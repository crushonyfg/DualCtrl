from __future__ import annotations

import inspect

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers import official
from controllers.kh_gp import (
    KHGPConfig,
    KHGPControllerCartPole,
    MultiOutputBayesLinearGP,
    construct_kh_gp_posterior_blocks,
    kh_ad_trace_uncertainty_cost,
)
from controllers.official import KHDualControlCartPole, OfficialCartPoleConfig


def _controller(num_features: int = 12, horizon: int = 2, action_grid_size: int = 3) -> KHGPControllerCartPole:
    config = KHGPConfig(
        horizon=horizon,
        action_grid_size=action_grid_size,
        num_features=num_features,
        lengthscale=1.2,
        prior_var=0.7,
        noise_var=0.05,
        seed=3,
    )
    return KHGPControllerCartPole(CartPoleParams(), CartPoleCost(CartPoleCostConfig()), config)


def test_fourier_feature_kernel_approximates_se_kernel() -> None:
    config = KHGPConfig(num_features=4000, lengthscale=1.7, seed=11)
    model = MultiOutputBayesLinearGP(input_dim=5, output_dim=1, config=config)
    x = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
    y = np.array([-0.3, 0.25, 0.1, -0.1, 0.2])
    approx = model.features.kernel(x, y)
    exact = np.exp(-0.5 * np.sum((x - y) ** 2) / (config.lengthscale * config.lengthscale))
    assert abs(approx - exact) < 0.04


def test_finite_feature_posterior_update_matches_bayes_linear_equations_16_17() -> None:
    config = KHGPConfig(num_features=8, prior_var=2.0, noise_var=0.3, seed=0)
    model = MultiOutputBayesLinearGP(input_dim=5, output_dim=4, config=config)
    z = np.array([0.2, -0.1, 0.05, 0.3, 1.0])
    y = np.array([0.4, -0.2, 0.1, 0.3])
    phi = model.features(z)
    cov0 = model.cov[0].copy()
    mean0 = model.mean[0].copy()
    s = float(phi @ cov0 @ phi + config.noise_var)
    gain = cov0 @ phi / s
    expected_mean0 = mean0 + gain * (y[0] - mean0 @ phi)
    expected_cov0 = cov0 - np.outer(gain, phi @ cov0)

    model.update(z, y)

    assert np.allclose(model.mean[0], expected_mean0)
    assert np.allclose(model.cov[0], expected_cov0)
    assert np.all(np.linalg.eigvalsh(model.cov[0]) >= -1e-10)


def test_kh_gram_matrix_eq_16_17_matches_sequential_bayes_finite_features() -> None:
    # Assumption encoded for controlled CartPole residual dynamics: along a fixed
    # nominal action sequence, actions are deterministic design variables and the
    # KH GP posterior is applied to x[j+1] = A[j]x[j] + Phi[j]w + q[j].
    A = [np.array([[1.1, 0.2], [0.0, 0.9]]), np.array([[0.8, -0.1], [0.3, 1.0]])]
    Phi = [np.array([[1.0, 0.5, -0.2], [0.0, -0.3, 0.7]]), np.array([[0.4, -0.6, 0.1], [0.2, 0.5, -0.4]])]
    q = np.diag([0.03, 0.04])
    r = np.diag([0.05, 0.06])
    sw = np.diag([0.7, 1.1, 0.9])
    sx = np.array([[0.2, 0.03], [0.03, 0.4]])
    x0 = np.array([0.3, -0.2])
    w0 = np.array([0.1, -0.05, 0.2])
    y = np.array([[0.6, -0.1], [0.2, 0.4]])

    blocks = construct_kh_gp_posterior_blocks(A, Phi, q, r, sw, initial_state_cov=sx, initial_state_mean=x0, weight_mean=w0, observations=y)

    mu = np.concatenate([x0, w0])
    cov = np.zeros((5, 5))
    cov[:2, :2] = sx
    cov[2:, 2:] = sw
    for j in range(2):
        M = np.zeros((5, 5))
        M[:2, :2] = A[j]
        M[:2, 2:] = Phi[j]
        M[2:, 2:] = np.eye(3)
        Q_aug = np.zeros((5, 5))
        Q_aug[:2, :2] = q
        mu = M @ mu
        cov = M @ cov @ M.T + Q_aug
        H = np.zeros((2, 5))
        H[:, :2] = np.eye(2)
        S = H @ cov @ H.T + r
        K = cov @ H.T @ np.linalg.inv(S)
        mu = mu + K @ (y[j] - H @ mu)
        cov = cov - K @ H @ cov

    assert blocks.F.shape == (4, 4)
    assert blocks.F_inv.shape == (4, 4)
    assert np.allclose(blocks.F_inv @ blocks.F, np.eye(4))
    assert np.allclose(blocks.G, blocks.P + blocks.Q + blocks.K + blocks.X + blocks.X.T + blocks.F_inv @ blocks.R @ blocks.F_inv.T)
    assert np.allclose(blocks.posterior_z_mean, mu, atol=1e-10)
    assert np.allclose(blocks.posterior_z_cov, cov, atol=1e-10)


def test_augmented_ekf_covariance_update_reduces_uncertainty() -> None:
    controller = _controller()
    state = np.array([0.0, 0.1, 0.05, -0.02])
    a_tilde, _, _, _ = controller.linearize_augmented(state, 0.5)
    sigma0 = controller.initial_augmented_covariance()
    pred, updated = controller.ekf_covariance_step(sigma0, a_tilde)

    assert pred.shape == updated.shape == sigma0.shape
    assert np.trace(updated) < np.trace(pred)
    assert np.trace(updated[4:, 4:]) < np.trace(pred[4:, 4:])


def test_ad_dual_uncertainty_cost_matches_section4_trace_formula_exactly() -> None:
    rng = np.random.default_rng(4)
    zdim = 5
    horizon = 3
    filtered = []
    predicted = []
    base = rng.normal(size=(zdim, zdim))
    sigma = base @ base.T + np.eye(zdim) * 0.2
    filtered.append(sigma)
    for _ in range(horizon):
        reduction = np.diag(np.linspace(0.01, 0.05, zdim))
        pred = sigma + reduction
        predicted.append(pred)
        sigma = pred - 0.5 * reduction
        filtered.append(sigma)
    riccati = []
    for _ in range(horizon + 1):
        a = rng.normal(size=(zdim, zdim))
        riccati.append(a @ a.T)
    W = np.diag([1.0, 0.5, 0.2, 0.0, 0.0])
    WT = np.diag([2.0, 0.7, 0.0, 0.0, 0.0])

    expected = 0.0
    for j in range(horizon):
        expected += np.trace(W @ filtered[j])
        expected += np.trace((predicted[j] - filtered[j + 1]) @ riccati[j + 1])
    expected += np.trace(WT @ filtered[-1])
    expected *= 0.5

    assert kh_ad_trace_uncertainty_cost(filtered, predicted, riccati, W, WT) == expected


def test_controller_dual_uncertainty_cost_uses_section4_trace_formula() -> None:
    controller = _controller(num_features=8, horizon=2, action_grid_size=3)
    state = np.array([0.0, 0.0, 0.05, 0.0])
    xs, us = controller.nominal_ce_trajectory(state, 0.0, 0.0)
    riccati = controller.augmented_riccati(xs, us)
    sigma = controller.initial_augmented_covariance()
    filtered = [sigma]
    predicted = []
    for a_tilde in riccati.a_tilde:
        pred, updated = controller.ekf_covariance_step(sigma, a_tilde)
        predicted.append(pred)
        filtered.append(updated)
        sigma = updated
    p = controller.model.weight_dim
    W = np.zeros_like(filtered[0])
    W[:4, :4] = np.diag([controller.cost.config.w_p, controller.cost.config.w_v, controller.cost.config.w_phi, controller.cost.config.w_omega])
    WT = np.zeros_like(filtered[0])
    WT[:4, :4] = np.diag([controller.cost.config.terminal_p_weight, 0.0, controller.cost.config.terminal_phi_weight, 0.0])
    assert W.shape == WT.shape == (4 + p, 4 + p)
    expected = kh_ad_trace_uncertainty_cost(filtered, predicted, riccati.riccati, W, WT)

    assert np.isclose(controller.dual_uncertainty_cost(xs, us, riccati), expected)


def test_augmented_riccati_dimensions_for_cartpole_z_x_w() -> None:
    controller = _controller(num_features=10, horizon=3, action_grid_size=3)
    state = np.array([0.0, 0.0, 0.05, 0.0])
    xs, us = controller.nominal_ce_trajectory(state, 0.0, 0.0)
    result = controller.augmented_riccati(xs, us)
    zdim = 4 + 4 * controller.config.num_features

    assert xs.shape == (controller.config.horizon + 1, 4)
    assert us.shape == (controller.config.horizon,)
    assert len(result.riccati) == controller.config.horizon + 1
    assert len(result.gains) == controller.config.horizon
    for k in result.riccati:
        assert k.shape == (zdim, zdim)
    for gain in result.gains:
        assert gain.shape == (1, zdim)
    for a_tilde, b_tilde in zip(result.a_tilde, result.b_tilde):
        assert a_tilde.shape == (zdim, zdim)
        assert b_tilde.shape == (zdim, 1)


def test_cartpole_kh_uses_gp_transition_delta_not_actuator_gain_pseudo_observation() -> None:
    controller = KHDualControlCartPole(
        GaussianBelief(mean=1.0, var=0.1),
        CartPoleParams(),
        CartPoleCost(CartPoleCostConfig()),
        OfficialCartPoleConfig(horizon=2, action_grid_size=3),
    )
    assert isinstance(controller, KHGPControllerCartPole)
    assert not hasattr(controller, "belief")
    source = inspect.getsource(KHDualControlCartPole.observe)
    assert "_cartpole_theta_pseudo_observation" not in source

    state = np.array([0.0, 0.0, 0.05, 0.0])
    next_state = state + np.array([0.01, -0.02, 0.03, -0.04])
    before = controller.model.mean.copy()
    controller.observe(state, 0.5, next_state)
    assert not np.allclose(controller.model.mean, before)


def test_kh_and_arcari_cartpole_do_not_share_pseudo_observation_path() -> None:
    kh_source = inspect.getsource(official.KHDualControlCartPole)
    arcari_source = inspect.getsource(official.ArcariDualSMPCCartPole)
    assert "_cartpole_theta_pseudo_observation" not in kh_source
    assert "_cartpole_theta_pseudo_observation" in arcari_source
