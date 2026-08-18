from __future__ import annotations

import math

import numpy as np

from controllers.tv_gp_ucb import (
    RealizedCostBanditAdapter,
    TVGPLCB,
    TVGPUCB,
    TVGPUCBConfig,
    finite_action_beta,
    markov_time_space_kernel,
    squared_exponential_kernel,
    TimeVaryingGPPosterior,
)


def test_markov_time_space_kernel_values() -> None:
    config = TVGPUCBConfig(epsilon=0.36, lengthscale=2.0, signal_var=3.0)
    x = np.array([1.0, 0.0])
    y = np.array([3.0, 0.0])

    spatial = 3.0 * math.exp(-0.5)
    assert math.isclose(squared_exponential_kernel(x, y, lengthscale=2.0, signal_var=3.0), spatial, rel_tol=1e-12)
    assert math.isclose(
        markov_time_space_kernel(x, 2, y, 5, config),
        (1.0 - 0.36) ** (3.0 / 2.0) * spatial,
        rel_tol=1e-12,
    )
    assert math.isclose(markov_time_space_kernel(x, 4, x, 4, config), 3.0, rel_tol=1e-12)


def test_posterior_single_observation_matches_closed_form() -> None:
    config = TVGPUCBConfig(epsilon=0.36, noise_var=0.25, lengthscale=1.0, signal_var=1.0, jitter=0.0)
    posterior = TimeVaryingGPPosterior(config)
    posterior.update(np.array([0.0]), t=1, y=2.0)

    mean, var = posterior.predict(np.array([0.0]), t=2)

    # K + sigma^2 I = [1.25], k_* = sqrt(1 - epsilon) = 0.8.
    assert math.isclose(mean, 0.8 * 2.0 / 1.25, rel_tol=1e-12)
    assert math.isclose(var, 1.0 - (0.8 * 0.8) / 1.25, rel_tol=1e-12)


def test_finite_action_beta_formula() -> None:
    beta = finite_action_beta(t=3, num_actions=5, delta=0.2)
    expected = 2.0 * math.log(5.0 * (math.pi**2 * 3.0**2 / 6.0) / 0.2)
    assert math.isclose(beta, expected, rel_tol=1e-12)


def test_ucb_and_lcb_acquisition_on_deterministic_dataset() -> None:
    actions = [0.0, 1.0]
    feature_map = lambda arm: np.array([arm], dtype=float)
    config = TVGPUCBConfig(epsilon=0.0, noise_var=1e-6, delta=0.1, lengthscale=0.1, jitter=0.0)

    ucb = TVGPUCB(actions, feature_map, config)
    ucb.update(0.0, reward=0.0, t=1)
    assert ucb.select() == 1.0
    ucb_values = {value.arm: value for value in ucb.acquisition_values()}
    assert ucb_values[1.0].score > ucb_values[0.0].score

    lcb = TVGPLCB(actions, feature_map, config)
    lcb.update(0.0, cost=0.0, t=1)
    assert lcb.select() == 1.0
    lcb_values = {value.arm: value for value in lcb.acquisition_values()}
    assert lcb_values[1.0].score < lcb_values[0.0].score


def test_realized_cost_adapter_uses_observed_cost_only() -> None:
    config = TVGPUCBConfig(epsilon=0.0, noise_var=1e-6, delta=0.1, lengthscale=0.1, jitter=0.0)
    adapter = RealizedCostBanditAdapter(
        action_provider=lambda context: [-1.0, 1.0],
        feature_map=lambda context, action: np.array([float(context), float(action)], dtype=float),
        config=config,
    )

    first = adapter.select(context=2.0)
    adapter.update(realized_cost=7.5)

    assert first in {-1.0, 1.0}
    assert adapter.posterior.observations == [7.5]
    np.testing.assert_allclose(adapter.posterior.features[0], np.array([2.0, first]))
    assert adapter.posterior.times == [1]
