"""Klenske-Hennig finite-feature GP approximate dual control for CartPole.

This module implements the GP version of Klenske & Hennig (2016), Secs. 4,
5.2, 5.2.1, App. A.2, and App. B for the repository's explicit finite-action
CartPole benchmark.  The unknown dynamics are represented in weight space by a
finite random Fourier feature approximation to a squared-exponential GP over
state-action inputs.  The controller models CartPole transition deltas,

    x[k+1] = f_nominal(x[k], u[k]) + W phi([x[k], u[k]]) + eps,
    vec(W)[k+1] = vec(W)[k],

with independent GP outputs.  Planning uses the Sec. 4 approximate-dual recipe:
a certainty-equivalent nominal trajectory under the current posterior mean,
local Jacobians of the augmented dynamics z=(x, vec(W)), an augmented Riccati
recursion, and deterministic preposterior covariance filtering for the dual
uncertainty term.  Because the benchmark action set is explicitly finite, the
outer current-action optimization and CE tail are exact enumerations over that
finite grid rather than continuous NLP solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost
from benchmarks.cartpole_twin.dynamics import CartPoleParams, cartpole_step, reference_position
from controllers.action_optimization import minimize_scalar_action, minimize_vector_actions


@dataclass(frozen=True)
class KHGPConfig:
    horizon: int = 4
    action_grid_size: int = 7
    continuous_actions: bool = False
    optimizer_grid_size: int = 81
    optimizer_maxiter: int = 100
    optimizer_xatol: float = 1e-4
    num_features: int = 32
    lengthscale: float = 1.0
    signal_var: float = 1.0
    prior_var: float = 1.0
    noise_var: float = 1e-3
    gp_noise_var: float | None = None
    state_obs_var: float = 1e-9
    seed: int = 0
    finite_difference_eps: float = 1e-5
    process_noise_var: tuple[float, float, float, float] = (0.0, 1e-4, 0.0, 1e-4)
    min_variance: float = 1e-12

    def __post_init__(self) -> None:
        if self.gp_noise_var is None:
            object.__setattr__(self, "gp_noise_var", float(self.noise_var))
        else:
            object.__setattr__(self, "noise_var", float(self.gp_noise_var))

    @property
    def action_grid(self) -> np.ndarray:
        # The official benchmark exposes a finite CartPole action set, so KH's
        # root optimization is an exact finite minimization over this set.
        return np.linspace(-1.0, 1.0, self.action_grid_size)


class FourierFeatureMap:
    """Sec. 5.2.1 random Fourier features for the squared-exponential kernel."""

    def __init__(self, input_dim: int, num_features: int, lengthscale: float, seed: int, signal_var: float = 1.0):
        if num_features <= 0 or num_features % 2:
            raise ValueError("num_features must be a positive even integer for sine/cosine pairs")
        if lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive")
        self.input_dim = int(input_dim)
        self.num_features = int(num_features)
        self.lengthscale = float(lengthscale)
        self.signal_var = float(signal_var)
        rng = np.random.default_rng(seed)
        self.omega = rng.normal(0.0, 1.0 / self.lengthscale, size=(self.num_features // 2, self.input_dim))
        self.scale = float(np.sqrt(2.0 * self.signal_var / self.num_features))

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape[0] != self.input_dim:
            raise ValueError(f"expected input dimension {self.input_dim}, got {z.shape[0]}")
        proj = self.omega @ z
        return self.scale * np.concatenate([np.sin(proj), np.cos(proj)])

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Return d phi(z) / d z with shape (num_features, input_dim)."""

        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape[0] != self.input_dim:
            raise ValueError(f"expected input dimension {self.input_dim}, got {z.shape[0]}")
        proj = self.omega @ z
        return self.scale * np.vstack([np.cos(proj)[:, None] * self.omega, -np.sin(proj)[:, None] * self.omega])

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.approximate_kernel(x, y)

    def approximate_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self(x) @ self(y))

    def exact_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = np.asarray(x, dtype=float).reshape(-1) - np.asarray(y, dtype=float).reshape(-1)
        return float(self.signal_var * np.exp(-0.5 * np.dot(diff, diff) / (self.lengthscale * self.lengthscale)))


class MultiOutputBayesLinearGP:
    """Independent-output finite-feature GP in Bayesian linear weight space."""

    def __init__(self, input_dim: int, output_dim: int, config: KHGPConfig):
        self.features = FourierFeatureMap(input_dim, config.num_features, config.lengthscale, config.seed, config.signal_var)
        self.output_dim = int(output_dim)
        self.config = config
        self.mean = np.zeros((self.output_dim, config.num_features), dtype=float)
        self.cov = np.stack([np.eye(config.num_features, dtype=float) * config.prior_var for _ in range(self.output_dim)])

    @property
    def num_features(self) -> int:
        return self.config.num_features

    @property
    def weight_dim(self) -> int:
        return self.output_dim * self.num_features

    def predict(self, z: np.ndarray, include_noise: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi = self.features(z)
        mean = self.mean @ phi
        noise = self.config.gp_noise_var if include_noise else 0.0
        var = np.array(
            [max(float(phi @ self.cov[d] @ phi + noise), self.config.min_variance) for d in range(self.output_dim)],
            dtype=float,
        )
        return mean, var, phi

    def update(self, z: np.ndarray, y: np.ndarray, noise_var: float | None = None) -> None:
        """Bayesian linear-Gaussian posterior update, finite-feature Eq. (16)-(17)."""

        phi = self.features(z)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != self.output_dim:
            raise ValueError(f"expected output dimension {self.output_dim}, got {y.shape[0]}")
        r = max(self.config.gp_noise_var if noise_var is None else float(noise_var), self.config.min_variance)
        for d in range(self.output_dim):
            cov = self.cov[d]
            s = max(float(phi @ cov @ phi + r), self.config.min_variance)
            gain = cov @ phi / s
            residual = float(y[d] - self.mean[d] @ phi)
            self.mean[d] = self.mean[d] + gain * residual
            self.cov[d] = cov - np.outer(gain, phi @ cov)
            self.cov[d] = _symmetrize_with_min_diagonal(self.cov[d], self.config.min_variance)

    def flat_mean(self) -> np.ndarray:
        return self.mean.reshape(-1).copy()

    def weight_mean_vector(self) -> np.ndarray:
        return self.flat_mean()

    def flat_cov(self) -> np.ndarray:
        out = np.zeros((self.weight_dim, self.weight_dim), dtype=float)
        m = self.config.num_features
        for d in range(self.output_dim):
            out[d * m : (d + 1) * m, d * m : (d + 1) * m] = self.cov[d]
        return out

    def weight_covariance(self) -> np.ndarray:
        return self.flat_cov()

    def copy(self) -> "MultiOutputBayesLinearGP":
        other = object.__new__(MultiOutputBayesLinearGP)
        other.features = self.features
        other.output_dim = self.output_dim
        other.config = self.config
        other.mean = self.mean.copy()
        other.cov = self.cov.copy()
        return other


@dataclass(frozen=True)
class KHGPPosteriorBlocks:
    """Equation-level KH GP posterior blocks for finite-feature weight space.

    The controlled CartPole benchmark uses nominal-plus-residual dynamics
    ``x[j+1] = f_nominal(x[j], u[j]) + W phi([x[j], u[j]]) + q[j]``.  Along a fixed
    action sequence, actions are treated as deterministic design variables, so
    the KH Sec. 5.2/App. A.2 GP posterior is applied to the locally linearized
    model ``x[j+1] = A[j] x[j] + Phi[j] w + q[j]`` with observations of the
    next physical state.  This is the finite-dimensional weight-space form of
    Eqs. (16)/(17): ``K`` is represented by ``Phi Sigma_ww Phi^T`` rather than
    by a heuristic finite-feature covariance bonus.
    """

    F: np.ndarray
    F_inv: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    K: np.ndarray
    R: np.ndarray
    X: np.ndarray
    C: np.ndarray
    G: np.ndarray
    G_inv: np.ndarray
    PhiSigma: np.ndarray
    prior_state_mean_stack: np.ndarray
    compressed_observation_residual: np.ndarray | None
    posterior_x_mean: np.ndarray
    posterior_w_mean: np.ndarray
    posterior_x_cov: np.ndarray
    posterior_xw_cov: np.ndarray
    posterior_w_cov: np.ndarray

    @property
    def posterior_z_mean(self) -> np.ndarray:
        return np.concatenate([self.posterior_x_mean, self.posterior_w_mean])

    @property
    def posterior_z_cov(self) -> np.ndarray:
        n = self.posterior_x_cov.shape[0]
        p = self.posterior_w_cov.shape[0]
        out = np.zeros((n + p, n + p), dtype=float)
        out[:n, :n] = self.posterior_x_cov
        out[:n, n:] = self.posterior_xw_cov
        out[n:, :n] = self.posterior_xw_cov.T
        out[n:, n:] = self.posterior_w_cov
        return out


@dataclass(frozen=True)
class KHRiccatiResult:
    gains: list[np.ndarray]
    riccati: list[np.ndarray]
    a_tilde: list[np.ndarray]
    b_tilde: list[np.ndarray]


def construct_kh_gp_posterior_blocks(
    transition_jacobians: list[np.ndarray] | np.ndarray,
    feature_blocks: list[np.ndarray] | np.ndarray,
    process_cov: np.ndarray,
    observation_cov: np.ndarray,
    weight_cov: np.ndarray,
    initial_state_cov: np.ndarray | None = None,
    initial_state_weight_cov: np.ndarray | None = None,
    initial_state_mean: np.ndarray | None = None,
    weight_mean: np.ndarray | None = None,
    observations: np.ndarray | None = None,
) -> KHGPPosteriorBlocks:
    """Construct finite-feature KH Eq. (16)/(17) GP posterior quantities.

    Parameters describe the linearized model ``x[j+1]=A[j] x[j]+Phi[j] w+q[j]``
    and full-state observations ``y[j+1]=x[j+1]+r[j]``.  The returned matrices
    explicitly expose KH's multi-step ``F``, ``F^{-1}``, ``P,Q,K,R`` blocks,
    compressed Gram matrix ``G = P+Q+K+X+X.T+F^{-1} R F^{-T}``, and posterior
    mean/covariance blocks for ``z=(x,w)`` at the final step.  ``X`` is zero
    under the common KH nominal-shift assumption that current state and weights
    are uncorrelated; it is included to exactly support later AD filtering where
    ``Sigma_xw`` may be nonzero (the cross terms are the finite-feature version
    of Eq. (17a)).
    """

    A = [np.asarray(a, dtype=float) for a in transition_jacobians]
    Phi = [np.asarray(phi, dtype=float) for phi in feature_blocks]
    if len(A) == 0:
        raise ValueError("at least one transition is required")
    if len(A) != len(Phi):
        raise ValueError("transition_jacobians and feature_blocks must have the same length")
    m = len(A)
    n = A[0].shape[0]
    p = Phi[0].shape[1]
    for a in A:
        if a.shape != (n, n):
            raise ValueError("all transition_jacobians must have shape (n, n)")
    for phi in Phi:
        if phi.shape != (n, p):
            raise ValueError("all feature_blocks must have shape (n, p)")

    Q_step = _as_square_or_diag(process_cov, n, "process_cov")
    R_step = _as_square_or_diag(observation_cov, n, "observation_cov")
    Sigma_w = _as_square_or_diag(weight_cov, p, "weight_cov")
    Sigma_x0 = np.zeros((n, n), dtype=float) if initial_state_cov is None else _as_square_or_diag(initial_state_cov, n, "initial_state_cov")
    Sigma_xw0 = np.zeros((n, p), dtype=float) if initial_state_weight_cov is None else np.asarray(initial_state_weight_cov, dtype=float)
    if Sigma_xw0.shape != (n, p):
        raise ValueError("initial_state_weight_cov must have shape (n, p)")
    x0_mean = np.zeros(n, dtype=float) if initial_state_mean is None else np.asarray(initial_state_mean, dtype=float).reshape(n)
    w_mean = np.zeros(p, dtype=float) if weight_mean is None else np.asarray(weight_mean, dtype=float).reshape(p)

    mn = m * n
    F = np.zeros((mn, mn), dtype=float)
    F_inv = np.zeros((mn, mn), dtype=float)
    for i in range(m):
        F_inv[i * n : (i + 1) * n, i * n : (i + 1) * n] = np.eye(n)
        if i > 0:
            F_inv[i * n : (i + 1) * n, (i - 1) * n : i * n] = -A[i]
        for j in range(i + 1):
            block = np.eye(n)
            for ell in range(j + 1, i + 1):
                block = A[ell] @ block
            F[i * n : (i + 1) * n, j * n : (j + 1) * n] = block

    P = np.zeros((mn, mn), dtype=float)
    P[:n, :n] = A[0] @ Sigma_x0 @ A[0].T
    Q = np.zeros((mn, mn), dtype=float)
    R = np.zeros((mn, mn), dtype=float)
    for i in range(m):
        Q[i * n : (i + 1) * n, i * n : (i + 1) * n] = Q_step
        R[i * n : (i + 1) * n, i * n : (i + 1) * n] = R_step

    Phi_stack = np.vstack(Phi)
    K = Phi_stack @ Sigma_w @ Phi_stack.T
    PhiSigma = Phi_stack @ Sigma_w
    X = np.zeros((mn, mn), dtype=float)
    xw0_after_a0 = A[0] @ Sigma_xw0
    X[:n, :] = xw0_after_a0 @ PhiSigma.T
    C = P + Q + K + X + X.T
    G = C + F_inv @ R @ F_inv.T
    G_inv = np.linalg.pinv(G)

    compressed_mean = np.zeros(mn, dtype=float)
    compressed_mean[:n] = A[0] @ x0_mean + Phi[0] @ w_mean
    for i in range(1, m):
        compressed_mean[i * n : (i + 1) * n] = Phi[i] @ w_mean
    prior_state_mean_stack = F @ compressed_mean
    final_selector = F[(m - 1) * n : m * n, :]
    prior_x_final_mean = prior_state_mean_stack[(m - 1) * n : m * n]

    compressed_residual: np.ndarray | None = None
    posterior_x_mean = prior_x_final_mean.copy()
    posterior_w_mean = w_mean.copy()
    if observations is not None:
        obs = np.asarray(observations, dtype=float).reshape(mn)
        compressed_residual = F_inv @ (obs - prior_state_mean_stack)
        posterior_x_mean = posterior_x_mean + final_selector @ C @ G_inv @ compressed_residual
        posterior_w_mean = posterior_w_mean + PhiSigma.T @ G_inv @ compressed_residual

    posterior_x_cov = final_selector @ C @ final_selector.T - final_selector @ C @ G_inv @ C @ final_selector.T
    posterior_xw_cov = final_selector @ PhiSigma - final_selector @ C @ G_inv @ PhiSigma
    posterior_w_cov = Sigma_w - PhiSigma.T @ G_inv @ PhiSigma

    return KHGPPosteriorBlocks(
        F=F,
        F_inv=F_inv,
        P=P,
        Q=Q,
        K=K,
        R=R,
        X=X,
        C=C,
        G=G,
        G_inv=G_inv,
        PhiSigma=PhiSigma,
        prior_state_mean_stack=prior_state_mean_stack,
        compressed_observation_residual=compressed_residual,
        posterior_x_mean=posterior_x_mean,
        posterior_w_mean=posterior_w_mean,
        posterior_x_cov=0.5 * (posterior_x_cov + posterior_x_cov.T),
        posterior_xw_cov=posterior_xw_cov,
        posterior_w_cov=0.5 * (posterior_w_cov + posterior_w_cov.T),
    )


class KHGPControllerCartPole:
    """Strict CartPole-compatible KH approximate dual controller."""

    name = "kh_dual_control"

    def __init__(self, dynamics: CartPoleParams, cost: CartPoleCost, config: KHGPConfig):
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.model = MultiOutputBayesLinearGP(input_dim=5, output_dim=4, config=config)
        self.t = 0

    @property
    def belief_mean(self) -> float:
        return float(np.linalg.norm(self.model.flat_mean()))

    @property
    def belief_var(self) -> float:
        return float(np.trace(self.model.flat_cov()))

    def predict(self) -> None:
        # KH GP weights are static; no actuator-gain drift prediction is used.
        pass

    def _input(self, state: np.ndarray, action: float) -> np.ndarray:
        return np.concatenate([np.asarray(state, dtype=float).reshape(4), np.array([float(action)])])

    def transition_delta_mean(self, state: np.ndarray, action: float, flat_weights: np.ndarray | None = None) -> np.ndarray:
        phi = self.model.features(self._input(state, action))
        weights = self.model.mean if flat_weights is None else np.asarray(flat_weights, dtype=float).reshape(4, self.config.num_features)
        return weights @ phi

    def nominal_next_state(self, state: np.ndarray, action: float) -> np.ndarray:
        next_state, _ = cartpole_step(np.asarray(state, dtype=float).reshape(4), float(action), 1.0, self.dynamics)
        return next_state

    def mean_next_state(self, state: np.ndarray, action: float, flat_weights: np.ndarray | None = None) -> np.ndarray:
        return self.nominal_next_state(state, action) + self.transition_delta_mean(state, action, flat_weights)

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        # Strict GP dynamics observation: transition delta only.  This deliberately
        # avoids the old CartPole actuator-gain/theta pseudo-observation.
        delta = np.asarray(next_state, dtype=float).reshape(4) - self.nominal_next_state(state, action)
        self.model.update(self._input(state, action), delta, noise_var=self.config.gp_noise_var)
        self.t += 1

    def act(self, state: np.ndarray, prev_action: float) -> float:
        state = np.asarray(state, dtype=float).reshape(4)
        if self.config.continuous_actions:
            action, _ = minimize_scalar_action(
                lambda u: self.approximate_dual_cost(state, float(prev_action), float(u), self.t),
                -1.0,
                1.0,
                self.config.optimizer_grid_size,
                self.config.optimizer_maxiter,
                self.config.optimizer_xatol,
            )
            return float(action)

        best_action = 0.0
        best_value = float("inf")
        for action in self.config.action_grid:
            value = self.approximate_dual_cost(state, float(prev_action), float(action), self.t)
            if value < best_value:
                best_value = value
                best_action = float(action)
        return best_action

    def approximate_dual_cost(self, state: np.ndarray, prev_action: float, root_action: float, t0: int = 0) -> float:
        xs, us = self.nominal_ce_trajectory(state, prev_action, root_action, t0)
        nominal = self._nominal_cost(xs, us, prev_action, t0)
        riccati = self.augmented_riccati(xs, us, t0)
        dual = self.dual_uncertainty_cost(xs, us, riccati)
        return float(nominal + dual)

    def nominal_ce_trajectory(self, state: np.ndarray, prev_action: float, root_action: float, t0: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Certainty-equivalent trajectory with fixed root action and exact finite-grid tail."""

        del t0
        horizon = int(self.config.horizon)
        xs0 = [np.asarray(state, dtype=float).reshape(4).copy()]
        us0 = [float(np.clip(root_action, -1.0, 1.0))]
        xs0.append(self.mean_next_state(xs0[0], us0[0]))
        if horizon == 1:
            return np.asarray(xs0), np.asarray(us0)

        if self.config.continuous_actions:
            def tail_objective(tail_actions: np.ndarray) -> float:
                xs = [x.copy() for x in xs0]
                us = list(us0)
                for u in np.asarray(tail_actions, dtype=float).reshape(-1):
                    us.append(float(u))
                    xs.append(self.mean_next_state(xs[-1], float(u)))
                return _quadratic_rollout_cost_for_tail(np.asarray(xs), np.asarray(us), prev_action, self.cost)

            tail_actions, _ = minimize_vector_actions(
                tail_objective,
                [(-1.0, 1.0)] * (horizon - 1),
                x0=[0.0] * (horizon - 1),
                grid_size=self.config.optimizer_grid_size,
                maxiter=self.config.optimizer_maxiter,
                xatol=self.config.optimizer_xatol,
            )
            xs = [x.copy() for x in xs0]
            us = list(us0)
            for u in tail_actions:
                us.append(float(u))
                xs.append(self.mean_next_state(xs[-1], float(u)))
            return np.asarray(xs), np.asarray(us)

        action_tuple_grid = tuple(float(u) for u in self.config.action_grid)
        best_tail: tuple[float, list[np.ndarray], list[float]] | None = None
        for tail in product(action_tuple_grid, repeat=horizon - 1):
            xs = [x.copy() for x in xs0]
            us = list(us0)
            for u in tail:
                us.append(float(u))
                xs.append(self.mean_next_state(xs[-1], float(u)))
            value = _quadratic_rollout_cost_for_tail(np.asarray(xs), np.asarray(us), prev_action, self.cost)
            if best_tail is None or value < best_tail[0]:
                best_tail = (value, xs, us)
        if best_tail is None:
            raise RuntimeError("empty action grid")
        return np.asarray(best_tail[1]), np.asarray(best_tail[2])

    def linearize_augmented(self, state: np.ndarray, action: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Local Jacobians of z[k+1]=(x[k+1],w[k+1]) about the posterior mean."""

        n = 4
        p = self.model.weight_dim
        eps = self.config.finite_difference_eps
        state = np.asarray(state, dtype=float).reshape(4)
        weights = self.model.flat_mean()
        base_next = self.mean_next_state(state, action, weights)

        fx = np.zeros((n, n), dtype=float)
        for i in range(n):
            step = eps * max(1.0, abs(float(state[i])))
            plus = state.copy()
            minus = state.copy()
            plus[i] += step
            minus[i] -= step
            fx[:, i] = (self.mean_next_state(plus, action, weights) - self.mean_next_state(minus, action, weights)) / (2.0 * step)

        u_step = eps * max(1.0, abs(float(action)))
        fu = ((self.mean_next_state(state, action + u_step, weights) - self.mean_next_state(state, action - u_step, weights)) / (2.0 * u_step)).reshape(n, 1)

        phi = self.model.features(self._input(state, action))
        fw = np.zeros((n, p), dtype=float)
        m = self.config.num_features
        for d in range(n):
            fw[d, d * m : (d + 1) * m] = phi

        a_tilde = np.zeros((n + p, n + p), dtype=float)
        a_tilde[:n, :n] = fx
        a_tilde[:n, n:] = fw
        a_tilde[n:, n:] = np.eye(p)
        b_tilde = np.zeros((n + p, 1), dtype=float)
        b_tilde[:n, :] = fu
        return a_tilde, b_tilde, fx, base_next

    def feature_weight_map(self, state: np.ndarray, action: float) -> np.ndarray:
        """Map flat finite-feature weights to the CartPole transition residual."""

        n = 4
        p = self.model.weight_dim
        phi = self.model.features(self._input(state, action))
        out = np.zeros((n, p), dtype=float)
        m = self.config.num_features
        for d in range(n):
            out[d, d * m : (d + 1) * m] = phi
        return out

    def gram_matrix_posterior_for_observed_trajectory(
        self, xs: np.ndarray, us: np.ndarray, observation_cov: np.ndarray | None = None
    ) -> KHGPPosteriorBlocks:
        """Build Eq. (16)/(17) finite-feature Gram blocks for a fixed CartPole trajectory.

        The controlled residual-dynamics assumption is explicit: ``us`` is fixed,
        and the GP prior is over ``delta_x = W phi([x,u])``.
        """

        xs = np.asarray(xs, dtype=float)
        us = np.asarray(us, dtype=float).reshape(-1)
        a_mats = []
        phi_mats = []
        for j, u in enumerate(us):
            _, _, fx, _ = self.linearize_augmented(xs[j], float(u))
            a_mats.append(fx)
            phi_mats.append(self.feature_weight_map(xs[j], float(u)))
        process_cov = np.diag(np.asarray(self.config.process_noise_var, dtype=float))
        obs_cov = np.eye(4, dtype=float) * max(self.config.state_obs_var, self.config.min_variance) if observation_cov is None else observation_cov
        return construct_kh_gp_posterior_blocks(
            transition_jacobians=a_mats,
            feature_blocks=phi_mats,
            process_cov=process_cov,
            observation_cov=obs_cov,
            weight_cov=self.model.flat_cov(),
            initial_state_cov=np.zeros((4, 4), dtype=float),
            initial_state_mean=xs[0],
            weight_mean=self.model.flat_mean(),
            observations=xs[1:],
        )

    def augmented_riccati(self, xs: np.ndarray, us: np.ndarray, t0: int = 0) -> KHRiccatiResult:
        """Sec. 4 augmented Riccati recursion for z=(x,w)."""

        del t0
        horizon = len(us)
        n = 4
        p = self.model.weight_dim
        zdim = n + p
        q = _augmented_cost_matrix(self.cost, p, terminal=False)
        terminal = _augmented_cost_matrix(self.cost, p, terminal=True)
        riccati = [np.zeros((zdim, zdim), dtype=float) for _ in range(horizon + 1)]
        gains = [np.zeros((1, zdim), dtype=float) for _ in range(horizon)]
        a_list: list[np.ndarray] = [np.zeros((zdim, zdim), dtype=float) for _ in range(horizon)]
        b_list: list[np.ndarray] = [np.zeros((zdim, 1), dtype=float) for _ in range(horizon)]
        riccati[horizon] = terminal
        r = np.array([[max(self.cost.config.energy_weight + self.cost.config.switch_weight, self.config.min_variance)]], dtype=float)
        for j in range(horizon - 1, -1, -1):
            a_tilde, b_tilde, _, _ = self.linearize_augmented(xs[j], float(us[j]))
            a_list[j] = a_tilde
            b_list[j] = b_tilde
            s = b_tilde.T @ riccati[j + 1] @ b_tilde + r
            gain = np.linalg.pinv(s) @ b_tilde.T @ riccati[j + 1] @ a_tilde
            gains[j] = gain
            riccati[j] = q + a_tilde.T @ riccati[j + 1] @ (a_tilde - b_tilde @ gain)
            riccati[j] = 0.5 * (riccati[j] + riccati[j].T)
        return KHRiccatiResult(gains=gains, riccati=riccati, a_tilde=a_list, b_tilde=b_list)

    def initial_augmented_covariance(self) -> np.ndarray:
        n = 4
        p = self.model.weight_dim
        sigma = np.zeros((n + p, n + p), dtype=float)
        sigma[n:, n:] = self.model.flat_cov()
        return sigma

    def ekf_covariance_step(self, sigma: np.ndarray, a_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-step finite-feature KH Eq. (17) covariance prediction/update."""

        n = 4
        zdim = sigma.shape[0]
        p = zdim - n
        qx = np.diag(np.asarray(self.config.process_noise_var, dtype=float))
        A = a_tilde[:n, :n]
        Phi = a_tilde[:n, n:]
        blocks = construct_kh_gp_posterior_blocks(
            [A],
            [Phi],
            qx,
            np.eye(n, dtype=float) * max(self.config.state_obs_var, self.config.min_variance),
            sigma[n:, n:],
            initial_state_cov=sigma[:n, :n],
            initial_state_weight_cov=sigma[:n, n:],
        )
        pred_xx = A @ sigma[:n, :n] @ A.T + qx + Phi @ sigma[n:, n:] @ Phi.T + A @ sigma[:n, n:] @ Phi.T + Phi @ sigma[n:, :n] @ A.T
        pred_xw = A @ sigma[:n, n:] + Phi @ sigma[n:, n:]
        pred = np.zeros_like(sigma)
        pred[:n, :n] = pred_xx
        pred[:n, n:] = pred_xw
        pred[n:, :n] = pred_xw.T
        pred[n:, n:] = sigma[n:, n:]
        updated = blocks.posterior_z_cov
        return _symmetrize_with_min_diagonal(pred, self.config.min_variance), _symmetrize_with_min_diagonal(updated, self.config.min_variance)

    def dual_uncertainty_cost(self, xs: np.ndarray, us: np.ndarray, riccati: KHRiccatiResult) -> float:
        """Sec. 4 approximate-dual covariance cost along the CE trajectory."""

        n = 4
        p = self.model.weight_dim
        sigma = self.initial_augmented_covariance()
        q = _augmented_cost_matrix(self.cost, p, terminal=False)
        terminal = _augmented_cost_matrix(self.cost, p, terminal=True)
        filtered = [sigma]
        predicted = []
        for j in range(len(us)):
            pred, updated = self.ekf_covariance_step(sigma, riccati.a_tilde[j])
            predicted.append(pred)
            filtered.append(updated)
            sigma = updated
        dual = kh_ad_trace_uncertainty_cost(filtered, predicted, riccati.riccati, q, terminal)
        return max(float(dual), 0.0)

    def _nominal_cost(self, xs: np.ndarray, us: np.ndarray, prev_action: float, t0: int) -> float:
        total = 0.0
        last = float(prev_action)
        for j, action in enumerate(us):
            failed = self.cost.failed(xs[j + 1]) if j + 1 < len(xs) else self.cost.failed(xs[j])
            total += self.cost.stage(xs[j], float(action), last, reference_position(t0 + j), failed).total
            last = float(action)
        total += self.cost.terminal(xs[-1], reference_position(t0 + len(us))).total
        return float(total)


# Convenience functional API for tests/diagnostics.
def initial_augmented_covariance(model: MultiOutputBayesLinearGP) -> np.ndarray:
    n = 4
    sigma = np.zeros((n + model.weight_dim, n + model.weight_dim), dtype=float)
    sigma[n:, n:] = model.flat_cov()
    return sigma


def augmented_state_observation_update(sigma_pred: np.ndarray, config: KHGPConfig) -> np.ndarray:
    n = 4
    zdim = sigma_pred.shape[0]
    h = np.zeros((n, zdim), dtype=float)
    h[:, :n] = np.eye(n)
    r = np.eye(n, dtype=float) * max(config.state_obs_var, config.min_variance)
    innovation = h @ sigma_pred @ h.T + r
    gain = sigma_pred @ h.T @ np.linalg.pinv(innovation)
    return _symmetrize_with_min_diagonal(sigma_pred - gain @ h @ sigma_pred, config.min_variance)


def _as_square_or_diag(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.eye(dim, dtype=float) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"{name} must have length {dim}")
        return np.diag(arr)
    if arr.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim})")
    return arr


def kh_ad_trace_uncertainty_cost(
    filtered_covariances: list[np.ndarray],
    predictive_covariances: list[np.ndarray],
    riccati: list[np.ndarray],
    stage_cost: np.ndarray,
    terminal_cost: np.ndarray,
) -> float:
    """Section 4 trace-form AD uncertainty cost.

    This evaluates exactly
    ``1/2 tr(W_T Sigma_T|T + sum_j W_j Sigma_j|j +
    (Sigma_{j+1|j}-Sigma_{j+1|j+1}) K_{j+1})`` for compatible matrices.
    """

    if len(predictive_covariances) + 1 != len(filtered_covariances):
        raise ValueError("filtered_covariances must contain initial plus one updated covariance per prediction")
    if len(riccati) != len(filtered_covariances):
        raise ValueError("riccati must align with filtered_covariances")
    total = 0.0
    W = np.asarray(stage_cost, dtype=float)
    WT = np.asarray(terminal_cost, dtype=float)
    for j, pred in enumerate(predictive_covariances):
        total += float(np.trace(W @ filtered_covariances[j]))
        total += float(np.trace((np.asarray(pred, dtype=float) - filtered_covariances[j + 1]) @ riccati[j + 1]))
    total += float(np.trace(WT @ filtered_covariances[-1]))
    return 0.5 * float(total)


def _augmented_cost_matrix(cost: CartPoleCost, weight_dim: int, terminal: bool) -> np.ndarray:
    if terminal:
        qx = np.diag([cost.config.terminal_p_weight, 0.0, cost.config.terminal_phi_weight, 0.0])
    else:
        qx = np.diag([cost.config.w_p, cost.config.w_v, cost.config.w_phi, cost.config.w_omega])
    out = np.zeros((4 + weight_dim, 4 + weight_dim), dtype=float)
    out[:4, :4] = qx
    return out


def _quadratic_rollout_cost_for_tail(xs: np.ndarray, us: np.ndarray, prev_action: float, cost: CartPoleCost) -> float:
    # Time-independent surrogate used only to choose the CE tail; the final
    # nominal cost is recomputed with reference_position(t) in _nominal_cost.
    total = 0.0
    last = float(prev_action)
    for j, u in enumerate(us):
        failed = cost.failed(xs[j + 1]) if j + 1 < len(xs) else False
        total += cost.stage(xs[j], float(u), last, 0.0, failed).total
        last = float(u)
    total += cost.terminal(xs[-1], 0.0).total
    return float(total)


def _symmetrize_with_min_diagonal(matrix: np.ndarray, min_diag: float) -> np.ndarray:
    out = 0.5 * (matrix + matrix.T)
    diag = np.diag(out).copy()
    diag[diag < min_diag] = min_diag
    np.fill_diagonal(out, diag)
    return out
