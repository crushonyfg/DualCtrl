"""Official literature baselines for the benchmark report.

The public runners should use only these controllers plus the oracle. Debug
controllers in other files are intentionally excluded from official tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss

from benchmarks.cartpole_twin.costs import CartPoleCost
from benchmarks.cartpole_twin.dynamics import CartPoleParams, cartpole_step, reference_position
from benchmarks.scalar_dual.costs import ScalarCost
from benchmarks.scalar_dual.filters import GaussianBelief


@dataclass(frozen=True)
class OfficialScalarConfig:
    horizon: int = 3
    action_low: float = -3.0
    action_high: float = 3.0
    action_grid_size: int = 31
    process_var: float = 0.1
    kh_quadrature_points: int = 7
    smpc_dual_horizon: int = 2
    smpc_scenarios: int = 3
    tvgp_lengthscale: float = 1.0
    tvgp_noise_var: float = 1e-3
    tvgp_epsilon: float = 0.02
    tvgp_delta: float = 0.1

    @property
    def action_grid(self) -> np.ndarray:
        return np.linspace(self.action_low, self.action_high, self.action_grid_size)


@dataclass(frozen=True)
class OfficialCartPoleConfig:
    horizon: int = 8
    action_grid_size: int = 9
    theta_process_var: float = 1e-4
    theta_obs_var: float = 0.02
    kh_quadrature_points: int = 5
    smpc_dual_horizon: int = 2
    smpc_scenarios: int = 3
    smpc_noise_std: tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01)
    tvgp_lengthscale: float = 1.0
    tvgp_noise_var: float = 1e-3
    tvgp_epsilon: float = 0.02
    tvgp_delta: float = 0.1

    @property
    def action_grid(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.action_grid_size)


class KHDualControlScalar:
    """Klenske-Hennig style approximate dual control for scalar dynamics.

    Uses the paper's core augmented-belief mechanism for linear-in-parameter
    Gaussian dynamics: root actions are evaluated by propagating fantasy
    observations, updating the Gaussian posterior, and valuing the resulting
    future control problem. The scalar digital twin is x' = x + b u.
    """

    name = "kh_dual_control"

    def __init__(self, belief: GaussianBelief, cost: ScalarCost, config: OfficialScalarConfig):
        self.belief = belief
        self.cost = cost
        self.config = config

    @property
    def belief_mean(self) -> float:
        return self.belief.mean

    @property
    def belief_var(self) -> float:
        return self.belief.var

    def predict(self) -> None:
        self.belief.predict()

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        self.belief.update(x, u, observed_next_x, obs_var)

    def act(self, x: float, prev_u: float) -> float:
        return _kh_scalar_action(x, prev_u, self.belief, self.cost, self.config)


class ArcariDualSMPCScalar:
    """Arcari et al. dual stochastic MPC with a sampled dual part.

    The implementation follows the paper's dual/exploitation split. The first
    L steps are represented by sampled scenarios that update the information
    state. After L, the branch information state is fixed and exploitation
    controls are optimized branch-wise over the remaining horizon.
    """

    name = "arcari_dual_smpc"

    def __init__(self, belief: GaussianBelief, cost: ScalarCost, config: OfficialScalarConfig, seed: int = 0):
        self.belief = belief
        self.cost = cost
        self.config = config
        self.rng = np.random.default_rng(seed)

    @property
    def belief_mean(self) -> float:
        return self.belief.mean

    @property
    def belief_var(self) -> float:
        return self.belief.var

    def predict(self) -> None:
        self.belief.predict()

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        self.belief.update(x, u, observed_next_x, obs_var)

    def act(self, x: float, prev_u: float) -> float:
        return _arcari_scalar_action(x, prev_u, self.belief, self.cost, self.config, self.rng)


class TVGPLCBScalar:
    """Bogunovic et al. TV-GP-UCB adapted as LCB for cost minimization."""

    name = "tv_gp_lcb"

    def __init__(self, cost: ScalarCost, config: OfficialScalarConfig):
        self.cost = cost
        self.config = config
        self.t = 1
        self.features: list[np.ndarray] = []
        self.times: list[int] = []
        self.values: list[float] = []
        self._last_feature: np.ndarray | None = None
        self._last_prev_u = 0.0
        self._last_nominal_cost = 0.0

    @property
    def belief_mean(self) -> float:
        return float("nan")

    @property
    def belief_var(self) -> float:
        return float("nan")

    def predict(self) -> None:
        pass

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        if self._last_feature is not None:
            realized = self.cost.stage(x, u, self._last_prev_u).total
            self.features.append(self._last_feature)
            self.times.append(self.t)
            self.values.append(realized)
        self.t += 1

    def record_cost(self, feature: np.ndarray, stage_cost: float) -> None:
        self.features.append(feature)
        self.times.append(self.t)
        self.values.append(stage_cost)
        self.t += 1

    def act(self, x: float, prev_u: float) -> float:
        best_u = 0.0
        best_lcb = float("inf")
        beta = 2.0 * np.log((len(self.config.action_grid) * np.pi**2 * max(self.t, 1) ** 2) / (6.0 * self.config.tvgp_delta))
        for u in self.config.action_grid:
            u = float(u)
            feature = np.array([x, prev_u, u], dtype=float)
            mu, var = _tv_gp_posterior(feature, self.t, self.features, self.times, self.values, self.config)
            nominal = self.cost.stage(x, u, prev_u).total
            lcb = mu - np.sqrt(beta) * np.sqrt(max(var, 0.0))
            if lcb < best_lcb:
                best_lcb = lcb
                best_u = u
                self._last_feature = feature
                self._last_prev_u = prev_u
                self._last_nominal_cost = nominal
        return best_u


class OracleTrendScalar:
    """Oracle MPC knowing current/future theta trend and true physical gap."""

    name = "oracle_trend"

    def __init__(self, b_path: np.ndarray, cost: ScalarCost, config: OfficialScalarConfig, discrepancy_quadratic: float = 0.0):
        self.b_path = b_path
        self.cost = cost
        self.config = config
        self.discrepancy_quadratic = discrepancy_quadratic
        self.t = 0

    @property
    def belief_mean(self) -> float:
        return float(self.b_path[min(self.t, len(self.b_path) - 1)])

    @property
    def belief_var(self) -> float:
        return 0.0

    def predict(self) -> None:
        pass

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        self.t += 1

    def act(self, x: float, prev_u: float) -> float:
        horizon_b = self.b_path[self.t : min(len(self.b_path), self.t + self.config.horizon)]
        return _oracle_scalar_action(x, prev_u, horizon_b, self.cost, self.config, self.discrepancy_quadratic)


class KHDualControlCartPole:
    name = "kh_dual_control"

    def __init__(self, belief: GaussianBelief, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig):
        self.belief = belief
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.t = 0

    @property
    def belief_mean(self) -> float:
        return self.belief.mean

    @property
    def belief_var(self) -> float:
        return self.belief.var

    def predict(self) -> None:
        self.belief.predict()

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        theta_hat = _cartpole_theta_pseudo_observation(state, action, next_state, self.dynamics)
        self.belief.update(0.0, 1.0, theta_hat, self.config.theta_obs_var)
        self.t += 1

    def act(self, state: np.ndarray, prev_action: float) -> float:
        return _kh_cartpole_action(state, prev_action, self.t, self.belief, self.dynamics, self.cost, self.config)


class ArcariDualSMPCCartPole:
    name = "arcari_dual_smpc"

    def __init__(self, belief: GaussianBelief, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig, seed: int = 0):
        self.belief = belief
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.t = 0

    @property
    def belief_mean(self) -> float:
        return self.belief.mean

    @property
    def belief_var(self) -> float:
        return self.belief.var

    def predict(self) -> None:
        self.belief.predict()

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        theta_hat = _cartpole_theta_pseudo_observation(state, action, next_state, self.dynamics)
        self.belief.update(0.0, 1.0, theta_hat, self.config.theta_obs_var)
        self.t += 1

    def act(self, state: np.ndarray, prev_action: float) -> float:
        return _arcari_cartpole_action(state, prev_action, self.t, self.belief, self.dynamics, self.cost, self.config, self.rng)


class TVGPLCBCartPole:
    name = "tv_gp_lcb"

    def __init__(self, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig):
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.t = 1
        self.features: list[np.ndarray] = []
        self.times: list[int] = []
        self.values: list[float] = []
        self._last_feature: np.ndarray | None = None
        self._last_prev_action = 0.0
        self._last_nominal_cost = 0.0

    @property
    def belief_mean(self) -> float:
        return float("nan")

    @property
    def belief_var(self) -> float:
        return float("nan")

    def predict(self) -> None:
        pass

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        if self._last_feature is not None:
            failed = self.cost.failed(next_state)
            value = self.cost.stage(state, action, self._last_prev_action, reference_position(self.t - 1), failed).total
            self.features.append(self._last_feature)
            self.times.append(self.t)
            self.values.append(value)
        self.t += 1

    def act(self, state: np.ndarray, prev_action: float) -> float:
        best_u = 0.0
        best_lcb = float("inf")
        beta = 2.0 * np.log((len(self.config.action_grid) * np.pi**2 * max(self.t, 1) ** 2) / (6.0 * self.config.tvgp_delta))
        for u in self.config.action_grid:
            u = float(u)
            feature = np.concatenate([state, np.array([prev_action, u])])
            mu, var = _tv_gp_posterior_cartpole(feature, self.t, self.features, self.times, self.values, self.config)
            pred, _ = cartpole_step(state, u, 1.0, self.dynamics)
            failed = self.cost.failed(pred)
            nominal = self.cost.stage(state, u, prev_action, reference_position(self.t - 1), failed).total + self.cost.terminal(pred, reference_position(self.t)).total
            lcb = mu - np.sqrt(beta) * np.sqrt(max(var, 0.0))
            if lcb < best_lcb:
                best_lcb = lcb
                best_u = u
                self._last_feature = feature
                self._last_prev_action = prev_action
                self._last_nominal_cost = nominal
        return best_u


class OracleTrendCartPole:
    name = "oracle_trend"

    def __init__(self, theta_path: np.ndarray, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig):
        self.theta_path = theta_path
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.t = 0

    @property
    def belief_mean(self) -> float:
        return float(self.theta_path[min(self.t, len(self.theta_path) - 1)])

    @property
    def belief_var(self) -> float:
        return 0.0

    def predict(self) -> None:
        pass

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        self.t += 1

    def act(self, state: np.ndarray, prev_action: float) -> float:
        return _oracle_cartpole_action(state, prev_action, self.t, self.theta_path, self.dynamics, self.cost, self.config)


def _kh_scalar_action(x: float, prev_u: float, belief: GaussianBelief, cost: ScalarCost, config: OfficialScalarConfig) -> float:
    nodes, weights = hermgauss(config.kh_quadrature_points)
    best_u, best_value = 0.0, float("inf")
    for u in config.action_grid:
        u = float(u)
        immediate = cost.stage(x, u, prev_u).total
        pred_mean = x + belief.mean * u
        pred_var = max(config.process_var + belief.var * u * u, 1e-12)
        branch = 0.0
        for node, weight in zip(nodes, weights):
            x_next = pred_mean + np.sqrt(2.0 * pred_var) * float(node)
            fantasy = belief.copy()
            fantasy.update(x, u, float(x_next), config.process_var)
            branch += float(weight) * _scalar_exploitation_value(float(x_next), u, fantasy.mean, fantasy.var, cost, config, config.horizon - 1)
        value = immediate + branch / np.sqrt(np.pi)
        if value < best_value:
            best_value = value
            best_u = u
    return best_u


def _arcari_scalar_action(x: float, prev_u: float, belief: GaussianBelief, cost: ScalarCost, config: OfficialScalarConfig, rng: np.random.Generator) -> float:
    """Root action for Arcari et al. explicit dual/exploitation tree.

    For the no-structural-mode case, n_m=1. The first L stages form the
    dual scenario tree. Each node chooses its own action; each action branches
    into parameter/noise scenarios and updates the node information state. At
    depth L, the branch belief is fixed and an exploitation tail is solved.
    """
    dual_horizon = min(config.smpc_dual_horizon, config.horizon)
    best_u, best_value = 0.0, float("inf")
    for u in config.action_grid:
        u = float(u)
        value = cost.stage(x, u, prev_u).total
        value += _arcari_scalar_child_expectation(x, u, belief, cost, config, depth=1, dual_horizon=dual_horizon)
        if value < best_value:
            best_value = value
            best_u = u
    return best_u


def _arcari_scalar_node_value(
    x: float,
    prev_u: float,
    belief: GaussianBelief,
    cost: ScalarCost,
    config: OfficialScalarConfig,
    depth: int,
    dual_horizon: int,
) -> float:
    if depth >= dual_horizon:
        return _scalar_exploitation_value(x, prev_u, belief.mean, belief.var, cost, config, config.horizon - depth)
    best = float("inf")
    for u in config.action_grid:
        u = float(u)
        value = cost.stage(x, u, prev_u).total
        value += _arcari_scalar_child_expectation(x, u, belief, cost, config, depth + 1, dual_horizon)
        if value < best:
            best = value
    return best


def _arcari_scalar_child_expectation(
    x: float,
    u: float,
    belief: GaussianBelief,
    cost: ScalarCost,
    config: OfficialScalarConfig,
    depth: int,
    dual_horizon: int,
) -> float:
    scenarios = _sigma_scenarios(belief.mean, belief.var, config.smpc_scenarios)
    noise_nodes, noise_weights = _normal_sigma(config.process_var)
    expected = 0.0
    for b, bw in scenarios:
        for eps, ew in zip(noise_nodes, noise_weights):
            x_next = x + b * u + eps
            branch_belief = belief.copy()
            branch_belief.update(x, u, float(x_next), config.process_var)
            expected += bw * ew * _arcari_scalar_node_value(
                float(x_next), u, branch_belief, cost, config, depth, dual_horizon
            )
    return expected


def _scalar_exploitation_value(x: float, prev_u: float, mean: float, var: float, cost: ScalarCost, config: OfficialScalarConfig, horizon: int) -> float:
    if horizon <= 0:
        return cost.terminal(x).total
    best = float("inf")
    for u in config.action_grid:
        u = float(u)
        next_mean = x + mean * u
        expected_next_sq = next_mean * next_mean + var * u * u + config.process_var
        terminal = cost.config.terminal_weight * expected_next_sq if horizon == 1 else _scalar_exploitation_value(next_mean, u, mean, var, cost, config, horizon - 1)
        value = cost.stage(x, u, prev_u).total + terminal
        if value < best:
            best = value
    return best


def _oracle_scalar_action(x: float, prev_u: float, b_future: np.ndarray, cost: ScalarCost, config: OfficialScalarConfig, discrepancy_quadratic: float = 0.0) -> float:
    best_u, best_value = 0.0, float("inf")
    for u in config.action_grid:
        u = float(u)
        value = cost.stage(x, u, prev_u).total
        value += _oracle_scalar_value(x + b_future[0] * u + discrepancy_quadratic * u * u, u, b_future[1:], cost, config, discrepancy_quadratic)
        if value < best_value:
            best_value = value
            best_u = float(u)
    return best_u


def _oracle_scalar_value(x: float, prev_u: float, b_future: np.ndarray, cost: ScalarCost, config: OfficialScalarConfig, discrepancy_quadratic: float = 0.0) -> float:
    if len(b_future) == 0:
        return cost.terminal(x).total
    best = float("inf")
    for u in config.action_grid:
        u = float(u)
        value = cost.stage(x, u, prev_u).total + _oracle_scalar_value(x + b_future[0] * u + discrepancy_quadratic * u * u, u, b_future[1:], cost, config, discrepancy_quadratic)
        best = min(best, value)
    return best


def _tv_gp_posterior(feature: np.ndarray, t: int, features: list[np.ndarray], times: list[int], values: list[float], config: OfficialScalarConfig) -> tuple[float, float]:
    if not features:
        return 0.0, 1.0
    X = np.vstack(features)
    y = np.asarray(values, dtype=float)
    ts = np.asarray(times, dtype=float)
    spatial = _se_kernel_matrix(X, X, config.tvgp_lengthscale)
    temporal = (1.0 - config.tvgp_epsilon) ** (np.abs(ts[:, None] - ts[None, :]) / 2.0)
    K = spatial * temporal + (config.tvgp_noise_var + 1e-8) * np.eye(len(y))
    kx = _se_kernel_matrix(X, feature[None, :], config.tvgp_lengthscale).ravel()
    kt = (1.0 - config.tvgp_epsilon) ** ((t - ts) / 2.0)
    k = kx * kt
    return _gp_predict(K, k, y)


def _tv_gp_posterior_cartpole(feature: np.ndarray, t: int, features: list[np.ndarray], times: list[int], values: list[float], config: OfficialCartPoleConfig) -> tuple[float, float]:
    if not features:
        return 0.0, 1.0
    X = np.vstack(features)
    y = np.asarray(values, dtype=float)
    ts = np.asarray(times, dtype=float)
    spatial = _se_kernel_matrix(X, X, config.tvgp_lengthscale)
    temporal = (1.0 - config.tvgp_epsilon) ** (np.abs(ts[:, None] - ts[None, :]) / 2.0)
    K = spatial * temporal + (config.tvgp_noise_var + 1e-8) * np.eye(len(y))
    kx = _se_kernel_matrix(X, feature[None, :], config.tvgp_lengthscale).ravel()
    kt = (1.0 - config.tvgp_epsilon) ** ((t - ts) / 2.0)
    k = kx * kt
    return _gp_predict(K, k, y)


def _se_kernel_matrix(X: np.ndarray, Y: np.ndarray, lengthscale: float) -> np.ndarray:
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / (lengthscale * lengthscale))


def _gp_predict(K: np.ndarray, k: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        v = np.linalg.solve(L, k)
        mu = float(k @ alpha)
        var = float(max(1.0 - v @ v, 1e-9))
        return mu, var
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K)
        return float(k @ K_inv @ y), float(max(1.0 - k @ K_inv @ k, 1e-9))


def _sigma_scenarios(mean: float, var: float, n: int) -> list[tuple[float, float]]:
    if n <= 1 or var <= 1e-12:
        return [(mean, 1.0)]
    nodes, weights = hermgauss(n)
    return [(float(mean + np.sqrt(2.0 * var) * node), float(weight / np.sqrt(np.pi))) for node, weight in zip(nodes, weights)]


def _normal_sigma(var: float) -> tuple[np.ndarray, np.ndarray]:
    if var <= 1e-12:
        return np.array([0.0]), np.array([1.0])
    return np.array([-1.0, 0.0, 1.0]) * np.sqrt(var), np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


def _cartpole_theta_pseudo_observation(state: np.ndarray, action: float, next_state: np.ndarray, dynamics: CartPoleParams) -> float:
    candidates = np.linspace(0.55, 1.45, 21)
    errors = []
    for theta in candidates:
        pred, _ = cartpole_step(state, action, float(theta), dynamics)
        errors.append(float(np.sum((pred - next_state) ** 2)))
    return float(candidates[int(np.argmin(errors))])


def _kh_cartpole_action(state: np.ndarray, prev_action: float, t: int, belief: GaussianBelief, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig) -> float:
    scenarios = _sigma_scenarios(belief.mean, belief.var, config.kh_quadrature_points)
    best_u, best_value = 0.0, float("inf")
    for action in config.action_grid:
        action = float(action)
        value = 0.0
        for theta, weight in scenarios:
            next_state, _ = cartpole_step(state, action, float(theta), dynamics)
            failed = cost.failed(next_state)
            stage = cost.stage(state, action, prev_action, reference_position(t), failed).total
            fantasy = belief.copy()
            fantasy.update(0.0, 1.0, float(theta), config.theta_obs_var)
            value += weight * (stage + _cartpole_exploitation_value(next_state, action, t + 1, fantasy.mean, dynamics, cost, config, config.horizon - 1))
        if value < best_value:
            best_value = value
            best_u = action
    return best_u


def _arcari_cartpole_action(state: np.ndarray, prev_action: float, t: int, belief: GaussianBelief, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig, rng: np.random.Generator) -> float:
    dual_horizon = min(config.smpc_dual_horizon, config.horizon)
    best_u, best_value = 0.0, float("inf")
    for action in config.action_grid:
        action = float(action)
        value = _arcari_cartpole_child_expectation(
            state, action, prev_action, t, belief, dynamics, cost, config, depth=1, dual_horizon=dual_horizon
        )
        if value < best_value:
            best_value = value
            best_u = action
    return best_u


def _arcari_cartpole_node_value(
    state: np.ndarray,
    prev_action: float,
    t: int,
    belief: GaussianBelief,
    dynamics: CartPoleParams,
    cost: CartPoleCost,
    config: OfficialCartPoleConfig,
    depth: int,
    dual_horizon: int,
) -> float:
    if depth >= dual_horizon:
        return _cartpole_exploitation_value(state, prev_action, t, belief.mean, dynamics, cost, config, config.horizon - depth)
    best = float("inf")
    for action in config.action_grid:
        action = float(action)
        failed = cost.failed(state)
        value = cost.stage(state, action, prev_action, reference_position(t), failed).total
        value += _arcari_cartpole_child_expectation(state, action, prev_action, t, belief, dynamics, cost, config, depth + 1, dual_horizon)
        if value < best:
            best = value
    return best


def _arcari_cartpole_child_expectation(
    state: np.ndarray,
    action: float,
    prev_action: float,
    t: int,
    belief: GaussianBelief,
    dynamics: CartPoleParams,
    cost: CartPoleCost,
    config: OfficialCartPoleConfig,
    depth: int,
    dual_horizon: int,
) -> float:
    scenarios = _sigma_scenarios(belief.mean, belief.var, config.smpc_scenarios)
    noise_scenarios = _cartpole_noise_scenarios(config)
    expected = 0.0
    for theta, theta_weight in scenarios:
        nominal_next, _ = cartpole_step(state, action, float(theta), dynamics)
        for noise, noise_weight in noise_scenarios:
            next_state = nominal_next + noise
            failed = cost.failed(next_state)
            stage = cost.stage(state, action, prev_action, reference_position(t), failed).total
            if failed:
                branch = stage + cost.config.failure_cost * max(config.horizon - depth, 0)
            else:
                branch_belief = belief.copy()
                theta_obs = _cartpole_theta_pseudo_observation(state, action, next_state, dynamics)
                branch_belief.update(0.0, 1.0, theta_obs, config.theta_obs_var)
                branch = stage + _arcari_cartpole_node_value(
                    next_state, action, t + 1, branch_belief, dynamics, cost, config, depth, dual_horizon
                )
            expected += theta_weight * noise_weight * branch
    return expected


def _cartpole_noise_scenarios(config: OfficialCartPoleConfig) -> list[tuple[np.ndarray, float]]:
    std = np.asarray(config.smpc_noise_std, dtype=float)
    scenarios = [(np.zeros(4, dtype=float), 2.0 / 3.0)]
    nonzero = np.where(std > 0.0)[0]
    if len(nonzero) == 0:
        return [(np.zeros(4, dtype=float), 1.0)]
    side_weight = 1.0 / (6.0 * len(nonzero))
    for idx in nonzero:
        noise = np.zeros(4, dtype=float)
        noise[idx] = std[idx]
        scenarios.append((noise, side_weight))
        noise = np.zeros(4, dtype=float)
        noise[idx] = -std[idx]
        scenarios.append((noise, side_weight))
    total = sum(weight for _, weight in scenarios)
    return [(noise, weight / total) for noise, weight in scenarios]


def _cartpole_exploitation_value(state: np.ndarray, prev_action: float, t: int, theta: float, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig, horizon: int) -> float:
    if horizon <= 0:
        return cost.terminal(state, reference_position(t)).total
    best = float("inf")
    for action in config.action_grid:
        action = float(action)
        next_state, _ = cartpole_step(state, action, theta, dynamics)
        failed = cost.failed(next_state)
        value = cost.stage(state, action, prev_action, reference_position(t), failed).total
        if failed:
            value += cost.config.failure_cost * max(horizon - 1, 0)
        else:
            value += _cartpole_exploitation_value(next_state, action, t + 1, theta, dynamics, cost, config, horizon - 1)
        best = min(best, value)
    return best


def _oracle_cartpole_action(state: np.ndarray, prev_action: float, t: int, theta_path: np.ndarray, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig) -> float:
    best_u, best_value = 0.0, float("inf")
    for action in config.action_grid:
        action = float(action)
        theta = float(theta_path[min(t, len(theta_path) - 1)])
        next_state, _ = cartpole_step(state, action, theta, dynamics)
        failed = cost.failed(next_state)
        value = cost.stage(state, action, prev_action, reference_position(t), failed).total
        value += _oracle_cartpole_value(next_state, action, t + 1, theta_path, dynamics, cost, config, config.horizon - 1)
        if value < best_value:
            best_value = value
            best_u = action
    return best_u


def _oracle_cartpole_value(state: np.ndarray, prev_action: float, t: int, theta_path: np.ndarray, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig, horizon: int) -> float:
    if horizon <= 0:
        return cost.terminal(state, reference_position(t)).total
    best = float("inf")
    for action in config.action_grid:
        action = float(action)
        theta = float(theta_path[min(t, len(theta_path) - 1)])
        next_state, _ = cartpole_step(state, action, theta, dynamics)
        failed = cost.failed(next_state)
        value = cost.stage(state, action, prev_action, reference_position(t), failed).total
        if failed:
            value += cost.config.failure_cost * max(horizon - 1, 0)
        else:
            value += _oracle_cartpole_value(next_state, action, t + 1, theta_path, dynamics, cost, config, horizon - 1)
        best = min(best, value)
    return best
