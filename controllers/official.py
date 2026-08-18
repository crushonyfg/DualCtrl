"""Official literature baselines for the benchmark report.

The public runners should use only these controllers plus the oracle. Debug
controllers in other files are intentionally excluded from official tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.hermite import hermgauss

from benchmarks.cartpole_twin.costs import CartPoleCost
from benchmarks.cartpole_twin.dynamics import CartPoleParams, cartpole_step, reference_position
from benchmarks.scalar_dual.costs import ScalarCost
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers.kh_gp import KHGPConfig, KHGPControllerCartPole
from controllers.kh_strict import KHScalarADConfig, KHScalarADController
from controllers.tv_gp_ucb import TVGPUCBConfig, RealizedCostBanditAdapter


@dataclass(frozen=True)
class OfficialScalarConfig:
    horizon: int = 3
    # Finite-action benchmark assumption: Arcari et al.'s continuous NLP in
    # Eq. (10) is solved exactly over this explicit finite grid, while keeping
    # the same scenario-tree objective over all node actions.
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
    # Finite-action benchmark assumption; see OfficialScalarConfig.
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


class KHDualControlScalar(KHScalarADController):
    """Strict Klenske-Hennig scalar AD baseline routed through kh_strict.py."""

    name = "kh_dual_control"

    def __init__(self, belief: GaussianBelief, cost: ScalarCost, config: OfficialScalarConfig):
        kh_config = KHScalarADConfig(
            a=1.0,
            process_var=config.process_var,
            obs_var=0.0,
            horizon=config.horizon,
            action_low=config.action_low,
            action_high=config.action_high,
            action_grid_size=config.action_grid_size,
        )
        super().__init__(belief, cost, kh_config)


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

    def build_tree_for_action(self, x: float, prev_u: float, root_action: float) -> ArcariScalarScenarioTree:
        """Expose the explicit Eq. (5)-(10) tree for tests and diagnostics."""
        return build_arcari_scalar_scenario_tree(x, prev_u, root_action, self.belief, self.cost, self.config)


class TVGPLCBScalar:
    """Strict Bogunovic et al. finite-action TV-GP-LCB for realized costs.

    Each feasible root action in the current scalar context is a bandit arm with
    feature [x, previous_action, action]. The update uses only the realized stage
    cost supplied after the environment step; no nominal dynamics or terminal-cost
    heuristic is added to the acquisition.
    """

    name = "tv_gp_lcb"

    def __init__(self, cost: ScalarCost, config: OfficialScalarConfig):
        self.cost = cost
        self.config = config
        self.t = 1
        self._last_x = 0.0
        self._last_u = 0.0
        self._last_prev_u = 0.0
        self._pending_cost = False
        self.adapter = RealizedCostBanditAdapter(
            action_provider=lambda _context: [float(u) for u in self.config.action_grid],
            feature_map=lambda context, action: np.array([float(context[0]), float(context[1]), float(action)], dtype=float),
            config=TVGPUCBConfig(
                epsilon=self.config.tvgp_epsilon,
                noise_var=self.config.tvgp_noise_var,
                delta=self.config.tvgp_delta,
                lengthscale=self.config.tvgp_lengthscale,
            ),
        )

    @property
    def belief_mean(self) -> float:
        return float("nan")

    @property
    def belief_var(self) -> float:
        return float("nan")

    def predict(self) -> None:
        pass

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        self._finalize_pending_realized_cost()

    def _finalize_pending_realized_cost(self) -> None:
        if not self._pending_cost:
            return
        realized = self.cost.stage(self._last_x, self._last_u, self._last_prev_u).total
        self.adapter.update(realized)
        self.t = self.adapter.t
        self._pending_cost = False

    def record_cost(self, feature: np.ndarray, stage_cost: float) -> None:
        context = (float(feature[0]), float(feature[1]))
        action = float(feature[2])
        self.adapter.update(float(stage_cost), action=action, context=context)
        self.t = self.adapter.t

    def act(self, x: float, prev_u: float) -> float:
        self._finalize_pending_realized_cost()
        u = float(self.adapter.select((float(x), float(prev_u))))
        self._last_x = float(x)
        self._last_u = u
        self._last_prev_u = float(prev_u)
        self._pending_cost = True
        self.t = self.adapter.t
        return u


class OracleTrendScalar:
    """Pathwise oracle knowing future theta, physical gap, and realized noise."""

    name = "oracle_trend"

    def __init__(
        self,
        b_path: np.ndarray,
        cost: ScalarCost,
        config: OfficialScalarConfig,
        discrepancy_quadratic: float = 0.0,
        noise_path: np.ndarray | None = None,
    ):
        self.b_path = b_path
        self.cost = cost
        self.config = config
        self.discrepancy_quadratic = discrepancy_quadratic
        self.noise_path = np.zeros_like(b_path) if noise_path is None else np.asarray(noise_path, dtype=float)
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
        horizon_noise = self.noise_path[self.t : min(len(self.noise_path), self.t + self.config.horizon)]
        return _oracle_scalar_action(x, prev_u, horizon_b, horizon_noise, self.cost, self.config, self.discrepancy_quadratic)


class KHDualControlCartPole(KHGPControllerCartPole):
    """Official CartPole KH baseline: finite-feature GP AD, not theta pseudo-observation."""

    def __init__(self, belief: GaussianBelief, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig):
        del belief
        kh_config = KHGPConfig(
            horizon=config.horizon,
            action_grid_size=config.action_grid_size,
            lengthscale=config.tvgp_lengthscale,
            gp_noise_var=config.tvgp_noise_var,
        )
        super().__init__(dynamics, cost, kh_config)
        self.official_config = config


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

    def build_tree_for_action(self, state: np.ndarray, prev_action: float, root_action: float) -> "ArcariCartPoleScenarioTree":
        """Expose the explicit Eq. (5)-(10) CartPole tree for tests and diagnostics."""
        return build_arcari_cartpole_scenario_tree(state, prev_action, self.t, root_action, self.belief, self.dynamics, self.cost, self.config)


@dataclass
class ArcariScalarTreeNode:
    """One information-state node in Arcari et al.'s dual scenario tree."""

    node_id: int
    depth: int
    parent_id: int | None
    branch_index: int
    mode_index: int
    sample_index: int
    probability: float
    state: float
    prev_action: float
    belief: GaussianBelief
    mode_probability: float = 1.0
    parameter_sample: float | None = None
    noise_sample: float | None = None
    action: float | None = None
    stage_cost: float | None = None
    children: list[int] = field(default_factory=list)


@dataclass
class ArcariScalarScenarioTree:
    """Explicit tree and unified objective for Eqs. (5)-(10), nm=1."""

    nodes: list[ArcariScalarTreeNode]
    levels: list[list[int]]
    dual_cost: float
    exploitation_cost: float
    total_cost: float
    root_action: float
    dual_horizon: int
    prediction_horizon: int
    branch_factor: int

    def node_count_by_depth(self) -> list[int]:
        return [len(level) for level in self.levels]


@dataclass
class ArcariCartPoleTreeNode:
    """One CartPole information-state node in Arcari et al.'s scenario tree.

    The node stores the physical state, parent/child topology, recursive Eq. (7)
    probability weight, and the branch-specific posterior information state used
    to choose subsequent dual actions and the fixed-information exploitation tail.
    ``transition_stage_cost`` is the cost of the parent action on the sampled
    transition that created this node.
    """

    node_id: int
    depth: int
    parent_id: int | None
    branch_index: int
    mode_index: int
    sample_index: int
    probability: float
    state: np.ndarray
    prev_action: float
    time: int
    belief: GaussianBelief
    branch_weight: float = 1.0
    mode_probability: float = 1.0
    parameter_sample: float | None = None
    process_noise_sample: np.ndarray | None = None
    action: float | None = None
    transition_stage_cost: float | None = None
    failed: bool = False
    children: list[int] = field(default_factory=list)

    @property
    def information_state(self) -> GaussianBelief:
        return self.belief


@dataclass
class ArcariCartPoleScenarioTree:
    """Explicit CartPole scenario tree and Eq. (10) objective decomposition."""

    nodes: list[ArcariCartPoleTreeNode]
    levels: list[list[int]]
    dual_cost: float
    exploitation_cost: float
    total_cost: float
    root_action: float
    dual_horizon: int
    prediction_horizon: int
    branch_factor: int

    def node_count_by_depth(self) -> list[int]:
        return [len(level) for level in self.levels]


class TVGPLCBCartPole:
    """Strict Bogunovic et al. finite-action TV-GP-LCB for CartPole costs.

    Each feasible root action in the current state/context is a bandit arm with
    feature concat(state, previous_action, action). The update uses only the
    realized stage cost after the environment step; no nominal CartPole rollout,
    terminal cost, or model-predicted failure heuristic enters the acquisition.
    """

    name = "tv_gp_lcb"

    def __init__(self, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig):
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.t = 1
        self._last_state = np.zeros(4, dtype=float)
        self._last_action = 0.0
        self._last_prev_action = 0.0
        self._last_failed = False
        self._pending_cost = False
        self.adapter = RealizedCostBanditAdapter(
            action_provider=lambda _context: [float(u) for u in self.config.action_grid],
            feature_map=lambda context, action: np.concatenate([np.asarray(context[0], dtype=float), np.array([float(context[1]), float(action)])]),
            config=TVGPUCBConfig(
                epsilon=self.config.tvgp_epsilon,
                noise_var=self.config.tvgp_noise_var,
                delta=self.config.tvgp_delta,
                lengthscale=self.config.tvgp_lengthscale,
            ),
        )

    @property
    def belief_mean(self) -> float:
        return float("nan")

    @property
    def belief_var(self) -> float:
        return float("nan")

    def predict(self) -> None:
        pass

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        self._last_failed = self.cost.failed(next_state)
        self._finalize_pending_realized_cost()

    def _finalize_pending_realized_cost(self) -> None:
        if not self._pending_cost:
            return
        value = self.cost.stage(self._last_state, self._last_action, self._last_prev_action, reference_position(self.t - 1), self._last_failed).total
        self.adapter.update(value)
        self.t = self.adapter.t
        self._pending_cost = False
        self._last_failed = False

    def act(self, state: np.ndarray, prev_action: float) -> float:
        if self._pending_cost:
            self._last_failed = self.cost.failed(np.asarray(state, dtype=float))
        self._finalize_pending_realized_cost()
        action = float(self.adapter.select((np.asarray(state, dtype=float).copy(), float(prev_action))))
        self._last_state = np.asarray(state, dtype=float).copy()
        self._last_action = action
        self._last_prev_action = float(prev_action)
        self._pending_cost = True
        self.t = self.adapter.t
        return action


class OracleTrendCartPole:
    name = "oracle_trend"

    def __init__(
        self,
        theta_path: np.ndarray,
        dynamics: CartPoleParams,
        cost: CartPoleCost,
        config: OfficialCartPoleConfig,
        noise_path: np.ndarray | None = None,
    ):
        self.theta_path = theta_path
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.noise_path = np.zeros((len(theta_path), 4), dtype=float) if noise_path is None else np.asarray(noise_path, dtype=float)
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
        return _oracle_cartpole_action(state, prev_action, self.t, self.theta_path, self.noise_path, self.dynamics, self.cost, self.config)


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

    For the no-structural-mode case, n_m=1. We make the benchmark's finite
    action grid explicit and minimize the paper's unified Eq. (10) objective
    over all node actions in the sampled dual tree plus branch-wise exploitation
    tails, rather than using a one-step lookahead surrogate.
    """
    best_u, best_value = 0.0, float("inf")
    for u in config.action_grid:
        tree = build_arcari_scalar_scenario_tree(x, prev_u, float(u), belief, cost, config)
        if tree.total_cost < best_value:
            best_value = tree.total_cost
            best_u = float(u)
    return best_u


def build_arcari_scalar_scenario_tree(
    x: float,
    prev_u: float,
    root_action: float,
    belief: GaussianBelief,
    cost: ScalarCost,
    config: OfficialScalarConfig,
) -> ArcariScalarScenarioTree:
    """Build and optimize Arcari et al.'s scalar scenario tree (Eqs. 5-10).

    The paper samples parameters and noise at every dual node (Eq. 5), updates
    the information state of each child node via Bayes' rule (Eq. 4, used by
    GaussianBelief.update), assigns recursive node probabilities/weights
    (Eq. 7; nm=1 here, so mode probability is one), and then merges the dual
    and exploitation parts into one objective over all tree-node actions
    (Eqs. 6 and 10). This finite-action benchmark enumerates all actions on the
    configured grid instead of calling a continuous nonlinear optimizer.
    """
    dual_horizon = min(config.smpc_dual_horizon, config.horizon)
    root_parameter_scenarios = _sigma_scenarios(belief.mean, belief.var, config.smpc_scenarios)
    noise_nodes, noise_weights = _normal_sigma(config.process_var)
    branch_factor = len(root_parameter_scenarios) * len(noise_nodes)

    root = ArcariScalarTreeNode(
        node_id=0,
        depth=0,
        parent_id=None,
        branch_index=0,
        mode_index=0,
        sample_index=0,
        probability=1.0,
        state=float(x),
        prev_action=float(prev_u),
        belief=belief.copy(),
    )
    nodes = [root]
    levels: list[list[int]] = [[0]]
    next_id = 1

    for depth in range(dual_horizon):
        next_level: list[int] = []
        for node_id in levels[depth]:
            node = nodes[node_id]
            if depth == 0:
                action = float(root_action)
                value_to_go = None
            else:
                action, value_to_go = _arcari_scalar_best_dual_action_for_node(
                    node.state,
                    node.prev_action,
                    node.belief,
                    cost,
                    config,
                    depth,
                    dual_horizon,
                )
            node.action = action
            node.stage_cost = cost.stage(node.state, action, node.prev_action).total

            parameter_scenarios = _sigma_scenarios(node.belief.mean, node.belief.var, config.smpc_scenarios)
            for sample_index, ((theta, theta_weight), (noise, noise_weight)) in enumerate(
                (param_noise for param_noise in ((ps, nw) for ps in parameter_scenarios for nw in zip(noise_nodes, noise_weights)))
            ):
                child_belief = node.belief.copy()
                child_state = float(node.state + theta * action + float(noise))
                child_belief.update(node.state, action, child_state, config.process_var)
                child_weight = float(theta_weight * noise_weight)
                child = ArcariScalarTreeNode(
                    node_id=next_id,
                    depth=depth + 1,
                    parent_id=node_id,
                    branch_index=sample_index,
                    mode_index=0,
                    sample_index=sample_index,
                    probability=node.probability * child_weight,
                    state=child_state,
                    prev_action=action,
                    belief=child_belief,
                    mode_probability=1.0,
                    parameter_sample=float(theta),
                    noise_sample=float(noise),
                )
                nodes.append(child)
                node.children.append(next_id)
                next_level.append(next_id)
                next_id += 1
        levels.append(next_level)

    dual_cost = 0.0
    for depth in range(dual_horizon):
        for node_id in levels[depth]:
            node = nodes[node_id]
            if node.stage_cost is None:
                raise RuntimeError("dual node was not assigned an action")
            dual_cost += node.probability * node.stage_cost
    exploitation_cost = 0.0
    for leaf_id in levels[dual_horizon]:
        leaf = nodes[leaf_id]
        tail = _scalar_exploitation_value(
            leaf.state,
            leaf.prev_action,
            leaf.belief.mean,
            leaf.belief.var,
            cost,
            config,
            config.horizon - dual_horizon,
        )
        exploitation_cost += leaf.probability * tail

    return ArcariScalarScenarioTree(
        nodes=nodes,
        levels=levels,
        dual_cost=float(dual_cost),
        exploitation_cost=float(exploitation_cost),
        total_cost=float(dual_cost + exploitation_cost),
        root_action=float(root_action),
        dual_horizon=dual_horizon,
        prediction_horizon=config.horizon,
        branch_factor=branch_factor,
    )


def _arcari_scalar_best_dual_action_for_node(
    x: float,
    prev_u: float,
    belief: GaussianBelief,
    cost: ScalarCost,
    config: OfficialScalarConfig,
    depth: int,
    dual_horizon: int,
) -> tuple[float, float]:
    """Return the optimal finite-grid action for a non-root dual tree node."""
    best_u, best_value = 0.0, float("inf")
    for u in config.action_grid:
        u = float(u)
        value = cost.stage(x, u, prev_u).total
        value += _arcari_scalar_child_expectation(x, u, belief, cost, config, depth + 1, dual_horizon)
        if value < best_value:
            best_value = value
            best_u = u
    return best_u, float(best_value)


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
    _, best_value = _arcari_scalar_best_dual_action_for_node(x, prev_u, belief, cost, config, depth, dual_horizon)
    return best_value


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


def _oracle_scalar_action(x: float, prev_u: float, b_future: np.ndarray, noise_future: np.ndarray, cost: ScalarCost, config: OfficialScalarConfig, discrepancy_quadratic: float = 0.0) -> float:
    best_u, best_value = 0.0, float("inf")
    for u in config.action_grid:
        u = float(u)
        value = cost.stage(x, u, prev_u).total
        value += _oracle_scalar_value(x + b_future[0] * u + discrepancy_quadratic * u * u + noise_future[0], u, b_future[1:], noise_future[1:], cost, config, discrepancy_quadratic)
        if value < best_value:
            best_value = value
            best_u = float(u)
    return best_u


def _oracle_scalar_value(x: float, prev_u: float, b_future: np.ndarray, noise_future: np.ndarray, cost: ScalarCost, config: OfficialScalarConfig, discrepancy_quadratic: float = 0.0) -> float:
    if len(b_future) == 0:
        return cost.terminal(x).total
    best = float("inf")
    for u in config.action_grid:
        u = float(u)
        value = cost.stage(x, u, prev_u).total + _oracle_scalar_value(x + b_future[0] * u + discrepancy_quadratic * u * u + noise_future[0], u, b_future[1:], noise_future[1:], cost, config, discrepancy_quadratic)
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
    """Root action for Arcari et al.'s CartPole dual stochastic MPC tree.

    The benchmark solver is finite-action but preserves Eq. (10): every root
    candidate builds the full sampled dual tree to depth L, optimizes all
    non-root dual-node actions over the same grid, and appends branch-wise
    exploitation tails with information fixed at depth L.
    """
    del rng  # deterministic sigma-point samples make the benchmark reproducible.
    best_u, best_value = 0.0, float("inf")
    for action in config.action_grid:
        tree = build_arcari_cartpole_scenario_tree(state, prev_action, t, float(action), belief, dynamics, cost, config)
        if tree.total_cost < best_value:
            best_value = tree.total_cost
            best_u = float(action)
    return best_u


def build_arcari_cartpole_scenario_tree(
    state: np.ndarray,
    prev_action: float,
    t: int,
    root_action: float,
    belief: GaussianBelief,
    dynamics: CartPoleParams,
    cost: CartPoleCost,
    config: OfficialCartPoleConfig,
) -> ArcariCartPoleScenarioTree:
    """Build and optimize Arcari et al.'s CartPole scenario tree (Eqs. 5-10).

    Eq. (5) is represented by sampled CartPole transitions
    ``x_child = f(x_parent, u_parent, gamma_sample) + w_sample``.  Eq. (7)
    probabilities are recursive parent probabilities times parameter/noise sample
    weights (single structural mode, so mode probability is one).  Each child
    stores its branch-specific Bayesian information-state update from the sampled
    observation.  Eq. (10) is stored as ``dual_cost + exploitation_cost``: the
    dual part sums all sampled transition costs generated by actions on depths
    ``0..L-1``; exploitation solves each depth-L branch with that leaf posterior
    fixed for the remaining prediction horizon ``N-L``.
    """
    dual_horizon = min(config.smpc_dual_horizon, config.horizon)
    parameter_scenarios = _sigma_scenarios(belief.mean, belief.var, config.smpc_scenarios)
    noise_scenarios = _cartpole_noise_scenarios(config)
    branch_factor = len(parameter_scenarios) * len(noise_scenarios)

    root = ArcariCartPoleTreeNode(
        node_id=0,
        depth=0,
        parent_id=None,
        branch_index=0,
        mode_index=0,
        sample_index=0,
        probability=1.0,
        state=np.asarray(state, dtype=float).copy(),
        prev_action=float(prev_action),
        time=int(t),
        belief=belief.copy(),
    )
    nodes: list[ArcariCartPoleTreeNode] = [root]
    levels: list[list[int]] = [[0]]
    next_id = 1

    for depth in range(dual_horizon):
        next_level: list[int] = []
        for node_id in levels[depth]:
            node = nodes[node_id]
            if depth == 0:
                action = float(root_action)
            else:
                action, _ = _arcari_cartpole_best_dual_action_for_node(
                    node.state, node.prev_action, node.time, node.belief, dynamics, cost, config, depth, dual_horizon
                )
            node.action = action

            branch_index = 0
            for theta, theta_weight in _sigma_scenarios(node.belief.mean, node.belief.var, config.smpc_scenarios):
                nominal_next, _ = cartpole_step(node.state, action, float(theta), dynamics)
                for noise, noise_weight in _cartpole_noise_scenarios(config):
                    child_state = np.asarray(nominal_next + noise, dtype=float)
                    failed = cost.failed(child_state)
                    stage = cost.stage(node.state, action, node.prev_action, reference_position(node.time), failed).total
                    child_belief = node.belief.copy()
                    theta_obs = _cartpole_theta_pseudo_observation(node.state, action, child_state, dynamics)
                    child_belief.update(0.0, 1.0, theta_obs, config.theta_obs_var)
                    child_weight = float(theta_weight * noise_weight)
                    child = ArcariCartPoleTreeNode(
                        node_id=next_id,
                        depth=depth + 1,
                        parent_id=node_id,
                        branch_index=branch_index,
                        mode_index=0,
                        sample_index=branch_index,
                        probability=node.probability * child_weight,
                        state=child_state,
                        prev_action=action,
                        time=node.time + 1,
                        belief=child_belief,
                        branch_weight=child_weight,
                        mode_probability=1.0,
                        parameter_sample=float(theta),
                        process_noise_sample=np.asarray(noise, dtype=float).copy(),
                        transition_stage_cost=float(stage),
                        failed=failed,
                    )
                    nodes.append(child)
                    node.children.append(next_id)
                    next_level.append(next_id)
                    next_id += 1
                    branch_index += 1
        levels.append(next_level)

    dual_cost = 0.0
    for depth in range(1, dual_horizon + 1):
        for node_id in levels[depth]:
            node = nodes[node_id]
            if node.transition_stage_cost is None:
                raise RuntimeError("CartPole tree child has no sampled transition cost")
            dual_cost += node.probability * node.transition_stage_cost

    exploitation_cost = 0.0
    for leaf_id in levels[dual_horizon]:
        leaf = nodes[leaf_id]
        remaining = config.horizon - dual_horizon
        if leaf.failed:
            tail = cost.config.failure_cost * max(remaining, 0)
        else:
            tail = _cartpole_exploitation_value(
                leaf.state, leaf.prev_action, leaf.time, leaf.belief.mean, dynamics, cost, config, remaining
            )
        exploitation_cost += leaf.probability * tail

    return ArcariCartPoleScenarioTree(
        nodes=nodes,
        levels=levels,
        dual_cost=float(dual_cost),
        exploitation_cost=float(exploitation_cost),
        total_cost=float(dual_cost + exploitation_cost),
        root_action=float(root_action),
        dual_horizon=dual_horizon,
        prediction_horizon=config.horizon,
        branch_factor=branch_factor,
    )


def _arcari_cartpole_best_dual_action_for_node(
    state: np.ndarray,
    prev_action: float,
    t: int,
    belief: GaussianBelief,
    dynamics: CartPoleParams,
    cost: CartPoleCost,
    config: OfficialCartPoleConfig,
    depth: int,
    dual_horizon: int,
) -> tuple[float, float]:
    best_action, best_value = 0.0, float("inf")
    for action in config.action_grid:
        action = float(action)
        value = _arcari_cartpole_child_expectation(state, action, prev_action, t, belief, dynamics, cost, config, depth + 1, dual_horizon)
        if value < best_value:
            best_value = value
            best_action = action
    return best_action, float(best_value)


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
    _, best_value = _arcari_cartpole_best_dual_action_for_node(state, prev_action, t, belief, dynamics, cost, config, depth, dual_horizon)
    return best_value


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
    expected = 0.0
    for theta, theta_weight in _sigma_scenarios(belief.mean, belief.var, config.smpc_scenarios):
        nominal_next, _ = cartpole_step(state, action, float(theta), dynamics)
        for noise, noise_weight in _cartpole_noise_scenarios(config):
            next_state = np.asarray(nominal_next + noise, dtype=float)
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
    return float(expected)


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


def _oracle_cartpole_action(state: np.ndarray, prev_action: float, t: int, theta_path: np.ndarray, noise_path: np.ndarray, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig) -> float:
    best_u, best_value = 0.0, float("inf")
    for action in config.action_grid:
        action = float(action)
        theta = float(theta_path[min(t, len(theta_path) - 1)])
        next_state, _ = cartpole_step(state, action, theta, dynamics)
        next_state = next_state + noise_path[min(t, len(noise_path) - 1)]
        failed = cost.failed(next_state)
        value = cost.stage(state, action, prev_action, reference_position(t), failed).total
        value += _oracle_cartpole_value(next_state, action, t + 1, theta_path, noise_path, dynamics, cost, config, config.horizon - 1)
        if value < best_value:
            best_value = value
            best_u = action
    return best_u


def _oracle_cartpole_value(state: np.ndarray, prev_action: float, t: int, theta_path: np.ndarray, noise_path: np.ndarray, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig, horizon: int) -> float:
    if horizon <= 0:
        return cost.terminal(state, reference_position(t)).total
    best = float("inf")
    for action in config.action_grid:
        action = float(action)
        theta = float(theta_path[min(t, len(theta_path) - 1)])
        next_state, _ = cartpole_step(state, action, theta, dynamics)
        next_state = next_state + noise_path[min(t, len(noise_path) - 1)]
        failed = cost.failed(next_state)
        value = cost.stage(state, action, prev_action, reference_position(t), failed).total
        if failed:
            value += cost.config.failure_cost * max(horizon - 1, 0)
        else:
            value += _oracle_cartpole_value(next_state, action, t + 1, theta_path, noise_path, dynamics, cost, config, horizon - 1)
        best = min(best, value)
    return best
