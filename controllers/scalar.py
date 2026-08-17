"""Controllers for the scalar dual-control benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.polynomial.hermite import hermgauss

from benchmarks.scalar_dual.costs import ScalarCost
from benchmarks.scalar_dual.filters import GaussianBelief, GridBelief


class ScalarController(Protocol):
    name: str

    def act(self, x: float, prev_u: float) -> float: ...

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None: ...

    def predict(self) -> None: ...

    @property
    def belief_mean(self) -> float: ...

    @property
    def belief_var(self) -> float: ...


@dataclass(frozen=True)
class ScalarPlannerConfig:
    horizon: int = 2
    action_low: float = -3.0
    action_high: float = 3.0
    action_grid_size: int = 121
    process_var: float = 0.1
    dual_sigma_points: tuple[float, ...] = (-1.0, 0.0, 1.0)

    @property
    def action_grid(self) -> np.ndarray:
        return np.linspace(self.action_low, self.action_high, self.action_grid_size)


class BaseGaussianController:
    def __init__(self, name: str, belief: GaussianBelief, cost: ScalarCost, config: ScalarPlannerConfig):
        self.name = name
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


class CertaintyEquivalentController(BaseGaussianController):
    def act(self, x: float, prev_u: float) -> float:
        return _ce_action(x, prev_u, self.belief.mean, self.cost, self.config)


class CautiousController(BaseGaussianController):
    def act(self, x: float, prev_u: float) -> float:
        return _cautious_action(x, prev_u, self.belief.mean, self.belief.var, self.cost, self.config)


class AnalyticBayesDualController(BaseGaussianController):
    def act(self, x: float, prev_u: float) -> float:
        return _analytic_bayes_dual_action(x, prev_u, self.belief, self.cost, self.config)


class ApproxDualController(BaseGaussianController):
    """Small scalar approximate-dual controller.

    This is a diagnostic implementation: it evaluates root actions by
    propagating sigma-point fantasy observations and updating the Gaussian
    belief before optimizing the next certainty-equivalent action. It is not a
    literal full reproduction of every KH approximation term, but it preserves
    the scalar dual effect for first-pass benchmark plumbing.
    """

    def act(self, x: float, prev_u: float) -> float:
        grid = self.config.action_grid
        best_u = float(grid[0])
        best_cost = float("inf")
        for u0 in grid:
            immediate = self.cost.stage(x, float(u0), prev_u).total
            expected_next = 0.0
            sigma = np.sqrt(max(self.config.process_var + u0 * u0 * self.belief.var, 1e-12))
            weights = _sigma_weights(len(self.config.dual_sigma_points))
            for weight, point in zip(weights, self.config.dual_sigma_points):
                next_x = x + self.belief.mean * u0 + point * sigma
                fantasy = self.belief.copy()
                fantasy.update(x, float(u0), float(next_x), self.config.process_var)
                u1 = _ce_action(float(next_x), float(u0), fantasy.mean, self.cost, self.config)
                pred_x2 = next_x + fantasy.mean * u1
                expected_next += weight * (
                    self.cost.stage(float(next_x), u1, float(u0)).total
                    + self.cost.terminal(float(pred_x2)).total
                )
            total = immediate + expected_next
            if total < best_cost:
                best_cost = total
                best_u = float(u0)
        return best_u


class GridBayesController:
    def __init__(
        self,
        name: str,
        belief: GridBelief,
        cost: ScalarCost,
        config: ScalarPlannerConfig,
        branch_grid_size: int = 9,
    ):
        self.name = name
        self.belief = belief
        self.cost = cost
        self.config = config
        self.branch_grid_size = branch_grid_size

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
        return _grid_bayes_action(x, prev_u, self.belief, self.cost, self.config, self.branch_grid_size)


class ClairvoyantController:
    def __init__(self, b_path: np.ndarray, cost: ScalarCost, config: ScalarPlannerConfig):
        self.name = "clairvoyant"
        self.b_path = b_path
        self.cost = cost
        self.config = config
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
        b = float(self.b_path[min(self.t, len(self.b_path) - 1)])
        return _ce_action(x, prev_u, b, self.cost, self.config)


class ScheduledProbeController(CertaintyEquivalentController):
    def __init__(self, belief: GaussianBelief, cost: ScalarCost, config: ScalarPlannerConfig, period: int = 25, magnitude: float = 1.0):
        super().__init__("scheduled_probe", belief, cost, config)
        self.period = period
        self.magnitude = magnitude
        self.t = 0

    def act(self, x: float, prev_u: float) -> float:
        if self.t > 0 and self.t % self.period == 0:
            return self.magnitude if x <= 0 else -self.magnitude
        return super().act(x, prev_u)

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        super().observe(x, u, observed_next_x, obs_var)
        self.t += 1


def _ce_action(x: float, prev_u: float, b: float, cost: ScalarCost, config: ScalarPlannerConfig) -> float:
    return _cautious_action(x, prev_u, b, 0.0, cost, config)


def _cautious_action(x: float, prev_u: float, mean: float, var: float, cost: ScalarCost, config: ScalarPlannerConfig) -> float:
    best_u = 0.0
    best_cost = float("inf")
    for u in config.action_grid:
        u = float(u)
        next_mean = x + mean * u
        expected_terminal = cost.config.terminal_weight * (next_mean * next_mean + var * u * u + config.process_var)
        total = cost.stage(x, u, prev_u).total + expected_terminal
        if total < best_cost:
            best_cost = total
            best_u = u
    return best_u


def _analytic_bayes_dual_action(x: float, prev_u: float, belief: GaussianBelief, cost: ScalarCost, config: ScalarPlannerConfig) -> float:
    nodes, weights = hermgauss(9)
    best_u = 0.0
    best_cost = float("inf")
    for u0 in config.action_grid:
        u0 = float(u0)
        immediate = cost.stage(x, u0, prev_u).total
        pred_mean = x + belief.mean * u0
        pred_var = max(config.process_var + belief.var * u0 * u0, 1e-12)
        branch = 0.0
        for node, weight in zip(nodes, weights):
            next_x = pred_mean + np.sqrt(2.0 * pred_var) * float(node)
            fantasy = belief.copy()
            fantasy.update(x, u0, float(next_x), config.process_var)
            u1 = _cautious_action(float(next_x), u0, fantasy.mean, fantasy.var, cost, config)
            pred_x2_mean = next_x + fantasy.mean * u1
            expected_terminal = cost.config.terminal_weight * (
                pred_x2_mean * pred_x2_mean + fantasy.var * u1 * u1 + config.process_var
            )
            branch += float(weight) * (cost.stage(float(next_x), u1, u0).total + expected_terminal)
        total = immediate + branch / np.sqrt(np.pi)
        if total < best_cost:
            best_cost = total
            best_u = u0
    return best_u


def _grid_bayes_action(
    x: float,
    prev_u: float,
    belief: GridBelief,
    cost: ScalarCost,
    config: ScalarPlannerConfig,
    branch_grid_size: int = 9,
) -> float:
    best_u = 0.0
    best_cost = float("inf")
    action_grid = config.action_grid
    branch_weights = _branch_weights(belief, branch_grid_size)
    sigma_points = np.array([-1.0, 0.0, 1.0]) * np.sqrt(config.process_var)
    sigma_weights = _sigma_weights(len(sigma_points))
    for u0 in action_grid:
        immediate = cost.stage(x, float(u0), prev_u).total
        expected = 0.0
        for idx, bw in branch_weights:
            b = belief.grid[idx]
            for nw, eps in zip(sigma_weights, sigma_points):
                next_x = x + b * u0 + eps
                fantasy = belief.copy()
                fantasy.update(x, float(u0), float(next_x), config.process_var)
                u1 = _grid_one_step_action(float(next_x), float(u0), fantasy, cost, config)
                pred_x2s = next_x + fantasy.grid * u1
                expected_terminal = float(np.sum(fantasy.weights * (config.process_var + pred_x2s * pred_x2s)))
                expected += bw * nw * (cost.stage(float(next_x), u1, float(u0)).total + expected_terminal)
        total = immediate + expected
        if total < best_cost:
            best_cost = total
            best_u = float(u0)
    return best_u


def _grid_one_step_action(x: float, prev_u: float, belief: GridBelief, cost: ScalarCost, config: ScalarPlannerConfig) -> float:
    best_u = 0.0
    best_cost = float("inf")
    for u in config.action_grid:
        next_xs = x + belief.grid * u
        expected_terminal = float(np.sum(belief.weights * (config.process_var + next_xs * next_xs)))
        total = cost.stage(x, float(u), prev_u).total + expected_terminal
        if total < best_cost:
            best_cost = total
            best_u = float(u)
    return best_u


def _branch_weights(belief: GridBelief, branch_grid_size: int) -> list[tuple[int, float]]:
    if branch_grid_size >= len(belief.grid):
        return [(i, float(w)) for i, w in enumerate(belief.weights)]
    cdf = np.cumsum(belief.weights)
    quantiles = (np.arange(branch_grid_size) + 0.5) / branch_grid_size
    indices = np.searchsorted(cdf, quantiles)
    unique: dict[int, float] = {}
    for idx in indices:
        idx = int(np.clip(idx, 0, len(belief.grid) - 1))
        unique[idx] = unique.get(idx, 0.0) + 1.0 / branch_grid_size
    return sorted(unique.items())


def _sigma_weights(n: int) -> np.ndarray:
    if n == 3:
        return np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    weights = np.ones(n, dtype=float)
    return weights / weights.sum()
