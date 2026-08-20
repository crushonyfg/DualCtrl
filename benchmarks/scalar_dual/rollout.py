"""Rollout utilities for scalar deployment experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from controllers.scalar import ScalarController

from .costs import CostBreakdown, ScalarCost
from .env import ScalarPhysicalEnv


@dataclass
class ScalarTrajectory:
    controller: str
    total_cost: float
    terminal_cost: float
    physical_transitions: int
    observed_transitions: int
    xs: list[float] = field(default_factory=list)
    us: list[float] = field(default_factory=list)
    bs: list[float] = field(default_factory=list)
    observed: list[bool] = field(default_factory=list)
    stage_costs: list[float] = field(default_factory=list)
    state_costs: list[float] = field(default_factory=list)
    energy_costs: list[float] = field(default_factory=list)
    switch_costs: list[float] = field(default_factory=list)
    nonsmooth_switch_costs: list[float] = field(default_factory=list)
    belief_means: list[float] = field(default_factory=list)
    belief_vars: list[float] = field(default_factory=list)

    @property
    def mean_abs_action(self) -> float:
        return float(np.mean(np.abs(self.us))) if self.us else 0.0

    @property
    def frac_zero_action(self) -> float:
        return float(np.mean(np.isclose(self.us, 0.0))) if self.us else 0.0

    @property
    def action_changes(self) -> int:
        if not self.us:
            return 0
        prev = np.array([0.0, *self.us[:-1]], dtype=float)
        cur = np.array(self.us, dtype=float)
        return int(np.sum(~np.isclose(cur, prev)))


def run_scalar_rollout(
    env: ScalarPhysicalEnv,
    controller: ScalarController,
    cost: ScalarCost,
    observation_interval: int = 1,
    obs_var: float = 0.1,
) -> ScalarTrajectory:
    observed_transitions = 0
    traj = ScalarTrajectory(
        controller=controller.name,
        total_cost=0.0,
        terminal_cost=0.0,
        physical_transitions=env.horizon,
        observed_transitions=0,
    )
    for t in range(env.horizon):
        controller.predict()
        observe = (t % observation_interval) == 0
        u = controller.act(env.x, env.prev_u)
        step = env.step(u, observe=observe)
        traj.xs.append(step.x)
        traj.us.append(step.u)
        traj.bs.append(step.b)
        traj.observed.append(observe)
        traj.stage_costs.append(step.cost.total)
        traj.state_costs.append(step.cost.state)
        traj.energy_costs.append(step.cost.energy)
        traj.switch_costs.append(step.cost.switch)
        traj.nonsmooth_switch_costs.append(step.cost.nonsmooth_switch)
        traj.belief_means.append(controller.belief_mean)
        traj.belief_vars.append(controller.belief_var)
        traj.total_cost += step.cost.total
        if step.observed_next_x is not None:
            controller.observe(step.x, step.u, step.observed_next_x, obs_var)
            observed_transitions += 1
        else:
            _advance_unobserved_controller_time(controller)
    terminal = cost.terminal(env.x)
    traj.terminal_cost = terminal.total
    traj.total_cost += terminal.total
    traj.observed_transitions = observed_transitions
    traj.xs.append(float(env.x))
    return traj


def _advance_unobserved_controller_time(controller: ScalarController) -> None:
    """Advance deployment clocks without giving hidden transitions to filters."""
    if getattr(controller, "name", "") == "tv_gp_lcb":
        return
    if hasattr(controller, "t"):
        try:
            controller.t += 1
        except (AttributeError, TypeError):
            pass


def paired_bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means[i] = float(np.mean(values[idx]))
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
