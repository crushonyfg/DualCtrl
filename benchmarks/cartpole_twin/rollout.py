"""Rollout utilities for CartPole twin experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .costs import CartPoleCost
from .env import CartPolePhysicalEnv
from .dynamics import reference_position


@dataclass
class CartPoleCalibrationDataset:
    """Shared pre-deployment CartPole transitions used only for calibration."""

    policy: str
    transitions: list = field(default_factory=list)

    @property
    def n_observations(self) -> int:
        return len(self.transitions)


@dataclass
class CartPoleTrajectory:
    controller: str
    total_cost: float
    terminal_cost: float
    physical_transitions: int
    observed_transitions: int
    failures: int
    failure_events: int = 0
    theta: list[float] = field(default_factory=list)
    actions: list[float] = field(default_factory=list)
    stage_costs: list[float] = field(default_factory=list)
    task_costs: list[float] = field(default_factory=list)
    energy_costs: list[float] = field(default_factory=list)
    switch_costs: list[float] = field(default_factory=list)
    nonsmooth_switch_costs: list[float] = field(default_factory=list)
    failure_costs: list[float] = field(default_factory=list)
    belief_means: list[float] = field(default_factory=list)
    belief_vars: list[float] = field(default_factory=list)

    @property
    def mean_abs_action(self) -> float:
        return float(np.mean(np.abs(self.actions))) if self.actions else 0.0

    @property
    def frac_zero_action(self) -> float:
        return float(np.mean(np.isclose(self.actions, 0.0))) if self.actions else 0.0

    @property
    def action_changes(self) -> int:
        if not self.actions:
            return 0
        prev = np.array([0.0, *self.actions[:-1]], dtype=float)
        cur = np.array(self.actions, dtype=float)
        return int(np.sum(~np.isclose(cur, prev)))


def generate_initial_calibration_dataset(
    env: CartPolePhysicalEnv,
    policy: str,
    n_transitions: int,
    rng: np.random.Generator | None = None,
) -> CartPoleCalibrationDataset:
    """Generate shared physical pre-rollout observations without scoring reward.

    The supplied environment carries the same theta path, process-noise path, and
    dynamics configuration as deployment.  It is consumed only for the requested
    calibration transitions; callers should create fresh deployment environments
    afterwards so deployment reward/cost starts from the original initial state.
    """

    if n_transitions < 0:
        raise ValueError("n_transitions must be nonnegative")
    if n_transitions > env.horizon:
        raise ValueError("n_transitions cannot exceed the environment horizon")
    if policy not in {"small_random", "zero", "grid_probe"}:
        raise ValueError(f"unknown initial calibration policy: {policy}")
    rng = np.random.default_rng(0) if rng is None else rng
    dataset = CartPoleCalibrationDataset(policy=policy)
    grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
    for i in range(n_transitions):
        if policy == "zero":
            action = 0.0
        elif policy == "grid_probe":
            action = float(grid[i % len(grid)])
        else:
            action = float(rng.uniform(-0.2, 0.2))
        dataset.transitions.append(env.step(action, observe=True))
    return dataset


def apply_initial_calibration(controller, dataset: CartPoleCalibrationDataset) -> int:
    """Feed identical pre-deployment transitions to a controller if it observes data."""

    deployment_t = getattr(controller, "t", None)
    original_deployment_t = getattr(controller, "deployment_t", None)
    if original_deployment_t is not None:
        controller.deployment_t = 0
    for step in dataset.transitions:
        if hasattr(controller, "record_cost"):
            feature = np.concatenate(
                [np.asarray(step.state, dtype=float).reshape(4), np.array([float(step.prev_action), float(step.action)])]
            )
            controller.record_cost(feature, step.cost.total)
            continue
        if hasattr(controller, "predict"):
            controller.predict()
        controller.observe(step.state, step.action, step.next_state)
    if deployment_t is not None and not hasattr(controller, "record_cost"):
        controller.t = deployment_t
    if original_deployment_t is not None:
        controller.deployment_t = original_deployment_t
    controller.n_initial_calibration_observations = dataset.n_observations
    return dataset.n_observations


def run_cartpole_rollout(env: CartPolePhysicalEnv, controller, cost: CartPoleCost, observation_interval: int = 1) -> CartPoleTrajectory:
    traj = CartPoleTrajectory(
        controller=controller.name,
        total_cost=0.0,
        terminal_cost=0.0,
        physical_transitions=env.horizon,
        observed_transitions=0,
        failures=0,
    )
    for t in range(env.horizon):
        controller.predict()
        observe = (t % observation_interval) == 0
        action = controller.act(env.state, env.prev_action)
        step = env.step(action, observe=observe)
        traj.theta.append(step.theta)
        traj.actions.append(step.action)
        traj.stage_costs.append(step.cost.total)
        traj.task_costs.append(step.cost.task)
        traj.energy_costs.append(step.cost.energy)
        traj.switch_costs.append(step.cost.switch)
        traj.nonsmooth_switch_costs.append(step.cost.nonsmooth_switch)
        traj.failure_costs.append(step.cost.failure)
        traj.belief_means.append(controller.belief_mean)
        traj.belief_vars.append(controller.belief_var)
        traj.total_cost += step.cost.total
        traj.failures += int(step.failed)
        traj.failure_events += int(step.failure_event)
        if step.observed_next_state is not None:
            controller.observe(step.state, step.action, step.observed_next_state)
            traj.observed_transitions += 1
        else:
            _advance_unobserved_controller_time(controller)
    terminal = cost.terminal(env.state, reference_position(env.horizon))
    traj.terminal_cost = terminal.total
    traj.total_cost += terminal.total
    return traj


def _advance_unobserved_controller_time(controller) -> None:
    """Keep controller clocks aligned when sparse physical data hides transitions.

    Observation-gated filters should not update their belief on hidden physical
    transitions, but controllers with an explicit deployment clock still need to
    advance time for references/oracle indices. TV-GP-LCB is excluded because its
    adapter clock advances from realized-cost feedback in ``act``/``observe``.
    """
    if getattr(controller, "name", "") == "tv_gp_lcb":
        return
    attr = "deployment_t" if hasattr(controller, "deployment_t") else "t"
    if hasattr(controller, attr):
        try:
            setattr(controller, attr, getattr(controller, attr) + 1)
        except (AttributeError, TypeError):
            pass
