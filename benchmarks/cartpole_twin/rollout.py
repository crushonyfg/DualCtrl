"""Rollout utilities for CartPole twin experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .costs import CartPoleCost
from .env import CartPolePhysicalEnv
from .dynamics import reference_position


@dataclass
class CartPoleTrajectory:
    controller: str
    total_cost: float
    terminal_cost: float
    physical_transitions: int
    observed_transitions: int
    failures: int
    theta: list[float] = field(default_factory=list)
    actions: list[float] = field(default_factory=list)
    stage_costs: list[float] = field(default_factory=list)
    task_costs: list[float] = field(default_factory=list)
    energy_costs: list[float] = field(default_factory=list)
    switch_costs: list[float] = field(default_factory=list)
    failure_costs: list[float] = field(default_factory=list)
    belief_means: list[float] = field(default_factory=list)
    belief_vars: list[float] = field(default_factory=list)


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
        traj.failure_costs.append(step.cost.failure)
        traj.belief_means.append(controller.belief_mean)
        traj.belief_vars.append(controller.belief_var)
        traj.total_cost += step.cost.total
        traj.failures += int(step.failed)
        if step.observed_next_state is not None:
            controller.observe(step.state, step.action, step.observed_next_state)
            traj.observed_transitions += 1
    terminal = cost.terminal(env.state, reference_position(env.horizon))
    traj.terminal_cost = terminal.total
    traj.total_cost += terminal.total
    return traj
