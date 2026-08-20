"""Physical CartPole environment for evolving twin experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import CartPoleCost, CartPoleCostBreakdown
from .dynamics import CartPoleParams, cartpole_step, reference_position


@dataclass(frozen=True)
class CartPoleEnvConfig:
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.0)
    process_std: tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01)
    reference_segment: int = 125
    absorbing_failure: bool = True


@dataclass(frozen=True)
class CartPoleStep:
    state: np.ndarray
    action: float
    prev_action: float
    theta: float
    next_state: np.ndarray
    observed_next_state: np.ndarray | None
    cost: CartPoleCostBreakdown
    failed: bool
    failure_event: bool
    p_ref: float


class CartPolePhysicalEnv:
    def __init__(
        self,
        config: CartPoleEnvConfig,
        dynamics: CartPoleParams,
        cost: CartPoleCost,
        theta_path: np.ndarray,
        process_noise: np.ndarray,
    ):
        self.config = config
        self.dynamics = dynamics
        self.cost = cost
        self.theta_path = np.asarray(theta_path, dtype=float)
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.reset()

    @property
    def horizon(self) -> int:
        return len(self.theta_path)

    def reset(self) -> None:
        self.t = 0
        self.state = np.array(self.config.initial_state, dtype=float)
        self.prev_action = 0.0
        self.prev_force = 0.0
        self.absorbed = False
        self.failure_event_emitted = False

    def step(self, action: float, observe: bool = True) -> CartPoleStep:
        if self.t >= self.horizon:
            raise RuntimeError("Cannot step after horizon.")
        state = self.state.copy()
        prev_action = self.prev_action
        theta = float(self.theta_path[self.t])
        clipped_action = float(np.clip(action, -1.0, 1.0))
        p_ref = reference_position(self.t, self.config.reference_segment)
        already_absorbed = self.absorbed
        if already_absorbed:
            next_state = state.copy()
            failed = True
            failure_event = False
        else:
            next_state, self.prev_force = cartpole_step(state, clipped_action, theta, self.dynamics, self.prev_force)
            next_state = next_state + self.process_noise[self.t]
            failed = self.cost.failed(next_state)
            failure_event = failed and not self.failure_event_emitted
            if failure_event:
                self.failure_event_emitted = True
            if failed and self.config.absorbing_failure:
                self.absorbed = True
        stage_cost = self.cost.stage(state, clipped_action, prev_action, p_ref, failed)
        observed_next_state = next_state.copy() if observe else None
        self.state = next_state
        self.prev_action = clipped_action
        self.t += 1
        return CartPoleStep(
            state=state,
            action=clipped_action,
            prev_action=prev_action,
            theta=theta,
            next_state=next_state.copy(),
            observed_next_state=observed_next_state,
            cost=stage_cost,
            failed=failed,
            failure_event=failure_event,
            p_ref=p_ref,
        )
