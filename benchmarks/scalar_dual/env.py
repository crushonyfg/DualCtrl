"""Physical scalar system for deployment experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import CostBreakdown, ScalarCost


@dataclass(frozen=True)
class ScalarEnvConfig:
    process_std: float = np.sqrt(0.1)
    observation_std: float = 0.0
    x0: float = 1.0
    action_low: float = -3.0
    action_high: float = 3.0
    discrepancy_quadratic: float = 0.0
    discrepancy_threshold: float = 0.0
    discrepancy_threshold_value: float = 0.0


@dataclass(frozen=True)
class ScalarStep:
    x: float
    u: float
    prev_u: float
    b: float
    noise: float
    next_x: float
    observed_next_x: float | None
    cost: CostBreakdown


class ScalarPhysicalEnv:
    def __init__(
        self,
        config: ScalarEnvConfig,
        cost: ScalarCost,
        b_path: np.ndarray,
        process_noise: np.ndarray,
        observation_noise: np.ndarray | None = None,
    ):
        self.config = config
        self.cost = cost
        self.b_path = np.asarray(b_path, dtype=float)
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.observation_noise = observation_noise
        if self.observation_noise is None:
            self.observation_noise = np.zeros_like(self.process_noise)
        self.reset()

    @property
    def horizon(self) -> int:
        return len(self.b_path)

    def reset(self) -> None:
        self.t = 0
        self.x = float(self.config.x0)
        self.prev_u = 0.0

    def _discrepancy(self, u: float) -> float:
        value = self.config.discrepancy_quadratic * u * u
        if self.config.discrepancy_threshold_value and abs(u) > self.config.discrepancy_threshold:
            value += self.config.discrepancy_threshold_value
        return value

    def step(self, u: float, observe: bool = True) -> ScalarStep:
        if self.t >= self.horizon:
            raise RuntimeError("Cannot step after horizon.")
        clipped_u = float(np.clip(u, self.config.action_low, self.config.action_high))
        x = self.x
        prev_u = self.prev_u
        b = float(self.b_path[self.t])
        noise = float(self.process_noise[self.t])
        next_x = x + b * clipped_u + self._discrepancy(clipped_u) + noise
        observed_next_x = None
        if observe:
            observed_next_x = next_x + float(self.observation_noise[self.t])
        cost = self.cost.stage(x, clipped_u, prev_u)
        self.x = float(next_x)
        self.prev_u = clipped_u
        self.t += 1
        return ScalarStep(
            x=x,
            u=clipped_u,
            prev_u=prev_u,
            b=b,
            noise=noise,
            next_x=float(next_x),
            observed_next_x=None if observed_next_x is None else float(observed_next_x),
            cost=cost,
        )
