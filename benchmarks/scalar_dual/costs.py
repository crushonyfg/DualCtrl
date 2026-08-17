"""Cost functions for the scalar dual-control benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarCostConfig:
    state_weight: float = 1.0
    energy_weight: float = 0.1
    switch_weight: float = 0.0
    terminal_weight: float = 1.0
    nonsmooth_switch_cost: float = 0.0
    nonsmooth_switch_threshold: float = 1e-9


@dataclass(frozen=True)
class CostBreakdown:
    state: float
    energy: float
    switch: float
    nonsmooth_switch: float
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return self.state + self.energy + self.switch + self.nonsmooth_switch + self.terminal


class ScalarCost:
    def __init__(self, config: ScalarCostConfig):
        self.config = config

    def stage(self, x: float, u: float, prev_u: float) -> CostBreakdown:
        delta = u - prev_u
        nonsmooth = 0.0
        if self.config.nonsmooth_switch_cost > 0.0 and abs(delta) > self.config.nonsmooth_switch_threshold:
            nonsmooth = self.config.nonsmooth_switch_cost
        return CostBreakdown(
            state=self.config.state_weight * x * x,
            energy=self.config.energy_weight * u * u,
            switch=self.config.switch_weight * delta * delta,
            nonsmooth_switch=nonsmooth,
        )

    def terminal(self, x: float) -> CostBreakdown:
        return CostBreakdown(
            state=0.0,
            energy=0.0,
            switch=0.0,
            nonsmooth_switch=0.0,
            terminal=self.config.terminal_weight * x * x,
        )
