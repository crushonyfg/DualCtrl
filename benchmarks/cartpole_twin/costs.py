"""Costs for CartPole evolving digital-twin experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPoleCostConfig:
    w_p: float = 1.0
    w_phi: float = 20.0
    w_v: float = 0.1
    w_omega: float = 0.1
    energy_weight: float = 1e-3
    switch_weight: float = 1e-2
    nonsmooth_switch_cost: float = 0.0
    nonsmooth_switch_threshold: float = 1e-9
    failure_cost: float = 100.0
    terminal_p_weight: float = 5.0
    terminal_phi_weight: float = 50.0
    x_failure: float = 2.4
    angle_failure_rad: float = 20.0 * np.pi / 180.0


@dataclass(frozen=True)
class CartPoleCostBreakdown:
    task: float
    energy: float
    switch: float
    nonsmooth_switch: float
    failure: float
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return self.task + self.energy + self.switch + self.nonsmooth_switch + self.failure + self.terminal


class CartPoleCost:
    def __init__(self, config: CartPoleCostConfig):
        self.config = config

    def failed(self, state: np.ndarray) -> bool:
        return abs(float(state[0])) > self.config.x_failure or abs(float(state[2])) > self.config.angle_failure_rad

    def stage(self, state: np.ndarray, action: float, prev_action: float, p_ref: float, failed: bool) -> CartPoleCostBreakdown:
        delta = action - prev_action
        nonsmooth = 0.0
        if self.config.nonsmooth_switch_cost > 0.0 and abs(delta) > self.config.nonsmooth_switch_threshold:
            nonsmooth = self.config.nonsmooth_switch_cost
        task = (
            self.config.w_p * (float(state[0]) - p_ref) ** 2
            + self.config.w_phi * float(state[2]) ** 2
            + self.config.w_v * float(state[1]) ** 2
            + self.config.w_omega * float(state[3]) ** 2
        )
        return CartPoleCostBreakdown(
            task=task,
            energy=self.config.energy_weight * action * action,
            switch=self.config.switch_weight * delta * delta,
            nonsmooth_switch=nonsmooth,
            failure=self.config.failure_cost if failed else 0.0,
        )

    def terminal(self, state: np.ndarray, p_ref: float) -> CartPoleCostBreakdown:
        return CartPoleCostBreakdown(
            task=0.0,
            energy=0.0,
            switch=0.0,
            nonsmooth_switch=0.0,
            failure=0.0,
            terminal=self.config.terminal_p_weight * (float(state[0]) - p_ref) ** 2 + self.config.terminal_phi_weight * float(state[2]) ** 2,
        )
