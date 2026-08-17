"""Actuator-gain regimes for CartPole twin experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


RegimeKind = Literal["static", "ou", "fixed_jumps", "random_jumps", "multimodal"]


@dataclass(frozen=True)
class CartPoleRegimeConfig:
    kind: RegimeKind = "static"
    horizon: int = 600
    base: float = 1.0
    rho: float = 0.995
    sigma: float = 0.005
    lower: float = 0.55
    upper: float = 1.45
    fixed_values: tuple[float, ...] = (1.0, 0.65, 1.25)
    change_points: tuple[int, ...] = (200, 400)
    hazard: float = 1.0 / 200.0
    min_dwell: int = 80
    multimodal_values: tuple[float, ...] = (0.65, 1.0, 1.25)


def generate_theta_path(config: CartPoleRegimeConfig, rng: np.random.Generator) -> np.ndarray:
    theta = np.empty(config.horizon, dtype=float)
    if config.kind == "static":
        theta.fill(config.base)
        return theta
    if config.kind == "ou":
        theta[0] = config.base
        for t in range(config.horizon - 1):
            proposal = config.base + config.rho * (theta[t] - config.base) + config.sigma * rng.normal()
            theta[t + 1] = np.clip(proposal, config.lower, config.upper)
        return theta
    if config.kind == "fixed_jumps":
        for t in range(config.horizon):
            idx = sum(t >= cp for cp in config.change_points)
            theta[t] = config.fixed_values[min(idx, len(config.fixed_values) - 1)]
        return theta
    values = np.array(config.multimodal_values if config.kind == "multimodal" else config.fixed_values, dtype=float)
    current_idx = 0
    dwell = 0
    for t in range(config.horizon):
        if dwell >= config.min_dwell and rng.random() < config.hazard:
            choices = [i for i in range(len(values)) if i != current_idx]
            current_idx = int(rng.choice(choices))
            dwell = 0
        theta[t] = values[current_idx]
        dwell += 1
    return theta
