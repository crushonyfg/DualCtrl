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


def _effective_change_points(change_points: tuple[int, ...], horizon: int, n_values: int) -> tuple[int, ...]:
    cps = tuple(sorted({int(cp) for cp in change_points if 0 < int(cp) < horizon}))
    needed = max(n_values - 1, 0)
    if len(cps) >= needed:
        return cps[:needed]
    if needed == 0:
        return ()
    fallback = tuple(max(1, min(horizon - 1, int(round(horizon * k / n_values)))) for k in range(1, n_values))
    return tuple(sorted({*cps, *fallback}))[:needed]


def generate_theta_path(config: CartPoleRegimeConfig, rng: np.random.Generator) -> np.ndarray:
    if config.horizon <= 0:
        raise ValueError("horizon must be positive")
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
        change_points = _effective_change_points(config.change_points, config.horizon, len(config.fixed_values))
        for t in range(config.horizon):
            idx = sum(t >= cp for cp in change_points)
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
