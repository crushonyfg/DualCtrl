"""Parameter trajectories for the scalar benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


RegimeKind = Literal["static", "ou", "fixed_jumps", "random_jumps", "multimodal"]


@dataclass(frozen=True)
class ScalarRegimeConfig:
    kind: RegimeKind = "static"
    horizon: int = 300
    base: float = 2.0
    rho: float = 0.995
    sigma: float = 0.015
    lower: float = 0.4
    upper: float = 2.6
    fixed_values: tuple[float, ...] = (2.0, 1.2, 2.2)
    change_points: tuple[int, ...] = (100, 200)
    hazard: float = 1.0 / 200.0
    min_dwell: int = 80
    multimodal_values: tuple[float, ...] = (0.8, 1.6, 2.4)


def _effective_change_points(change_points: tuple[int, ...], horizon: int, n_values: int) -> tuple[int, ...]:
    cps = tuple(sorted({int(cp) for cp in change_points if 0 < int(cp) < horizon}))
    needed = max(n_values - 1, 0)
    if len(cps) >= needed:
        return cps[:needed]
    if needed == 0:
        return ()
    fallback = tuple(max(1, min(horizon - 1, int(round(horizon * k / n_values)))) for k in range(1, n_values))
    return tuple(sorted({*cps, *fallback}))[:needed]


def generate_b_path(config: ScalarRegimeConfig, rng: np.random.Generator) -> np.ndarray:
    if config.horizon <= 0:
        raise ValueError("horizon must be positive")
    b = np.empty(config.horizon, dtype=float)
    if config.kind == "static":
        b.fill(config.base)
        return b

    if config.kind == "ou":
        b[0] = config.base
        for t in range(config.horizon - 1):
            proposal = config.base + config.rho * (b[t] - config.base) + config.sigma * rng.normal()
            b[t + 1] = np.clip(proposal, config.lower, config.upper)
        return b

    if config.kind == "fixed_jumps":
        values = config.fixed_values
        cps = _effective_change_points(config.change_points, config.horizon, len(values))
        for t in range(config.horizon):
            idx = sum(t >= cp for cp in cps)
            b[t] = values[min(idx, len(values) - 1)]
        return b

    if config.kind == "random_jumps":
        values = np.array(config.fixed_values, dtype=float)
        current_idx = 0
        dwell = 0
        for t in range(config.horizon):
            if dwell >= config.min_dwell and rng.random() < config.hazard:
                choices = [i for i in range(len(values)) if i != current_idx]
                current_idx = int(rng.choice(choices))
                dwell = 0
            b[t] = values[current_idx]
            dwell += 1
        return b

    if config.kind == "multimodal":
        values = np.array(config.multimodal_values, dtype=float)
        current_idx = int(rng.integers(len(values)))
        dwell = 0
        for t in range(config.horizon):
            if dwell >= config.min_dwell and rng.random() < config.hazard:
                choices = [i for i in range(len(values)) if i != current_idx]
                current_idx = int(rng.choice(choices))
                dwell = 0
            b[t] = values[current_idx]
            dwell += 1
        return b

    raise ValueError(f"Unknown regime kind: {config.kind}")
