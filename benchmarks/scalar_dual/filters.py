"""Belief updates for scalar calibration parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianBelief:
    mean: float = 1.0
    var: float = 10.0
    process_var: float = 0.0
    min_var: float = 1e-9

    def predict(self) -> None:
        self.var = max(self.var + self.process_var, self.min_var)

    def update(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        if abs(u) < 1e-12:
            return
        z = observed_next_x - x
        pred_var = max(self.var, self.min_var)
        noise_var = max(obs_var, self.min_var)
        innovation_var = u * u * pred_var + noise_var
        gain = pred_var * u / innovation_var
        self.mean = self.mean + gain * (z - u * self.mean)
        self.var = max((1.0 - gain * u) * pred_var, self.min_var)

    def copy(self) -> "GaussianBelief":
        return GaussianBelief(self.mean, self.var, self.process_var, self.min_var)


@dataclass
class GridBelief:
    grid: np.ndarray
    weights: np.ndarray
    transition: np.ndarray | None = None
    min_weight: float = 1e-300

    @classmethod
    def normal_prior(cls, grid: np.ndarray, mean: float, var: float, transition: np.ndarray | None = None) -> "GridBelief":
        weights = np.exp(-0.5 * (grid - mean) ** 2 / var)
        weights /= weights.sum()
        return cls(grid=np.asarray(grid, dtype=float), weights=weights, transition=transition)

    @property
    def mean(self) -> float:
        return float(np.sum(self.grid * self.weights))

    @property
    def var(self) -> float:
        mean = self.mean
        return float(np.sum((self.grid - mean) ** 2 * self.weights))

    def predict(self) -> None:
        if self.transition is not None:
            self.weights = self.transition.T @ self.weights
            self._normalize()

    def update(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        if abs(u) < 1e-12:
            return
        z = observed_next_x - x
        noise_var = max(obs_var, 1e-12)
        log_like = -0.5 * (z - self.grid * u) ** 2 / noise_var
        log_like -= np.max(log_like)
        self.weights *= np.exp(log_like)
        self._normalize()

    def _normalize(self) -> None:
        self.weights = np.maximum(self.weights, self.min_weight)
        total = self.weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            self.weights = np.ones_like(self.weights) / len(self.weights)
        else:
            self.weights /= total

    def copy(self) -> "GridBelief":
        return GridBelief(self.grid.copy(), self.weights.copy(), None if self.transition is None else self.transition.copy(), self.min_weight)


def random_walk_transition(grid: np.ndarray, process_std: float) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if process_std <= 0:
        return np.eye(len(grid))
    diff = grid[:, None] - grid[None, :]
    mat = np.exp(-0.5 * diff * diff / (process_std * process_std))
    mat /= mat.sum(axis=1, keepdims=True)
    return mat
