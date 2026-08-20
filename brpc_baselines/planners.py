"""CE and posterior-sampling receding-horizon planners with shared CEM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class PredictiveCalibrator(Protocol):
    def predictive_mean(self, inputs: np.ndarray) -> np.ndarray: ...
    def sample_latent(self, rng: np.random.Generator | None = None) -> dict: ...


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 3
    population: int = 64
    elite_fraction: float = 0.10
    iterations: int = 3
    smoothing: float = 0.20
    action_low: float = -1.0
    action_high: float = 1.0
    random_seed: int | None = None


class CEPlanner:
    """Certainty-equivalent MPC: rolls out posterior predictive mean only."""

    def __init__(
        self,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], float],
        config: CEMConfig = CEMConfig(),
    ):
        self.reward_fn = reward_fn
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.query_count = 0
        self._last_mean: np.ndarray | None = None

    def act(self, state: np.ndarray, previous_action: np.ndarray, calibrator: PredictiveCalibrator, t: int = 0) -> np.ndarray:
        def rollout(actions: np.ndarray) -> float:
            s = np.asarray(state, dtype=float).copy()
            prev = np.asarray(previous_action, dtype=float).copy()
            total = 0.0
            for k, a in enumerate(actions):
                a_arr = np.array([float(a)], dtype=float)
                x_input = np.concatenate([s.reshape(-1), a_arr])
                next_out = calibrator.predictive_mean(x_input[None, :])[0]
                self.query_count += 1
                total += self.reward_fn(s, a_arr, prev, t + k)
                # Toy 2 predictive output is response while state is previous action.
                s = next_out.reshape(-1) if next_out.size == s.size else a_arr.copy()
                prev = a_arr
            return total

        return np.array([_cem_optimize(rollout, self.config, self.rng, self._last_mean)[0]], dtype=float)


class PosteriorSamplingPlanner:
    """PS-step MPC: sample one latent model each physical decision and plan in it."""

    def __init__(
        self,
        twin,
        inducing_points: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], float],
        config: CEMConfig = CEMConfig(),
    ):
        self.twin = twin
        self.inducing_points = np.atleast_2d(inducing_points)
        self.kernel_fn = kernel_fn
        self.reward_fn = reward_fn
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.query_count = 0
        self._last_mean: np.ndarray | None = None

    def act(self, state: np.ndarray, previous_action: np.ndarray, calibrator: PredictiveCalibrator, t: int = 0) -> np.ndarray:
        sample = calibrator.sample_latent(self.rng)
        theta = sample["theta"]
        u = np.asarray(sample["u"])
        kzz = self.kernel_fn(self.inducing_points, self.inducing_points)
        kzz += 1e-6 * np.eye(kzz.shape[0])

        def discrepancy(x_input: np.ndarray) -> np.ndarray:
            kxz = self.kernel_fn(np.atleast_2d(x_input), self.inducing_points)
            coeff = np.linalg.solve(kzz, u.T).T
            return np.array([kxz @ coeff[j] for j in range(coeff.shape[0])]).reshape(-1)

        def rollout(actions: np.ndarray) -> float:
            s = np.asarray(state, dtype=float).copy()
            prev = np.asarray(previous_action, dtype=float).copy()
            total = 0.0
            for k, a in enumerate(actions):
                a_arr = np.array([float(a)], dtype=float)
                x_input = np.concatenate([s.reshape(-1), a_arr])
                nominal = self.twin.batch_step(x_input[None, :], theta)[0]
                delta = discrepancy(x_input)
                next_out = nominal + delta
                self.query_count += 1
                total += self.reward_fn(s, a_arr, prev, t + k)
                s = next_out.reshape(-1) if next_out.size == s.size else a_arr.copy()
                prev = a_arr
            return total

        return np.array([_cem_optimize(rollout, self.config, self.rng, self._last_mean)[0]], dtype=float)


def _cem_optimize(
    objective: Callable[[np.ndarray], float],
    config: CEMConfig,
    rng: np.random.Generator,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    horizon = config.horizon
    low = float(config.action_low)
    high = float(config.action_high)
    if warm_start is None or len(warm_start) != horizon:
        mean = np.full(horizon, 0.5 * (low + high), dtype=float)
    else:
        mean = np.r_[warm_start[1:], warm_start[-1]]
    std = np.full(horizon, 0.5 * (high - low), dtype=float)
    elite_n = max(1, int(np.ceil(config.elite_fraction * config.population)))
    best = mean.copy()
    best_value = -np.inf
    for _ in range(config.iterations):
        samples = rng.normal(mean, std, size=(config.population, horizon))
        samples = np.clip(samples, low, high)
        values = np.asarray([objective(seq) for seq in samples])
        order = np.argsort(values)[::-1]
        elites = samples[order[:elite_n]]
        elite_mean = np.mean(elites, axis=0)
        elite_std = np.std(elites, axis=0) + 1e-6
        mean = config.smoothing * mean + (1.0 - config.smoothing) * elite_mean
        std = config.smoothing * std + (1.0 - config.smoothing) * elite_std
        if values[order[0]] > best_value:
            best_value = float(values[order[0]])
            best = samples[order[0]].copy()
    return best
