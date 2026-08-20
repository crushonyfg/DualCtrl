"""CE and posterior-sampling receding-horizon planners with shared CEM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np


class PredictiveCalibrator(Protocol):
    def predictive_mean(self, inputs: np.ndarray) -> np.ndarray: ...
    def sample_latent_path(self, horizon: int, rng: np.random.Generator | None = None) -> list[dict]: ...


def stage_reward(
    predicted_response: np.ndarray | float,
    action: np.ndarray | float,
    previous_action: np.ndarray | float,
    lambda_energy: float,
    lambda_switch: float,
) -> float:
    """Shared physical stage reward accounting for response-style objectives.

    This implements response - lambda_E * a^2 - lambda_switch * (a - a_prev)^2.
    Planners should use this for Toy 2 and other response-as-task-reward settings
    rather than an action-cost-only surrogate.
    """

    response = float(np.asarray(predicted_response, dtype=float).reshape(-1)[0])
    a = float(np.asarray(action, dtype=float).reshape(-1)[0])
    prev = float(np.asarray(previous_action, dtype=float).reshape(-1)[0])
    return response - float(lambda_energy) * a * a - float(lambda_switch) * (a - prev) ** 2


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
        stage_reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        config: CEMConfig = CEMConfig(),
    ):
        self.stage_reward_fn = stage_reward_fn
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.query_count = 0
        self._last_mean: np.ndarray | None = None

    def act(self, state: np.ndarray, previous_action: np.ndarray, calibrator: PredictiveCalibrator, t: int = 0) -> np.ndarray:
        del t

        def rollout(actions: np.ndarray) -> float:
            s = np.asarray(state, dtype=float).copy()
            prev = np.asarray(previous_action, dtype=float).copy()
            total = 0.0
            for _, a in enumerate(actions):
                a_arr = np.array([float(a)], dtype=float)
                x_input = np.concatenate([s.reshape(-1), a_arr])
                next_out = calibrator.predictive_mean(x_input[None, :])[0]
                self.query_count += 1
                total += self.stage_reward_fn(next_out, a_arr, prev)
                # Toy 2 predictive output is response while state is previous action.
                s = next_out.reshape(-1) if next_out.size == s.size else a_arr.copy()
                prev = a_arr
            return total

        best = _cem_optimize(rollout, self.config, self.rng, self._last_mean)
        self._last_mean = best.copy()
        return np.array([best[0]], dtype=float)


class PosteriorSamplingPlanner:
    """Posterior-sampling MPC: sample one coherent future latent path per decision."""

    def __init__(
        self,
        twin,
        inducing_points: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        stage_reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        config: CEMConfig = CEMConfig(),
    ):
        self.twin = twin
        self.inducing_points = np.atleast_2d(inducing_points)
        self.kernel_fn = kernel_fn
        self.stage_reward_fn = stage_reward_fn
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.query_count = 0
        self._last_mean: np.ndarray | None = None

    def act(self, state: np.ndarray, previous_action: np.ndarray, calibrator: PredictiveCalibrator, t: int = 0) -> np.ndarray:
        del t
        latent_path = calibrator.sample_latent_path(self.config.horizon, self.rng)
        if len(latent_path) != self.config.horizon:
            raise ValueError("sample_latent_path must return one latent sample per horizon step.")
        kzz = self.kernel_fn(self.inducing_points, self.inducing_points)
        kzz += 1e-6 * np.eye(kzz.shape[0])
        path_coefficients = [np.linalg.solve(kzz, np.asarray(sample["u"], dtype=float).T).T for sample in latent_path]

        def discrepancy(x_input: np.ndarray, step: int) -> np.ndarray:
            kxz = self.kernel_fn(np.atleast_2d(x_input), self.inducing_points)
            coeff = path_coefficients[step]
            return np.array([kxz @ coeff[j] for j in range(coeff.shape[0])]).reshape(-1)

        def rollout(actions: np.ndarray) -> float:
            s = np.asarray(state, dtype=float).copy()
            prev = np.asarray(previous_action, dtype=float).copy()
            total = 0.0
            for k, a in enumerate(actions):
                sample = latent_path[k]
                a_arr = np.array([float(a)], dtype=float)
                x_input = np.concatenate([s.reshape(-1), a_arr])
                nominal = self.twin.batch_step(x_input[None, :], sample["theta"])[0]
                delta = discrepancy(x_input, k)
                next_out = nominal + delta
                self.query_count += 1
                total += self.stage_reward_fn(next_out, a_arr, prev)
                s = next_out.reshape(-1) if next_out.size == s.size else a_arr.copy()
                prev = a_arr
            return total

        best = _cem_optimize(rollout, self.config, self.rng, self._last_mean)
        self._last_mean = best.copy()
        return np.array([best[0]], dtype=float)


class ToyCurrentDynamicsOraclePlanner:
    """CEM-MPC oracle for Toy1/Toy2 using true current dynamics only.

    At decision time t this planner may read the current true theta and current
    discrepancy from the toy environment. During the planning horizon it freezes those
    current dynamics and uses noise-free rollouts, so it does not see future process
    noise or future regime changes. This is the primary oracle in the BRPC toy suite.
    """

    oracle_kind = "oracle_current"

    def __init__(
        self,
        env,
        config: CEMConfig = CEMConfig(),
        environment: Literal["Toy1", "Toy2"] | None = None,
    ):
        self.env = env
        self.config = config
        self.environment = environment or _infer_toy_environment(env)
        self.rng = np.random.default_rng(config.random_seed)
        self.query_count = 0
        self._last_mean: np.ndarray | None = None

    def act(self, state: np.ndarray, previous_action: np.ndarray, calibrator: PredictiveCalibrator | None = None, t: int | None = None) -> np.ndarray:
        del calibrator
        current_t = int(self.env.t if t is None else t)
        theta = float(self.env.theta_path[current_t])
        beta = float(self.env.beta_path[current_t]) if self.environment == "Toy1" else None

        def rollout(actions: np.ndarray) -> float:
            s = np.asarray(state, dtype=float).copy()
            prev = np.asarray(previous_action, dtype=float).copy()
            total = 0.0
            for a in actions:
                a_arr = np.array([float(a)], dtype=float)
                if self.environment == "Toy1":
                    next_out = self.env.twin.step(s, a_arr, theta)
                    next_out = np.array([float(next_out[0]) + self.env.discrepancy(s, a_arr, beta)], dtype=float)
                    total += (
                        -self.env.config.q_x * (float(next_out[0]) - self.env.config.production_ref) ** 2
                        - self.env.config.lambda_energy * float(a_arr[0]) ** 2
                        - self.env.config.lambda_switch * (float(a_arr[0]) - float(prev[0])) ** 2
                    )
                    s = next_out.reshape(-1)
                else:
                    response = self.env.expected_response(a_arr, theta)
                    next_out = np.array([response], dtype=float)
                    total += stage_reward(next_out, a_arr, prev, self.env.config.lambda_energy, self.env.config.lambda_switch)
                    s = a_arr.copy()
                self.query_count += 1
                prev = a_arr
            return float(total)

        best = _cem_optimize(rollout, self.config, self.rng, self._last_mean)
        self._last_mean = best.copy()
        return np.array([best[0]], dtype=float)


class ToyFutureRegimeOraclePlanner:
    """Appendix/ceiling CEM-MPC oracle for toys with known future regime path.

    This toy-only ceiling knows theta/discrepancy regimes over the planning horizon but
    still uses noise-free rollouts and never sees future process noise or future states.
    """

    oracle_kind = "oracle_future_appendix_ceiling"

    def __init__(
        self,
        env,
        config: CEMConfig = CEMConfig(),
        environment: Literal["Toy1", "Toy2"] | None = None,
    ):
        self.env = env
        self.config = config
        self.environment = environment or _infer_toy_environment(env)
        self.rng = np.random.default_rng(config.random_seed)
        self.query_count = 0
        self._last_mean: np.ndarray | None = None

    def act(self, state: np.ndarray, previous_action: np.ndarray, calibrator: PredictiveCalibrator | None = None, t: int | None = None) -> np.ndarray:
        del calibrator
        start_t = int(self.env.t if t is None else t)

        def rollout(actions: np.ndarray) -> float:
            s = np.asarray(state, dtype=float).copy()
            prev = np.asarray(previous_action, dtype=float).copy()
            total = 0.0
            for k, a in enumerate(actions):
                regime_t = min(start_t + k, self.env.config.horizon_T - 1)
                theta = float(self.env.theta_path[regime_t])
                a_arr = np.array([float(a)], dtype=float)
                if self.environment == "Toy1":
                    beta = float(self.env.beta_path[regime_t])
                    next_out = self.env.twin.step(s, a_arr, theta)
                    next_out = np.array([float(next_out[0]) + self.env.discrepancy(s, a_arr, beta)], dtype=float)
                    ref = self.env.reference(regime_t)
                    total += (
                        -self.env.config.q_x * (float(next_out[0]) - ref) ** 2
                        - self.env.config.lambda_energy * float(a_arr[0]) ** 2
                        - self.env.config.lambda_switch * (float(a_arr[0]) - float(prev[0])) ** 2
                    )
                    s = next_out.reshape(-1)
                else:
                    response = self.env.expected_response(a_arr, theta)
                    next_out = np.array([response], dtype=float)
                    total += stage_reward(next_out, a_arr, prev, self.env.config.lambda_energy, self.env.config.lambda_switch)
                    s = a_arr.copy()
                self.query_count += 1
                prev = a_arr
            return float(total)

        best = _cem_optimize(rollout, self.config, self.rng, self._last_mean)
        self._last_mean = best.copy()
        return np.array([best[0]], dtype=float)


def _infer_toy_environment(env) -> Literal["Toy1", "Toy2"]:
    if hasattr(env, "beta_path"):
        return "Toy1"
    if hasattr(env, "expected_response"):
        return "Toy2"
    raise ValueError("Could not infer toy environment; pass environment='Toy1' or 'Toy2'.")


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
