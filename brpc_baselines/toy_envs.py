"""Toy environments for the BRPC baseline suite.

The classes here implement only the new benchmark spec and deliberately do not import
or reuse the older KH/Arcari/TVGP baseline modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    task_reward: float
    energy_cost: float
    switching_cost: float
    failure_cost: float = 0.0

    @property
    def net_reward(self) -> float:
        return self.task_reward - self.energy_cost - self.switching_cost - self.failure_cost


@dataclass(frozen=True)
class Toy1Config:
    horizon_T: int = 400
    theta_initial: float = 0.85
    beta_initial: float = 0.08
    kappa_delta: float = 0.03
    sigma_w: float = 0.03
    q_x: float = 1.0
    lambda_energy: float = 0.05
    lambda_switch: float = 0.20
    action_low: float = -1.0
    action_high: float = 1.0
    x0: float = 0.0
    production_ref: float = 0.75
    quiet_fraction: float = 0.5


class Toy1DigitalTwin:
    """Digital twin f_DT(x, a; theta) = theta x + a."""

    state_dim = 1
    action_dim = 1
    output_dim = 1

    def step(self, state: np.ndarray | float, action: np.ndarray | float, theta: np.ndarray | float) -> np.ndarray:
        x = float(np.asarray(state).reshape(-1)[0])
        a = float(np.asarray(action).reshape(-1)[0])
        th = float(np.asarray(theta).reshape(-1)[0])
        return np.array([th * x + a], dtype=float)

    def batch_step(self, inputs: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        th = float(np.asarray(theta).reshape(-1)[0])
        return (th * X[:, 0] + X[:, 1])[:, None]


class Toy1PhysicalEnv:
    def __init__(
        self,
        config: Toy1Config = Toy1Config(),
        theta_path: np.ndarray | None = None,
        beta_path: np.ndarray | None = None,
        noise_path: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.config = config
        rng = np.random.default_rng(seed)
        self.theta_path = np.asarray(
            theta_path if theta_path is not None else np.full(config.horizon_T, config.theta_initial),
            dtype=float,
        )
        self.beta_path = np.asarray(
            beta_path if beta_path is not None else np.full(config.horizon_T, config.beta_initial),
            dtype=float,
        )
        self.noise_path = np.asarray(
            noise_path if noise_path is not None else rng.normal(0.0, config.sigma_w, config.horizon_T),
            dtype=float,
        )
        if len(self.theta_path) != config.horizon_T or len(self.beta_path) != config.horizon_T:
            raise ValueError("theta_path and beta_path must have length horizon_T.")
        if len(self.noise_path) != config.horizon_T:
            raise ValueError("noise_path must have length horizon_T.")
        self.twin = Toy1DigitalTwin()
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.state = np.array([self.config.x0], dtype=float)
        self.previous_action = np.array([0.0], dtype=float)
        return self.state.copy()

    def reference(self, t: int) -> float:
        return 0.0 if t < int(self.config.quiet_fraction * self.config.horizon_T) else self.config.production_ref

    def discrepancy(self, state: np.ndarray | float, action: np.ndarray | float, beta: float | None = None) -> float:
        x = float(np.asarray(state).reshape(-1)[0])
        a = float(np.asarray(action).reshape(-1)[0])
        b = float(self.beta_path[self.t] if beta is None else beta)
        return b * np.tanh(2.0 * x) + self.config.kappa_delta * a * abs(a)

    def reward(self, state: np.ndarray, action: np.ndarray, previous_action: np.ndarray, next_state: np.ndarray | None = None) -> RewardBreakdown:
        del next_state
        x = float(np.asarray(state).reshape(-1)[0])
        a = float(np.asarray(action).reshape(-1)[0])
        prev = float(np.asarray(previous_action).reshape(-1)[0])
        task = -self.config.q_x * (x - self.reference(self.t)) ** 2
        energy = self.config.lambda_energy * a * a
        switch = self.config.lambda_switch * (a - prev) ** 2
        return RewardBreakdown(task, energy, switch)

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, RewardBreakdown, bool, dict]:
        if self.t >= self.config.horizon_T:
            raise RuntimeError("Cannot step after horizon.")
        a = float(np.clip(float(np.asarray(action).reshape(-1)[0]), self.config.action_low, self.config.action_high))
        action_arr = np.array([a], dtype=float)
        theta = self.theta_path[self.t]
        x_next_nominal = self.twin.step(self.state, action_arr, theta)[0]
        x_next = x_next_nominal + self.discrepancy(self.state, action_arr, self.beta_path[self.t]) + self.noise_path[self.t]
        next_state = np.array([x_next], dtype=float)
        reward = self.reward(self.state, action_arr, self.previous_action, next_state)
        info = {
            "theta": float(theta),
            "beta": float(self.beta_path[self.t]),
            "noise": float(self.noise_path[self.t]),
            "calibration_input": np.array([self.state[0], a], dtype=float),
            "calibration_output": next_state.copy(),
        }
        self.state = next_state
        self.previous_action = action_arr
        self.t += 1
        return next_state.copy(), reward, self.t >= self.config.horizon_T, info


@dataclass(frozen=True)
class Toy2Config:
    horizon_T: int = 300
    theta_initial: float = 0.10
    theta_after_jump: float = 1.00
    change_time: int | None = None
    sigma_basis: float = 0.08
    b_left: float = 1.20
    b_right: float = 0.70
    b_diag: float = -0.40
    c_right: float = 0.90
    c_diag: float = 1.50
    discrepancy_sine_amplitude: float = 0.05
    discrepancy_cubic: float = 0.0
    sigma_y: float = 0.03
    lambda_energy: float = 0.05
    lambda_switch: float = 0.80
    action_low: float = 0.0
    action_high: float = 1.0
    a_left: float = 0.2
    a_diag: float = 0.5
    a_right: float = 0.8


class Toy2DigitalTwin:
    """Stealth changepoint operating landscape response model."""

    state_dim = 1
    action_dim = 1
    output_dim = 1

    def __init__(self, config: Toy2Config = Toy2Config()):
        self.config = config

    def basis(self, action: np.ndarray | float, center: float) -> np.ndarray:
        a = np.asarray(action, dtype=float)
        return np.exp(-0.5 * ((a - center) / self.config.sigma_basis) ** 2)

    def response(self, action: np.ndarray | float, theta: np.ndarray | float) -> np.ndarray:
        a = np.asarray(action, dtype=float)
        th = float(np.asarray(theta).reshape(-1)[0])
        cfg = self.config
        return (
            cfg.b_left * self.basis(a, cfg.a_left)
            + (cfg.b_right + cfg.c_right * th) * self.basis(a, cfg.a_right)
            + (cfg.b_diag + cfg.c_diag * th) * self.basis(a, cfg.a_diag)
        )

    def step(self, state: np.ndarray | float, action: np.ndarray | float, theta: np.ndarray | float) -> np.ndarray:
        del state
        return np.array([float(self.response(float(np.asarray(action).reshape(-1)[0]), theta))], dtype=float)

    def batch_step(self, inputs: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        # Calibration input is either action alone or (previous_action, action).
        actions = X[:, -1]
        return np.asarray(self.response(actions, theta), dtype=float).reshape(-1, 1)


class Toy2PhysicalEnv:
    def __init__(
        self,
        config: Toy2Config = Toy2Config(),
        theta_path: np.ndarray | None = None,
        noise_path: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.config = config
        rng = np.random.default_rng(seed)
        if theta_path is None:
            cp = config.change_time if config.change_time is not None else config.horizon_T // 2
            theta_path = np.full(config.horizon_T, config.theta_initial, dtype=float)
            theta_path[int(cp) :] = config.theta_after_jump
        self.theta_path = np.asarray(theta_path, dtype=float)
        self.noise_path = np.asarray(
            noise_path if noise_path is not None else rng.normal(0.0, config.sigma_y, config.horizon_T),
            dtype=float,
        )
        if len(self.theta_path) != config.horizon_T or len(self.noise_path) != config.horizon_T:
            raise ValueError("theta_path and noise_path must have length horizon_T.")
        self.twin = Toy2DigitalTwin(config)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.previous_action = np.array([self.config.a_left], dtype=float)
        # Per spec, state is previous action for Toy 2.
        self.state = self.previous_action.copy()
        return self.state.copy()

    def discrepancy(self, action: np.ndarray | float) -> float:
        a = float(np.asarray(action).reshape(-1)[0])
        cfg = self.config
        return cfg.discrepancy_sine_amplitude * np.sin(4.0 * np.pi * a) + cfg.discrepancy_cubic * (a - 0.5) ** 3

    def expected_response(self, action: np.ndarray | float, theta: float) -> float:
        return float(self.twin.step(self.state, action, theta)[0] + self.discrepancy(action))

    def reward_from_response(self, response: float, action: np.ndarray, previous_action: np.ndarray) -> RewardBreakdown:
        a = float(np.asarray(action).reshape(-1)[0])
        prev = float(np.asarray(previous_action).reshape(-1)[0])
        energy = self.config.lambda_energy * a * a
        switch = self.config.lambda_switch * (a - prev) ** 2
        return RewardBreakdown(task_reward=float(response), energy_cost=energy, switching_cost=switch)

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, RewardBreakdown, bool, dict]:
        if self.t >= self.config.horizon_T:
            raise RuntimeError("Cannot step after horizon.")
        a = float(np.clip(float(np.asarray(action).reshape(-1)[0]), self.config.action_low, self.config.action_high))
        action_arr = np.array([a], dtype=float)
        theta = float(self.theta_path[self.t])
        y = self.expected_response(action_arr, theta) + float(self.noise_path[self.t])
        reward = self.reward_from_response(y, action_arr, self.previous_action)
        next_state = action_arr.copy()
        info = {
            "theta": theta,
            "noise": float(self.noise_path[self.t]),
            "response": float(y),
            "calibration_input": np.array([self.previous_action[0], a], dtype=float),
            "calibration_output": np.array([y], dtype=float),
        }
        self.previous_action = action_arr
        self.state = next_state
        self.t += 1
        return next_state.copy(), reward, self.t >= self.config.horizon_T, info


def make_reference_schedule_toy1(config: Toy1Config) -> Callable[[int], float]:
    return lambda t: 0.0 if t < int(config.quiet_fraction * config.horizon_T) else config.production_ref
