"""Baseline CartPole controllers.

These are deliberately simple first-pass controllers for benchmark plumbing.
They are not the final KH/particle Bayes references.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost
from benchmarks.cartpole_twin.dynamics import CartPoleParams, cartpole_step, reference_position
from benchmarks.scalar_dual.filters import GaussianBelief


@dataclass(frozen=True)
class CartPolePlannerConfig:
    horizon: int = 10
    action_grid_size: int = 11
    theta_obs_var: float = 0.05
    sequence_samples: int = 64
    sequence_std: float = 0.4

    @property
    def action_grid(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.action_grid_size)


class CartPoleCEController:
    def __init__(self, name: str, belief: GaussianBelief, dynamics: CartPoleParams, cost: CartPoleCost, config: CartPolePlannerConfig):
        self.name = name
        self.belief = belief
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        self.t = 0

    @property
    def belief_mean(self) -> float:
        return self.belief.mean

    @property
    def belief_var(self) -> float:
        return self.belief.var

    def predict(self) -> None:
        self.belief.predict()

    def act(self, state: np.ndarray, prev_action: float) -> float:
        return _shooting_mpc_action(state, prev_action, self.belief.mean, self.t, self.dynamics, self.cost, self.config)

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        # Crude actuator-gain pseudo-observation from one-step residual search.
        candidates = np.linspace(0.55, 1.45, 61)
        errors = []
        for theta in candidates:
            pred, _ = cartpole_step(state, action, theta, self.dynamics)
            errors.append(float(np.sum((pred - next_state) ** 2)))
        theta_hat = float(candidates[int(np.argmin(errors))])
        self.belief.update(0.0, 1.0, theta_hat, self.config.theta_obs_var)
        self.t += 1


class CartPoleClairvoyantController(CartPoleCEController):
    def __init__(self, theta_path: np.ndarray, dynamics: CartPoleParams, cost: CartPoleCost, config: CartPolePlannerConfig):
        super().__init__("clairvoyant", GaussianBelief(mean=1.0, var=0.0), dynamics, cost, config)
        self.theta_path = theta_path

    @property
    def belief_mean(self) -> float:
        return float(self.theta_path[min(self.t, len(self.theta_path) - 1)])

    @property
    def belief_var(self) -> float:
        return 0.0

    def predict(self) -> None:
        pass

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        self.t += 1

    def act(self, state: np.ndarray, prev_action: float) -> float:
        theta = float(self.theta_path[min(self.t, len(self.theta_path) - 1)])
        return _shooting_mpc_action(state, prev_action, theta, self.t, self.dynamics, self.cost, self.config)


def _shooting_mpc_action(
    state: np.ndarray,
    prev_action: float,
    theta: float,
    t: int,
    dynamics: CartPoleParams,
    cost: CartPoleCost,
    config: CartPolePlannerConfig,
) -> float:
    candidate_first_actions = config.action_grid
    tails = _deterministic_tail_sequences(config)
    best_action = 0.0
    best_cost = float("inf")
    for first_action in candidate_first_actions:
        for tail in tails:
            sim_state = state.copy()
            sim_prev_action = prev_action
            sim_force = 0.0
            total = 0.0
            failed_any = False
            sequence = np.concatenate([[float(first_action)], tail])
            for h, action in enumerate(sequence[: config.horizon]):
                p_ref = reference_position(t + h)
                next_state, sim_force = cartpole_step(sim_state, float(action), theta, dynamics, sim_force)
                failed = cost.failed(next_state)
                failed_any = failed_any or failed
                total += cost.stage(sim_state, float(action), sim_prev_action, p_ref, failed).total
                sim_state = next_state
                sim_prev_action = float(action)
                if failed:
                    total += cost.config.failure_cost * (config.horizon - h - 1)
                    break
            if not failed_any:
                total += cost.terminal(sim_state, reference_position(t + config.horizon)).total
            if total < best_cost:
                best_cost = total
                best_action = float(first_action)
    return best_action


def _deterministic_tail_sequences(config: CartPolePlannerConfig) -> list[np.ndarray]:
    tail_len = max(config.horizon - 1, 0)
    if tail_len == 0:
        return [np.array([], dtype=float)]
    sequences = [np.zeros(tail_len, dtype=float)]
    for level in (-0.5, 0.5, -1.0, 1.0):
        sequences.append(np.full(tail_len, level, dtype=float))
    for switch in (tail_len // 3, 2 * tail_len // 3):
        seq = np.zeros(tail_len, dtype=float)
        seq[switch:] = 0.5
        sequences.append(seq)
        seq = np.zeros(tail_len, dtype=float)
        seq[switch:] = -0.5
        sequences.append(seq)
    return sequences
