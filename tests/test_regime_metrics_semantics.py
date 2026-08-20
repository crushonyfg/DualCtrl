from __future__ import annotations

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams
from benchmarks.cartpole_twin.env import CartPoleEnvConfig, CartPolePhysicalEnv
from benchmarks.cartpole_twin.rollout import run_cartpole_rollout
from benchmarks.cartpole_twin.regimes import generate_theta_path
from benchmarks.scalar_dual.regimes import generate_b_path
from experiments import run_official_cartpole, run_official_scalar


class ConstantCartPoleController:
    name = "constant_zero"
    belief_mean = 1.0
    belief_var = 0.0

    def predict(self) -> None:
        pass

    def act(self, state: np.ndarray, prev_action: float) -> float:
        return 0.0

    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        pass


def test_official_scalar_piecewise_changes_inside_short_horizon() -> None:
    horizon = 5
    regime = run_official_scalar.make_regime("piecewise", horizon)
    path = generate_b_path(regime, np.random.default_rng(0))
    summary = run_official_scalar.path_summary(path, "b")

    assert regime.change_points == (2, 3)
    assert summary["b_path_n_changes"] == 2
    assert 0 < summary["b_path_first_change_step"] < horizon
    assert summary["b_path_values"] == "2;2;1.2;2.2;2.2"


def test_official_cartpole_piecewise_changes_inside_short_horizon() -> None:
    horizon = 5
    regime = run_official_cartpole.make_regime("piecewise", horizon)
    path = generate_theta_path(regime, np.random.default_rng(0))
    summary = run_official_cartpole.path_summary(path, "theta")

    assert regime.change_points == (2, 3)
    assert summary["theta_path_n_changes"] == 2
    assert 0 < summary["theta_path_first_change_step"] < horizon
    assert summary["theta_path_values"] == "1;1;0.65;1.25;1.25"


def test_official_drifting_regimes_have_nontrivial_short_horizon_variation() -> None:
    horizon = 5
    scalar_regime = run_official_scalar.make_regime("drifting", horizon)
    cartpole_regime = run_official_cartpole.make_regime("drifting", horizon)
    b_path = generate_b_path(scalar_regime, np.random.default_rng(1))
    theta_path = generate_theta_path(cartpole_regime, np.random.default_rng(1))

    assert scalar_regime.sigma >= 0.05
    assert cartpole_regime.sigma >= 0.02
    assert run_official_scalar.path_summary(b_path, "b")["b_path_range"] > 0.0
    assert run_official_cartpole.path_summary(theta_path, "theta")["theta_path_range"] > 0.0


def test_cartpole_absorbing_failure_event_count_is_at_most_one_per_trajectory() -> None:
    horizon = 4
    theta_path = np.ones(horizon)
    process_noise = np.zeros((horizon, 4))
    cost = CartPoleCost(CartPoleCostConfig(angle_failure_rad=0.01, failure_cost=7.0))
    env = CartPolePhysicalEnv(
        CartPoleEnvConfig(initial_state=(0.0, 0.0, 0.2, 0.0), absorbing_failure=True),
        CartPoleParams(),
        cost,
        theta_path,
        process_noise,
    )

    traj = run_cartpole_rollout(env, ConstantCartPoleController(), cost)

    assert traj.failure_events <= 1
    assert traj.failure_events == 1
    assert traj.failures == horizon
    assert sum(c > 0.0 for c in traj.failure_costs) == horizon
    assert sum(traj.failure_costs) == horizon * 7.0
