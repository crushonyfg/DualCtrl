from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams
from benchmarks.cartpole_twin.env import CartPoleEnvConfig, CartPolePhysicalEnv
from benchmarks.cartpole_twin.rollout import apply_initial_calibration, generate_initial_calibration_dataset
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers.kh_gp import KHGPConfig, KHGPControllerCartPole
from controllers.official import ArcariDualSMPCCartPole, OfficialCartPoleConfig, TVGPLCBCartPole
from experiments import run_official_cartpole


def _physical_setup(horizon: int = 5):
    dynamics = CartPoleParams()
    cost = CartPoleCost(CartPoleCostConfig())
    theta_path = np.linspace(0.9, 1.1, horizon)
    process_noise = np.zeros((horizon, 4), dtype=float)
    return dynamics, cost, theta_path, process_noise


def test_initial_calibration_feeds_same_number_to_cartpole_baselines() -> None:
    dynamics, cost, theta_path, process_noise = _physical_setup()
    env = CartPolePhysicalEnv(CartPoleEnvConfig(), dynamics, cost, theta_path, process_noise)
    dataset = generate_initial_calibration_dataset(env, "grid_probe", 3, np.random.default_rng(0))
    config = OfficialCartPoleConfig(horizon=2, action_grid_size=3, smpc_dual_horizon=1, smpc_scenarios=1)
    controllers = [
        KHGPControllerCartPole(dynamics, cost, KHGPConfig(horizon=2, action_grid_size=3, num_features=8, seed=1)),
        ArcariDualSMPCCartPole(GaussianBelief(mean=1.0, var=0.1), dynamics, cost, config, seed=1),
        TVGPLCBCartPole(dynamics, cost, config),
    ]

    counts = [apply_initial_calibration(controller, dataset) for controller in controllers]

    assert counts == [3, 3, 3]
    assert [controller.n_initial_calibration_observations for controller in controllers] == [3, 3, 3]
    assert controllers[0].t == 0
    assert controllers[1].t == 0
    assert controllers[2].adapter.t == 4
    assert len(controllers[2].adapter.posterior.observations) == 3


def test_kh_gp_initial_calibration_changes_posterior() -> None:
    dynamics, cost, theta_path, process_noise = _physical_setup()
    env = CartPolePhysicalEnv(CartPoleEnvConfig(), dynamics, cost, theta_path, process_noise)
    dataset = generate_initial_calibration_dataset(env, "grid_probe", 2, np.random.default_rng(0))
    controller = KHGPControllerCartPole(
        dynamics,
        cost,
        KHGPConfig(horizon=2, action_grid_size=3, num_features=8, prior_var=1.0, noise_var=0.05, seed=2),
    )
    mean_before = controller.model.flat_mean()
    cov_trace_before = np.trace(controller.model.flat_cov())

    apply_initial_calibration(controller, dataset)

    assert not np.allclose(controller.model.flat_mean(), mean_before)
    assert np.trace(controller.model.flat_cov()) < cov_trace_before


def test_official_cartpole_initial_calibration_outputs_config_and_raw_counts(tmp_path) -> None:
    out_dir = tmp_path / "cartpole"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run_official_cartpole",
            "--out-dir",
            str(out_dir),
            "--horizon",
            "2",
            "--n-seeds",
            "1",
            "--planning-horizon",
            "1",
            "--action-grid-size",
            "3",
            "--smpc-dual-horizon",
            "1",
            "--smpc-scenarios",
            "1",
            "--n-initial-calibration",
            "1",
            "--initial-calibration-policy",
            "zero",
        ],
        check=True,
    )

    config = json.loads((out_dir / "config.json").read_text())
    assert config["n_initial_calibration"] == 1
    assert config["initial_calibration_policy"] == "zero"
    with (out_dir / "cartpole_main_raw.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {row["n_initial_calibration"] for row in rows if row["baseline"] != "oracle_trend"} == {"1"}
    assert {row["n_initial_calibration"] for row in rows if row["baseline"] == "oracle_trend"} == {"0"}
    assert {row["physical_transitions"] for row in rows} == {"2"}


def test_run_one_setting_records_initial_calibration_without_deployment_reward() -> None:
    args = argparse.Namespace(
        horizon=2,
        n_seeds=1,
        seed=0,
        planning_horizon=1,
        action_grid_size=3,
        continuous_actions=False,
        optimizer_grid_size=81,
        optimizer_maxiter=100,
        optimizer_xatol=1e-4,
        smpc_dual_horizon=1,
        smpc_scenarios=1,
        observation_interval=1,
        include_diagnostics=False,
        nonsmooth_switch_cost=0.0,
        nonsmooth_switch_threshold=1e-9,
        kh_gp_features=8,
        kh_gp_lengthscale=1.0,
        gap_lag_alpha=0.6,
        gap_friction=0.02,
        n_initial_calibration=2,
        initial_calibration_policy="zero",
    )

    rows = run_official_cartpole.run_one_setting(args, "no_gap", "static")

    assert {row["physical_transitions"] for row in rows} == {2}
    assert {row["observed_transitions"] for row in rows} == {2}
    non_oracle = [row for row in rows if row["baseline"] != "oracle_trend"]
    assert {row["n_initial_calibration"] for row in non_oracle} == {2}
