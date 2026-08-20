"""Run diagnostic controls separately from the official three-baseline tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams
from benchmarks.cartpole_twin.env import CartPoleEnvConfig, CartPolePhysicalEnv
from benchmarks.cartpole_twin.regimes import generate_theta_path
from benchmarks.cartpole_twin.rollout import run_cartpole_rollout
from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.env import ScalarEnvConfig, ScalarPhysicalEnv
from benchmarks.scalar_dual.filters import GaussianBelief
from benchmarks.scalar_dual.regimes import generate_b_path
from benchmarks.scalar_dual.rollout import run_scalar_rollout
from controllers.kh_gp import KHGPConfig
from controllers.official import (
    ArcariPassiveExploitationCartPole,
    ArcariPassiveExploitationScalar,
    NominalMPCCartPole,
    NominalMPCScalar,
    OfficialCartPoleConfig,
    OfficialScalarConfig,
)
from experiments.run_official_cartpole import make_regime as make_cartpole_regime
from experiments.run_official_cartpole import path_summary as cartpole_path_summary
from experiments.run_official_cartpole import summarize as summarize_cartpole
from experiments.run_official_cartpole import write_csv
from experiments.run_official_scalar import make_regime as make_scalar_regime
from experiments.run_official_scalar import path_summary as scalar_path_summary
from experiments.run_official_scalar import summarize as summarize_scalar


def build_scalar_diagnostics(cost: ScalarCost, config: OfficialScalarConfig, seed: int, include_nominal_mpc: bool):
    controllers = [
        ArcariPassiveExploitationScalar(GaussianBelief(mean=1.0, var=10.0), cost, config, seed=seed),
    ]
    if include_nominal_mpc:
        controllers.append(NominalMPCScalar(GaussianBelief(mean=1.0, var=10.0), cost, config))
    return controllers


def build_cartpole_diagnostics(dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig, include_nominal_mpc: bool, seed: int):
    controllers = [
        ArcariPassiveExploitationCartPole(GaussianBelief(mean=1.0, var=0.1), dynamics, cost, config, seed=seed),
    ]
    if include_nominal_mpc:
        controllers.append(NominalMPCCartPole(GaussianBelief(mean=1.0, var=0.1), dynamics, cost, config))
    return controllers


def run_scalar_diagnostic_setting(args: argparse.Namespace, twin_gap: str, regime_kind: str) -> list[dict[str, float | str | int]]:
    rows = []
    cost = ScalarCost(
        ScalarCostConfig(
            energy_weight=args.energy_weight,
            switch_weight=args.switch_weight,
            nonsmooth_switch_cost=args.nonsmooth_switch_cost,
            nonsmooth_switch_threshold=args.nonsmooth_switch_threshold,
        )
    )
    config = OfficialScalarConfig(
        horizon=args.planning_horizon,
        action_grid_size=args.scalar_action_grid_size,
        continuous_actions=args.continuous_actions,
        optimizer_grid_size=args.optimizer_grid_size,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_xatol=args.optimizer_xatol,
        process_var=args.process_std**2,
        smpc_dual_horizon=args.smpc_dual_horizon,
        smpc_scenarios=args.smpc_scenarios,
    )
    regime = make_scalar_regime(regime_kind, args.horizon)
    discrepancy = args.gap_discrepancy_quadratic if twin_gap == "gap" else 0.0
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        b_path = generate_b_path(regime, rng)
        process_noise = rng.normal(0.0, args.process_std, size=args.horizon)
        obs_noise = rng.normal(0.0, args.observation_std, size=args.horizon)
        controllers = build_scalar_diagnostics(cost, config, seed, args.include_nominal_mpc)
        b_summary = scalar_path_summary(b_path, "b")
        regime_metadata = {
            "regime_config_kind": regime.kind,
            "regime_sigma": regime.sigma,
            "regime_rho": regime.rho,
            "regime_change_points": ";".join(str(cp) for cp in regime.change_points),
            "regime_variation_label": "nontrivial_ou_drift" if regime_kind == "drifting" else regime_kind,
        }
        for controller in controllers:
            env = ScalarPhysicalEnv(
                ScalarEnvConfig(
                    process_std=args.process_std,
                    observation_std=args.observation_std,
                    x0=args.x0,
                    discrepancy_quadratic=discrepancy,
                ),
                cost,
                b_path=b_path,
                process_noise=process_noise,
                observation_noise=obs_noise,
            )
            obs_var = args.process_std**2 + args.observation_std**2
            traj = run_scalar_rollout(env, controller, cost, args.observation_interval, obs_var)
            state_cost = float(np.sum(traj.state_costs))
            energy_cost = float(np.sum(traj.energy_costs))
            switch_cost = float(np.sum(traj.switch_costs))
            nonsmooth_switch_cost = float(np.sum(traj.nonsmooth_switch_costs))
            rows.append(
                {
                    "environment": "scalar",
                    "diagnostic": "true",
                    "twin_gap": twin_gap,
                    "regime": regime_kind,
                    "seed": seed,
                    "baseline": controller.name,
                    **regime_metadata,
                    **b_summary,
                    "total_cost": traj.total_cost,
                    "net_reward": -traj.total_cost,
                    "state_cost": state_cost,
                    "task_cost": state_cost,
                    "acc_task_reward": -state_cost,
                    "energy_cost": energy_cost,
                    "acc_energy_cost": energy_cost,
                    "switch_cost": switch_cost,
                    "nonsmooth_switch_cost": nonsmooth_switch_cost,
                    "acc_switch_cost": switch_cost + nonsmooth_switch_cost,
                    "failure_cost": 0.0,
                    "acc_failure_cost": 0.0,
                    "terminal_cost": traj.terminal_cost,
                    "mean_abs_action": traj.mean_abs_action,
                    "frac_zero_action": traj.frac_zero_action,
                    "action_changes": traj.action_changes,
                    "physical_transitions": traj.physical_transitions,
                    "observed_transitions": traj.observed_transitions,
                    "observation_interval": args.observation_interval,
                    "oracle_regret": float("nan"),
                }
            )
    return rows


def run_cartpole_diagnostic_setting(args: argparse.Namespace, twin_gap: str, regime_kind: str) -> list[dict[str, float | str | int]]:
    rows = []
    cost = CartPoleCost(
        CartPoleCostConfig(
            nonsmooth_switch_cost=args.nonsmooth_switch_cost,
            nonsmooth_switch_threshold=args.nonsmooth_switch_threshold,
        )
    )
    dynamics = CartPoleParams(
        actuator_lag_alpha=args.gap_lag_alpha if twin_gap == "gap" else 1.0,
        coulomb_friction=args.gap_friction if twin_gap == "gap" else 0.0,
    )
    nominal_dynamics = CartPoleParams()
    config = OfficialCartPoleConfig(
        horizon=args.planning_horizon,
        action_grid_size=args.cartpole_action_grid_size,
        continuous_actions=args.continuous_actions,
        optimizer_grid_size=args.optimizer_grid_size,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_xatol=args.optimizer_xatol,
        smpc_dual_horizon=args.smpc_dual_horizon,
        smpc_scenarios=args.smpc_scenarios,
    )
    KHGPConfig(
        horizon=args.planning_horizon,
        action_grid_size=args.cartpole_action_grid_size,
        continuous_actions=args.continuous_actions,
        optimizer_grid_size=args.optimizer_grid_size,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_xatol=args.optimizer_xatol,
        num_features=args.kh_gp_features,
        lengthscale=args.kh_gp_lengthscale,
        seed=0,
    )
    regime = make_cartpole_regime(regime_kind, args.horizon)
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        theta_path = generate_theta_path(regime, rng)
        process_std = np.array(CartPoleEnvConfig().process_std)
        process_noise = rng.normal(0.0, process_std, size=(args.horizon, 4))
        theta_summary = cartpole_path_summary(theta_path, "theta")
        regime_metadata = {
            "regime_config_kind": regime.kind,
            "regime_sigma": regime.sigma,
            "regime_rho": regime.rho,
            "regime_change_points": ";".join(str(cp) for cp in regime.change_points),
            "regime_variation_label": "nontrivial_ou_drift" if regime_kind == "drifting" else regime_kind,
        }
        controllers = build_cartpole_diagnostics(nominal_dynamics, cost, config, args.include_nominal_mpc, seed)
        for controller in controllers:
            if hasattr(controller, "dynamics"):
                controller.dynamics = nominal_dynamics
            env = CartPolePhysicalEnv(CartPoleEnvConfig(), dynamics, cost, theta_path, process_noise)
            traj = run_cartpole_rollout(env, controller, cost, args.observation_interval)
            task_cost = float(np.sum(traj.task_costs))
            energy_cost = float(np.sum(traj.energy_costs))
            switch_cost = float(np.sum(traj.switch_costs))
            nonsmooth_switch_cost = float(np.sum(traj.nonsmooth_switch_costs))
            failure_cost = float(np.sum(traj.failure_costs))
            rows.append(
                {
                    "environment": "cartpole",
                    "diagnostic": "true",
                    "twin_gap": twin_gap,
                    "regime": regime_kind,
                    "seed": seed,
                    "baseline": controller.name,
                    **regime_metadata,
                    **theta_summary,
                    "total_cost": traj.total_cost,
                    "net_reward": -traj.total_cost,
                    "task_cost": task_cost,
                    "acc_task_reward": -task_cost,
                    "energy_cost": energy_cost,
                    "acc_energy_cost": energy_cost,
                    "switch_cost": switch_cost,
                    "nonsmooth_switch_cost": nonsmooth_switch_cost,
                    "acc_switch_cost": switch_cost + nonsmooth_switch_cost,
                    "failure_cost": failure_cost,
                    "acc_failure_cost": failure_cost,
                    "terminal_cost": traj.terminal_cost,
                    "violation_steps": traj.failures,
                    "failures": traj.failures,
                    "failure_events": traj.failure_events,
                    "mean_abs_action": traj.mean_abs_action,
                    "frac_zero_action": traj.frac_zero_action,
                    "action_changes": traj.action_changes,
                    "physical_transitions": traj.physical_transitions,
                    "observed_transitions": traj.observed_transitions,
                    "observation_interval": args.observation_interval,
                    "oracle_regret": float("nan"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x0", type=float, default=1.0)
    parser.add_argument("--process-std", type=float, default=np.sqrt(0.1))
    parser.add_argument("--observation-std", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--switch-weight", type=float, default=0.0)
    parser.add_argument("--nonsmooth-switch-cost", type=float, default=0.0)
    parser.add_argument("--nonsmooth-switch-threshold", type=float, default=1e-9)
    parser.add_argument("--gap-discrepancy-quadratic", type=float, default=0.1)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--scalar-action-grid-size", type=int, default=21)
    parser.add_argument("--cartpole-action-grid-size", type=int, default=7)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--optimizer-grid-size", type=int, default=81)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--optimizer-xatol", type=float, default=1e-4)
    parser.add_argument("--smpc-dual-horizon", type=int, default=2)
    parser.add_argument("--smpc-scenarios", type=int, default=3)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--kh-gp-features", type=int, default=16)
    parser.add_argument("--kh-gp-lengthscale", type=float, default=1.0)
    parser.add_argument("--gap-lag-alpha", type=float, default=0.6)
    parser.add_argument("--gap-friction", type=float, default=0.02)
    parser.add_argument("--include-nominal-mpc", action="store_true", help="Include certainty-equivalent nominal MPC diagnostic.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/diagnostics"))
    args = parser.parse_args()

    scalar_rows = []
    cartpole_rows = []
    for twin_gap in ("no_gap", "gap"):
        for regime in ("static", "piecewise", "drifting"):
            scalar_rows.extend(run_scalar_diagnostic_setting(args, twin_gap, regime))
            cartpole_rows.extend(run_cartpole_diagnostic_setting(args, twin_gap, regime))

    scalar_summary = summarize_scalar(scalar_rows)
    cartpole_summary = summarize_cartpole(cartpole_rows)
    write_csv(args.out_dir / "scalar_diagnostics_raw.csv", scalar_rows)
    write_csv(args.out_dir / "scalar_diagnostics_summary.csv", scalar_summary)
    write_csv(args.out_dir / "cartpole_diagnostics_raw.csv", cartpole_rows)
    write_csv(args.out_dir / "cartpole_diagnostics_summary.csv", cartpole_summary)
    print(f"wrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
