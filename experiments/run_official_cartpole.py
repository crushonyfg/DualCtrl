"""Run official CartPole benchmark matrix with literature baselines only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams
from benchmarks.cartpole_twin.env import CartPoleEnvConfig, CartPolePhysicalEnv
from benchmarks.cartpole_twin.regimes import CartPoleRegimeConfig, generate_theta_path
from benchmarks.cartpole_twin.rollout import apply_initial_calibration, generate_initial_calibration_dataset, run_cartpole_rollout
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers.kh_gp import KHGPConfig, KHGPControllerCartPole
from controllers.official import (
    ArcariDualSMPCCartPole,
    ArcariPassiveExploitationCartPole,
    NominalMPCCartPole,
    OfficialCartPoleConfig,
    OracleTrendCartPole,
    TVGPLCBCartPole,
)


def short_horizon_change_points(horizon: int, n_values: int = 3) -> tuple[int, ...]:
    """Place piecewise changes inside the run, even for short official horizons."""
    if horizon < 2:
        return ()
    needed = min(n_values - 1, horizon - 1)
    return tuple(max(1, min(horizon - 1, int(round(horizon * k / n_values)))) for k in range(1, needed + 1))


def make_regime(kind: str, horizon: int) -> CartPoleRegimeConfig:
    if kind == "static":
        return CartPoleRegimeConfig(kind="static", horizon=horizon, base=1.0)
    if kind == "piecewise":
        values = (1.0, 0.65, 1.25)
        return CartPoleRegimeConfig(kind="fixed_jumps", horizon=horizon, fixed_values=values, change_points=short_horizon_change_points(horizon, len(values)))
    if kind == "drifting":
        # Official short-horizon drifting should be visibly non-static; label/config columns below record the exact OU setting.
        return CartPoleRegimeConfig(kind="ou", horizon=horizon, base=1.0, sigma=0.02)
    raise ValueError(kind)


def path_summary(path: np.ndarray, prefix: str) -> dict[str, float | int | str]:
    changes = np.flatnonzero(~np.isclose(np.diff(path), 0.0)) + 1
    return {
        f"{prefix}_path_min": float(np.min(path)),
        f"{prefix}_path_max": float(np.max(path)),
        f"{prefix}_path_range": float(np.max(path) - np.min(path)),
        f"{prefix}_path_std": float(np.std(path)),
        f"{prefix}_path_n_unique": int(len(np.unique(np.round(path, decimals=12)))),
        f"{prefix}_path_n_changes": int(len(changes)),
        f"{prefix}_path_first_change_step": int(changes[0]) if len(changes) else -1,
        f"{prefix}_path_last_change_step": int(changes[-1]) if len(changes) else -1,
        f"{prefix}_path_values": ";".join(f"{v:.12g}" for v in path),
    }


def build_controllers(
    theta_path: np.ndarray,
    noise_path: np.ndarray,
    dynamics: CartPoleParams,
    cost: CartPoleCost,
    config: OfficialCartPoleConfig,
    kh_config: KHGPConfig,
    seed: int,
    include_diagnostics: bool = False,
):
    controllers = [
        KHGPControllerCartPole(dynamics, cost, kh_config),
        ArcariDualSMPCCartPole(GaussianBelief(mean=1.0, var=0.1), dynamics, cost, config, seed=seed),
        TVGPLCBCartPole(dynamics, cost, config),
        OracleTrendCartPole(theta_path, dynamics, cost, config, noise_path=noise_path),
    ]
    if include_diagnostics:
        controllers.extend(
            [
                ArcariPassiveExploitationCartPole(GaussianBelief(mean=1.0, var=0.1), dynamics, cost, config, seed=seed),
                NominalMPCCartPole(GaussianBelief(mean=1.0, var=0.1), dynamics, cost, config),
            ]
        )
    return controllers


def run_one_setting(args: argparse.Namespace, twin_gap: str, regime_kind: str) -> list[dict[str, float | str | int]]:
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
        action_grid_size=args.action_grid_size,
        continuous_actions=args.continuous_actions,
        optimizer_grid_size=args.optimizer_grid_size,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_xatol=args.optimizer_xatol,
        smpc_dual_horizon=args.smpc_dual_horizon,
        smpc_scenarios=args.smpc_scenarios,
    )
    kh_config = KHGPConfig(
        horizon=args.planning_horizon,
        action_grid_size=args.action_grid_size,
        continuous_actions=args.continuous_actions,
        optimizer_grid_size=args.optimizer_grid_size,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_xatol=args.optimizer_xatol,
        num_features=args.kh_gp_features,
        lengthscale=args.kh_gp_lengthscale,
        seed=0,
    )
    regime = make_regime(regime_kind, args.horizon)
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        theta_path = generate_theta_path(regime, rng)
        process_std = np.array(CartPoleEnvConfig().process_std)
        process_noise = rng.normal(0.0, process_std, size=(args.horizon, 4))
        theta_summary = path_summary(theta_path, "theta")
        regime_metadata = {
            "regime_config_kind": regime.kind,
            "regime_sigma": regime.sigma,
            "regime_rho": regime.rho,
            "regime_change_points": ";".join(str(cp) for cp in regime.change_points),
            "regime_variation_label": "nontrivial_ou_drift" if regime_kind == "drifting" else regime_kind,
        }
        controllers = build_controllers(theta_path, process_noise, dynamics if twin_gap == "gap" else nominal_dynamics, cost, config, kh_config, seed, args.include_diagnostics)
        calibration_env = CartPolePhysicalEnv(CartPoleEnvConfig(), dynamics, cost, theta_path, process_noise)
        calibration_dataset = generate_initial_calibration_dataset(
            calibration_env,
            args.initial_calibration_policy,
            args.n_initial_calibration,
            np.random.default_rng(seed + 1_000_003),
        )
        calibration_counts = {}
        for controller in controllers:
            if controller.name != "oracle_trend":
                if hasattr(controller, "dynamics"):
                    controller.dynamics = nominal_dynamics
                calibration_counts[controller.name] = apply_initial_calibration(controller, calibration_dataset)
            else:
                calibration_counts[controller.name] = 0
        seed_costs = {}
        for controller in controllers:
            env = CartPolePhysicalEnv(CartPoleEnvConfig(), dynamics, cost, theta_path, process_noise)
            traj = run_cartpole_rollout(env, controller, cost, args.observation_interval)
            seed_costs[controller.name] = traj.total_cost
            task_cost = float(np.sum(traj.task_costs))
            energy_cost = float(np.sum(traj.energy_costs))
            switch_cost = float(np.sum(traj.switch_costs))
            nonsmooth_switch_cost = float(np.sum(traj.nonsmooth_switch_costs))
            failure_cost = float(np.sum(traj.failure_costs))
            rows.append(
                {
                    "environment": "cartpole",
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
                    "n_initial_calibration": calibration_counts[controller.name],
                    "initial_calibration_policy": args.initial_calibration_policy,
                    "observation_interval": args.observation_interval,
                }
            )
        for row in rows[-len(controllers):]:
            row["oracle_regret"] = float(row["total_cost"] - seed_costs["oracle_trend"])
    return rows


def summarize(rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str]]:
    summary = []
    keys = sorted({(r["environment"], r["twin_gap"], r["regime"], r["baseline"]) for r in rows})
    for environment, twin_gap, regime, baseline in keys:
        sub = [r for r in rows if (r["environment"], r["twin_gap"], r["regime"], r["baseline"]) == (environment, twin_gap, regime, baseline)]
        vals = [float(r["total_cost"]) for r in sub]
        regrets = [float(r["oracle_regret"]) for r in sub]
        failures = [float(r["failures"]) for r in sub]
        failure_events = [float(r.get("failure_events", 0.0)) for r in sub]
        summary.append(
            {
                "environment": environment,
                "twin_gap": twin_gap,
                "regime": regime,
                "baseline": baseline,
                "mean_total_cost": float(np.mean(vals)),
                "stderr_total_cost": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "mean_net_reward": float(np.mean([float(r["net_reward"]) for r in sub])),
                "mean_oracle_regret": float(np.mean(regrets)),
                "mean_task_cost": float(np.mean([float(r["task_cost"]) for r in sub])),
                "mean_energy_cost": float(np.mean([float(r["energy_cost"]) for r in sub])),
                "mean_switch_cost": float(np.mean([float(r["switch_cost"]) for r in sub])),
                "mean_failure_cost": float(np.mean([float(r["failure_cost"]) for r in sub])),
                "mean_terminal_cost": float(np.mean([float(r["terminal_cost"]) for r in sub])),
                "mean_violation_steps": float(np.mean([float(r["violation_steps"]) for r in sub])),
                "mean_failures": float(np.mean(failures)),
                "mean_failure_events": float(np.mean(failure_events)),
                "max_failure_events": int(max(int(r["failure_events"]) for r in sub)),
                "mean_abs_action": float(np.mean([float(r["mean_abs_action"]) for r in sub])),
                "mean_frac_zero_action": float(np.mean([float(r["frac_zero_action"]) for r in sub])),
                "mean_action_changes": float(np.mean([float(r["action_changes"]) for r in sub])),
                "mean_theta_path_range": float(np.mean([float(r["theta_path_range"]) for r in sub])),
                "min_theta_path_n_changes": int(min(int(r["theta_path_n_changes"]) for r in sub)),
                "min_theta_path_first_change_step": int(min(int(r["theta_path_first_change_step"]) for r in sub)),
                "n": len(vals),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_config(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_initial_calibration": int(args.n_initial_calibration),
        "initial_calibration_policy": args.initial_calibration_policy,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-horizon", type=int, default=4)
    parser.add_argument("--action-grid-size", type=int, default=7)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--optimizer-grid-size", type=int, default=81)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--optimizer-xatol", type=float, default=1e-4)
    parser.add_argument("--smpc-dual-horizon", type=int, default=2)
    parser.add_argument("--smpc-scenarios", type=int, default=3)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--n-initial-calibration", type=int, default=0)
    parser.add_argument("--initial-calibration-policy", choices=["small_random", "zero", "grid_probe"], default="small_random")
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--nonsmooth-switch-cost", type=float, default=0.0)
    parser.add_argument("--nonsmooth-switch-threshold", type=float, default=1e-9)
    parser.add_argument("--kh-gp-features", type=int, default=16)
    parser.add_argument("--kh-gp-lengthscale", type=float, default=1.0)
    parser.add_argument("--gap-lag-alpha", type=float, default=0.6)
    parser.add_argument("--gap-friction", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/official"))
    args = parser.parse_args()

    rows = []
    for twin_gap in ("no_gap", "gap"):
        for regime in ("static", "piecewise", "drifting"):
            rows.extend(run_one_setting(args, twin_gap, regime))
    diagnostic_names = {"arcari_l0_passive_exploitation", "nominal_mpc_ce"}
    main_rows = [row for row in rows if row["baseline"] not in diagnostic_names]
    diagnostic_rows = [row for row in rows if row["baseline"] in diagnostic_names]
    summary = summarize(main_rows)
    write_csv(args.out_dir / "cartpole_main_raw.csv", main_rows)
    write_csv(args.out_dir / "cartpole_main_summary.csv", summary)
    write_config(args.out_dir / "config.json", args)
    if diagnostic_rows:
        write_csv(args.out_dir / "cartpole_diagnostic_raw.csv", diagnostic_rows)
        write_csv(args.out_dir / "cartpole_diagnostic_summary.csv", summarize(diagnostic_rows))
    print(f"wrote {len(main_rows)} raw rows and {len(summary)} summary rows to {args.out_dir}")
    if diagnostic_rows:
        print(f"wrote {len(diagnostic_rows)} diagnostic rows to {args.out_dir}")
    for row in summary:
        print(f"{row['environment']},{row['twin_gap']},{row['regime']},{row['baseline']},{row['mean_total_cost']:.4g},{row['mean_oracle_regret']:.4g}")


if __name__ == "__main__":
    main()
