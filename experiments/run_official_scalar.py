"""Run official scalar benchmark matrix with literature baselines only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.env import ScalarEnvConfig, ScalarPhysicalEnv
from benchmarks.scalar_dual.filters import GaussianBelief
from benchmarks.scalar_dual.regimes import ScalarRegimeConfig, generate_b_path
from benchmarks.scalar_dual.rollout import run_scalar_rollout
from controllers.official import (
    ArcariDualSMPCScalar,
    ArcariPassiveExploitationScalar,
    KHDualControlScalar,
    NominalMPCScalar,
    OfficialScalarConfig,
    OracleTrendScalar,
    TVGPLCBScalar,
)


def short_horizon_change_points(horizon: int, n_values: int = 3) -> tuple[int, ...]:
    """Place piecewise changes inside the run, even for short official horizons."""
    if horizon < 2:
        return ()
    needed = min(n_values - 1, horizon - 1)
    return tuple(max(1, min(horizon - 1, int(round(horizon * k / n_values)))) for k in range(1, needed + 1))


def make_regime(kind: str, horizon: int) -> ScalarRegimeConfig:
    if kind == "static":
        return ScalarRegimeConfig(kind="static", horizon=horizon, base=2.0)
    if kind == "piecewise":
        values = (2.0, 1.2, 2.2)
        return ScalarRegimeConfig(kind="fixed_jumps", horizon=horizon, fixed_values=values, change_points=short_horizon_change_points(horizon, len(values)))
    if kind == "drifting":
        # Official short-horizon drifting should be visibly non-static; label/config columns below record the exact OU setting.
        return ScalarRegimeConfig(kind="ou", horizon=horizon, base=2.0, sigma=0.05)
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
    b_path: np.ndarray,
    noise_path: np.ndarray,
    cost: ScalarCost,
    config: OfficialScalarConfig,
    seed: int,
    discrepancy: float,
    include_diagnostics: bool = False,
):
    controllers = [
        KHDualControlScalar(GaussianBelief(mean=1.0, var=10.0), cost, config),
        ArcariDualSMPCScalar(GaussianBelief(mean=1.0, var=10.0), cost, config, seed=seed),
        TVGPLCBScalar(cost, config),
        OracleTrendScalar(b_path, cost, config, discrepancy_quadratic=discrepancy, noise_path=noise_path),
    ]
    if include_diagnostics:
        controllers.extend(
            [
                ArcariPassiveExploitationScalar(GaussianBelief(mean=1.0, var=10.0), cost, config, seed=seed),
                NominalMPCScalar(GaussianBelief(mean=1.0, var=10.0), cost, config),
            ]
        )
    return controllers


def run_one_setting(args: argparse.Namespace, twin_gap: str, regime_kind: str) -> list[dict[str, float | str | int]]:
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
        action_grid_size=args.action_grid_size,
        continuous_actions=args.continuous_actions,
        optimizer_grid_size=args.optimizer_grid_size,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_xatol=args.optimizer_xatol,
        process_var=args.process_std**2,
        smpc_dual_horizon=args.smpc_dual_horizon,
        smpc_scenarios=args.smpc_scenarios,
    )
    regime = make_regime(regime_kind, args.horizon)
    discrepancy = args.gap_discrepancy_quadratic if twin_gap == "gap" else 0.0
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        b_path = generate_b_path(regime, rng)
        process_noise = rng.normal(0.0, args.process_std, size=args.horizon)
        obs_noise = rng.normal(0.0, args.observation_std, size=args.horizon)
        controllers = build_controllers(b_path, process_noise, cost, config, seed, discrepancy, args.include_diagnostics)
        b_summary = path_summary(b_path, "b")
        regime_metadata = {
            "regime_config_kind": regime.kind,
            "regime_sigma": regime.sigma,
            "regime_rho": regime.rho,
            "regime_change_points": ";".join(str(cp) for cp in regime.change_points),
            "regime_variation_label": "nontrivial_ou_drift" if regime_kind == "drifting" else regime_kind,
        }
        seed_costs = {}
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
            seed_costs[controller.name] = traj.total_cost
            state_cost = float(np.sum(traj.state_costs))
            energy_cost = float(np.sum(traj.energy_costs))
            switch_cost = float(np.sum(traj.switch_costs))
            nonsmooth_switch_cost = float(np.sum(traj.nonsmooth_switch_costs))
            rows.append(
                {
                    "environment": "scalar",
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
                "mean_terminal_cost": float(np.mean([float(r["terminal_cost"]) for r in sub])),
                "mean_abs_action": float(np.mean([float(r["mean_abs_action"]) for r in sub])),
                "mean_frac_zero_action": float(np.mean([float(r["frac_zero_action"]) for r in sub])),
                "mean_action_changes": float(np.mean([float(r["action_changes"]) for r in sub])),
                "mean_b_path_range": float(np.mean([float(r["b_path_range"]) for r in sub])),
                "min_b_path_n_changes": int(min(int(r["b_path_n_changes"]) for r in sub)),
                "min_b_path_first_change_step": int(min(int(r["b_path_first_change_step"]) for r in sub)),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
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
    parser.add_argument("--action-grid-size", type=int, default=21)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--optimizer-grid-size", type=int, default=81)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--optimizer-xatol", type=float, default=1e-4)
    parser.add_argument("--smpc-dual-horizon", type=int, default=2)
    parser.add_argument("--smpc-scenarios", type=int, default=3)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--include-diagnostics", action="store_true")
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
    write_csv(args.out_dir / "scalar_main_raw.csv", main_rows)
    write_csv(args.out_dir / "scalar_main_summary.csv", summary)
    if diagnostic_rows:
        write_csv(args.out_dir / "scalar_diagnostic_raw.csv", diagnostic_rows)
        write_csv(args.out_dir / "scalar_diagnostic_summary.csv", summarize(diagnostic_rows))
    print(f"wrote {len(main_rows)} raw rows and {len(summary)} summary rows to {args.out_dir}")
    if diagnostic_rows:
        print(f"wrote {len(diagnostic_rows)} diagnostic rows to {args.out_dir}")
    for row in summary:
        print(f"{row['environment']},{row['twin_gap']},{row['regime']},{row['baseline']},{row['mean_total_cost']:.4g},{row['mean_oracle_regret']:.4g}")


if __name__ == "__main__":
    main()
