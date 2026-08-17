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
    KHDualControlScalar,
    OfficialScalarConfig,
    OracleTrendScalar,
    TVGPLCBScalar,
)


def make_regime(kind: str, horizon: int) -> ScalarRegimeConfig:
    if kind == "static":
        return ScalarRegimeConfig(kind="static", horizon=horizon, base=2.0)
    if kind == "piecewise":
        return ScalarRegimeConfig(kind="fixed_jumps", horizon=horizon, fixed_values=(2.0, 1.2, 2.2), change_points=(horizon // 3, 2 * horizon // 3))
    if kind == "drifting":
        return ScalarRegimeConfig(kind="ou", horizon=horizon, base=2.0, sigma=0.015)
    raise ValueError(kind)


def build_controllers(b_path: np.ndarray, cost: ScalarCost, config: OfficialScalarConfig, seed: int, discrepancy: float):
    return [
        KHDualControlScalar(GaussianBelief(mean=1.0, var=10.0), cost, config),
        ArcariDualSMPCScalar(GaussianBelief(mean=1.0, var=10.0), cost, config, seed=seed),
        TVGPLCBScalar(cost, config),
        OracleTrendScalar(b_path, cost, config, discrepancy_quadratic=discrepancy),
    ]


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
        controllers = build_controllers(b_path, cost, config, seed, discrepancy)
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
            rows.append(
                {
                    "environment": "scalar",
                    "twin_gap": twin_gap,
                    "regime": regime_kind,
                    "seed": seed,
                    "baseline": controller.name,
                    "total_cost": traj.total_cost,
                    "state_cost": float(np.sum(traj.state_costs)),
                    "energy_cost": float(np.sum(traj.energy_costs)),
                    "switch_cost": float(np.sum(traj.switch_costs)),
                    "nonsmooth_switch_cost": float(np.sum(traj.nonsmooth_switch_costs)),
                    "terminal_cost": traj.terminal_cost,
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
        vals = [float(r["total_cost"]) for r in rows if (r["environment"], r["twin_gap"], r["regime"], r["baseline"]) == (environment, twin_gap, regime, baseline)]
        regrets = [float(r["oracle_regret"]) for r in rows if (r["environment"], r["twin_gap"], r["regime"], r["baseline"]) == (environment, twin_gap, regime, baseline)]
        summary.append(
            {
                "environment": environment,
                "twin_gap": twin_gap,
                "regime": regime,
                "baseline": baseline,
                "mean_total_cost": float(np.mean(vals)),
                "stderr_total_cost": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "mean_oracle_regret": float(np.mean(regrets)),
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
    parser.add_argument("--smpc-dual-horizon", type=int, default=2)
    parser.add_argument("--smpc-scenarios", type=int, default=3)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/official"))
    args = parser.parse_args()

    rows = []
    for twin_gap in ("no_gap", "gap"):
        for regime in ("static", "piecewise", "drifting"):
            rows.extend(run_one_setting(args, twin_gap, regime))
    summary = summarize(rows)
    write_csv(args.out_dir / "scalar_main_raw.csv", rows)
    write_csv(args.out_dir / "scalar_main_summary.csv", summary)
    print(f"wrote {len(rows)} raw rows and {len(summary)} summary rows to {args.out_dir}")
    for row in summary:
        print(f"{row['environment']},{row['twin_gap']},{row['regime']},{row['baseline']},{row['mean_total_cost']:.4g},{row['mean_oracle_regret']:.4g}")


if __name__ == "__main__":
    main()
