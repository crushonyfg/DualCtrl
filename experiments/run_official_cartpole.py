"""Run official CartPole benchmark matrix with literature baselines only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams
from benchmarks.cartpole_twin.env import CartPoleEnvConfig, CartPolePhysicalEnv
from benchmarks.cartpole_twin.regimes import CartPoleRegimeConfig, generate_theta_path
from benchmarks.cartpole_twin.rollout import run_cartpole_rollout
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers.kh_gp import KHGPConfig, KHGPControllerCartPole
from controllers.official import (
    ArcariDualSMPCCartPole,
    OfficialCartPoleConfig,
    OracleTrendCartPole,
    TVGPLCBCartPole,
)


def make_regime(kind: str, horizon: int) -> CartPoleRegimeConfig:
    if kind == "static":
        return CartPoleRegimeConfig(kind="static", horizon=horizon, base=1.0)
    if kind == "piecewise":
        return CartPoleRegimeConfig(kind="fixed_jumps", horizon=horizon, fixed_values=(1.0, 0.65, 1.25), change_points=(horizon // 3, 2 * horizon // 3))
    if kind == "drifting":
        return CartPoleRegimeConfig(kind="ou", horizon=horizon, base=1.0, sigma=0.005)
    raise ValueError(kind)


def build_controllers(theta_path: np.ndarray, noise_path: np.ndarray, dynamics: CartPoleParams, cost: CartPoleCost, config: OfficialCartPoleConfig, kh_config: KHGPConfig, seed: int):
    return [
        KHGPControllerCartPole(dynamics, cost, kh_config),
        ArcariDualSMPCCartPole(GaussianBelief(mean=1.0, var=0.1), dynamics, cost, config, seed=seed),
        TVGPLCBCartPole(dynamics, cost, config),
        OracleTrendCartPole(theta_path, dynamics, cost, config, noise_path=noise_path),
    ]


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
        smpc_dual_horizon=args.smpc_dual_horizon,
        smpc_scenarios=args.smpc_scenarios,
    )
    kh_config = KHGPConfig(
        horizon=args.planning_horizon,
        action_grid_size=args.action_grid_size,
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
        controllers = build_controllers(theta_path, process_noise, dynamics if twin_gap == "gap" else nominal_dynamics, cost, config, kh_config, seed)
        for controller in controllers:
            if controller.name != "oracle_trend":
                if hasattr(controller, "dynamics"):
                    controller.dynamics = nominal_dynamics
        seed_costs = {}
        for controller in controllers:
            env = CartPolePhysicalEnv(CartPoleEnvConfig(), dynamics, cost, theta_path, process_noise)
            traj = run_cartpole_rollout(env, controller, cost, args.observation_interval)
            seed_costs[controller.name] = traj.total_cost
            rows.append(
                {
                    "environment": "cartpole",
                    "twin_gap": twin_gap,
                    "regime": regime_kind,
                    "seed": seed,
                    "baseline": controller.name,
                    "total_cost": traj.total_cost,
                    "task_cost": float(np.sum(traj.task_costs)),
                    "energy_cost": float(np.sum(traj.energy_costs)),
                    "switch_cost": float(np.sum(traj.switch_costs)),
                    "failure_cost": float(np.sum(traj.failure_costs)),
                    "terminal_cost": traj.terminal_cost,
                    "failures": traj.failures,
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
        failures = [float(r["failures"]) for r in sub]
        summary.append(
            {
                "environment": environment,
                "twin_gap": twin_gap,
                "regime": regime,
                "baseline": baseline,
                "mean_total_cost": float(np.mean(vals)),
                "stderr_total_cost": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "mean_oracle_regret": float(np.mean(regrets)),
                "mean_failures": float(np.mean(failures)),
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
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-horizon", type=int, default=4)
    parser.add_argument("--action-grid-size", type=int, default=7)
    parser.add_argument("--smpc-dual-horizon", type=int, default=2)
    parser.add_argument("--smpc-scenarios", type=int, default=3)
    parser.add_argument("--observation-interval", type=int, default=1)
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
    summary = summarize(rows)
    write_csv(args.out_dir / "cartpole_main_raw.csv", rows)
    write_csv(args.out_dir / "cartpole_main_summary.csv", summary)
    print(f"wrote {len(rows)} raw rows and {len(summary)} summary rows to {args.out_dir}")
    for row in summary:
        print(f"{row['environment']},{row['twin_gap']},{row['regime']},{row['baseline']},{row['mean_total_cost']:.4g},{row['mean_oracle_regret']:.4g}")


if __name__ == "__main__":
    main()
