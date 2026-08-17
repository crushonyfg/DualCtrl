"""Run scalar dual-control benchmark experiments.

This entry point is intentionally lightweight: it gives a first diagnostic
implementation for static/drift/jump/sparse settings before any proposed
planner is added.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.env import ScalarEnvConfig, ScalarPhysicalEnv
from benchmarks.scalar_dual.filters import GaussianBelief, GridBelief, random_walk_transition
from benchmarks.scalar_dual.regimes import ScalarRegimeConfig, generate_b_path
from benchmarks.scalar_dual.rollout import run_scalar_rollout
from controllers.scalar import (
    AnalyticBayesDualController,
    ApproxDualController,
    CautiousController,
    CertaintyEquivalentController,
    ClairvoyantController,
    GridBayesController,
    ScalarPlannerConfig,
    ScheduledProbeController,
)


def build_controllers(
    b_path: np.ndarray,
    cost: ScalarCost,
    planner: ScalarPlannerConfig,
    process_var_rw: float,
    include_grid: bool,
    grid_size: int,
    grid_branch_size: int,
) -> list:
    controllers = [
        CertaintyEquivalentController("ce_static", GaussianBelief(mean=1.0, var=10.0, process_var=0.0), cost, planner),
        CautiousController("cautious_static", GaussianBelief(mean=1.0, var=10.0, process_var=0.0), cost, planner),
        AnalyticBayesDualController("bayes2_static", GaussianBelief(mean=1.0, var=10.0, process_var=0.0), cost, planner),
        ApproxDualController("ad_static", GaussianBelief(mean=1.0, var=10.0, process_var=0.0), cost, planner),
        CertaintyEquivalentController("ce_rw", GaussianBelief(mean=1.0, var=10.0, process_var=process_var_rw), cost, planner),
        CautiousController("cautious_rw", GaussianBelief(mean=1.0, var=10.0, process_var=process_var_rw), cost, planner),
        AnalyticBayesDualController("bayes2_rw", GaussianBelief(mean=1.0, var=10.0, process_var=process_var_rw), cost, planner),
        ApproxDualController("ad_rw", GaussianBelief(mean=1.0, var=10.0, process_var=process_var_rw), cost, planner),
        ScheduledProbeController(GaussianBelief(mean=1.0, var=10.0, process_var=process_var_rw), cost, planner),
        ClairvoyantController(b_path, cost, planner),
    ]
    if include_grid:
        grid = np.linspace(0.3, 2.7, grid_size)
        trans = random_walk_transition(grid, np.sqrt(process_var_rw))
        belief = GridBelief.normal_prior(grid, mean=1.0, var=10.0, transition=trans)
        controllers.append(GridBayesController("grid_bayes_ref", belief, cost, planner, branch_grid_size=grid_branch_size))
    return controllers


def run(args: argparse.Namespace) -> list[dict[str, float | str | int]]:
    out_rows = []
    cost = ScalarCost(
        ScalarCostConfig(
            energy_weight=args.energy_weight,
            switch_weight=args.switch_weight,
            nonsmooth_switch_cost=args.nonsmooth_switch_cost,
            nonsmooth_switch_threshold=args.nonsmooth_switch_threshold,
        )
    )
    planner = ScalarPlannerConfig(
        horizon=args.planning_horizon,
        action_grid_size=args.action_grid_size,
        process_var=args.process_std * args.process_std,
    )
    regime = ScalarRegimeConfig(kind=args.regime, horizon=args.horizon, sigma=args.drift_sigma)
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        b_path = generate_b_path(regime, rng)
        process_noise = rng.normal(0.0, args.process_std, size=args.horizon)
        obs_noise = rng.normal(0.0, args.observation_std, size=args.horizon)
        controllers = build_controllers(
            b_path=b_path,
            cost=cost,
            planner=planner,
            process_var_rw=args.rw_process_std * args.rw_process_std,
            include_grid=args.include_grid_bayes,
            grid_size=args.grid_size,
            grid_branch_size=args.grid_branch_size,
        )
        seed_costs = {}
        for controller in controllers:
            env = ScalarPhysicalEnv(
                ScalarEnvConfig(
                    process_std=args.process_std,
                    observation_std=args.observation_std,
                    x0=args.x0,
                    discrepancy_quadratic=args.discrepancy_quadratic,
                    discrepancy_threshold=args.discrepancy_threshold,
                    discrepancy_threshold_value=args.discrepancy_threshold_value,
                ),
                cost,
                b_path=b_path,
                process_noise=process_noise,
                observation_noise=obs_noise,
            )
            obs_var = args.process_std * args.process_std + args.observation_std * args.observation_std
            traj = run_scalar_rollout(env, controller, cost, args.observation_interval, obs_var)
            seed_costs[controller.name] = traj.total_cost
            out_rows.append(
                {
                    "seed": seed,
                    "controller": controller.name,
                    "regime": args.regime,
                    "total_cost": traj.total_cost,
                    "terminal_cost": traj.terminal_cost,
                    "physical_transitions": traj.physical_transitions,
                    "observed_transitions": traj.observed_transitions,
                    "observation_interval": args.observation_interval,
                    "state_cost": float(np.sum(traj.state_costs)),
                    "energy_cost": float(np.sum(traj.energy_costs)),
                    "switch_cost": float(np.sum(traj.switch_costs)),
                    "nonsmooth_switch_cost": float(np.sum(traj.nonsmooth_switch_costs)),
                    "final_belief_mean": traj.belief_means[-1],
                    "final_belief_var": traj.belief_vars[-1],
                }
            )
        for row in out_rows[-len(controllers):]:
            row["clairvoyant_gap"] = float(row["total_cost"] - seed_costs.get("clairvoyant", np.nan))
            row["bayes_regret"] = float(row["total_cost"] - seed_costs.get("grid_bayes_ref", np.nan))
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["static", "ou", "fixed_jumps", "random_jumps", "multimodal"], default="static")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x0", type=float, default=1.0)
    parser.add_argument("--process-std", type=float, default=np.sqrt(0.1))
    parser.add_argument("--observation-std", type=float, default=0.0)
    parser.add_argument("--drift-sigma", type=float, default=0.015)
    parser.add_argument("--rw-process-std", type=float, default=0.015)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--switch-weight", type=float, default=0.0)
    parser.add_argument("--nonsmooth-switch-cost", type=float, default=0.0)
    parser.add_argument("--nonsmooth-switch-threshold", type=float, default=1e-9)
    parser.add_argument("--planning-horizon", type=int, default=2)
    parser.add_argument("--action-grid-size", type=int, default=61)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--include-grid-bayes", action="store_true")
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--grid-branch-size", type=int, default=9)
    parser.add_argument("--discrepancy-quadratic", type=float, default=0.0)
    parser.add_argument("--discrepancy-threshold", type=float, default=0.0)
    parser.add_argument("--discrepancy-threshold-value", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("reports/tables/scalar_results.csv"))
    args = parser.parse_args()

    rows = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    controllers = sorted({row["controller"] for row in rows})
    print(f"wrote {len(rows)} rows to {args.out}")
    for controller in controllers:
        costs = [float(row["total_cost"]) for row in rows if row["controller"] == controller]
        print(f"{controller:16s} mean_total_cost={np.mean(costs):.3f}")


if __name__ == "__main__":
    main()
