"""Run first-pass CartPole evolving-twin benchmark skeleton."""

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
from controllers.cartpole import CartPoleCEController, CartPoleClairvoyantController, CartPolePlannerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["static", "ou", "fixed_jumps", "random_jumps", "multimodal"], default="static")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--rw-process-std", type=float, default=0.005)
    parser.add_argument("--out", type=Path, default=Path("reports/tables/cartpole_results.csv"))
    args = parser.parse_args()

    rows = []
    cost = CartPoleCost(CartPoleCostConfig())
    dynamics = CartPoleParams()
    planner = CartPolePlannerConfig()
    regime = CartPoleRegimeConfig(kind=args.regime, horizon=args.horizon)
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        theta_path = generate_theta_path(regime, rng)
        process_std = np.array(CartPoleEnvConfig().process_std)
        process_noise = rng.normal(0.0, process_std, size=(args.horizon, 4))
        controllers = [
            CartPoleCEController("ce_static", GaussianBelief(mean=1.0, var=0.1, process_var=0.0), dynamics, cost, planner),
            CartPoleCEController("ce_rw", GaussianBelief(mean=1.0, var=0.1, process_var=args.rw_process_std**2), dynamics, cost, planner),
            CartPoleClairvoyantController(theta_path, dynamics, cost, planner),
        ]
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
                    "seed": seed,
                    "controller": controller.name,
                    "regime": args.regime,
                    "total_cost": traj.total_cost,
                    "net_reward": -traj.total_cost,
                    "terminal_cost": traj.terminal_cost,
                    "physical_transitions": traj.physical_transitions,
                    "observed_transitions": traj.observed_transitions,
                    "observation_interval": args.observation_interval,
                    "violation_steps": traj.failures,
                    "failures": traj.failures,
                    "failure_events": traj.failure_events,
                    "task_cost": task_cost,
                    "acc_task_reward": -task_cost,
                    "energy_cost": energy_cost,
                    "acc_energy_cost": energy_cost,
                    "switch_cost": switch_cost,
                    "nonsmooth_switch_cost": nonsmooth_switch_cost,
                    "acc_switch_cost": switch_cost + nonsmooth_switch_cost,
                    "failure_cost": failure_cost,
                    "acc_failure_cost": failure_cost,
                    "mean_abs_action": traj.mean_abs_action,
                    "frac_zero_action": traj.frac_zero_action,
                    "action_changes": traj.action_changes,
                    "final_belief_mean": traj.belief_means[-1],
                    "final_belief_var": traj.belief_vars[-1],
                }
            )
        for row in rows[-len(controllers):]:
            row["clairvoyant_gap"] = float(row["total_cost"] - seed_costs["clairvoyant"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    for controller in sorted({row["controller"] for row in rows}):
        costs = [float(row["total_cost"]) for row in rows if row["controller"] == controller]
        print(f"{controller:16s} mean_total_cost={np.mean(costs):.3f}")


if __name__ == "__main__":
    main()
