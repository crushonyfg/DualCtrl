"""Run convergence diagnostics for grid size, scenario count, and horizon.

Outputs are intentionally separate from the official main baseline summaries.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepJob:
    environment: str
    sweep: str
    value: int
    action_grid_size: int
    smpc_scenarios: int
    planning_horizon: int
    out_dir: Path


def parse_int_list(value: str) -> list[int]:
    vals = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_sweep_jobs(args: argparse.Namespace) -> list[SweepJob]:
    environments = ["scalar", "cartpole"] if args.environment == "both" else [args.environment]
    jobs: list[SweepJob] = []
    for environment in environments:
        fixed_grid = args.scalar_fixed_action_grid_size if environment == "scalar" else args.cartpole_fixed_action_grid_size
        grid_values = args.scalar_action_grid_sizes if environment == "scalar" else args.cartpole_action_grid_sizes
        for value in grid_values:
            jobs.append(
                SweepJob(
                    environment=environment,
                    sweep="action_grid_size",
                    value=value,
                    action_grid_size=value,
                    smpc_scenarios=args.fixed_smpc_scenarios,
                    planning_horizon=args.fixed_planning_horizon,
                    out_dir=args.out_dir / environment / "action_grid_size" / str(value),
                )
            )
        for value in args.smpc_scenarios_list:
            jobs.append(
                SweepJob(
                    environment=environment,
                    sweep="smpc_scenarios",
                    value=value,
                    action_grid_size=fixed_grid,
                    smpc_scenarios=value,
                    planning_horizon=args.fixed_planning_horizon,
                    out_dir=args.out_dir / environment / "smpc_scenarios" / str(value),
                )
            )
        for value in args.planning_horizons:
            jobs.append(
                SweepJob(
                    environment=environment,
                    sweep="planning_horizon",
                    value=value,
                    action_grid_size=fixed_grid,
                    smpc_scenarios=args.fixed_smpc_scenarios,
                    planning_horizon=value,
                    out_dir=args.out_dir / environment / "planning_horizon" / str(value),
                )
            )
    return jobs


def run_job(job: SweepJob, args: argparse.Namespace) -> list[dict[str, str]]:
    module = f"experiments.run_official_{job.environment}"
    cmd = [
        "python",
        "-m",
        module,
        "--out-dir",
        str(job.out_dir),
        "--horizon",
        str(args.horizon),
        "--n-seeds",
        str(args.n_seeds),
        "--seed",
        str(args.seed),
        "--planning-horizon",
        str(job.planning_horizon),
        "--action-grid-size",
        str(job.action_grid_size),
        "--smpc-scenarios",
        str(job.smpc_scenarios),
        "--smpc-dual-horizon",
        str(args.smpc_dual_horizon),
        "--observation-interval",
        str(args.observation_interval),
    ]
    if args.include_diagnostics:
        cmd.append("--include-diagnostics")
    subprocess.run(cmd, check=True)

    summary_path = job.out_dir / f"{job.environment}_main_summary.csv"
    rows = []
    for row in read_csv(summary_path):
        rows.append(
            {
                "sweep": job.sweep,
                "sweep_value": job.value,
                "action_grid_size": job.action_grid_size,
                "smpc_scenarios": job.smpc_scenarios,
                "planning_horizon": job.planning_horizon,
                **row,
            }
        )
    if args.include_diagnostics:
        diagnostic_path = job.out_dir / f"{job.environment}_diagnostic_summary.csv"
        if diagnostic_path.exists():
            for row in read_csv(diagnostic_path):
                rows.append(
                    {
                        "sweep": job.sweep,
                        "sweep_value": job.value,
                        "action_grid_size": job.action_grid_size,
                        "smpc_scenarios": job.smpc_scenarios,
                        "planning_horizon": job.planning_horizon,
                        **row,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/convergence"))
    parser.add_argument("--environment", choices=("scalar", "cartpole", "both"), default="both")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--scalar-action-grid-sizes", type=parse_int_list, default=parse_int_list("7,11,15"))
    parser.add_argument("--cartpole-action-grid-sizes", type=parse_int_list, default=parse_int_list("3,5,7"))
    parser.add_argument("--smpc-scenarios-list", type=parse_int_list, default=parse_int_list("1,3,5"))
    parser.add_argument("--planning-horizons", type=parse_int_list, default=parse_int_list("2,3,4"))
    parser.add_argument("--scalar-fixed-action-grid-size", type=int, default=11)
    parser.add_argument("--cartpole-fixed-action-grid-size", type=int, default=5)
    parser.add_argument("--fixed-smpc-scenarios", type=int, default=3)
    parser.add_argument("--fixed-planning-horizon", type=int, default=3)
    parser.add_argument("--smpc-dual-horizon", type=int, default=2)
    parser.add_argument("--include-diagnostics", action="store_true")
    args = parser.parse_args()

    all_rows = []
    for job in build_sweep_jobs(args):
        all_rows.extend(run_job(job, args))
    write_csv(args.out_dir / "convergence_sweep_summary.csv", all_rows)
    print(f"wrote convergence sweep summary to {args.out_dir / 'convergence_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
