"""Sweep CEM population/iteration budgets for BRPC toy oracle stability.

This is a lightweight utility, not a large experiment runner.  Defaults prioritize the
current-dynamics oracle on Toy1/Toy2 and write per-step/summary CSVs with actions,
predicted/realized rewards when available, and CEM query budgets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_brpc_baseline_report import BASELINES, RunnerConfig, run_one


def parse_int_list(value: str) -> list[int]:
    vals = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (
            row["environment"],
            row["baseline"],
            row["cem_population"],
            row["cem_iterations"],
            row["seed"],
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (environment, baseline, population, iterations, seed), rows in sorted(groups.items()):
        actions = np.asarray([float(r["action"]) for r in rows], dtype=float)
        rewards = np.asarray([float(r["realized_reward"]) for r in rows], dtype=float)
        predicted = np.asarray([float(r["predicted_reward"]) for r in rows], dtype=float)
        queries = np.asarray([float(r["planner_queries_step"]) for r in rows], dtype=float)
        out.append(
            {
                "environment": environment,
                "baseline": baseline,
                "cem_population": population,
                "cem_iterations": iterations,
                "seed": seed,
                "steps": len(rows),
                "total_realized_reward": float(np.sum(rewards)),
                "mean_realized_reward": float(np.mean(rewards)),
                "mean_predicted_reward": float(np.mean(predicted)) if np.all(np.isfinite(predicted)) else float("nan"),
                "mean_action": float(np.mean(actions)),
                "std_action_over_time": float(np.std(actions)),
                "planner_queries_total": int(np.sum(queries)),
                "mean_planner_queries_step": float(np.mean(queries)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep CEM population/iterations for BRPC toy oracle convergence.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/brpc_cem_convergence"))
    parser.add_argument("--environment", choices=("Toy1", "Toy2", "both"), default="both")
    parser.add_argument("--baselines", nargs="+", default=["oracle_current"], help="Default prioritizes oracle stability. Use oracle_future for appendix ceiling.")
    parser.add_argument("--include-future-oracle", action="store_true", help="Also sweep toy-only future-regime appendix/ceiling oracle.")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--populations", type=parse_int_list, default=parse_int_list("8,16"))
    parser.add_argument("--iterations", type=parse_int_list, default=parse_int_list("1,2"))
    parser.add_argument("--cold-start-transitions", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=16)
    parser.add_argument("--inducing-points", type=int, default=8)
    args = parser.parse_args()

    valid_baselines = {name for name, _, _ in BASELINES}
    baselines = list(args.baselines)
    if args.include_future_oracle and "oracle_future" not in baselines:
        baselines.append("oracle_future")
    unknown = sorted(set(baselines) - valid_baselines)
    if unknown:
        raise ValueError(f"Unknown baselines {unknown}; expected subset of {sorted(valid_baselines)}")

    environments = ["Toy1", "Toy2"] if args.environment == "both" else [args.environment]
    baseline_meta = {name: (calibration, planner) for name, calibration, planner in BASELINES}
    raw_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []

    for population in args.populations:
        for iterations in args.iterations:
            cfg = RunnerConfig(
                horizon=args.horizon,
                seeds=tuple(args.seeds),
                cold_start_transitions=args.cold_start_transitions,
                num_particles=args.num_particles,
                inducing_points=args.inducing_points,
                cem_population=population,
                cem_iterations=iterations,
            )
            for environment in environments:
                for seed in args.seeds:
                    for baseline in baselines:
                        calibration, planner = baseline_meta[baseline]
                        rows, seed_summary = run_one(environment, baseline, calibration, planner, cfg, seed)
                        for row in rows:
                            row.update(
                                {
                                    "sweep": "cem_population_iterations",
                                    "cem_population": population,
                                    "cem_iterations": iterations,
                                    "selected_action": row["action"],
                                    "query_budget": row["planner_queries_step"],
                                }
                            )
                        seed_summary.update(
                            {
                                "sweep": "cem_population_iterations",
                                "cem_population": population,
                                "cem_iterations": iterations,
                            }
                        )
                        raw_rows.extend(rows)
                        seed_summary_rows.append(seed_summary)

    summary_rows = summarize(raw_rows)
    out_dir = args.out_dir.resolve()
    write_csv(out_dir / "brpc_cem_convergence_raw.csv", raw_rows)
    write_csv(out_dir / "brpc_cem_convergence_seed_summary.csv", seed_summary_rows)
    write_csv(out_dir / "brpc_cem_convergence_summary.csv", summary_rows)
    print(f"wrote BRPC CEM convergence CSVs to {out_dir}")


if __name__ == "__main__":
    main()
