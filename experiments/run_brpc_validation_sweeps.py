"""Validation-scale sweeps for the BRPC toy benchmark suite.

This runner is deliberately narrow: it varies one mechanism at a time rather than
running a full Cartesian product.  It reuses the BRPC baseline runner so schema and
reward accounting stay aligned with the smoke/pilot reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_brpc_baseline_report import BASELINES, RunnerConfig, run_one, summarize


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


def toy2_changepoint_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    toy2 = [row for row in raw_rows if row["environment"] == "Toy2"]
    if not toy2:
        return []
    out = []
    baselines = sorted({row["baseline"] for row in toy2})
    for baseline in baselines:
        rows = [row for row in toy2 if row["baseline"] == baseline]
        post = [row for row in rows if float(row["true_theta"]) > 0.5]
        if not post:
            continue
        actions = np.asarray([float(row["action"]) for row in post], dtype=float)
        rewards = np.asarray([float(row["net_reward"]) for row in post], dtype=float)
        change_times = []
        for seed in sorted({row["seed"] for row in rows}, key=int):
            seed_rows = sorted([row for row in rows if row["seed"] == seed], key=lambda r: int(r["t"]))
            cps = [int(row["t"]) for row in seed_rows if float(row["true_theta"]) > 0.5]
            if cps:
                change_times.append(min(cps))
        out.append(
            {
                "baseline": baseline,
                "mean_detected_changepoint_time": float(np.mean(change_times)) if change_times else "",
                "post_mean_action": float(np.mean(actions)),
                "post_near_old_rate_abs_le_0p1": float(np.mean(np.abs(actions - 0.2) <= 0.1)),
                "post_near_diag_rate_abs_le_0p1": float(np.mean(np.abs(actions - 0.5) <= 0.1)),
                "post_near_new_rate_abs_le_0p1": float(np.mean(np.abs(actions - 0.8) <= 0.1)),
                "post_mean_distance_to_new": float(np.mean(np.abs(actions - 0.8))),
                "post_mean_net_reward_step": float(np.mean(rewards)),
            }
        )
    return out


def run_suite(label: str, cfg: RunnerConfig, out_dir: Path, baselines: tuple[str, ...]) -> dict[str, int]:
    baseline_meta = {name: (calibration, planner) for name, calibration, planner in BASELINES}
    unknown = sorted(set(baselines) - set(baseline_meta))
    if unknown:
        raise ValueError(f"Unknown baselines {unknown}; expected subset of {sorted(baseline_meta)}")

    raw_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for environment in ("Toy1", "Toy2"):
        for seed in cfg.seeds:
            for baseline in baselines:
                calibration, planner = baseline_meta[baseline]
                rows, seed_summary = run_one(environment, baseline, calibration, planner, cfg, seed)
                for row in rows:
                    row["sweep_label"] = label
                seed_summary["sweep_label"] = label
                raw_rows.extend(rows)
                seed_rows.append(seed_summary)

    summary_rows = summarize(seed_rows)
    for row in summary_rows:
        row["sweep_label"] = label
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "brpc_validation_raw.csv", raw_rows)
    write_csv(out_dir / "brpc_validation_seed_summary.csv", seed_rows)
    write_csv(out_dir / "brpc_validation_summary.csv", summary_rows)
    cp_rows = toy2_changepoint_summary(raw_rows)
    if cp_rows:
        write_csv(out_dir / "toy2_changepoint_summary.csv", cp_rows)
    (out_dir / "config.json").write_text(json.dumps({"label": label, "config": asdict(cfg), "baselines": baselines}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"raw_rows": len(raw_rows), "seed_summary_rows": len(seed_rows), "summary_rows": len(summary_rows)}


def run_horizon_sweep(base_cfg: RunnerConfig, out_dir: Path, horizons: list[int], baselines: tuple[str, ...]) -> None:
    aggregate = []
    for planning_horizon in horizons:
        label = f"M{planning_horizon}"
        cfg = replace(base_cfg, cem_horizon=planning_horizon)
        subdir = out_dir / label
        counts = run_suite(label, cfg, subdir, baselines)
        with (subdir / "brpc_validation_summary.csv").open(newline="") as f:
            aggregate.extend(csv.DictReader(f))
        print(f"{label}: {counts}")
    write_csv(out_dir / "horizon_sweep_summary.csv", aggregate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-at-a-time BRPC benchmark-validation sweeps.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/brpc_validation_sweeps"))
    parser.add_argument("--sweep", choices=("horizon",), default="horizon")
    parser.add_argument("--horizon", type=int, default=60, help="Deployment horizon T.")
    parser.add_argument("--planning-horizons", type=parse_int_list, default=parse_int_list("1,2,3,5,10,20"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cold-start-transitions", type=int, default=20)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--inducing-points", type=int, default=24)
    parser.add_argument("--cem-population", type=int, default=128)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--toy2-optimizer", choices=("cem", "grid_dp"), default="grid_dp")
    parser.add_argument("--toy2-grid-size", type=int, default=201)
    parser.add_argument("--bocpd-hazard", type=float, default=0.02)
    parser.add_argument("--bocpd-max-experts", type=int, default=12)
    parser.add_argument("--bocpd-min-segment-length", type=int, default=5)
    parser.add_argument("--baselines", nargs="+", default=["ce_brpc", "ps_brpc", "ce_bbrpc", "ps_bbrpc", "oracle_current", "oracle_future"])
    args = parser.parse_args()

    cfg = RunnerConfig(
        horizon=args.horizon,
        seeds=tuple(args.seeds),
        cold_start_transitions=args.cold_start_transitions,
        num_particles=args.num_particles,
        inducing_points=args.inducing_points,
        cem_horizon=1,
        cem_population=args.cem_population,
        cem_iterations=args.cem_iterations,
        toy2_optimizer=args.toy2_optimizer,
        toy2_grid_size=args.toy2_grid_size,
        bocpd_hazard=args.bocpd_hazard,
        bocpd_max_experts=args.bocpd_max_experts,
        bocpd_min_segment_length=args.bocpd_min_segment_length,
        toy2_changepoint_low_fraction=0.35,
        toy2_changepoint_high_fraction=0.65,
        output_note="benchmark validation sweep; one mechanism varied at a time",
    )

    if args.sweep == "horizon":
        run_horizon_sweep(cfg, args.out_dir.resolve(), args.planning_horizons, tuple(args.baselines))
    print(f"wrote validation sweep outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
