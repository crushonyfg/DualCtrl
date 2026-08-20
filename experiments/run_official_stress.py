"""Run separated official stress-test panels."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_panel(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_summary(path: Path, stress: str, level: str) -> list[dict]:
    rows = []
    for r in read_csv(path):
        rows.append(
            {
                "stress": stress,
                "level": level,
                "environment": r["environment"],
                "twin_gap": r["twin_gap"],
                "regime": r["regime"],
                "baseline": r["baseline"],
                "mean_total_cost": r["mean_total_cost"],
                "stderr_total_cost": r["stderr_total_cost"],
                "mean_oracle_regret": r["mean_oracle_regret"],
                "n": r["n"],
            }
        )
    return rows


def run_cmd(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/official_stress"))
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--scalar-action-grid-size", type=int, default=11)
    parser.add_argument("--cartpole-action-grid-size", type=int, default=5)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--optimizer-grid-size", type=int, default=81)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--optimizer-xatol", type=float, default=1e-4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sparse_rows = []
    for interval in (1, 5, 10, 20):
        sub = args.out_dir / f"sparse_m{interval}"
        run_cmd([
            "python", "-m", "experiments.run_official_scalar",
            "--horizon", str(args.horizon), "--n-seeds", str(args.n_seeds),
            "--planning-horizon", str(args.planning_horizon), "--action-grid-size", str(args.scalar_action_grid_size),
            *( ["--continuous-actions"] if args.continuous_actions else [] ),
            "--optimizer-grid-size", str(args.optimizer_grid_size), "--optimizer-maxiter", str(args.optimizer_maxiter),
            "--optimizer-xatol", str(args.optimizer_xatol),
            "--observation-interval", str(interval), "--out-dir", str(sub),
        ])
        sparse_rows.extend(collect_summary(sub / "scalar_main_summary.csv", "sparse_physical", f"m={interval}"))
    write_panel(args.out_dir / "sparse_physical_panel.csv", sparse_rows)

    nondiff_rows = []
    for k in (0.0, 0.05, 0.1, 0.2):
        sub = args.out_dir / f"nondiff_k{k}"
        run_cmd([
            "python", "-m", "experiments.run_official_scalar",
            "--horizon", str(args.horizon), "--n-seeds", str(args.n_seeds),
            "--planning-horizon", str(args.planning_horizon), "--action-grid-size", str(args.scalar_action_grid_size),
            "--nonsmooth-switch-cost", str(k), "--nonsmooth-switch-threshold", "0.05",
            *( ["--continuous-actions"] if args.continuous_actions else [] ),
            "--optimizer-grid-size", str(args.optimizer_grid_size), "--optimizer-maxiter", str(args.optimizer_maxiter),
            "--optimizer-xatol", str(args.optimizer_xatol),
            "--out-dir", str(sub),
        ])
        nondiff_rows.extend(collect_summary(sub / "scalar_main_summary.csv", "nondiff_switch", f"k={k}"))
    write_panel(args.out_dir / "nondiff_switch_panel.csv", nondiff_rows)

    multimodal_note = args.out_dir / "multimodal_notes.md"
    multimodal_note.write_text(
        "# Multimodal diagnostic notes\n\n"
        "The official three literature baselines are run with their native belief assumptions. "
        "A separate multimodal panel should not silently replace those assumptions with an invented mixture controller. "
        "For Arcari, multimodality can be represented as structural/model modes only when the same nominal digital-twin model set is available to all methods. "
        "For KH and TV-GP-LCB, multimodality is an assumption-stress test rather than a like-for-like method extension.\n"
    )
    print(f"wrote stress panels to {args.out_dir}")


if __name__ == "__main__":
    main()
