"""Run the official scalar and CartPole benchmark suite and write a report."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def markdown_table(rows: list[dict[str, str]], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return ""
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row[col]
            try:
                f = float(val)
                val = f"{f:.4g}"
            except ValueError:
                pass
            vals.append(val)
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/official"))
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--scalar-action-grid-size", type=int, default=15)
    parser.add_argument("--cartpole-action-grid-size", type=int, default=5)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--optimizer-grid-size", type=int, default=81)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--optimizer-xatol", type=float, default=1e-4)
    parser.add_argument("--smpc-dual-horizon", type=int, default=2)
    parser.add_argument("--smpc-scenarios", type=int, default=3)
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--include-diagnostics", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--horizon", str(args.horizon),
        "--n-seeds", str(args.n_seeds),
        "--planning-horizon", str(args.planning_horizon),
        "--smpc-dual-horizon", str(args.smpc_dual_horizon),
        "--optimizer-grid-size", str(args.optimizer_grid_size),
        "--optimizer-maxiter", str(args.optimizer_maxiter),
        "--optimizer-xatol", str(args.optimizer_xatol),
        "--smpc-scenarios", str(args.smpc_scenarios),
        "--out-dir", str(args.out_dir),
    ]
    diagnostic_args = ["--include-diagnostics"] if args.include_diagnostics else []
    continuous_args = ["--continuous-actions"] if args.continuous_actions else []
    subprocess.run(["python", "-m", "experiments.reproduce_kh_section6", "--out", str(args.out_dir / "kh_section6_curve.csv")], check=True)
    subprocess.run(["python", "-m", "experiments.run_official_scalar", "--action-grid-size", str(args.scalar_action_grid_size), *diagnostic_args, *continuous_args, *common], check=True)
    subprocess.run(["python", "-m", "experiments.run_official_cartpole", "--action-grid-size", str(args.cartpole_action_grid_size), *diagnostic_args, *continuous_args, *common], check=True)

    if args.include_stress:
        subprocess.run([
            "python", "-m", "experiments.run_official_stress",
            "--out-dir", str(args.out_dir / "stress"),
            "--horizon", str(args.horizon),
            "--n-seeds", str(args.n_seeds),
            "--planning-horizon", str(args.planning_horizon),
            "--scalar-action-grid-size", str(args.scalar_action_grid_size),
            "--cartpole-action-grid-size", str(args.cartpole_action_grid_size),
            *continuous_args,
            "--optimizer-grid-size", str(args.optimizer_grid_size),
            "--optimizer-maxiter", str(args.optimizer_maxiter),
            "--optimizer-xatol", str(args.optimizer_xatol),
        ], check=True)

    scalar = read_csv(args.out_dir / "scalar_main_summary.csv")
    cartpole = read_csv(args.out_dir / "cartpole_main_summary.csv")
    report = [
        "# Official Baseline Benchmark Report",
        "",
        "Baselines included: KH dual control, Arcari dual stochastic MPC, TV-GP-LCB, oracle trend planner.",
        "All non-oracle baselines receive the same nominal digital twin and physical observations. In gap settings, only the physical environment has the gap; baselines are not given the true gap function.",
        "",
        "## Scalar main matrix",
        markdown_table(scalar),
        "",
        "## CartPole main matrix",
        markdown_table(cartpole),
        "",
        "## Notes",
        "- Arcari uses an explicit dual scenario tree with configurable dual horizon L and parameter/noise scenario branching where implemented.",
        "- TV-GP-LCB is the Bogunovic time-varying GP-UCB acquisition adapted to cost minimization via LCB.",
        "- KH reproduction curve is written to kh_section6_curve.csv.",
    ]
    if args.include_diagnostics:
        report.extend([
            "- Diagnostics, excluded from the main tables, are written to scalar_diagnostic_summary.csv and cartpole_diagnostic_summary.csv.",
        ])
    if args.include_stress:
        report.extend([
            "",
            "## Stress panels",
            "- Sparse physical data panel: stress/sparse_physical_panel.csv",
            "- Non-differentiable switch-cost panel: stress/nondiff_switch_panel.csv",
            "- Multimodal diagnostic notes: stress/multimodal_notes.md",
        ])
    report_path = args.out_dir / "official_report.md"
    report_path.write_text("\n".join(report))
    print(f"wrote report to {report_path}")


if __name__ == "__main__":
    main()
