"""Plot scalar benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_curve(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    rows = read_rows(args.input)
    u0 = np.array([float(r["u0"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [
        ("ce_cost", "CE"),
        ("cautious_cost", "cautious"),
        ("bayes_dual_cost", "Bayes dual"),
        ("approx_dual_cost", "approx dual"),
        ("grid_bayes_cost", "grid Bayes"),
    ]:
        if key in rows[0]:
            ax.plot(u0, [float(r[key]) for r in rows], label=label)
    ax.set_xlabel("root action u0")
    ax.set_ylabel("two-step expected cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


def summarize(args: argparse.Namespace) -> None:
    rows = read_rows(args.input)
    by_controller: dict[str, list[float]] = defaultdict(list)
    by_gap: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        controller = row["controller"]
        by_controller[controller].append(float(row["total_cost"]))
        if "clairvoyant_gap" in row and row["clairvoyant_gap"]:
            by_gap[controller].append(float(row["clairvoyant_gap"]))
    print("controller,total_cost_mean,total_cost_std,clairvoyant_gap_mean")
    for controller in sorted(by_controller):
        costs = np.array(by_controller[controller])
        gaps = np.array(by_gap.get(controller, [np.nan]))
        print(f"{controller},{costs.mean():.6g},{costs.std(ddof=1) if len(costs)>1 else 0:.6g},{np.nanmean(gaps):.6g}")


def plot_bars(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    rows = read_rows(args.input)
    by_controller: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_controller[row["controller"]].append(float(row[args.metric]))
    labels = sorted(by_controller)
    means = [np.mean(by_controller[label]) for label in labels]
    errs = [np.std(by_controller[label], ddof=1) / np.sqrt(len(by_controller[label])) if len(by_controller[label]) > 1 else 0.0 for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4))
    ax.bar(labels, means, yerr=errs)
    ax.set_ylabel(args.metric)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    curve = sub.add_parser("curve")
    curve.add_argument("--input", type=Path, required=True)
    curve.add_argument("--output", type=Path, default=Path("reports/figures/scalar_curve.png"))
    curve.set_defaults(func=plot_curve)

    summary = sub.add_parser("summary")
    summary.add_argument("--input", type=Path, required=True)
    summary.set_defaults(func=summarize)

    bars = sub.add_parser("bars")
    bars.add_argument("--input", type=Path, required=True)
    bars.add_argument("--metric", default="total_cost")
    bars.add_argument("--output", type=Path, default=Path("reports/figures/scalar_bars.png"))
    bars.set_defaults(func=plot_bars)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
