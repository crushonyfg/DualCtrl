"""Create baseline-only reward/cost summaries from official raw outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASELINES = {"kh_dual_control", "arcari_dual_smpc", "tv_gp_lcb"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def enrich_scalar(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for r in rows:
        if r["baseline"] not in BASELINES:
            continue
        state = float(r["state_cost"])
        energy = float(r["energy_cost"])
        switch = float(r["switch_cost"])
        nonsmooth = float(r["nonsmooth_switch_cost"])
        terminal = float(r["terminal_cost"])
        total = float(r["total_cost"])
        out.append(
            {
                **{k: r[k] for k in ["environment", "twin_gap", "regime", "seed", "baseline"]},
                "acc_task_reward": -state,
                "acc_energy_cost": energy,
                "acc_switch_cost": switch + nonsmooth,
                "acc_failure_cost": 0.0,
                "terminal_cost": terminal,
                "total_cost": total,
                "net_reward": -total,
                "physical_transitions": int(float(r["physical_transitions"])),
                "observed_transitions": int(float(r["observed_transitions"])),
                "observation_interval": int(float(r["observation_interval"])),
            }
        )
    return out


def enrich_cartpole(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for r in rows:
        if r["baseline"] not in BASELINES:
            continue
        task = float(r["task_cost"])
        energy = float(r["energy_cost"])
        switch = float(r["switch_cost"])
        failure = float(r["failure_cost"])
        terminal = float(r["terminal_cost"])
        total = float(r["total_cost"])
        out.append(
            {
                **{k: r[k] for k in ["environment", "twin_gap", "regime", "seed", "baseline"]},
                "acc_task_reward": -task,
                "acc_energy_cost": energy,
                "acc_switch_cost": switch,
                "acc_failure_cost": failure,
                "terminal_cost": terminal,
                "total_cost": total,
                "net_reward": -total,
                "failures": float(r["failures"]),
                "physical_transitions": int(float(r["physical_transitions"])),
                "observed_transitions": int(float(r["observed_transitions"])),
                "observation_interval": int(float(r["observation_interval"])),
            }
        )
    return out


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["environment"], r["twin_gap"], r["regime"], r["baseline"])].append(r)
    summary = []
    metrics = ["net_reward", "total_cost", "acc_task_reward", "acc_energy_cost", "acc_switch_cost", "acc_failure_cost", "terminal_cost"]
    for key in sorted(groups):
        sub = groups[key]
        row = {
            "environment": key[0],
            "twin_gap": key[1],
            "regime": key[2],
            "baseline": key[3],
            "n": len(sub),
        }
        for m in metrics:
            vals = np.array([float(r[m]) for r in sub], dtype=float)
            row[f"mean_{m}"] = float(vals.mean())
            row[f"stderr_{m}"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        if "failures" in sub[0]:
            vals = np.array([float(r["failures"]) for r in sub], dtype=float)
            row["mean_failures"] = float(vals.mean())
        summary.append(row)
    return summary


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=str, default="{}")
    args = parser.parse_args()

    scalar_raw = enrich_scalar(read_csv(args.official_dir / "scalar_main_raw.csv"))
    cartpole_raw = enrich_cartpole(read_csv(args.official_dir / "cartpole_main_raw.csv"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "scalar_baseline_raw.csv", scalar_raw)
    write_csv(args.out_dir / "cartpole_baseline_raw.csv", cartpole_raw)
    scalar_summary = summarize(scalar_raw)
    cartpole_summary = summarize(cartpole_raw)
    write_csv(args.out_dir / "scalar_baseline_summary.csv", scalar_summary)
    write_csv(args.out_dir / "cartpole_baseline_summary.csv", cartpole_summary)
    (args.out_dir / "config.json").write_text(json.dumps(json.loads(args.config), indent=2, ensure_ascii=False))

    report = [
        "# Baseline-only reward/cost report",
        "",
        "本报告只包含三个文献 baseline：`kh_dual_control`、`arcari_dual_smpc`、`tv_gp_lcb`。不包含 oracle，也不报告 oracle regret。",
        "",
        "指标定义：`net_reward = - total_cost`；`acc_task_reward = - task/state cost`；action cost 分为 energy 与 switch。",
        "",
        "## Scalar summary",
        markdown_table(scalar_summary),
        "",
        "## CartPole summary",
        markdown_table(cartpole_summary),
    ]
    (args.out_dir / "baseline_report_zh.md").write_text("\n".join(report))
    print(f"wrote baseline-only outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
