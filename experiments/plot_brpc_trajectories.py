"""Plot trajectory-level diagnostics for BRPC toy validation runs.

The plots are mechanism diagnostics rather than score tables.  Toy2 receives the most
attention: a time-action heatmap shows the true operating reward landscape, with
stagewise greedy optimum, full-horizon switching-cost oracle, and each method's
actual closed-loop actions overlaid.
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from brpc_baselines.toy_envs import Toy1Config, Toy2Config, Toy2DigitalTwin


DEFAULT_METHODS = ("ce_brpc", "ce_bbrpc", "ps_brpc", "ps_bbrpc", "oracle_current", "oracle_future")
METHOD_STYLE = {
    "ce_brpc": {"color": "tab:blue", "linestyle": "-", "label": "CE-BRPC"},
    "ce_bbrpc": {"color": "tab:cyan", "linestyle": "--", "label": "CE-BBRPC"},
    "ps_brpc": {"color": "tab:orange", "linestyle": "-", "label": "PS-BRPC"},
    "ps_bbrpc": {"color": "tab:red", "linestyle": "--", "label": "PS-BBRPC"},
    "oracle_current": {"color": "black", "linestyle": "-", "label": "current oracle"},
    "oracle_future": {"color": "white", "linestyle": ":", "label": "future oracle"},
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _group(rows: list[dict[str, str]], environment: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("environment") != environment:
            continue
        groups[(row["seed"], row["baseline"])].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(r["t"]))
    return groups


def _total_return(rows: list[dict[str, str]]) -> float:
    if not rows:
        return float("nan")
    return float(rows[-1].get("cumulative_net_reward", "nan"))


def select_median_seed(rows: list[dict[str, str]], environment: str, reference_baseline: str) -> str:
    groups = _group(rows, environment)
    seed_returns = []
    for (seed, baseline), trajectory in groups.items():
        if baseline == reference_baseline:
            seed_returns.append((seed, _total_return(trajectory)))
    if not seed_returns:
        seeds = sorted({row["seed"] for row in rows if row.get("environment") == environment}, key=int)
        if not seeds:
            raise ValueError(f"No rows for environment={environment!r}")
        return seeds[len(seeds) // 2]
    seed_returns.sort(key=lambda item: item[1])
    return seed_returns[len(seed_returns) // 2][0]


def toy2_operating_landscape(theta_path: np.ndarray, action_grid: np.ndarray, cfg: Toy2Config) -> np.ndarray:
    twin = Toy2DigitalTwin(cfg)
    out = np.empty((len(action_grid), len(theta_path)), dtype=float)
    for t, theta in enumerate(theta_path):
        response = twin.response(action_grid, float(theta))
        discrepancy = cfg.discrepancy_sine_amplitude * np.sin(4.0 * np.pi * action_grid)
        out[:, t] = response + discrepancy - cfg.lambda_energy * action_grid * action_grid
    return out


def toy2_greedy_actions(theta_path: np.ndarray, action_grid: np.ndarray, cfg: Toy2Config) -> np.ndarray:
    reward = toy2_operating_landscape(theta_path, action_grid, cfg)
    return action_grid[np.argmax(reward, axis=0)]


def toy2_full_horizon_oracle(theta_path: np.ndarray, action_grid: np.ndarray, cfg: Toy2Config, initial_previous_action: float | None = None) -> np.ndarray:
    """Full-deployment Toy2 oracle trajectory including switching cost.

    This is not a stagewise greedy line.  It solves the discretized full-horizon
    deterministic problem with known theta path and zero observation noise.
    """

    initial_previous_action = cfg.a_left if initial_previous_action is None else float(initial_previous_action)
    operating = toy2_operating_landscape(theta_path, action_grid, cfg).T  # (T, A)
    A = len(action_grid)
    T = len(theta_path)
    switch = cfg.lambda_switch * (action_grid[:, None] - action_grid[None, :]) ** 2
    values = [np.zeros(A, dtype=float) for _ in range(T + 1)]
    policies = [np.zeros(A, dtype=int) for _ in range(T)]
    for t in range(T - 1, -1, -1):
        q = operating[t][None, :] - switch + values[t + 1][None, :]
        policies[t] = np.argmax(q, axis=1)
        values[t] = np.max(q, axis=1)
    initial_q = operating[0] - cfg.lambda_switch * (action_grid - initial_previous_action) ** 2 + values[1]
    action_idx = int(np.argmax(initial_q))
    traj = []
    for t in range(T):
        traj.append(action_grid[action_idx])
        if t + 1 < T:
            action_idx = int(policies[t + 1][action_idx])
    return np.asarray(traj, dtype=float)


def _series_for_seed(groups: dict[tuple[str, str], list[dict[str, str]]], seed: str, baseline: str, key: str) -> np.ndarray | None:
    rows = groups.get((seed, baseline))
    if not rows:
        return None
    return np.asarray([_float(row, key) for row in rows], dtype=float)


def _change_times(theta_path: np.ndarray) -> list[int]:
    if len(theta_path) <= 1:
        return []
    return [idx for idx in range(1, len(theta_path)) if abs(theta_path[idx] - theta_path[idx - 1]) > 1e-9]


def _bocpd_probability(rows: list[dict[str, str]]) -> np.ndarray | None:
    vals = []
    for row in rows:
        val = _float(row, "recent_change_probability")
        vals.append(val)
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return None
    return arr


def _theta_mean(rows: list[dict[str, str]]) -> np.ndarray | None:
    vals = []
    for row in rows:
        val = _float(row, "theta_mean")
        vals.append(val)
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return None
    return arr


def _expert_new_mass(rows: list[dict[str, str]]) -> np.ndarray | None:
    vals = []
    for row in rows:
        starts_raw = row.get("expert_start_times", "")
        masses_raw = row.get("expert_masses", "")
        if not starts_raw or not masses_raw:
            vals.append(np.nan)
            continue
        try:
            starts = np.asarray(ast.literal_eval(starts_raw), dtype=float)
            masses = np.asarray(ast.literal_eval(masses_raw), dtype=float)
            t = float(row["t"])
            vals.append(float(np.sum(masses[starts > 0.5 * t])))
        except (SyntaxError, ValueError, TypeError):
            vals.append(np.nan)
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return None
    return arr


def plot_toy2_representative(raw_rows: list[dict[str, str]], out_path: Path, methods: tuple[str, ...], reference_seed_method: str, action_grid_size: int) -> None:
    cfg = Toy2Config()
    groups = _group(raw_rows, "Toy2")
    seed = select_median_seed(raw_rows, "Toy2", reference_seed_method)
    reference_rows = groups.get((seed, "oracle_current")) or groups.get((seed, methods[0]))
    if not reference_rows:
        raise ValueError("Could not find representative Toy2 trajectory rows.")
    theta_path = np.asarray([_float(row, "true_theta") for row in reference_rows], dtype=float)
    T = len(theta_path)
    t_grid = np.arange(T)
    action_grid = np.linspace(cfg.action_low, cfg.action_high, action_grid_size)
    reward = toy2_operating_landscape(theta_path, action_grid, cfg)
    greedy = toy2_greedy_actions(theta_path, action_grid, cfg)
    oracle_full = toy2_full_horizon_oracle(theta_path, action_grid, cfg)

    fig, (ax, ax_prob) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [4.0, 1.0]})
    im = ax.imshow(reward, origin="lower", aspect="auto", extent=[-0.5, T - 0.5, cfg.action_low, cfg.action_high], cmap="viridis")
    fig.colorbar(im, ax=ax, label="true operating reward (no switching)")
    ax.plot(t_grid, greedy, color="yellow", linewidth=2.0, label="stagewise greedy $a_t^{greedy}$")
    ax.plot(t_grid, oracle_full, color="white", linewidth=2.4, linestyle="-", label="full-horizon switching oracle")

    for method in methods:
        series = _series_for_seed(groups, seed, method, "action")
        if series is None:
            continue
        style = METHOD_STYLE.get(method, {"label": method})
        ax.plot(t_grid[: len(series)], series, linewidth=1.8, marker="o", markersize=2.4, **style)
        if "bbrpc" in method:
            prob = _bocpd_probability(groups[(seed, method)])
            if prob is not None:
                ax_prob.plot(t_grid[: len(prob)], prob, linewidth=1.8, label=METHOD_STYLE.get(method, {}).get("label", method))

    for cp in _change_times(theta_path):
        ax.axvline(cp - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.9)
        ax_prob.axvline(cp - 0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.9)

    ax.set_ylabel("action")
    ax.set_title(f"Toy2 representative trajectory (median {reference_seed_method} seed={seed})")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax_prob.set_ylabel("recent CP prob")
    ax_prob.set_xlabel("deployment time t")
    ax_prob.set_ylim(-0.02, 1.02)
    if ax_prob.lines:
        ax_prob.legend(loc="upper left", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_toy2_aggregate(raw_rows: list[dict[str, str]], out_path: Path, methods: tuple[str, ...], action_grid_size: int) -> None:
    cfg = Toy2Config()
    groups = _group(raw_rows, "Toy2")
    seeds = sorted({seed for seed, _ in groups}, key=int)
    if not seeds:
        raise ValueError("No Toy2 rows found.")
    reference_baseline = "oracle_current" if any(b == "oracle_current" for _, b in groups) else methods[0]
    theta_paths = []
    for seed in seeds:
        rows = groups.get((seed, reference_baseline))
        if rows:
            theta_paths.append([_float(row, "true_theta") for row in rows])
    theta_arr = np.asarray(theta_paths, dtype=float)
    mean_theta = np.mean(theta_arr, axis=0)
    T = theta_arr.shape[1]
    t_grid = np.arange(T)
    action_grid = np.linspace(cfg.action_low, cfg.action_high, action_grid_size)
    reward = toy2_operating_landscape(mean_theta, action_grid, cfg)
    greedy = toy2_greedy_actions(mean_theta, action_grid, cfg)
    oracle_full = toy2_full_horizon_oracle(mean_theta, action_grid, cfg)

    fig, ax = plt.subplots(figsize=(13, 5.6))
    im = ax.imshow(reward, origin="lower", aspect="auto", extent=[-0.5, T - 0.5, cfg.action_low, cfg.action_high], cmap="viridis")
    fig.colorbar(im, ax=ax, label="mean true operating reward (no switching)")
    ax.plot(t_grid, greedy, color="yellow", linewidth=2.0, label="greedy on mean theta")
    ax.plot(t_grid, oracle_full, color="white", linewidth=2.4, label="full-horizon oracle on mean theta")

    for method in methods:
        traces = []
        for seed in seeds:
            series = _series_for_seed(groups, seed, method, "action")
            if series is not None and len(series) == T:
                traces.append(series)
        if not traces:
            continue
        arr = np.asarray(traces, dtype=float)
        q10, med, q90 = np.quantile(arr, [0.1, 0.5, 0.9], axis=0)
        style = METHOD_STYLE.get(method, {"color": None, "linestyle": "-", "label": method})
        color = style.get("color")
        ax.plot(t_grid, med, color=color, linestyle=style.get("linestyle", "-"), linewidth=2.0, label=f"{style.get('label', method)} median")
        if method not in {"oracle_current", "oracle_future"}:
            ax.fill_between(t_grid, q10, q90, color=color, alpha=0.16, linewidth=0)

    # Mark all realized changepoints lightly because aggregate seeds may differ.
    for theta_path in theta_arr:
        for cp in _change_times(theta_path):
            ax.axvline(cp - 0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.18)

    ax.set_xlabel("deployment time t")
    ax.set_ylabel("action")
    ax.set_title("Toy2 aggregate trajectories (median with 10%-90% bands)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_toy2_belief_diagnostics(raw_rows: list[dict[str, str]], out_path: Path, methods: tuple[str, ...], reference_seed_method: str) -> None:
    groups = _group(raw_rows, "Toy2")
    seed = select_median_seed(raw_rows, "Toy2", reference_seed_method)
    reference_rows = groups.get((seed, "oracle_current")) or groups.get((seed, methods[0]))
    if not reference_rows:
        raise ValueError("Could not find representative Toy2 trajectory rows.")
    T = len(reference_rows)
    t_grid = np.arange(T)
    theta = np.asarray([_float(row, "true_theta") for row in reference_rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.0), sharex=True)
    axes[0].plot(t_grid, theta, color="black", linewidth=2.0, label="true theta")
    for method in methods:
        rows = groups.get((seed, method))
        if not rows:
            continue
        style = METHOD_STYLE.get(method, {"label": method})
        theta_mean = _theta_mean(rows)
        if theta_mean is not None:
            axes[0].plot(t_grid[: len(theta_mean)], theta_mean, linewidth=1.7, **style)
        prob = _bocpd_probability(rows)
        if prob is not None:
            axes[1].plot(t_grid[: len(prob)], prob, linewidth=1.7, **style)
        new_mass = _expert_new_mass(rows)
        if new_mass is not None:
            axes[2].plot(t_grid[: len(new_mass)], new_mass, linewidth=1.7, **style)
    for cp in _change_times(theta):
        for ax in axes:
            ax.axvline(cp - 0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.9)
    axes[0].set_ylabel(r"$\theta$")
    axes[1].set_ylabel("recent CP prob")
    axes[2].set_ylabel("new expert mass")
    axes[2].set_xlabel("deployment time t")
    axes[0].set_title(f"Toy2 belief diagnostics (median {reference_seed_method} seed={seed})")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_toy2_horizon_small_multiples(raw_paths: list[Path], out_path: Path, methods: tuple[str, ...], reference_seed_method: str, action_grid_size: int) -> None:
    if not raw_paths:
        return
    cfg = Toy2Config()
    panels = []
    for raw_path in raw_paths:
        rows = _read_rows(raw_path)
        groups = _group(rows, "Toy2")
        seed = select_median_seed(rows, "Toy2", reference_seed_method)
        reference_rows = groups.get((seed, "oracle_current")) or groups.get((seed, methods[0]))
        if not reference_rows:
            continue
        theta_path = np.asarray([_float(row, "true_theta") for row in reference_rows], dtype=float)
        panels.append((raw_path.parent.name, groups, seed, theta_path))
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), sharex=False, squeeze=False)
    action_grid = np.linspace(cfg.action_low, cfg.action_high, action_grid_size)
    for ax, (label, groups, seed, theta_path) in zip(axes[:, 0], panels):
        T = len(theta_path)
        t_grid = np.arange(T)
        reward = toy2_operating_landscape(theta_path, action_grid, cfg)
        greedy = toy2_greedy_actions(theta_path, action_grid, cfg)
        oracle_full = toy2_full_horizon_oracle(theta_path, action_grid, cfg)
        ax.imshow(reward, origin="lower", aspect="auto", extent=[-0.5, T - 0.5, cfg.action_low, cfg.action_high], cmap="viridis")
        ax.plot(t_grid, greedy, color="yellow", linewidth=1.7, label="greedy")
        ax.plot(t_grid, oracle_full, color="white", linewidth=2.0, label="full oracle")
        for method in methods:
            series = _series_for_seed(groups, seed, method, "action")
            if series is None:
                continue
            style = METHOD_STYLE.get(method, {"label": method})
            ax.plot(t_grid[: len(series)], series, linewidth=1.5, **style)
        for cp in _change_times(theta_path):
            ax.axvline(cp - 0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(f"Toy2 {label} representative seed={seed}")
        ax.set_ylabel("action")
    axes[-1, 0].set_xlabel("deployment time t")
    axes[0, 0].legend(loc="upper left", ncol=4, fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_toy1_representative(raw_rows: list[dict[str, str]], out_path: Path, methods: tuple[str, ...], reference_seed_method: str) -> None:
    cfg = Toy1Config()
    groups = _group(raw_rows, "Toy1")
    seed = select_median_seed(raw_rows, "Toy1", reference_seed_method)
    reference_rows = groups.get((seed, "oracle_current")) or groups.get((seed, methods[0]))
    if not reference_rows:
        raise ValueError("Could not find representative Toy1 trajectory rows.")
    T = len(reference_rows)
    t_grid = np.arange(T)
    theta = np.asarray([_float(row, "true_theta") for row in reference_rows], dtype=float)
    ref = np.asarray([0.0 if t < int(cfg.quiet_fraction * T) else cfg.production_ref for t in range(T)], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True)
    axes[0].plot(t_grid, theta, color="black", linewidth=2.0, label="true theta")
    axes[0].set_ylabel(r"$\theta_t$")
    axes[0].legend(loc="upper left")

    for method in methods:
        rows = groups.get((seed, method))
        if not rows:
            continue
        style = METHOD_STYLE.get(method, {"label": method})
        state = np.asarray([_float(row, "state") for row in rows], dtype=float)
        action = np.asarray([_float(row, "action") for row in rows], dtype=float)
        cum = np.asarray([_float(row, "cumulative_net_reward") for row in rows], dtype=float)
        axes[1].plot(t_grid[: len(state)], state, linewidth=1.6, **style)
        axes[2].plot(t_grid[: len(action)], action, linewidth=1.6, **style)
        axes[3].plot(t_grid[: len(cum)], cum, linewidth=1.6, **style)
    axes[1].plot(t_grid, ref, color="black", linestyle=":", linewidth=2.0, label="reference")
    axes[1].set_ylabel("state x")
    axes[2].set_ylabel("action a")
    axes[3].set_ylabel("cumulative net reward")
    axes[3].set_xlabel("deployment time t")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[0].set_title(f"Toy1 representative trajectory (median {reference_seed_method} seed={seed})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BRPC toy trajectory diagnostics from raw CSV outputs.")
    parser.add_argument("--raw", type=Path, required=True, help="Path to brpc_validation_raw.csv or brpc_baseline_raw.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--reference-seed-method", default="ce_brpc")
    parser.add_argument("--action-grid-size", type=int, default=201)
    parser.add_argument("--toy", choices=("Toy1", "Toy2", "both"), default="both")
    parser.add_argument("--horizon-raw", type=Path, nargs="*", default=[], help="Optional raw CSVs for Toy2 M-sweep small multiples.")
    args = parser.parse_args()

    rows = _read_rows(args.raw)
    methods = tuple(args.methods)
    if args.toy in ("Toy2", "both"):
        plot_toy2_representative(rows, args.out_dir / "toy2_representative_trajectory.png", methods, args.reference_seed_method, args.action_grid_size)
        plot_toy2_aggregate(rows, args.out_dir / "toy2_aggregate_trajectory.png", methods, args.action_grid_size)
        plot_toy2_belief_diagnostics(rows, args.out_dir / "toy2_belief_diagnostics.png", methods, args.reference_seed_method)
        if args.horizon_raw:
            plot_toy2_horizon_small_multiples(args.horizon_raw, args.out_dir / "toy2_horizon_small_multiples.png", methods, args.reference_seed_method, args.action_grid_size)
    if args.toy in ("Toy1", "both"):
        plot_toy1_representative(rows, args.out_dir / "toy1_representative_trajectory.png", methods, args.reference_seed_method)
    print(f"wrote trajectory plots to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
