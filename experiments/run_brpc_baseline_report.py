"""Run a lightweight BRPC-only 2x2 baseline matrix and write a Chinese report.

This runner is intentionally separate from the older KH/Arcari/TVGP reports.  It is
for engineering smoke/small checks only: small horizons, few particles, and no paper
claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brpc_baselines.bocpd_brpc import BOCPDBRPC, BOCPDConfig
from brpc_baselines.brpc import BRPCConfig, FixedSupportBRPC
from brpc_baselines.planners import (
    CEPlanner,
    CEMConfig,
    PosteriorSamplingPlanner,
    ToyCurrentDynamicsOraclePlanner,
    ToyFutureRegimeOraclePlanner,
    stage_reward,
)
from brpc_baselines.toy_envs import (
    Toy1Config,
    Toy1DigitalTwin,
    Toy1PhysicalEnv,
    Toy2Config,
    Toy2DigitalTwin,
    Toy2PhysicalEnv,
)

BASELINES: tuple[tuple[str, str, str], ...] = (
    ("ce_brpc", "BRPC", "CE"),
    ("ps_brpc", "BRPC", "PS"),
    ("ce_bbrpc", "BOCPD-BRPC", "CE"),
    ("ps_bbrpc", "BOCPD-BRPC", "PS"),
    ("oracle_current", "true_current_dynamics", "CEM-MPC oracle"),
    ("oracle_future", "future_regime_path_appendix_ceiling", "CEM-MPC oracle"),
)


@dataclass(frozen=True)
class RunnerConfig:
    horizon: int = 6
    seeds: tuple[int, ...] = (0,)
    cold_start_transitions: int = 4
    num_particles: int = 24
    inducing_points: int = 12
    cem_horizon: int = 2
    cem_population: int = 12
    cem_iterations: int = 2
    bocpd_hazard: float = 0.15
    bocpd_max_experts: int = 4
    bocpd_min_segment_length: int = 1
    output_note: str = "lightweight smoke/small run; not a formal statistical experiment"


def _safe_float(value: Any) -> float | str:
    try:
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _toy1_paths(cfg: Toy1Config, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    theta_path = np.linspace(cfg.theta_initial, min(1.12, cfg.theta_initial + 0.20), cfg.horizon_T)
    beta_path = np.full(cfg.horizon_T, cfg.beta_initial)
    noise_path = rng.normal(0.0, cfg.sigma_w, cfg.horizon_T)
    return theta_path, beta_path, noise_path


def _toy2_paths(cfg: Toy2Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20_000 + seed)
    cp = cfg.change_time if cfg.change_time is not None else max(1, cfg.horizon_T // 2)
    theta_path = np.full(cfg.horizon_T, cfg.theta_initial)
    theta_path[int(cp) :] = cfg.theta_after_jump
    noise_path = rng.normal(0.0, cfg.sigma_y, cfg.horizon_T)
    return theta_path, noise_path


def _cold_start_toy1(cfg: Toy1Config, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.empty((0, 2)), np.empty((0, 1))
    env = Toy1PhysicalEnv(
        Toy1Config(
            horizon_T=n,
            theta_initial=cfg.theta_initial,
            beta_initial=cfg.beta_initial,
            kappa_delta=cfg.kappa_delta,
            sigma_w=0.0,
            q_x=cfg.q_x,
            lambda_energy=cfg.lambda_energy,
            lambda_switch=cfg.lambda_switch,
            action_low=cfg.action_low,
            action_high=cfg.action_high,
            x0=cfg.x0,
            production_ref=cfg.production_ref,
            quiet_fraction=cfg.quiet_fraction,
        ),
        noise_path=np.zeros(n),
    )
    actions = np.linspace(-0.5, 0.5, n)
    xs, ys = [], []
    env.reset()
    for action in actions:
        _, _, _, info = env.step(np.array([action]))
        xs.append(info["calibration_input"])
        ys.append(info["calibration_output"])
    return np.asarray(xs), np.asarray(ys)


def _cold_start_toy2(cfg: Toy2Config, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.empty((0, 2)), np.empty((0, 1))
    cold_cfg = Toy2Config(
        horizon_T=n,
        theta_initial=cfg.theta_initial,
        theta_after_jump=cfg.theta_initial,
        change_time=n + 1,
        sigma_basis=cfg.sigma_basis,
        b_left=cfg.b_left,
        b_right=cfg.b_right,
        b_diag=cfg.b_diag,
        c_right=cfg.c_right,
        c_diag=cfg.c_diag,
        discrepancy_sine_amplitude=cfg.discrepancy_sine_amplitude,
        discrepancy_cubic=cfg.discrepancy_cubic,
        sigma_y=0.0,
        lambda_energy=cfg.lambda_energy,
        lambda_switch=cfg.lambda_switch,
        action_low=cfg.action_low,
        action_high=cfg.action_high,
        a_left=cfg.a_left,
        a_diag=cfg.a_diag,
        a_right=cfg.a_right,
    )
    env = Toy2PhysicalEnv(cold_cfg, noise_path=np.zeros(n))
    actions = np.linspace(cfg.action_low, cfg.action_high, n)
    xs, ys = [], []
    env.reset()
    for action in actions:
        _, _, _, info = env.step(np.array([action]))
        xs.append(info["calibration_input"])
        ys.append(info["calibration_output"])
    return np.asarray(xs), np.asarray(ys)


def _make_brpc(environment: str, runner_cfg: RunnerConfig, seed: int) -> FixedSupportBRPC:
    if environment == "Toy1":
        twin = Toy1DigitalTwin()
        grid = np.linspace(-1.0, 1.0, runner_cfg.inducing_points)
        inducing = np.column_stack([grid, np.roll(grid, runner_cfg.inducing_points // 3)])
        cfg = BRPCConfig(
            theta_low=0.60,
            theta_high=1.25,
            num_particles=runner_cfg.num_particles,
            sigma_theta=0.20,
            sigma_epsilon=0.04,
            kernel_output_scale=0.08,
            kernel_length_scale=(0.55, 0.55),
            theta_process_std=0.015,
            random_seed=1_000 + seed,
        )
    elif environment == "Toy2":
        twin = Toy2DigitalTwin()
        grid = np.linspace(0.0, 1.0, runner_cfg.inducing_points)
        inducing = np.column_stack([grid, grid])
        cfg = BRPCConfig(
            theta_low=0.0,
            theta_high=1.0,
            num_particles=runner_cfg.num_particles,
            sigma_theta=0.20,
            sigma_epsilon=0.04,
            kernel_output_scale=0.06,
            kernel_length_scale=(0.35, 0.35),
            theta_process_std=0.02,
            random_seed=2_000 + seed,
        )
    else:
        raise ValueError(environment)
    return FixedSupportBRPC(twin, inducing, cfg)


def _initialize_calibrator(environment: str, baseline: str, runner_cfg: RunnerConfig, seed: int):
    brpc = _make_brpc(environment, runner_cfg, seed)
    if environment == "Toy1":
        cold_x, cold_y = _cold_start_toy1(Toy1Config(horizon_T=max(1, runner_cfg.horizon)), runner_cfg.cold_start_transitions)
    else:
        cold_x, cold_y = _cold_start_toy2(Toy2Config(horizon_T=max(1, runner_cfg.horizon)), runner_cfg.cold_start_transitions)
    if len(cold_x):
        brpc.update(cold_x, cold_y)
    if "bbrpc" in baseline:
        return BOCPDBRPC(
            brpc,
            BOCPDConfig(
                hazard=runner_cfg.bocpd_hazard,
                max_experts=runner_cfg.bocpd_max_experts,
                min_segment_length=runner_cfg.bocpd_min_segment_length,
            ),
        )
    return brpc


def _is_oracle_baseline(baseline: str) -> bool:
    return baseline.startswith("oracle_")


def _stage_reward_fn(environment: str, env_cfg: Toy1Config | Toy2Config):
    if environment == "Toy1":
        cfg = env_cfg
        assert isinstance(cfg, Toy1Config)

        def reward(predicted_response: np.ndarray, action: np.ndarray, previous_action: np.ndarray) -> float:
            x_next = float(np.asarray(predicted_response).reshape(-1)[0])
            a = float(np.asarray(action).reshape(-1)[0])
            prev = float(np.asarray(previous_action).reshape(-1)[0])
            ref = cfg.production_ref
            return -cfg.q_x * (x_next - ref) ** 2 - cfg.lambda_energy * a * a - cfg.lambda_switch * (a - prev) ** 2

        return reward

    cfg = env_cfg
    assert isinstance(cfg, Toy2Config)

    def reward(predicted_response: np.ndarray, action: np.ndarray, previous_action: np.ndarray) -> float:
        return stage_reward(predicted_response, action, previous_action, cfg.lambda_energy, cfg.lambda_switch)

    return reward


def _make_planner(environment: str, baseline: str, env_cfg: Toy1Config | Toy2Config, runner_cfg: RunnerConfig, seed: int, calibrator, env=None):
    if environment == "Toy1":
        low, high = env_cfg.action_low, env_cfg.action_high
        twin = Toy1DigitalTwin()
    else:
        low, high = env_cfg.action_low, env_cfg.action_high
        twin = Toy2DigitalTwin(env_cfg)
    cem = CEMConfig(
        horizon=runner_cfg.cem_horizon,
        population=runner_cfg.cem_population,
        iterations=runner_cfg.cem_iterations,
        elite_fraction=0.25,
        smoothing=0.20,
        action_low=low,
        action_high=high,
        random_seed=3_000 + 97 * seed + len(baseline),
    )
    if baseline == "oracle_current":
        if env is None:
            raise ValueError("oracle_current requires the physical toy env")
        return ToyCurrentDynamicsOraclePlanner(env, cem, environment=environment)
    if baseline == "oracle_future":
        if env is None:
            raise ValueError("oracle_future requires the physical toy env")
        return ToyFutureRegimeOraclePlanner(env, cem, environment=environment)
    reward = _stage_reward_fn(environment, env_cfg)
    if baseline.startswith("ce"):
        return CEPlanner(reward, cem)
    # The PS planner needs the base BRPC geometry.  For BOCPD-BRPC, use the anchor
    # expert's BRPC support/kernel; sample_latent still comes from the expert mixture.
    base_brpc = calibrator.experts[0].brpc if hasattr(calibrator, "experts") else calibrator
    return PosteriorSamplingPlanner(twin, base_brpc.inducing_points, base_brpc.kernel, reward, cem)


def _oracle_noise_free_one_step_reward(environment: str, env: Toy1PhysicalEnv | Toy2PhysicalEnv, state: np.ndarray, action: np.ndarray, previous_action: np.ndarray) -> float:
    if environment == "Toy1":
        theta = float(env.theta_path[env.t])
        beta = float(env.beta_path[env.t])
        next_state = env.twin.step(state, action, theta)
        next_state = np.array([float(next_state[0]) + env.discrepancy(state, action, beta)], dtype=float)
        a = float(np.asarray(action).reshape(-1)[0])
        prev = float(np.asarray(previous_action).reshape(-1)[0])
        return float(
            -env.config.q_x * (float(next_state[0]) - env.config.production_ref) ** 2
            - env.config.lambda_energy * a * a
            - env.config.lambda_switch * (a - prev) ** 2
        )
    theta = float(env.theta_path[env.t])
    response = env.expected_response(action, theta)
    return stage_reward(np.array([response]), action, previous_action, env.config.lambda_energy, env.config.lambda_switch)


def _diagnostic_scalars(diagnostics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if key in {"expert_masses", "expert_start_times", "inner_weight_sums"}:
            out[key] = json.dumps(np.asarray(value).tolist())
        else:
            out[key] = _safe_float(value)
    return out


def run_one(environment: str, baseline: str, calibration: str, planner_name: str, runner_cfg: RunnerConfig, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if environment == "Toy1":
        env_cfg = Toy1Config(horizon_T=runner_cfg.horizon)
        theta_path, beta_path, noise_path = _toy1_paths(env_cfg, seed)
        env = Toy1PhysicalEnv(env_cfg, theta_path=theta_path, beta_path=beta_path, noise_path=noise_path)
    elif environment == "Toy2":
        env_cfg = Toy2Config(horizon_T=runner_cfg.horizon, change_time=max(1, runner_cfg.horizon // 2))
        theta_path, noise_path = _toy2_paths(env_cfg, seed)
        env = Toy2PhysicalEnv(env_cfg, theta_path=theta_path, noise_path=noise_path)
    else:
        raise ValueError(environment)

    calibrator = None if _is_oracle_baseline(baseline) else _initialize_calibrator(environment, baseline, runner_cfg, seed)
    planner = _make_planner(environment, baseline, env_cfg, runner_cfg, seed, calibrator, env=env)
    state = env.reset()
    previous_action = env.previous_action.copy()
    rows: list[dict[str, Any]] = []
    totals = {
        "task_reward": 0.0,
        "energy_cost": 0.0,
        "switching_cost": 0.0,
        "failure_cost": 0.0,
        "net_reward": 0.0,
    }
    previous_queries = 0
    restart_count = 0

    for t in range(runner_cfg.horizon):
        action = planner.act(state, previous_action, calibrator, t=t)
        predicted_reward = _oracle_noise_free_one_step_reward(environment, env, state, action, previous_action) if _is_oracle_baseline(baseline) else float("nan")
        next_state, reward, done, info = env.step(action)
        if calibrator is not None:
            calibrator.update(info["calibration_input"][None, :], info["calibration_output"][None, :])
            diagnostics = _diagnostic_scalars(calibrator.diagnostics())
        else:
            diagnostics = {"oracle_kind": getattr(planner, "oracle_kind", baseline)}
        restart_count += int(bool(diagnostics.get("restart_event", False)))
        step_queries = int(planner.query_count - previous_queries)
        previous_queries = int(planner.query_count)

        totals["task_reward"] += reward.task_reward
        totals["energy_cost"] += reward.energy_cost
        totals["switching_cost"] += reward.switching_cost
        totals["failure_cost"] += reward.failure_cost
        totals["net_reward"] += reward.net_reward

        row = {
            "environment": environment,
            "seed": seed,
            "baseline": baseline,
            "calibration": calibration,
            "planner": planner_name,
            "t": t,
            "state": float(np.asarray(state).reshape(-1)[0]),
            "previous_action": float(np.asarray(previous_action).reshape(-1)[0]),
            "action": float(np.asarray(action).reshape(-1)[0]),
            "next_state": float(np.asarray(next_state).reshape(-1)[0]),
            "calibration_output": float(np.asarray(info["calibration_output"]).reshape(-1)[0]),
            "true_theta": float(info["theta"]),
            "predicted_reward": float(predicted_reward),
            "realized_reward": float(reward.net_reward),
            "task_reward": float(reward.task_reward),
            "energy_cost": float(reward.energy_cost),
            "switching_cost": float(reward.switching_cost),
            "failure_cost": float(reward.failure_cost),
            "net_reward": float(reward.net_reward),
            "cumulative_task_reward": totals["task_reward"],
            "cumulative_energy_cost": totals["energy_cost"],
            "cumulative_switching_cost": totals["switching_cost"],
            "cumulative_failure_cost": totals["failure_cost"],
            "cumulative_net_reward": totals["net_reward"],
            "planner_queries_step": step_queries,
            "planner_queries_total": int(planner.query_count),
            "cold_start_transitions": runner_cfg.cold_start_transitions,
        }
        row.update(diagnostics)
        rows.append(row)
        state = next_state
        previous_action = action
        if done:
            break

    seed_summary = {
        "environment": environment,
        "seed": seed,
        "baseline": baseline,
        "calibration": calibration,
        "planner": planner_name,
        "steps": len(rows),
        "cold_start_transitions": runner_cfg.cold_start_transitions,
        "total_task_reward": totals["task_reward"],
        "total_energy_cost": totals["energy_cost"],
        "total_switching_cost": totals["switching_cost"],
        "total_failure_cost": totals["failure_cost"],
        "total_net_reward": totals["net_reward"],
        "mean_net_reward_per_step": totals["net_reward"] / max(1, len(rows)),
        "planner_queries_total": int(planner.query_count),
        "mean_planner_queries_per_step": int(planner.query_count) / max(1, len(rows)),
        "restart_count": restart_count,
    }
    return rows, seed_summary


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        groups.setdefault((row["environment"], row["baseline"]), []).append(row)
    out = []
    metrics = [
        "total_task_reward",
        "total_energy_cost",
        "total_switching_cost",
        "total_failure_cost",
        "total_net_reward",
        "mean_net_reward_per_step",
        "planner_queries_total",
        "mean_planner_queries_per_step",
        "restart_count",
    ]
    for (environment, baseline), rows in sorted(groups.items()):
        first = rows[0]
        item: dict[str, Any] = {
            "environment": environment,
            "baseline": baseline,
            "calibration": first["calibration"],
            "planner": first["planner"],
            "n_seeds": len(rows),
            "horizon": first["steps"],
            "cold_start_transitions": first["cold_start_transitions"],
        }
        for metric in metrics:
            values = [float(r[metric]) for r in rows]
            item[f"mean_{metric}"] = float(np.mean(values))
            item[f"stderr_{metric}"] = _stderr(values)
        out.append(item)
    return out


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "environment",
        "baseline",
        "n_seeds",
        "mean_total_net_reward",
        "mean_total_task_reward",
        "mean_total_energy_cost",
        "mean_total_switching_cost",
        "mean_planner_queries_total",
        "mean_restart_count",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row[col]
            vals.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, runner_cfg: RunnerConfig, summary_rows: list[dict[str, Any]]) -> None:
    report = [
        "# BRPC baseline 轻量实验报告",
        "",
        "本报告只包含新的 BRPC baseline 矩阵：`ce_brpc`、`ps_brpc`、`ce_bbrpc`、`ps_bbrpc`。它写入独立目录，不合并旧的 KH / Arcari / TVGP 结果。",
        "",
        "## 实验范围",
        "",
        f"- 环境：Toy1 与 Toy2。",
        f"- 矩阵：CE/PS planner × BRPC/BOCPD-BRPC calibration。",
        f"- horizon={runner_cfg.horizon}，seeds={list(runner_cfg.seeds)}，particles={runner_cfg.num_particles}，inducing points={runner_cfg.inducing_points}。",
        f"- CEM：horizon={runner_cfg.cem_horizon}，population={runner_cfg.cem_population}，iterations={runner_cfg.cem_iterations}。",
        "- 这是 smoke/small run，用于检查接口、记账和输出格式；不用于论文级统计结论。",
        "",
        "## 数学建模摘要",
        "",
        "Toy1 使用标量动态：digital twin 为 $f_{DT}(x,a;\\theta)=\\theta x+a$，physical transition 额外包含 $\\beta\\tanh(2x)$、动作非线性 discrepancy 与高斯噪声。净回报按真实部署轨迹记账：任务项 $-q_x(x-x^{ref})^2$ 减 energy cost $\\lambda_E a^2$ 与 switching cost $\\lambda_\\Delta(a-a_{prev})^2$。",
        "",
        "Toy2 使用连续动作 response landscape：digital twin 由左、右、诊断三个 Gaussian basis 组成，physical response 再加固定 discrepancy 与观测噪声。净回报为 response 减 energy 与 switching cost；状态按 benchmark 约定记录为上一步动作。",
        "",
        "BRPC 使用固定 inducing/support set、参数粒子、particle-specific discrepancy mean 与 shared discrepancy covariance。参数权重通过 discrepancy-free likelihood 做 tempered update，discrepancy 使用 fixed-support GP 条件更新，并在 ESS 低时把参数粒子与 discrepancy mean 一起 resample。",
        "",
        "BOCPD-BRPC 在 BRPC expert mixture 上做 prequential evidence、hazard restart 分支、expert mass 归一化与 pruning。CE planner 使用 posterior predictive mean；PS planner 每个 physical step 采样一个 latent model 后规划。Toy2 planner 的 stage objective 使用同一个 physical accounting：predicted response 减 energy cost 与 previous-action switching cost。",
        "",
        "## 结果摘要",
        "",
        _markdown_table(summary_rows),
        "",
        "## 输出文件",
        "",
        "- `brpc_baseline_raw.csv`：逐步 raw 记录，包含 reward/cost decomposition、真实 theta、动作、累计回报、planner query count 和 calibrator diagnostics。",
        "- `brpc_baseline_seed_summary.csv`：每个 environment/seed/baseline 的累计分项。",
        "- `brpc_baseline_summary.csv`：按 environment/baseline 聚合的 mean 与 standard error。",
        "- `config.json`：本次轻量运行配置。",
        "",
        "## Caveats",
        "",
        "- 该运行 horizon、seed 数、粒子数和 CEM budget 都很小，只能作为 smoke/small check。",
        "- 未实现 oracle，因此不报告 oracle regret。",
        "- Toy2 geometry/reporting 区分 production operating optimum（不含 switching）和 previous-action-dependent one-step net reward；二者不应混用。",
        "- 未做 geometry plots、bootstrap CI、hazard sensitivity 或正式 paired statistical protocol。",
        "- 所有结论应限于：代码路径可运行、CSV 记账完整、BRPC-only 输出与旧 baseline 输出分离。",
        "",
    ]
    (out_dir / "brpc_baseline_report_zh.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight BRPC 2x2 baselines on Toy1 and Toy2.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables/brpc_baseline_small"))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--cold-start-transitions", type=int, default=4)
    parser.add_argument("--num-particles", type=int, default=24)
    parser.add_argument("--inducing-points", type=int, default=12)
    parser.add_argument("--cem-population", type=int, default=12)
    parser.add_argument("--cem-iterations", type=int, default=2)
    args = parser.parse_args()

    runner_cfg = RunnerConfig(
        horizon=args.horizon,
        seeds=tuple(args.seeds),
        cold_start_transitions=args.cold_start_transitions,
        num_particles=args.num_particles,
        inducing_points=args.inducing_points,
        cem_population=args.cem_population,
        cem_iterations=args.cem_iterations,
    )

    raw_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    for environment in ("Toy1", "Toy2"):
        for seed in runner_cfg.seeds:
            for baseline, calibration, planner_name in BASELINES:
                rows, seed_summary = run_one(environment, baseline, calibration, planner_name, runner_cfg, seed)
                raw_rows.extend(rows)
                seed_summary_rows.append(seed_summary)

    summary_rows = summarize(seed_summary_rows)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "brpc_baseline_raw.csv", raw_rows)
    _write_csv(out_dir / "brpc_baseline_seed_summary.csv", seed_summary_rows)
    _write_csv(out_dir / "brpc_baseline_summary.csv", summary_rows)
    (out_dir / "config.json").write_text(json.dumps(asdict(runner_cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir, runner_cfg, summary_rows)
    print(f"wrote BRPC baseline outputs to {out_dir}")
    print(f"raw_rows={len(raw_rows)} seed_summary_rows={len(seed_summary_rows)} summary_rows={len(summary_rows)}")


if __name__ == "__main__":
    main()
