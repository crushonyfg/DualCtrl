"""Smoke runner for the 2x2 BRPC baseline matrix.

This is a tiny engineering smoke test, not an experiment runner and not a source of
paper claims. It verifies that CE/PS planners can run with BRPC and BOCPD-BRPC on a
toy environment with matched interfaces.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .bocpd_brpc import BOCPDBRPC, BOCPDConfig
from .brpc import BRPCConfig, FixedSupportBRPC
from .planners import CEPlanner, CEMConfig, PosteriorSamplingPlanner
from .toy_envs import Toy2Config, Toy2DigitalTwin, Toy2PhysicalEnv


BASELINE_MATRIX = ("ce_brpc", "ps_brpc", "ce_bbrpc", "ps_bbrpc")


@dataclass(frozen=True)
class SmokeResult:
    baseline: str
    total_return: float
    steps: int
    planner_queries: int
    diagnostics: dict


def _toy2_reward_fn(config: Toy2Config):
    def reward(state: np.ndarray, action: np.ndarray, previous_action: np.ndarray, t: int) -> float:
        del state, t
        a = float(action[0])
        prev = float(previous_action[0])
        # CEM only needs a lightweight stage objective; response is supplied through
        # predictive transition for model dynamics, while immediate production reward
        # is approximated by action cost terms in this smoke runner.
        return -(config.lambda_energy * a * a + config.lambda_switch * (a - prev) ** 2)

    return reward


def make_toy2_brpc(seed: int) -> FixedSupportBRPC:
    cfg = Toy2Config(horizon_T=20, sigma_y=0.03)
    twin = Toy2DigitalTwin(cfg)
    inducing = np.linspace(cfg.action_low, cfg.action_high, 16)
    inducing = np.column_stack([inducing, inducing])
    brpc_cfg = BRPCConfig(
        theta_low=0.0,
        theta_high=1.0,
        num_particles=32,
        sigma_theta=0.20,
        sigma_epsilon=cfg.sigma_y,
        kernel_output_scale=0.05,
        kernel_length_scale=(0.3, 0.3),
        theta_process_std=0.01,
        random_seed=seed,
    )
    return FixedSupportBRPC(twin, inducing, brpc_cfg)


def run_smoke_baseline(baseline: str, seed: int = 0, horizon: int = 8) -> SmokeResult:
    if baseline not in BASELINE_MATRIX:
        raise ValueError(f"Unknown baseline {baseline!r}; expected one of {BASELINE_MATRIX}.")
    env_cfg = Toy2Config(horizon_T=horizon, change_time=max(1, horizon // 2), sigma_y=0.0)
    env = Toy2PhysicalEnv(env_cfg, seed=seed)
    brpc = make_toy2_brpc(seed + 11)
    calibrator = BOCPDBRPC(brpc, BOCPDConfig(hazard=0.10, max_experts=4, min_segment_length=1)) if "bbrpc" in baseline else brpc
    cem_cfg = CEMConfig(horizon=2, population=16, iterations=2, action_low=0.0, action_high=1.0, random_seed=seed + 23)
    reward_fn = _toy2_reward_fn(env_cfg)
    if baseline.startswith("ce"):
        planner = CEPlanner(reward_fn, cem_cfg)
    else:
        planner = PosteriorSamplingPlanner(brpc.twin, brpc.inducing_points, brpc.kernel, reward_fn, cem_cfg)

    state = env.reset()
    previous = env.previous_action.copy()
    total = 0.0
    for t in range(horizon):
        action = planner.act(state, previous, calibrator, t=t)
        next_state, reward, done, info = env.step(action)
        calibrator.update(info["calibration_input"][None, :], info["calibration_output"][None, :])
        total += reward.net_reward
        state = next_state
        previous = action
        if done:
            break
    diagnostics = calibrator.diagnostics()
    return SmokeResult(baseline, float(total), t + 1, int(planner.query_count), diagnostics)


def run_matrix(seed: int = 0, horizon: int = 8) -> list[SmokeResult]:
    return [run_smoke_baseline(name, seed=seed, horizon=horizon) for name in BASELINE_MATRIX]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight 2x2 BRPC baseline smoke matrix.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    for result in run_matrix(seed=args.seed, horizon=args.horizon):
        print(
            f"{result.baseline}: return={result.total_return:.6f} steps={result.steps} "
            f"queries={result.planner_queries}"
        )


if __name__ == "__main__":
    main()
