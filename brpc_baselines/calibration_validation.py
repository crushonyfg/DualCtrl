"""Calibration-only validation for BRPC and BOCPD-BRPC.

This module deliberately avoids planner calls.  It generates fixed exogenous action
sequences, records one physical trajectory per Toy/scenario, then feeds exactly the
same calibration inputs and outputs to standalone BRPC and BOCPD-BRPC.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .bocpd_brpc import BOCPDBRPC, BOCPDConfig
from .brpc import BRPCConfig, FixedSupportBRPC, effective_sample_size, logsumexp
from .toy_envs import Toy1Config, Toy1DigitalTwin, Toy1PhysicalEnv, Toy2Config, Toy2DigitalTwin, Toy2PhysicalEnv

SCENARIOS = ("stationary", "gradual_drift", "abrupt_change", "no_change_false_alarm")
TOYS = ("toy1", "toy2")
ALGORITHMS = ("brpc", "bocpd_brpc")

RESULT_SCHEMA = (
    "toy",
    "scenario",
    "algorithm",
    "horizon",
    "theta_rmse",
    "response_rmse",
    "nlpd",
    "coverage_proxy_95",
    "ess_mean",
    "ess_min",
    "ess_final",
    "discrepancy_error_proxy_rmse",
    "restart_count",
    "restart_false_alarm_rate",
    "restart_delay_mean",
    "restart_precision",
    "restart_recall",
    "expert_mass_summary",
)


@dataclass(frozen=True)
class CalibrationValidationConfig:
    horizon: int = 40
    num_particles: int = 32
    seed: int = 0
    hazard: float = 0.04
    max_experts: int = 6
    min_segment_length: int = 6
    restart_margin_rho_B: float = 100.0
    restart_tolerance: int = 6


@dataclass(frozen=True)
class CalibrationRecord:
    time: int
    calibration_input: np.ndarray
    calibration_output: np.ndarray
    true_theta: float
    true_response: float
    true_discrepancy: float


def fixed_action_sequence(toy: str, horizon: int) -> np.ndarray:
    """Return deterministic exogenous actions independent of either calibrator."""

    t = np.arange(int(horizon), dtype=float)
    if toy == "toy1":
        actions = 0.55 * np.sin(2.0 * np.pi * t / 7.0) + 0.25 * np.cos(2.0 * np.pi * t / 13.0)
        return np.clip(actions, -1.0, 1.0)
    if toy == "toy2":
        pattern = np.asarray([0.2, 0.5, 0.8, 0.5, 0.35, 0.65], dtype=float)
        return np.resize(pattern, int(horizon))
    raise ValueError(f"Unknown toy {toy!r}; expected one of {TOYS}.")


def _theta_path(scenario: str, horizon: int, low: float, high: float) -> tuple[np.ndarray, list[int]]:
    if scenario in ("stationary", "no_change_false_alarm"):
        return np.full(horizon, low, dtype=float), []
    if scenario == "gradual_drift":
        return np.linspace(low, high, horizon, dtype=float), []
    if scenario == "abrupt_change":
        cp = horizon // 2
        path = np.full(horizon, low, dtype=float)
        path[cp:] = high
        return path, [cp]
    raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}.")


def make_calibration_dataset(toy: str, scenario: str, horizon: int, seed: int = 0) -> tuple[list[CalibrationRecord], object, list[int], np.ndarray]:
    """Generate one trajectory for a toy/scenario and fixed action sequence."""

    horizon = int(horizon)
    actions = fixed_action_sequence(toy, horizon)
    deterministic = scenario == "no_change_false_alarm"
    if toy == "toy1":
        theta, change_times = _theta_path(scenario, horizon, 0.45, 0.85)
        cfg = Toy1Config(horizon_T=horizon, sigma_w=0.0 if deterministic else 0.01, theta_initial=float(theta[0]))
        env = Toy1PhysicalEnv(
            cfg,
            theta_path=theta,
            beta_path=np.full(horizon, cfg.beta_initial, dtype=float),
            noise_path=np.zeros(horizon) if deterministic else None,
            seed=seed,
        )
    elif toy == "toy2":
        theta, change_times = _theta_path(scenario, horizon, 0.10, 1.00)
        cfg = Toy2Config(horizon_T=horizon, sigma_y=0.0 if deterministic else 0.01, change_time=horizon + 1)
        env = Toy2PhysicalEnv(cfg, theta_path=theta, noise_path=np.zeros(horizon) if deterministic else None, seed=seed)
    else:
        raise ValueError(f"Unknown toy {toy!r}; expected one of {TOYS}.")

    env.reset()
    records: list[CalibrationRecord] = []
    for t, action in enumerate(actions):
        _, _, _, info = env.step(np.array([action], dtype=float))
        x = np.asarray(info["calibration_input"], dtype=float)
        y = np.asarray(info["calibration_output"], dtype=float)
        true_response = float(y.reshape(-1)[0] - float(info.get("noise", 0.0)))
        true_theta = float(info["theta"])
        twin_response = float(env.twin.batch_step(x[None, :], true_theta)[0, 0])
        records.append(
            CalibrationRecord(
                time=t,
                calibration_input=x,
                calibration_output=y,
                true_theta=true_theta,
                true_response=true_response,
                true_discrepancy=true_response - twin_response,
            )
        )
    return records, env.twin, change_times, actions


def _inducing_points(toy: str) -> np.ndarray:
    if toy == "toy1":
        xs = np.linspace(-1.5, 1.5, 4)
        actions = np.linspace(-1.0, 1.0, 4)
    elif toy == "toy2":
        xs = np.linspace(0.0, 1.0, 4)
        actions = np.linspace(0.0, 1.0, 4)
    else:
        raise ValueError(f"Unknown toy {toy!r}; expected one of {TOYS}.")
    return np.asarray([[x, a] for x in xs for a in actions], dtype=float)


def _brpc_config(toy: str, cfg: CalibrationValidationConfig, seed: int) -> BRPCConfig:
    if toy == "toy1":
        return BRPCConfig(
            theta_low=0.0,
            theta_high=1.0,
            num_particles=cfg.num_particles,
            sigma_theta=0.12,
            sigma_epsilon=0.02,
            kernel_output_scale=0.08,
            kernel_length_scale=(0.8, 0.5),
            theta_process_std=0.01,
            random_seed=seed,
        )
    if toy == "toy2":
        return BRPCConfig(
            theta_low=0.0,
            theta_high=1.0,
            num_particles=cfg.num_particles,
            sigma_theta=0.18,
            sigma_epsilon=0.02,
            kernel_output_scale=0.06,
            kernel_length_scale=(0.35, 0.35),
            theta_process_std=0.01,
            random_seed=seed,
        )
    raise ValueError(f"Unknown toy {toy!r}; expected one of {TOYS}.")


def _make_calibrators(toy: str, twin: object, cfg: CalibrationValidationConfig) -> dict[str, object]:
    inducing = _inducing_points(toy)
    brpc = FixedSupportBRPC(twin, inducing, _brpc_config(toy, cfg, cfg.seed + 101))
    bbrpc_anchor = FixedSupportBRPC(twin, inducing, _brpc_config(toy, cfg, cfg.seed + 101))
    bbrpc = BOCPDBRPC(
        bbrpc_anchor,
        BOCPDConfig(
            hazard=cfg.hazard,
            max_experts=cfg.max_experts,
            restart_margin_rho_B=cfg.restart_margin_rho_B,
            min_segment_length=cfg.min_segment_length,
        ),
    )
    return {"brpc": brpc, "bocpd_brpc": bbrpc}


def _predictive_components(calibrator: object, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.atleast_2d(np.asarray(inputs, dtype=float))
    weights: list[float] = []
    means: list[float] = []
    variances: list[float] = []

    if isinstance(calibrator, BOCPDBRPC):
        for expert in calibrator.experts:
            pred = expert.brpc.predict(X)
            for idx, weight in enumerate(pred["weights"]):
                weights.append(float(expert.mass * weight))
                means.append(float(pred["means"][idx, 0, 0]))
                variances.append(float(pred["covariances"][idx, 0, 0, 0]))
    else:
        pred = calibrator.predict(X)  # type: ignore[attr-defined]
        for idx, weight in enumerate(pred["weights"]):
            weights.append(float(weight))
            means.append(float(pred["means"][idx, 0, 0]))
            variances.append(float(pred["covariances"][idx, 0, 0, 0]))

    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return w, np.asarray(means, dtype=float), np.maximum(np.asarray(variances, dtype=float), 1e-12)


def _predictive_summary(calibrator: object, inputs: np.ndarray) -> tuple[float, float]:
    w, means, variances = _predictive_components(calibrator, inputs)
    mean = float(np.sum(w * means))
    second = float(np.sum(w * (variances + means * means)))
    return mean, max(second - mean * mean, 1e-12)


def _log_predictive(calibrator: object, inputs: np.ndarray, outputs: np.ndarray) -> float | None:
    try:
        if isinstance(calibrator, BOCPDBRPC):
            components = [expert.log_mass + expert.brpc.log_predictive(inputs, outputs) for expert in calibrator.experts]
            return float(logsumexp(np.asarray(components, dtype=float)))
        return float(calibrator.log_predictive(inputs, outputs))  # type: ignore[attr-defined]
    except (AttributeError, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None


def _theta_mean(calibrator: object) -> float:
    if isinstance(calibrator, BOCPDBRPC):
        return float(sum(expert.mass * expert.brpc.diagnostics()["theta_mean"] for expert in calibrator.experts))
    return float(calibrator.diagnostics()["theta_mean"])  # type: ignore[attr-defined]


def _ess(calibrator: object) -> float:
    if isinstance(calibrator, BOCPDBRPC):
        flat_weights = []
        for expert in calibrator.experts:
            flat_weights.extend((expert.mass * expert.brpc.state.theta_weights).tolist())
        return effective_sample_size(np.asarray(flat_weights, dtype=float))
    return float(calibrator.diagnostics()["ess"])  # type: ignore[attr-defined]


def _expert_mass_snapshot(calibrator: object) -> dict[str, float | int | None]:
    if not isinstance(calibrator, BOCPDBRPC):
        return {
            "num_experts_final": None,
            "anchor_start_time_final": None,
            "recent_change_probability_final": None,
            "recent_change_probability_max": None,
            "expert_entropy_mean": None,
        }
    diag = calibrator.diagnostics()
    masses = np.asarray(diag["expert_masses"], dtype=float)
    entropy = float(-np.sum(masses * np.log(np.maximum(masses, 1e-300))))
    return {
        "num_experts_final": int(diag["num_experts"]),
        "anchor_start_time_final": int(diag["anchor_start_time"]),
        "recent_change_probability_final": float(diag["recent_change_probability"]),
        "recent_change_probability_max": float(diag["recent_change_probability"]),
        "expert_entropy_mean": entropy,
    }


def _restart_metrics(events: list[int], true_changes: list[int], horizon: int, tolerance: int) -> dict[str, float | int | None]:
    matched_events: set[int] = set()
    delays: list[int] = []
    for cp in true_changes:
        candidates = [event for event in events if event >= cp and event - cp <= tolerance and event not in matched_events]
        if candidates:
            event = min(candidates)
            matched_events.add(event)
            delays.append(event - cp)
    tp = len(matched_events)
    precision = None if not events else tp / len(events)
    recall = None if not true_changes else tp / len(true_changes)
    return {
        "restart_count": int(len(events)),
        "restart_false_alarm_rate": float(max(0, len(events) - tp) / max(1, horizon)),
        "restart_delay_mean": None if not delays else float(np.mean(delays)),
        "restart_precision": precision,
        "restart_recall": recall,
    }


def evaluate_calibrator(
    algorithm: str,
    calibrator: object,
    twin: object,
    records: list[CalibrationRecord],
    toy: str,
    scenario: str,
    true_changes: list[int],
    tolerance: int,
) -> dict:
    theta_errors: list[float] = []
    response_errors: list[float] = []
    nlpd_values: list[float] = []
    coverage_hits: list[float] = []
    ess_values: list[float] = []
    discrepancy_errors: list[float] = []
    restart_events: list[int] = []
    recent_probabilities: list[float] = []
    entropies: list[float] = []

    for record in records:
        X = record.calibration_input[None, :]
        Y = record.calibration_output[None, :]
        pred_mean, pred_var = _predictive_summary(calibrator, X)
        response_errors.append(pred_mean - record.true_response)
        coverage_hits.append(float(abs(record.true_response - pred_mean) <= 1.96 * np.sqrt(pred_var)))
        logp = _log_predictive(calibrator, X, Y)
        if logp is not None and np.isfinite(logp):
            nlpd_values.append(-float(logp))

        calibrator.update(X, Y)  # type: ignore[attr-defined]
        theta_hat = _theta_mean(calibrator)
        theta_errors.append(theta_hat - record.true_theta)
        ess_values.append(_ess(calibrator))

        post_mean, _ = _predictive_summary(calibrator, X)
        twin_at_theta_hat = float(twin.batch_step(X, theta_hat)[0, 0])  # type: ignore[attr-defined]
        discrepancy_errors.append((post_mean - twin_at_theta_hat) - record.true_discrepancy)

        if isinstance(calibrator, BOCPDBRPC):
            diag = calibrator.diagnostics()
            if bool(diag["restart_event"]):
                restart_events.append(record.time)
            masses = np.asarray(diag["expert_masses"], dtype=float)
            recent_probabilities.append(float(diag["recent_change_probability"]))
            entropies.append(float(-np.sum(masses * np.log(np.maximum(masses, 1e-300)))))

    expert_summary = _expert_mass_snapshot(calibrator)
    if recent_probabilities:
        expert_summary["recent_change_probability_max"] = float(np.max(recent_probabilities))
    if entropies:
        expert_summary["expert_entropy_mean"] = float(np.mean(entropies))

    restart = _restart_metrics(restart_events, true_changes, len(records), tolerance)
    if algorithm == "brpc":
        restart = {
            "restart_count": 0,
            "restart_false_alarm_rate": 0.0,
            "restart_delay_mean": None,
            "restart_precision": None,
            "restart_recall": None,
        }

    return {
        "toy": toy,
        "scenario": scenario,
        "algorithm": algorithm,
        "horizon": int(len(records)),
        "theta_rmse": float(np.sqrt(np.mean(np.square(theta_errors)))),
        "response_rmse": float(np.sqrt(np.mean(np.square(response_errors)))),
        "nlpd": None if not nlpd_values else float(np.mean(nlpd_values)),
        "coverage_proxy_95": None if not coverage_hits else float(np.mean(coverage_hits)),
        "ess_mean": float(np.mean(ess_values)),
        "ess_min": float(np.min(ess_values)),
        "ess_final": float(ess_values[-1]),
        "discrepancy_error_proxy_rmse": float(np.sqrt(np.mean(np.square(discrepancy_errors)))),
        "restart_count": restart["restart_count"],
        "restart_false_alarm_rate": restart["restart_false_alarm_rate"],
        "restart_delay_mean": restart["restart_delay_mean"],
        "restart_precision": restart["restart_precision"],
        "restart_recall": restart["restart_recall"],
        "expert_mass_summary": expert_summary,
    }


def run_calibration_validation(
    toys: Iterable[str] = TOYS,
    scenarios: Iterable[str] = SCENARIOS,
    config: CalibrationValidationConfig = CalibrationValidationConfig(),
) -> list[dict]:
    """Run calibration-only validation and return one schema-stable row per method."""

    rows: list[dict] = []
    for toy in toys:
        for scenario in scenarios:
            records, twin, true_changes, _ = make_calibration_dataset(toy, scenario, config.horizon, seed=config.seed)
            calibrators = _make_calibrators(toy, twin, config)
            for algorithm in ALGORITHMS:
                rows.append(
                    evaluate_calibrator(
                        algorithm,
                        calibrators[algorithm],
                        twin,
                        records,
                        toy,
                        scenario,
                        true_changes,
                        config.restart_tolerance,
                    )
                )
    return rows


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_rows_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_SCHEMA)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["expert_mass_summary"] = json.dumps(
                serializable["expert_mass_summary"],
                sort_keys=True,
                default=_json_default,
            )
            writer.writerow(serializable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration-only BRPC/BOCPD-BRPC validation.")
    parser.add_argument("--toy", choices=TOYS, action="append", help="Toy to run; repeat for multiple. Default: both.")
    parser.add_argument("--scenario", choices=SCENARIOS, action="append", help="Scenario to run; repeat for multiple. Default: all.")
    parser.add_argument("--horizon", type=int, default=CalibrationValidationConfig.horizon)
    parser.add_argument("--num-particles", type=int, default=CalibrationValidationConfig.num_particles)
    parser.add_argument("--seed", type=int, default=CalibrationValidationConfig.seed)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path. Defaults to printing JSON to stdout.")
    args = parser.parse_args()
    cfg = CalibrationValidationConfig(horizon=args.horizon, num_particles=args.num_particles, seed=args.seed)
    rows = run_calibration_validation(toys=args.toy or TOYS, scenarios=args.scenario or SCENARIOS, config=cfg)
    if args.out is not None:
        write_rows_csv(rows, args.out)
    else:
        print(json.dumps(rows, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
