"""Geometry-gate utilities for lightweight toy benchmark screening.

The benchmark spec requires a pre-baseline screening pass that compares production
reward, parameter information, old/new predictive KL, and switching geometry on a
state/action grid.  These helpers keep that logic deterministic and independent of
calibrators/planners so the toy environments can be screened directly from their
known physical models.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .toy_envs import Toy1Config, Toy1DigitalTwin, Toy2Config, Toy2DigitalTwin


@dataclass(frozen=True)
class GeometryReport:
    action_grid: np.ndarray
    reward_old: np.ndarray
    reward_new: np.ndarray
    sensitivity: np.ndarray
    predictive_kl_old_new: np.ndarray
    switching_cost_from_previous: np.ndarray
    old_opt_action: float
    new_opt_action: float
    diagnostic_action: float
    passes_g1: bool
    passes_g2: bool
    passes_g3: bool
    passes_g5: bool


@dataclass(frozen=True)
class Toy2DiagnosticConditions:
    """Numerical gate checks from spec section 5.3."""

    production_gap_at_diag: float
    diagnostic_information: float
    left_information: float
    right_information: float
    bridge_cost: float
    direct_cost: float
    a_diag_not_production_optimal: bool
    information_higher_at_a_diag: bool
    bridge_switching_inequality: bool

    @property
    def passes(self) -> bool:
        return (
            self.a_diag_not_production_optimal
            and self.information_higher_at_a_diag
            and self.bridge_switching_inequality
        )


def gaussian_kl_same_variance(mean_new: np.ndarray, mean_old: np.ndarray, variance: float) -> np.ndarray:
    variance = max(float(variance), 1e-12)
    return 0.5 * (np.asarray(mean_new) - np.asarray(mean_old)) ** 2 / variance


def _toy2_discrepancy(config: Toy2Config, action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=float)
    return config.discrepancy_sine_amplitude * np.sin(4.0 * np.pi * a) + config.discrepancy_cubic * (a - 0.5) ** 3


def toy2_parameter_derivative(twin: Toy2DigitalTwin, action: np.ndarray) -> np.ndarray:
    cfg = twin.config
    a = np.asarray(action, dtype=float)
    return cfg.c_right * twin.basis(a, cfg.a_right) + cfg.c_diag * twin.basis(a, cfg.a_diag)


def toy2_information(twin: Toy2DigitalTwin, action: np.ndarray, noise_variance: float | None = None) -> np.ndarray:
    cfg = twin.config
    variance = cfg.sigma_y**2 if noise_variance is None else noise_variance
    return toy2_parameter_derivative(twin, action) ** 2 / max(float(variance), 1e-12)


def toy2_expected_net_reward(
    twin: Toy2DigitalTwin,
    action: np.ndarray,
    theta: float,
    previous_action: np.ndarray | float,
    include_switching: bool = True,
) -> np.ndarray:
    cfg = twin.config
    a = np.asarray(action, dtype=float)
    prev = np.asarray(previous_action, dtype=float)
    reward = twin.response(a, theta) + _toy2_discrepancy(cfg, a) - cfg.lambda_energy * a * a
    if include_switching:
        reward = reward - cfg.lambda_switch * (a - prev) ** 2
    return np.asarray(reward, dtype=float)


def toy2_diagnostic_conditions(
    config: Toy2Config = Toy2Config(),
    theta_grid: np.ndarray | None = None,
    action_grid: np.ndarray | None = None,
    epsilon_prod: float = 1e-3,
) -> Toy2DiagnosticConditions:
    """Check Toy 2 diagnostic-action conditions required by the benchmark spec.

    The production optimality test excludes switching cost because G5 is about
    whether the diagnostic action is an argmax for any single sampled production
    model.  The bridge inequality uses the configured square switching geometry.
    """

    twin = Toy2DigitalTwin(config)
    theta_grid = np.linspace(0.0, 1.0, 101) if theta_grid is None else np.asarray(theta_grid, dtype=float)
    action_grid = (
        np.linspace(config.action_low, config.action_high, 2001)
        if action_grid is None
        else np.asarray(action_grid, dtype=float)
    )

    diag_reward_gaps = []
    for theta in theta_grid:
        rewards = toy2_expected_net_reward(
            twin,
            action_grid,
            float(theta),
            previous_action=config.a_left,
            include_switching=False,
        )
        diag_reward = float(
            toy2_expected_net_reward(
                twin,
                np.array([config.a_diag]),
                float(theta),
                previous_action=config.a_left,
                include_switching=False,
            )[0]
        )
        diag_reward_gaps.append(float(np.max(rewards) - diag_reward))

    diagnostic_information = float(toy2_information(twin, np.array([config.a_diag]))[0])
    left_information = float(toy2_information(twin, np.array([config.a_left]))[0])
    right_information = float(toy2_information(twin, np.array([config.a_right]))[0])
    bridge_cost = config.lambda_switch * ((config.a_diag - config.a_left) ** 2 + (config.a_right - config.a_diag) ** 2)
    direct_cost = config.lambda_switch * (config.a_right - config.a_left) ** 2
    production_gap_at_diag = float(np.min(diag_reward_gaps))

    return Toy2DiagnosticConditions(
        production_gap_at_diag=production_gap_at_diag,
        diagnostic_information=diagnostic_information,
        left_information=left_information,
        right_information=right_information,
        bridge_cost=float(bridge_cost),
        direct_cost=float(direct_cost),
        a_diag_not_production_optimal=production_gap_at_diag >= epsilon_prod,
        information_higher_at_a_diag=diagnostic_information > max(left_information, right_information),
        bridge_switching_inequality=bridge_cost < direct_cost,
    )


def screen_toy2(
    twin,
    config,
    theta_old: float | None = None,
    theta_new: float | None = None,
    num_grid: int = 401,
    previous_action: float | None = None,
) -> GeometryReport:
    theta_old = config.theta_initial if theta_old is None else theta_old
    theta_new = config.theta_after_jump if theta_new is None else theta_new
    previous_action = config.a_left if previous_action is None else previous_action
    a = np.linspace(config.action_low, config.action_high, num_grid)
    zero_state = np.zeros((num_grid, 1))
    X = np.column_stack([zero_state[:, 0], a])
    old_mean = twin.batch_step(X, theta_old)[:, 0]
    new_mean = twin.batch_step(X, theta_new)[:, 0]
    reward_old = toy2_expected_net_reward(twin, a, theta_old, previous_action)
    reward_new = toy2_expected_net_reward(twin, a, theta_new, previous_action)
    sensitivity = toy2_parameter_derivative(twin, a) ** 2
    kl = gaussian_kl_same_variance(new_mean, old_mean, config.sigma_y**2)
    switch = config.lambda_switch * (a - previous_action) ** 2
    old_opt = float(a[int(np.argmax(reward_old))])
    new_opt = float(a[int(np.argmax(reward_new))])
    diag = float(a[int(np.argmax(sensitivity))])
    diag_idx = int(np.argmin(np.abs(a - config.a_diag)))
    left_idx = int(np.argmin(np.abs(a - config.a_left)))
    right_idx = int(np.argmin(np.abs(a - config.a_right)))
    prod_gap = min(float(np.max(reward_old) - reward_old[diag_idx]), float(np.max(reward_new) - reward_new[diag_idx]))
    return GeometryReport(
        action_grid=a,
        reward_old=reward_old,
        reward_new=reward_new,
        sensitivity=sensitivity,
        predictive_kl_old_new=kl,
        switching_cost_from_previous=switch,
        old_opt_action=old_opt,
        new_opt_action=new_opt,
        diagnostic_action=diag,
        passes_g1=abs(old_opt - new_opt) > 0.05 and reward_new[int(np.argmin(np.abs(a - old_opt)))] < np.max(reward_new),
        passes_g2=kl[left_idx] < 0.25 * max(float(np.max(kl)), 1e-12),
        passes_g3=sensitivity[diag_idx] > max(sensitivity[left_idx], sensitivity[right_idx]),
        passes_g5=prod_gap > 0.0,
    )


def grid_optimum(action_grid: np.ndarray, objective: Callable[[float], float]) -> tuple[float, float]:
    values = np.asarray([objective(float(a)) for a in action_grid], dtype=float)
    idx = int(np.argmax(values))
    return float(action_grid[idx]), float(values[idx])


def toy1_geometry_rows(
    config: Toy1Config = Toy1Config(),
    theta_old: float | None = None,
    theta_new: float = 1.25,
    beta: float | None = None,
    num_state: int = 101,
    num_action: int = 101,
    state_low: float = -1.0,
    state_high: float = 1.0,
    previous_action: float = 0.0,
    reference: float | None = None,
) -> list[dict[str, float | str]]:
    """Return Toy 1 geometry-screening rows on an x/a grid."""

    theta_old = config.theta_initial if theta_old is None else theta_old
    beta = config.beta_initial if beta is None else beta
    reference = 0.0 if reference is None else reference
    twin = Toy1DigitalTwin()
    states = np.linspace(state_low, state_high, num_state)
    actions = np.linspace(config.action_low, config.action_high, num_action)
    rows: list[dict[str, float | str]] = []
    variance = max(config.sigma_w**2, 1e-12)
    for x in states:
        for a in actions:
            X = np.array([[x, a]], dtype=float)
            old_mean = float(twin.batch_step(X, theta_old)[0, 0])
            new_mean = float(twin.batch_step(X, theta_new)[0, 0])
            discrepancy = beta * np.tanh(2.0 * x) + config.kappa_delta * a * abs(a)
            expected_reward = -config.q_x * (x - reference) ** 2 - config.lambda_energy * a * a - config.lambda_switch * (a - previous_action) ** 2
            derivative = x
            information = derivative * derivative / variance
            rows.append(
                {
                    "environment": "toy1",
                    "state": float(x),
                    "action": float(a),
                    "previous_action": float(previous_action),
                    "theta_old": float(theta_old),
                    "theta_new": float(theta_new),
                    "expected_next_old": old_mean + discrepancy,
                    "expected_next_new": new_mean + discrepancy,
                    "expected_reward_old": float(expected_reward),
                    "expected_reward_new": float(expected_reward),
                    "parameter_sensitivity": float(derivative * derivative),
                    "fisher_proxy": float(information),
                    "variance_reduction_proxy": float(information),
                    "predictive_kl_old_new": float(gaussian_kl_same_variance(np.array([new_mean]), np.array([old_mean]), variance)[0]),
                    "switching_cost": float(config.lambda_switch * (a - previous_action) ** 2),
                }
            )
    return rows


def toy2_geometry_rows(
    config: Toy2Config = Toy2Config(),
    theta_old: float | None = None,
    theta_new: float | None = None,
    num_state: int = 101,
    num_action: int = 101,
) -> list[dict[str, float | str]]:
    """Return Toy 2 geometry-screening rows on previous-action/action grid."""

    theta_old = config.theta_initial if theta_old is None else theta_old
    theta_new = config.theta_after_jump if theta_new is None else theta_new
    twin = Toy2DigitalTwin(config)
    previous_actions = np.linspace(config.action_low, config.action_high, num_state)
    actions = np.linspace(config.action_low, config.action_high, num_action)
    rows: list[dict[str, float | str]] = []
    variance = max(config.sigma_y**2, 1e-12)
    for previous_action in previous_actions:
        for action in actions:
            old_mean = float(twin.response(action, theta_old))
            new_mean = float(twin.response(action, theta_new))
            discrepancy = float(_toy2_discrepancy(config, np.array([action]))[0])
            derivative = float(toy2_parameter_derivative(twin, np.array([action]))[0])
            information = derivative * derivative / variance
            switch = config.lambda_switch * (action - previous_action) ** 2
            rows.append(
                {
                    "environment": "toy2",
                    "state": float(previous_action),
                    "action": float(action),
                    "previous_action": float(previous_action),
                    "theta_old": float(theta_old),
                    "theta_new": float(theta_new),
                    "expected_response_old": old_mean + discrepancy,
                    "expected_response_new": new_mean + discrepancy,
                    "expected_reward_old": float(old_mean + discrepancy - config.lambda_energy * action * action - switch),
                    "expected_reward_new": float(new_mean + discrepancy - config.lambda_energy * action * action - switch),
                    "parameter_sensitivity": float(derivative * derivative),
                    "fisher_proxy": float(information),
                    "variance_reduction_proxy": float(information),
                    "predictive_kl_old_new": float(gaussian_kl_same_variance(np.array([new_mean]), np.array([old_mean]), variance)[0]),
                    "switching_cost": float(switch),
                }
            )
    return rows


def write_geometry_csv(rows: Iterable[dict[str, float | str]], output_path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty geometry CSV.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_geometry_csv(
    environment: str,
    output_path: str | Path,
    num_state: int = 101,
    num_action: int = 101,
    theta_old: float | None = None,
    theta_new: float | None = None,
) -> Path:
    """Generate the required geometry-screening CSV for Toy 1 or Toy 2."""

    env = environment.lower()
    if env == "toy1":
        rows = toy1_geometry_rows(num_state=num_state, num_action=num_action, theta_old=theta_old, theta_new=1.25 if theta_new is None else theta_new)
    elif env == "toy2":
        rows = toy2_geometry_rows(num_state=num_state, num_action=num_action, theta_old=theta_old, theta_new=theta_new)
    else:
        raise ValueError("environment must be 'toy1' or 'toy2'.")
    write_geometry_csv(rows, output_path)
    return Path(output_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Toy1/Toy2 geometry screening CSVs.")
    parser.add_argument("--environment", choices=("toy1", "toy2", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--num-state", type=int, default=101)
    parser.add_argument("--num-action", type=int, default=101)
    args = parser.parse_args(argv)

    environments = ("toy1", "toy2") if args.environment == "both" else (args.environment,)
    for environment in environments:
        path = args.out_dir / f"{environment}_geometry_screening.csv"
        generate_geometry_csv(environment, path, num_state=args.num_state, num_action=args.num_action)
        print(path)


if __name__ == "__main__":
    main()
