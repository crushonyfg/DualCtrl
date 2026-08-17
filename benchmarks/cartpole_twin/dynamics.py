"""CartPole dynamics for the evolving actuator-gain twin benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPoleParams:
    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    half_length: float = 0.5
    dt: float = 0.02
    force_mag: float = 10.0
    actuator_lag_alpha: float = 1.0
    coulomb_friction: float = 0.0


def cartpole_step(state: np.ndarray, action: float, theta: float, params: CartPoleParams, prev_force: float = 0.0) -> tuple[np.ndarray, float]:
    x, x_dot, angle, angle_dot = map(float, state)
    commanded_force = float(np.clip(action, -1.0, 1.0)) * params.force_mag
    target_force = theta * commanded_force
    force = (1.0 - params.actuator_lag_alpha) * prev_force + params.actuator_lag_alpha * target_force

    total_mass = params.mass_cart + params.mass_pole
    polemass_length = params.mass_pole * params.half_length
    costheta = np.cos(angle)
    sintheta = np.sin(angle)

    temp = (force + polemass_length * angle_dot * angle_dot * sintheta) / total_mass
    theta_acc = (params.gravity * sintheta - costheta * temp) / (
        params.half_length * (4.0 / 3.0 - params.mass_pole * costheta * costheta / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * costheta / total_mass
    if params.coulomb_friction:
        x_acc -= params.coulomb_friction * np.sign(x_dot if abs(x_dot) > 1e-9 else force)

    x = x + params.dt * x_dot
    x_dot = x_dot + params.dt * x_acc
    angle = angle + params.dt * angle_dot
    angle_dot = angle_dot + params.dt * theta_acc
    return np.array([x, x_dot, angle, angle_dot], dtype=float), float(force)


def reference_position(t: int, segment: int = 125) -> float:
    schedule = [0.0, 0.8, 0.0, -0.8]
    idx = (t // segment) % len(schedule)
    next_idx = (idx + 1) % len(schedule)
    phase = (t % segment) / float(segment)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    return (1.0 - smooth) * schedule[idx] + smooth * schedule[next_idx]
