"""Deterministic bounded action optimizers used by continuous-action baselines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import warnings

import numpy as np

try:  # pragma: no cover - exercised implicitly when scipy is installed.
    from scipy.optimize import minimize, minimize_scalar

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback is covered by helper structure.
    minimize = None
    minimize_scalar = None
    SCIPY_AVAILABLE = False


def minimize_scalar_action(
    objective: Callable[[float], float],
    low: float = -1.0,
    high: float = 1.0,
    grid_size: int = 81,
    maxiter: int = 100,
    xatol: float = 1e-4,
) -> tuple[float, float]:
    """Minimize a bounded scalar action objective and return ``(action, value)``."""

    low = float(low)
    high = float(high)
    if high < low:
        raise ValueError("high must be >= low")
    if high == low:
        value = float(objective(low))
        return low, value

    candidates = [(low, float(objective(low))), (high, float(objective(high)))]
    if SCIPY_AVAILABLE:
        result = minimize_scalar(  # type: ignore[misc]
            lambda u: float(objective(float(u))),
            bounds=(low, high),
            method="bounded",
            options={"xatol": float(xatol), "maxiter": int(maxiter)},
        )
        if getattr(result, "success", False) and np.isfinite(result.fun):
            u = float(np.clip(result.x, low, high))
            candidates.append((u, float(objective(u))))
    else:
        warnings.warn(
            "scipy.optimize is unavailable; using deterministic dense scalar action optimizer fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        for u in np.linspace(low, high, max(2, int(grid_size))):
            candidates.append((float(u), float(objective(float(u)))))
    return min(candidates, key=lambda item: item[1])


def minimize_vector_actions(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    x0: Sequence[float] | None = None,
    grid_size: int = 41,
    maxiter: int = 100,
    xatol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Minimize a bounded vector action objective and return ``(actions, value)``."""

    bounds = [(float(low), float(high)) for low, high in bounds]
    dim = len(bounds)
    if dim == 0:
        x = np.zeros(0, dtype=float)
        return x, float(objective(x))
    for low, high in bounds:
        if high < low:
            raise ValueError("all bounds must satisfy high >= low")

    if x0 is None:
        current = np.array([(low + high) * 0.5 for low, high in bounds], dtype=float)
    else:
        current = np.asarray(x0, dtype=float).reshape(dim)
        current = np.array([np.clip(current[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)
    best_x = current.copy()
    best_value = float(objective(best_x))

    if SCIPY_AVAILABLE:
        result = minimize(  # type: ignore[misc]
            lambda values: float(objective(np.asarray(values, dtype=float))),
            best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": float(xatol)},
        )
        if getattr(result, "success", False) and np.isfinite(result.fun):
            candidate = np.asarray(result.x, dtype=float)
            candidate = np.array([np.clip(candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)
            value = float(objective(candidate))
            if value < best_value:
                best_x, best_value = candidate, value
    else:
        warnings.warn(
            "scipy.optimize is unavailable; using deterministic dense coordinate action optimizer fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        # Deterministic dense coordinate fallback: sweep each action coordinate over
        # a fixed dense grid until no coordinate improves or maxiter is reached.
        grids = [np.linspace(low, high, max(2, int(grid_size))) for low, high in bounds]
        for _ in range(max(1, int(maxiter))):
            improved = False
            for i, grid in enumerate(grids):
                coord_best = best_x.copy()
                coord_value = best_value
                for value_i in grid:
                    candidate = best_x.copy()
                    candidate[i] = float(value_i)
                    value = float(objective(candidate))
                    if value < coord_value:
                        coord_best, coord_value = candidate, value
                if coord_value + float(xatol) < best_value:
                    best_x, best_value = coord_best, coord_value
                    improved = True
            if not improved:
                break
    return best_x, best_value
