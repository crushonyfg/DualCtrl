"""Reproduce the scalar sanity setting for Klenske-Hennig style dual control.

This script is a reproduction harness, not a new baseline. It writes cost
landscapes for the paper's scalar linear-Gaussian setting using the same
Gaussian posterior update and approximate dual planning logic used by the
official KH controller.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers.official import OfficialScalarConfig, _kh_scalar_action, _scalar_exploitation_value


def forced_kh_cost(x: float, prev_u: float, u0: float, belief: GaussianBelief, cost: ScalarCost, config: OfficialScalarConfig) -> float:
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(config.kh_quadrature_points)
    immediate = cost.stage(x, u0, prev_u).total
    pred_mean = x + belief.mean * u0
    pred_var = max(config.process_var + belief.var * u0 * u0, 1e-12)
    branch = 0.0
    for node, weight in zip(nodes, weights):
        x_next = pred_mean + np.sqrt(2.0 * pred_var) * float(node)
        fantasy = belief.copy()
        fantasy.update(x, u0, float(x_next), config.process_var)
        branch += float(weight) * _scalar_exploitation_value(float(x_next), u0, fantasy.mean, fantasy.var, cost, config, config.horizon - 1)
    return immediate + branch / np.sqrt(np.pi)


def forced_ce_cost(x: float, prev_u: float, u0: float, belief: GaussianBelief, cost: ScalarCost, config: OfficialScalarConfig) -> float:
    immediate = cost.stage(x, u0, prev_u).total
    x_next = x + belief.mean * u0
    return immediate + _scalar_exploitation_value(x_next, u0, belief.mean, 0.0, cost, config, config.horizon - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/tables/official/kh_section6_curve.csv"))
    parser.add_argument("--action-grid-size", type=int, default=401)
    parser.add_argument("--process-var", type=float, default=0.1)
    parser.add_argument("--prior-mean", type=float, default=1.0)
    parser.add_argument("--prior-var", type=float, default=10.0)
    parser.add_argument("--x0", type=float, default=1.0)
    args = parser.parse_args()

    cost = ScalarCost(ScalarCostConfig(energy_weight=1.0, switch_weight=0.0, terminal_weight=1.0))
    config = OfficialScalarConfig(
        horizon=2,
        action_grid_size=args.action_grid_size,
        process_var=args.process_var,
        action_low=-3.0,
        action_high=3.0,
        kh_quadrature_points=21,
    )
    belief = GaussianBelief(mean=args.prior_mean, var=args.prior_var)
    rows = []
    for u0 in config.action_grid:
        u0 = float(u0)
        rows.append(
            {
                "u0": u0,
                "ce_cost": forced_ce_cost(args.x0, 0.0, u0, belief, cost, config),
                "kh_dual_cost": forced_kh_cost(args.x0, 0.0, u0, belief, cost, config),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ce_min = min(rows, key=lambda r: r["ce_cost"])
    kh_min = min(rows, key=lambda r: r["kh_dual_cost"])
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"CE min u0={ce_min['u0']:.4f}, cost={ce_min['ce_cost']:.4f}")
    print(f"KH-dual min u0={kh_min['u0']:.4f}, cost={kh_min['kh_dual_cost']:.4f}")


if __name__ == "__main__":
    main()
