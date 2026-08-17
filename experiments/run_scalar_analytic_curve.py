"""Write analytic scalar two-step B0 sanity curves to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmarks.scalar_dual.analytic_reference import AnalyticScalarConfig, compute_analytic_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-grid-size", type=int, default=401)
    parser.add_argument("--quadrature-points", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("reports/tables/scalar_analytic_curve.csv"))
    args = parser.parse_args()

    config = AnalyticScalarConfig(action_grid_size=args.action_grid_size, quadrature_points=args.quadrature_points)
    curve = compute_analytic_curve(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["u0", "ce_cost", "cautious_cost", "bayes_dual_cost"])
        writer.writeheader()
        for u0, ce, cautious, bayes in zip(curve.u0, curve.ce_cost, curve.cautious_cost, curve.bayes_dual_cost):
            writer.writerow({"u0": u0, "ce_cost": ce, "cautious_cost": cautious, "bayes_dual_cost": bayes})
    print(f"wrote {len(curve.u0)} rows to {args.out}")
    print(f"CE min u0={curve.u0[curve.ce_cost.argmin()]:.4f}, cost={curve.ce_cost.min():.4f}")
    print(f"cautious min u0={curve.u0[curve.cautious_cost.argmin()]:.4f}, cost={curve.cautious_cost.min():.4f}")
    print(f"Bayes dual min u0={curve.u0[curve.bayes_dual_cost.argmin()]:.4f}, cost={curve.bayes_dual_cost.min():.4f}")


if __name__ == "__main__":
    main()
