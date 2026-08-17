"""Write the scalar static two-step sanity curve to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmarks.scalar_dual.kh_reference import compute_static_two_step_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-grid-size", type=int, default=121)
    parser.add_argument("--out", type=Path, default=Path("reports/tables/scalar_static_curve.csv"))
    args = parser.parse_args()
    curve = compute_static_two_step_curve(args.action_grid_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["u0", "ce_cost", "approx_dual_cost", "grid_bayes_cost"])
        writer.writeheader()
        for u0, ce, ad, bayes in zip(curve.u0, curve.ce_cost, curve.approx_dual_cost, curve.grid_bayes_cost):
            writer.writerow({"u0": u0, "ce_cost": ce, "approx_dual_cost": ad, "grid_bayes_cost": bayes})
    print(f"wrote {len(curve.u0)} rows to {args.out}")
    print(f"CE min u0={curve.u0[curve.ce_cost.argmin()]:.3f}")
    print(f"approx dual min u0={curve.u0[curve.approx_dual_cost.argmin()]:.3f}")
    print(f"grid bayes min u0={curve.u0[curve.grid_bayes_cost.argmin()]:.3f}")


if __name__ == "__main__":
    main()
