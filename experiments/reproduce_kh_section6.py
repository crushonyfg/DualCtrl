"""Write Klenske-Hennig Sec. 6.1 scalar cost landscapes.

This is a strict reproduction harness for the paper's scalar T=2 setting.  It
uses only the equations in Sec. 3.1, Sec. 4, and Sec. 6.1:

* CE reference: Sec. 3.1 certainty-equivalent nominal cost.
* OF reference: Sec. 3.1 cautious one-step expected-cost controller.
* approximate dual: Sec. 4 nominal trajectory plus quadratic uncertainty cost.
* sampling reference: Monte-Carlo evaluation of Eq. (9), if requested.

No BEB or other invented baselines are written.  The paper does not state x0 in
Sec. 6.1; the default is x0=1.0 and the CSV records this ambiguity.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from controllers.kh_strict import (
    ce_scalar_cost,
    exact_sampling_dual_cost,
    kh_ad_scalar_cost,
    make_section61_problem,
    of_scalar_cost,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/tables/official/kh_section6_curve.csv"))
    parser.add_argument("--action-grid-size", type=int, default=401)
    parser.add_argument("--action-low", type=float, default=-1.0)
    parser.add_argument("--action-high", type=float, default=1.0)
    parser.add_argument("--x0", type=float, default=None, help="initial state; Sec. 6.1 does not state it, default uses documented reproduction value")
    parser.add_argument("--sampling-samples", type=int, default=0, help="if >0, add Monte-Carlo Eq. (9) sampling reference")
    parser.add_argument("--sampling-seed", type=int, default=0)
    args = parser.parse_args()

    constants, cost, config, belief = make_section61_problem(
        action_grid_size=args.action_grid_size,
        action_low=args.action_low,
        action_high=args.action_high,
    )
    x0 = constants.x0_default if args.x0 is None else float(args.x0)

    fieldnames = [
        "u0",
        "ce_cost",
        "of_cost",
        "kh_ad_cost",
        "x0",
        "a",
        "true_b",
        "prior_mean",
        "prior_var",
        "process_var_Q",
        "obs_var_R",
        "state_weight_W",
        "control_weight_Lambda",
        "terminal_weight_WT",
        "horizon_T",
        "x0_ambiguity",
    ]
    if args.sampling_samples > 0:
        fieldnames.append("sampling_dual_cost")

    rows = []
    for u0 in config.action_grid:
        u0 = float(u0)
        row = {
            "u0": u0,
            "ce_cost": ce_scalar_cost(x0, u0, belief.mean, cost, config),
            "of_cost": of_scalar_cost(x0, u0, belief.mean, belief.var, cost, config),
            "kh_ad_cost": kh_ad_scalar_cost(x0, u0, belief.mean, belief.var, cost, config),
            "x0": x0,
            "a": constants.a,
            "true_b": constants.true_b,
            "prior_mean": constants.prior_mean,
            "prior_var": constants.prior_var,
            "process_var_Q": constants.process_var,
            "obs_var_R": constants.obs_var,
            "state_weight_W": constants.state_weight,
            "control_weight_Lambda": constants.energy_weight,
            "terminal_weight_WT": constants.terminal_weight,
            "horizon_T": constants.horizon,
            "x0_ambiguity": "Sec. 6.1 states all listed constants except x0; default x0=1.0 is documented in code.",
        }
        if args.sampling_samples > 0:
            row["sampling_dual_cost"] = exact_sampling_dual_cost(
                x0,
                u0,
                belief.mean,
                belief.var,
                constants.true_b,
                cost,
                config,
                num_samples=args.sampling_samples,
                seed=args.sampling_seed,
            )
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ce_min = min(rows, key=lambda r: r["ce_cost"])
    of_min = min(rows, key=lambda r: r["of_cost"])
    ad_min = min(rows, key=lambda r: r["kh_ad_cost"])
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"CE min u0={ce_min['u0']:.4f}, cost={ce_min['ce_cost']:.6f}")
    print(f"OF min u0={of_min['u0']:.4f}, cost={of_min['of_cost']:.6f}")
    print(f"KH-AD min u0={ad_min['u0']:.4f}, cost={ad_min['kh_ad_cost']:.6f}")
    if args.sampling_samples > 0:
        sampling_min = min(rows, key=lambda r: r["sampling_dual_cost"])
        print(f"sampling Eq. (9) min u0={sampling_min['u0']:.4f}, cost={sampling_min['sampling_dual_cost']:.6f}")


if __name__ == "__main__":
    main()
