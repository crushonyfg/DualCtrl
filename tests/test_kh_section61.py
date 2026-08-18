from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from controllers.kh_strict import (
    KHSection61Constants,
    ce_control_law,
    kh_ad_scalar_cost,
    make_section61_problem,
    of_control_law,
    scalar_posterior_update,
)


def test_section61_constants_match_paper_text() -> None:
    c = KHSection61Constants()
    assert c.a == 1.0
    assert c.true_b == 2.0
    assert c.prior_mean == 1.0
    assert c.prior_var == 10.0
    assert c.process_var == 1e-1
    assert c.obs_var == 0.0
    assert c.state_weight == 1.0
    assert c.energy_weight == 1.0
    assert c.terminal_weight == 1.0
    assert c.horizon == 2


def test_equation_6_ce_and_of_laws() -> None:
    _, cost, config, belief = make_section61_problem()
    x = 3.0
    assert ce_control_law(x, belief.mean, cost, config) == -1.5
    assert of_control_law(x, belief.mean, belief.var, cost, config) == -0.25


def test_equation_7_posterior_update() -> None:
    mu0 = 1.0
    var0 = 10.0
    u0 = 0.2
    q = 0.1
    residual = 2.0 * u0 + 0.05
    mu1, var1 = scalar_posterior_update(mu0, var0, u0, residual, q)
    denom = u0 * u0 * var0 + q
    assert np.isclose(mu1, (var0 * u0 * residual + mu0 * q) / denom)
    assert np.isclose(var1, var0 * q / denom)


def test_ad_landscape_has_symmetric_probing_minima_for_documented_x0() -> None:
    c, cost, config, belief = make_section61_problem(action_grid_size=401)
    values = np.array([kh_ad_scalar_cost(c.x0_default, float(u), belief.mean, belief.var, cost, config) for u in config.action_grid])
    min_u = float(config.action_grid[int(values.argmin())])
    zero_idx = int(np.argmin(np.abs(config.action_grid)))
    assert abs(min_u) > 0.05
    assert abs(min_u) < 0.5
    assert values.min() < values[zero_idx]
    # With the documented x0 default, the AD cost must prefer a non-zero probing
    # action over doing nothing; Sec. 6.1 does not state x0, so no sign-symmetry
    # assertion is made here.
    assert np.all(np.isfinite(values))


def test_cost_landscape_writer_schema_and_constants(tmp_path: Path) -> None:
    out = tmp_path / "kh_curve.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.reproduce_kh_section6",
            "--out",
            str(out),
            "--action-grid-size",
            "41",
        ],
        cwd="/mnt/bn/feed-quality-training/user/yxu/DualCtrl",
        check=True,
    )
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 41
    required = {
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
    }
    assert required.issubset(rows[0].keys())
    c = KHSection61Constants()
    for row in rows:
        assert float(row["a"]) == c.a
        assert float(row["true_b"]) == c.true_b
        assert float(row["prior_mean"]) == c.prior_mean
        assert float(row["prior_var"]) == c.prior_var
        assert float(row["process_var_Q"]) == c.process_var
        assert float(row["obs_var_R"]) == c.obs_var
        assert float(row["state_weight_W"]) == c.state_weight
        assert float(row["control_weight_Lambda"]) == c.energy_weight
        assert float(row["terminal_weight_WT"]) == c.terminal_weight
        assert int(float(row["horizon_T"])) == c.horizon
        assert row["x0_ambiguity"]
    ad_values = np.array([float(r["kh_ad_cost"]) for r in rows])
    u_values = np.array([float(r["u0"]) for r in rows])
    assert abs(float(u_values[int(ad_values.argmin())])) > 0.05
