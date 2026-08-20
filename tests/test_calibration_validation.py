import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brpc_baselines.calibration_validation import (  # noqa: E402
    ALGORITHMS,
    RESULT_SCHEMA,
    CalibrationValidationConfig,
    make_calibration_dataset,
    run_calibration_validation,
)


def test_calibration_validation_output_schema_and_identical_data_feeds():
    cfg = CalibrationValidationConfig(horizon=8, num_particles=8, seed=3, min_segment_length=3, max_experts=3)
    rows = run_calibration_validation(toys=("toy2",), scenarios=("stationary",), config=cfg)

    assert [row["algorithm"] for row in rows] == list(ALGORITHMS)
    for row in rows:
        assert tuple(row.keys()) == RESULT_SCHEMA
        assert row["toy"] == "toy2"
        assert row["scenario"] == "stationary"
        assert row["horizon"] == cfg.horizon
        assert np.isfinite(row["theta_rmse"])
        assert np.isfinite(row["response_rmse"])
        assert row["nlpd"] is None or np.isfinite(row["nlpd"])
        assert row["coverage_proxy_95"] is None or 0.0 <= row["coverage_proxy_95"] <= 1.0
        assert row["ess_min"] > 0.0
        assert np.isfinite(row["discrepancy_error_proxy_rmse"])
        assert set(row["expert_mass_summary"]) == {
            "num_experts_final",
            "anchor_start_time_final",
            "recent_change_probability_final",
            "recent_change_probability_max",
            "expert_entropy_mean",
        }

    records_a, _, _, actions_a = make_calibration_dataset("toy2", "stationary", cfg.horizon, seed=cfg.seed)
    records_b, _, _, actions_b = make_calibration_dataset("toy2", "stationary", cfg.horizon, seed=cfg.seed)
    assert np.allclose(actions_a, actions_b)
    assert np.allclose(
        np.asarray([record.calibration_input for record in records_a]),
        np.asarray([record.calibration_input for record in records_b]),
    )
    assert np.allclose(
        np.asarray([record.calibration_output for record in records_a]),
        np.asarray([record.calibration_output for record in records_b]),
    )


def test_no_change_false_alarm_control_on_small_deterministic_sequence():
    cfg = CalibrationValidationConfig(
        horizon=10,
        num_particles=8,
        seed=11,
        hazard=0.02,
        max_experts=4,
        min_segment_length=4,
        restart_margin_rho_B=10.0,
        restart_tolerance=3,
    )
    rows = run_calibration_validation(toys=("toy2",), scenarios=("no_change_false_alarm",), config=cfg)
    bbrpc = next(row for row in rows if row["algorithm"] == "bocpd_brpc")

    assert bbrpc["restart_count"] == 0
    assert bbrpc["restart_false_alarm_rate"] == 0.0
    assert bbrpc["restart_precision"] is None
    assert bbrpc["restart_recall"] is None
    assert bbrpc["expert_mass_summary"]["anchor_start_time_final"] == 0


def test_calibration_validation_cli_writes_out_csv(tmp_path):
    out_path = tmp_path / "nested" / "calibration_validation.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "brpc_baselines.calibration_validation",
            "--toy",
            "toy1",
            "--scenario",
            "stationary",
            "--horizon",
            "5",
            "--num-particles",
            "5",
            "--out",
            str(out_path),
        ],
        cwd=ROOT,
        check=True,
    )

    with out_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(ALGORITHMS)
    assert tuple(rows[0].keys()) == RESULT_SCHEMA
    assert {row["algorithm"] for row in rows} == set(ALGORITHMS)
    assert {row["toy"] for row in rows} == {"toy1"}
    assert {row["scenario"] for row in rows} == {"stationary"}
    assert {row["horizon"] for row in rows} == {"5"}
    for row in rows:
        assert set(json.loads(row["expert_mass_summary"])) == {
            "num_experts_final",
            "anchor_start_time_final",
            "recent_change_probability_final",
            "recent_change_probability_max",
            "expert_entropy_mean",
        }
