from __future__ import annotations

import csv
import json
import subprocess
import sys


DIAGNOSTICS = {"arcari_l0_passive_exploitation", "nominal_mpc_ce"}
MAIN_BASELINES = {"kh_dual_control", "arcari_dual_smpc", "tv_gp_lcb", "oracle_trend"}


def _baselines(path):
    with path.open() as f:
        return {row["baseline"] for row in csv.DictReader(f)}


def test_scalar_diagnostics_are_written_separately(tmp_path):
    out_dir = tmp_path / "scalar"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run_official_scalar",
            "--out-dir",
            str(out_dir),
            "--horizon",
            "2",
            "--n-seeds",
            "1",
            "--planning-horizon",
            "2",
            "--action-grid-size",
            "3",
            "--smpc-dual-horizon",
            "1",
            "--smpc-scenarios",
            "1",
            "--include-diagnostics",
        ],
        check=True,
    )

    assert _baselines(out_dir / "scalar_main_summary.csv") == MAIN_BASELINES
    assert _baselines(out_dir / "scalar_diagnostic_summary.csv") == DIAGNOSTICS
    assert not DIAGNOSTICS & _baselines(out_dir / "scalar_main_raw.csv")


def test_cartpole_diagnostics_are_written_separately(tmp_path):
    out_dir = tmp_path / "cartpole"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run_official_cartpole",
            "--out-dir",
            str(out_dir),
            "--horizon",
            "2",
            "--n-seeds",
            "1",
            "--planning-horizon",
            "2",
            "--action-grid-size",
            "3",
            "--smpc-dual-horizon",
            "1",
            "--smpc-scenarios",
            "1",
            "--include-diagnostics",
            "--n-initial-calibration",
            "1",
            "--initial-calibration-policy",
            "zero",
        ],
        check=True,
    )

    assert _baselines(out_dir / "cartpole_main_summary.csv") == MAIN_BASELINES
    assert _baselines(out_dir / "cartpole_diagnostic_summary.csv") == DIAGNOSTICS
    assert not DIAGNOSTICS & _baselines(out_dir / "cartpole_main_raw.csv")
    with (out_dir / "config.json").open() as f:
        config = json.load(f)
    assert config["n_initial_calibration"] == 1
    assert config["initial_calibration_policy"] == "zero"
