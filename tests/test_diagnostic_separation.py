import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.make_baseline_only_report import BASELINES, enrich_scalar


def test_diagnostic_baselines_are_not_main_three_baselines():
    assert BASELINES == {"kh_dual_control", "arcari_dual_smpc", "tv_gp_lcb"}
    assert "arcari_l0_passive_exploitation" not in BASELINES
    assert "nominal_mpc_ce" not in BASELINES


def test_enrich_scalar_filters_diagnostics_from_baseline_report():
    common = {
        "environment": "scalar",
        "twin_gap": "no_gap",
        "regime": "static",
        "seed": "0",
        "state_cost": "1.0",
        "energy_cost": "0.1",
        "switch_cost": "0.0",
        "nonsmooth_switch_cost": "0.0",
        "terminal_cost": "0.2",
        "total_cost": "1.3",
        "mean_abs_action": "0.5",
        "frac_zero_action": "0.0",
        "action_changes": "1",
        "physical_transitions": "2",
        "observed_transitions": "2",
        "observation_interval": "1",
    }
    rows = [
        {**common, "baseline": "kh_dual_control"},
        {**common, "baseline": "arcari_l0_passive_exploitation"},
        {**common, "baseline": "nominal_mpc_ce"},
    ]
    enriched = enrich_scalar(rows)
    assert [row["baseline"] for row in enriched] == ["kh_dual_control"]
