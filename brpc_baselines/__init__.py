"""Lightweight BRPC baseline suite from benchmark_brpc_baseline_implementation_spec.md.

This package is intentionally isolated from the older KH/Arcari/TVGP baseline code.
"""

from .toy_envs import (
    RewardBreakdown,
    Toy1Config,
    Toy1DigitalTwin,
    Toy1PhysicalEnv,
    Toy2Config,
    Toy2DigitalTwin,
    Toy2PhysicalEnv,
)
from .brpc import BRPCConfig, BRPCState, FixedSupportBRPC
from .bocpd_brpc import BOCPDBRPC, BOCPDConfig
from .calibration_validation import CalibrationValidationConfig, RESULT_SCHEMA, run_calibration_validation
from .geometry import generate_geometry_csv, toy2_diagnostic_conditions, toy2_geometry_rows, toy2_operating_reward, toy2_one_step_net_reward
from .planners import CEPlanner, CEMConfig, GridDPConfig, PosteriorSamplingPlanner, Toy2GridDPCEPlanner, Toy2GridDPOraclePlanner, Toy2GridDPPSPlanner, ToyCurrentDynamicsOraclePlanner, ToyFutureRegimeOraclePlanner, stage_reward

__all__ = [
    "RewardBreakdown",
    "Toy1Config",
    "Toy1DigitalTwin",
    "Toy1PhysicalEnv",
    "Toy2Config",
    "Toy2DigitalTwin",
    "Toy2PhysicalEnv",
    "BRPCConfig",
    "BRPCState",
    "FixedSupportBRPC",
    "BOCPDBRPC",
    "BOCPDConfig",
    "CalibrationValidationConfig",
    "RESULT_SCHEMA",
    "run_calibration_validation",
    "generate_geometry_csv",
    "toy2_diagnostic_conditions",
    "toy2_geometry_rows",
    "toy2_operating_reward",
    "toy2_one_step_net_reward",
    "CEPlanner",
    "CEMConfig",
    "GridDPConfig",
    "PosteriorSamplingPlanner",
    "Toy2GridDPCEPlanner",
    "Toy2GridDPOraclePlanner",
    "Toy2GridDPPSPlanner",
    "ToyCurrentDynamicsOraclePlanner",
    "ToyFutureRegimeOraclePlanner",
    "stage_reward",
]
