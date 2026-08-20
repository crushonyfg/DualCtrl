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
from .geometry import generate_geometry_csv, toy2_diagnostic_conditions, toy2_geometry_rows
from .planners import CEPlanner, CEMConfig, PosteriorSamplingPlanner

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
    "generate_geometry_csv",
    "toy2_diagnostic_conditions",
    "toy2_geometry_rows",
    "CEPlanner",
    "CEMConfig",
    "PosteriorSamplingPlanner",
]
