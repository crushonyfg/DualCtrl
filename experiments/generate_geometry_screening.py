"""CLI wrapper for Toy1/Toy2 geometry-screening CSV generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brpc_baselines.geometry import main


if __name__ == "__main__":
    main()
