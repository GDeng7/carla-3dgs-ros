#!/usr/bin/env python3
"""Entry point: macOS co-simulation (no CARLA server required)."""

from carla_3dgs_ros.cosim.macos.orchestrator import run_from_cli
from carla_3dgs_ros.utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    raise SystemExit(run_from_cli())