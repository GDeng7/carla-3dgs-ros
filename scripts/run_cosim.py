#!/usr/bin/env python3
"""Entry point: co-simulation mode (CARLA physics + 3DGS renderer + ROS2)."""

from carla_3dgs_ros.cosim.orchestrator import run_from_cli
from carla_3dgs_ros.utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    raise SystemExit(run_from_cli())