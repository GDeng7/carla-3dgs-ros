#!/usr/bin/env python3
"""Entry point: replay a 3DGS/NuRec scene in CARLA."""

from carla_3dgs_ros.replay import run_from_cli
from carla_3dgs_ros.utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    raise SystemExit(run_from_cli())