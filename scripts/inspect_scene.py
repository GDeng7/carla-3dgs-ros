#!/usr/bin/env python3
"""Inspect a NuRec USDZ scene without starting CARLA."""

import argparse
import json
from pathlib import Path

from carla_3dgs_ros.scene import load_scene, validate_scene_for_carla
from carla_3dgs_ros.utils import setup_logging


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Inspect a NuRec USDZ scene.")
    parser.add_argument("usdz", nargs="?", help="Path to .usdz file")
    parser.add_argument("--labels", default=None, help="Optional labels.json path")
    args = parser.parse_args()

    default = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "default.yaml"
    )
    usdz_path = args.usdz
    if usdz_path is None:
        import yaml

        with default.open("r", encoding="utf-8") as handle:
            usdz_path = yaml.safe_load(handle)["scene"]["usdz_path"]

    scene = load_scene(usdz_path, labels_path=args.labels)
    warnings = validate_scene_for_carla(scene)

    print(json.dumps(
        {
            "usdz_path": str(scene.usdz_path),
            "scene_id": scene.scene_id,
            "uuid": scene.uuid,
            "version": scene.version_string,
            "camera_ids": scene.camera_ids,
            "lidar_ids": scene.lidar_ids,
            "duration_seconds": scene.duration_seconds,
            "labels": scene.labels,
            "warnings": warnings,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())