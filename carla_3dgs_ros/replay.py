"""Main orchestrator: CARLA + NuRec (3DGS) replay with optional ROS2 bridge."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from carla_3dgs_ros.carla_bridge import CarlaBridge, CarlaConnectionConfig
from carla_3dgs_ros.config_loader import load_config
from carla_3dgs_ros.nurec_client import NuRecContainer, NuRecContainerConfig
from carla_3dgs_ros.ros_bridge import RosBridge, RosBridgeConfig
from carla_3dgs_ros.scene import load_scene, validate_scene_for_carla
from carla_3dgs_ros.utils import carla_official_nurec_script, ensure_directory, prepend_carla_pythonpath

logger = logging.getLogger(__name__)


@dataclass
class ReplayOptions:
    config_path: str
    usdz_path: str | None = None
    use_official_script: bool = True
    start_nurec_container: bool = True
    enable_ros_bridge: bool = True
    dry_run: bool = False


class ReplayOrchestrator:
    """
    End-to-end pipeline:

      1. Load NuRec USDZ (your self-created 3DGS map)
      2. Start NuRec gRPC Docker renderer
      3. Connect to CARLA server (launch with --ros2 for native DDS)
      4. Replay scene and publish sensors via ROS2

    Architecture (from NVIDIA NuRec + CARLA integration):

        ROS2 Stack  <---DDS--->  CARLA (--ros2)  <---API--->  Replay Script
                                                                    |
                                                              NuRec gRPC
                                                                    |
                                                              3DGS USDZ scene
    """

    def __init__(self, options: ReplayOptions) -> None:
        self.options = options
        self.config = load_config(options.config_path)
        self.scene = load_scene(
            options.usdz_path or self.config["scene"]["usdz_path"],
        )
        self._nurec: NuRecContainer | None = None
        self._ros: RosBridge | None = None
        self._carla: CarlaBridge | None = None

    def validate(self) -> None:
        warnings = validate_scene_for_carla(self.scene)
        for warning in warnings:
            logger.warning(warning)

        if warnings:
            logger.warning("Scene validation produced %d warning(s).", len(warnings))
        else:
            logger.info("Scene validation passed for %s", self.scene.usdz_path.name)

        logger.info(
            "Scene: id=%s cameras=%s duration=%.1fs",
            self.scene.scene_id or "unknown",
            len(self.scene.camera_ids),
            self.scene.duration_seconds,
        )

    def _setup_nurec_env(self) -> None:
        nurec_cfg = self.config["nurec"]
        os.environ.setdefault("NUREC_IMAGE", nurec_cfg["image"])
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", nurec_cfg["cuda_visible_devices"])

    def _start_nurec(self) -> None:
        if not self.options.start_nurec_container:
            logger.info("Skipping NuRec container start (start_nurec_container=False).")
            return

        nurec_cfg = self.config["nurec"]
        self._nurec = NuRecContainer(
            NuRecContainerConfig(
                image=nurec_cfg["image"],
                grpc_port=int(nurec_cfg["grpc_port"]),
                cuda_visible_devices=str(nurec_cfg["cuda_visible_devices"]),
            )
        )
        self._nurec.start()

    def _start_ros_bridge(self) -> None:
        if not self.options.enable_ros_bridge:
            return
        if not RosBridge.available():
            logger.warning("rclpy not installed — ROS2 bridge disabled.")
            return

        ros_cfg = self.config["ros2"]
        self._ros = RosBridge(
            RosBridgeConfig(
                namespace=ros_cfg["namespace"],
                ego_name=ros_cfg["ego_name"],
                frame_id=ros_cfg["frame_id"],
                publish_rate_hz=int(ros_cfg["publish_rate_hz"]),
                topics=dict(ros_cfg.get("topics", {})),
            )
        )
        self._ros.initialize()

    def _delegate_to_official_script(self) -> int:
        script = carla_official_nurec_script()
        if script is None:
            raise FileNotFoundError(
                "CARLA_ROOT is not set or official NuRec example script is missing.\n"
                "Set CARLA_ROOT to your CARLA package and run install_nurec.sh first."
            )

        carla_cfg = self.config["carla"]
        nurec_cfg = self.config["nurec"]
        replay_cfg = self.config["replay"]

        cmd = [
            sys.executable,
            str(script),
            "--host",
            carla_cfg["host"],
            "--port",
            str(carla_cfg["port"]),
            "--usdz-filename",
            str(self.scene.usdz_path),
            "--nurec-port",
            str(nurec_cfg["grpc_port"]),
        ]
        if replay_cfg.get("move_spectator"):
            cmd.append("--move-spectator")
        if replay_cfg.get("save_images"):
            cmd.append("--saveimages")
            output_dir = replay_cfg.get("output_dir", "output/captured_images")
            cmd.extend(["--output-dir", str(ensure_directory(output_dir))])

        logger.info("Delegating replay to CARLA official script: %s", script.name)
        env = os.environ.copy()
        env["NUREC_IMAGE"] = nurec_cfg["image"]
        return subprocess.call(cmd, env=env, cwd=script.parent)

    def _run_integrated_replay(self) -> None:
        """
        Integrated replay path when CARLA official script is unavailable.

        Connects to CARLA, enables sync mode, spawns ROS-enabled ego, and
        keeps the ROS2 bridge alive. NuRec frame injection requires the
        official gRPC client from CARLA's nurec examples — use
        use_official_script=True on Ubuntu with CARLA installed.
        """
        prepend_carla_pythonpath()
        carla_cfg = self.config["carla"]
        replay_cfg = self.config["replay"]

        self._carla = CarlaBridge(
            CarlaConnectionConfig(
                host=carla_cfg["host"],
                port=int(carla_cfg["port"]),
                timeout=float(carla_cfg["timeout"]),
                ego_role_name=carla_cfg["ego_role_name"],
                ros2_enabled=bool(carla_cfg.get("ros2_enabled", True)),
            )
        )
        self._carla.connect()

        if replay_cfg.get("tick_sync", True):
            self._carla.configure_sync_mode(float(replay_cfg.get("fixed_delta_seconds", 0.05)))

        ego = self._carla.spawn_ego_for_ros()
        camera_config_path = Path(self.config["scene"]["camera_config"]).resolve()
        with camera_config_path.open("r", encoding="utf-8") as handle:
            camera_configs = yaml.safe_load(handle)["cameras"]

        for cam in camera_configs:
            self._carla.attach_ros_camera(
                parent=ego,
                ros_name=cam["ros_name"],
                width=int(cam["width"]),
                height=int(cam["height"]),
                fov=float(cam["fov"]),
            )

        logger.info(
            "Integrated replay running. Launch CARLA with --ros2 for native DDS sensor topics.\n"
            "For full NuRec neural rendering, set CARLA_ROOT and use --official."
        )

        tick = 0
        try:
            while tick < 100:
                self._carla.tick()
                if self._ros is not None:
                    self._ros.publish_clock(stamp_sec=tick, stamp_nanosec=0)
                    self._ros.spin_once(timeout_sec=0.0)
                tick += 1
        except KeyboardInterrupt:
            logger.info("Replay interrupted by user.")
        finally:
            if self._carla is not None:
                self._carla.cleanup()

    def run(self) -> int:
        self.validate()
        if self.options.dry_run:
            logger.info("Dry run complete — configuration and scene are valid.")
            return 0

        self._setup_nurec_env()
        self._start_nurec()
        self._start_ros_bridge()

        try:
            if self.options.use_official_script and carla_official_nurec_script() is not None:
                return self._delegate_to_official_script()
            self._run_integrated_replay()
            return 0
        finally:
            if self._ros is not None:
                self._ros.shutdown()
            if self._nurec is not None:
                self._nurec.stop()


def run_from_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay a NuRec (3DGS) scene in CARLA with ROS2/DDS publishing.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "config" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--usdz", default=None, help="Override USDZ scene path.")
    parser.add_argument(
        "--official",
        action="store_true",
        default=True,
        help="Use CARLA's official NuRec replay script when CARLA_ROOT is set.",
    )
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Force integrated replay (no official CARLA script).",
    )
    parser.add_argument("--no-ros", action="store_true", help="Disable Python ROS2 bridge.")
    parser.add_argument(
        "--no-nurec-container",
        action="store_true",
        help="Do not start the NuRec Docker container.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and scene only.",
    )
    args = parser.parse_args(argv)

    options = ReplayOptions(
        config_path=args.config,
        usdz_path=args.usdz,
        use_official_script=not args.integrated,
        start_nurec_container=not args.no_nurec_container,
        enable_ros_bridge=not args.no_ros,
        dry_run=args.dry_run,
    )
    orchestrator = ReplayOrchestrator(options)
    return orchestrator.run()