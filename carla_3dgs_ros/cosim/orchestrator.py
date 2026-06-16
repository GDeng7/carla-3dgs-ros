"""Co-simulation orchestrator: CARLA physics + 3DGS renderer + ROS2/DDS."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from carla_3dgs_ros.carla_bridge import CarlaBridge, CarlaConnectionConfig
from carla_3dgs_ros.config_loader import load_config
from carla_3dgs_ros.cosim.gsplat_renderer import SceneAlignment, create_renderer
from carla_3dgs_ros.cosim.pose import CameraMount, carla_snapshot_to_stamp, extract_camera_pose
from carla_3dgs_ros.cosim.renderer_base import RendererConfig
from carla_3dgs_ros.ros_bridge import RosBridge, RosBridgeConfig
from carla_3dgs_ros.utils import prepend_carla_pythonpath

logger = logging.getLogger(__name__)


@dataclass
class CoSimOptions:
    config_path: str
    enable_ros: bool = True
    max_ticks: int = 0  # 0 = run until interrupted


class CoSimOrchestrator:
    """
    SplatSim-style co-simulation architecture:

        ┌─────────────────┐
        │  CARLA Server   │  physics, traffic, actor behaviour, scene logic
        └────────┬────────┘
                 │ each tick: read ego camera pose
                 ▼
        ┌─────────────────┐
        │ 3DGS Renderer   │  gsplat (your own pipeline, full control)
        └────────┬────────┘
                 │ rendered image
                 ▼
        ┌─────────────────┐
        │  ROS2 / DDS     │  sensor_msgs/Image → perception stack
        └─────────────────┘

    CARLA does NOT produce the perception image. It only drives the world.
    """

    def __init__(self, options: CoSimOptions) -> None:
        self.options = options
        self.config = load_config(options.config_path)
        self._carla: CarlaBridge | None = None
        self._renderer = None
        self._ros: RosBridge | None = None
        self._ego: Any = None
        self._traffic_manager: Any = None

    def _start_ros(self) -> None:
        if not self.options.enable_ros or not RosBridge.available():
            if self.options.enable_ros:
                logger.warning("rclpy not available — ROS publishing disabled.")
            return

        ros_cfg = self.config["ros2"]
        topics = dict(ros_cfg.get("topics", {}))
        topics.setdefault("front_wide_image", topics.pop("image", "camera/front/image_raw"))
        self._ros = RosBridge(
            RosBridgeConfig(
                namespace=ros_cfg["namespace"],
                ego_name=ros_cfg["ego_name"],
                frame_id=ros_cfg.get("frame_id", "front_camera"),
                publish_rate_hz=int(ros_cfg.get("publish_rate_hz", 20)),
                topics=topics,
            )
        )
        self._ros.initialize()

    def _spawn_traffic(self, carla_module: Any, world: Any, ego: Any) -> None:
        carla_cfg = self.config["carla"]
        if not carla_cfg.get("spawn_traffic", True):
            return

        num_vehicles = int(carla_cfg.get("num_traffic_vehicles", 20))
        client = self._carla.client  # type: ignore[union-attr]
        self._traffic_manager = client.get_trafficmanager(8000)
        self._traffic_manager.set_synchronous_mode(True)

        bp_lib = world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawn_points = world.get_map().get_spawn_points()

        spawned = 0
        for point in spawn_points:
            if spawned >= num_vehicles:
                break
            bp = vehicle_bps[spawned % len(vehicle_bps)]
            if bp.has_attribute("color"):
                bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
            vehicle = world.try_spawn_actor(bp, point)
            if vehicle is None:
                continue
            vehicle.set_autopilot(True, self._traffic_manager.get_port())
            spawned += 1

        self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        logger.info("Spawned %d traffic vehicles.", spawned)

    def run(self) -> int:
        prepend_carla_pythonpath()
        carla_cfg = self.config["carla"]
        renderer_cfg = self.config["renderer"]
        camera_cfg = self.config["camera"]

        self._carla = CarlaBridge(
            CarlaConnectionConfig(
                host=carla_cfg["host"],
                port=int(carla_cfg["port"]),
                timeout=float(carla_cfg["timeout"]),
                ego_role_name=carla_cfg["ego_role_name"],
                ros2_enabled=False,  # perception images come from our renderer, not CARLA
            )
        )
        carla_module = self._carla.import_carla()
        self._carla.connect(carla_module)

        if carla_cfg.get("sync_mode", True):
            self._carla.configure_sync_mode(float(carla_cfg.get("fixed_delta_seconds", 0.05)))

        self._ego = self._carla.spawn_ego_for_ros()
        self._spawn_traffic(carla_module, self._carla.world, self._ego)

        mount = CameraMount(
            position=tuple(camera_cfg["position"]),
            rotation=tuple(camera_cfg["rotation"]),
        )

        align_cfg = self.config.get("alignment", {})
        alignment = SceneAlignment(
            offset=tuple(align_cfg.get("offset", [0.0, 0.0, 0.0])),
            yaw_offset_deg=float(align_cfg.get("yaw_offset_deg", 0.0)),
        )

        self._renderer = create_renderer(
            RendererConfig(
                scene_path=renderer_cfg["scene_path"],
                width=int(renderer_cfg["width"]),
                height=int(renderer_cfg["height"]),
                fov=float(renderer_cfg["fov"]),
                device=str(renderer_cfg.get("device", "cuda")),
                near_plane=float(renderer_cfg.get("near_plane", 0.01)),
                far_plane=float(renderer_cfg.get("far_plane", 1000.0)),
                background_color=tuple(renderer_cfg.get("background_color", [0, 0, 0])),
                backend=str(renderer_cfg.get("backend", "gsplat")),
            ),
            alignment=alignment,
        )
        self._renderer.load_scene()
        self._start_ros()

        logger.info(
            "Co-simulation running. CARLA=physics/traffic, renderer=%s, ROS=%s",
            renderer_cfg.get("backend"),
            "enabled" if self._ros else "disabled",
        )

        tick = 0
        try:
            while self.options.max_ticks == 0 or tick < self.options.max_ticks:
                self._carla.tick()
                snapshot = self._carla.world.get_snapshot()
                stamp_sec, stamp_nanosec = carla_snapshot_to_stamp(snapshot)

                pose = extract_camera_pose(
                    self._ego,
                    carla_module,
                    mount,
                    stamp_sec,
                    stamp_nanosec,
                )
                image = self._renderer.render(pose)

                if self._ros is not None:
                    self._ros.publish_image(
                        "front_wide_image",
                        image,
                        stamp_sec,
                        stamp_nanosec,
                    )
                    transform = self._ego.get_transform()
                    self._ros.publish_odometry(
                        x=transform.location.x,
                        y=transform.location.y,
                        z=transform.location.z,
                        yaw_rad=float(transform.rotation.yaw) * 3.14159265 / 180.0,
                        stamp_sec=stamp_sec,
                        stamp_nanosec=stamp_nanosec,
                    )
                    self._ros.publish_clock(stamp_sec, stamp_nanosec)
                    self._ros.spin_once(timeout_sec=0.0)

                tick += 1
                if tick % 100 == 0:
                    logger.info("Tick %d — pose=(%.1f, %.1f, %.1f)", tick, *pose.position)
        except KeyboardInterrupt:
            logger.info("Co-simulation stopped by user.")
        finally:
            if self._ros is not None:
                self._ros.shutdown()
            if self._carla is not None:
                self._carla.cleanup()

        return 0


def run_from_cli(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Co-simulation: CARLA physics + separate 3DGS renderer + ROS2/DDS.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent.parent / "config" / "cosim.yaml"),
    )
    parser.add_argument("--no-ros", action="store_true")
    parser.add_argument("--max-ticks", type=int, default=0)
    args = parser.parse_args(argv)

    orchestrator = CoSimOrchestrator(
        CoSimOptions(
            config_path=args.config,
            enable_ros=not args.no_ros,
            max_ticks=args.max_ticks,
        )
    )
    return orchestrator.run()