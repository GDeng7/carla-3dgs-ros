"""CARLA client helpers for NuRec replay and ROS2-native sensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CarlaConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 30.0
    ego_role_name: str = "ego"
    ros2_enabled: bool = True


class CarlaBridge:
    """Thin wrapper around the CARLA Python API."""

    def __init__(self, config: CarlaConnectionConfig) -> None:
        self.config = config
        self._carla: Any = None
        self._client: Any = None
        self._world: Any = None
        self._ego: Any = None

    @staticmethod
    def import_carla(carla_egg: str | None = None) -> Any:
        try:
            import carla  # type: ignore

            return carla
        except ImportError as exc:
            hint = (
                "Install the CARLA Python API wheel, e.g.\n"
                "  pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-*-py3*.whl"
            )
            if carla_egg:
                hint = f"Add CARLA egg to PYTHONPATH: {carla_egg}"
            raise ImportError(f"carla module not found. {hint}") from exc

    def connect(self, carla_module: Any | None = None) -> None:
        self._carla = carla_module or self.import_carla()
        self._client = self._carla.Client(self.config.host, self.config.port)
        self._client.set_timeout(self.config.timeout)
        self._world = self._client.get_world()
        logger.info("Connected to CARLA at %s:%s", self.config.host, self.config.port)

    @property
    def world(self) -> Any:
        if self._world is None:
            raise RuntimeError("Not connected to CARLA. Call connect() first.")
        return self._world

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError("Not connected to CARLA. Call connect() first.")
        return self._client

    def configure_sync_mode(self, fixed_delta_seconds: float) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(settings)
        logger.info("Synchronous mode enabled (dt=%.3fs)", fixed_delta_seconds)

    def spawn_ego_for_ros(self, blueprint_filter: str = "vehicle.lincoln.mkz_2020") -> Any:
        """Spawn ego vehicle with ROS2 topic naming enabled."""
        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.filter(blueprint_filter)[0]
        bp.set_attribute("role_name", self.config.ego_role_name)
        if self.config.ros2_enabled:
            bp.set_attribute("ros_name", self.config.ego_role_name)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in the current CARLA map.")

        self._ego = self.world.spawn_actor(bp, spawn_points[0])
        logger.info("Spawned ego vehicle '%s' (id=%s)", self.config.ego_role_name, self._ego.id)
        return self._ego

    def attach_ros_camera(
        self,
        parent: Any,
        ros_name: str,
        width: int = 1920,
        height: int = 1080,
        fov: float = 90.0,
    ) -> Any:
        """Attach a CARLA RGB camera with native ROS2 publishing enabled."""
        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("ros_name", ros_name)

        transform = self._carla.Transform(
            self._carla.Location(x=2.0, z=1.5),
            self._carla.Rotation(pitch=0.0),
        )
        sensor = self.world.spawn_actor(bp, transform, attach_to=parent)
        if self.config.ros2_enabled:
            sensor.enable_for_ros()
        logger.info("Attached ROS camera '%s' (id=%s)", ros_name, sensor.id)
        return sensor

    def tick(self) -> None:
        self.world.tick()

    def cleanup(self) -> None:
        if self._ego is not None:
            self._ego.destroy()
            self._ego = None
        logger.info("CARLA actors cleaned up.")