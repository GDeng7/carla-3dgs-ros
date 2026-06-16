"""Camera pose extraction from CARLA actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """Timestamped 6-DoF camera pose for the 3DGS renderer."""

    stamp_sec: int
    stamp_nanosec: int
    position: tuple[float, float, float]
    rotation_matrix: np.ndarray  # 3x3, camera-to-world

    @property
    def view_matrix(self) -> np.ndarray:
        """World-to-camera 4x4 matrix for gsplat rendering."""
        view = np.eye(4, dtype=np.float64)
        R = self.rotation_matrix
        t = np.array(self.position, dtype=np.float64)
        view[:3, :3] = R.T
        view[:3, 3] = -(R.T @ t)
        return view


@dataclass(frozen=True)
class CameraMount:
    """Camera offset relative to a CARLA actor."""

    position: tuple[float, float, float] = (2.0, 0.0, 1.5)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw


def _actor_transform_matrix(actor: Any) -> np.ndarray:
    return np.asarray(actor.get_transform().get_matrix(), dtype=np.float64)


def _offset_matrix(carla_module: Any, mount: CameraMount) -> np.ndarray:
    transform = carla_module.Transform(
        carla_module.Location(x=mount.position[0], y=mount.position[1], z=mount.position[2]),
        carla_module.Rotation(
            roll=mount.rotation[0],
            pitch=mount.rotation[1],
            yaw=mount.rotation[2],
        ),
    )
    return np.asarray(transform.get_matrix(), dtype=np.float64)


def extract_camera_pose(
    actor: Any,
    carla_module: Any,
    mount: CameraMount,
    stamp_sec: int,
    stamp_nanosec: int,
) -> CameraPose:
    """Compute world-frame camera pose from a CARLA actor and sensor mount."""
    T_world_actor = _actor_transform_matrix(actor)
    T_actor_camera = _offset_matrix(carla_module, mount)
    T_world_camera = T_world_actor @ T_actor_camera

    position = tuple(float(v) for v in T_world_camera[:3, 3])
    rotation = T_world_camera[:3, :3].copy()
    return CameraPose(
        stamp_sec=stamp_sec,
        stamp_nanosec=stamp_nanosec,
        position=position,
        rotation_matrix=rotation,
    )


def carla_snapshot_to_stamp(snapshot: Any) -> tuple[int, int]:
    elapsed_ns = int(snapshot.timestamp.elapsed_seconds * 1e9)
    sec, nanosec = divmod(elapsed_ns, 10**9)
    return sec, nanosec