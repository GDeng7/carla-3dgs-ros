"""Synthetic ego/camera pose generator for macOS development (no CARLA)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from carla_3dgs_ros.cosim.pose import CameraMount, CameraPose


def _rotation_matrix_carla(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build 3x3 rotation for CARLA convention: X=forward, Y=right, Z=up."""
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Rz(yaw) @ Ry(pitch) @ Rx(roll) — Unreal/CARLA Euler order
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _apply_camera_mount(
    ego_position: tuple[float, float, float],
    ego_rotation: np.ndarray,
    mount: CameraMount,
) -> tuple[tuple[float, float, float], np.ndarray]:
    """Apply sensor offset to ego pose (same math as CARLA, without CARLA API)."""
    T_ego = np.eye(4, dtype=np.float64)
    T_ego[:3, :3] = ego_rotation
    T_ego[:3, 3] = ego_position

    R_mount = _rotation_matrix_carla(*mount.rotation)
    T_mount = np.eye(4, dtype=np.float64)
    T_mount[:3, :3] = R_mount
    T_mount[:3, 3] = mount.position

    T_cam = T_ego @ T_mount
    position = tuple(float(v) for v in T_cam[:3, 3])
    return position, T_cam[:3, :3].copy()


@dataclass
class PoseSimulatorConfig:
    trajectory: str = "straight"   # straight | circle | replay
    speed_mps: float = 10.0
    duration_sec: float = 10.0
    tick_hz: float = 20.0
    start_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_yaw_deg: float = 0.0
    circle_radius_m: float = 30.0
    replay_file: str = ""


class PoseSimulator:
    """Generate camera poses that mimic a CARLA ego vehicle driving."""

    def __init__(self, config: PoseSimulatorConfig, mount: CameraMount) -> None:
        self.config = config
        self.mount = mount

    def poses(self) -> Iterator[CameraPose]:
        if self.config.trajectory == "replay":
            yield from self._replay_poses()
        elif self.config.trajectory == "circle":
            yield from self._circle_poses()
        else:
            yield from self._straight_poses()

    def _straight_poses(self) -> Iterator[CameraPose]:
        cfg = self.config
        dt = 1.0 / cfg.tick_hz
        total_ticks = int(cfg.duration_sec * cfg.tick_hz)
        R = _rotation_matrix_carla(0.0, 0.0, cfg.start_yaw_deg)

        for tick in range(total_ticks):
            t = tick * dt
            dist = cfg.speed_mps * t
            ego_pos = (
                cfg.start_position[0] + dist * R[0, 0],
                cfg.start_position[1] + dist * R[1, 0],
                cfg.start_position[2] + dist * R[2, 0],
            )
            stamp_ns = int(t * 1e9)
            sec, nanosec = divmod(stamp_ns, 10**9)
            cam_pos, cam_rot = _apply_camera_mount(ego_pos, R, self.mount)
            yield CameraPose(sec, nanosec, cam_pos, cam_rot)

    def _circle_poses(self) -> Iterator[CameraPose]:
        cfg = self.config
        dt = 1.0 / cfg.tick_hz
        total_ticks = int(cfg.duration_sec * cfg.tick_hz)
        omega = cfg.speed_mps / max(cfg.circle_radius_m, 1.0)

        for tick in range(total_ticks):
            t = tick * dt
            angle = omega * t + math.radians(cfg.start_yaw_deg)
            cx = cfg.start_position[0] + cfg.circle_radius_m * math.cos(angle)
            cy = cfg.start_position[1] + cfg.circle_radius_m * math.sin(angle)
            cz = cfg.start_position[2]
            yaw_deg = math.degrees(angle + math.pi / 2)
            R = _rotation_matrix_carla(0.0, 0.0, yaw_deg)

            stamp_ns = int(t * 1e9)
            sec, nanosec = divmod(stamp_ns, 10**9)
            cam_pos, cam_rot = _apply_camera_mount((cx, cy, cz), R, self.mount)
            yield CameraPose(sec, nanosec, cam_pos, cam_rot)

    def _replay_poses(self) -> Iterator[CameraPose]:
        path = Path(self.config.replay_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Pose replay file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        entries = data.get("poses", data)
        for entry in entries:
            t = float(entry.get("t", 0.0))
            stamp_ns = int(t * 1e9)
            sec, nanosec = divmod(stamp_ns, 10**9)
            ego_pos = (
                float(entry["x"]),
                float(entry["y"]),
                float(entry.get("z", 0.0)),
            )
            yaw = float(entry.get("yaw_deg", entry.get("yaw", 0.0)))
            pitch = float(entry.get("pitch_deg", 0.0))
            roll = float(entry.get("roll_deg", 0.0))
            R = _rotation_matrix_carla(roll, pitch, yaw)
            cam_pos, cam_rot = _apply_camera_mount(ego_pos, R, self.mount)
            yield CameraPose(sec, nanosec, cam_pos, cam_rot)


def export_poses_template(path: str | Path) -> None:
    """Write a sample pose replay JSON for use on macOS after CARLA recording."""
    template = {
        "description": "Ego poses exported from CARLA (replay on macOS with trajectory=replay)",
        "poses": [
            {"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            {"t": 0.05, "x": 0.5, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            {"t": 0.10, "x": 1.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
        ],
    }
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(template, handle, indent=2)