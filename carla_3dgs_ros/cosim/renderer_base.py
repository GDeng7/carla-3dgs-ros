"""Abstract 3DGS renderer interface (decoupled from CARLA)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from carla_3dgs_ros.cosim.pose import CameraPose


@dataclass
class RendererConfig:
    scene_path: str
    width: int = 1920
    height: int = 1080
    fov: float = 90.0
    device: str = "cuda"
    near_plane: float = 0.01
    far_plane: float = 1000.0
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    backend: str = "gsplat"


class RendererBase(ABC):
    """Own your rendering pipeline — no NuRec Docker dependency."""

    def __init__(self, config: RendererConfig) -> None:
        self.config = config

    @abstractmethod
    def load_scene(self) -> None:
        """Load 3DGS assets (PLY, glTF splat tileset, etc.)."""

    @abstractmethod
    def render(self, pose: CameraPose) -> NDArray[np.uint8]:
        """Render BGR uint8 image from the given camera pose."""

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 pinhole intrinsics from FOV and resolution."""
        w, h = self.config.width, self.config.height
        fov_rad = np.deg2rad(self.config.fov)
        fx = fy = w / (2.0 * np.tan(fov_rad / 2.0))
        cx, cy = w / 2.0, h / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)