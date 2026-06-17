"""CPU preview renderer for macOS — projects Gaussian centers without CUDA."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from carla_3dgs_ros.cosim.gsplat_renderer import SceneAlignment, carla_pose_to_gsplat_viewmat
from carla_3dgs_ros.cosim.ply_loader import GaussianCloud, load_gaussian_ply
from carla_3dgs_ros.cosim.pose import CameraPose
from carla_3dgs_ros.cosim.renderer_base import RendererBase, RendererConfig

logger = logging.getLogger(__name__)


class PreviewRenderer(RendererBase):
    """
    Lightweight CPU renderer for macOS development.

    Projects 3D Gaussian centers into the image plane. Not photorealistic,
    but lets you verify pose sync and scene alignment without CUDA/gsplat.
    """

    def __init__(
        self,
        config: RendererConfig,
        alignment: SceneAlignment | None = None,
        max_points: int = 200_000,
    ) -> None:
        super().__init__(config)
        self.alignment = alignment or SceneAlignment()
        self.max_points = max_points
        self._cloud: GaussianCloud | None = None

    def load_scene(self) -> None:
        scene_path = Path(self.config.scene_path).expanduser().resolve()
        if not scene_path.exists():
            raise FileNotFoundError(
                f"Scene not found: {scene_path}\n"
                "Use backend: mock to test without a PLY file."
            )
        self._cloud = load_gaussian_ply(scene_path)
        n = self._cloud.means.shape[0]
        if n > self.max_points:
            idx = np.linspace(0, n - 1, self.max_points, dtype=np.int64)
            self._cloud = GaussianCloud(
                means=self._cloud.means[idx],
                quats=self._cloud.quats[idx],
                scales=self._cloud.scales[idx],
                opacities=self._cloud.opacities[idx],
                colors=self._cloud.colors[idx],
                sh_degree=self._cloud.sh_degree,
            )
        logger.info(
            "PreviewRenderer loaded %d Gaussians from %s",
            self._cloud.means.shape[0],
            scene_path.name,
        )

    def render(self, pose: CameraPose) -> NDArray[np.uint8]:
        if self._cloud is None:
            raise RuntimeError("Call load_scene() first.")

        view = carla_pose_to_gsplat_viewmat(pose, self.alignment)
        K = self.intrinsic_matrix
        w, h = self.config.width, self.config.height

        # Homogeneous world points → camera frame
        ones = np.ones((self._cloud.means.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([self._cloud.means.astype(np.float64), ones])
        cam = (view @ pts_h.T).T[:, :3]

        # Keep points in front of camera (gsplat: +Z forward)
        mask = cam[:, 2] > self.config.near_plane
        cam = cam[mask]
        colors = self._cloud.colors[mask]

        if cam.shape[0] == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Project to image plane
        u = K[0, 0] * (cam[:, 0] / cam[:, 2]) + K[0, 2]
        v = K[1, 1] * (cam[:, 1] / cam[:, 2]) + K[1, 2]

        ui = np.clip(u.astype(np.int32), 0, w - 1)
        vi = np.clip(v.astype(np.int32), 0, h - 1)
        depth = cam[:, 2]

        image = np.zeros((h, w, 3), dtype=np.float64)
        zbuf = np.full((h, w), np.inf, dtype=np.float64)

        order = np.argsort(-depth)  # far to near
        for i in order:
            row, col = vi[i], ui[i]
            if depth[i] < zbuf[row, col]:
                zbuf[row, col] = depth[i]
                image[row, col] = colors[i]

        return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)[:, :, ::-1]  # BGR