"""gsplat-based 3DGS renderer — no NuRec dependency."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from carla_3dgs_ros.cosim.ply_loader import GaussianCloud, load_gaussian_ply
from carla_3dgs_ros.cosim.pose import CameraPose
from carla_3dgs_ros.cosim.renderer_base import RendererBase, RendererConfig

logger = logging.getLogger(__name__)


@dataclass
class SceneAlignment:
    """Align CARLA world coordinates to your 3DGS scene origin."""

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_offset_deg: float = 0.0


def carla_pose_to_gsplat_viewmat(
    pose: CameraPose,
    alignment: SceneAlignment,
) -> np.ndarray:
    """
    Convert CARLA camera pose to gsplat world-to-camera view matrix.

    CARLA: X=forward, Y=right, Z=up
    gsplat: right-down-forward (RDF)
    """
    import math

    yaw_rad = math.radians(alignment.yaw_offset_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    R_align = np.array(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    offset = np.array(alignment.offset, dtype=np.float64)
    position = R_align @ (np.array(pose.position, dtype=np.float64) - offset)
    R_carla = R_align @ pose.rotation_matrix

    # CARLA camera axes → gsplat RDF
    R_rdf = np.column_stack([R_carla[:, 1], -R_carla[:, 2], R_carla[:, 0]])

    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = R_rdf.T
    view[:3, 3] = -(R_rdf.T @ position)
    return view


class GsplatRenderer(RendererBase):
    """
    Self-owned 3DGS renderer using gsplat.

    Works on RTX 5090 (Blackwell) when PyTorch + gsplat are built for CUDA 12.8+.
    No NuRec Docker required.
    """

    def __init__(
        self,
        config: RendererConfig,
        alignment: SceneAlignment | None = None,
    ) -> None:
        super().__init__(config)
        self.alignment = alignment or SceneAlignment()
        self._cloud: GaussianCloud | None = None
        self._device = None
        self._tensors: dict | None = None

    def load_scene(self) -> None:
        scene_path = Path(self.config.scene_path).expanduser().resolve()
        if not scene_path.exists():
            raise FileNotFoundError(
                f"3DGS scene not found: {scene_path}\n"
                "Export your self-built map as standard 3DGS PLY "
                "(from 3DGUT, nerfstudio, or graphdeco-inria/gaussian-splatting)."
            )

        self._cloud = load_gaussian_ply(scene_path)
        logger.info(
            "Loaded %d Gaussians from %s (sh_degree=%d)",
            self._cloud.means.shape[0],
            scene_path.name,
            self._cloud.sh_degree,
        )

        try:
            import torch

            self._device = torch.device(self.config.device)
            self._tensors = {
                "means": torch.from_numpy(self._cloud.means).to(self._device),
                "quats": torch.from_numpy(self._cloud.quats).to(self._device),
                "scales": torch.from_numpy(self._cloud.scales).to(self._device),
                "opacities": torch.from_numpy(self._cloud.opacities).to(self._device),
                "colors": torch.from_numpy(self._cloud.colors).to(self._device),
            }
            logger.info("GPU tensors ready on %s", self._device)
        except ImportError:
            logger.warning("torch not installed — rendering will use CPU placeholder.")

    def render(self, pose: CameraPose) -> NDArray[np.uint8]:
        if self._cloud is None:
            raise RuntimeError("Call load_scene() before render().")

        if self._tensors is None:
            return self._placeholder_image(pose)

        import torch
        from gsplat import rasterization

        view_np = carla_pose_to_gsplat_viewmat(pose, self.alignment)
        K_np = self.intrinsic_matrix

        viewmat = torch.tensor(view_np, device=self._device, dtype=torch.float32)
        K = torch.tensor(K_np, device=self._device, dtype=torch.float32)
        bg = torch.tensor(
            [list(self.config.background_color)],
            device=self._device,
            dtype=torch.float32,
        )

        sh_degree = self._cloud.sh_degree if self._cloud.sh_degree > 0 else None

        with torch.no_grad():
            colors, _alphas, _meta = rasterization(
                means=self._tensors["means"],
                quats=self._tensors["quats"],
                scales=self._tensors["scales"],
                opacities=self._tensors["opacities"],
                colors=self._tensors["colors"],
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=self.config.width,
                height=self.config.height,
                sh_degree=sh_degree,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_mode="RGB",
                packed=False,
                backgrounds=bg,
            )

        rgb = colors[0].clamp(0.0, 1.0).byte().cpu().numpy()
        return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR for ROS

    def _placeholder_image(self, pose: CameraPose) -> NDArray[np.uint8]:
        h, w = self.config.height, self.config.width
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:, :, 1] = int(abs(pose.position[0]) * 10) % 255
        return image


class MockRenderer(RendererBase):
    """Test the co-simulation loop without GPU or PLY files."""

    def load_scene(self) -> None:
        logger.info("MockRenderer ready — no NuRec, no PLY required.")

    def render(self, pose: CameraPose) -> NDArray[np.uint8]:
        h, w = self.config.height, self.config.width
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:, :, 2] = (int(pose.stamp_sec * 10) % 255)
        return image


def create_renderer(
    config: RendererConfig,
    alignment: SceneAlignment | None = None,
) -> RendererBase:
    if config.backend == "mock":
        return MockRenderer(config)
    return GsplatRenderer(config, alignment=alignment)