"""Load standard 3DGS PLY exports (Inria / gsplat / nerfstudio format)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SH_C0 = 0.2820947917738781


@dataclass
class GaussianCloud:
    """CPU-side Gaussian splat data ready for GPU upload."""

    means: np.ndarray       # [N, 3]
    quats: np.ndarray       # [N, 4] wxyz
    scales: np.ndarray      # [N, 3] linear scale
    opacities: np.ndarray   # [N]
    colors: np.ndarray      # [N, 3] RGB in [0, 1]
    sh_degree: int = 0


def _read_ply_header(path: Path) -> tuple[list[str], int, bool]:
    properties: list[str] = []
    vertex_count = 0
    binary = False
    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("format binary"):
                binary = True
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.startswith("property "):
                properties.append(line.split()[-1])
            if line == "end_header":
                break
    return properties, vertex_count, binary


def load_gaussian_ply(path: str | Path) -> GaussianCloud:
    """
    Load a 3D Gaussian Splatting PLY file.

    Supports the common Inria / graphdeco export format with fields:
    x,y,z, f_dc_*, f_rest_*, opacity, scale_*, rot_*
    """
    ply_path = Path(path).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Install plyfile: pip install plyfile") from exc

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()

    def col(name: str) -> np.ndarray:
        if name not in names:
            raise ValueError(f"PLY missing field '{name}'. Found: {names}")
        return np.asarray(vertex[name], dtype=np.float64)

    means = np.stack([col("x"), col("y"), col("z")], axis=1)

    # DC spherical harmonics → RGB
    if all(f in names for f in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
        colors = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0)
        sh_rest = [n for n in names if n.startswith("f_rest_")]
        sh_degree = int(np.sqrt(len(sh_rest) / 3)) if sh_rest else 0
    elif all(f in names for f in ("red", "green", "blue")):
        colors = np.stack([col("red"), col("green"), col("blue")], axis=1) / 255.0
        sh_degree = 0
    else:
        colors = np.full((means.shape[0], 3), 0.5, dtype=np.float64)
        sh_degree = 0

    opacity = col("opacity")
    opacities = 1.0 / (1.0 + np.exp(-opacity))  # sigmoid

    scales = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1)
    scales = np.exp(scales)

    quats = np.stack(
        [col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")],
        axis=1,
    )
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(min=1e-8)

    return GaussianCloud(
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales.astype(np.float32),
        opacities=opacities.astype(np.float32),
        colors=colors.astype(np.float32),
        sh_degree=sh_degree,
    )