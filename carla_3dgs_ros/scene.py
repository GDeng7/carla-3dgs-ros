"""NuRec USDZ scene loader and metadata extraction."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class NuRecScene:
    """Metadata for a self-created or downloaded NuRec (3DGS) scene."""

    usdz_path: Path
    scene_id: str = ""
    uuid: str = ""
    camera_ids: list[str] = field(default_factory=list)
    lidar_ids: list[str] = field(default_factory=list)
    time_range: dict[str, int] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    version_string: str = ""

    @property
    def duration_ns(self) -> int:
        if not self.time_range:
            return 0
        return int(self.time_range.get("end", 0) - self.time_range.get("start", 0))

    @property
    def duration_seconds(self) -> float:
        return self.duration_ns / 1e9


def _read_zip_member(archive: zipfile.ZipFile, member: str) -> str | None:
    try:
        return archive.read(member).decode("utf-8")
    except KeyError:
        return None


def load_scene(usdz_path: str | Path, labels_path: str | Path | None = None) -> NuRecScene:
    """Load scene metadata from a NuRec USDZ archive."""
    path = Path(usdz_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"USDZ scene not found: {path}")
    if path.suffix.lower() != ".usdz":
        raise ValueError(f"Expected a .usdz file, got: {path}")

    scene = NuRecScene(usdz_path=path)

    with zipfile.ZipFile(path, "r") as archive:
        metadata_text = _read_zip_member(archive, "metadata.yaml")
        if metadata_text:
            metadata = yaml.safe_load(metadata_text) or {}
            scene.scene_id = str(metadata.get("scene_id", ""))
            scene.uuid = str(metadata.get("uuid", ""))
            scene.version_string = str(metadata.get("version_string", ""))
            scene.time_range = metadata.get("time_range", {}) or {}
            sensors = metadata.get("sensors", {}) or {}
            scene.camera_ids = list(sensors.get("camera_ids", []) or [])
            scene.lidar_ids = list(sensors.get("lidar_ids", []) or [])

        datasource_text = _read_zip_member(archive, "datasource_summary.json")
        if datasource_text and not scene.scene_id:
            summary = json.loads(datasource_text)
            scene.scene_id = str(summary.get("scene_id", scene.scene_id))

    if labels_path:
        labels_file = Path(labels_path).expanduser().resolve()
        if labels_file.exists():
            with labels_file.open("r", encoding="utf-8") as handle:
                scene.labels = json.load(handle)
    else:
        sibling_labels = path.parent / "labels.json"
        if sibling_labels.exists():
            with sibling_labels.open("r", encoding="utf-8") as handle:
                scene.labels = json.load(handle)

    return scene


def validate_scene_for_carla(scene: NuRecScene) -> list[str]:
    """Return warnings if the scene may be incompatible with CARLA NuRec replay."""
    warnings: list[str] = []
    required_members = ["metadata.yaml", "volume.nurec", "rig_trajectories.json", "map.xodr"]

    with zipfile.ZipFile(scene.usdz_path, "r") as archive:
        names = set(archive.namelist())
        for member in required_members:
            if member not in names:
                warnings.append(f"Missing required USDZ member: {member}")

    if not scene.camera_ids:
        warnings.append("No camera_ids found in metadata — replay may use defaults only.")

    return warnings