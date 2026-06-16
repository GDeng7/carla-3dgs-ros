"""Shared helpers."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def carla_official_nurec_script() -> Path | None:
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        return None
    script = (
        Path(carla_root)
        / "PythonAPI"
        / "examples"
        / "nvidia"
        / "nurec"
        / "example_nurec_replay_save_images.py"
    )
    return script if script.exists() else None


def prepend_carla_pythonpath() -> None:
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        return
    egg_dir = Path(carla_root) / "PythonAPI" / "carla"
    for candidate in sorted(egg_dir.glob("dist/carla-*-py3*.egg"), reverse=True):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))