"""Load and resolve YAML configuration with environment variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.environ.get(key, match.group(0))

    return _ENV_PATTERN.sub(replacer, value)


def _resolve_values(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _resolve_values(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_values(item) for item in node]
    if isinstance(node, str):
        return _expand_env(node)
    return node


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _resolve_values(raw)