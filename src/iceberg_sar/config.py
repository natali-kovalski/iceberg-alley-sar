"""Load and access `config.yaml`.

`include: [file, ...]` in the main config merges more top-level sections from other files
(paths relative to the main config). Paths inside them still resolve against the main config's
directory, and a section may be defined only once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config.yaml")

REQUIRED_SECTIONS = ("paths", "search", "hyp3", "preprocess", "cfar", "detections")


@dataclass(frozen=True)
class Config:
    """Parsed configuration. Paths are resolved against the config file's directory."""

    root: Path
    raw: dict[str, Any] = field(repr=False)
    files: tuple[Path, ...] = ()  # main config first, then its includes

    def section(self, name: str) -> dict[str, Any]:
        if name not in self.raw:
            raise KeyError(f"Config section '{name}' not found")
        return self.raw[name]

    def path(self, key: str) -> Path:
        """Absolute path for an entry in the `paths` section."""
        paths = self.section("paths")
        if key not in paths:
            raise KeyError(f"Path '{key}' not found in config 'paths' section")
        return (self.root / paths[key]).resolve()

    @property
    def data_dirs(self) -> list[Path]:
        return [self.path(k) for k in ("raw", "interim", "outputs")]


def metres_to_px(metres: float, pixel_m: float, minimum: int = 1) -> int:
    """Config lengths are in metres so they hold at any resolution; round to whole pixels."""
    return max(minimum, round(metres / pixel_m))


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return raw


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    config_path = Path(path).resolve()
    raw = _read_mapping(config_path)
    files = [config_path]
    for inc in raw.pop("include", None) or []:
        inc_path = (config_path.parent / inc).resolve()
        extra = _read_mapping(inc_path)
        if "include" in extra:
            raise ValueError(f"Nested include is not supported: {inc_path}")
        dup = sorted(raw.keys() & extra.keys())
        if dup:
            raise ValueError(f"Config sections defined twice ({inc_path}): {', '.join(dup)}")
        raw.update(extra)
        files.append(inc_path)
    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(missing)}")
    return Config(root=config_path.parent, raw=raw, files=tuple(files))
