"""Milestone 7c: fetch the trained THOR head from its GitHub release (`thor-fetch-head`).

Weights are never committed. The final linear head is published as a release asset and pinned
by SHA-256 in `thor.release`; a download that doesn't match is discarded. `load_head` reads
checkpoints with `weights_only=True`, so the file can't run code when it is loaded.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from iceberg_sar.config import Config


@dataclass(frozen=True)
class ReleaseParams:
    url: str
    sha256: str
    head: str   # head kind (linear / centernet): saved as models/thor/<head>/final/head.pt

    @classmethod
    def from_config(cls, cfg: Config) -> ReleaseParams:
        r = cfg.section("thor")["release"]
        return cls(url=str(r["url"]), sha256=str(r["sha256"]).lower(), head=str(r["head"]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def head_path(cfg: Config, p: ReleaseParams) -> Path:
    return cfg.path("models") / "thor" / p.head / "final" / "head.pt"


def fetch_head(cfg: Config, force: bool = False) -> tuple[Path, bool]:
    """Download the released head unless a file with the pinned hash is already there.

    Returns (path, downloaded). Raises ValueError on a hash mismatch, leaving no file behind.
    """
    p = ReleaseParams.from_config(cfg)
    out = head_path(cfg, p)
    if not force and out.is_file() and sha256(out) == p.sha256:
        return out, False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    try:
        with urllib.request.urlopen(p.url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if (got := sha256(tmp)) != p.sha256:
            raise ValueError(f"{p.url}: SHA-256 {got} does not match thor.release.sha256 "
                             f"{p.sha256}; not saved")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out, True
