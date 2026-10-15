"""thor-fetch-head: download the released head, pinned by SHA-256 (served from a file:// URL)."""

import hashlib
from pathlib import Path

import pytest

from iceberg_sar.config import Config
from iceberg_sar.thor.release import fetch_head

PAYLOAD = b"not really a torch checkpoint"


def _cfg(tmp_path: Path, payload: bytes = PAYLOAD, sha: str | None = None) -> Config:
    asset = tmp_path / "release" / "thor-linear-head.pt"
    asset.parent.mkdir(exist_ok=True)
    asset.write_bytes(payload)
    return Config(root=tmp_path, raw={
        "paths": {"models": "models"},
        "thor": {"release": {"url": asset.as_uri(), "head": "linear",
                             "sha256": sha or hashlib.sha256(payload).hexdigest()}},
    })


def test_fetch_saves_where_run_looks(tmp_path: Path) -> None:
    out, downloaded = fetch_head(_cfg(tmp_path))
    assert downloaded and out == tmp_path / "models" / "thor" / "linear" / "final" / "head.pt"
    assert out.read_bytes() == PAYLOAD


def test_fetch_skips_matching_file_and_replaces_stale(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    out, _ = fetch_head(cfg)
    assert fetch_head(cfg) == (out, False)              # same hash: nothing downloaded
    out.write_bytes(b"retrained locally")
    assert fetch_head(cfg) == (out, True)               # different hash: replaced
    assert out.read_bytes() == PAYLOAD


def test_fetch_rejects_hash_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, sha="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        fetch_head(cfg)
    assert not any((tmp_path / "models").rglob("*.p*"))   # neither head.pt nor .part left
