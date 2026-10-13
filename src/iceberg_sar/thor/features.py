"""Frozen THOR features cached per chip, so head training never touches the encoder.

One .npy per chip under `<thor dir>/features/<scene>/<chip_id>.npy`: (len(blocks) * D, h, w)
float16, encoder blocks concatenated (THOR Tiny, 4 blocks, 512 px at patch 8: 768 x 64 x 64,
~6 MB). A `meta.json` records model, blocks and patch size; a cache built with other settings
is rebuilt rather than mixed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from iceberg_sar.config import Config
from iceberg_sar.thor.chips import load_index
from iceberg_sar.thor.encoder import ThorParams, embed_blocks, load_encoder, pixel_m_from_name


@dataclass(frozen=True)
class FeatureParams:
    blocks: tuple[int, ...]
    batch: int

    @classmethod
    def from_config(cls, cfg: Config) -> FeatureParams:
        f = cfg.section("thor")["features"]
        return cls(blocks=tuple(int(b) for b in f["blocks"]), batch=int(f["batch"]))


def cache_meta(tp: ThorParams, fp: FeatureParams) -> dict[str, object]:
    return {"model": tp.model, "bands": list(tp.bands), "patch_px": tp.patch_px,
            "crop_px": tp.crop_px, "blocks": list(fp.blocks)}


def feature_path(features_dir: Path, chip: pd.Series) -> Path:
    return features_dir / chip["scene"] / f"{chip['chip_id']}.npy"


def _batches(rows: pd.DataFrame, n: int) -> Iterator[pd.DataFrame]:
    for i in range(0, len(rows), n):
        yield rows.iloc[i:i + n]


def cache_features(
    chips_dir: Path, features_dir: Path, cfg: Config, device: str = "cuda",
    overwrite: bool = False,
) -> tuple[int, int]:
    """Embed every chip not cached yet. Returns (embedded, already cached)."""
    tp, fp = ThorParams.from_config(cfg), FeatureParams.from_config(cfg)
    meta = cache_meta(tp, fp)
    meta_path = features_dir / "meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        overwrite = True  # different encoder settings: rebuild everything
    features_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    index = load_index(chips_dir)
    todo = index if overwrite else index[
        [not feature_path(features_dir, r).is_file() for _, r in index.iterrows()]]
    if todo.empty:
        return 0, len(index)
    pixel_m = {pixel_m_from_name(s) for s in todo["scene"]}
    if len(pixel_m) != 1:
        raise ValueError(f"Chips mix pixel sizes {sorted(pixel_m)}; cache them separately")
    model = load_encoder(tp, pixel_m.pop(), device)
    for rows in _batches(todo, fp.batch):
        x = np.stack([np.load(chips_dir / p)["x"] for p in rows["path"]]).astype(np.float32)
        feats = embed_blocks(model, x, fp.blocks, device)
        for (_, r), f in zip(rows.iterrows(), feats, strict=True):
            out = feature_path(features_dir, r)
            out.parent.mkdir(exist_ok=True)
            np.save(out, f)
    return len(todo), len(index) - len(todo)
