"""ENL of a dual-pol stack from open-water boxes (mean² / variance of linear intensity).

The equivalent number of looks (ENL) measures speckle strength: for gamma-distributed
clutter, intensity variance = mean² / ENL. CFAR thresholds depend on it, and it changes
with multilooking, so it is estimated on the raster the detectors actually run on.

`auto` tiles the raster into square boxes, keeps boxes that are fully valid, and takes the
box whose HH and HV ENL are both closest to the median over all boxes: a typical open-water
box. Boxes with a berg or texture in them have a lower ENL, so the median is robust to them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from iceberg_sar.benchmark.preprocess import BANDS


def moment_enl(x: np.ndarray) -> float:
    """ENL = mean² / variance over the finite samples."""
    v = x[np.isfinite(x)].astype(np.float64)
    return float(v.mean() ** 2 / v.var())


def box_table(stack: np.ndarray, valid: np.ndarray, box_px: int) -> pd.DataFrame:
    """Mean and ENL per band for every fully valid `box_px` square box."""
    _, h, w = stack.shape
    hb, wb = h // box_px, w // box_px
    full = valid[: hb * box_px, : wb * box_px].reshape(hb, box_px, wb, box_px).all(axis=(1, 3))
    rows, cols = np.nonzero(full)
    out: dict[str, Any] = {"row0": rows * box_px, "col0": cols * box_px}
    for i, pol in enumerate(BANDS):
        b = stack[i, : hb * box_px, : wb * box_px].reshape(hb, box_px, wb, box_px)
        b = b.transpose(0, 2, 1, 3)[rows, cols].reshape(len(rows), -1).astype(np.float64)
        m, v = b.mean(axis=1), b.var(axis=1)
        out[f"mean_{pol}"] = m
        out[f"enl_{pol}"] = m**2 / v
    return pd.DataFrame(out)


def combine(enl: dict[str, float], how: str) -> float:
    """The library takes one ENL for both bands."""
    if how == "min":
        return min(enl.values())
    if how == "mean":
        return float(np.mean(list(enl.values())))
    if how in enl:
        return enl[how]
    raise ValueError(f"enl_combine must be min, mean, HH or HV, got {how!r}")


def estimate_enl(
    raster: Path, mask: Path, box_m: float, box: str | list[float] = "auto",
    combine_how: str = "min", sidecar: Path | None = None,
) -> dict[str, Any]:
    """ENL per band (and the combined value passed to the library); optionally written as JSON."""
    with rasterio.open(raster) as src, rasterio.open(mask) as msrc:
        pixel_m = abs(src.transform.a)
        if box == "auto":
            stack, valid = src.read(), msrc.read(1) > 0
            box_px = max(2, round(box_m / pixel_m))
            t = box_table(stack, valid, box_px)
            if t.empty:
                raise ValueError(f"No fully valid {box_px} px box for the ENL estimate")
            dist = sum(np.abs(np.log(t[f"enl_{p}"] / t[f"enl_{p}"].median())) for p in BANDS)
            pick = t.loc[dist.idxmin()]
            r0, c0 = int(pick["row0"]), int(pick["col0"])
            win = rasterio.windows.Window(c0, r0, box_px, box_px)
            spread = {p: {q: round(float(t[f"enl_{p}"].quantile(q / 100)), 3)
                          for q in (10, 50, 90)} for p in BANDS}
            n_boxes = len(t)
        else:
            to_scene = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x0, y0 = to_scene.transform(box[0], box[1])
            x1, y1 = to_scene.transform(box[2], box[3])
            win = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                              src.transform).round_offsets().round_lengths()
            spread, n_boxes = None, None
        data = src.read(window=win)
        ok = msrc.read(1, window=win) > 0
        enl = {p: round(moment_enl(np.where(ok, data[i], np.nan)), 3)
               for i, p in enumerate(BANDS)}
        left, bottom, right, top = rasterio.windows.bounds(win, src.transform)
        to_wgs = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        w, s = to_wgs.transform(left, bottom)
        e, n = to_wgs.transform(right, top)
    result = {
        "raster": raster.name, "pixel_m": pixel_m, "method": "mean^2 / variance, linear",
        "box_selection": "auto (median box)" if box == "auto" else "config",
        "box_px": [int(win.height), int(win.width)],
        "box_rowcol": [int(win.row_off), int(win.col_off)],
        "box_wgs84": [round(v, 5) for v in (w, s, e, n)],
        "box_mean_db": {p: round(float(10 * np.log10(np.nanmean(np.where(ok, data[i], np.nan)))), 2)
                        for i, p in enumerate(BANDS)},
        "enl": enl, "enl_combine": combine_how, "enl_used": combine(enl, combine_how),
        "boxes_considered": n_boxes, "enl_percentiles_over_boxes": spread,
    }
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
