"""Point labels for the THOR head: IIP positions snapped to the berg, rendered as Gaussians.

IIP stores positions in tenths of an arc-minute (~185 m north-south, ~115 m east-west here),
coarser than a THOR token (80 m at patch 8). Each label is moved to the brightest HH peak
nearby when that peak clearly stands out; otherwise it stays and is flagged as unsnapped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from iceberg_sar.config import Config, metres_to_px


@dataclass(frozen=True)
class TargetParams:
    snap_m: float
    snap_background_m: float
    snap_min_db: float
    sigma_m: float

    @classmethod
    def from_config(cls, cfg: Config) -> TargetParams:
        t = cfg.section("thor")["targets"]
        return cls(snap_m=float(t["snap_m"]), snap_background_m=float(t["snap_background_m"]),
                   snap_min_db=float(t["snap_min_db"]), sigma_m=float(t["sigma_m"]))


def snap_points(
    db: np.ndarray, rc: np.ndarray, params: TargetParams, pixel_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move each (row, col) to the brightest 3x3-mean peak within `snap_m`, if it stands out.

    `db` is a dB image (NaN = invalid) covering the points. Contrast is the peak over the
    median of the ring between `snap_m` and `snap_background_m`. Returns the new points,
    whether each one moved, and its contrast (dB, NaN if not measurable).
    """
    r_in = metres_to_px(params.snap_m, pixel_m)
    r_out = metres_to_px(params.snap_background_m, pixel_m)
    lin = np.where(np.isfinite(db), 10 ** (db / 10), np.nan)
    smooth = ndimage.uniform_filter(np.nan_to_num(lin), 3)
    count = ndimage.uniform_filter(np.isfinite(lin).astype(np.float32), 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        smooth_db = 10 * np.log10(np.where(count > 0.5, smooth / count, np.nan))

    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    dist = np.hypot(yy, xx)
    inner, ring = dist <= r_in, (dist > r_in) & (dist <= r_out)

    out = rc.astype(np.float64).copy()
    moved = np.zeros(len(rc), bool)
    contrast = np.full(len(rc), np.nan)
    h, w = db.shape
    for i, (r, c) in enumerate(np.round(rc).astype(int)):
        r0, c0 = r - r_out, c - r_out
        if r0 < 0 or c0 < 0 or r + r_out >= h or c + r_out >= w:
            continue
        win = smooth_db[r0:r0 + 2 * r_out + 1, c0:c0 + 2 * r_out + 1]
        peak_win = np.where(inner, win, np.nan)
        if not np.isfinite(peak_win).any() or not np.isfinite(win[ring]).any():
            continue
        k = np.nanargmax(peak_win)
        contrast[i] = float(peak_win.flat[k] - np.nanmedian(win[ring]))
        if contrast[i] >= params.snap_min_db:
            dr, dc = np.unravel_index(k, win.shape)
            out[i] = (r0 + dr, c0 + dc)
            moved[i] = True
    return out, moved, contrast


def render_heatmap(
    rc: np.ndarray, shape: tuple[int, int], stride_px: int, sigma_px: float
) -> np.ndarray:
    """Gaussian per point on a grid of `shape` cells, each `stride_px` input pixels wide.

    Cell centres sit at (i + 0.5) * stride. Overlapping Gaussians take the max, and the cell
    holding each point is exactly 1: the positives of the CenterNet focal loss.
    """
    heat = np.zeros(shape, np.float32)
    cy = (np.arange(shape[0]) + 0.5) * stride_px
    cx = (np.arange(shape[1]) + 0.5) * stride_px
    for r, c in rc:
        g = np.exp(-((cy[:, None] - r) ** 2 + (cx[None, :] - c) ** 2) / (2 * sigma_px ** 2))
        np.maximum(heat, g.astype(np.float32), out=heat)
        i, j = int(r // stride_px), int(c // stride_px)
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            heat[i, j] = 1.0
    return heat
