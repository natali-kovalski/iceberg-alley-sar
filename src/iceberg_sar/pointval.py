"""Point-level validation: SAR detections vs. truth points (IIP sightings, Sentinel-2 targets).

Only targets both sources could see are compared:
- truth points must fall on valid (unmasked) SAR pixels at SAR time, i.e. after undoing the
  drift offset, away from land, pack ice and the scene edge;
- detections are only counted against optical truth where the optical image was clear
  (pass `truth_area`). Without it (aircraft surveys cover an unknown swath) precision is
  not defined, only recall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from iceberg_sar.config import Config
from iceberg_sar.matching import OffsetResult, estimate_offset, match_points


@dataclass(frozen=True)
class PointValParams:
    max_dist_m: float = 500.0  # match tolerance after drift correction
    offset_search_m: float = 15_000.0  # drift search half-width when sources are not simultaneous
    offset_step_m: float = 250.0
    simultaneous_s: float = 1800.0  # closer in time than this: no drift correction
    min_offset_matches: int = 5  # a drift peak needs this many matches and 3x the chance level
    valid_block_px: int = 10  # truth must sit in a fully valid block of this many SAR pixels


def _xy(gdf: gpd.GeoDataFrame, crs: object) -> np.ndarray:
    g = gdf.to_crs(crs).geometry
    return np.column_stack([g.x.to_numpy(), g.y.to_numpy()]) if len(g) else np.empty((0, 2))


def sar_valid_at(masked_tif: Path, xy: np.ndarray, block_px: int) -> np.ndarray:
    """True where the SAR raster is valid over the whole `block_px` block containing the point."""
    with rasterio.open(masked_tif) as src:
        h, w = src.height // block_px, src.width // block_px
        ok = np.zeros((h, w), dtype=bool)
        step = block_px * 256
        for r0 in range(0, h * block_px, step):
            n = min(step, h * block_px - r0)
            a = src.read(1, window=Window(0, r0, w * block_px, n))
            blocks = np.isfinite(a).reshape(n // block_px, block_px, w, block_px)
            ok[r0 // block_px : (r0 + n) // block_px] = blocks.all(axis=(1, 3))
        t = src.transform * rasterio.Affine.scale(block_px)
    cols, rows = ~t * (xy[:, 0], xy[:, 1])
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.zeros(len(xy), dtype=bool)
    out[inside] = ok[rows[inside], cols[inside]]
    return out


def validate_points(
    detections: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    masked_tif: Path,
    sar_time: pd.Timestamp,
    p: PointValParams,
    truth_area: gpd.GeoDataFrame | None = None,
) -> tuple[dict, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Score detections against truth points. Returns summary, truth and detections with flags."""
    with rasterio.open(masked_tif) as src:
        crs = src.crs
    det_xy, truth_xy = _xy(detections, crs), _xy(truth, crs)
    gaps = np.abs((truth["time"] - sar_time).dt.total_seconds())
    gap_s = float(np.median(gaps)) if len(truth) else 0.0
    if gap_s > p.simultaneous_s:
        off = estimate_offset(det_xy, truth_xy, p.offset_search_m, p.offset_step_m, p.max_dist_m)
    else:
        off = OffsetResult(0.0, 0.0, 0, 0.0)
    # A weak peak is a chance alignment: applying it would score against random positions.
    reliable = gap_s <= p.simultaneous_s or (
        off.n_matched >= max(p.min_offset_matches, 3 * off.chance_n_matched))
    shift = np.array([off.dx_m, off.dy_m]) if reliable else np.zeros(2)

    truth_ok = sar_valid_at(masked_tif, truth_xy - shift, p.valid_block_px)
    det_ok = np.ones(len(det_xy), dtype=bool)
    if truth_area is not None and len(det_xy):
        moved = gpd.GeoSeries(gpd.points_from_xy(*(det_xy + shift).T), crs=crs)
        moved = moved.to_crs(truth_area.crs)
        det_ok = moved.within(truth_area.union_all()).to_numpy()

    m = match_points(det_xy[det_ok] + shift, truth_xy[truth_ok], p.max_dist_m)
    det_idx, truth_idx = np.flatnonzero(det_ok), np.flatnonzero(truth_ok)
    det_out = detections.assign(compared=det_ok, matched=False)
    truth_out = truth.assign(compared=truth_ok, matched=False, match_dist_m=np.nan)
    det_out.iloc[det_idx[m.pairs[:, 0]], det_out.columns.get_loc("matched")] = True
    truth_out.iloc[truth_idx[m.pairs[:, 1]], truth_out.columns.get_loc("matched")] = True
    dist_col = truth_out.columns.get_loc("match_dist_m")
    truth_out.iloc[truth_idx[m.pairs[:, 1]], dist_col] = m.distances_m

    summary = {
        "time_gap_h": round(gap_s / 3600, 2),
        "offset": asdict(off),
        "offset_applied": bool(reliable and shift.any()),
        "drift_unresolved": not reliable,
        "truth_total": len(truth),
        "truth_compared": m.n_truth,
        "detections_total": len(detections),
        "detections_compared": m.n_detections,
        "matched": m.true_positives,
        "recall": round(m.recall, 3),
        "precision": round(m.precision, 3) if truth_area is not None else None,
        "median_match_dist_m": (round(float(np.median(m.distances_m))) if len(m.distances_m)
                                else None),
    }
    return summary, truth_out, det_out


TRUTH_KINDS = ("iip-satellite", "iip-aircraft", "s2")


def scene_footprint(raw_dir: Path, name: str) -> gpd.GeoSeries:
    """Largest valid-data polygon of a HyP3 product (its `_shape.shp`), EPSG:4326."""
    fp = gpd.read_file(raw_dir / name / f"{name}_shape.shp")
    return fp.iloc[[int(fp.area.argmax())]].geometry.to_crs("EPSG:4326")


def at_cell_centres(pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """IIP sightings with truncated positions moved to their 0.1' cell centre."""
    from iceberg_sar.iip import cell_centres

    lat, lon = cell_centres(pts["lat"].to_numpy(), pts["lon"].to_numpy())
    return gpd.GeoDataFrame(pts.assign(lat=lat, lon=lon),
                            geometry=gpd.points_from_xy(lon, lat), crs=pts.crs)


def load_truth(
    cfg: Config, name: str, sar_time: pd.Timestamp, kind: str
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    """Truth points for one scene, and the area they cover if known (Sentinel-2 only)."""
    from iceberg_sar.iip import load_sightings, select_sightings

    v = cfg.section("validation")
    if kind == "s2":
        d = cfg.root / v["s2_dir"]
        day = sar_time.strftime("%Y-%m-%d")
        pts = gpd.read_file(d / f"{day}_s2_targets.geojson")
        pts["time"] = pd.to_datetime(pts["time"])
        # Files written before _no_targets() existed can hold these as strings.
        num = [c for c in ("row", "col", "area_m2", "peak_reflectance") if c in pts]
        pts[num] = pts[num].apply(pd.to_numeric)
        return pts, gpd.read_file(d / f"{day}_s2_clear.geojson")
    if kind not in TRUTH_KINDS:
        raise ValueError(f"truth must be one of {TRUTH_KINDS}")
    files = sorted((cfg.root / v["iip_dir"]).glob("IIP_*IcebergSeason.csv"))
    if not files:
        raise FileNotFoundError("No IIP sightings; run `iip-sightings` first")
    sightings = pd.concat([load_sightings(f) for f in files], ignore_index=True)
    area = scene_footprint(cfg.path("raw"), name).iloc[0]
    if kind == "iip-satellite":
        w = pd.Timedelta(minutes=float(v["satellite_window_min"]))
        return at_cell_centres(
            select_sightings(sightings, area, sar_time - w, sar_time + w, satellite=True)), None
    w = pd.Timedelta(hours=float(v["aircraft_window_h"]))
    return at_cell_centres(
        select_sightings(sightings, area, sar_time, sar_time + w, satellite=False)), None


def point_val_params(cfg: Config) -> PointValParams:
    v = cfg.section("validation")
    return PointValParams(max_dist_m=float(v["max_dist_m"]),
                          offset_search_m=float(v["offset_search_m"]),
                          offset_step_m=float(v["offset_step_m"]))
