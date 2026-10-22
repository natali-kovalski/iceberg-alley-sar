"""Milestone 1d, Path B: iceberg labels rebuilt from the Sentinel-2 L1C scene.

Every output of this module is RECONSTRUCTED LABELS, not the paper's validation set.

The paper (sec. 3.4, "Validation data"):

  "In the Sentinel-2 image icebergs were identified by applying an N-Sigma CFAR detector to the
   sum of the high-resolution bands: B2, B3, B4, and B8. Pixels brighter than the mean
   background plus 4 times the standard deviations were marked as possible icebergs. The
   results were then manually checked to remove artifacts from clouds and land, and a few
   icebergs missed by the automatic detection were added."

What is automated here, and what the paper leaves open (all values in `labels:` in config):
- Background window: not given. The paper's SAR window is reused (square, 360 m guard,
  520 m outer); mean and standard deviation over the ring between them.
- "Manually checked ... clouds and land": replaced by rules. OSM land polygons (buffered
  `land_buffer_m`) and large L1C cloud-mask polygons are excluded; an object must be dark in
  SWIR (B11), where ice is dark and clouds are bright.
- "a few icebergs missed ... were added": not reproducible, skipped.
- Size: length = major axis of the object (as the paper), classes from Table 2.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy import ndimage

from iceberg_sar.benchmark.download import s2_band
from iceberg_sar.config import Config

SIZE_CLASSES = (("small", 0.0, 60.0), ("medium", 60.0, 120.0), ("large", 120.0, np.inf))
LABEL_SOURCE = "reconstructed (Path B, Sentinel-2 4-sigma)"


def radiometry(safe: Path) -> tuple[float, float]:
    """(quantification value, additive offset) for L1C DN -> TOA reflectance.

    Baseline 04.00+ products add RADIO_ADD_OFFSET (-1000); N0207 has none.
    """
    text = next(safe.glob("MTD_MSIL1C.xml")).read_text(encoding="utf-8")
    q = float(re.search(r"<QUANTIFICATION_VALUE[^>]*>([\d.]+)<", text)[1])
    off = re.search(r"<RADIO_ADD_OFFSET[^>]*>(-?[\d.]+)<", text)
    return q, float(off[1]) if off else 0.0


def read_reflectance(path: Path, quant: float, offset: float,
                     out_shape: tuple[int, int] | None = None) -> np.ndarray:
    """TOA reflectance (float32, NaN where DN = 0), optionally resampled (nearest) to a shape."""
    with rasterio.open(path) as src:
        dn = src.read(1, out_shape=out_shape, resampling=Resampling.nearest)
    out = (dn.astype(np.float32) + offset) / quant
    out[dn == 0] = np.nan
    return out


def ring_stats(x: np.ndarray, valid: np.ndarray, guard_px: int, outer_px: int,
               block_rows: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, std and valid fraction over the square ring between `guard_px` and `outer_px`.

    Invalid pixels are left out of the sums. Computed in row blocks (with a margin of half
    the outer window) to keep the float64 temporaries small on 10980 px tiles.
    """
    h = outer_px // 2
    n_ring = outer_px**2 - guard_px**2
    xv = np.where(valid, x, 0.0)
    mean = np.full(x.shape, np.nan, np.float32)
    std = np.full(x.shape, np.nan, np.float32)
    frac = np.zeros(x.shape, np.float32)
    for r0 in range(0, x.shape[0], block_rows):
        r1 = min(r0 + block_rows, x.shape[0])
        a0, a1 = max(0, r0 - h), min(x.shape[0], r1 + h)
        b = xv[a0:a1].astype(np.float64)
        v = valid[a0:a1].astype(np.float64)

        def box(arr: np.ndarray, size: int) -> np.ndarray:
            return ndimage.uniform_filter(arr, size, mode="constant") * size**2

        n = box(v, outer_px) - box(v, guard_px)
        s1 = box(b, outer_px) - box(b, guard_px)
        s2 = box(b * b, outer_px) - box(b * b, guard_px)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s1 / n
            sd = np.sqrt(np.maximum(s2 / n - m * m, 0.0))
        sl = slice(r0 - a0, r0 - a0 + (r1 - r0))
        mean[r0:r1], std[r0:r1] = m[sl], sd[sl]
        frac[r0:r1] = n[sl] / n_ring
    return mean, std, frac


def nsigma_detect(x: np.ndarray, valid: np.ndarray, p: dict[str, Any], pixel_m: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pixels brighter than ring mean + n_sigma * ring std (paper: 4), and the ring mean."""
    guard = _odd(p["guard_m"] / pixel_m)
    outer = _odd(p["outer_m"] / pixel_m)
    mean, std, frac = ring_stats(x, valid, guard, outer)
    with np.errstate(invalid="ignore"):
        det = valid & (frac >= p["min_ring_fraction"]) & (x > mean + p["n_sigma"] * std)
    return det, mean


def _odd(px: float) -> int:
    n = max(1, round(px))
    return n if n % 2 else n + 1


def size_class(length_m: np.ndarray) -> np.ndarray:
    out = np.empty(len(length_m), dtype=object)
    for name, lo, hi in SIZE_CLASSES:
        out[(length_m >= lo) & (length_m < hi)] = name
    return out


def object_table(det: np.ndarray, x: np.ndarray, swir: np.ndarray, pixel_m: float
                 ) -> pd.DataFrame:
    """Connected objects (8-connectivity): centroid, area, major-axis length, brightness, SWIR."""
    from skimage.measure import regionprops_table

    labels, n = ndimage.label(det, structure=np.ones((3, 3), bool))
    if n == 0:
        return pd.DataFrame(columns=["row", "col", "area_px", "length_m", "peak", "mean",
                                     "swir_mean"])
    t = regionprops_table(labels, intensity_image=np.dstack([x, np.nan_to_num(swir, nan=1.0)]),
                          properties=("centroid", "area", "major_axis_length",
                                      "intensity_max", "intensity_mean"))
    # One pixel has a zero major axis; it is still one pixel (10 m) long.
    length_px = np.maximum(t["major_axis_length"], 1.0)
    return pd.DataFrame({
        "row": t["centroid-0"], "col": t["centroid-1"], "area_px": t["area"].astype(int),
        "length_m": length_px * pixel_m, "peak": t["intensity_max-0"],
        "mean": t["intensity_mean-0"], "swir_mean": t["intensity_mean-1"],
    })


def cloud_polygons(safe: Path, crs: str, min_area_m2: float, buffer_m: float
                   ) -> gpd.GeoSeries:
    """Large polygons of the L1C cloud mask (MSK_CLOUDS_B00.gml), buffered. Small ones are
    skipped: the mask's brightness tests also flag bright bergs."""
    hits = sorted(safe.glob("GRANULE/*/QI_DATA/MSK_CLOUDS_B00.gml"))
    try:
        g = gpd.read_file(hits[0]) if hits else gpd.GeoDataFrame(geometry=[])
    except Exception:  # an empty mask is written without features; GDAL can't open it
        g = gpd.GeoDataFrame(geometry=[])
    if g.empty:
        return gpd.GeoSeries([], crs=crs)
    if g.crs is None:
        g = g.set_crs(crs)
    g = g.to_crs(crs)
    big = g[g.area >= min_area_m2]
    return big.geometry.buffer(buffer_m)


def tile_targets(cfg: Config, safe: Path, p: dict[str, Any], log: Any = print
                 ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Targets (points, tile CRS) and the usable area (polygons) of one S2 tile."""
    from iceberg_sar.benchmark.landmask import land_polygons

    quant, offset = radiometry(safe)
    with rasterio.open(s2_band(safe, p["bands"][0])) as src:
        shape, transform, crs = src.shape, src.transform, src.crs
        pixel_m = abs(transform.a)
        bbox = transform_bounds(crs, "EPSG:4326", *src.bounds)
    x = np.zeros(shape, np.float32)
    nodata = np.zeros(shape, bool)
    for b in p["bands"]:
        r = read_reflectance(s2_band(safe, b), quant, offset)
        nodata |= ~np.isfinite(r)
        x += np.nan_to_num(r)
    swir = read_reflectance(s2_band(safe, p["swir_band"]), quant, offset, out_shape=shape)
    land = land_polygons(cfg, bbox, crs.to_string()).geometry
    if p["land_buffer_m"]:
        land = land.buffer(p["land_buffer_m"])
    clouds = cloud_polygons(safe, crs.to_string(), p["cloud_min_area_m2"], p["cloud_buffer_m"])
    excl = list(land) + list(clouds)
    excluded = rasterize(((g, 1) for g in excl if not g.is_empty), out_shape=shape,
                         transform=transform, fill=0, dtype="uint8").astype(bool) if excl else \
        np.zeros(shape, bool)
    valid = ~nodata & ~excluded
    det, bg = nsigma_detect(x, valid, p, pixel_m)
    t = object_table(det, x, swir, pixel_m)
    n_raw = len(t)
    # Contrast: object peak over the ring mean at its centroid (open water ~0.2 summed).
    r, c = t["row"].round().astype(int).to_numpy(), t["col"].round().astype(int).to_numpy()
    t["contrast"] = t["peak"].to_numpy() / bg[r, c]
    keep = ((t["area_px"] >= p["min_area_px"]) & (t["area_px"] * pixel_m**2 <= p["max_area_m2"])
            & (t["contrast"] >= p["min_contrast"]) & (t["swir_mean"] <= p["max_swir"]))
    t = t[keep].reset_index(drop=True)
    xs, ys = rasterio.transform.xy(transform, t["row"].to_numpy(), t["col"].to_numpy())
    pts = gpd.GeoDataFrame(t.assign(tile=safe.name.split("_")[5], size_class=size_class(
        t["length_m"].to_numpy())), geometry=gpd.points_from_xy(xs, ys), crs=crs)
    log(f"  {safe.name[:60]}: {int(det.sum())} px over {p['n_sigma']} sigma, {n_raw} objects, "
        f"{len(t)} kept (size / contrast / SWIR rules); clouds {len(clouds)} polygons")
    return pts, _area_polygons(valid, transform, crs)


def _area_polygons(valid: np.ndarray, transform: rasterio.Affine, crs: Any,
                   factor: int = 10) -> gpd.GeoDataFrame:
    """Polygons where a `factor`-coarser block is mostly valid (100 m blocks at 10 m)."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    h, w = valid.shape[0] // factor * factor, valid.shape[1] // factor * factor
    coarse = valid[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)) > 0.5
    t = transform * rasterio.Affine.scale(factor)
    geoms = [shape(g) for g, v in shapes(coarse.astype(np.uint8), mask=coarse, transform=t) if v]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def dedupe(points: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    """One target per object where tiles overlap: keep the largest within `tol_m`."""
    from scipy.spatial import cKDTree

    if points.empty:
        return points
    pts = points.sort_values("area_px", ascending=False).reset_index(drop=True)
    xy = np.column_stack([pts.geometry.x, pts.geometry.y])
    keep = np.ones(len(pts), bool)
    for i, j in sorted(cKDTree(xy).query_pairs(tol_m)):
        if keep[i] and keep[j]:
            keep[j] = False
    return pts[keep].reset_index(drop=True)


def s2_targets(cfg: Config, log: Any = print) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """All tiles: deduplicated targets and their union of usable area, in the scene CRS."""
    from iceberg_sar.benchmark.download import raw_dir

    fc = cfg.section("benchmark")["faerch2023"]
    p = fc["labels"]
    pts, areas = [], []
    for s in fc["scenes"]:
        if s["sensor"] != "Sentinel-2":
            continue
        t, a = tile_targets(cfg, raw_dir(cfg) / f"{s['product_id']}.SAFE", p, log)
        pts.append(t.to_crs(fc["crs"]))
        areas.append(a.to_crs(fc["crs"]))
    targets = dedupe(pd.concat(pts, ignore_index=True), p["dedupe_m"])
    area = gpd.GeoDataFrame(geometry=[pd.concat(areas).union_all()], crs=fc["crs"])
    return targets, area
