"""Automatic pack-ice mask: large, bright areas of *uniform* cross-pol backscatter.

Why cross-pol (HV/VH): sea ice returns strong volume scattering in cross-pol, while
open water stays near the noise floor regardless of wind. Co-pol (HH/VV) water gets
bright in high wind and can look like ice.

Observed on S1A 2025-05-08 (HV, 200 m cells): open water ~ -31 dB with CV ~1.1 (noise
dominated), pack ice ~ -20 dB with CV ~0.5. So at this scale pack ice is bright AND
smooth; the visible floe/lead texture is at km scale and is handled by closing.
A cell holding one iceberg is bright but spiky (high CV), so the CV cap protects icebergs.

Method (on a coarse grid, e.g. 200 m cells built from 20 m pixels):
1. Per cell: mean backscatter (linear -> dB) and coefficient of variation (std/mean).
2. Open-water reference = a low percentile of cell dB (robust if ice covers < ~half the scene).
3. Ice candidate = dB above reference by `min_excess_db` AND CV below `max_cv`.
4. Morphological closing (fills leads), remove patches < `min_area_km2` (icebergs are
   far smaller), then dilate by `buffer_m` to catch loose floes at the ice edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import shape


@dataclass(frozen=True)
class CoarseStats:
    mean_lin: np.ndarray  # mean linear power per cell (NaN if too few valid pixels)
    cv: np.ndarray  # std/mean of linear power per cell
    transform: Affine  # coarse-grid transform
    factor: int  # fine pixels per coarse cell (per side)
    fine_shape: tuple[int, int]
    crs: rasterio.crs.CRS


def coarse_stats(path: Path, factor: int, min_valid_frac: float = 0.5) -> CoarseStats:
    """Block mean and coefficient of variation of a linear-power raster (NaN = masked)."""
    with rasterio.open(path) as src:
        h, w, crs = src.height, src.width, src.crs
        ch, cw = -(-h // factor), -(-w // factor)  # ceil
        mean = np.full((ch, cw), np.nan, dtype=np.float32)
        cv = np.full((ch, cw), np.nan, dtype=np.float32)
        rows_per_read = factor * max(1, 2048 // factor)
        for r0 in range(0, h, rows_per_read):
            nrows = min(rows_per_read, h - r0)
            data = src.read(1, window=Window(0, r0, w, nrows)).astype(np.float64)
            # Pad to whole cells with NaN so edge cells use only their real pixels.
            ph, pw = -(-nrows // factor) * factor, cw * factor
            padded = np.full((ph, pw), np.nan)
            padded[:nrows, :w] = data
            blocks = padded.reshape(ph // factor, factor, cw, factor)
            valid = np.isfinite(blocks)
            n = valid.sum(axis=(1, 3))
            s1 = np.where(valid, blocks, 0).sum(axis=(1, 3))
            s2 = np.where(valid, blocks**2, 0).sum(axis=(1, 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                m = s1 / n
                sd = np.sqrt(np.maximum(s2 / n - m**2, 0))
                c = sd / m
            ok = n >= min_valid_frac * factor * factor
            i0 = r0 // factor
            mean[i0 : i0 + m.shape[0]] = np.where(ok, m, np.nan)
            cv[i0 : i0 + m.shape[0]] = np.where(ok, c, np.nan)
        transform = src.transform * Affine.scale(factor)
    return CoarseStats(mean, cv, transform, factor, (h, w), crs)


def detect_ice(
    mean_lin: np.ndarray,
    cv: np.ndarray,
    cell_m: float,
    min_excess_db: float,
    max_cv: float,
    min_area_km2: float,
    buffer_m: float,
    strip_min_km2: float = 0.5,
    water_percentile: float = 25.0,
    closing_m: float = 600.0,
    min_db: float | None = None,
) -> tuple[np.ndarray, float]:
    """Boolean ice mask on the coarse grid, plus the open-water reference level (dB).

    A cell is "bright" if it is `min_excess_db` above the open-water reference and, when
    `min_db` is set, also above that absolute level. HyP3 removes the HV noise floor, so on
    calm days the reference can sit near -40 dB and slightly brighter water by coasts and
    swath edges passes the relative test alone.

    Two rules, OR-ed, then buffered:
    - pack-ice cores: bright AND uniform (CV < max_cv), closed, >= min_area_km2.
    - ice strips/edges: any connected bright cells (any CV) covering >= strip_min_km2.
      Thin bands only partly fill a cell, so they look spiky like an iceberg cell, but an
      iceberg lights up 1-4 isolated cells, far below this area.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(mean_lin)
    finite = np.isfinite(db)
    if not finite.any():
        return np.zeros(mean_lin.shape, dtype=bool), float("nan")
    water_db = float(np.percentile(db[finite], water_percentile))
    bright = finite & (db > water_db + min_excess_db)
    if min_db is not None:
        bright &= db > min_db
    cell_km2 = cell_m**2 / 1e6

    r_close = max(1, round(closing_m / cell_m))
    core = ndimage.binary_closing(bright & (cv < max_cv), structure=_disk(r_close))
    core = _remove_small(core, min_area_km2 / cell_km2)

    strips = _remove_small(bright, strip_min_km2 / cell_km2, connectivity=2)

    ice = core | strips
    r_buf = round(buffer_m / cell_m)
    if r_buf > 0 and ice.any():
        ice = ndimage.binary_dilation(ice, structure=_disk(r_buf))
    return ice, water_db


def ice_core(mask: np.ndarray, cell_m: float, buffer_m: float) -> np.ndarray:
    """Undo detect_ice's buffer: erode the buffered mask by the same disk.

    Gives the ice before buffering, except that gaps narrower than ~2 x buffer_m between ice
    areas stay filled (erosion after dilation is a closing). Cells beyond the array border
    count as ice so cores touching the scene edge are not eaten away.
    """
    r_buf = round(buffer_m / cell_m)
    if r_buf <= 0 or not mask.any():
        return mask.copy()
    return ndimage.binary_erosion(mask, structure=_disk(r_buf), border_value=1)


def write_core_mask(mask_path: Path, buffer_m: float, out_path: Path) -> Path:
    """Unbuffered pack-ice mask next to a buffered one (same grid, 0/1, 255 = nodata)."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1) == 1
        cell_m = abs(src.transform.a)
        profile = src.profile
    core = ice_core(mask, cell_m, buffer_m)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(core.astype(profile["dtype"]), 1)
    return out_path


def _remove_small(mask: np.ndarray, min_cells: float, connectivity: int = 1) -> np.ndarray:
    """Keep connected components with at least `min_cells` cells."""
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, n = ndimage.label(mask, structure=structure)
    if not n:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_cells
    return keep[labels]


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x**2 + y**2 <= radius**2


def ice_polygons(mask: np.ndarray, transform: Affine, crs: rasterio.crs.CRS) -> gpd.GeoDataFrame:
    """Vectorize the coarse ice mask into EPSG:4326 polygons with area in km²."""
    features = shapes(mask.astype(np.uint8), mask=mask, transform=transform)
    geoms = [shape(g) for g, v in features if v]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["area_km2"] = (gdf.area / 1e6).round(2)
    return gdf.to_crs("EPSG:4326")


def write_mask(mask: np.ndarray, transform: Affine, crs: rasterio.crs.CRS, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1], count=1,
        dtype="uint8", crs=crs, transform=transform, nodata=255, compress="deflate",
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)
    return path


def apply_coarse_mask(path: Path, mask: np.ndarray, factor: int, block_rows: int = 2048) -> int:
    """Set fine pixels under True coarse cells to NaN, in place. Returns pixels newly masked."""
    n_masked = 0
    block_rows = factor * max(1, block_rows // factor)
    with rasterio.open(path, "r+") as dst:
        h, w = dst.height, dst.width
        for r0 in range(0, h, block_rows):
            nrows = min(block_rows, h - r0)
            coarse = mask[r0 // factor : -(-(r0 + nrows) // factor)]
            if not coarse.any():
                continue
            fine = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)[:nrows, :w]
            win = Window(0, r0, w, nrows)
            data = dst.read(1, window=win)
            newly = fine & np.isfinite(data)
            n_masked += int(newly.sum())
            data[fine] = np.nan
            dst.write(data, 1, window=win)
    return n_masked


def distance_to_ice_km(mask_path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Distance (km) from map points (mask CRS) to the nearest masked pack-ice cell.

    Measured to the edge of the *buffered* mask, so the distance to the ice itself is about
    `buffer_m` more. NaN if the scene has no pack ice or a point is off the grid.
    """
    with rasterio.open(mask_path) as src:
        ice = src.read(1) == 1
        transform = src.transform
    if not ice.any():
        return np.full(len(xs), np.nan)
    dist_km = ndimage.distance_transform_edt(~ice) * abs(transform.a) / 1000.0
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < ice.shape[0]) & (cols >= 0) & (cols < ice.shape[1])
    out = np.full(len(xs), np.nan)
    out[inside] = dist_km[rows[inside], cols[inside]]
    return out
