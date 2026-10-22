"""Run the reference CFAR detectors (external, GPL-3.0) and turn their output into objects.

`run(detector, raster, mask, pfas, ...)` calls `external/cfar_runner/run_cfar.py` in the
`cfar-bench` env with `subprocess` and reads back one binary detection raster per PFA. Each
raster becomes connected-component objects with the **same schema as the CA-CFAR output**
(`detections.to_geodataframe`), plus `detector`, `pfa`, `pfa_per_channel` and `length_m`.
Attributes the reference detectors don't produce (clutter background, contrast, grown
structure) are NaN.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from iceberg_sar.benchmark.pfa import detector_channel_pfa
from iceberg_sar.config import Config
from iceberg_sar.detections import _LabelStats, to_db, to_geodataframe

DETECTORS = ("gamma", "lognorm", "k", "wishart", "nis", "idpolrad")
# Per-channel detectors that take an ENL; with `enl_bands` each band gets its own (runner flag)
PER_BAND_ENL = frozenset({"gamma", "k"})
BANDS = ("HH", "HV")
NAN_COLUMNS = ("structure_px", "background_db", "contrast_db", "copol_contrast_db")


class WindowLimitError(ValueError):
    """A CFAR window is wider than the library's stencil and cropping wasn't allowed."""


@dataclass(frozen=True)
class DetectorSettings:
    name: str
    wi_px: int
    wo_px: int
    allow_window_crop: bool = False


def odd_px(metres: float, pixel_m: float) -> int:
    """Window diameter in metres -> nearest odd pixel count (ties round up), at least 1."""
    return max(1, 2 * int(np.floor(metres / pixel_m / 2.0)) + 1)


def detector_settings(name: str, pixel_m: float, bench: dict[str, Any]) -> DetectorSettings:
    """Pixel windows for `name` at `pixel_m`; fail loudly if one exceeds the library limit."""
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector '{name}', expected one of {DETECTORS}")
    d = bench["detectors"][name]
    s = DetectorSettings(name, odd_px(d["inner_m"], pixel_m), odd_px(d["outer_m"], pixel_m),
                         bool(d.get("allow_window_crop", False)))
    if s.wi_px >= s.wo_px:
        raise ValueError(f"{name}: inner window {s.wi_px} px must be smaller than "
                         f"outer {s.wo_px} px")
    limit = int(bench["window_limit_px"])
    if s.wo_px > limit and not s.allow_window_crop:
        raise WindowLimitError(
            f"{name}: outer window {d['outer_m']} m = {s.wo_px} px at {pixel_m:g} m exceeds the "
            f"library's {limit} px limit. Multilook the raster or set allow_window_crop.")
    return s


def pixel_size_m(path: Path) -> float:
    with rasterio.open(path) as src:
        a, e = abs(src.transform.a), abs(src.transform.e)
    if not np.isclose(a, e, rtol=1e-3):
        raise ValueError(f"{path.name}: pixels are not square ({a} x {e} m)")
    return float(a)


def pfa_tag(pfa: float) -> str:
    """Same file tag as the runner's: 1e-09."""
    return f"{pfa:.0e}".replace("+", "")


def runner_command(
    bench: dict[str, Any], root: Path, s: DetectorSettings, raster: Path, mask: Path,
    pfas: Sequence[float], enl: float, out_dir: Path,
    enl_bands: dict[str, float] | None = None,
) -> list[str]:
    cmd = [str(root / bench["cfar_env_python"]), str(root / bench["cfar_runner"]),
           "--image", str(raster), "--mask", str(mask), "--detector", s.name,
           "--wi", str(s.wi_px), "--wo", str(s.wo_px), "--enl", f"{enl:.6g}",
           "--method", bench["merge"], "--minsize", str(bench["minsize_px"]),
           "--min-ring-fraction", str(bench["min_ring_fraction"]),
           "--masked-as-nan" if bench["masked_as_nan"] else "--no-masked-as-nan",
           "--out-dir", str(out_dir)]
    for p in pfas:
        cmd += ["--pfa", repr(float(p))]
    if s.allow_window_crop:
        cmd.append("--allow-window-crop")
    if enl_bands is not None and s.name in PER_BAND_ENL:
        cmd += ["--enl-hh", f"{enl_bands['HH']:.6g}", "--enl-hv", f"{enl_bands['HV']:.6g}"]
    return cmd


def _major_axis_px(st: _LabelStats, area: np.ndarray) -> np.ndarray:
    """Length along the major axis from second moments: sqrt(12 λ1), the side of a uniform
    rectangle with the same spread, so a straight run of n pixels has length n (1 px -> 1)."""
    r, c = st.rows.astype(np.float64), st.cols.astype(np.float64)
    mr, mc = st.sum(vals=r) / area, st.sum(vals=c) / area
    idx = st.lab - 1
    dr, dc = r - mr[idx], c - mc[idx]
    # +1/12: variance of a unit pixel, so single pixels and lines get their true length
    vrr = st.sum(vals=dr * dr) / area + 1 / 12
    vcc = st.sum(vals=dc * dc) / area + 1 / 12
    vrc = st.sum(vals=dr * dc) / area
    lam = (vrr + vcc) / 2 + np.sqrt(((vrr - vcc) / 2) ** 2 + vrc**2)
    return np.sqrt(12.0 * lam)


def object_table(
    det: np.ndarray, bands: dict[str, np.ndarray], connectivity: int, max_object_px: int,
) -> pd.DataFrame:
    """Connected components of `det` -> per-object pixel stats (CA-CFAR table columns)."""
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, n = ndimage.label(det, structure=structure)
    if n == 0:
        return pd.DataFrame()
    st = _LabelStats(labels, n)
    area = st.count()
    slices = ndimage.find_objects(labels)
    extent = np.array([max(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in slices])
    keep = area <= max_object_px
    out = pd.DataFrame({
        "row": (st.sum(vals=st.rows.astype(np.float64)) / area)[keep],
        "col": (st.sum(vals=st.cols.astype(np.float64)) / area)[keep],
        "area_px": area[keep].astype(int),
        "extent_px": extent[keep],
    })
    for col in NAN_COLUMNS[:2]:
        out[col] = np.nan
    for pol, arr in bands.items():
        finite = np.isfinite(arr)
        vals = np.where(finite, arr, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = st.sum(vals) / st.sum(finite)
        out[f"peak_db_{pol}"] = to_db(st.max(vals)[keep])
        out[f"mean_db_{pol}"] = to_db(mean[keep])
    for col in NAN_COLUMNS[2:]:
        out[col] = np.nan
    out["length_px"] = _major_axis_px(st, area)[keep]
    return out


def objects_from_raster(
    det_path: Path, raster: Path, *, detector: str, pfa: float, merge: str,
    connectivity: int, max_object_px: int, scene_id: str,
    inc_path: Path | None = None, ice_mask: Path | None = None,
) -> gpd.GeoDataFrame:
    with rasterio.open(det_path) as src:
        det = src.read(1) > 0
    with rasterio.open(raster) as src:
        bands = {pol: src.read(i + 1).astype(np.float32) for i, pol in enumerate(BANDS)}
        transform, crs = src.transform, src.crs
    table = object_table(det, bands, connectivity, max_object_px)
    gdf = to_geodataframe(table, transform, crs, scene_id, inc_path, ice_mask)
    if gdf.empty:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        gdf["length_m"] = (gdf.pop("length_px") * abs(transform.a)).round(1)
    gdf["detector"] = detector
    gdf["pfa"] = pfa
    gdf["pfa_per_channel"] = detector_channel_pfa(detector, pfa, merge)  # type: ignore[arg-type]
    return gdf


def run(
    detector: str,
    raster: Path,
    mask: Path,
    pfas: Sequence[float],
    *,
    enl: float,
    enl_bands: dict[str, float] | None = None,
    cfg: Config,
    out_dir: Path,
    scene_id: str,
    inc_path: Path | None = None,
    ice_mask: Path | None = None,
) -> dict[float, gpd.GeoDataFrame]:
    """Run `detector` at each total PFA; returns (and writes) one GeoDataFrame per PFA.

    `raster`: 2-band GeoTIFF [HH, HV], linear intensity. `mask`: nonzero = valid.
    `enl` is the single ENL (library behaviour, and always Wishart's). With `enl_bands`
    ({"HH": .., "HV": ..}), gamma and K run each band with its own ENL instead.
    Writes `<detector>_pfa<pfa>.tif` / `.geojson` and `<detector>_run.json` to `out_dir`.
    """
    bench = cfg.section("benchmark")
    s = detector_settings(detector, pixel_size_m(raster), bench)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = runner_command(bench, cfg.root, s, raster, mask, pfas, enl, out_dir, enl_bands)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cfar runner failed for {detector} (exit {proc.returncode}):\n"
                           f"{proc.stderr.strip()[-3000:]}")
    meta = json.loads((out_dir / f"{detector}_run.json").read_text(encoding="utf-8"))
    results: dict[float, gpd.GeoDataFrame] = {}
    for p, r in zip(pfas, meta["runs"], strict=True):
        gdf = objects_from_raster(
            out_dir / r["raster"], raster, detector=detector, pfa=float(p),
            merge=bench["merge"], connectivity=int(bench["object_connectivity"]),
            max_object_px=int(bench["max_object_px"]), scene_id=scene_id,
            inc_path=inc_path, ice_mask=ice_mask)
        gdf.to_file(out_dir / f"{detector}_pfa{pfa_tag(float(p))}.geojson", driver="GeoJSON")
        results[float(p)] = gdf
    return results
