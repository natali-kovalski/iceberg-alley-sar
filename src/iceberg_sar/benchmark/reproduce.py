"""Milestone 1b-1c: the Færch 2023 Sentinel-1 scene as a 40 m sigma0 stack plus land mask.

File layout under `benchmark.faerch2023.dir`:
  raw/<S1 product>.zip                    downloaded by `bench-download`
  interim/<S1>_snap_40m.tif               SNAP output [Sigma0_HH, Sigma0_HV, incidence]
  interim/<S1>_40m_HHHV.tif               [HH, HV] linear sigma0, NaN outside the swath
  interim/<S1>_40m_inc.tif / _swath.tif   incidence angle (deg) / 1 = inside the swath
  interim/<S1>_40m_land.tif               1 = land or within buffer_m of it
  interim/<S1>_40m_valid.tif              1 = swath and not land (the CFAR mask)
  interim/<S1>_40m_enl.json               open-water ENL per band on the valid area
  quicklooks/                             dB quicklooks, mask overlay
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from iceberg_sar.benchmark import plots
from iceberg_sar.benchmark.enl import estimate_enl
from iceberg_sar.benchmark.landmask import land_mask, mask_overlay
from iceberg_sar.benchmark.preprocess import run_snap, split_snap_output
from iceberg_sar.config import Config


@dataclass(frozen=True)
class ScenePaths:
    product: str
    stem: str          # file prefix: the product, or `<product>_noTNR`
    raw_zip: Path
    snap: Path
    stack: Path
    inc: Path
    swath: Path
    land: Path
    valid: Path
    enl: Path
    quicklooks: Path


def s1_product(cfg: Config, s1_format: str | None = None) -> str:
    """The Sentinel-1 product to use: `s1_format` (GRDH / GRDM), else `benchmark.faerch2023.s1_format`."""
    fc = cfg.section("benchmark")["faerch2023"]
    fmt = (s1_format or fc["s1_format"]).upper()
    hits = [s["product_id"] for s in fc["scenes"]
            if s["sensor"] == "Sentinel-1" and f"_EW_{fmt}_" in s["product_id"]]
    if len(hits) != 1:
        raise ValueError(f"Expected one Sentinel-1 EW {fmt} product in the scene list, got {hits}")
    return hits[0]


def scene_paths(cfg: Config, s1_format: str | None = None) -> ScenePaths:
    fc = cfg.section("benchmark")["faerch2023"]
    s1 = s1_product(cfg, s1_format)
    base = cfg.root / fc["dir"]
    res = f"{fc['pixel_m']:g}m"
    i = base / "interim"
    # Diagnostic variant without thermal noise removal: own files, so the main stack stays
    st = s1 if fc["snap"].get("remove_thermal_noise", True) else f"{s1}_noTNR"
    return ScenePaths(
        product=s1, stem=st, raw_zip=base / "raw" / f"{s1}.zip", snap=i / f"{st}_snap_{res}.tif",
        stack=i / f"{st}_{res}_HHHV.tif", inc=i / f"{st}_{res}_inc.tif",
        swath=i / f"{st}_{res}_swath.tif", land=i / f"{st}_{res}_land.tif",
        valid=i / f"{st}_{res}_valid.tif", enl=i / f"{st}_{res}_enl.json",
        quicklooks=base / "quicklooks")


def preprocess_scene(cfg: Config, force: bool = False, s1_format: str | None = None,
                     log: Any = print) -> dict[str, Any]:
    """1b: SNAP graph -> [HH, HV] stack, incidence, swath mask, HH / HV dB quicklooks."""
    fc = cfg.section("benchmark")["faerch2023"]
    p = scene_paths(cfg, s1_format)
    if not p.raw_zip.is_file():
        raise FileNotFoundError(f"{p.raw_zip} missing; run `bench-download` first")
    if force:
        for f in (p.snap, p.stack):
            f.unlink(missing_ok=True)
    run_snap(fc["snap"], cfg.root / fc["snap"]["graph"], p.raw_zip, p.snap, fc["crs"],
             float(fc["pixel_m"]), log)
    if force or not p.stack.is_file():
        info = split_snap_output(p.snap, p.stack, p.inc, p.swath)
    else:
        with rasterio.open(p.stack) as src:
            info = {"shape": src.shape, "pixel_m": (abs(src.transform.a), abs(src.transform.e))}
    if not np.allclose(info["pixel_m"], fc["pixel_m"], atol=0.01):
        raise ValueError(f"Pixel spacing {info['pixel_m']} != {fc['pixel_m']} m")
    info["quicklooks"] = db_quicklooks(p, int(fc["quicklook_max_px"]))
    return info


def db_quicklooks(p: ScenePaths, max_px: int) -> list[Path]:
    """HH and HV in dB (2-98 % stretch over the swath), one PNG each."""
    import geopandas as gpd

    out = []
    with rasterio.open(p.stack) as src:
        for i, pol in enumerate(("HH", "HV"), start=1):
            db = plots.to_db(src.read(i))
            png = p.quicklooks / f"{p.stem}_{pol}_db.png"
            plots.detection_quicklook(db, gpd.GeoDataFrame(), png,
                                      f"{pol} sigma0 dB, {abs(src.transform.a):g} m",
                                      max_px, plots.db_stretch(db))
            out.append(png)
    return out


def build_land_mask(cfg: Config, s1_format: str | None = None, log: Any = print
                    ) -> dict[str, Any]:
    """1c: land mask on the scene grid, CFAR valid mask, overlay quicklook, ENL sidecar."""
    fc = cfg.section("benchmark")["faerch2023"]
    p = scene_paths(cfg, s1_format)
    if not p.stack.is_file():
        raise FileNotFoundError(f"{p.stack} missing; run `bench-preprocess` first")
    buffer_m = float(fc["land"]["buffer_m"])
    land_mask(cfg, p.stack, p.land, buffer_m)
    with rasterio.open(p.land) as lsrc, rasterio.open(p.swath) as ssrc:
        land, swath = lsrc.read(1) > 0, ssrc.read(1) > 0
        profile = ssrc.profile
    valid = swath & ~land
    with rasterio.open(p.valid, "w", **profile) as dst:
        dst.write(valid.astype(np.uint8), 1)
    with rasterio.open(p.stack) as src:
        hh_db = plots.to_db(src.read(1))
    png = mask_overlay(hh_db, land & swath, p.quicklooks / f"{p.stem}_landmask.png",
                       f"HH dB + OSM land, {buffer_m:g} m buffer (purple)",
                       plots.db_stretch(hh_db), int(fc["quicklook_max_px"]))
    # ENL on open water, as in 0c (median of 2 km boxes): compare with Table 1's 10.7.
    enl = estimate_enl(p.stack, p.valid, 2000.0, "auto", "min", sidecar=p.enl)
    log(f"ENL (median 2 km open-water box): HH {enl['enl']['HH']}, HV {enl['enl']['HV']}; "
        f"spread p10/p50/p90 {json.dumps(enl['enl_percentiles_over_boxes'])}")
    return {"land_px_in_swath": int((land & swath).sum()), "valid_px": int(valid.sum()),
            "overlay": png, "enl": enl}
