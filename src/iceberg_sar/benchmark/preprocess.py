"""Benchmark inputs: dual-pol stacks for the reference CFAR runner.

`multilook_stack` turns co-registered HH / HV rasters (linear intensity, NaN = masked) into
the runner's input: one 2-band GeoTIFF [HH, HV] plus a uint8 valid mask, averaged over
`factor` x `factor` blocks. Multilooking (averaging intensity over neighbouring pixels)
reduces speckle, so the ENL goes up and has to be re-estimated on the result.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

BANDS = ("HH", "HV")


def gpt_command(snap: dict[str, Any], graph: Path, src: Path, out: Path, crs: str,
                pixel_m: float) -> list[str]:
    """`gpt` call for the preprocessing graph; graph variables are passed as -P options."""
    return [str(Path(snap["gpt"]).expanduser()), str(graph), "-c", str(snap["cache"]),
            "-q", str(snap["threads"]), f"-Pinput={src}", f"-Poutput={out}", f"-Pcrs={crs}",
            f"-PpixelSpacing={pixel_m:g}", f"-Presampling={snap['resampling']}"]


def run_snap(snap: dict[str, Any], graph: Path, src: Path, out: Path, crs: str,
             pixel_m: float, log: Any = print) -> Path:
    """Run the SNAP graph (skipped if `out` exists). SNAP's own log goes to <out>.gpt.log."""
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = gpt_command(snap, graph, src, out, crs, pixel_m)
    log(" ".join(cmd))
    t0 = time.perf_counter()
    log_path = out.with_suffix(".gpt.log")
    with log_path.open("w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if res.returncode != 0 or not out.is_file():
        raise RuntimeError(f"gpt failed (exit {res.returncode}); see {log_path}")
    log(f"gpt done in {time.perf_counter() - t0:.0f} s")
    return out


def split_snap_output(snap_tif: Path, stack: Path, inc: Path, mask: Path) -> dict[str, Any]:
    """SNAP's [Sigma0_HH, Sigma0_HV, incidence] GeoTIFF -> [HH, HV] float32 stack (NaN outside
    the swath), incidence angle (deg) and a uint8 valid mask (1 = inside swath, HH > 0).

    SNAP writes 0 outside the swath. Thermal noise removal also clips noise-dominated HV pixels
    to 0, so the swath is taken from HH alone and HV zeros inside it stay valid zeros.
    """
    with rasterio.open(snap_tif) as src:
        if src.count < 3:
            raise ValueError(f"{snap_tif}: expected 3 bands (HH, HV, incidence), got {src.count}")
        hh, hv, ang = (src.read(i).astype(np.float32) for i in (1, 2, 3))
        profile = src.profile
    valid = hh > 0
    profile.update(driver="GTiff", compress="deflate", tiled=True, blockxsize=512,
                   blockysize=512, BIGTIFF="IF_SAFER")
    stack.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stack, "w", **{**profile, "count": 2, "dtype": "float32",
                                       "nodata": np.nan, "predictor": 3}) as dst:
        dst.write(np.where(valid, hh, np.nan), 1)
        dst.write(np.where(valid, hv, np.nan), 2)
        dst.descriptions = BANDS
        dst.update_tags(units="sigma0, linear intensity", source=snap_tif.name)
    with rasterio.open(inc, "w", **{**profile, "count": 1, "dtype": "float32",
                                     "nodata": np.nan, "predictor": 3}) as dst:
        dst.write(np.where(valid, ang, np.nan), 1)
        dst.descriptions = ("incidence_angle_ellipsoid_deg",)
    with rasterio.open(mask, "w", **{**profile, "count": 1, "dtype": "uint8", "nodata": None,
                                      "predictor": 1}) as dst:
        dst.write(valid.astype(np.uint8), 1)
    return {"shape": hh.shape, "pixel_m": (abs(profile["transform"].a),
                                           abs(profile["transform"].e)),
            "valid_px": int(valid.sum()), "hv_zero_in_swath": float((hv[valid] <= 0).mean())}


def block_mean(x: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of finite values per `factor` x `factor` block, and the finite share per block."""
    h, w = (x.shape[0] // factor) * factor, (x.shape[1] // factor) * factor
    b = x[:h, :w].reshape(h // factor, factor, w // factor, factor)
    finite = np.isfinite(b)
    n = finite.sum(axis=(1, 3))
    s = np.where(finite, b, 0.0).sum(axis=(1, 3), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
    return mean.astype(np.float32), n / factor**2


def multilook_stack(
    paths: dict[str, Path],
    factor: int,
    out_raster: Path,
    out_mask: Path,
    min_valid: float = 1.0,
    block_rows: int = 2048,
) -> tuple[Path, Path]:
    """HH/HV -> 2-band multilooked stack + mask (1 = valid in both bands)."""
    if block_rows % factor:
        raise ValueError("block_rows must be a multiple of the multilook factor")
    srcs = {pol: rasterio.open(paths[pol]) for pol in BANDS}
    try:
        ref = srcs["HH"]
        if (srcs["HV"].shape, srcs["HV"].transform) != (ref.shape, ref.transform):
            raise ValueError("HH and HV are not on the same grid")
        h, w = ref.height // factor, ref.width // factor
        transform = ref.transform * ref.transform.scale(factor, factor)
        common = dict(driver="GTiff", height=h, width=w, crs=ref.crs, transform=transform,
                      compress="deflate", tiled=True, blockxsize=512, blockysize=512)
        out_raster.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_raster, "w", count=2, dtype="float32", nodata=np.nan,
                           predictor=3, **common) as dr, \
                rasterio.open(out_mask, "w", count=1, dtype="uint8", **common) as dm:
            dr.descriptions = BANDS
            for r0 in range(0, h * factor, block_rows):
                nrows = min(block_rows, h * factor - r0)
                win = Window(0, r0, w * factor, nrows)
                out_win = Window(0, r0 // factor, w, nrows // factor)
                valid = np.ones((nrows // factor, w), bool)
                for i, pol in enumerate(BANDS):
                    mean, frac = block_mean(srcs[pol].read(1, window=win), factor)
                    ok = (frac >= min_valid) & (mean > 0)
                    valid &= ok
                    dr.write(np.where(ok, mean, np.nan).astype(np.float32), i + 1, window=out_win)
                dm.write(valid.astype(np.uint8), 1, window=out_win)
    finally:
        for s in srcs.values():
            s.close()
    return out_raster, out_mask
