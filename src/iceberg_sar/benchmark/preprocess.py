"""Benchmark inputs: dual-pol stacks for the reference CFAR runner.

`multilook_stack` turns co-registered HH / HV rasters (linear intensity, NaN = masked) into
the runner's input: one 2-band GeoTIFF [HH, HV] plus a uint8 valid mask, averaged over
`factor` x `factor` blocks. Multilooking (averaging intensity over neighbouring pixels)
reduces speckle, so the ENL goes up and has to be re-estimated on the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

BANDS = ("HH", "HV")


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
