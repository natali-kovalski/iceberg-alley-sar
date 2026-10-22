"""Run one cfar-object-detection detector on a dual-pol GeoTIFF at one or more PFAs.

Runs in the `cfar-bench` env (see benchmarks/ENVIRONMENT.md) and is the only code that
imports the GPL-3.0 `cfar_filters` library. It talks to the main package through files:

  in : --image  2-band GeoTIFF [HH, HV], linear intensity, float32 (NaN = nodata)
       --mask   1-band GeoTIFF, nonzero = valid
  out: <out-dir>/<detector>_pfa<pfa>.tif  uint8 detections (1 = outlier), same grid as --image
       <out-dir>/<detector>_run.json       parameters, per-PFA runtime and counts, warnings

Library behaviour this runner handles (benchmarks/cfar_api_notes.md):
- `detector.run` expects dB; the input here is linear, so it is converted at the boundary.
- The mask is only checked at the centre pixel. With --masked-as-nan (default), invalid
  pixels are NaN so they stay out of the clutter ring.
- An all-NaN ring gives a clutter mean of 0 (every pixel detected), so pixels whose ring is
  less than --min-ring-fraction valid are removed from the mask. 0 = library as-is.
- Windows wider than the 41 px stencil are silently cropped by the library; here that is an
  error unless --allow-window-crop is given.

Copyright (C) 2026 Natali Kovalski. GPL-3.0-or-later, see LICENSE in this folder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio

t_import = time.perf_counter()
import cfar_filters  # noqa: E402  (Numba compiles the window kernels at import)
from cfar_filters import detector as cfar_detector  # noqa: E402
from cfar_filters import gamma, kdistribution, utils  # noqa: E402
from cfar_filters.fast_functions import fast_edge_mean  # noqa: E402

IMPORT_S = time.perf_counter() - t_import

DETECTORS = ("gamma", "lognorm", "k", "wishart", "nis", "idpolrad")
STENCIL_LIMIT_PX = 41  # fast_functions: neighbourhood (-20, 20)
# Per-channel detectors that take an ENL: with --enl-hh/--enl-hv each band gets its own
PER_BAND_ENL = {"gamma": gamma.detector, "k": kdistribution.detector}


def channel_pfa(pfa: float, method: str) -> float:
    """Per-channel PFA (Færch 2023 sec. 2.1.1); OR in a form that doesn't cancel to 0."""
    return float(np.sqrt(pfa)) if method == "AND" else float(-np.expm1(0.5 * np.log1p(-pfa)))


def run_per_band(image_db: np.ndarray, mask: np.ndarray, detector: str, method: str,
                 pfa: float, enl_hh: float, enl_hv: float, minsize: int, wi: int,
                 wo: int) -> np.ndarray:
    """`detector.run` for gamma / K, except each band uses its own ENL.

    Same steps as the library: dB -> linear, per-channel PFA, HH and HV detectors, Boolean
    merge, small-object removal. The library passes one ENL to both bands, which doesn't
    fit HyP3 data, where water HV has a much lower ENL than HH (benchmarks/m0_smoke.md).
    """
    lin = utils.db2in(image_db)
    p = channel_pfa(pfa, method)
    f = PER_BAND_ENL[detector]
    hh = f(lin[0, ...], mask=mask, pfa=p, enl=enl_hh, wi=wi, wo=wo)
    hv = f(lin[1, ...], mask=mask, pfa=p, enl=enl_hv, wi=wi, wo=wo)
    return utils.remove_small_objects(hh & hv if method == "AND" else hh | hv, minsize)


def pfa_tag(pfa: float) -> str:
    return f"{pfa:.0e}".replace("+", "")


def library_commit() -> str | None:
    repo = Path(cfar_filters.__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ring_valid_fraction(valid: np.ndarray, wi: int, wo: int) -> np.ndarray:
    """Share of the library's clutter ring that is valid, per pixel (0 at the array border)."""
    everywhere = np.ones(valid.shape, dtype=bool)
    return fast_edge_mean(valid.astype(np.float32), everywhere, wi, wo)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--detector", choices=DETECTORS, required=True)
    p.add_argument("--pfa", type=float, action="append", required=True,
                   help="Total PFA; repeat for several levels.")
    p.add_argument("--wi", type=int, required=True, help="Inner window diameter (px).")
    p.add_argument("--wo", type=int, required=True, help="Outer window diameter (px).")
    p.add_argument("--enl", type=float, required=True,
                   help="Single ENL (library behaviour; Wishart always uses this one).")
    p.add_argument("--enl-hh", type=float, help="With --enl-hv: per-band ENL for gamma / k.")
    p.add_argument("--enl-hv", type=float)
    p.add_argument("--method", choices=("AND", "OR"), default="AND")
    p.add_argument("--minsize", type=int, default=2)
    p.add_argument("--min-ring-fraction", type=float, default=0.5)
    p.add_argument("--masked-as-nan", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-window-crop", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    for name, w in (("wi", a.wi), ("wo", a.wo)):
        if w > STENCIL_LIMIT_PX and not a.allow_window_crop:
            raise SystemExit(f"{name}={w} px exceeds the library's {STENCIL_LIMIT_PX} px window "
                             "(it would be cropped); pass --allow-window-crop to accept that")
    if not all(0 < p < 1 for p in a.pfa):
        raise SystemExit(f"PFA must be in (0, 1): {a.pfa}")
    if (a.enl_hh is None) != (a.enl_hv is None):
        raise SystemExit("--enl-hh and --enl-hv go together")
    per_band = a.enl_hh is not None and a.detector in PER_BAND_ENL
    a.out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(a.image) as src:
        if src.count != 2:
            raise SystemExit(f"--image must have 2 bands [HH, HV], has {src.count}")
        image = src.read().astype(np.float32)
        profile = src.profile
    with rasterio.open(a.mask) as src:
        if (src.height, src.width) != image.shape[1:]:
            raise SystemExit("--mask is not on the --image grid")
        valid = src.read(1) > 0

    valid &= np.all(np.isfinite(image) & (image > 0), axis=0)  # dB needs positive intensity
    if a.masked_as_nan:
        image[:, ~valid] = np.nan
    # iDPolRAD's windows are a test and a training disk (inner radius 0)
    ring_wi = 0 if a.detector == "idpolrad" else a.wi
    if a.min_ring_fraction > 0:
        run_mask = valid & (ring_valid_fraction(valid, ring_wi, a.wo) >= a.min_ring_fraction)
    else:
        run_mask = valid.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        image_db = (10.0 * np.log10(image)).astype(np.float32)

    out_profile = dict(profile) | {"count": 1, "dtype": "uint8", "nodata": None, "compress": "lzw",
                             "predictor": 1}
    runs = []
    for pfa in a.pfa:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            if per_band:
                outliers = run_per_band(image_db, run_mask, a.detector, a.method, pfa,
                                        a.enl_hh, a.enl_hv, a.minsize, a.wi, a.wo)
            else:
                outliers = cfar_detector.run(image_db, run_mask, detector=a.detector,
                                             method=a.method, pfa=pfa, enl=a.enl,
                                             minsize=a.minsize, wi=a.wi, wo=a.wo)
            runtime = time.perf_counter() - t0
        if not isinstance(outliers, np.ndarray):
            raise SystemExit(f"Library returned no result for detector '{a.detector}'")
        outliers &= run_mask
        path = a.out_dir / f"{a.detector}_pfa{pfa_tag(pfa)}.tif"
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(outliers.astype(np.uint8), 1)
        msgs = sorted({f"{w.category.__name__}: {' '.join(str(w.message).split())}"
                       for w in caught})
        runs.append({"pfa": pfa, "raster": path.name, "runtime_s": round(runtime, 2),
                     "outlier_px": int(outliers.sum()), "warnings": msgs})
        print(f"{a.detector} pfa={pfa:.0e}: {int(outliers.sum())} px in {runtime:.1f} s",
              flush=True)

    meta = {
        "detector": a.detector, "method": a.method, "wi_px": a.wi, "wo_px": a.wo,
        "enl_mode": "per_band" if per_band else "single",
        "enl": {"HH": a.enl_hh, "HV": a.enl_hv} if per_band else a.enl,
        "minsize": a.minsize, "min_ring_fraction": a.min_ring_fraction,
        "masked_as_nan": a.masked_as_nan, "window_cropped": max(a.wi, a.wo) > STENCIL_LIMIT_PX,
        "image": str(a.image), "mask": str(a.mask),
        "valid_px": int(valid.sum()), "run_mask_px": int(run_mask.sum()),
        "import_s": round(IMPORT_S, 2), "library_commit": library_commit(),
        "python": sys.version.split()[0],
        "versions": {m.__name__: m.__version__ for m in (np, rasterio)},
        "runs": runs,
    }
    (a.out_dir / f"{a.detector}_run.json").write_text(json.dumps(meta, indent=2),
                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
