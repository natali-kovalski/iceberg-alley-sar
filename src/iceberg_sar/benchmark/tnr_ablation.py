"""Revision C12: thermal-noise ablation on the Færch EW scene.

SNAP 14's ThermalNoiseRemoval subtracts the noise-equivalent sigma0 (N) and clips negatives to
~3.7e-12. Whether the detectors degrade because of the subtraction or because of the clipping
was never separated. The two SNAP stacks share one 40 m grid, so N is read off them directly:
N = noTNR - TNR wherever TNR did not clip (exact: SNAP's own noise LUT, applied by SNAP), and a
Gaussian-weighted average of the exact N around it where it did (N varies over km, not pixels).
Four preprocessing variants then differ only in what happens below the noise floor:
- `none`: no subtraction (the noTNR stack, the reproduction run);
- `clip`: subtraction, negatives clipped (SNAP's TNR stack);
- `floor`: subtraction, floored at N (values below the noise level set to the noise level);
- `noclip`: subtraction, negatives kept. The library takes dB input and can't run on negative
  intensities, so only my linear CA-CFAR runs here.
ENL is measured in the same open-water box on every stack (the noTNR run's box).
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from iceberg_sar.benchmark import score as sc
from iceberg_sar.config import Config

VARIANTS = ("floor", "noclip")


def variant_config(cfg: Config, variant: str | None, tnr: bool = False) -> Config:
    """`cfg` for one preprocessing variant (`snap.tnr_variant`; None = the SNAP stacks)."""
    raw = copy.deepcopy(cfg.raw)
    snap = raw["benchmark"]["faerch2023"]["snap"]
    snap["remove_thermal_noise"] = tnr
    snap["tnr_variant"] = variant
    return Config(root=cfg.root, raw=raw)


def noise_field(no_tnr: np.ndarray, tnr: np.ndarray, valid: np.ndarray, clip_level: float,
                sigma_px: float) -> tuple[np.ndarray, float]:
    """N per pixel and the share of valid pixels where it had to be filled (TNR clipped)."""
    n = no_tnr - tnr
    exact = valid & np.isfinite(n) & (tnr > clip_level)
    w = ndimage.gaussian_filter(exact.astype(np.float32), sigma_px)
    s = ndimage.gaussian_filter(np.where(exact, n, 0).astype(np.float32), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = np.where(w > 1e-3, s / w, np.nan)
    out = np.where(exact, n, fill).astype(np.float32)
    filled = float((valid & ~exact).sum() / max(valid.sum(), 1))
    return out, filled


def build_stacks(cfg: Config, a: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    """Write the `floor` and `noclip` stacks (+ copies of the noTNR masks); returns stats."""
    from iceberg_sar.benchmark.enl import estimate_enl
    from iceberg_sar.benchmark.reproduce import scene_paths

    src_n, src_t = scene_paths(variant_config(cfg, None)), scene_paths(variant_config(cfg, None,
                                                                                      tnr=True))
    outs = {v: scene_paths(variant_config(cfg, v)) for v in VARIANTS}
    stats: dict[str, Any] = {}
    if all(o.stack.is_file() for o in outs.values()):
        log("stacks: done")
    else:
        with rasterio.open(src_n.stack) as fn, rasterio.open(src_t.stack) as ft, \
                rasterio.open(src_n.valid) as fv:
            profile = fn.profile
            valid = fv.read(1) > 0
            data = {v: [] for v in VARIANTS}
            for i, pol in enumerate(fn.descriptions):
                x, t = fn.read(i + 1), ft.read(i + 1)
                n, filled = noise_field(x, t, valid, float(a["clip_level"]),
                                        float(a["fill_sigma_px"]))
                d = x - n
                data["noclip"].append(np.where(valid, d, np.nan).astype(np.float32))
                data["floor"].append(np.where(valid, np.maximum(d, n), np.nan).astype(np.float32))
                stats[pol] = {"noise_median_db": round(float(10 * np.log10(np.nanmedian(
                                  n[valid]))), 2),
                              "filled_share": round(filled, 4),
                              "below_noise_share": round(float(np.mean(d[valid] < 0)), 4),
                              "snap_clipped_share": round(float(np.mean(
                                  t[valid] <= float(a["clip_level"]))), 4)}
                log(f"{pol}: N median {stats[pol]['noise_median_db']} dB, filled "
                    f"{100 * filled:.1f} %, below noise "
                    f"{100 * stats[pol]['below_noise_share']:.1f} %")
            descriptions = fn.descriptions
        for v, o in outs.items():
            with rasterio.open(o.stack, "w", **(profile | {"dtype": "float32"})) as dst:
                for i, arr in enumerate(data[v]):
                    dst.write(arr, i + 1)
                    dst.set_band_description(i + 1, descriptions[i])
            for f in ("inc", "swath", "land", "valid"):
                shutil.copyfile(getattr(src_n, f), getattr(o, f))
        (outs["floor"].stack.parent / "tnr_ablation_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
    box = json.loads(src_n.enl.read_text(encoding="utf-8"))["box_wgs84"]
    enl = {}
    for name, sp in [("none", src_n), ("clip", src_t), *outs.items()]:
        r = estimate_enl(sp.stack, sp.valid, 2000.0, box, "min",
                         sidecar=sp.enl if name in VARIANTS else None)
        enl[name] = r["enl"]
    return {"noise": stats, "enl": enl}


def run_tnr_ablation(cfg: Config, sweep: bool = True, log: Callable[[str], None] = print
                     ) -> dict[str, Path]:
    from iceberg_sar.benchmark.sweep import run_faerch_sweep, sweep_dir

    fc = cfg.section("benchmark")["faerch2023"]
    a = cfg.section("benchmark")["tnr_ablation"]
    info = build_stacks(cfg, a, log)
    if sweep:
        for v in VARIANTS:
            only = list(a["detectors"][v])
            log(f"{v}: sweep {only}")
            run_faerch_sweep(variant_config(cfg, v), only=only, log=log)
    labels, unknown = sc.load_labels(cfg, fc["labels_source"])
    domain = sc.eval_domain(cfg, fc["score"]["aoi"])
    rows = []
    for name, vc in [("none", variant_config(cfg, None)),
                     ("clip", variant_config(cfg, None, tnr=True)),
                     *[(v, variant_config(cfg, v)) for v in VARIANTS]]:
        if not (sweep_dir(vc) / "sweep_runtime.csv").is_file():
            continue
        s = sc.score_sweep(vc, labels, unknown, domain, lambda _m: None)
        for det, b in sc.optimum(s).iterrows():
            if det not in a["report_detectors"]:
                continue
            rows.append({"variant": name, "detector": det, "pfa": b.pfa, "TP": int(b.TP),
                         "FP": int(b.FP), "precision": b.precision, "recall": b.recall,
                         "f": b.f, "enl_hh": info["enl"][name]["HH"],
                         "enl_hv": info["enl"][name]["HV"]})
    t = pd.DataFrame(rows)
    out_dir = cfg.root / a["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "tnr_ablation.csv", index=False)
    rep = cfg.root / a["report"]
    rep.write_text(report(t, info), encoding="utf-8")
    return {"table": out_dir / "tnr_ablation.csv", "report": rep}


NAMES = {"none": "No subtraction (noTNR, reproduction run)",
         "clip": "Subtraction, negatives clipped (SNAP 14 TNR)",
         "floor": "Subtraction, floored at the noise level",
         "noclip": "Subtraction, negatives kept"}


def report(t: pd.DataFrame, info: dict[str, Any]) -> str:
    L = ["# Thermal-noise ablation (revision C12)", "",
         "Generated by `python -m iceberg_sar.cli bench-tnr-ablation`. Færch EW scene, 40 m. "
         "The noise field N is SNAP 14's own (noTNR − TNR where TNR did not clip, smoothed in "
         "where it did), so the four variants differ only below the noise floor. ENL: mean² / "
         "variance in one open-water box, the same box on every stack. Gamma runs with Table 1's "
         "ENL 10.7, as in the paper; my CA-CFAR measures its own ENL per band.", "",
         "| Preprocessing | ENL HH / HV | Detector | Best PFA | TP / FP | P | R | F |",
         "|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {NAMES[r.variant]} | {r.enl_hh:.2f} / {r.enl_hv:.2f} | "
                 f"{sc.DISPLAY.get(r.detector, r.detector)} | {r.pfa:.0e} | {r.TP} / {r.FP} | "
                 f"{r.precision:.3f} | {r.recall:.3f} | {r.f:.3f} |")
    L += ["", "Noise field per band:", ""]
    for pol, s in info["noise"].items():
        L.append(f"- {pol}: N median {s['noise_median_db']} dB; SNAP clipped "
                 f"{100 * s['snap_clipped_share']:.1f} % of valid pixels, filled "
                 f"{100 * s['filled_share']:.1f} %; below the noise level after subtraction "
                 f"{100 * s['below_noise_share']:.1f} %.")
    L += ["", "The library's detectors take dB input, so they can't run on `negatives kept`.", ""]
    return "\n".join(L)
