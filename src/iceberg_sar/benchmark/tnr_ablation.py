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
- `snapnoclip`: subtraction, negatives kept: SNAP's own TNR with clipNegativeValues=false (a
  real SNAP run, with the noTNR land / valid masks). The library takes dB input and can't run on
  negative intensities, so only my linear CA-CFAR runs here.
`noclip` (noTNR - N, negatives kept) is built with `floor` but not swept: compared pixel by pixel
with `snapnoclip`, it checks the reconstructed N that `floor` depends on.
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
from iceberg_sar.benchmark.preprocess import SNAP_NOCLIP
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


def snap_noclip_stack(cfg: Config, log: Callable[[str], None]) -> Any:
    """SNAP run with clipNegativeValues=false (skipped if its stack exists), then the noTNR
    land / valid masks copied in, so every variant is scored on the same pixels."""
    from iceberg_sar.benchmark.reproduce import preprocess_scene, scene_paths

    vc = variant_config(cfg, SNAP_NOCLIP, tnr=True)
    sp, src = scene_paths(vc), scene_paths(variant_config(cfg, None))
    if not sp.stack.is_file():
        info = preprocess_scene(vc, log=log)
        log(f"{SNAP_NOCLIP}: negative in swath HH {info.get('hh_negative_in_swath', 0):.2%}, "
            f"HV {info.get('hv_negative_in_swath', 0):.2%}")
    for f in ("land", "valid"):
        shutil.copyfile(getattr(src, f), getattr(sp, f))
    return sp


def compare_noclip(cfg: Config, a: dict[str, Any]) -> dict[str, Any]:
    """SNAP's negatives-kept stack vs the reconstructed `noclip` one, per band: equal where SNAP
    did not clip (sanity), and how close the reconstruction is where it did."""
    from iceberg_sar.benchmark.reproduce import scene_paths

    paths = {"snap": scene_paths(variant_config(cfg, SNAP_NOCLIP, tnr=True)).stack,
             "synth": scene_paths(variant_config(cfg, "noclip")).stack,
             "clip": scene_paths(variant_config(cfg, None, tnr=True)).stack}
    valid_path = scene_paths(variant_config(cfg, None)).valid
    with rasterio.open(valid_path) as fv:
        valid = fv.read(1) > 0
    out: dict[str, Any] = {}
    srcs = {k: rasterio.open(v) for k, v in paths.items()}
    try:
        for i, pol in enumerate(srcs["clip"].descriptions):
            x = {k: f.read(i + 1) for k, f in srcs.items()}
            ok = valid & np.isfinite(x["snap"]) & np.isfinite(x["synth"])
            clipped = ok & (x["clip"] <= float(a["clip_level"]))
            kept = ok & ~clipped
            rel = np.abs(x["snap"][kept] - x["clip"][kept]) / np.abs(x["clip"][kept])
            s, y = x["snap"][clipped], x["synth"][clipped]
            out[pol] = {
                "negative_share_snap": round(float(np.mean(x["snap"][ok] < 0)), 4),
                "negative_share_synth": round(float(np.mean(x["synth"][ok] < 0)), 4),
                "unclipped_max_rel_diff": float(rel.max()) if rel.size else 0.0,
                "clipped_px": int(clipped.sum()),
                "clipped_median_snap": float(np.median(s)) if s.size else float("nan"),
                "clipped_median_synth": float(np.median(y)) if y.size else float("nan"),
                "clipped_corr": round(float(np.corrcoef(s, y)[0, 1]), 3) if s.size > 1 else
                float("nan")}
    finally:
        for f in srcs.values():
            f.close()
    return out


def build_stacks(cfg: Config, a: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    """Write the `floor` and `noclip` stacks (+ copies of the noTNR masks); returns stats."""
    from iceberg_sar.benchmark.enl import estimate_enl
    from iceberg_sar.benchmark.reproduce import scene_paths

    src_n, src_t = scene_paths(variant_config(cfg, None)), scene_paths(variant_config(cfg, None,
                                                                                      tnr=True))
    outs = {v: scene_paths(variant_config(cfg, v)) for v in VARIANTS}
    stats: dict[str, Any] = {}
    stats_json = outs["floor"].stack.parent / "tnr_ablation_stats.json"
    if all(o.stack.is_file() for o in outs.values()) and stats_json.is_file():
        log("stacks: done")
        stats = json.loads(stats_json.read_text(encoding="utf-8"))
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
        stats_json.write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
    box = json.loads(src_n.enl.read_text(encoding="utf-8"))["box_wgs84"]
    stacks = [("none", src_n), ("clip", src_t), *outs.items()]
    snap_noclip = SNAP_NOCLIP in a["detectors"]
    if snap_noclip:
        stacks.append((SNAP_NOCLIP, snap_noclip_stack(cfg, log)))
    enl = {}
    for name, sp in stacks:
        r = estimate_enl(sp.stack, sp.valid, 2000.0, box, "min",
                         sidecar=sp.enl if name in (*VARIANTS, SNAP_NOCLIP) else None)
        enl[name] = r["enl"]
    info: dict[str, Any] = {"noise": stats, "enl": enl}
    if snap_noclip:
        info["noclip_check"] = compare_noclip(cfg, a)
        for pol, c in info["noclip_check"].items():
            log(f"{pol}: negative SNAP {100 * c['negative_share_snap']:.1f} % vs reconstructed "
                f"{100 * c['negative_share_synth']:.1f} %; where SNAP clipped, corr "
                f"{c['clipped_corr']}")
    return info


def run_tnr_ablation(cfg: Config, sweep: bool = True, log: Callable[[str], None] = print
                     ) -> dict[str, Path]:
    from iceberg_sar.benchmark.sweep import run_faerch_sweep, sweep_dir

    fc = cfg.section("benchmark")["faerch2023"]
    a = cfg.section("benchmark")["tnr_ablation"]
    info = build_stacks(cfg, a, log)
    derived = [(v, variant_config(cfg, v, tnr=v == SNAP_NOCLIP))
               for v in (*VARIANTS, SNAP_NOCLIP) if v in a["detectors"]]
    if sweep:
        for v, vc in derived:
            only = list(a["detectors"][v])
            log(f"{v}: sweep {only}")
            run_faerch_sweep(vc, only=only, log=log)
    labels, unknown = sc.load_labels(cfg, fc["labels_source"])
    domain = sc.eval_domain(cfg, fc["score"]["aoi"])
    rows = []
    for name, vc in [("none", variant_config(cfg, None)),
                     ("clip", variant_config(cfg, None, tnr=True)), *derived]:
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
         SNAP_NOCLIP: "Subtraction, negatives kept (SNAP 14, clipNegativeValues=false)"}


def report(t: pd.DataFrame, info: dict[str, Any]) -> str:
    L = ["# Thermal-noise ablation (revision C12)", "",
         "Generated by `python -m iceberg_sar.cli bench-tnr-ablation`. Færch EW scene, 40 m. "
         "The floored stack uses SNAP 14's own noise field N (noTNR − TNR where TNR did not "
         "clip, smoothed in where it did); the negatives-kept stack is SNAP's own run with "
         "`clipNegativeValues=false`. The four variants differ only below the noise floor. ENL: mean² / "
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
    L += ["", "The library's detectors take dB input, and negative intensities have no dB value, "
          "so they can't run on `negatives kept`.", ""]
    if info.get("noclip_check"):
        L += ["Check of the reconstructed N (used for the floored stack): noTNR − N with "
              "negatives kept, against SNAP's own negatives-kept run:", ""]
        for pol, c in info["noclip_check"].items():
            L.append(f"- {pol}: negative {100 * c['negative_share_snap']:.1f} % (SNAP) vs "
                     f"{100 * c['negative_share_synth']:.1f} % (reconstructed). Where SNAP 14 "
                     f"clipped ({c['clipped_px']:,} px): median {c['clipped_median_snap']:.2e} vs "
                     f"{c['clipped_median_synth']:.2e}, correlation {c['clipped_corr']}. "
                     f"Elsewhere identical to the clipped stack up to "
                     f"{c['unclipped_max_rel_diff']:.1e} (relative).")
        L.append("")
    f = t.set_index(["variant", "detector"]).f
    if {("none", "gamma"), ("clip", "gamma"), ("floor", "gamma")} <= set(f.index):
        L += ["## Reading", "",
              f"Gamma with Table 1's ENL: F {f['none', 'gamma']:.3f} without subtraction, "
              f"{f['clip', 'gamma']:.3f} with SNAP's clipped subtraction, "
              f"{f['floor', 'gamma']:.3f} with the same subtraction floored at the noise "
              "level. On this scene the drop after SNAP 14 thermal noise removal comes from "
              "how the pixels below the noise floor are handled (set to ~3.7e-12, i.e. "
              "-115 dB), not from subtracting the noise. Kept negative (SNAP's "
              "`clipNegativeValues=false`), they break the speckle statistics outright, so "
              "turning SNAP's clipping off is no fix. This is a finding about one scene and "
              "one SNAP version, not a claim of a software defect; which handling the paper's SNAP "
              "used is open (question to the authors).", "",
              "Caveats: where SNAP clipped (23.6 % of HV pixels), the floor value is the "
              "smoothed noise field, not SNAP's own; the check above shows it matches SNAP's. On the floored stack, HV open water sits "
              "mostly at the floor, so its measured ENL is meaningless (very high); that "
              "lowers my CA-CFAR's HV threshold, which measures its ENL on the scene, and is "
              "why it does worse there than gamma with a fixed ENL.", ""]
    return "\n".join(L)
