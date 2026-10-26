"""Milestone 0c: every reference detector on one IW frame, multilooked to 40 m.

Steps: multilook the masked HH/HV rasters (land, pack ice and nodata already NaN) ->
estimate the ENL on the result -> run each detector at one PFA through the adapter ->
quicklook per detector, a side-by-side zoom, and a counts table in `benchmarks/m0_smoke.md`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio

from iceberg_sar.benchmark import adapter, plots
from iceberg_sar.benchmark.enl import estimate_enl
from iceberg_sar.benchmark.pfa import detector_channel_pfa
from iceberg_sar.benchmark.preprocess import multilook_stack
from iceberg_sar.config import Config


def prepare(cfg: Config, force: bool = False, sc: dict[str, Any] | None = None,
            scene: str | None = None, work_root: Path | None = None) -> dict[str, Any]:
    """Multilooked stack, mask and ENL sidecar for one IW frame (cached on disk).

    Defaults to the smoke scene. `sc` is a config block with the multilook and ENL settings
    (`benchmark.smoke` by default); the stack goes to `work_root / scene`.
    """
    bench = cfg.section("benchmark")
    sc = sc if sc is not None else bench["smoke"]
    scene = scene or sc["scene"]
    interim = cfg.path("interim") / scene
    paths = {pol: interim / f"{scene}_{pol}_masked.tif" for pol in ("HH", "HV")}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing; run `preprocess` for {scene} first")
    with rasterio.open(paths["HH"]) as src:
        pixel_m = abs(src.transform.a)
    res_m = round(pixel_m * int(sc["multilook"]))
    work = (work_root or cfg.root / bench["out_dir"] / "m0") / scene
    raster = work / f"{scene}_{res_m}m_HHHV.tif"
    mask = work / f"{scene}_{res_m}m_valid.tif"
    if force or not (raster.is_file() and mask.is_file()):
        t0 = time.perf_counter()
        multilook_stack(paths, int(sc["multilook"]), raster, mask,
                        min_valid=float(sc["multilook_min_valid"]))
        print(f"multilook {pixel_m:g} -> {res_m} m: {time.perf_counter() - t0:.0f} s")
    sidecar = work / f"{scene}_{res_m}m_enl.json"
    if force or not sidecar.is_file():
        estimate_enl(raster, mask, float(sc["enl_box_m"]), sc["enl_box"],
                     sc["enl_combine"], sidecar=sidecar)
    enl = json.loads(sidecar.read_text(encoding="utf-8"))
    inc = cfg.path("raw") / scene / f"{scene}_inc_map.tif"
    return {"scene": scene, "work": work, "raster": raster, "mask": mask, "enl": enl,
            "res_m": res_m, "src_m": pixel_m, "inc": inc if inc.is_file() else None}


def run_smoke(cfg: Config, report: Path, force: bool = False) -> Path:
    bench = cfg.section("benchmark")
    sc = bench["smoke"]
    prep = prepare(cfg, force)
    pfa = float(sc["pfa"])
    per_band = bench.get("enl_mode", "single") == "per_band"
    enl_bands = prep["enl"]["enl"] if per_band else None
    results: dict[str, gpd.GeoDataFrame] = {}
    rows = []
    for name in adapter.DETECTORS:
        out_dir = prep["work"] / name
        t0 = time.perf_counter()
        res = adapter.run(name, prep["raster"], prep["mask"], [pfa], enl=prep["enl"]["enl_used"],
                          enl_bands=enl_bands, cfg=cfg, out_dir=out_dir,
                          scene_id=prep["scene"], inc_path=prep["inc"])
        meta = json.loads((out_dir / f"{name}_run.json").read_text(encoding="utf-8"))
        run = meta["runs"][0]
        g = res[pfa]
        results[name] = g
        n_raw = _count_objects(out_dir / run["raster"], int(bench["object_connectivity"]))
        rows.append({
            "detector": name, "wi_px": meta["wi_px"], "wo_px": meta["wo_px"],
            "cropped": meta["window_cropped"], "enl": _enl_label(name, meta),
            "pfa_per_channel": detector_channel_pfa(name, pfa, bench["merge"]),
            "outlier_px": run["outlier_px"], "objects": len(g),
            "dropped_large": n_raw - len(g),
            "median_area_px": float(np.median(g["area_px"])) if len(g) else float("nan"),
            "runtime_s": run["runtime_s"], "warnings": run["warnings"],
        })
        print(f"{name}: {len(g)} objects, detector {run['runtime_s']:.0f} s, "
              f"total {time.perf_counter() - t0:.0f} s")
    pngs = _figures(prep, results, sc, pfa)
    notes = ""
    if report.is_file():  # keep hand-written notes below the marker across re-runs
        old = report.read_text(encoding="utf-8")
        if NOTES_MARKER in old:
            notes = old[old.index(NOTES_MARKER):]
    text = _report(prep, rows, pngs, bench, pfa, cfg.root)
    report.write_text(text + ("\n" + notes if notes else ""), encoding="utf-8")
    return report


NOTES_MARKER = "<!-- hand-written notes below; kept when bench-smoke re-runs -->"


def _enl_label(name: str, meta: dict[str, Any]) -> str:
    """ENL a detector actually used, for the counts table."""
    if name == "nis":
        return "own (NIS image)"
    if name in ("lognorm", "idpolrad"):
        return "not used"
    e = meta["enl"]
    return f"HH {e['HH']:g} / HV {e['HV']:g}" if isinstance(e, dict) else f"{e:g}"


def _gamma_multiplier(pfa: float, enl: float) -> float:
    """Gamma CA-CFAR threshold over the clutter mean (the library's `_gamma_pfa` relation)."""
    from scipy.optimize import brentq
    from scipy.special import gammaincc

    return float(brentq(lambda t: gammaincc(enl, t * enl) - pfa, 1.0, 1e4))


def _count_objects(det_path: Path, connectivity: int) -> int:
    from scipy import ndimage

    with rasterio.open(det_path) as src:
        det = src.read(1) > 0
    return int(ndimage.label(det, ndimage.generate_binary_structure(2, 1 if connectivity == 4
                                                                    else 2))[1])


def _figures(prep: dict[str, Any], results: dict[str, gpd.GeoDataFrame], sc: dict[str, Any],
             pfa: float) -> dict[str, Path]:
    with rasterio.open(prep["raster"]) as src:
        hv_db = plots.to_db(src.read(2))
    with rasterio.open(prep["mask"]) as src:
        valid = src.read(1) > 0
    stretch = plots.db_stretch(hv_db)
    qdir = prep["work"] / "quicklooks"
    pngs = {name: plots.detection_quicklook(
                hv_db, g, qdir / f"{name}_pfa{adapter.pfa_tag(pfa)}.png",
                f"{name}  PFA {pfa:.0e}  {len(g)} objects", int(sc["quicklook_max_px"]), stretch)
            for name, g in results.items()}
    size = 250  # 10 km at 40 m
    r0, c0 = plots.densest_window(list(results.values()), hv_db.shape, size, valid)
    pngs["side_by_side"] = plots.side_by_side(
        hv_db, results, (r0, c0, size), qdir / f"side_by_side_pfa{adapter.pfa_tag(pfa)}.png",
        stretch, f"{prep['scene']}  HV dB, {prep['res_m']} m, PFA {pfa:.0e} total; "
                 f"zoom rows {r0}-{r0 + size}, cols {c0}-{c0 + size}")
    return pngs


def _report(prep: dict[str, Any], rows: list[dict[str, Any]], pngs: dict[str, Path],
            bench: dict[str, Any], pfa: float, root: Path) -> str:
    enl = prep["enl"]
    rel = {k: p.relative_to(root).as_posix() for k, p in pngs.items()}
    lines = [
        "# Milestone 0c: reference CFAR detectors on one IW frame",
        "",
        "Generated by `python -m iceberg_sar.cli bench-smoke`. Smoke test only: no labels, no",
        "scoring. The counts show that every detector runs on our data and how far apart they are.",
        "",
        "## Setup",
        "",
        f"- Frame: `{prep['scene']}` (HyP3 RTC, **gamma0**, HH/HV). The paper used sigma0;",
        "  a local CFAR barely notices, since cos(incidence) is almost constant across a "
        "520 m window.",
        f"- Multilooked {int(bench['smoke']['multilook'])}x{int(bench['smoke']['multilook'])} "
        f"from {prep['src_m']:g} m to **{prep['res_m']} m** (mean of linear intensity, "
        "block kept only if fully valid).",
        "- Mask: the existing `_masked` rasters: OSM land + 500 m buffer, pack ice, nodata. "
        "Masked pixels are NaN, and pixels whose clutter ring is less than "
        f"{bench['min_ring_fraction']:.0%} valid are skipped (see `cfar_api_notes.md`).",
        f"- ENL (mean²/var, linear) in a {enl['box_px'][0]}x{enl['box_px'][1]} px open-water box "
        f"at {enl['box_wgs84']} (W, S, E, N), the median of {enl['boxes_considered']} fully valid "
        f"boxes: **HH {enl['enl']['HH']}, HV {enl['enl']['HV']}**. "
        + ("Gamma and K use each band's own ENL (`enl_mode: per_band`, a runner option; the "
           "library itself takes one value for both). Wishart's statistic takes one ENL: "
           if bench.get("enl_mode", "single") == "per_band" else
           "The library takes one value for both bands: ")
        + f"**{enl['enl_used']}** ({enl['enl_combine']}).",
    ]
    if enl.get("enl_percentiles_over_boxes"):
        sp = enl["enl_percentiles_over_boxes"]
        lines.append("  Spread over boxes (p10 / p50 / p90): " + ", ".join(
            f"{p} {sp[p]['10']} / {sp[p]['50']} / {sp[p]['90']}" for p in sp) + ".")
    hh, hv = enl["enl"]["HH"], enl["enl"]["HV"]
    if max(hh, hv) / min(hh, hv) > 1.5 and bench.get("enl_mode", "single") != "per_band":
        p_ch = detector_channel_pfa("gamma", pfa, bench["merge"])
        m = {k: 10 * np.log10(_gamma_multiplier(p_ch, v)) for k, v in
             (("HH", hh), ("HV", hv), ("used", enl["enl_used"]))}
        lines.append(
            f"- **ENL mismatch.** At the per-channel PFA, the gamma threshold is "
            f"{m['HH']:.1f} dB over the clutter mean for HH's own ENL and {m['HV']:.1f} dB for "
            f"HV's. With one ENL for both, it is {m['used']:.1f} dB on both, so one channel "
            "misses its nominal PFA. HyP3's HV noise-floor removal leaves water HV with a low "
            "ENL. The paper's EW scene has one ENL (10.7) for both bands. To decide in 2b.")
    lines += [
        f"- Total PFA **{pfa:.0e}**; gamma, log-normal and K merge HH AND HV "
        f"(per channel {detector_channel_pfa('gamma', pfa, 'AND'):.3g}). NIS, Wishart and "
        "iDPolRAD test one statistic at the total PFA.",
        f"- Objects: {bench['object_connectivity']}-connected components; library minsize "
        f"{bench['minsize_px']} px, objects over {bench['max_object_px']} px dropped "
        "(paper sec. 3.3).",
        "",
        "## Counts",
        "",
        "| Detector | wi / wo px | ENL | PFA per test | Outlier px | Objects | Dropped >"
        f"{bench['max_object_px']} px | Median area px | Detector runtime s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        crop = " (cropped to 41)" if r["cropped"] else ""
        lines.append(f"| {r['detector']} | {r['wi_px']} / {r['wo_px']}{crop} | {r['enl']} | "
                     f"{r['pfa_per_channel']:.3g} | {r['outlier_px']} | {r['objects']} | "
                     f"{r['dropped_large']} | {r['median_area_px']:g} | {r['runtime_s']:.1f} |")
    warn = [(r["detector"], w) for r in rows for w in r["warnings"]]
    lines += ["", "Runtime excludes Numba compilation (it happens at import in the runner).", ""]
    if warn:
        lines += ["Library warnings:", ""] + [f"- {d}: `{w}`" for d, w in warn] + [""]
    lines += ["## Figures (local, gitignored)", "",
              f"- Side by side, same zoom: `{rel['side_by_side']}`"]
    lines += [f"- {name}: `{rel[name]}`" for name in adapter.DETECTORS]
    lines.append("")
    return "\n".join(lines)
