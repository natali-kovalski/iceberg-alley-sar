"""Milestone 2b (one frame first): PFA sweep of the reference detectors and my CA-CFAR on an IW
frame multilooked to 40 m (`benchmark.iw_sweep`).

Per scene, under `<out_dir>/<scene>/`:
  <scene>_40m_HHHV.tif / _valid.tif / _enl.json   multilooked stack, mask, ENL (smoke.prepare)
  <scene>_40m_HH_masked.tif / _HV_masked.tif      the stack as single bands, NaN outside the mask
                                                  (the CA-CFAR's input format)
  <detector>/<detector>_pfa<pfa>.geojson          objects per PFA, CA-CFAR schema
  <detector>/<detector>_run.json                  per-PFA runtime, counts and settings
  cacfar*/                                        my CA-CFAR variants (`cacfar_variants`): length
                                                  rule on / off x paper or own windows
  cacfar-nolength-10m/                            my CA-CFAR on the 10 m masked rasters, no
                                                  length rule (`cacfar_10m_pfas`, one pass)
  sweep_runtime.csv                               runtime per detector and PFA

PFAs run in ascending order, and a detector stops after the first PFA with more objects than
`max_objects`. Finished detectors are skipped on a restart (`force` redoes them); a CA-CFAR
variant whose run.json has other settings (windows, ENL, PFAs, cap) is redone.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from iceberg_sar.benchmark import adapter
from iceberg_sar.benchmark.pfa import detector_channel_pfa
from iceberg_sar.benchmark.smoke import prepare
from iceberg_sar.cfar import CfarParams
from iceberg_sar.config import Config
from iceberg_sar.detections import (
    NO_STRUCTURE_LIMIT,
    DetectionParams,
    detect_raster,
    detect_raster_multi,
    params_for_raster,
    to_geodataframe,
)

BANDS = ("HH", "HV")
CACFAR_10M = "cacfar-nolength-10m"


WINDOW_SETS = ("paper", "own")


@dataclass(frozen=True)
class CacfarVariant:
    """One run of my CA-CFAR: length rule on / off, and the window set. `paper` = the reference
    detectors' windows (`iw_sweep.cacfar_windows`), `own` = `cfar.guard_m` / `background_m`."""
    length_rule: bool
    windows: str = "paper"

    @property
    def tag(self) -> str:
        return cacfar_tag(self.length_rule) + ("-ownwin" if self.windows == "own" else "")


@dataclass(frozen=True)
class SweepParams:
    scenes: list[str]
    out_dir: Path
    pfas: list[float]
    max_objects: int
    detectors: list[str]
    cacfar_variants: list[CacfarVariant]
    cacfar_windows: dict[str, float]
    cacfar_10m_pfas: list[float]
    cacfar_10m_block_rows: int | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> SweepParams:
        s = cfg.section("benchmark")["iw_sweep"]
        pfas = sorted(float(p) for p in s["pfas"])
        if not all(0 < p < 1 for p in pfas):
            raise ValueError(f"iw_sweep.pfas must be in (0, 1): {pfas}")
        unknown = set(s["detectors"]) - set(adapter.DETECTORS)
        if unknown:
            raise ValueError(f"iw_sweep.detectors: unknown {sorted(unknown)}")
        variants = [CacfarVariant(bool(v["length_rule"]), str(v.get("windows", "paper")))
                    for v in s["cacfar_variants"]]
        bad = sorted({v.windows for v in variants} - set(WINDOW_SETS))
        if bad:
            raise ValueError(f"iw_sweep.cacfar_variants: windows must be {WINDOW_SETS}: {bad}")
        return cls(scenes=list(s["scenes"]), out_dir=cfg.root / s["out_dir"], pfas=pfas,
                   max_objects=int(s["max_objects"]), detectors=list(s["detectors"]),
                   cacfar_variants=variants,
                   cacfar_windows={k: float(v) for k, v in s["cacfar_windows"].items()},
                   cacfar_10m_pfas=sorted(float(x) for x in s.get("cacfar_10m_pfas") or []),
                   cacfar_10m_block_rows=s.get("cacfar_10m_block_rows"))

    @property
    def cacfar_tags(self) -> dict[str, CacfarVariant]:
        """CA-CFAR variant tag -> variant."""
        return {v.tag: v for v in self.cacfar_variants}


def cacfar_tag(length_rule: bool) -> str:
    return "cacfar" if length_rule else "cacfar-nolength"


def cacfar_settings(cfg: Config, hv: Path, length_rule: bool,
                    windows: dict[str, float] | None = None
                    ) -> tuple[float, CfarParams, DetectionParams]:
    """`params_for_raster`, with the length rule forced on or off. `windows` ({inner_m, outer_m},
    diameters as the reference detectors take them -> odd px -> half-widths) replaces the
    config's `cfar.guard_m` / `background_m`; None keeps them."""
    pixel_m, base, params = params_for_raster(cfg, hv)
    if windows is not None:
        wi = adapter.odd_px(windows["inner_m"], pixel_m)
        wo = adapter.odd_px(windows["outer_m"], pixel_m)
        base = replace(base, guard_px=(wi - 1) // 2, background_px=(wo - 1) // 2)
    if not length_rule:
        params = replace(params, max_structure_px=NO_STRUCTURE_LIMIT)
    return pixel_m, base, params


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    else:
        gdf.to_file(path, driver="GeoJSON")
    return path


def split_bands(stack: Path, mask: Path, out: dict[str, Path]) -> dict[str, Path]:
    """2-band [HH, HV] stack -> one masked GeoTIFF per band (NaN where `mask` is 0)."""
    with rasterio.open(stack) as src, rasterio.open(mask) as m:
        valid = m.read(1) > 0
        profile = src.profile | {"count": 1}
        for i, pol in enumerate(BANDS):
            band = src.read(i + 1).astype(np.float32)
            band[~valid] = np.nan
            with rasterio.open(out[pol], "w", **profile) as dst:
                dst.write(band, 1)
    return out


def band_paths(prep: dict[str, Any]) -> dict[str, Path]:
    stem = f"{prep['scene']}_{prep['res_m']}m"
    paths = {pol: prep["work"] / f"{stem}_{pol}_masked.tif" for pol in BANDS}
    if not all(p.is_file() for p in paths.values()):
        split_bands(prep["raster"], prep["mask"], paths)
    return paths


def _reference_done(out_dir: Path, name: str, p: SweepParams) -> dict[float, Path] | None:
    """GeoJSON per PFA of a finished reference run with these settings, else None."""
    f = out_dir / f"{name}_run.json"
    if not f.is_file():
        return None
    meta = json.loads(f.read_text(encoding="utf-8"))
    if (sorted(meta.get("pfas_requested") or []) != p.pfas
            or meta.get("max_objects") != p.max_objects):
        return None
    files = {float(r["pfa"]): out_dir / f"{name}_pfa{adapter.pfa_tag(float(r['pfa']))}.geojson"
             for r in meta["runs"]}
    return files if all(g.is_file() for g in files.values()) else None


def run_reference(cfg: Config, prep: dict[str, Any], name: str, p: SweepParams,
                  ice_mask: Path | None, force: bool, log: Callable[[str], None]
                  ) -> list[dict[str, Any]]:
    """One reference detector over the sweep; returns runtime rows (cached runs: none)."""
    out_dir = prep["work"] / name
    if not force and _reference_done(out_dir, name, p) is not None:
        log(f"{name}: done, skipped")
        return []
    bench = cfg.section("benchmark")
    per_band = bench.get("enl_mode", "single") == "per_band"
    t0 = time.perf_counter()
    res = adapter.run(name, prep["raster"], prep["mask"], p.pfas, enl=prep["enl"]["enl_used"],
                      enl_bands=prep["enl"]["enl"] if per_band else None, cfg=cfg,
                      out_dir=out_dir, scene_id=prep["scene"], inc_path=prep["inc"],
                      ice_mask=ice_mask, max_objects=p.max_objects)
    meta = json.loads((out_dir / f"{name}_run.json").read_text(encoding="utf-8"))
    rows = [{"detector": name, "pfa": r["pfa"], "objects": len(res[float(r["pfa"])]),
             "runtime_s": r["runtime_s"]} for r in meta["runs"]]
    stop = " (stopped: object cap)" if meta.get("stopped_early") else ""
    log(f"{name}: {len(rows)} PFAs{stop}, detector {sum(r['runtime_s'] for r in rows):.0f} s, "
        f"total {time.perf_counter() - t0:.0f} s")
    return rows


def run_cacfar(cfg: Config, prep: dict[str, Any], p: SweepParams, v: CacfarVariant,
               ice_mask: Path | None, force: bool, log: Callable[[str], None]
               ) -> list[dict[str, Any]]:
    """My CA-CFAR (HV, co-pol check, `detections` rules) on the multilooked bands, per PFA."""
    tag, length_rule = v.tag, v.length_rule
    out_dir = prep["work"] / tag
    paths = band_paths(prep)
    pixel_m, base, params = cacfar_settings(
        cfg, paths["HV"], length_rule, p.cacfar_windows if v.windows == "paper" else None)
    settings = {"windows": v.windows, "guard_px": base.guard_px,
                "background_px": base.background_px, "enl": round(base.enl, 3),
                "length_rule": length_rule, "pfas_requested": p.pfas,
                "max_objects": p.max_objects}
    run_json = out_dir / f"{tag}_run.json"
    if run_json.is_file() and not force:
        old = json.loads(run_json.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in settings} != settings:
            log(f"{tag}: settings changed, redoing")
            force = True
    if force:   # PFAs past a new stopping point must not linger
        for f in out_dir.glob(f"{tag}_pfa*.geojson"):
            f.unlink()
    rows, runs, stopped = [], [], False
    for pfa in p.pfas:
        f = out_dir / f"{tag}_pfa{adapter.pfa_tag(pfa)}.geojson"
        if f.is_file() and not force:
            n = len(gpd.read_file(f))
            runtime = None
        else:
            t0 = time.perf_counter()
            table, transform, crs = detect_raster(paths, "HV", replace(base, pfa=pfa), params)
            gdf = to_geodataframe(table, transform, crs, prep["scene"], prep["inc"], ice_mask)
            gdf["detector"] = tag
            gdf["pfa"] = pfa
            gdf["pfa_per_channel"] = pfa   # one test, on HV
            write_geojson(gdf, f)
            n, runtime = len(gdf), round(time.perf_counter() - t0, 2)
            rows.append({"detector": tag, "pfa": pfa, "objects": n, "runtime_s": runtime})
        runs.append({"pfa": pfa, "objects": n, "runtime_s": runtime,
                     "alpha_db": round(10 * np.log10(replace(base, pfa=pfa).alpha), 2)})
        if n > p.max_objects:
            stopped = True
            break
    meta = {"detector": tag, "band": "HV", "pixel_m": pixel_m, **settings,
            "max_structure_px": params.max_structure_px if length_rule else None,
            "copol_min_contrast_db": params.copol_min_contrast_db, "stopped_early": stopped,
            "runs": runs}
    out_dir.mkdir(parents=True, exist_ok=True)
    run_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stop = " (stopped: object cap)" if stopped else ""
    log(f"{tag}: {len(runs)} PFAs{stop}, ENL {base.enl:.2f}, guard/background "
        f"{base.guard_px}/{base.background_px} px, {sum(r['runtime_s'] for r in rows):.0f} s new")
    return rows


def run_cacfar_10m(cfg: Config, scene: str, p: SweepParams, work: Path, inc: Path | None,
                   ice_mask: Path | None, force: bool, log: Callable[[str], None]
                   ) -> list[dict[str, Any]]:
    """My CA-CFAR on the native 10 m masked rasters without the length rule, all of
    `cacfar_10m_pfas` in one pass (as 5d's sweep). Skipped if every PFA is on disk."""
    out_dir = work / CACFAR_10M
    files = {pfa: out_dir / f"{CACFAR_10M}_pfa{adapter.pfa_tag(pfa)}.geojson"
             for pfa in p.cacfar_10m_pfas}
    done = (all(f.is_file() for f in files.values())
            and (out_dir / f"{CACFAR_10M}_run.json").is_file())
    if done and not force:
        log(f"{CACFAR_10M}: done, skipped")
        return []
    interim = cfg.path("interim") / scene
    paths = {pol: interim / f"{scene}_{pol}_masked.tif" for pol in BANDS}
    pixel_m, base, params = params_for_raster(cfg, paths["HV"])
    params = replace(params, max_structure_px=NO_STRUCTURE_LIMIT)
    if p.cacfar_10m_block_rows:
        params = replace(params, block_rows=int(p.cacfar_10m_block_rows))
    t0 = time.perf_counter()
    cfars = [replace(base, pfa=pfa) for pfa in p.cacfar_10m_pfas]
    tables, transform, crs = detect_raster_multi(paths, "HV", cfars, params)
    # One GeoDataFrame for the whole sweep: incidence and distance to ice are sampled once
    tagged = [t.assign(sweep_k=k) for k, t in enumerate(tables) if not t.empty]
    allg = to_geodataframe(pd.concat(tagged, ignore_index=True) if tagged else pd.DataFrame(),
                           transform, crs, scene, inc, ice_mask)
    runtime = round(time.perf_counter() - t0, 2)
    runs = []
    for k, c in enumerate(cfars):
        gdf = allg[allg["sweep_k"] == k].drop(columns="sweep_k") if len(allg) else allg
        gdf = gdf.assign(detector=CACFAR_10M, pfa=c.pfa, pfa_per_channel=c.pfa)
        write_geojson(gdf, files[c.pfa])
        runs.append({"pfa": c.pfa, "objects": len(gdf),
                     "alpha_db": round(10 * np.log10(c.alpha), 2)})
    meta = {"detector": CACFAR_10M, "band": "HV", "pixel_m": pixel_m, "guard_px": base.guard_px,
            "background_px": base.background_px, "enl": round(base.enl, 3), "length_rule": False,
            "copol_min_contrast_db": params.copol_min_contrast_db,
            "pfas_requested": p.cacfar_10m_pfas, "runtime_s_all_pfas": runtime, "runs": runs}
    (out_dir / f"{CACFAR_10M}_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"{CACFAR_10M}: {len(runs)} PFAs in one pass, ENL {base.enl:.2f}, guard/background "
        f"{base.guard_px}/{base.background_px} px, {runtime:.0f} s")
    return [{"detector": CACFAR_10M, "pfa": r["pfa"], "objects": r["objects"],
             "runtime_s": runtime / len(runs)} for r in runs]


def run_sweep(cfg: Config, force: bool = False, only: list[str] | None = None,
              log: Callable[[str], None] = print) -> list[Path]:
    """Every scene of `iw_sweep`: reference detectors, then my CA-CFAR. Returns the work dirs."""
    p = SweepParams.from_config(cfg)
    sc = cfg.section("benchmark")["iw_sweep"]
    names = [*p.detectors, *p.cacfar_tags] + ([CACFAR_10M] if p.cacfar_10m_pfas else [])
    if only:
        names = [n for n in names if n in only]
    out = []
    for scene in p.scenes:
        prep = prepare(cfg, sc=sc, scene=scene, work_root=p.out_dir)
        ice = cfg.path("interim") / scene / f"{scene}_seaice_mask.tif"
        ice_mask = ice if ice.is_file() else None
        log(f"{scene}: {prep['res_m']} m, ENL HH {prep['enl']['enl']['HH']} / "
            f"HV {prep['enl']['enl']['HV']}, {len(p.pfas)} PFAs, cap {p.max_objects} objects")
        rows: list[dict[str, Any]] = []
        for name in names:
            if name in p.cacfar_tags:
                rows += run_cacfar(cfg, prep, p, p.cacfar_tags[name], ice_mask, force, log)
            elif name == CACFAR_10M:
                rows += run_cacfar_10m(cfg, scene, p, prep["work"], prep["inc"], ice_mask,
                                       force, log)
            else:
                rows += run_reference(cfg, prep, name, p, ice_mask, force, log)
        if rows:
            f = prep["work"] / "sweep_runtime.csv"
            new = pd.DataFrame(rows)
            if f.is_file():   # keep the rows of detectors that were skipped this time
                old = pd.read_csv(f)
                new = pd.concat([old[~old["detector"].isin(new["detector"])], new],
                                ignore_index=True)
            new.to_csv(f, index=False)
        out.append(prep["work"])
    return out


# --- Milestone 1e: the Færch 2023 EW scene, every detector x 21 PFAs (`benchmark.faerch2023.sweep`)
#
# Under `sweep.out_dir/<detector>/`:
#   <detector>_pfa<pfa>.tif / .json   runner output: detections, and that PFA's runtime and counts
#   <detector>_pfa<pfa>.geojson       objects, CA-CFAR schema (not written above `max_objects`)
# and `sweep_runtime.csv` (one row per detector and PFA) plus `sweep_log.txt` in `out_dir`.
# Resumable: a PFA counts as done once its .json exists, so a restart (or a crash halfway through
# a detector) only runs what is missing. `force` reruns everything.


@dataclass(frozen=True)
class FaerchSweepParams:
    out_dir: Path
    pfas: list[float]
    detectors: list[str]
    enl: float
    max_objects: int

    @classmethod
    def from_config(cls, cfg: Config) -> FaerchSweepParams:
        s = cfg.section("benchmark")["faerch2023"]["sweep"]
        lo, hi = int(s["pfa_min_exp"]), int(s["pfa_max_exp"])
        if not lo <= hi < 0:
            raise ValueError(f"faerch2023.sweep: PFA exponents {lo}..{hi} must be < 0")
        unknown = set(s["detectors"]) - set(adapter.DETECTORS)
        if unknown:
            raise ValueError(f"faerch2023.sweep.detectors: unknown {sorted(unknown)}")
        enl = table1_enl(cfg) if s["enl"] == "table1" else float(s["enl"])
        return cls(out_dir=sweep_dir(cfg),
                   pfas=[float(f"1e{e}") for e in range(lo, hi + 1)],
                   detectors=list(s["detectors"]), enl=enl, max_objects=int(s["max_objects"]))


def sweep_dir(cfg: Config) -> Path:
    """`sweep.out_dir`, plus `_noTNR` for the stack without thermal noise removal."""
    fc = cfg.section("benchmark")["faerch2023"]
    tnr = fc["snap"].get("remove_thermal_noise", True)
    return cfg.root / (fc["sweep"]["out_dir"] + ("" if tnr else "_noTNR"))


def table1_enl(cfg: Config) -> float:
    """The paper's ENL for the Sentinel-1 product in use (scenes.csv `paper_enl`, from Table 1)."""
    from iceberg_sar.benchmark.reproduce import s1_product
    from iceberg_sar.benchmark.scenes import load_scenes

    fc = cfg.section("benchmark")["faerch2023"]
    product = s1_product(cfg)
    rows = [r for r in load_scenes(cfg.root / fc["scenes_csv"]) if r["product_id"] == product]
    if len(rows) != 1 or not rows[0].get("paper_enl"):
        raise ValueError(f"No paper_enl for {product} in {fc['scenes_csv']}")
    return float(rows[0]["paper_enl"])


def pfa_status(out_dir: Path, name: str, pfas: list[float], max_objects: int
               ) -> tuple[dict[float, dict[str, Any]], list[float]]:
    """Finished PFAs (runner sidecars) and the PFAs still to run. Ascending: nothing above the
    first PFA over `max_objects` runs."""
    done: dict[float, dict[str, Any]] = {}
    todo: list[float] = []
    for pfa in sorted(pfas):
        f = out_dir / f"{name}_pfa{adapter.pfa_tag(pfa)}.json"
        if f.is_file():
            done[pfa] = json.loads(f.read_text(encoding="utf-8"))
            if done[pfa]["objects"] > max_objects:
                break
        else:
            todo.append(pfa)
    return done, todo


def _run_runner(cmd: list[str], log: Callable[[str], None]) -> int:
    """Run the external runner, passing its per-PFA progress lines to `log`."""
    import subprocess

    tail: list[str] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          bufsize=1) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            if " pfa=" in line:
                log(f"  {line}")
    if proc.returncode != 0:
        log("  runner failed:\n    " + "\n    ".join(tail))
    return proc.returncode


def sweep_detector(cfg: Config, name: str, p: FaerchSweepParams, force: bool,
                   log: Callable[[str], None]) -> list[dict[str, Any]]:
    """One detector over the sweep on the EW stack; returns one row per finished PFA."""
    from iceberg_sar.benchmark.reproduce import scene_paths

    bench = cfg.section("benchmark")
    sp = scene_paths(cfg)
    out_dir = p.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for f in out_dir.glob(f"{name}_pfa*.json"):
            f.unlink()
    done, todo = pfa_status(out_dir, name, p.pfas, p.max_objects)
    if todo:
        s = adapter.detector_settings(name, adapter.pixel_size_m(sp.stack), bench)
        cmd = adapter.runner_command(bench, cfg.root, s, sp.stack, sp.valid, todo, p.enl, out_dir,
                                     max_objects=p.max_objects)
        log(f"{name}: running {len(todo)} PFAs ({todo[0]:.0e} .. {todo[-1]:.0e}), "
            f"wi/wo {s.wi_px}/{s.wo_px} px, ENL {p.enl:g}")
        if _run_runner(cmd, log) != 0:
            log(f"{name}: FAILED; finished PFAs are kept, rerun to continue")
        done, todo = pfa_status(out_dir, name, p.pfas, p.max_objects)
    rows = []
    for pfa, st in sorted(done.items()):
        geo = out_dir / f"{name}_pfa{adapter.pfa_tag(pfa)}.geojson"
        over = st["objects"] > p.max_objects
        if not over and (force or not geo.is_file()):
            gdf = adapter.objects_from_raster(
                out_dir / st["raster"], sp.stack, detector=name, pfa=pfa, merge=bench["merge"],
                connectivity=int(bench["object_connectivity"]),
                max_object_px=int(bench["max_object_px"]), scene_id=sp.product, inc_path=sp.inc)
            write_geojson(gdf, geo)
        rows.append({"detector": name, "pfa": pfa,
                     "pfa_per_channel": detector_channel_pfa(name, pfa, bench["merge"]),
                     "outlier_px": st["outlier_px"], "objects_raw": st["objects"],
                     "runtime_s": st["runtime_s"], "over_cap": over,
                     "geojson": geo.name if not over else ""})
    left = "" if not todo else f", {len(todo)} PFAs still missing"
    log(f"{name}: {len(rows)} PFAs on disk{left}, detector runtime "
        f"{sum(r['runtime_s'] for r in rows) / 60:.1f} min")
    return rows


def run_faerch_sweep(cfg: Config, force: bool = False, only: list[str] | None = None,
                     log: Callable[[str], None] = print) -> Path:
    """1e: every detector of `faerch2023.sweep`, then (2a) my CA-CFAR variants; returns
    `sweep_runtime.csv`. `only` takes detector names and CA-CFAR tags."""
    import datetime as dt

    p = FaerchSweepParams.from_config(cfg)
    p.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = p.out_dir / "sweep_log.txt"

    def tlog(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        log(line)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    names = [n for n in p.detectors if not only or n in only]
    cacfar = {t: r for t, r in faerch_cacfar_tags(cfg).items() if not only or t in only}
    tlog(f"sweep: {names + list(cacfar)}, {len(p.pfas)} PFAs {p.pfas[0]:.0e}..{p.pfas[-1]:.0e}, "
         f"ENL {p.enl:g} (reference detectors), cap {p.max_objects} objects")
    rows: list[dict[str, Any]] = []
    for name in names:
        t0 = time.perf_counter()
        rows += sweep_detector(cfg, name, p, force, tlog)
        tlog(f"{name}: wall time {(time.perf_counter() - t0) / 60:.1f} min")
    for rule in cacfar.values():
        rows += sweep_cacfar_faerch(cfg, p, rule, force, tlog)
    return merge_runtime_rows(p.out_dir / "sweep_runtime.csv", rows)


def merge_runtime_rows(f: Path, rows: list[dict[str, Any]]) -> Path:
    """Write `rows` to `sweep_runtime.csv`, keeping the rows of detectors not in them."""
    new = pd.DataFrame(rows)
    if f.is_file() and not new.empty:
        old = pd.read_csv(f)
        new = pd.concat([old[~old["detector"].isin(new["detector"])], new], ignore_index=True)
    new.to_csv(f, index=False)
    return f


# --- Milestone 2a: my CA-CFAR on the Færch 2023 EW scene, same protocol as 1e
#
# Under `sweep_dir/<tag>/` (tag cacfar / cacfar-nolength): `<tag>_pfa<pfa>.geojson` per PFA up to
# the first one over `max_objects` (that one has no GeoJSON, as in 1e) and `<tag>_run.json`.
# Rows go into the same `sweep_runtime.csv`, so `bench-score` scores them with the rest.
# Input: the stack as masked single bands next to it (`<stem>_40m_HH_masked.tif`, NaN outside
# the valid mask), the format `detections.detect_raster` reads.


def faerch_cacfar_tags(cfg: Config) -> dict[str, bool]:
    """CA-CFAR variant tag -> length rule on, from `faerch2023.sweep.cacfar_length_rules`."""
    rules = cfg.section("benchmark")["faerch2023"]["sweep"].get("cacfar_length_rules") or []
    return {cacfar_tag(bool(r)): bool(r) for r in rules}


def faerch_cacfar_settings(cfg: Config, hv: Path, length_rule: bool
                           ) -> tuple[float, CfarParams, DetectionParams]:
    """`params_for_raster` with the paper's windows (diameters -> odd px -> half-widths) and
    the length rule forced on or off."""
    s = cfg.section("benchmark")["faerch2023"]["sweep"]["cacfar_windows"]
    return cacfar_settings(cfg, hv, length_rule, s)


def cap_runs(counts: list[int], max_objects: int) -> int:
    """How many of the ascending PFAs to keep: up to and including the first over the cap."""
    for i, n in enumerate(counts):
        if n > max_objects:
            return i + 1
    return len(counts)


def sweep_cacfar_faerch(cfg: Config, p: FaerchSweepParams, length_rule: bool, force: bool,
                        log: Callable[[str], None]) -> list[dict[str, Any]]:
    """My CA-CFAR over every PFA of the sweep in one pass; returns `sweep_runtime.csv` rows."""
    from iceberg_sar.benchmark.reproduce import scene_paths

    tag = cacfar_tag(length_rule)
    sp = scene_paths(cfg)
    out_dir = p.out_dir / tag
    run_json = out_dir / f"{tag}_run.json"
    pixel_m = adapter.pixel_size_m(sp.stack)
    stem = f"{sp.stem}_{pixel_m:g}m"
    paths = {pol: sp.stack.parent / f"{stem}_{pol}_masked.tif" for pol in BANDS}
    if force or not all(f.is_file() for f in paths.values()):
        split_bands(sp.stack, sp.valid, paths)
    pixel_m, base, params = faerch_cacfar_settings(cfg, paths["HV"], length_rule)
    settings = {"guard_px": base.guard_px, "background_px": base.background_px,
                "enl": round(base.enl, 3), "length_rule": length_rule,
                "pfas_requested": p.pfas, "max_objects": p.max_objects}
    if run_json.is_file() and not force:
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        if {k: meta.get(k) for k in settings} == settings:
            log(f"{tag}: done, skipped")
            return meta["rows"]
    t0 = time.perf_counter()
    cfars = [replace(base, pfa=pfa) for pfa in p.pfas]
    tables, transform, crs = detect_raster_multi(paths, "HV", cfars, params)
    runtime = round(time.perf_counter() - t0, 2)
    keep = cap_runs([len(t) for t in tables], p.max_objects)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, t in zip(cfars[:keep], tables[:keep], strict=True):
        over = len(t) > p.max_objects
        geo = out_dir / f"{tag}_pfa{adapter.pfa_tag(c.pfa)}.geojson"
        if not over:
            gdf = to_geodataframe(t, transform, crs, sp.product, sp.inc)
            write_geojson(gdf.assign(detector=tag, pfa=c.pfa, pfa_per_channel=c.pfa), geo)
        rows.append({"detector": tag, "pfa": c.pfa, "pfa_per_channel": c.pfa,   # one test, HV
                     "outlier_px": np.nan, "objects_raw": len(t),
                     "runtime_s": round(runtime / len(cfars), 2), "over_cap": over,
                     "geojson": geo.name if not over else ""})
    meta = {"detector": tag, "band": "HV", "pixel_m": pixel_m, **settings,
            "alpha_db": {f"{c.pfa:.0e}": round(10 * np.log10(c.alpha), 2) for c in cfars},
            "max_structure_px": None if not length_rule else params.max_structure_px,
            "copol_min_contrast_db": params.copol_min_contrast_db,
            "runtime_s_all_pfas": runtime, "stopped_early": keep < len(cfars), "rows": rows}
    run_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stop = f", over the cap from {_e(cfars[keep - 1].pfa)}" if rows[-1]["over_cap"] else ""
    log(f"{tag}: {len(rows)} PFAs{stop}, ENL {base.enl:.2f}, guard/background "
        f"{base.guard_px}/{base.background_px} px, {runtime / 60:.1f} min")
    return rows


def _e(p: float) -> str:
    return f"{p:.0e}".replace("e-0", "e-")
