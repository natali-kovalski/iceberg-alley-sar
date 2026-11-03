"""Revision C11a: the THOR pipeline on the same 40 m IW input the CFARs see (2b).

The IW comparison put the THOR pipeline at 10 m against reference CFARs at 40 m, so resolution
was a variable. Here the frozen encoder + IW-trained linear head runs on 2b's 4x4-multilooked
frames (`iw_sweep.out_dir/<scene>/<scene>_40m_HHHV.tif`, same valid mask), as on the Færch EW
scene: 5.12 km crops, `upsample` (bands declared 10 m, THOR resizes to 512 px) unless config
says otherwise. HyP3 gamma0 goes to sigma0 with the incidence map averaged onto the 40 m grid.
Peaks get `distance_to_ice_km` from the frame's pack-ice mask; scoring is 2b's (shared domain,
one-to-one at `validation.max_dist_m`, recall at the 2-per-label budget from real runs).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

from iceberg_sar.config import Config
from iceberg_sar.thor import PIPELINE_SHORT

TAG = "thor-pipeline-40m"


@dataclass
class _Stack:
    stack: Path
    valid: Path


def inc_40m(cfg: Config, scene: str, stack: Path) -> Path:
    """The 10 m incidence map (radians, 0 = nodata) averaged onto the 40 m grid, in degrees."""
    from iceberg_sar.preprocess import find_rtc_bands

    out = stack.with_name(f"{scene}_40m_inc_deg.tif")
    if out.is_file():
        return out
    src_path = find_rtc_bands(cfg.path("raw") / scene)["inc"]
    with rasterio.open(stack) as ref, rasterio.open(src_path) as src:
        dst = np.full((ref.height, ref.width), np.nan, np.float32)
        a = src.read(1).astype(np.float32)
        a[a == 0] = np.nan
        reproject(a, dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref.transform, dst_crs=ref.crs, resampling=Resampling.average,
                  src_nodata=np.nan, dst_nodata=np.nan)
        profile = ref.profile | {"count": 1, "dtype": "float32", "nodata": np.nan}
    with rasterio.open(out, "w", **profile) as f:
        f.write(np.degrees(dst).astype(np.float32), 1)
    return out


def run_detect(cfg: Config, device: str | None, force: bool, log: Callable[[str], None]
               ) -> dict[str, Path]:
    import torch

    from iceberg_sar.benchmark.sweep import SweepParams
    from iceberg_sar.benchmark.thor_faerch import (
        ThorFaerchParams,
        crop_geometry,
        scene_peaks,
    )
    from iceberg_sar.seaice import distance_to_ice_km
    from iceberg_sar.thor.encoder import ThorParams, load_encoder
    from iceberg_sar.thor.train import load_head

    p, sp = ThorFaerchParams.from_config(cfg), SweepParams.from_config(cfg)
    variant = cfg.section("benchmark")["thor_iw40"]["variant"]
    v = next(x for x in p.variants if x.name == variant)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head, encoder, out = None, None, {}
    for scene in sp.scenes:
        work = sp.out_dir / scene
        gj = work / TAG / f"{TAG}.geojson"
        out[scene] = gj
        if gj.is_file() and not force:
            log(f"{scene}: done")
            continue
        st = _Stack(work / f"{scene}_40m_HHHV.tif", work / f"{scene}_40m_valid.tif")
        with rasterio.open(st.stack) as src:
            pixel_m, crs, transform, bounds = abs(src.transform.a), src.crs, src.transform, \
                src.bounds
        if head is None:
            head = load_head(p.head, device)
            crop_px, _ = crop_geometry(p.crop_m, pixel_m, v)
            encoder = load_encoder(replace(ThorParams.from_config(cfg), crop_px=crop_px,
                                           patch_px=v.patch_px), pixel_m, device, v.band_gsd_m)
        inc = inc_40m(cfg, scene, st.stack)
        peaks, _, _ = scene_peaks(st, v, encoder, head, cfg, box(*bounds), device, log,
                                  radiometry="gamma0", inc=inc)
        xs, ys = transform * (peaks.col.to_numpy() + 0.5, peaks.row.to_numpy() + 0.5)
        ice = cfg.path("interim") / scene / f"{scene}_seaice_mask.tif"
        dist = distance_to_ice_km(ice, np.asarray(xs), np.asarray(ys)) if ice.is_file() \
            else np.full(len(peaks), np.nan)
        gdf = gpd.GeoDataFrame(peaks.assign(detector=TAG, distance_to_ice_km=dist), crs=crs,
                               geometry=gpd.points_from_xy(xs, ys))
        gj.parent.mkdir(parents=True, exist_ok=True)
        gj.write_text(gdf.to_json(drop_id=True), encoding="utf-8")
        log(f"{scene}: {len(gdf)} peaks")
    return out


def run_thor_iw40(cfg: Config, device: str | None = None, force: bool = False,
                  log: Callable[[str], None] = print) -> dict[str, Path]:
    from iceberg_sar.benchmark.iwcompare import (
        IwCompareParams,
        at_budget,
        pool_sweep,
        scene_truth,
    )
    from iceberg_sar.thor.compare import SCORE_GRID, in_domain, score_points, xy

    files = run_detect(cfg, device, force, log)
    p = IwCompareParams.from_config(cfg)
    rows = []
    for scene, gj in files.items():
        _, t_xy, crs = scene_truth(cfg, scene, p.domain)
        det = in_domain(gpd.read_file(gj), p.domain)
        d_xy, sc = xy(det, crs), det["score"].to_numpy()
        for t in SCORE_GRID:
            keep = sc >= t
            tp, _ = score_points(d_xy[keep], t_xy, p.radius_m)
            rows.append({"scene": scene, "detector": TAG, "setting": float(t),
                         "detections": int(keep.sum()), "matched": tp, "truth": len(t_xy)})
    per = pd.DataFrame(rows)
    per["recall"] = per.matched / per.truth
    pooled = pool_sweep(per, per.scene.nunique())
    ref = pd.read_csv(p.sweep.out_dir / "pooled_sweep.csv")
    allc = pd.concat([pooled, ref[ref.detector != TAG]], ignore_index=True)
    n_truth = int(per.drop_duplicates("scene").truth.sum())
    budget = int(round(p.per_label * n_truth))
    tab = []
    for det, g in allc.groupby("detector", sort=False):
        r, n, flag = at_budget(g, budget, p.budget_min_fraction)
        tab.append({"detector": det, "recall": r, "detections": n, "flag": flag})
    tab = pd.DataFrame(tab).sort_values("recall", ascending=False)
    out_dir = p.sweep.out_dir
    per.to_csv(out_dir / f"{TAG}_sweep.csv", index=False)
    tab.to_csv(out_dir / f"{TAG}_at_budget.csv", index=False)
    rep = cfg.root / cfg.section("benchmark")["thor_iw40"]["report"]
    rep.write_text(report(tab, pooled, allc, budget, n_truth), encoding="utf-8")
    return {"sweep": out_dir / f"{TAG}_sweep.csv", "report": rep}


def report(tab: pd.DataFrame, pooled: pd.DataFrame, allc: pd.DataFrame, budget: int,
           n_truth: int) -> str:
    from iceberg_sar.benchmark.iwcompare import LABELS

    names = LABELS | {TAG: f"{PIPELINE_SHORT}, linear head, 40 m (this run)"}
    t10 = allc[allc.detector == "thor-linear-10m"].set_index("detections")
    L = ["# THOR pipeline on the 40 m IW frames (revision C11a)", "",
         "Generated by `python -m iceberg_sar.cli bench-thor-iw40`. Same input as the 2b CFARs "
         f"(4x4-multilooked HyP3 frames, same masks), {n_truth} same-pass labels in the shared "
         f"domain, recall at the {budget}-detection budget from real runs.", "",
         "| Detector | Recall at budget | Detections used | Flag | THOR pipeline 40 m at that "
         "count |", "|---|---|---|---|---|"]
    g40 = pooled[pooled.detections > 0].set_index("detections").sort_index()
    for r in tab.itertuples():
        same = ""
        if pd.notna(r.detections) and r.detector != TAG:
            a = g40[g40.index <= r.detections]
            if len(a):
                same = f"{a.recall.iloc[-1]:.3f} ({int(a.index[-1])})"
        L.append(f"| {names.get(r.detector, r.detector)} | "
                 + ("" if pd.isna(r.recall) else f"{r.recall:.3f}") + " | "
                 + ("" if pd.isna(r.detections) else f"{int(r.detections)}")
                 + f" | {r.flag} | {same} |")
    L += ["", "The THOR pipeline sweeps its score threshold in 0.01 steps, so it lands close "
          "to any budget; the CFARs move in PFA decades (see the full curves in "
          "`benchmarks/m2b_iw40.md`).", ""]
    if len(t10):
        L += ["10 m vs 40 m, THOR pipeline, pooled at the same detection counts:", "",
              "| Detections | Recall 40 m | Recall 10 m (5d) |", "|---|---|---|"]
        for n in (250, 500, 750, budget):
            a = g40[g40.index <= n]
            b = t10[t10.index <= n]
            if len(a) and len(b):
                L.append(f"| ≤ {n} | {a.recall.iloc[a.index.argmax()]:.3f} "
                         f"({int(a.index.max())}) | {b.recall.iloc[b.index.argmax()]:.3f} "
                         f"({int(b.index.max())}) |")
        L.append("")
    return "\n".join(L)
