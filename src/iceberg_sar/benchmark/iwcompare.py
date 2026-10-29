"""Milestone 2b: score the `bench-iw-sweep` detections against IIP labels, per frame and
pooled over the frames.

Same protocol as 5d (`thor/compare.py`), so the 40 m curves sit next to the 10 m ones:
- truth: same-pass IIP labels (`pointval.load_truth`, iip-satellite), kept where the 10 m SAR
  is valid (`sar_valid_at`): the labels 5d compares (512 on its four frames);
- detections: open water only (nothing within `viewer.near_ice_km` of pack ice);
- matching: one-to-one at `validation.max_dist_m`.

Precision isn't measurable without AIS, so detectors are compared by recall at matched numbers
of detections, read off each PFA curve by interpolation in log(detections).
The 10 m CA-CFAR (with the length rule, from the old 5d run with the rule always on) and THOR
curves come from the 5d tables as reference.

Pooled: detections and matches summed per (detector, setting) over the frames, as 5d pools,
for the settings every frame ran. Frames differ a lot in labels (46-267), so the cross-frame
summary reads each curve at `summary_detections_per_label` x that frame's labels.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from iceberg_sar.benchmark import adapter
from iceberg_sar.benchmark.sweep import CACFAR_10M, SweepParams
from iceberg_sar.config import Config
from iceberg_sar.pointval import load_truth, point_val_params, sar_valid_at
from iceberg_sar.thor.compare import POOLED, SIZE_GROUPS, open_water, score_points, xy

# 10 m reference curves from 5d: 5d detector name -> label used here
REF_10M = {"cfar": "cacfar-10m", "thor-linear": "thor-linear-10m"}
REF_10M_OPS = {"cfar": "cacfar-10m", "thor-linear matched-count": "thor-linear-10m"}
# Current 5d (length rule only in icy frames): its CFAR curve, pooled section only. Its THOR
# operating point (matched to that CFAR's count) is the headline one.
CURRENT_5D = "cacfar-10m-5d"
CUR_10M = {"cfar": CURRENT_5D}
CUR_10M_OPS = {"cfar": CURRENT_5D, "thor-linear matched-count": "thor-linear-10m"}
LABELS = {
    "cacfar-nolength": "my CA-CFAR, HV, no length rule",
    "cacfar": "my CA-CFAR, HV, with length rule",
    "cacfar-nolength-ownwin": "my CA-CFAR, HV, no length rule, own windows",
    "cacfar-ownwin": "my CA-CFAR, HV, with length rule, own windows",
    "gamma": "gamma (HH AND HV)",
    "lognorm": "log-normal (HH AND HV)",
    "k": "K (HH AND HV)",
    "wishart": "Wishart",
    "nis": "NIS",
    "idpolrad": "iDPolRAD (window cropped to 41 px)",
    "cacfar-10m": "my CA-CFAR 10 m, with length rule (old 5d)",
    CACFAR_10M: "my CA-CFAR 10 m, no length rule",
    CURRENT_5D: "my CA-CFAR 10 m, current 5d (length rule in icy frames)",
    "thor-linear-10m": "THOR linear 10 m (5d)",
}
NOTES_MARKER = "<!-- hand-written notes below; kept when bench-iw-compare re-runs -->"


@dataclass(frozen=True)
class IwCompareParams:
    sweep: SweepParams
    matched_counts: list[int]
    breakdown_count: int
    near_ice_km: float
    radius_m: float
    report: Path
    ref_5d: Path
    current_5d: Path
    per_label: float
    pooled_counts: list[int]
    pooled_figure: Path | None

    @classmethod
    def from_config(cls, cfg: Config) -> IwCompareParams:
        s = cfg.section("benchmark")["iw_sweep"]
        return cls(sweep=SweepParams.from_config(cfg),
                   matched_counts=[int(n) for n in s["matched_counts"]],
                   breakdown_count=int(s["breakdown_count"]),
                   near_ice_km=float(cfg.raw.get("viewer", {}).get("near_ice_km", 5)),
                   radius_m=point_val_params(cfg).max_dist_m,
                   report=cfg.root / s["report"],
                   ref_5d=cfg.root / (s.get("ref_5d_dir")
                                       or cfg.section("thor")["compare"]["out_dir"]),
                   current_5d=cfg.root / cfg.section("thor")["compare"]["out_dir"],
                   per_label=float(s.get("summary_detections_per_label", 2)),
                   pooled_counts=[int(n) for n in s.get("pooled_matched_counts") or []],
                   pooled_figure=(cfg.root / s["pooled_figure"]) if s.get("pooled_figure")
                   else None)


def scene_truth(cfg: Config, scene: str) -> tuple[gpd.GeoDataFrame, np.ndarray, Any]:
    """5d's compared labels for a same-pass scene: truth, xy in the scene CRS, the CRS."""
    from iceberg_sar.detections import scene_timestamp

    masked = cfg.path("interim") / scene / f"{scene}_HV_masked.tif"
    with rasterio.open(masked) as src:
        crs = src.crs
    sar_time = pd.Timestamp(scene_timestamp(scene)).tz_localize(None)
    truth, _ = load_truth(cfg, scene, sar_time, "iip-satellite")
    t_xy = xy(truth, crs)
    ok = sar_valid_at(masked, t_xy, point_val_params(cfg).valid_block_px)
    return truth[ok].reset_index(drop=True), t_xy[ok], crs


def load_detections(det_dir: Path, tag: str, near_ice_km: float) -> dict[float, gpd.GeoDataFrame]:
    """Open-water detections per PFA of one detector's sweep (the PFAs it ran)."""
    meta_path = det_dir / f"{tag}_run.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path}; run `bench-iw-sweep` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out = {}
    for r in meta["runs"]:
        pfa = float(r["pfa"])
        out[pfa] = open_water(gpd.read_file(det_dir / f"{tag}_pfa{adapter.pfa_tag(pfa)}.geojson"),
                              near_ice_km)
    return out


def sweep_table(dets: dict[str, dict[float, gpd.GeoDataFrame]], t_xy: np.ndarray, crs: Any,
                radius_m: float) -> pd.DataFrame:
    rows = []
    for tag, by_pfa in dets.items():
        for pfa, d in by_pfa.items():
            tp, _ = score_points(xy(d, crs), t_xy, radius_m)
            rows.append({"detector": tag, "setting": pfa, "detections": len(d), "matched": tp})
    out = pd.DataFrame(rows)
    out["truth"] = len(t_xy)
    out["recall"] = (out["matched"] / len(t_xy)).round(3) if len(t_xy) else np.nan
    return out


def reference_10m(ref_5d: Path, scene: str, names: dict[str, str] = REF_10M) -> pd.DataFrame:
    """The 5d sweep rows of `names` (5d detector -> tag here) for this scene."""
    f = ref_5d / "sweep.csv"
    if not f.is_file():
        return pd.DataFrame()
    t = pd.read_csv(f)
    t = t[(t["scene"] == scene) & t["detector"].isin(list(names))].copy()
    t["detector"] = t["detector"].map(names)
    return t.drop(columns=["scene", "radius_m"], errors="ignore")


def recall_at(curve: pd.DataFrame, counts: list[int]) -> list[float]:
    """Recall at each detection count, interpolated in log(detections) along one detector's
    curve; NaN outside the range the curve covers (no extrapolation)."""
    c = curve[curve["detections"] > 0].groupby("detections", as_index=False)["recall"].max()
    if c.empty:
        return [np.nan] * len(counts)
    lx, y = np.log10(c["detections"].to_numpy(float)), c["recall"].to_numpy(float)
    return [float(np.interp(np.log10(n), lx, y)) if lx[0] <= np.log10(n) <= lx[-1] else np.nan
            for n in counts]


def matched_table(sweep: pd.DataFrame, counts: list[int]) -> pd.DataFrame:
    rows = []
    for det, g in sweep.groupby("detector", sort=False):
        r = recall_at(g, counts)
        rows.append({"detector": det, **{f"recall@{n}": v for n, v in zip(counts, r,
                                                                          strict=True)},
                     "detections_min": int(g["detections"].min()),
                     "detections_max": int(g["detections"].max()),
                     "pfas_run": len(g)})
    return pd.DataFrame(rows)


def size_breakdown(dets: dict[str, dict[float, gpd.GeoDataFrame]], truth: gpd.GeoDataFrame,
                   t_xy: np.ndarray, crs: Any, radius_m: float, count: int,
                   pfa_by_tag: dict[str, float] | None = None) -> pd.DataFrame:
    """Recall per IIP size group at each detector's PFA with the detection count nearest
    `count` (a real operating point, not an interpolation), or at `pfa_by_tag` where given
    (pooled: one PFA for every frame; detectors not in it are skipped)."""
    size = truth["size"].map(SIZE_GROUPS).fillna("other / unknown").to_numpy()
    rows = []
    for tag, by_pfa in dets.items():
        if not by_pfa or (pfa_by_tag is not None and pfa_by_tag.get(tag) not in by_pfa):
            continue
        pfa = pfa_by_tag[tag] if pfa_by_tag is not None else \
            min(by_pfa, key=lambda p: abs(np.log(max(len(by_pfa[p]), 1) / count)))
        d = by_pfa[pfa]
        _, hit = score_points(xy(d, crs), t_xy, radius_m)
        for group in sorted(set(size)):
            sel = size == group
            rows.append({"detector": tag, "pfa": pfa, "detections": len(d), "size": group,
                         "labels": int(sel.sum()), "matched": int(hit[sel].sum())})
    out = pd.DataFrame(rows)
    out["recall"] = (out["matched"] / out["labels"]).round(3)
    return out


def size_breakdown_10m(ref_5d: Path, scene: str,
                       ops: dict[str, str] = REF_10M_OPS) -> pd.DataFrame:
    """5d's size breakdown at its operating points (10 m CA-CFAR at its PFA, THOR at the same
    detection count) for this scene; `ops`: 5d operating point -> tag here."""
    f = ref_5d / "breakdown.csv"
    if not f.is_file():
        return pd.DataFrame()
    b = pd.read_csv(f)
    b = b[(b["scene"] == scene) & (b["by"] == "size") & b["op"].isin(list(ops))]
    b = b.drop_duplicates(["op", "bin"])
    return pd.DataFrame({"detector": b["op"].map(ops), "pfa": np.nan,
                         "detections": np.nan, "size": b["bin"], "labels": b["labels"],
                         "matched": b["matched"], "recall": b["recall"]})


def run_iw_compare(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Path]:
    p = IwCompareParams.from_config(cfg)
    sp = p.sweep
    out: dict[str, Path] = {}
    sections = []
    scenes: dict[str, dict[str, Any]] = {}   # per-scene data for the pooled section
    for scene in sp.scenes:
        work = sp.out_dir / scene
        truth, t_xy, crs = scene_truth(cfg, scene)
        tags = [*sp.detectors, *sp.cacfar_tags] + ([CACFAR_10M] if sp.cacfar_10m_pfas else [])
        dets = {t: load_detections(work / t, t, p.near_ice_km) for t in tags}
        sweep = sweep_table(dets, t_xy, crs, p.radius_m)
        ref = reference_10m(p.ref_5d, scene)
        full = pd.concat([sweep, ref], ignore_index=True)
        full.insert(0, "scene", scene)
        full.to_csv(work / "sweep_scored.csv", index=False)
        matched = matched_table(full, p.matched_counts)
        matched.to_csv(work / "recall_at_matched_counts.csv", index=False)
        sizes = pd.concat([size_breakdown(dets, truth, t_xy, crs, p.radius_m,
                                          p.breakdown_count),
                           size_breakdown_10m(p.ref_5d, scene)], ignore_index=True)
        sizes.to_csv(work / "size_breakdown.csv", index=False)
        enl = json.loads(next(work.glob(f"{scene}_*m_enl.json")).read_text(encoding="utf-8"))
        cac = {}   # window set -> run.json of its first variant (ENL and windows per set)
        for tag, v in sp.cacfar_tags.items():
            if v.windows not in cac:
                cac[v.windows] = json.loads((work / tag / f"{tag}_run.json")
                                            .read_text(encoding="utf-8"))
        from iceberg_sar.thor.compare_figure import scene_date

        png = recall_figure(full, sp, len(t_xy), p, f"{scene_date(scene)} {scene[:3]}",
                            work / "recall_vs_detections.png")
        log(f"{scene}: {len(t_xy)} labels; figure {png}")
        out[scene] = png
        sections.append(_scene_section(scene, full, matched, sizes, enl, cac, p, png, cfg.root))
        scenes[scene] = {"sweep": pd.concat([full, reference_10m(p.current_5d, scene, CUR_10M)
                                             .assign(scene=scene)], ignore_index=True),
                         "dets": dets, "truth": truth, "t_xy": t_xy, "crs": crs}
    if len(scenes) > 1:
        pooled_out = run_pooled(scenes, p, cfg.root, log)
        out.update(pooled_out["paths"])
        sections.append(pooled_out["section"])
    notes = ""
    if p.report.is_file():
        old = p.report.read_text(encoding="utf-8")
        if NOTES_MARKER in old:
            notes = old[old.index(NOTES_MARKER):]
    p.report.write_text(_header(p) + "\n".join(sections) + ("\n" + notes if notes else ""),
                        encoding="utf-8")
    out["report"] = p.report
    return out


# --- pooled over the frames ------------------------------------------------------------------

def pool_sweep(per_scene: pd.DataFrame, n_scenes: int) -> pd.DataFrame:
    """Detections, matches and labels summed per (detector, setting) over the frames, for the
    settings every frame has (a detector stopped by the object cap in one frame has no pooled
    point past it)."""
    out = per_scene.groupby(["detector", "setting"], as_index=False, sort=False).agg(
        detections=("detections", "sum"), matched=("matched", "sum"),
        truth=("truth", "sum"), frames=("scene", "nunique"))
    out = out[out["frames"] == n_scenes].drop(columns="frames").reset_index(drop=True)
    out["recall"] = (out["matched"] / out["truth"]).round(3)
    return out


def summary_table(per_scene: pd.DataFrame, pooled: pd.DataFrame, scenes: list[str],
                  per_label: float) -> pd.DataFrame:
    """Recall of each detector at `per_label` x the labels, in each frame and pooled
    (interpolated in log(detections); NaN where the curve doesn't reach that count)."""
    rows = []
    for det in pooled["detector"].unique().tolist() + [
            d for d in per_scene["detector"].unique() if d not in set(pooled["detector"])]:
        row: dict[str, Any] = {"detector": det}
        for scene in scenes:
            c = per_scene[(per_scene["scene"] == scene) & (per_scene["detector"] == det)]
            n = int(round(per_label * c["truth"].iloc[0])) if len(c) else 0
            row[scene] = recall_at(c, [n])[0] if len(c) else np.nan
        c = pooled[pooled["detector"] == det]
        row[POOLED] = (recall_at(c, [int(round(per_label * c["truth"].iloc[0]))])[0]
                       if len(c) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_sizes(scenes: dict[str, dict[str, Any]], pooled: pd.DataFrame, p: IwCompareParams,
                 count: int) -> pd.DataFrame:
    """Recall by size, pooled: each detector of this sweep at the one setting whose pooled
    detection count is nearest `count`, summed over the frames; 10 m CA-CFARs and THOR at
    5d's pooled operating points."""
    pfa_by_tag = {}
    for det, g in pooled.groupby("detector", sort=False):
        g = g[g["detections"] > 0]
        if len(g):
            pfa_by_tag[det] = float(g.loc[(np.log(g["detections"] / count)).abs().idxmin(),
                                          "setting"])
    parts = [size_breakdown(d["dets"], d["truth"], d["t_xy"], d["crs"], p.radius_m, count,
                            pfa_by_tag) for d in scenes.values()]
    t = pd.concat(parts, ignore_index=True)
    out = t.groupby(["detector", "size"], as_index=False, sort=False).agg(
        pfa=("pfa", "first"), labels=("labels", "sum"), matched=("matched", "sum"))
    # Detections from the pooled sweep: a size group missing in one frame (04-17 has no large
    # bergs) would otherwise leave that frame's detections out of its row.
    n_det = pooled.set_index(["detector", "setting"])["detections"]
    out["detections"] = [n_det.get((d, s), np.nan) for d, s in zip(out["detector"], out["pfa"],
                                                                     strict=True)]
    out["recall"] = (out["matched"] / out["labels"]).round(3)
    ref = pd.concat([size_breakdown_10m(p.ref_5d, POOLED, {"cfar": "cacfar-10m"}),
                     size_breakdown_10m(p.current_5d, POOLED, CUR_10M_OPS)], ignore_index=True)
    return pd.concat([out, ref], ignore_index=True)


def run_pooled(scenes: dict[str, dict[str, Any]], p: IwCompareParams, root: Path,
               log: Callable[[str], None]) -> dict[str, Any]:
    """Pooled tables, figure and report section; tables in `iw_sweep.out_dir`."""
    from iceberg_sar.thor.compare_figure import scene_date

    sp = p.sweep
    per_scene = pd.concat([d["sweep"] for d in scenes.values()], ignore_index=True)
    pooled = pool_sweep(per_scene, len(scenes))
    n_truth = int(sum(len(d["t_xy"]) for d in scenes.values()))
    order = sorted(scenes, key=scene_date)
    summary = summary_table(per_scene, pooled, order, p.per_label)
    count = int(round(p.per_label * n_truth))
    sizes = pooled_sizes(scenes, pooled, p, count)
    out_dir = sp.out_dir
    paths = {}
    for name, t in [("pooled_sweep", pooled), ("summary_recall", summary),
                    ("pooled_size_breakdown", sizes)]:
        paths[name] = out_dir / f"{name}.csv"
        t.to_csv(paths[name], index=False)
    png = p.pooled_figure or out_dir / "recall_vs_detections_pooled.png"
    dates = ", ".join(scene_date(s)[5:] for s in order)
    recall_figure(pooled, sp, n_truth, p, f"{len(scenes)} frames pooled ({dates} 2019)", png,
                  grey={CURRENT_5D: ("s", ":"), "thor-linear-10m": (None, "--")})
    paths["pooled figure"] = png
    log(f"pooled: {len(scenes)} frames, {n_truth} labels; figure {png}")
    section = _pooled_section(scenes, order, pooled, summary, sizes, n_truth, count, p, png,
                              root)
    return {"paths": paths, "section": section}


def _pooled_section(scenes: dict[str, dict[str, Any]], order: list[str], pooled: pd.DataFrame,
                    summary: pd.DataFrame, sizes: pd.DataFrame, n_truth: int, count: int,
                    p: IwCompareParams, png: Path, root: Path) -> str:
    from iceberg_sar.thor.compare_figure import scene_date

    n_lab = {s: len(scenes[s]["t_xy"]) for s in order}
    head = [f"{scene_date(s)[5:]} ({n_lab[s]})" for s in order] + [f"pooled ({n_truth})"]
    lines = [
        f"## All {len(order)} frames, pooled",
        "",
        "Detections and matches summed per detector and PFA (THOR: score threshold) over the "
        "frames, as 5d pools; only settings every frame ran. Frames are named by date with "
        "their label count.",
        "",
        f"### Recall at {p.per_label:g} x the labels",
        "",
        f"Each curve read at {p.per_label:g} detections per label of that frame (pooled: "
        f"{count}), since the frames hold 46-267 labels and a fixed count would sit at a "
        "different point of each curve.",
        "",
        "| Detector | " + " | ".join(head) + " |",
        "|---|" + "---|" * len(head),
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {LABELS.get(r['detector'], r['detector'])} | "
                     + " | ".join(_fmt(r[s]) for s in [*order, POOLED]) + " |")
    lines += ["", "Blank: the curve doesn't reach that count in that frame (pooled: in "
              "every frame at a shared setting).", ""]
    if p.pooled_counts:
        matched = matched_table(pooled, p.pooled_counts)
        lines += ["### Pooled recall at matched detection counts", "",
                  "| Detector | " + " | ".join(f"@{n}" for n in p.pooled_counts)
                  + " | Detections covered |",
                  "|---|" + "---|" * len(p.pooled_counts) + "---|"]
        for _, r in matched.iterrows():
            lines.append(f"| {LABELS.get(r['detector'], r['detector'])} | "
                         + " | ".join(_fmt(r[f"recall@{n}"]) for n in p.pooled_counts)
                         + f" | {r['detections_min']}-{r['detections_max']} |")
        lines.append("")
    groups = [g for g in ("small 15-60 m", "medium 61-120 m", "large >120 m",
                          "other / unknown") if g in set(sizes["size"])]
    n_by = sizes.drop_duplicates("size").set_index("size")["labels"]
    lines += [f"### Pooled recall by IIP size, near {count} detections", "",
              "| Detector | PFA | Detections | "
              + " | ".join(f"{g} (n={int(n_by[g])})" for g in groups) + " |",
              "|---|---|---|" + "---|" * len(groups)]
    for det, g in sizes.groupby("detector", sort=False):
        r = g.set_index("size")
        pfa = "" if pd.isna(r["pfa"].iloc[0]) else f"{r['pfa'].iloc[0]:.0e}"
        n = "" if pd.isna(r["detections"].iloc[0]) else str(int(r["detections"].iloc[0]))
        cells = [f"{int(r.loc[s, 'matched'])}/{int(r.loc[s, 'labels'])}" if s in r.index
                 else "" for s in groups]
        lines.append(f"| {LABELS.get(det, det)} | {pfa} | {n} | " + " | ".join(cells) + " |")
    rel = png.relative_to(root).as_posix() if png.is_relative_to(root) else png.as_posix()
    lines += ["", "10 m rows without a PFA: 5d's pooled operating points (CA-CFAR at "
              "`cfar.pfa`, THOR at the current CA-CFAR's count).", "",
              f"![Pooled recall vs detections]"
              f"({Path(os.path.relpath(png, p.report.parent)).as_posix()})",
              "",
              f"Figure: `{rel}`. Tables: `pooled_sweep.csv`, `summary_recall.csv`, "
              f"`pooled_size_breakdown.csv` in `{p.sweep.out_dir.relative_to(root).as_posix()}`.",
              ""]
    return "\n".join(lines)


# --- figure ----------------------------------------------------------------------------------

def recall_figure(sweep: pd.DataFrame, sp: SweepParams, n_truth: int, p: IwCompareParams,
                  caption: str, png: Path,
                  grey: dict[str, tuple[str | None, str]] | None = None) -> Path:
    """Small multiples: one panel per reference detector, each against my CA-CFAR at 40 m,
    with the 10 m CA-CFAR and THOR from 5d in grey. Two hues per panel (validated pair)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iceberg_sar.figures import GRID, SURFACE, TEXT, TEXT_2

    mine, ref_c = "#eb6834", "#6250d6"   # orange = CA-CFAR (as in 5d), violet = reference
    # 10 m curves in grey: CA-CFAR with the rule (dotted, filled), without (dash-dot, hollow), THOR
    if grey is None:
        grey = {"cacfar-10m": ("s", ":"), CACFAR_10M: ("s", "-."),
                "thor-linear-10m": (None, "--")}
    n = len(sp.detectors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows + 0.9), sharex=True,
                             sharey=True, facecolor=SURFACE, squeeze=False)
    # Clipped at 10^4: past that the flooded PFAs (1e-1 gives ~10^5 objects) only squash the
    # range where the curves differ. Zero-detection points can't sit on a log axis.
    xmax = 1e4
    sweep = sweep[sweep["detections"] > 0]
    xs = np.append(np.geomspace(1, n_truth, 60), 1e5)
    for ax, det in zip(axes.flat, sp.detectors, strict=False):
        ax.set_facecolor(SURFACE)
        ax.plot(xs, np.minimum(xs / n_truth, 1), color=TEXT_2, lw=1, ls=(0, (1, 3)), zorder=1)
        for tag, (marker, ls) in grey.items():
            g = sweep[sweep["detector"] == tag].sort_values("detections")
            if g.empty:
                continue
            ax.plot(g["detections"], g["recall"], color="#9a9890", lw=1.5, ls=ls,
                    marker=marker, ms=4, zorder=2, label=LABELS[tag],
                    markerfacecolor=SURFACE if tag == CACFAR_10M else "#9a9890")
        # CA-CFAR variants share the orange (one detector); the length rule is dashed + hollow.
        # Only the paper-window variants: the like-for-like ones (own windows: tables only).
        paper = {t: v for t, v in sp.cacfar_tags.items() if v.windows == "paper"}
        mine_styles = {t: ("-", mine) if not v.length_rule else ("--", SURFACE)
                       for t, v in paper.items()}
        for tag, color, marker in ([(t, mine, "s") for t in paper] + [(det, ref_c, "o")]):
            g = sweep[sweep["detector"] == tag].sort_values("detections")
            ls, face = mine_styles.get(tag, ("-", color))
            ax.plot(g["detections"], g["recall"], color=color, lw=2, ls=ls, marker=marker, ms=6,
                    markerfacecolor=face, markeredgecolor=color if face == SURFACE else SURFACE,
                    markeredgewidth=1.2 if face == SURFACE else 1, zorder=3,
                    label=LABELS[tag] + " (40 m, paper windows)" if tag in paper
                    else "reference detector (40 m)")
            if tag == det:
                last = (-np.inf, -np.inf)   # skip labels that would sit on the previous one
                for _, r in g[g["detections"] <= xmax].iterrows():
                    pos = (np.log10(r["detections"]), r["recall"])
                    if abs(pos[0] - last[0]) < 0.15 and abs(pos[1] - last[1]) < 0.05:
                        continue
                    last = pos
                    ax.annotate(f"{r['setting']:.0e}".replace("e-0", "e-"),
                                (r["detections"], r["recall"]), textcoords="offset points",
                                xytext=(4, -11), fontsize=6.5, color=TEXT_2)
        ax.set_title(LABELS[det], loc="left", fontsize=9.5, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(10, xmax)
        ax.set_ylim(0, 1)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=8)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Detections in open water", color=TEXT_2, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Recall (match ≤ {p.radius_m:.0f} m)", color=TEXT_2, fontsize=8.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.01, 0.995), labelcolor=TEXT)
    fig.text(0.01, 0.005,
             f"{caption}, {n_truth} same-pass IIP labels. Reference "
             "points are labelled with their total PFA. Dotted diagonal: every detection is a "
             f"label. Detections within {p.near_ice_km:g} km of pack ice are dropped.",
             fontsize=7.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png


# --- report ----------------------------------------------------------------------------------

def _header(p: IwCompareParams) -> str:
    sp = p.sweep
    return "\n".join([
        "# Milestone 2b: reference CFAR detectors vs my CA-CFAR on IIP labels, April 2019 frames",
        "",
        "Generated by `python -m iceberg_sar.cli bench-iw-sweep` and `bench-iw-compare`.",
        "Same scoring as 5d (`thor-compare`), so the 40 m curves sit next to 5d's 10 m ones.",
        "",
        "## Setup",
        "",
        "- HyP3 RTC frame (gamma0, HH/HV) multilooked 4x4 to **40 m** (mean of linear "
        "intensity), so the paper's windows fit the library (9 / 13 px). Masks: land + 500 m, "
        "pack ice, nodata, as in 5d.",
        "- Reference detectors: the library through `external/cfar_runner`, paper windows "
        "(360 / 520 m), HH AND HV merge, per-band ENL for gamma / K, objects of 2-500 px.",
        "- My CA-CFAR: HV, co-pol check, ENL estimated on the 40 m HV. Variants: "
        + ", ".join(f"**{'with' if v.length_rule else 'without'} the length rule, "
                    f"{'paper' if v.windows == 'paper' else 'own'} windows**"
                    for v in sp.cacfar_tags.values())
        + ". Paper windows = the reference detectors' "
        f"({sp.cacfar_windows['inner_m']:g} / {sp.cacfar_windows['outer_m']:g} m), so "
        "paper windows without the length rule is the like-for-like run (the reference "
        "detectors have no such rule). Own windows = `cfar.guard_m` / `background_m`, as at "
        "10 m. Unlabelled CA-CFAR rows below use the paper windows.",
        f"- Total PFA: {', '.join(f'{x:.0e}' for x in sp.pfas)}; a detector stops after the "
        f"first PFA with more than {sp.max_objects} objects.",
        f"- Truth: same-pass IIP labels where the SAR is valid. Detections within "
        f"{p.near_ice_km:g} km of pack ice dropped. One-to-one matching at {p.radius_m:.0f} m.",
        "- Precision isn't measurable without AIS: compare **recall at matched detection "
        "counts**, interpolated in log(detections) between PFA points.",
        "",
    ])


def _fmt(v: float) -> str:
    return "" if pd.isna(v) else f"{v:.2f}"


def _scene_section(scene: str, sweep: pd.DataFrame, matched: pd.DataFrame, sizes: pd.DataFrame,
                   enl: dict[str, Any], cac: dict[str, dict[str, Any]], p: IwCompareParams,
                   png: Path, root: Path) -> str:
    counts = p.matched_counts
    c0 = next(iter(cac.values()))
    windows = ", ".join(f"{w} {c['guard_px']} / {c['background_px']}" for w, c in cac.items())
    lines = [
        f"## {scene}",
        "",
        f"- Labels compared: {int(sweep['truth'].iloc[0])}. ENL at 40 m (open-water box): "
        f"HH {enl['enl']['HH']}, HV {enl['enl']['HV']}. My CA-CFAR: ENL {c0['enl']}, "
        f"guard / background half-widths (px): {windows}.",
        "",
        "### Recall at matched detection counts",
        "",
        "| Detector | " + " | ".join(f"@{n}" for n in counts)
        + " | Detections covered | PFAs run |",
        "|---|" + "---|" * len(counts) + "---|---|",
    ]
    for _, r in matched.iterrows():
        lines.append(f"| {LABELS.get(r['detector'], r['detector'])} | "
                     + " | ".join(_fmt(r[f"recall@{n}"]) for n in counts)
                     + f" | {r['detections_min']}-{r['detections_max']} | "
                     + (f"{r['pfas_run']}" if not r["detector"].endswith("-10m") else "5d")
                     + " |")
    lines += ["", "Blank: the curve doesn't reach that count. 10 m rows (own windows): the "
              "CA-CFAR without the length rule is this sweep's `cacfar-nolength-10m` (current "
              "5d's curve on open-water frames); with the rule, the old 5d run with the rule "
              "always on (`ref_5d_dir`; current 5d's curve on icy frames); THOR, 5d's score "
              "sweep.", "",
              f"### Recall by IIP size, near {p.breakdown_count} detections", ""]
    groups = [g for g in ("small 15-60 m", "medium 61-120 m", "large >120 m",
                          "other / unknown") if g in set(sizes["size"])]
    n_by = sizes.drop_duplicates("size").set_index("size")["labels"]
    lines += ["| Detector | PFA | Detections | "
              + " | ".join(f"{g} (n={int(n_by[g])})" for g in groups) + " |",
              "|---|---|---|" + "---|" * len(groups)]
    for det, g in sizes.groupby("detector", sort=False):
        r = g.set_index("size")
        pfa = "" if pd.isna(r["pfa"].iloc[0]) else f"{r['pfa'].iloc[0]:.0e}"
        n = "" if pd.isna(r["detections"].iloc[0]) else str(int(r["detections"].iloc[0]))
        cells = [f"{int(r.loc[s, 'matched'])}/{int(r.loc[s, 'labels'])}" if s in r.index
                 else "" for s in groups]
        lines.append(f"| {LABELS.get(det, det)} | {pfa} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "10 m rows without a PFA: 5d's operating points (CA-CFAR at `cfar.pfa`, THOR "
              "at the same detection count).", "",
              f"Figure (local, gitignored): `{png.relative_to(root).as_posix()}`. Tables: "
              f"`sweep_scored.csv`, `recall_at_matched_counts.csv`, `size_breakdown.csv` "
              "next to it.", ""]
    return "\n".join(lines)
