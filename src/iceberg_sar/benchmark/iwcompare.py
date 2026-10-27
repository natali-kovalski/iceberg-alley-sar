"""Milestone 2b (one frame first): score the `bench-iw-sweep` detections against IIP labels.

Same protocol as 5d (`thor/compare.py`), so the 40 m curves sit next to the 10 m ones:
- truth: same-pass IIP labels (`pointval.load_truth`, iip-satellite), kept where the 10 m SAR
  is valid (`sar_valid_at`): the same 267 labels on 2019-04-29;
- detections: open water only (nothing within `viewer.near_ice_km` of pack ice);
- matching: one-to-one at `validation.max_dist_m`.

Precision isn't measurable without AIS, so detectors are compared by recall at matched numbers
of detections, read off each PFA curve by interpolation in log(detections).
The 10 m CA-CFAR (with the length rule) and THOR curves come from the 5d tables as reference.
"""

from __future__ import annotations

import json
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
from iceberg_sar.thor.compare import SIZE_GROUPS, open_water, score_points, xy

# 10 m reference curves from 5d: 5d detector name -> label used here
REF_10M = {"cfar": "cacfar-10m", "thor-linear": "thor-linear-10m"}
REF_10M_OPS = {"cfar": "cacfar-10m", "thor-linear matched-count": "thor-linear-10m"}
LABELS = {
    "cacfar-nolength": "my CA-CFAR, HV, no length rule",
    "cacfar": "my CA-CFAR, HV, with length rule",
    "gamma": "gamma (HH AND HV)",
    "lognorm": "log-normal (HH AND HV)",
    "k": "K (HH AND HV)",
    "wishart": "Wishart",
    "nis": "NIS",
    "idpolrad": "iDPolRAD (window cropped to 41 px)",
    "cacfar-10m": "my CA-CFAR 10 m, with length rule (5d)",
    CACFAR_10M: "my CA-CFAR 10 m, no length rule",
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
                                       or cfg.section("thor")["compare"]["out_dir"]))


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


def reference_10m(ref_5d: Path, scene: str) -> pd.DataFrame:
    """The 5d sweep rows (10 m CA-CFAR with length rule, THOR linear) for this scene."""
    f = ref_5d / "sweep.csv"
    if not f.is_file():
        return pd.DataFrame()
    t = pd.read_csv(f)
    t = t[(t["scene"] == scene) & t["detector"].isin(list(REF_10M))].copy()
    t["detector"] = t["detector"].map(REF_10M)
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
                   t_xy: np.ndarray, crs: Any, radius_m: float, count: int) -> pd.DataFrame:
    """Recall per IIP size group at each detector's PFA with the detection count nearest
    `count` (a real operating point, not an interpolation)."""
    size = truth["size"].map(SIZE_GROUPS).fillna("other / unknown").to_numpy()
    rows = []
    for tag, by_pfa in dets.items():
        if not by_pfa:
            continue
        pfa = min(by_pfa, key=lambda p: abs(np.log(max(len(by_pfa[p]), 1) / count)))
        d = by_pfa[pfa]
        _, hit = score_points(xy(d, crs), t_xy, radius_m)
        for group in sorted(set(size)):
            sel = size == group
            rows.append({"detector": tag, "pfa": pfa, "detections": len(d), "size": group,
                         "labels": int(sel.sum()), "matched": int(hit[sel].sum())})
    out = pd.DataFrame(rows)
    out["recall"] = (out["matched"] / out["labels"]).round(3)
    return out


def size_breakdown_10m(ref_5d: Path, scene: str) -> pd.DataFrame:
    """5d's size breakdown at its operating points (10 m CA-CFAR at its PFA, THOR at the same
    detection count) for this scene."""
    f = ref_5d / "breakdown.csv"
    if not f.is_file():
        return pd.DataFrame()
    b = pd.read_csv(f)
    b = b[(b["scene"] == scene) & (b["by"] == "size") & b["op"].isin(list(REF_10M_OPS))]
    b = b.drop_duplicates(["op", "bin"])
    return pd.DataFrame({"detector": b["op"].map(REF_10M_OPS), "pfa": np.nan,
                         "detections": np.nan, "size": b["bin"], "labels": b["labels"],
                         "matched": b["matched"], "recall": b["recall"]})


def run_iw_compare(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Path]:
    p = IwCompareParams.from_config(cfg)
    sp = p.sweep
    out: dict[str, Path] = {}
    sections = []
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
        tag0 = next(iter(sp.cacfar_tags))   # ENL and windows are the same for every variant
        cac = json.loads((work / tag0 / f"{tag0}_run.json").read_text(encoding="utf-8"))
        png = recall_figure(full, sp, len(t_xy), p, scene, work / "recall_vs_detections.png")
        log(f"{scene}: {len(t_xy)} labels; figure {png}")
        out[scene] = png
        sections.append(_scene_section(scene, full, matched, sizes, enl, cac, p, png, cfg.root))
    notes = ""
    if p.report.is_file():
        old = p.report.read_text(encoding="utf-8")
        if NOTES_MARKER in old:
            notes = old[old.index(NOTES_MARKER):]
    p.report.write_text(_header(p) + "\n".join(sections) + ("\n" + notes if notes else ""),
                        encoding="utf-8")
    out["report"] = p.report
    return out


# --- figure ----------------------------------------------------------------------------------

def recall_figure(sweep: pd.DataFrame, sp: SweepParams, n_truth: int, p: IwCompareParams,
                  scene: str, png: Path) -> Path:
    """Small multiples: one panel per reference detector, each against my CA-CFAR at 40 m,
    with the 10 m CA-CFAR and THOR from 5d in grey. Two hues per panel (validated pair)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iceberg_sar.figures import GRID, SURFACE, TEXT, TEXT_2

    mine, ref_c = "#eb6834", "#6250d6"   # orange = CA-CFAR (as in 5d), violet = reference
    # 10 m curves in grey: CA-CFAR with the rule (dotted, filled), without (dash-dot, hollow), THOR
    grey = {"cacfar-10m": ("s", ":"), CACFAR_10M: ("s", "-."), "thor-linear-10m": (None, "--")}
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
        # CA-CFAR variants share the orange (one detector); the length rule is dashed + hollow
        mine_styles = {t: ("-", mine) if not rule else ("--", SURFACE)
                       for t, rule in sp.cacfar_tags.items()}
        for tag, color, marker in ([(t, mine, "s") for t in sp.cacfar_tags] + [(det, ref_c, "o")]):
            g = sweep[sweep["detector"] == tag].sort_values("detections")
            ls, face = mine_styles.get(tag, ("-", color))
            ax.plot(g["detections"], g["recall"], color=color, lw=2, ls=ls, marker=marker, ms=6,
                    markerfacecolor=face, markeredgecolor=color if face == SURFACE else SURFACE,
                    markeredgewidth=1.2 if face == SURFACE else 1, zorder=3,
                    label=LABELS[tag] + " (40 m)" if tag in sp.cacfar_tags
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
    from iceberg_sar.thor.compare_figure import scene_date

    fig.text(0.01, 0.005,
             f"{scene_date(scene)} {scene[:3]}, {n_truth} same-pass IIP labels. Reference "
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
        "# Milestone 2b (one frame): reference CFAR detectors vs my CA-CFAR on IIP labels",
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
        "- My CA-CFAR: HV, its own windows (`cfar.guard_m` / `background_m`), co-pol check, "
        "ENL estimated on the 40 m HV. Run "
        + " and ".join(f"**{'with' if r else 'without'} the length rule**"
                       for r in sp.cacfar_tags.values())
        + " (the reference detectors have no such rule).",
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
                   enl: dict[str, Any], cac: dict[str, Any], p: IwCompareParams, png: Path,
                   root: Path) -> str:
    counts = p.matched_counts
    lines = [
        f"## {scene}",
        "",
        f"- Labels compared: {int(sweep['truth'].iloc[0])}. ENL at 40 m (open-water box): "
        f"HH {enl['enl']['HH']}, HV {enl['enl']['HV']}. My CA-CFAR: ENL {cac['enl']}, "
        f"guard / background {cac['guard_px']} / {cac['background_px']} px half-width.",
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
    lines += ["", "Blank: the curve doesn't reach that count. 10 m rows: 5d's CA-CFAR PFA "
              "sweep (with the length rule) and THOR's score sweep.", "",
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
    lines += ["", "10 m rows: 5d's operating points (CA-CFAR at `cfar.pfa`, THOR at the same "
              "detection count).", "",
              f"Figure (local, gitignored): `{png.relative_to(root).as_posix()}`. Tables: "
              f"`sweep_scored.csv`, `recall_at_matched_counts.csv`, `size_breakdown.csv` "
              "next to it.", ""]
    return "\n".join(lines)
