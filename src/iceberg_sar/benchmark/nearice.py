"""Revision C8: the labels within `viewer.near_ice_km` of pack ice, scored on their own.

Until the shared evaluation mask (revision C3) these labels counted as misses while every
detection near them was dropped, so whether the CA-CFAR's ice-strip length rule helps near ice
was never measured. Here labels and detections are both restricted to the near-ice strip
(`EvalDomain(zone="near_ice")`) on 5d's same-pass frames, and each detector is swept:
- CA-CFAR, HV + 6 dB HH check: length rule always on (the archived 5d run, `rule_always_dir`)
  vs off (2b's `cacfar-nolength-10m`);
- CA-CFAR, HH AND HV: current 5d (rule on in frames with >= 500 km² of pack ice: 04-17, 04-23)
  vs off (2b's `cacfar-nolength-10m-and`);
- the THOR pipeline (linear head), score threshold swept.
Recall is plotted against the detections in the strip: there the unmatched ones are mostly ice
clutter, which is what the rule is for. Retrospective: the frames were looked at before.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from iceberg_sar.config import Config
from iceberg_sar.evaluation import EvalDomain
from iceberg_sar.thor import PIPELINE_SHORT


@dataclass(frozen=True)
class NearIceParams:
    rule_always_dir: Path
    rule_always_pfas: list[float]
    report: Path
    figure: Path
    table: Path

    @classmethod
    def from_config(cls, cfg: Config) -> NearIceParams:
        n = cfg.section("benchmark")["near_ice"]
        return cls(rule_always_dir=cfg.root / n["rule_always_dir"],
                   rule_always_pfas=[float(x) for x in n["rule_always_pfas"]],
                   report=cfg.root / n["report"], figure=cfg.root / n["figure"],
                   table=cfg.root / n["table"])


SERIES = {   # key -> (label, colour, line style)
    "6db-rule": ("CA-CFAR, HV + 6 dB check, length rule always on", "#eb6834", "--"),
    "6db-norule": ("CA-CFAR, HV + 6 dB check, no length rule", "#eb6834", "-"),
    "and-auto": ("CA-CFAR, HH AND HV, rule in icy frames (current)", "#6250d6", "--"),
    "and-norule": ("CA-CFAR, HH AND HV, no length rule", "#6250d6", "-"),
    "thor": (f"{PIPELINE_SHORT}, linear head", "#2a78d6", "-"),
}


def sweep_near_ice(cfg: Config, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Per frame, series and setting: detections in the strip, labels, matched."""
    from iceberg_sar.benchmark.iwcompare import IwCompareParams, load_detections
    from iceberg_sar.benchmark.sweep import CACFAR_10M, CACFAR_10M_AND
    from iceberg_sar.thor.compare import SCORE_GRID, CompareParams, load_scene, score_points, xy

    p = NearIceParams.from_config(cfg)
    cp = replace(CompareParams.from_config(cfg), zone="near_ice")
    old = replace(cp, out_dir=p.rule_always_dir, cfar_pfas=p.rule_always_pfas,
                  cfar_pfa=p.rule_always_pfas[0])
    ip = IwCompareParams.from_config(cfg)
    domain = EvalDomain(near_ice_km=cp.near_ice_km, zone="near_ice")
    radius = ip.radius_m
    rows = []
    for name, kind in cp.scenes.items():
        if kind != "iip-satellite":
            continue
        s = load_scene(cfg, name, kind, cp)
        s_old = load_scene(cfg, name, kind, old)
        work = ip.sweep.out_dir / name
        series = {"and-auto": s.cfar, "6db-rule": s_old.cfar,
                  "6db-norule": load_detections(work / CACFAR_10M, CACFAR_10M, domain),
                  "and-norule": load_detections(work / CACFAR_10M_AND, CACFAR_10M_AND, domain)}
        for key, by in series.items():
            for setting, det in by.items():
                tp, _ = score_points(xy(det, s.crs), s.truth_xy, radius)
                rows.append({"scene": name, "series": key, "setting": setting,
                             "detections": len(det), "labels": len(s.truth_xy), "matched": tp})
        thor = s.thor.get("thor-linear")
        if thor is not None:
            d_xy, sc = xy(thor, s.crs), thor["score"].to_numpy()
            for t in SCORE_GRID:
                tp, _ = score_points(d_xy[sc >= t], s.truth_xy, radius)
                rows.append({"scene": name, "series": "thor", "setting": float(t),
                             "detections": int((sc >= t).sum()), "labels": len(s.truth_xy),
                             "matched": tp})
        log(f"{name}: {len(s.truth_xy)} labels within {cp.near_ice_km:g} km of pack ice")
    return pd.DataFrame(rows)


def pooled_near_ice(t: pd.DataFrame) -> pd.DataFrame:
    """Summed over the frames per (series, setting), for settings every frame has."""
    n = t.scene.nunique()
    g = t.groupby(["series", "setting"], as_index=False).agg(
        detections=("detections", "sum"), labels=("labels", "sum"), matched=("matched", "sum"),
        frames=("scene", "nunique"))
    g = g[g.frames == n].drop(columns="frames")
    g["recall"] = (g.matched / g.labels).round(3)
    return g


def figure(pooled: pd.DataFrame, n_labels: int, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iceberg_sar.figures import GRID, SURFACE, TEXT, TEXT_2

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, (label, color, ls) in SERIES.items():
        g = pooled[(pooled.series == key) & (pooled.detections > 0)].sort_values("detections")
        if g.empty:
            continue
        ax.plot(g.detections, g.recall, color=color, ls=ls, lw=2,
                marker=None if key == "thor" else "o", ms=4, label=label)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Detections within 5 km of pack ice (4 frames)", color=TEXT_2)
    ax.set_ylabel(f"Recall of the {n_labels} near-ice labels", color=TEXT_2)
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    ax.set_title("Near pack ice: length rule on vs off (retrospective, April 2019 frames)",
                 loc="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def report(t: pd.DataFrame, pooled: pd.DataFrame, fig: Path, p: NearIceParams) -> str:
    import os

    from iceberg_sar.thor.compare_figure import scene_date

    n_lab = int(t.drop_duplicates("scene").labels.sum())
    per = t.drop_duplicates("scene")[["scene", "labels"]]
    rel = Path(os.path.relpath(fig, p.report.parent)).as_posix()
    L = ["# Near pack ice: does the length rule help? (revision C8)", "",
         "Generated by `python -m iceberg_sar.cli bench-near-ice`. Labels and detections both "
         f"restricted to within 5 km of the buffered pack-ice mask: {n_lab} labels ("
         + ", ".join(f"{scene_date(r.scene)}: {r.labels}" for r in per.itertuples()) + "). "
         "Retrospective: these frames were looked at when the rule and its 500 km² trigger "
         "were set.", "",
         f"![Near-ice recall vs detections]({rel})",
         "", "## Shared settings, rule on vs off", "",
         "| Pair | PFA | Rule on: detections / matched | Rule off: detections / matched |",
         "|---|---|---|---|"]
    for name, on, off in (("HV + 6 dB check", "6db-rule", "6db-norule"),
                          ("HH AND HV", "and-auto", "and-norule")):
        a = pooled[pooled.series == on].set_index("setting")
        b = pooled[pooled.series == off].set_index("setting")
        for s in sorted(set(a.index) & set(b.index), reverse=True):
            L.append(f"| {name} | {s:.0e} | {int(a.loc[s, 'detections'])} / "
                     f"{int(a.loc[s, 'matched'])} | {int(b.loc[s, 'detections'])} / "
                     f"{int(b.loc[s, 'matched'])} |")
    L += ["", "HH AND HV: the rule is on only in 04-17 and 04-23 (>= 500 km² of pack ice), so on "
          "04-29 both columns are the same run. Unmatched detections in the strip are mostly ice "
          "clutter, but not all: an unlabelled berg or ship counts there too.", "",
          "Read the figure, not the rows: at a shared PFA the rule always has fewer detections, "
          "so the question is recall at the same number of detections in the strip.", ""]
    return "\n".join(L)


def run_near_ice(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Path]:
    p = NearIceParams.from_config(cfg)
    t = sweep_near_ice(cfg, log)
    pooled = pooled_near_ice(t)
    p.table.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(p.table, index=False)
    pooled.to_csv(p.table.with_name(p.table.stem + "_pooled.csv"), index=False)
    n_lab = int(t.drop_duplicates("scene").labels.sum())
    fig = figure(pooled, n_lab, p.figure)
    p.report.write_text(report(t, pooled, fig, p), encoding="utf-8")
    return {"table": p.table, "figure": fig, "report": p.report}

