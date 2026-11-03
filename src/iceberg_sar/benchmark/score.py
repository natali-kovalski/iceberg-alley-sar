"""Milestone 1f: score the 1e sweep against the Færch 2023 labels and compare with the paper.

Protocol (paper sec. 3.4, `faerch2023.xml`): a detection within a label's search radius
(250 m verified, 1789 m interpolated) is a true positive, one detection per label, the nearest
first; other detections are false positives; labels without one are false negatives.
Matching (`score.matching`, revision C4): `greedy` by distance over all (label, detection)
pairs inside the label's radius, closest first, each side used once (the paper's wording, the
default); or `optimal`, the one-to-one assignment with the most matches (then the least total
distance). Radii in true metres (`radius_units: metres`, default) or in EPSG:5938 metres, the
paper's grid (`epsg5938`: each radius / that label's scale factor, ~1.07).

- All-labels: every label in the domain, its own radius (the paper's protocol, Table 3).
- Verified-only recall (revision C5): the share of verified labels matched in that same
  all-labels matching. It replaces the old verified-only F, which mixed a different label set
  with its own false-positive accounting.
- Scoring domain (`evaluation.EvalDomain`, revision C3): the scoring area, applied to labels
  (at their S1 position) and detections alike. Before 2026-11-02 only detections were clipped
  and all 492 labels counted, so the 5 labels outside the area were misses by construction.
  Detections near a Path A "unknown SAR object" are dropped too (paper: masked out).
- Numerically unreliable runs (revision C2): iDPolRAD computes `1 - pfa`, which is exactly 1.0
  in float64 for PFA <= 1e-17, so those runs all use one threshold. Runs whose effective PFA
  (1 - (1 - pfa)) is off by more than `unreliable_rel_err` are flagged `unreliable` and left
  out of every optimum.

`reference.yaml` holds Tables 3 and 5, parsed from the XML by `write_reference`; F at the paper's
optimum is derived from Table 3 (2 TP / (2 TP + FP + FN)), which the paper does not print.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from iceberg_sar.benchmark.labels import SIZE_CLASSES
from iceberg_sar.benchmark.scenes import TABLE_NS, _text
from iceberg_sar.benchmark.sweep import sweep_dir
from iceberg_sar.config import Config
from iceberg_sar.evaluation import EvalDomain

# Column headers of Tables 3 and 5 -> detector names in config / the runner
PAPER_NAMES = {"Gamma": "gamma", "iDPolRAD": "idpolrad", "K": "k", "Log-normal": "lognorm",
               "NIS": "nis", "Wishart": "wishart"}
# 2a: my CA-CFAR (`sweep.cacfar_tag`), scored with the rest; no paper value to compare with
MINE = {"cacfar": "My CA-CFAR", "cacfar-nolength": "My CA-CFAR, no length rule",
        # 2c: HH AND HV CA-CFAR instead of the 6 dB co-pol check
        "cacfar-and": "My CA-CFAR, HH AND HV",
        "cacfar-nolength-and": "My CA-CFAR, HH AND HV, no length rule"}
DISPLAY = {v: k for k, v in PAPER_NAMES.items()} | MINE
_PFA = re.compile(r"1\s*×\s*10\s*[-−–]\s*(\d+)")


# --- reference.yaml --------------------------------------------------------------------------

def _table(root: ET.Element, label: str) -> list[list[str]]:
    for tw in root.iter("table-wrap"):
        if _text(tw.find("label")) == label:
            return [[_text(c) for c in r.iter(TABLE_NS + "entry")]
                    for r in tw.iter(TABLE_NS + "row")]
    raise ValueError(f"{label} not found in the paper XML")


def _pfa(cell: str) -> float:
    m = _PFA.search(cell)
    if m is None:
        raise ValueError(f"No PFA in table cell {cell!r}")
    return float(f"1e-{m[1]}")


def parse_reference(xml_path: Path) -> dict[str, Any]:
    """Tables 3 and 5 (Sentinel-1) from the paper, with table and row of every value."""
    root = ET.parse(xml_path).getroot()
    t3 = _table(root, "Table 3")
    head, pfas = t3[0], t3[1]
    rows3 = {r[0]: (i, r) for i, r in enumerate(t3) if r[0] in ("FN", "FP", "TP")}
    table3: dict[str, Any] = {}
    for col, name in enumerate(head):
        if name not in PAPER_NAMES:
            continue
        e: dict[str, Any] = {"pfa": _pfa(pfas[col]), "source": {"table": 3, "column": name,
                                                                  "pfa_row": 2}}
        for key, (i, r) in rows3.items():
            e[key] = int(r[col])
            e["source"][f"{key}_row"] = i + 1
        tp, fp, fn = e["TP"], e["FP"], e["FN"]
        e["derived"] = {"recall": round(tp / (tp + fn), 4), "precision": round(tp / (tp + fp), 4),
                        "f": round(2 * tp / (2 * tp + fp + fn), 4)}
        table3[PAPER_NAMES[name]] = e

    t5 = _table(root, "Table 5")
    head5, pfas5 = t5[0], t5[1]
    table5: dict[str, Any] = {}
    for col, name in enumerate(head5):
        if name not in PAPER_NAMES:
            continue
        e = {"pfa": _pfa(pfas5[col]), "source": {"table": 5, "column": name, "pfa_row": 2},
             "detected_pct": {}, "n": {}}
        for i, r in enumerate(t5[2:], start=3):
            cls = r[0].lower()
            e["detected_pct"][cls] = float(r[col].replace("%", "").strip())
            e["n"][cls] = int(r[1])
            e["source"][f"{cls}_row"] = i
        table5[PAPER_NAMES[name]] = e
    notes = [f"{DISPLAY[d]}: Table 3 optimum PFA {table3[d]['pfa']:.0e}, "
             f"Table 5 {table5[d]['pfa']:.0e}"
             for d in table3 if table3[d]["pfa"] != table5[d]["pfa"]]
    return {"table3": table3, "table5": table5, "pfa_mismatch_table3_vs_table5": notes}


def write_reference(cfg: Config) -> Path:
    fc = cfg.section("benchmark")["faerch2023"]
    ref = parse_reference(cfg.root / fc["paper_xml"])
    out = cfg.root / fc["score"]["reference"]
    header = (
        "# Færch et al. 2023, Sentinel-1 results. Generated by `bench-reference` from\n"
        f"# {fc['paper_xml']} (Tables 3 and 5); never edit values by hand.\n"
        "# `derived` (recall, precision, F) is computed from Table 3; the paper prints no F\n"
        "# values.\n"
        "# Rows are 1-based rows of the table body, header rows included.\n"
        "# checked_against_paper: set to true once checked by hand against the published tables.\n")
    body = yaml.safe_dump({"checked_against_paper": False, **ref}, sort_keys=False,
                          allow_unicode=True)
    out.write_text(header + body, encoding="utf-8")
    return out


def load_reference(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- matching ----------------------------------------------------------------------------------

MATCHERS = ("greedy", "optimal")


@dataclass(frozen=True)
class MatchSettings:
    matcher: str = "greedy"
    radius_units: str = "metres"                # metres (true) | epsg5938
    radius_interpolated_m: float | None = None  # None: the labels' own (1789 m)

    @classmethod
    def from_config(cls, cfg: Config) -> MatchSettings:
        m = cfg.section("benchmark")["faerch2023"]["score"].get("matching") or {}
        r = m.get("radius_interpolated_m")
        out = cls(matcher=m.get("matcher", "greedy"),
                  radius_units=m.get("radius_units", "metres"),
                  radius_interpolated_m=None if r is None else float(r))
        out.check()
        return out

    def check(self) -> None:
        if self.matcher not in MATCHERS:
            raise ValueError(f"score.matching.matcher must be one of {MATCHERS}")
        if self.radius_units not in ("metres", "epsg5938"):
            raise ValueError("score.matching.radius_units must be metres or epsg5938")

    def describe(self) -> str:
        r = ("labels' own" if self.radius_interpolated_m is None
             else f"{self.radius_interpolated_m:g} m")
        return (f"{self.matcher} matching, radii in "
                f"{'true' if self.radius_units == 'metres' else 'EPSG:5938'} metres, "
                f"interpolated radius {r}")


def apply_match_settings(labels: pd.DataFrame, ms: MatchSettings) -> pd.DataFrame:
    """Labels with `radius_m` set for these settings (scoring CRS metres)."""
    r = labels.radius_m.to_numpy(float).copy()
    if ms.radius_interpolated_m is not None:
        r[(labels.status != "verified").to_numpy()] = ms.radius_interpolated_m
    if ms.radius_units == "epsg5938":
        r = r / labels["scale_5938"].to_numpy(float)
    return labels.assign(radius_m=r)


def optimal_match(det_xy: np.ndarray, lab_xy: np.ndarray, radius: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairs with the most matches, then the least total distance, each within the
    label's radius (`linear_sum_assignment` on the detections near any label)."""
    from scipy.optimize import linear_sum_assignment

    if len(det_xy) == 0 or len(lab_xy) == 0:
        return np.empty(0, int), np.empty(0, int)
    tree = cKDTree(det_xy)
    near = sorted({i for j in range(len(lab_xy))
                   for i in tree.query_ball_point(lab_xy[j], radius[j])})
    if not near:
        return np.empty(0, int), np.empty(0, int)
    cand = det_xy[near]
    d = np.linalg.norm(cand[:, None, :] - lab_xy[None, :, :], axis=2)
    ok = d <= radius[None, :]
    big = float(radius.max()) * (len(lab_xy) + 1) * 10   # any match beats any distance saving
    rows, cols = linear_sum_assignment(np.where(ok, d, big))
    keep = ok[rows, cols]
    return np.asarray(near)[rows[keep]], cols[keep]


def match(det_xy: np.ndarray, lab_xy: np.ndarray, radius: np.ndarray, matcher: str = "greedy"
          ) -> tuple[np.ndarray, np.ndarray]:
    return (optimal_match if matcher == "optimal" else greedy_match)(det_xy, lab_xy, radius)


def greedy_match(det_xy: np.ndarray, lab_xy: np.ndarray, radius: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairs (det index, label index), closest first, each within the label's radius."""
    if len(det_xy) == 0 or len(lab_xy) == 0:
        return np.empty(0, int), np.empty(0, int)
    tree = cKDTree(det_xy)
    cand = [(float(np.hypot(*(det_xy[i] - lab_xy[j]))), i, j)
            for j in range(len(lab_xy)) for i in tree.query_ball_point(lab_xy[j], radius[j])]
    cand.sort()
    used_d: set[int] = set()
    used_l: set[int] = set()
    pairs = []
    for _, i, j in cand:
        if i not in used_d and j not in used_l:
            used_d.add(i)
            used_l.add(j)
            pairs.append((i, j))
    p = np.array(pairs, int).reshape(-1, 2)
    return p[:, 0], p[:, 1]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    r = tp / (tp + fn) if tp + fn else np.nan
    p = tp / (tp + fp) if tp + fp else np.nan
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return r, p, f


def score_points(det_xy: np.ndarray, labels: pd.DataFrame, matcher: str = "greedy"
                 ) -> dict[str, Any]:
    """All-labels scores and verified-only recall for one detection set. `labels` (already in
    the scoring domain): x, y, radius_m, status, size_class."""
    lab_xy = labels[["x", "y"]].to_numpy(float)
    radius = labels.radius_m.to_numpy(float)
    ver = (labels.status == "verified").to_numpy()
    _, li = match(det_xy, lab_xy, radius, matcher)
    tp = len(li)
    out: dict[str, Any] = {"n_det": len(det_xy), "TP": tp, "FP": len(det_xy) - tp,
                           "FN": len(labels) - tp}
    out["recall"], out["precision"], out["f"] = _prf(tp, out["FP"], out["FN"])
    hit = np.zeros(len(labels), bool)
    hit[li] = True
    for name, _, _ in SIZE_CLASSES:
        m = (labels.size_class == name).to_numpy()
        out[f"recall_{name}"] = hit[m].mean() if m.any() else np.nan

    n_v = int(ver.sum())
    out |= {"n_v": n_v, "TP_v": int(hit[ver].sum()),
            "recall_v": hit[ver].mean() if n_v else np.nan}
    return out


# --- inputs -------------------------------------------------------------------------------------

def load_labels(cfg: Config, source: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Labels (x, y in the scene CRS) and the unknown-object positions (Path A only)."""
    fc = cfg.section("benchmark")["faerch2023"]
    crs = fc["crs"]
    if source == "faerch":
        d = cfg.root / fc["labels_faerch"]["out_dir"]
        lab = gpd.read_file(d / "labels_faerch.geojson").to_crs(crs)
        unk = gpd.read_file(d / "unknown_objects_faerch.geojson").to_crs(crs)
        unknown = np.column_stack([unk.geometry.x, unk.geometry.y])
    elif source == "reconstructed":
        lab = gpd.read_file(cfg.root / fc["labels"]["dir"] / "labels_reconstructed.geojson")
        lab = lab.to_crs(crs)
        unknown = np.empty((0, 2))
    else:
        raise ValueError(f"labels source must be faerch or reconstructed, not {source!r}")
    df = pd.DataFrame({"label_id": lab.label_id, "status": lab.status, "radius_m": lab.radius_m,
                       "size_class": lab.size_class, "x": lab.geometry.x, "y": lab.geometry.y})
    # EPSG:5938 metres per true metre at the label (Path A: from the paper-unit length)
    if {"length_m", "length_m_paper"} <= set(lab.columns):
        k = (lab.length_m_paper / lab.length_m.where(lab.length_m > 0)).to_numpy(float)
        df["scale_5938"] = np.where(np.isfinite(k), k, np.nanmedian(k))
    else:
        df["scale_5938"] = np.nan
    return df, unknown


def eval_domain(cfg: Config, kind: str) -> EvalDomain:
    """The EW scoring domain (revision C3): the scoring area `kind`, for labels and detections."""
    return EvalDomain(area=scoring_aoi(cfg, kind))


def labels_in_domain(labels: pd.DataFrame, domain: EvalDomain) -> pd.DataFrame:
    return labels[domain.keep(labels[["x", "y"]].to_numpy(float))].reset_index(drop=True)


def effective_pfa(pfa: float) -> float:
    """PFA after the `1 - pfa` round trip in float64 (iDPolRAD's `pde`)."""
    return float(1.0 - np.float64(1.0 - pfa))


def is_unreliable(detector: str, pfa: float, cfg_score: dict[str, Any]) -> bool:
    u = cfg_score.get("unreliable") or {}
    if detector not in u.get("pde_detectors", []):
        return False
    return abs(effective_pfa(pfa) / pfa - 1) > float(u.get("rel_err", 0.5))


def scoring_aoi(cfg: Config, kind: str) -> Any:
    """Scoring area in the scene CRS (shapely geometry)."""
    from iceberg_sar.benchmark.pathb import _swath_polygon
    from iceberg_sar.benchmark.reproduce import scene_paths

    fc = cfg.section("benchmark")["faerch2023"]
    if kind == "fig1":
        outl = gpd.read_file(cfg.root / fc["fig1_outlines"]).to_crs(fc["crs"])
        geom = _swath_polygon(scene_paths(cfg).swath).union_all()
        for name in ("pink_s2", "alos2"):
            geom = geom.intersection(outl[outl.name == name].union_all())
        return geom
    if kind == "reconstructed":
        f = cfg.root / fc["labels"]["dir"] / "aoi_reconstructed.geojson"
        return gpd.read_file(f).to_crs(fc["crs"]).union_all()
    raise ValueError(f"score.aoi must be fig1 or reconstructed, not {kind!r}")


def detections_xy(path: Path, crs: str, domain: EvalDomain, unknown: np.ndarray, ignore_m: float
                  ) -> tuple[np.ndarray, int]:
    """Detection centroids in the scoring domain, minus those near unknown objects (and how
    many)."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        return np.empty((0, 2)), 0
    return detections_xy_from(gdf.to_crs(crs), domain, unknown, ignore_m)


def detections_xy_from(gdf: gpd.GeoDataFrame, domain: EvalDomain, unknown: np.ndarray,
                       ignore_m: float) -> tuple[np.ndarray, int]:
    """`detections_xy` for detections already loaded in the scoring CRS."""
    if gdf.empty:
        return np.empty((0, 2)), 0
    xy = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    xy = xy[domain.keep(xy)]
    if len(unknown) and len(xy):
        near = cKDTree(unknown).query(xy, distance_upper_bound=ignore_m)[0] <= ignore_m
        return xy[~near], int(near.sum())
    return xy, 0


def score_sweep(cfg: Config, labels: pd.DataFrame, unknown: np.ndarray, domain: EvalDomain,
                log: Callable[[str], None] = print, ms: MatchSettings | None = None
                ) -> pd.DataFrame:
    """One row per (detector, PFA) of the sweep; over-cap PFAs get NaN scores. `labels` are
    taken into `domain` here, so labels and detections share it."""
    fc = cfg.section("benchmark")["faerch2023"]
    ms = ms or MatchSettings.from_config(cfg)
    labels = apply_match_settings(labels_in_domain(labels, domain), ms)
    out_dir = sweep_dir(cfg)
    runtime = pd.read_csv(out_dir / "sweep_runtime.csv")
    rows = []
    for r in runtime.itertuples():
        row = {"detector": r.detector, "pfa": r.pfa, "pfa_per_channel": r.pfa_per_channel,
               "objects_raw": r.objects_raw, "over_cap": bool(r.over_cap),
               "unreliable": is_unreliable(r.detector, r.pfa, fc["score"]),
               "labels_scored": len(labels)}
        if not r.over_cap:
            path = out_dir / r.detector / str(r.geojson)
            xy, n_unk = detections_xy(path, fc["crs"], domain, unknown,
                                      float(fc["score"]["unknown_ignore_m"]))
            row |= {"ignored_unknown": n_unk, **score_points(xy, labels, ms.matcher)}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["detector", "pfa"]).reset_index(drop=True)
    log(f"scored {len(df)} (detector, PFA) runs")
    return df


def optimum(scores: pd.DataFrame, col: str = "f") -> pd.DataFrame:
    """Row of the highest `col` per detector (ties: the lower PFA); runs flagged `unreliable`
    (revision C2) never count."""
    ok = scores.dropna(subset=[col])
    if "unreliable" in ok:
        ok = ok[~ok.unreliable.astype(bool)]
    ok = ok.sort_values(["detector", col, "pfa"],
                                                 ascending=[True, False, True])
    return ok.groupby("detector", sort=False).head(1).set_index("detector")


# --- figure and report --------------------------------------------------------------------------

LIGHT_BLUE = "#2a78d6"   # dataviz categorical slot 1 (validated); one hue per panel
ORANGE = "#eb6834"       # slot 2: digitized Fig. 6, if present
INK = "#1f1f1e"
GREY = "#9a9a96"         # the other stack variant (context, not a series of its own)


def f_figure(scores: pd.DataFrame, ref: dict[str, Any], fig6: pd.DataFrame | None,
             detectors: list[str], out: Path, note: str,
             alt: pd.DataFrame | None = None, alt_label: str = "") -> Path:
    """Small multiples: F vs PFA per detector (all labels), verified-only recall, paper
    optimum; `alt`: the other stack variant's all-labels F, drawn in grey. Runs flagged
    `unreliable` (revision C2) are grey crosses, not part of the curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = max(3, -(-len(detectors) // 2))
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7), sharex=True, sharey=True)
    for ax in axes.flat[len(detectors):]:
        ax.set_visible(False)
    for ax, det in zip(axes.flat, detectors, strict=False):
        d = scores[scores.detector == det].sort_values("pfa")
        bad = d.unreliable.astype(bool) if "unreliable" in d else pd.Series(False, d.index)
        u, d = d[bad], d[~bad]
        x = np.log10(d.pfa)
        ax.plot(x, d.f, color=LIGHT_BLUE, lw=2, marker="o", ms=4, label="mine, F (all labels)")
        ax.plot(x, d.recall_v, color=LIGHT_BLUE, lw=1.2, ls=(0, (1, 2)),
                label="mine, verified-only recall")
        if len(u):
            ax.plot(np.log10(u.pfa), u.f, color=GREY, marker="x", ms=6, ls="none",
                    label="numerically unreliable PFA (not scored)")
        if alt is not None:
            a = alt[alt.detector == det].sort_values("pfa")
            if "unreliable" in a:
                a = a[~a.unreliable.astype(bool)]
            ax.plot(np.log10(a.pfa), a.f, color=GREY, lw=1.5, label=alt_label)
        if fig6 is not None:
            g = fig6[fig6.detector == det].sort_values("pfa")
            ax.plot(np.log10(g.pfa), g.f, color=ORANGE, lw=2, label="paper Fig. 6 (digitized)")
        if det in ref["table3"]:
            t = ref["table3"][det]
            ax.plot(np.log10(t["pfa"]), t["derived"]["f"], marker="*", ms=14, color=INK,
                    ls="none", label="paper optimum (Table 3)")
        cap = d[d.over_cap]
        if len(cap):
            ax.axvline(np.log10(cap.pfa.min()), color="#9a9a96", lw=1, ls=":")
        ax.set_title(DISPLAY.get(det, det), fontsize=11, color=INK)
        if det in MINE:
            ax.text(0.03, 0.03, "no paper value", transform=ax.transAxes, fontsize=8,
                    color="#5f5e5a")
        ax.grid(True, color="#e6e6e3", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlim(-21.5, -0.5)
        ax.set_xticks(range(-21, 0, 4))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("log10 PFA (total)")
    for ax in axes[:, 0]:
        ax.set_ylabel("F score / verified-only recall")
    leg: dict[str, Any] = {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels(), strict=True):
            leg.setdefault(l_, h_)
    fig.legend(list(leg.values()), list(leg), loc="upper center", ncol=min(len(leg), 4),
               frameon=False)
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


# Paper Fig. 6 look (seaborn "darkgrid"): legend order, colours and markers read off the figure
FIG6_STYLE = {"lognorm": ("#3e6fa8", "1"), "gamma": ("#5ec8ec", "2"), "k": ("#2e9a3c", "3"),
              "idpolrad": ("#dcc23a", "4"), "nis": ("#ec6470", "+"), "wishart": ("#a3336f", "x"),
              # 2a, not in the paper: black, so they read as the extra series
              "cacfar": ("#1f1f1e", "o"), "cacfar-nolength": ("#1f1f1e", "s"),
              "cacfar-and": ("#eb6834", "o"), "cacfar-nolength-and": ("#eb6834", "s")}


def fig6_style_figure(scores: pd.DataFrame, detectors: list[str], out: Path, subtitle: str,
                      note: str) -> Path:
    """All detectors' all-labels F vs PFA on one panel, styled like the paper's Fig. 6.
    Over-cap PFAs (no score) are left out, so a curve can end before 1e-1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_facecolor("#eaeaf2")
    for det in [d for d in FIG6_STYLE if d in detectors]:
        d = scores[(scores.detector == det) & np.isfinite(scores.f)]
        if "unreliable" in d:   # revision C2: not part of the curve
            d = d[~d.unreliable.astype(bool)]
        d = d.sort_values("pfa")
        color, marker = FIG6_STYLE[det]
        mine = det in MINE
        ax.plot(d.pfa, d.f, color=color, marker=marker, ms=6 if mine else 9, mew=1.5, lw=2,
                ls="--" if det.startswith("cacfar-nolength") else "-",
                mfc="none" if mine else color,
                label=DISPLAY[det].replace("iDPolRAD", "iDPolRad"))
    ax.set_xscale("log")
    ax.set_xlim(10 ** -22.5, 10 ** -0.4)
    ax.set_xticks([10.0 ** e for e in range(-22, 0, 4)])
    ax.minorticks_off()
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(length=0, labelsize=12, colors="#262626")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_rotation(45)
    ax.grid(True, color="white", lw=1.2)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("PFA", fontsize=14)
    ax.set_ylabel("F-Score", fontsize=14, rotation=45, labelpad=18)
    ax.set_title(subtitle, fontsize=14)
    # outside the axes: with 8 curves, every corner inside covers one of them
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=12, facecolor="#eaeaf2",
              edgecolor="#cccccc")
    fig.suptitle("F Score (Accuracy)", fontsize=18)
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


def compare_table(scores: pd.DataFrame, ref: dict[str, Any], tol: float,
                  sens: pd.DataFrame | None, alt: pd.DataFrame | None = None) -> pd.DataFrame:
    best = optimum(scores)
    rows = []
    for det, t in ref["table3"].items():
        if det not in best.index:
            continue
        b = best.loc[det]
        f_ref = t["derived"]["f"]
        row = {"detector": det, "paper_pfa": t["pfa"], "pfa": b.pfa,
               "paper_TP": t["TP"], "TP": int(b.TP), "paper_FP": t["FP"], "FP": int(b.FP),
               "paper_FN": t["FN"], "FN": int(b.FN), "paper_f": f_ref, "f": b.f,
               "diff": b.f - f_ref, "within": abs(b.f - f_ref) <= tol,
               "recall_v": b.recall_v}
        if sens is not None and det in optimum(sens).index:
            row["f_sens"] = optimum(sens).loc[det].f
        if alt is not None:
            a = optimum(alt)
            row["f_alt"] = a.loc[det].f if det in a.index else np.nan
            row["pfa_alt"] = a.loc[det].pfa if det in a.index else np.nan
            for k in ("TP", "FP", "FN"):
                row[f"{k}_alt"] = int(a.loc[det][k]) if det in a.index else None
        rows.append(row)
    for det in [d for d in MINE if d in best.index]:
        b = best.loc[det]
        row = {"detector": det, "paper_pfa": np.nan, "pfa": b.pfa, "paper_TP": None,
               "TP": int(b.TP), "paper_FP": None, "FP": int(b.FP), "paper_FN": None,
               "FN": int(b.FN), "paper_f": np.nan, "f": b.f, "diff": np.nan, "within": None,
               "recall_v": b.recall_v}
        if sens is not None:
            s_best = optimum(sens)
            row["f_sens"] = s_best.loc[det].f if det in s_best.index else np.nan
        if alt is not None:
            a = optimum(alt)
            row["f_alt"] = a.loc[det].f if det in a.index else np.nan
            row["pfa_alt"] = a.loc[det].pfa if det in a.index else np.nan
            for k in ("TP", "FP", "FN"):
                row[f"{k}_alt"] = int(a.loc[det][k]) if det in a.index else None
        rows.append(row)
    return pd.DataFrame(rows)


def same_pfa(pfas: Any, pfa: float) -> np.ndarray:
    """Exact match on the log-spaced PFA grid (revision C1). `np.isclose` with its default
    atol=1e-8 calls every PFA below 1e-8 equal, so 1e-19 used to select the 1e-21 row."""
    return np.isclose(np.log10(np.asarray(pfas, float)), np.log10(pfa), rtol=0, atol=1e-6)


def size_table(scores: pd.DataFrame, ref: dict[str, Any]) -> pd.DataFrame:
    best = optimum(scores)
    rows = []
    for det, t in ref["table5"].items():
        if det not in best.index:
            continue
        at5 = scores[(scores.detector == det) & same_pfa(scores.pfa, t["pfa"])]
        at5 = at5.iloc[0] if len(at5) else None
        for name, _, _ in SIZE_CLASSES:
            rows.append({"detector": det, "size": name, "paper_pct": t["detected_pct"][name],
                         "paper_pfa": t["pfa"],
                         "pct_at_paper_pfa": np.nan if at5 is None
                         else 100 * at5[f"recall_{name}"],
                         "pct_at_my_opt": 100 * best.loc[det][f"recall_{name}"]})
    for det in [d for d in MINE if d in best.index]:
        for name, _, _ in SIZE_CLASSES:
            rows.append({"detector": det, "size": name, "paper_pct": np.nan,
                         "paper_pfa": np.nan, "pct_at_paper_pfa": np.nan,
                         "pct_at_my_opt": 100 * best.loc[det][f"recall_{name}"]})
    return pd.DataFrame(rows)


def _e(p: float) -> str:
    return f"{p:.0e}".replace("e-0", "e-")


def report(cfg: Config, source: str, ref: dict[str, Any], cmp: pd.DataFrame, sizes: pd.DataFrame,
           scores: pd.DataFrame, n_labels: int, n_outside: int, fig: Path, fig6: bool,
           csvs: dict[str, Path], fig_s: Path | None = None) -> str:
    fc = cfg.section("benchmark")["faerch2023"]
    sc = fc["score"]
    tol = float(cfg.section("benchmark")["repro_tolerance_f1"])
    rel = lambda f: Path(f).resolve().relative_to(cfg.root.resolve()).as_posix()  # noqa: E731
    from iceberg_sar.benchmark.reproduce import scene_paths
    from iceberg_sar.benchmark.sweep import FaerchSweepParams

    sp = FaerchSweepParams.from_config(cfg)
    scene = scene_paths(cfg)
    enl = json.loads(scene.enl.read_text(encoding="utf-8"))["enl"] if scene.enl.is_file() else {}
    tnr = fc["snap"].get("remove_thermal_noise", True)
    lines = ["# Færch 2023 reproduction (Stage 2, Milestone 1)", ""]
    if not ref.get("checked_against_paper"):
        lines += ["> **`reference.yaml` is not yet checked against the paper by hand.** The paper",
                  "> values below are parsed from the XML; treat the comparison as provisional.",
                  ""]
    if source != "faerch":
        lines += ["> **RECONSTRUCTED LABELS** (Path B): not the paper's validation set.", ""]
    lines += [
        "Generated by `python -m iceberg_sar.cli bench-score`. Setup:",
        "",
        f"- Scene: `{scene_paths(cfg).product}`, SNAP 40 m sigma0 (1b), OSM land + "
        f"{fc['land']['buffer_m']} m (1c).",
        f"- Detectors: the paper's code (`cfar-object-detection`, GPL-3.0, run externally), "
        f"{cfg.section('benchmark')['merge']} merge for gamma / log-normal / K, windows "
        "360 / 520 m "
        "(9 / 13 px), iDPolRAD 3 / 57 px cropped to the library's 41 px, objects 2-500 px.",
        *_mine_setup(sp.out_dir, cfg),
        "- Thermal noise removal: " + ("on, as the paper describes." if tnr else
            "**off** (decided 2026-10-26). After SNAP 14's noise subtraction, negatives are "
            "clipped to -115 dB and HV's ENL drops to ~1; every detector degrades. Floored at "
            "the noise level instead, gamma keeps its F (`docs/revision-tnr-ablation.md`). "
            "The run with SNAP's removal is the grey curve and the last table column."),
        f"- ENL {sp.enl:g} for both bands (Table 1). Measured on this stack (median 2 km "
        f"open-water box): HH {enl.get('HH', float('nan')):g}, HV {enl.get('HV', float('nan')):g}.",
        f"- PFA: {len(sp.pfas)} levels, {_e(sp.pfas[0])} .. {_e(sp.pfas[-1])} total; a detector "
        f"stops after the first PFA with > {sp.max_objects} objects in the scene (dotted line).",
        f"- Labels: {'Færch (Path A)' if source == 'faerch' else 'reconstructed (Path B)'}, "
        f"{n_labels - n_outside} of {n_labels} bergs scored: the {n_outside} outside the scoring "
        "area are dropped with the detections there (one shared evaluation mask, revision C3; "
        "the paper counts all 492). Radii 250 m verified / 1789 m interpolated.",
        f"- Scoring area: `{sc['aoi']}` (S1 valid area x the S2 and ALOS-2 outlines georeferenced "
        "from Fig. 1, a few km accuracy). Detections within "
        f"{sc['unknown_ignore_m']} m of an unknown SAR object are ignored.",
        f"- Matching: {MatchSettings.from_config(cfg).describe()}. Verified-only recall: the "
        "share of verified labels matched in the same all-labels matching.",
        "- Numerically unreliable runs (iDPolRAD below ~1e-16, where `1 - pfa` rounds to 1.0) "
        "are flagged and left out of every optimum.",
        "",
        f"## Partial agreement: F at the optimum PFA (tolerance {tol})",
        "",
        "Paper F is derived from Table 3 (TP, FP, FN). My optimum is the PFA with the highest",
        "all-labels F. Preprocessing differs from the paper's description: the main rows are "
        f"{'with' if tnr else 'without'} thermal noise removal; the table after this one has "
        f"the {'without' if tnr else 'with'}-removal rows next to the paper.",
        "",
        "| Detector | Paper PFA | My PFA | Paper TP / FP / FN | Mine TP / FP / FN | Paper F |"
        " My F |"
        " Diff | Within | Verified-only recall (at my PFA) |"
        + (" F, AOI `" + str(sc["aoi_sensitivity"]) + "` |" if "f_sens" in cmp else "")
        + (f" F {'with' if not tnr else 'without'} TNR (PFA) |" if "f_alt" in cmp else ""),
        "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if "f_sens" in cmp else "")
        + ("---|" if "f_alt" in cmp else ""),
    ]
    for r in cmp.itertuples():
        if r.detector in MINE:   # 2a: no paper value, no diff or tolerance
            lines.append(
                f"| {DISPLAY[r.detector]} | – | {_e(r.pfa)} | – | {r.TP} / {r.FP} / {r.FN} | – | "
                f"{r.f:.3f} | – | – | {r.recall_v:.3f} |"
                + (f" {r.f_sens:.3f} |" if "f_sens" in cmp else "")
                + ((f" {r.f_alt:.3f} ({_e(r.pfa_alt)}) |" if np.isfinite(r.f_alt)
                    else " not run |") if "f_alt" in cmp else ""))
            continue
        lines.append(
            f"| {DISPLAY[r.detector]} | {_e(r.paper_pfa)} | {_e(r.pfa)} | "
            f"{r.paper_TP:.0f} / {r.paper_FP:.0f} / {r.paper_FN:.0f} | {r.TP} / {r.FP} / {r.FN} | "
            f"{r.paper_f:.3f} | {r.f:.3f} | {r.diff:+.3f} | {'yes' if r.within else '**no**'} | "
            f"{r.recall_v:.3f} |"
            + (f" {r.f_sens:.3f} |" if "f_sens" in cmp else "")
            + ((f" {r.f_alt:.3f} ({_e(r.pfa_alt)}) |" if np.isfinite(r.f_alt) else " none |")
               if "f_alt" in cmp else ""))
    if "f_alt" in cmp:   # revision T1: the other preprocessing, next to the paper
        other = "with thermal noise removal (as described)" if not tnr else \
            "without thermal noise removal"
        lines += ["", f"Same detectors, {other}:", "",
                  "| Detector | Preprocessing | Paper PFA | My PFA | Paper TP / FP / FN | "
                  "Mine TP / FP / FN | Paper F | My F | Diff |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in cmp.itertuples():
            ours = r.detector in MINE
            ran = np.isfinite(r.f_alt)
            cells = [
                DISPLAY[r.detector], "TNR" if not tnr else "no TNR",
                "–" if ours else _e(r.paper_pfa),
                _e(r.pfa_alt) if ran else "–",
                "–" if ours else f"{r.paper_TP:.0f} / {r.paper_FP:.0f} / {r.paper_FN:.0f}",
                f"{int(r.TP_alt)} / {int(r.FP_alt)} / {int(r.FN_alt)}" if ran
                else "over the object cap",
                "–" if ours else f"{r.paper_f:.3f}",
                f"{r.f_alt:.3f}" if ran else "–",
                f"{r.f_alt - r.paper_f:+.3f}" if ran and not ours else "–"]
            lines.append("| " + " | ".join(cells) + " |")
    fig_rel = Path(os.path.relpath(fig, (cfg.root / sc["report"]).parent)).as_posix()
    lines += ["", f"![F vs PFA]({fig_rel})", ""]
    if not fig6:
        lines += [f"No `{sc['fig6_csv']}`, so the paper's Fig. 6 curves are not overlaid; the star",
                  "is Table 3's optimum.", ""]
    if fig_s is not None:
        s_rel = Path(os.path.relpath(fig_s, (cfg.root / sc["report"]).parent)).as_posix()
        lines += ["All detectors on one panel, all labels, styled like the paper's Fig. 6 for a",
                  "side-by-side read:", "", f"![F vs PFA, Fig. 6 style]({s_rel})", ""]
    lines += ["## Detected share by size at the optimum (Table 5)", "",
              "Table 5 uses its own optimum PFA, which differs from Table 3 for "
              + ", ".join(n.split(":")[0] for n in ref["pfa_mismatch_table3_vs_table5"]) + ".",
              "",
              "| Detector | Size | Paper % (PFA) | Mine at that PFA | Mine at my optimum |",
              "|---|---|---|---|---|"]
    for r in sizes.itertuples():
        if r.detector in MINE:
            lines.append(f"| {DISPLAY[r.detector]} | {r.size} | – | – | {r.pct_at_my_opt:.1f} |")
            continue
        lines.append(f"| {DISPLAY[r.detector]} | {r.size} | {r.paper_pct:.1f} "
                     f"({_e(r.paper_pfa)}) | "
                     + (f"{r.pct_at_paper_pfa:.1f}" if np.isfinite(r.pct_at_paper_pfa)
                        else "over cap") + f" | {r.pct_at_my_opt:.1f} |")
    notes = cfg.root / "benchmarks" / "faerch2023" / "reproduction_notes.md"
    lines += ["", "## Discussion", ""]
    lines += [notes.read_text(encoding="utf-8").strip() if notes.is_file()
              else f"(not written yet: `{rel(notes)}` is included here)"]
    lines += ["", "## Files", ""]
    lines += [f"- per-run scores, {k}: `{rel(v)}` (local)" for k, v in csvs.items()]
    lines += [f"- runtime per detector and PFA: `{rel(sp.out_dir / 'sweep_runtime.csv')}`",
              f"- reference values: `{sc['reference']}`", ""]
    return "\n".join(lines)


def _mine_setup(out_dir: Path, cfg: Config) -> list[str]:
    """Report bullet for my CA-CFAR (2a), from its run sidecar; [] if it hasn't run."""
    f = out_dir / "cacfar" / "cacfar_run.json"
    if not f.is_file():
        return []
    m = json.loads(f.read_text(encoding="utf-8"))
    d = cfg.section("detections")
    return [
        "- My CA-CFAR (2a; `cacfar`, `cacfar-nolength`): Stage 1's detector as it is, on the "
        f"same stack and mask. HV only (one test, so per-channel PFA = total PFA), the paper's "
        f"windows ({2 * m['guard_px'] + 1} / {2 * m['background_px'] + 1} px), ENL "
        f"{m['enl']:g} measured on the HV band (`cfar.enl: auto`), HH co-pol check "
        f"{m['copol_min_contrast_db']:g} dB, objects >= {d['min_area_px']} px merged by grown "
        f"structure, none within {d['edge_buffer_m']} m of the mask. Length rule "
        f"({d['max_structure_m']} m) on / off. The object cap counts its final objects.",
    ]


def run_score(cfg: Config, source: str | None = None, log: Callable[[str], None] = print
              ) -> dict[str, Path]:
    fc = cfg.section("benchmark")["faerch2023"]
    sc = fc["score"]
    source = source or fc["labels_source"]
    ref_path = cfg.root / sc["reference"]
    if not ref_path.is_file():
        raise FileNotFoundError(f"{ref_path} missing; run `bench-reference` and check it")
    ref = load_reference(ref_path)
    labels, unknown = load_labels(cfg, source)
    domain = eval_domain(cfg, sc["aoi"])
    n_out = len(labels) - len(labels_in_domain(labels, domain))
    log(f"{len(labels)} labels ({source}), {n_out} outside the `{sc['aoi']}` area (not "
        f"scored: {len(labels) - n_out} labels); {len(unknown)} unknown objects")
    scores = score_sweep(cfg, labels, unknown, domain, log)
    other = _other_variant(cfg)
    alt = None
    if (sweep_dir(other) / "sweep_runtime.csv").is_file():
        log(f"also scoring the {_variant(other)} sweep")
        alt = score_sweep(other, labels, unknown, domain, log)
    sens = None
    if sc.get("aoi_sensitivity"):
        sens = score_sweep(cfg, labels, unknown, eval_domain(cfg, sc["aoi_sensitivity"]), log)
    base = cfg.root / sc["table_csv"]
    base.parent.mkdir(parents=True, exist_ok=True)
    csvs = {_variant(cfg): base.with_name(f"{base.stem}_{_variant(cfg)}.csv")}
    scores.to_csv(csvs[_variant(cfg)], index=False)
    if alt is not None:
        csvs[_variant(other)] = base.with_name(f"{base.stem}_{_variant(other)}.csv")
        alt.to_csv(csvs[_variant(other)], index=False)

    tol = float(cfg.section("benchmark")["repro_tolerance_f1"])
    cmp = compare_table(scores, ref, tol, sens, alt)
    sizes = size_table(scores, ref)
    fig6_path = cfg.root / sc["fig6_csv"]
    fig6 = pd.read_csv(fig6_path) if fig6_path.is_file() else None
    dets = [d for d in [*fc["sweep"]["detectors"], *MINE] if d in set(scores.detector)]
    note = ("Færch labels (Path A)" if source == "faerch" else "RECONSTRUCTED LABELS (Path B)") + \
        f"; {_variant(cfg)} stack; scoring area `{sc['aoi']}`; dotted line: object cap reached"
    fig = f_figure(scores, ref, fig6, dets, cfg.root / sc["figure"], note, alt,
                   f"{_variant(other)} stack, all labels")
    fig_s = fig6_style_figure(
        scores, dets, cfg.root / sc["figure_fig6_style"],
        f"Sentinel-1 ({'with' if _variant(cfg) == 'TNR' else 'no'} thermal noise removal)",
        note.split("; dotted")[0] + "; all labels; over-cap PFAs left out")
    rep = cfg.root / sc["report"]
    rep.write_text(report(cfg, source, ref, cmp, sizes, scores, len(labels),
                          n_out, fig, fig6 is not None, csvs, fig_s),
                   encoding="utf-8")
    for r in cmp.itertuples():
        log(f"{r.detector:9s} F {r.f:.3f} at {_e(r.pfa)}" + ("" if r.detector in MINE else
            f" vs paper {r.paper_f:.3f} at {_e(r.paper_pfa)} ({r.diff:+.3f})"))
    return {**{f"scores {k}": v for k, v in csvs.items()}, "figure": fig,
            "figure (Fig. 6 style)": fig_s, "report": rep}


def _variant(cfg: Config) -> str:
    snap = cfg.section("benchmark")["faerch2023"]["snap"]
    if snap.get("tnr_variant"):
        return f"TNR{snap['tnr_variant']}"
    return "TNR" if snap.get("remove_thermal_noise", True) else "noTNR"


def _other_variant(cfg: Config) -> Config:
    """A copy of `cfg` for the other stack variant (thermal noise removal on <-> off)."""
    import copy

    other = Config(root=cfg.root, raw=copy.deepcopy(cfg.raw))
    snap = other.section("benchmark")["faerch2023"]["snap"]
    snap["remove_thermal_noise"] = not snap.get("remove_thermal_noise", True)
    return other

