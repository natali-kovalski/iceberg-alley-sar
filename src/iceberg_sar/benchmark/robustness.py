"""Revision phase 3: how much the headline numbers depend on matching, chance and sampling.

Re-scores detections already on disk, no detector runs:
- N1 matching sensitivity (C4, Færch EW scene): best F and rank per detector under greedy vs
  optimal matching, true vs EPSG:5938 metre radii, and tighter interpolated-label radii.
- N2 chance matches (C9): every detection set at its operating point shifted by `shift_m` in
  random directions, re-matched; the hits that survive are what the radius gives for free.
- C10 bootstrap intervals: recall (and F on EW) at each detector's operating point, labels
  resampled with replacement (IW: within each frame; also whole frames resampled), paired with
  the reference detector so differences get an interval too. Detections are held fixed, so EW
  false positives don't resample: the F interval is a label-sampling interval only. The same
  berg can appear in several IW passes; the frame bootstrap is the conservative read.

Everything shares the evaluation domain (revision C3) of the scorer it mirrors.
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
from scipy.spatial import cKDTree

from iceberg_sar.benchmark import score as sc
from iceberg_sar.config import Config
from iceberg_sar.thor import PIPELINE_SHORT


@dataclass(frozen=True)
class RobustnessParams:
    out_dir: Path
    report: Path
    n_boot: int
    seed: int
    shift_m: float
    n_shifts: int
    matching_variants: list[dict[str, Any]]
    ew_reference: str
    iw_reference: str
    thor_ew_series: tuple[str, str]   # (stack, series) of the THOR run scored on EW

    @classmethod
    def from_config(cls, cfg: Config) -> RobustnessParams:
        r = cfg.section("benchmark")["robustness"]
        return cls(out_dir=cfg.root / r["out_dir"], report=cfg.root / r["report"],
                   n_boot=int(r["n_boot"]), seed=int(r["seed"]), shift_m=float(r["shift_m"]),
                   n_shifts=int(r["n_shifts"]), matching_variants=list(r["matching_variants"]),
                   ew_reference=str(r["ew_reference"]), iw_reference=str(r["iw_reference"]),
                   thor_ew_series=(r["thor_ew"]["stack"], r["thor_ew"]["series"]))


# --- shared helpers -----------------------------------------------------------------------------

def shift_points(xy: np.ndarray, dist_m: float, rng: np.random.Generator) -> np.ndarray:
    """All points moved by one vector of length `dist_m` in a random direction."""
    a = rng.uniform(0, 2 * np.pi)
    return xy + dist_m * np.array([np.cos(a), np.sin(a)])


def boot_index(groups: np.ndarray, n_boot: int, rng: np.random.Generator,
               by: str = "label") -> np.ndarray:
    """(n_boot, n) indices: labels resampled within each group (`label`), or whole groups
    resampled and their labels taken as they are (`group`; n varies, padded with -1)."""
    n = len(groups)
    if by == "label":
        out = np.empty((n_boot, n), int)
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            out[:, idx] = rng.choice(idx, size=(n_boot, len(idx)))
        return out
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    rows = [np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
            for _ in range(n_boot)]
    width = max(len(r) for r in rows)
    return np.array([np.pad(r, (0, width - len(r)), constant_values=-1) for r in rows])


def boot_recall(hit: np.ndarray, idx: np.ndarray) -> np.ndarray:
    ok = idx >= 0
    h = np.where(ok, hit[np.clip(idx, 0, None)], 0)
    return h.sum(1) / ok.sum(1)


def ci(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# --- Færch EW scene ------------------------------------------------------------------------------

@dataclass
class EwRun:
    name: str
    setting: float
    det_xy: np.ndarray
    # THOR only: every candidate peak in the domain (unknowns ignored) with its head score, and
    # the thresholds of its sweep, so N1 can re-optimise the threshold as it does the CFAR PFA.
    cand_xy: np.ndarray | None = None
    cand_score: np.ndarray | None = None
    thresholds: tuple[float, ...] = ()


def best_threshold(run: EwRun, labels: pd.DataFrame, matcher: str
                   ) -> tuple[float, dict[str, Any]]:
    """Threshold with the highest F under these labels / matcher (ties: the lower threshold,
    as `bench-thor` picks it), and its scores."""
    if run.cand_xy is None or run.cand_score is None or not run.thresholds:
        raise ValueError(f"{run.name}: no candidate peaks to sweep")
    best: tuple[float, dict[str, Any]] | None = None
    for t in sorted(run.thresholds):
        r = sc.score_points(run.cand_xy[run.cand_score >= t], labels, matcher)
        if best is None or r["f"] > best[1]["f"]:
            best = (t, r)
    assert best is not None
    return best


def ew_runs(cfg: Config, labels: pd.DataFrame, unknown: np.ndarray, domain: Any,
            p: RobustnessParams) -> list[EwRun]:
    """Every detector's detections at its best-F setting (default matching), in the domain."""
    from iceberg_sar.benchmark.sweep import sweep_dir
    from iceberg_sar.benchmark.thor_faerch import ThorRunParams

    fc = cfg.section("benchmark")["faerch2023"]
    base = cfg.root / fc["score"]["table_csv"]
    scores = pd.read_csv(base.with_name(f"{base.stem}_{sc._variant(cfg)}.csv"))
    runtime = pd.read_csv(sweep_dir(cfg) / "sweep_runtime.csv")
    ignore = float(fc["score"]["unknown_ignore_m"])
    out = []
    for det, b in sc.optimum(scores).iterrows():
        r = runtime[(runtime.detector == det) & sc.same_pfa(runtime.pfa, b.pfa)].iloc[0]
        xy, _ = sc.detections_xy(sweep_dir(cfg) / det / str(r.geojson), fc["crs"], domain,
                                 unknown, ignore)
        out.append(EwRun(det, float(b.pfa), xy))
    rp = ThorRunParams.from_config(cfg)
    stack, series = p.thor_ew_series
    t = pd.read_csv(rp.out_dir / "thor_scores.csv")
    t = t[(t["stack"] == stack) & (t.series == series)]
    run = next(r for r in rp.runs if (r.stack, r.series) == (stack, series))
    if len(t):
        thr = float(t.loc[t.f.idxmax(), "threshold"])
        g = gpd.read_file(rp.out_dir / f"{run.tag}.geojson").to_crs(fc["crs"])
        g = g[g.score >= float(t.threshold.min())]
        xy = np.column_stack([g.geometry.x, g.geometry.y]).reshape(-1, 2)
        s = g.score.to_numpy(float)
        keep = domain.keep(xy)
        xy, s = xy[keep], s[keep]
        if len(unknown) and len(xy):
            far = cKDTree(unknown).query(xy, distance_upper_bound=ignore)[0] > ignore
            xy, s = xy[far], s[far]
        out.append(EwRun("thor-pipeline", thr, xy[s >= thr], cand_xy=xy, cand_score=s,
                         thresholds=tuple(sorted(float(v) for v in t.threshold.unique()))))
    return out


def ew_hits(run: EwRun, labels: pd.DataFrame, matcher: str) -> np.ndarray:
    _, li = sc.match(run.det_xy, labels[["x", "y"]].to_numpy(float),
                     labels.radius_m.to_numpy(float), matcher)
    hit = np.zeros(len(labels), bool)
    hit[li] = True
    return hit


def matching_sensitivity(cfg: Config, labels: pd.DataFrame, unknown: np.ndarray, domain: Any,
                         runs: list[EwRun], p: RobustnessParams,
                         log: Callable[[str], None]) -> pd.DataFrame:
    """N1: best F per detector (PFA re-optimised) under each matching variant, and its rank."""
    rows = []
    thor = next((r for r in runs if r.name == "thor-pipeline"), None)
    for v in p.matching_variants:
        ms = sc.MatchSettings(**v)
        ms.check()
        s = sc.score_sweep(cfg, labels, unknown, domain, lambda _m: None, ms)
        best = sc.optimum(s)
        for det, b in best.iterrows():
            rows.append({"variant": ms.describe(), **v, "detector": det, "pfa": b.pfa,
                         "TP": int(b.TP), "FP": int(b.FP), "f": b.f, "recall_v": b.recall_v})
        if thor is not None:   # THOR threshold re-optimised, like the CFAR PFA above
            lab = sc.apply_match_settings(sc.labels_in_domain(labels, domain), ms)
            thr, r = best_threshold(thor, lab, ms.matcher)
            rows.append({"variant": ms.describe(), **v, "detector": "thor-pipeline",
                         "pfa": thr, "TP": r["TP"], "FP": r["FP"], "f": r["f"],
                         "recall_v": r["recall_v"]})
        log(f"N1: {ms.describe()} done")
    t = pd.DataFrame(rows)
    t["rank"] = t.groupby("variant").f.rank(ascending=False, method="min").astype(int)
    return t


def ew_chance(runs: list[EwRun], labels: pd.DataFrame, p: RobustnessParams,
              rng: np.random.Generator) -> pd.DataFrame:
    """N2 on EW: hits after random shifts, all labels and by label status."""
    ver = (labels.status == "verified").to_numpy()
    rows = []
    for r in runs:
        base = ew_hits(r, labels, "greedy")
        tp = []
        for _ in range(p.n_shifts):
            moved = EwRun(r.name, r.setting, shift_points(r.det_xy, p.shift_m, rng))
            h = ew_hits(moved, labels, "greedy")
            tp.append((h.sum(), h[ver].sum(), h[~ver].sum()))
        a = np.array(tp)
        rows.append({"detector": r.name, "setting": r.setting, "detections": len(r.det_xy),
                     "TP": int(base.sum()), "TP_verified": int(base[ver].sum()),
                     "TP_interpolated": int(base[~ver].sum()),
                     "chance_TP_mean": a[:, 0].mean(), "chance_TP_p95": np.percentile(a[:, 0], 95),
                     "chance_verified_mean": a[:, 1].mean(),
                     "chance_interpolated_mean": a[:, 2].mean(),
                     "chance_share_of_TP": a[:, 0].mean() / max(base.sum(), 1)})
    return pd.DataFrame(rows)


def ew_bootstrap(runs: list[EwRun], labels: pd.DataFrame, p: RobustnessParams,
                 rng: np.random.Generator) -> pd.DataFrame:
    """C10 on EW: recall and F intervals, labels resampled; paired difference vs the reference."""
    idx = boot_index(np.zeros(len(labels), int), p.n_boot, rng)
    n = len(labels)
    res = {}
    for r in runs:
        hit = ew_hits(r, labels, "greedy")
        tp_b = hit[idx].sum(1)
        fp = len(r.det_xy) - int(hit.sum())
        f_b = 2 * tp_b / (2 * tp_b + fp + (n - tp_b))
        res[r.name] = (hit, tp_b / n, f_b, fp)
    ref = res.get(p.ew_reference)
    rows = []
    for name, (hit, r_b, f_b, fp) in res.items():
        tp = int(hit.sum())
        row = {"detector": name, "labels": n, "TP": tp, "FP": fp, "recall": tp / n,
               "f": 2 * tp / (2 * tp + fp + n - tp)}
        row["recall_lo"], row["recall_hi"] = ci(r_b)
        row["f_lo"], row["f_hi"] = ci(f_b)
        if ref is not None and name != p.ew_reference:
            row["df_vs_ref_lo"], row["df_vs_ref_hi"] = ci(f_b - ref[2])
            row["df_vs_ref"] = row["f"] - (2 * ref[0].sum() / (2 * ref[0].sum() + ref[3] + n
                                                              - ref[0].sum()))
        rows.append(row)
    return pd.DataFrame(rows)


# --- IW frames (5d at 10 m, 2b at 40 m) ----------------------------------------------------------

@dataclass
class IwOp:
    name: str
    setting: str
    detections: int
    hit: np.ndarray          # per label, labels of every frame concatenated
    det_xy: list[np.ndarray]  # per frame


def iw_ops(cfg: Config, log: Callable[[str], None]) -> tuple[list[IwOp], np.ndarray, list]:
    """Operating points with per-label hits on 5d's same-pass frames: 5d's CFAR at `cfar.pfa`
    and THOR at the same count (10 m), and every 2b 40 m detector at the pooled setting with
    the most detections at or below the 2-per-label budget (revision C6). Returns the ops, the
    frame of each label, and the per-frame truth xy."""
    from iceberg_sar.benchmark.iwcompare import (
        IwCompareParams,
        at_budget,
        load_detections,
        pool_sweep,
        scene_truth,
        sweep_table,
    )
    from iceberg_sar.thor.compare import (
        CompareParams,
        load_scene,
        operating_points,
        score_points,
        xy,
    )

    radius = float(cfg.section("validation")["max_dist_m"])
    cp = CompareParams.from_config(cfg)
    frames, truths, ops10 = [], [], {}
    for name, kind in cp.scenes.items():
        if kind != "iip-satellite":
            continue
        s = load_scene(cfg, name, kind, cp)
        frames.append(np.full(len(s.truth_xy), len(truths)))
        truths.append(s.truth_xy)
        for op, label, det in operating_points(s, cp):
            if op.endswith("view"):
                continue
            d_xy = xy(det, s.crs)
            _, hit = score_points(d_xy, s.truth_xy, radius)
            o = ops10.setdefault(op, {"label": label, "n": 0, "hit": [], "xy": []})
            o["n"] += len(det)
            o["hit"].append(hit)
            o["xy"].append(d_xy)
        log(f"IW 10 m: {name} ({len(s.truth_xy)} labels)")
    names10 = {"cfar": "cacfar-10m-5d", "thor-linear matched-count": "thor-pipeline-linear-10m",
               "thor-centernet matched-count": "thor-pipeline-centernet-10m"}
    ops = [IwOp(names10.get(k, k), v["label"], v["n"], np.concatenate(v["hit"]), v["xy"])
           for k, v in ops10.items()]

    p = IwCompareParams.from_config(cfg)
    sp = p.sweep
    order = [n for n in cp.scenes if cp.scenes[n] == "iip-satellite"]
    tags = [*sp.detectors, *sp.cacfar_tags, *sp.cacfar_10m_tags]
    per_scene, dets = [], {}
    for scene in order:
        _, t_xy, crs = scene_truth(cfg, scene, p.domain)
        work = sp.out_dir / scene
        dets[scene] = ({t: load_detections(work / t, t, p.domain) for t in tags}, t_xy, crs)
        per_scene.append(sweep_table(dets[scene][0], t_xy, crs, p.radius_m).assign(scene=scene))
        log(f"IW 40 m: {scene}")
    pooled = pool_sweep(pd.concat(per_scene, ignore_index=True), len(order))
    budget = int(round(p.per_label * sum(len(t) for t in truths)))
    for tag in tags:
        g = pooled[pooled.detector == tag]
        _, n, flag = at_budget(g, budget, p.budget_min_fraction)
        if flag == "none" or not np.isfinite(n):
            continue
        setting = float(g.loc[g.detections == n, "setting"].iloc[0])
        hits, xys = [], []
        for scene in order:
            d = dets[scene][0][tag][setting]
            d_xy = xy(d, dets[scene][2])
            hits.append(score_points(d_xy, dets[scene][1], p.radius_m)[1])
            xys.append(d_xy)
        ops.append(IwOp(tag, f"PFA {setting:.0e}" + (" (short)" if flag else ""), int(n),
                        np.concatenate(hits), xys))
    return ops, np.concatenate(frames), truths


def iw_bootstrap(ops: list[IwOp], frames: np.ndarray, p: RobustnessParams,
                 rng: np.random.Generator) -> pd.DataFrame:
    ref = next((o for o in ops if o.name == p.iw_reference), None)
    rows = []
    for by in ("label", "group"):
        idx = boot_index(frames, p.n_boot, rng, by)
        r_ref = boot_recall(ref.hit, idx) if ref is not None else None
        for o in ops:
            rb = boot_recall(o.hit, idx)
            row = {"resample": "labels within frames" if by == "label" else "frames",
                   "detector": o.name, "setting": o.setting, "detections": o.detections,
                   "labels": len(o.hit), "recall": o.hit.mean()}
            row["recall_lo"], row["recall_hi"] = ci(rb)
            if r_ref is not None and o is not ref:
                row["dr_vs_ref"] = o.hit.mean() - ref.hit.mean()
                row["dr_vs_ref_lo"], row["dr_vs_ref_hi"] = ci(rb - r_ref)
            rows.append(row)
    return pd.DataFrame(rows)


def iw_chance(ops: list[IwOp], truths: list[np.ndarray], radius_m: float, p: RobustnessParams,
              rng: np.random.Generator) -> pd.DataFrame:
    from iceberg_sar.matching import match_points

    rows = []
    for o in ops:
        tp = []
        for _ in range(p.n_shifts):
            tp.append(sum(match_points(shift_points(d, p.shift_m, rng), t, radius_m)
                          .true_positives for d, t in zip(o.det_xy, truths, strict=True)))
        rows.append({"detector": o.name, "setting": o.setting, "detections": o.detections,
                     "TP": int(o.hit.sum()), "chance_TP_mean": float(np.mean(tp)),
                     "chance_TP_p95": float(np.percentile(tp, 95))})
    return pd.DataFrame(rows)


# --- report --------------------------------------------------------------------------------------

def _f(v: float, d: int = 3) -> str:
    return "" if pd.isna(v) else f"{v:.{d}f}"


def _name(n: str) -> str:
    from iceberg_sar.benchmark.iwcompare import LABELS

    if n.startswith("thor-pipeline"):
        rest = n.removeprefix("thor-pipeline").strip("-").replace("-", ", ")
        return PIPELINE_SHORT + (f" ({rest})" if rest else "")
    return sc.DISPLAY.get(n) or LABELS.get(n, n)


def report(p: RobustnessParams, n1: pd.DataFrame, ew_c: pd.DataFrame, ew_b: pd.DataFrame,
           iw_b: pd.DataFrame, iw_c: pd.DataFrame, meta: dict[str, Any]) -> str:
    L = ["# Robustness of the benchmark numbers (revision phase 3)", "",
         "Generated by `python -m iceberg_sar.cli bench-robustness`; re-scores detections on "
         "disk. Intervals are 95 % percentile bootstrap intervals "
         f"({p.n_boot} resamples, seed {p.seed}). Chance matches: detections shifted by "
         f"{p.shift_m / 1000:g} km in {p.n_shifts} random directions.", "",
         "## N1: matching sensitivity (Færch EW scene, noTNR)", "",
         f"{meta['ew_labels']} labels in the scoring area. Best F per detector with its "
         f"setting re-optimised under each matching variant (PFA for the CFARs, head score "
         f"threshold for the {PIPELINE_SHORT}).", ""]
    piv = n1.pivot_table(index="detector", columns="variant", values="f")
    rk = n1.pivot_table(index="detector", columns="variant", values="rank")
    cols = list(dict.fromkeys(n1.variant))
    L += ["| Detector | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for det in piv.sort_values(cols[0], ascending=False).index:
        L.append(f"| {_name(det)} | " + " | ".join(
            f"{piv.loc[det, c]:.3f} (#{int(rk.loc[det, c])})" for c in cols) + " |")
    L += ["", "## N2: chance matches", "",
          "Hits that survive a random shift of every detection: what the search radius gives "
          "without any real alignment.", "",
          "### Færch EW scene (250 m verified / 1789 m interpolated)", "",
          "| Detector | Detections | TP (verified / interpolated) | Chance TP, mean (p95) | "
          "Chance verified / interpolated | Chance share of TP |", "|---|---|---|---|---|---|"]
    for r in ew_c.sort_values("TP", ascending=False).itertuples():
        L.append(f"| {_name(r.detector)} | {r.detections} | {r.TP} ({r.TP_verified} / "
                 f"{r.TP_interpolated}) | {r.chance_TP_mean:.1f} ({r.chance_TP_p95:.0f}) | "
                 f"{r.chance_verified_mean:.1f} / {r.chance_interpolated_mean:.1f} | "
                 f"{100 * r.chance_share_of_TP:.0f} % |")
    L += ["", f"### IW frames ({meta['iw_labels']} labels, {meta['iw_radius']:.0f} m)", "",
          "| Detector | Setting | Detections | TP | Chance TP, mean (p95) |",
          "|---|---|---|---|---|"]
    for r in iw_c.itertuples():
        L.append(f"| {_name(r.detector)} | {r.setting} | {r.detections} | {r.TP} | "
                 f"{r.chance_TP_mean:.1f} ({r.chance_TP_p95:.0f}) |")
    L += ["", "## C10: bootstrap intervals", "",
          f"### Færch EW scene, each detector at its best-F PFA; paired vs {_name(p.ew_reference)}",
          "", "Labels resampled; detections and so false positives held fixed.", "",
          "| Detector | F [95 % CI] | Recall [95 % CI] | ΔF vs reference [95 % CI] |",
          "|---|---|---|---|"]
    for r in ew_b.sort_values("f", ascending=False).itertuples():
        d = ("" if pd.isna(getattr(r, "df_vs_ref", np.nan)) else
             f"{r.df_vs_ref:+.3f} [{r.df_vs_ref_lo:+.3f}, {r.df_vs_ref_hi:+.3f}]")
        L.append(f"| {_name(r.detector)} | {r.f:.3f} [{r.f_lo:.3f}, {r.f_hi:.3f}] | "
                 f"{r.recall:.3f} [{r.recall_lo:.3f}, {r.recall_hi:.3f}] | {d} |")
    for by, g in iw_b.groupby("resample", sort=False):
        L += ["", f"### IW frames, recall at the operating points; resampling {by}; paired vs "
              f"{_name(p.iw_reference)}", "",
              "| Detector | Setting | Detections | Recall [95 % CI] | ΔR vs reference [95 % CI] |",
              "|---|---|---|---|---|"]
        for r in g.sort_values("recall", ascending=False).itertuples():
            d = ("" if pd.isna(getattr(r, "dr_vs_ref", np.nan)) else
                 f"{r.dr_vs_ref:+.3f} [{r.dr_vs_ref_lo:+.3f}, {r.dr_vs_ref_hi:+.3f}]")
            L.append(f"| {_name(r.detector)} | {r.setting} | {r.detections} | {r.recall:.3f} "
                     f"[{r.recall_lo:.3f}, {r.recall_hi:.3f}] | {d} |")
    L += ["", "The 10 m CA-CFAR and the THOR pipeline share one detection count (975), so their "
          "difference is like for like. The 40 m detectors sit at their real run with the most "
          "detections at or below the 2-per-label budget (revision C6), which is a different "
          "count for each: part of their gap to the reference is the budget they reach, so read "
          "those rows with the Detections column and the full curves.", "",
          "Frames: only four, so the frame bootstrap has 35 distinct resamples; read it as "
          "the spread between frames, not a precise interval. The same berg can appear in "
          "several passes, so labels within frames understates the uncertainty.", ""]
    return "\n".join(L)


def run_robustness(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Path]:
    p = RobustnessParams.from_config(cfg)
    p.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(p.seed)
    fc = cfg.section("benchmark")["faerch2023"]
    labels, unknown = sc.load_labels(cfg, fc["labels_source"])
    domain = sc.eval_domain(cfg, fc["score"]["aoi"])
    lab = sc.apply_match_settings(sc.labels_in_domain(labels, domain), sc.MatchSettings())
    runs = ew_runs(cfg, labels, unknown, domain, p)
    log(f"EW: {len(lab)} labels, {len(runs)} operating points")
    n1 = matching_sensitivity(cfg, labels, unknown, domain, runs, p, log)
    ew_c = ew_chance(runs, lab, p, rng)
    ew_b = ew_bootstrap(runs, lab, p, rng)
    ops, frames, truths = iw_ops(cfg, log)
    radius = float(cfg.section("validation")["max_dist_m"])
    iw_b = iw_bootstrap(ops, frames, p, rng)
    iw_c = iw_chance(ops, truths, radius, p, rng)
    out = {}
    for name, t in [("matching_sensitivity", n1), ("chance_ew", ew_c), ("bootstrap_ew", ew_b),
                    ("bootstrap_iw", iw_b), ("chance_iw", iw_c)]:
        out[name] = p.out_dir / f"{name}.csv"
        t.to_csv(out[name], index=False)
    meta = {"ew_labels": len(lab), "iw_labels": len(frames), "iw_radius": radius}
    (p.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    p.report.write_text(report(p, n1, ew_c, ew_b, iw_b, iw_c, meta), encoding="utf-8")
    out["report"] = p.report
    return out
