"""Milestone 5d: THOR vs. CA-CFAR on the held-out April 2019 scenes.

Precision can't be measured without AIS (an unlabelled detection may be a real berg, a ship or
clutter), so the detectors are compared by **recall at a matched number of detections**: the
CFAR is swept over its PFA, THOR over its score threshold, and each operating point gives
(detections per scene, labels matched). Both see the same things:
- detections: the CFAR's masked HV raster (THOR peaks already pass it in `infer`), then the
  open-water filter of `pointval.run_point_validation` (nothing within `viewer.near_ice_km`);
- truth: `pointval.load_truth`, kept where the SAR is valid (`sar_valid_at`) after the drift
  shift;
- matching: `matching.match_points`, one-to-one, at `validation.max_dist_m`.

Drift: same-pass satellite labels need none. Aircraft truth (hours later) gets one offset per
scene, estimated once from the pooled detections of both detectors at their reference operating
points and applied to every operating point of both, so neither detector gets its own fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from iceberg_sar.config import Config
from iceberg_sar.matching import estimate_offset, match_points
from iceberg_sar.pointval import (
    PointValParams,
    load_truth,
    point_val_params,
    sar_valid_at,
)

CFAR = "cfar"


@dataclass(frozen=True)
class CompareParams:
    out_dir: Path
    scenes: dict[str, str]          # product -> truth kind
    cfar_pfas: list[float]
    cfar_pfa: float                 # the CFAR's configured operating point
    thor_models: list[str]
    view_score: float
    radii_m: list[float]
    ice_bins_km: list[float]
    incidence_bins_deg: list[float]
    near_ice_km: float

    @classmethod
    def from_config(cls, cfg: Config) -> CompareParams:
        c = cfg.section("thor")["compare"]
        return cls(out_dir=cfg.root / c["out_dir"], scenes=dict(c["scenes"]),
                   cfar_pfas=[float(p) for p in c["cfar_pfas"]],
                   cfar_pfa=float(cfg.section("cfar")["pfa"]),
                   thor_models=list(c["thor_models"]), view_score=float(c["view_score"]),
                   radii_m=[float(r) for r in c["radii_m"]],
                   ice_bins_km=[float(b) for b in c["ice_bins_km"]],
                   incidence_bins_deg=[float(b) for b in c["incidence_bins_deg"]],
                   near_ice_km=float(cfg.raw.get("viewer", {}).get("near_ice_km", 5)))


def pfa_label(pfa: float) -> str:
    return f"pfa_{pfa:.0e}"


def cfar_sweep(cfg: Config, product_dir: Path, p: CompareParams, log=print) -> list[Path]:
    """CFAR detections at every PFA of the sweep (one pass), in <out_dir>/cfar_sweep/pfa_<x>/."""
    from iceberg_sar.detections import detect_product_sweep

    out = detect_product_sweep(Path(product_dir), cfg, p.cfar_pfas,
                               lambda pfa: p.out_dir / "cfar_sweep" / pfa_label(pfa), band="HV")
    for pfa, f in out.items():
        n = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))["n_detections"]
        log(f"{Path(product_dir).name}  pfa={pfa:g}  detections={n}")
    return list(out.values())


def open_water(det: gpd.GeoDataFrame, near_ice_km: float) -> gpd.GeoDataFrame:
    """The `validate-points` filter: drop detections within `near_ice_km` of pack ice."""
    if det.empty:
        return det
    return det[~(det["distance_to_ice_km"] < near_ice_km)]


def xy(gdf: gpd.GeoDataFrame, crs: object) -> np.ndarray:
    g = gdf.to_crs(crs).geometry
    return np.column_stack([g.x.to_numpy(), g.y.to_numpy()]) if len(g) else np.empty((0, 2))


@dataclass(frozen=True)
class Drift:
    dx_m: float
    dy_m: float
    n_matched: int
    chance_n_matched: float
    applied: bool


def shared_drift(
    det_xy: list[np.ndarray], truth_xy: np.ndarray, gap_s: float, vp: PointValParams,
) -> Drift:
    """One offset (truth ~ det + shift) from the pooled detections, with pointval's rules:
    none for same-pass truth, and only applied if the peak beats chance."""
    if gap_s <= vp.simultaneous_s:
        return Drift(0.0, 0.0, 0, 0.0, False)
    pooled = np.vstack([d for d in det_xy if len(d)]) if any(len(d) for d in det_xy) \
        else np.empty((0, 2))
    off = estimate_offset(pooled, truth_xy, vp.offset_search_m, vp.offset_step_m, vp.max_dist_m)
    ok = off.n_matched >= max(vp.min_offset_matches, 3 * off.chance_n_matched)
    return Drift(off.dx_m, off.dy_m, off.n_matched, off.chance_n_matched, bool(ok))


def score_points(
    det_xy: np.ndarray, truth_xy: np.ndarray, radius_m: float,
) -> tuple[int, np.ndarray]:
    """Labels matched, and a per-label matched flag (truth already filtered and shifted)."""
    m = match_points(det_xy, truth_xy, radius_m)
    hit = np.zeros(len(truth_xy), bool)
    hit[m.pairs[:, 1]] = True
    return m.true_positives, hit


def thor_thresholds(scores: np.ndarray, n_max: int = 60) -> np.ndarray:
    """Score thresholds spaced evenly in log(detection count), so the curve is dense where the
    CFAR points are (tens to thousands of detections)."""
    s = np.sort(scores)[::-1]
    if not len(s):
        return np.empty(0)
    counts = np.unique(np.geomspace(1, len(s), n_max).round().astype(int))
    return s[counts - 1]


def threshold_for_count(scores: np.ndarray, n: int) -> float:
    """The THOR threshold that keeps the `n` best-scoring detections (ties may add a few)."""
    s = np.sort(scores)[::-1]
    return float(s[min(max(n, 1), len(s)) - 1]) if len(s) else np.inf


@dataclass
class SceneData:
    name: str
    truth_kind: str
    crs: object
    truth: gpd.GeoDataFrame        # compared labels only, with attributes
    truth_xy: np.ndarray           # in SAR time (drift undone)
    drift: Drift
    cfar: dict[float, gpd.GeoDataFrame]   # pfa -> open-water detections
    thor: dict[str, gpd.GeoDataFrame]     # model tag -> open-water detections (all scores)


def load_scene(cfg: Config, name: str, truth_kind: str, p: CompareParams) -> SceneData:
    """Truth and every detector's open-water detections for one scene, plus the shared drift."""
    from iceberg_sar.detections import sample_incidence_deg, scene_timestamp
    from iceberg_sar.preprocess import find_rtc_bands
    from iceberg_sar.seaice import distance_to_ice_km

    vp = point_val_params(cfg)
    interim = cfg.path("interim") / name
    masked = interim / f"{name}_HV_masked.tif"
    with rasterio.open(masked) as src:
        crs = src.crs
    sar_time = pd.Timestamp(scene_timestamp(name)).tz_localize(None)
    truth, _ = load_truth(cfg, name, sar_time, truth_kind)

    cfar = {}
    for pfa in p.cfar_pfas:
        f = p.out_dir / "cfar_sweep" / pfa_label(pfa) / f"{name}_HV_detections.geojson"
        if not f.is_file():
            raise FileNotFoundError(f"{f}; run `thor-compare --sweep` first")
        cfar[pfa] = open_water(gpd.read_file(f), p.near_ice_km)
    det_dir = cfg.path("outputs") / "detections"
    thor = {}
    for tag in p.thor_models:
        f = det_dir / f"{name}_{tag}_detections.geojson"
        if not f.is_file():
            raise FileNotFoundError(f"{f}; run `thor-detect` first")
        thor[tag] = open_water(gpd.read_file(f), p.near_ice_km)

    t_xy = xy(truth, crs)
    gap_s = float(np.median(np.abs((truth["time"] - sar_time).dt.total_seconds()))) \
        if len(truth) else 0.0
    ref = [xy(cfar[p.cfar_pfa], crs)] + [xy(d[d["score"] >= p.view_score], crs)
                                         for d in thor.values()]
    drift = shared_drift(ref, t_xy, gap_s, vp)
    shift = np.array([drift.dx_m, drift.dy_m]) if drift.applied else np.zeros(2)
    t_xy = t_xy - shift
    ok = sar_valid_at(masked, t_xy, vp.valid_block_px)
    truth, t_xy = truth[ok].reset_index(drop=True), t_xy[ok]

    inc = find_rtc_bands(cfg.path("raw") / name).get("inc")
    truth["incidence_deg"] = sample_incidence_deg(inc, t_xy[:, 0], t_xy[:, 1]) if inc else np.nan
    ice = interim / f"{name}_seaice_mask.tif"
    truth["distance_to_ice_km"] = (distance_to_ice_km(ice, t_xy[:, 0], t_xy[:, 1])
                                   if ice.is_file() else np.nan)
    return SceneData(name, truth_kind, crs, truth, t_xy, drift, cfar, thor)


def sweep_table(s: SceneData, radius_m: float) -> pd.DataFrame:
    """One row per operating point: detector, setting, detections, matched, recall."""
    rows = []
    n_truth = len(s.truth_xy)
    for pfa, det in s.cfar.items():
        tp, _ = score_points(xy(det, s.crs), s.truth_xy, radius_m)
        rows.append({"detector": CFAR, "setting": pfa, "detections": len(det), "matched": tp})
    for tag, det in s.thor.items():
        d_xy, scores = xy(det, s.crs), det["score"].to_numpy()
        for t in thor_thresholds(scores):
            keep = scores >= t
            tp, _ = score_points(d_xy[keep], s.truth_xy, radius_m)
            rows.append({"detector": tag, "setting": float(t), "detections": int(keep.sum()),
                         "matched": tp})
    out = pd.DataFrame(rows)
    out.insert(0, "scene", s.name)
    out["truth"] = n_truth
    out["recall"] = (out["matched"] / n_truth).round(3) if n_truth else np.nan
    out["radius_m"] = radius_m
    return out


def operating_points(s: SceneData, p: CompareParams) -> dict[str, gpd.GeoDataFrame]:
    """Detections at matched counts: the CFAR at its configured PFA, and each THOR model at
    the threshold giving the same number of detections. Plus THOR at `view_score`."""
    base = s.cfar[p.cfar_pfa]
    out = {f"{CFAR} (pfa {p.cfar_pfa:g})": base}
    for tag, det in s.thor.items():
        t = threshold_for_count(det["score"].to_numpy(), len(base))
        out[f"{tag} (n-matched, score >= {t:.2f})"] = det[det["score"] >= t]
    for tag, det in s.thor.items():
        out[f"{tag} (score >= {p.view_score:g})"] = det[det["score"] >= p.view_score]
    return out


SIZE_GROUPS = {"SM": "small 15-60 m", "MED": "medium 61-120 m", "LG": "large >120 m",
               "VLG": "large >120 m"}


def _bin_labels(edges: list[float], unit: str) -> list[str]:
    """'0-5 km', '5-20 km', '>20 km' (a last edge >= 1000 means open-ended)."""
    labels = [f"{a:g}-{b:g}{unit}" for a, b in zip(edges[:-1], edges[1:], strict=False)]
    if edges[-1] >= 1000:
        labels[-1] = f">{edges[-2]:g}{unit}"
    return labels


def label_bins(truth: gpd.GeoDataFrame, p: CompareParams) -> pd.DataFrame:
    """Per label: size group, distance-to-ice bin and incidence bin."""
    size = truth["size"].map(SIZE_GROUPS).fillna("other / unknown")
    dist = truth["distance_to_ice_km"]
    ice = pd.cut(dist, p.ice_bins_km, right=False, labels=_bin_labels(p.ice_bins_km, " km"))
    ice = ice.astype(str).where(dist.notna(), "no ice in scene")
    inc = pd.cut(truth["incidence_deg"], p.incidence_bins_deg,
                 labels=_bin_labels(p.incidence_bins_deg, "°")).astype(str)
    return pd.DataFrame({"size": size, "ice_km": ice, "incidence_deg": inc})


def breakdown(s: SceneData, p: CompareParams, radius_m: float) -> pd.DataFrame:
    """Recall per label bin at each operating point (long format)."""
    bins = label_bins(s.truth, p)
    rows = []
    for op, det in operating_points(s, p).items():
        _, hit = score_points(xy(det, s.crs), s.truth_xy, radius_m)
        for by in bins.columns:
            g = pd.DataFrame({"bin": bins[by], "hit": hit}).groupby("bin")["hit"]
            for b, n, k in zip(g.size().index, g.size(), g.sum(), strict=True):
                rows.append({"scene": s.name, "operating_point": op, "by": by, "bin": b,
                             "labels": int(n), "matched": int(k)})
    out = pd.DataFrame(rows)
    out["recall"] = (out["matched"] / out["labels"]).round(3)
    return out


def detection_breakdown(s: SceneData, p: CompareParams, radius_m: float) -> pd.DataFrame:
    """Detections per incidence bin at each operating point, split into matched / unmatched
    (unmatched = unlabelled berg, ship or clutter; a clutter proxy, not a false-alarm count)."""
    rows = []
    for op, det in operating_points(s, p).items():
        m = match_points(xy(det, s.crs), s.truth_xy, radius_m)
        matched = np.zeros(len(det), bool)
        matched[m.pairs[:, 0]] = True
        inc = pd.cut(det["incidence_deg"].to_numpy(), p.incidence_bins_deg,
                     labels=_bin_labels(p.incidence_bins_deg, "°")).astype(str)
        g = pd.DataFrame({"bin": inc, "matched": matched}).groupby("bin")["matched"]
        for b, n, k in zip(g.size().index, g.size(), g.sum(), strict=True):
            rows.append({"scene": s.name, "operating_point": op, "incidence_deg": b,
                         "detections": int(n), "matched": int(k), "unmatched": int(n - k)})
    return pd.DataFrame(rows)


def radius_table(s: SceneData, p: CompareParams) -> pd.DataFrame:
    """Recall vs. matching radius at the matched-count operating points."""
    rows = []
    for op, det in operating_points(s, p).items():
        d_xy = xy(det, s.crs)
        for r in p.radii_m:
            tp, _ = score_points(d_xy, s.truth_xy, r)
            rows.append({"scene": s.name, "operating_point": op, "radius_m": r,
                         "detections": len(det), "matched": tp,
                         "recall": round(tp / len(s.truth_xy), 3) if len(s.truth_xy) else np.nan})
    return pd.DataFrame(rows)


def run_compare(cfg: Config, log=print) -> dict[str, Path]:
    """All 5d tables and the figure, in <out_dir> (figure also in docs/img)."""
    from iceberg_sar.thor.compare_figure import recall_figure

    p = CompareParams.from_config(cfg)
    radius = point_val_params(cfg).max_dist_m
    p.out_dir.mkdir(parents=True, exist_ok=True)
    sweeps, bd, dbd, rad, meta = [], [], [], [], {}
    scenes = []
    for name, kind in p.scenes.items():
        s = load_scene(cfg, name, kind, p)
        scenes.append(s)
        log(f"{name}: {len(s.truth_xy)} labels compared ({kind}), drift {asdict(s.drift)}")
        meta[name] = {"truth": kind, "labels_compared": len(s.truth_xy),
                      "drift": asdict(s.drift),
                      "detections": {**{f"cfar {pfa:g}": len(d) for pfa, d in s.cfar.items()},
                                     **{f"{t} >= {p.view_score:g}":
                                        int((d["score"] >= p.view_score).sum())
                                        for t, d in s.thor.items()}}}
        # Truth hours from the pass with no reliable drift fit sits km from where the bergs
        # were imaged: scoring it would measure the drift, not the detectors.
        if s.truth_kind != "iip-satellite" and not s.drift.applied:
            meta[name]["scored"] = False
            log(f"  not scored: drift unresolved ({s.drift.n_matched} matches)")
            continue
        meta[name]["scored"] = bool(len(s.truth_xy))
        if not len(s.truth_xy):
            continue
        sweeps.append(sweep_table(s, radius))
        bd.append(breakdown(s, p, radius))
        dbd.append(detection_breakdown(s, p, radius))
        rad.append(radius_table(s, p))
    out = {}
    for key, frames in [("sweep", sweeps), ("breakdown", bd), ("detections_by_incidence", dbd),
                        ("radius", rad)]:
        path = p.out_dir / f"{key}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        out[key] = path
    meta.update({"radius_m": radius, "near_ice_km": p.near_ice_km, "cfar_pfa": p.cfar_pfa,
                 "view_score": p.view_score})
    (p.out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out["summary"] = p.out_dir / "summary.json"
    sweep = pd.concat(sweeps, ignore_index=True)
    img = cfg.root / cfg.section("figures")["out_dir"] / "thor_vs_cfar_recall.png"
    out["figure"] = recall_figure(sweep, p, meta, img)
    return out
