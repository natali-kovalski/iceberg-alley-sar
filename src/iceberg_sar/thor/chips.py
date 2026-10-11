"""Training chips for the THOR head: 512 px crops with point labels, from one scene at a time.

Per scene:
- labels: same-pass IIP Sentinel-1 sightings, moved to their 0.1' cell centre, then snapped
  to the berg (targets.py);
- positive crops cover every label once, with random jitter so bergs are not always centred;
- negative crops (open water, pack-ice edge, land/nodata edge) come from inside the area the
  analysts labelled, so "no label" is more likely to mean "no berg";
- CFAR detections far from any label are stored as `unlabelled`, so training can ignore them
  instead of teaching the head that they are water.

Each chip is an .npz with the normalized THOR input (float16), the valid and ice masks, and
the points in crop pixel coordinates. Heatmaps are rendered at training time.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint

from iceberg_sar.config import Config, metres_to_px
from iceberg_sar.detections import scene_timestamp
from iceberg_sar.pointval import load_truth
from iceberg_sar.preprocess import find_rtc_bands, product_radiometry
from iceberg_sar.thor.encoder import ThorParams, to_model_input
from iceberg_sar.thor.targets import TargetParams, snap_points

ICE_NODATA = 255
KINDS = ("berg", "water", "ice_edge", "edge")


@dataclass(frozen=True)
class ChipParams:
    margin_px: int                 # labels at least this far inside a positive crop
    min_sea_fraction: float        # crops need this much water or ice (not land / nodata)
    negatives_per_positive: float
    negative_mix: dict[str, float]  # water / ice_edge / edge shares of the negatives
    ice_edge_range: tuple[float, float]
    edge_range: tuple[float, float]
    labelled_area_buffer_m: float
    unlabelled_radius_m: float     # CFAR detections farther than this from a label
    coarse_factor: int             # downsampling for crop planning
    seed: int
    val: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    holdout: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]

    @classmethod
    def from_config(cls, cfg: Config) -> ChipParams:
        t = cfg.section("thor")
        c = t["chips"]

        def ranges(rs: list[list[str]]) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
            return tuple((pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)) for a, b in rs)

        return cls(
            margin_px=int(c["margin_px"]),
            min_sea_fraction=float(c["min_sea_fraction"]),
            negatives_per_positive=float(c["negatives_per_positive"]),
            negative_mix={k: float(v) for k, v in c["negative_mix"].items()},
            ice_edge_range=(float(c["ice_edge_range"][0]), float(c["ice_edge_range"][1])),
            edge_range=(float(c["edge_range"][0]), float(c["edge_range"][1])),
            labelled_area_buffer_m=float(c["labelled_area_buffer_m"]),
            unlabelled_radius_m=float(c["unlabelled_radius_m"]),
            coarse_factor=int(c["coarse_factor"]),
            seed=int(c["seed"]),
            val=ranges(c["val"]),
            holdout=ranges(t["training"]["holdout"]),
        )


def scene_split(sar_time: pd.Timestamp, params: ChipParams) -> str:
    """'test' inside the holdout, 'val' inside a validation range, else 'train'."""
    if any(a <= sar_time < b for a, b in params.holdout):
        return "test"
    if any(a <= sar_time < b for a, b in params.val):
        return "val"
    return "train"


def read_crop(
    product_dir: Path, interim_dir: Path, r0: int, c0: int, params: ThorParams
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, rasterio.Affine]:
    """Raw linear bands with land set to NaN, incidence (deg), ice and land masks for a crop.

    Bands come from the raw product so pack ice stays visible; "land" is everything the
    preprocess step masked that is not pack ice (land, outside the AOI, nodata).
    """
    name = product_dir.name
    paths = find_rtc_bands(product_dir)
    win = Window(c0, r0, params.crop_px, params.crop_px)
    with rasterio.open(paths["inc"]) as src:
        inc = src.read(1, window=win).astype(np.float32)
    inc = np.where(inc > 0, inc, np.nan)
    inc_deg = np.degrees(inc) if np.nanmax(inc, initial=0) < 2 * np.pi else inc

    with rasterio.open(interim_dir / f"{name}_seaice_mask.tif") as src, \
            rasterio.open(paths[params.bands[0]]) as ref:
        transform = ref.window_transform(win)
        bounds = rasterio.windows.bounds(win, ref.transform)
        ice_win = src.window(*bounds)
        ice = src.read(1, window=ice_win, out_shape=(params.crop_px, params.crop_px),
                       resampling=Resampling.nearest, boundless=True,
                       fill_value=ICE_NODATA) == 1

    bands: dict[str, np.ndarray] = {}
    sea = np.ones((params.crop_px, params.crop_px), bool)
    for b in params.bands:
        with rasterio.open(paths[b]) as src:
            raw = src.read(1, window=win).astype(np.float32)
        with rasterio.open(interim_dir / f"{name}_{b}_masked.tif") as src:
            masked = src.read(1, window=win)
        sea &= np.isfinite(masked) | (ice & (raw > 0))
        bands[b] = raw
    land = ~sea
    for b in bands:
        bands[b] = np.where(land, np.nan, bands[b])
    return bands, inc_deg, ice, land, transform


def scene_labels(
    product_dir: Path, cfg: Config, tparams: TargetParams
) -> gpd.GeoDataFrame:
    """Same-pass IIP labels for a scene with pixel (row, col), at cell centres and snapped."""
    name = product_dir.name
    sar_time = pd.Timestamp(scene_timestamp(name)).tz_localize(None)
    pts, _ = load_truth(cfg, name, sar_time, "iip-satellite")
    hh = find_rtc_bands(product_dir)["HH"]
    with rasterio.open(hh) as src:
        x, y = Transformer.from_crs(4326, src.crs, always_xy=True).transform(
            pts["lon"].to_numpy(), pts["lat"].to_numpy())  # already at 0.1' cell centres
        rows, cols = rasterio.transform.rowcol(src.transform, x, y, op=float)
        rc = np.c_[rows, cols].astype(np.float64)
        inside = ((rc >= 0) & (rc < [src.height, src.width])).all(axis=1)
        pts, rc = pts[inside].reset_index(drop=True), rc[inside]
        pixel_m = float(src.res[0])
        pad = metres_to_px(tparams.snap_background_m, pixel_m) + 2
        snapped = rc.copy()
        moved = np.zeros(len(rc), bool)
        contrast = np.full(len(rc), np.nan)
        for i, (r, c) in enumerate(np.round(rc).astype(int)):
            win = Window(c - pad, r - pad, 2 * pad + 1, 2 * pad + 1)
            a = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = np.where(a > 0, 10 * np.log10(np.maximum(a, 1e-12)), np.nan)
            p, m, k = snap_points(db, np.array([[pad, pad]]), tparams, pixel_m)
            snapped[i] = p[0] + [r - pad, c - pad]
            moved[i], contrast[i] = m[0], k[0]
    out = pts.assign(row=snapped[:, 0], col=snapped[:, 1], snapped=moved,
                     snap_contrast_db=contrast,
                     snap_m=np.hypot(*(snapped - rc).T) * pixel_m)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=pts.crs)


def coarse_masks(
    product_dir: Path, interim_dir: Path, band: str, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse (sea fraction, ice fraction) grids for crop planning, `factor` px per cell."""
    name = product_dir.name
    with rasterio.open(interim_dir / f"{name}_{band}_masked.tif") as src:
        shape = (src.height // factor, src.width // factor)
        valid = np.isfinite(src.read(1, out_shape=shape, resampling=Resampling.nearest))
        bounds = src.bounds
    with rasterio.open(interim_dir / f"{name}_seaice_mask.tif") as src:
        ice = src.read(1, window=src.window(*bounds), out_shape=shape,
                       resampling=Resampling.nearest, boundless=True, fill_value=ICE_NODATA) == 1
    return (valid | ice).astype(np.float32), ice.astype(np.float32)


def crop_fractions(
    sea: np.ndarray, ice: np.ndarray, r0: int, c0: int, crop_px: int, factor: int
) -> tuple[float, float]:
    """(sea fraction, ice fraction) of a crop from the coarse grids."""
    rs = slice(r0 // factor, (r0 + crop_px) // factor)
    cs = slice(c0 // factor, (c0 + crop_px) // factor)
    s, i = sea[rs, cs], ice[rs, cs]
    return (float(s.mean()), float(i.mean())) if s.size else (0.0, 0.0)


def plan_positive_crops(
    rc: np.ndarray, shape: tuple[int, int], crop_px: int, margin_px: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Top-left corners so that every point lies at least `margin_px` inside some crop.

    Points are visited in random order; an uncovered point gets a crop with a random offset.
    """
    crops: list[tuple[int, int]] = []
    lo, hi = margin_px, crop_px - margin_px

    def covered(r: float, c: float) -> bool:
        return any(lo <= r - r0 < hi and lo <= c - c0 < hi for r0, c0 in crops)

    for k in rng.permutation(len(rc)):
        r, c = rc[k]
        if covered(r, c):
            continue
        r0 = int(np.clip(r - rng.integers(lo, hi), 0, shape[0] - crop_px))
        c0 = int(np.clip(c - rng.integers(lo, hi), 0, shape[1] - crop_px))
        crops.append((r0, c0))
    return crops


def negative_kind(
    sea_frac: float, ice_frac: float, params: ChipParams
) -> str | None:
    """Which negative category a crop falls in, or None."""
    if params.ice_edge_range[0] <= ice_frac <= params.ice_edge_range[1]:
        return "ice_edge"
    if ice_frac == 0 and params.edge_range[0] <= 1 - sea_frac <= params.edge_range[1]:
        return "edge"
    if ice_frac == 0 and sea_frac >= 0.99:
        return "water"
    return None


def plan_negative_crops(
    n: int, area_rc: np.ndarray, sea: np.ndarray, ice: np.ndarray, crop_px: int,
    params: ChipParams, rng: np.random.Generator, tries: int = 20000,
) -> list[tuple[int, int, str]]:
    """Random crops whose centre is inside the labelled area, filled per category quota.

    `area_rc` is a (rows, cols) boolean grid at coarse resolution marking the labelled area.
    """
    quota = {k: int(round(n * w)) for k, w in params.negative_mix.items()}
    f = params.coarse_factor
    cand = np.argwhere(area_rc)
    out: list[tuple[int, int, str]] = []
    if len(cand) == 0:
        return out
    h, w = sea.shape[0] * f, sea.shape[1] * f
    for _ in range(tries):
        if all(v <= 0 for v in quota.values()):
            break
        cr, cc = cand[rng.integers(len(cand))] * f
        r0 = int(np.clip(cr - crop_px // 2, 0, h - crop_px))
        c0 = int(np.clip(cc - crop_px // 2, 0, w - crop_px))
        sea_frac, ice_frac = crop_fractions(sea, ice, r0, c0, crop_px, f)
        if sea_frac < params.min_sea_fraction:
            continue
        kind = negative_kind(sea_frac, ice_frac, params)
        if kind is None or quota.get(kind, 0) <= 0:
            continue
        if any(abs(r0 - a) < crop_px // 2 and abs(c0 - b) < crop_px // 2 for a, b, _ in out):
            continue  # keep negatives from overlapping much
        out.append((r0, c0, kind))
        quota[kind] -= 1
    return out


def labelled_area_grid(
    rc: np.ndarray, shape: tuple[int, int], buffer_px: float, factor: int
) -> np.ndarray:
    """Coarse grid: True inside the buffered convex hull of the labels."""
    from rasterio.features import rasterize

    if len(rc) == 0:
        return np.zeros(shape, bool)
    hull = MultiPoint([(c / factor, r / factor) for r, c in rc]).convex_hull.buffer(
        buffer_px / factor)
    return rasterize([(hull, 1)], out_shape=shape, fill=0, dtype="uint8").astype(bool)


def unlabelled_detections(det_rc: np.ndarray, label_rc: np.ndarray, radius_px: float) -> np.ndarray:
    """CFAR detections with no label within `radius_px`."""
    if len(det_rc) == 0 or len(label_rc) == 0:
        return det_rc
    d, _ = cKDTree(label_rc).query(det_rc)
    return det_rc[d > radius_px]


def visible_at(product_dir: Path, interim_dir: Path, band: str, rc: np.ndarray) -> np.ndarray:
    """True where a (row, col) is water or pack ice in the preprocessed scene, i.e. not land,
    the land buffer, outside the AOI or nodata. Labels elsewhere can't be seen by the model."""
    name = product_dir.name
    if len(rc) == 0:
        return np.zeros(0, bool)
    with rasterio.open(interim_dir / f"{name}_{band}_masked.tif") as src, \
            rasterio.open(interim_dir / f"{name}_seaice_mask.tif") as ice:
        xy = list(zip(*rasterio.transform.xy(src.transform, rc[:, 0], rc[:, 1]), strict=True))
        water = np.isfinite([v[0] for v in src.sample(xy)])
        packed = np.array([v[0] for v in ice.sample(xy)]) == 1
    return water | packed


def to_local(rc: np.ndarray, r0: int, c0: int, crop_px: int) -> np.ndarray:
    """Scene (row, col) points -> crop coordinates, keeping only those inside the crop."""
    if len(rc) == 0:
        return np.empty((0, 2), np.float32)
    rel = rc - [r0, c0]
    return rel[((rel >= 0) & (rel < crop_px)).all(axis=1)].astype(np.float32)


def chip_scene(product_dir: Path, cfg: Config, out_dir: Path) -> pd.DataFrame:
    """Cut chips for one preprocessed scene. Returns one index row per chip."""
    tp = ThorParams.from_config(cfg)
    cp = ChipParams.from_config(cfg)
    tgt = TargetParams.from_config(cfg)
    product_dir = Path(product_dir)
    name = product_dir.name
    interim = cfg.path("interim") / name
    radiometry = product_radiometry(name)
    sar_time = pd.Timestamp(scene_timestamp(name)).tz_localize(None)
    split = scene_split(sar_time, cp)
    rng = np.random.default_rng([cp.seed, int(sar_time.timestamp())])

    with rasterio.open(find_rtc_bands(product_dir)[tp.bands[0]]) as src:
        shape, pixel_m = (src.height, src.width), float(src.res[0])

    labels = scene_labels(product_dir, cfg, tgt)
    label_rc = labels[["row", "col"]].to_numpy(np.float64)
    labels["visible"] = visible_at(product_dir, interim, tp.bands[-1], label_rc)
    seen = labels["visible"].to_numpy()
    sea, ice = coarse_masks(product_dir, interim, tp.bands[-1], cp.coarse_factor)

    det_path = (cfg.path("outputs") / "detections"
                / f"{name}_{cfg.section('cfar')['band']}_detections.geojson")
    det_rc = (gpd.read_file(det_path)[["row", "col"]].to_numpy(np.float64)
              if det_path.is_file() else np.empty((0, 2)))
    unlabelled = unlabelled_detections(
        det_rc, label_rc, cp.unlabelled_radius_m / pixel_m)

    pos = [(r0, c0, "berg") for r0, c0 in
           plan_positive_crops(label_rc[seen], shape, tp.crop_px, cp.margin_px, rng)]
    pos = [p for p in pos
           if crop_fractions(sea, ice, p[0], p[1], tp.crop_px, cp.coarse_factor)[0]
           >= cp.min_sea_fraction]
    area = labelled_area_grid(label_rc, sea.shape, cp.labelled_area_buffer_m / pixel_m,
                              cp.coarse_factor)
    neg = plan_negative_crops(int(round(len(pos) * cp.negatives_per_positive)), area, sea, ice,
                              tp.crop_px, cp, rng)

    scene_dir = out_dir / name
    if scene_dir.is_dir():
        shutil.rmtree(scene_dir)  # a rerun may plan fewer chips; no stale files
    scene_dir.mkdir(parents=True)
    rows = []
    for k, (r0, c0, kind) in enumerate(pos + neg):
        bands, inc_deg, ice_px, land, _ = read_crop(product_dir, interim, r0, c0, tp)
        x = to_model_input(bands, inc_deg, tp, radiometry).astype(np.float16)

        in_box = ((label_rc >= [r0, c0]) & (label_rc < [r0 + tp.crop_px, c0 + tp.crop_px])).all(1)
        in_crop = in_box & seen
        chip_id = f"{name}_{k:03d}"
        np.savez(
            scene_dir / f"{chip_id}.npz",
            x=x, valid=~land, ice=ice_px,
            berg_rc=to_local(label_rc[in_crop], r0, c0, tp.crop_px),
            berg_snapped=labels["snapped"].to_numpy()[in_crop],
            berg_size=labels["size"].astype(str).to_numpy()[in_crop],
            unlabelled_rc=to_local(unlabelled, r0, c0, tp.crop_px),
            masked_rc=to_local(label_rc[in_box & ~seen], r0, c0, tp.crop_px),  # never targets
        )
        rows.append({
            "chip_id": chip_id, "scene": name, "time": sar_time, "split": split, "kind": kind,
            "row0": r0, "col0": c0, "n_bergs": int(in_crop.sum()),
            "n_unlabelled": len(to_local(unlabelled, r0, c0, tp.crop_px)),
            "valid_fraction": round(float((~land).mean()), 3),
            "ice_fraction": round(float(ice_px.mean()), 3),
            "path": str((scene_dir / f"{chip_id}.npz").relative_to(out_dir)),
        })
    index = pd.DataFrame(rows)
    in_any = np.zeros(len(label_rc), bool)
    for r0, c0, _ in pos + neg:
        in_any |= ((label_rc >= [r0, c0]) & (label_rc < [r0 + tp.crop_px, c0 + tp.crop_px])).all(1)
    in_any &= seen
    meta = {
        "scene": name, "split": split, "radiometry": radiometry, "labels": len(labels),
        "labels_snapped": int(labels["snapped"].sum()),
        "labels_masked": int((~seen).sum()),  # outside the AOI or within the land buffer
        "labels_in_chips": int(in_any.sum()),  # visible labels inside some chip
        "median_snap_m": round(float(labels.loc[labels["snapped"], "snap_m"].median()), 1)
        if labels["snapped"].any() else None,
        "cfar_detections": len(det_rc), "unlabelled_detections": len(unlabelled),
        "chips": index["kind"].value_counts().to_dict() if len(index) else {},
    }
    (scene_dir / "scene.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    labels.drop(columns="geometry").assign(time=labels["time"].astype(str)).to_csv(
        scene_dir / "labels.csv", index=False)
    index.to_csv(scene_dir / "index.csv", index=False)
    return index


def load_index(chips_dir: Path) -> pd.DataFrame:
    """All per-scene chip indexes under `chips_dir`."""
    parts = [pd.read_csv(p) for p in sorted(chips_dir.glob("*/index.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summary_table(index: pd.DataFrame) -> pd.DataFrame:
    """Crops and labelled bergs per scene and split (a berg in two crops counts twice)."""
    t = index.pivot_table(index=["split", "scene"], columns="kind", values="chip_id",
                          aggfunc="count", fill_value=0)
    t["crops"] = t.sum(axis=1)
    t["bergs_in_crops"] = index.groupby(["split", "scene"])["n_bergs"].sum()
    totals = t.groupby(level="split").sum()
    totals.index = pd.MultiIndex.from_tuples([(s, "TOTAL") for s in totals.index])
    return pd.concat([t, totals]).sort_index()


def contact_sheet(
    index: pd.DataFrame, chips_dir: Path, png: Path, cfg: Config, n: int = 20, seed: int = 0
) -> Path:
    """Grid of chips (HH dB) with the token-grid target heatmap and points overlaid.

    Picks berg chips with the most labels plus a few of each negative kind. Labels are
    red (snapped) or yellow (not snapped); unlabelled CFAR detections are cyan.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iceberg_sar.thor.targets import render_heatmap

    tp = ThorParams.from_config(cfg)
    tgt = TargetParams.from_config(cfg)
    rng = np.random.default_rng(seed)
    n_neg = n // 4
    bergs = index[index["kind"] == "berg"]
    picks = [bergs.sort_values("n_bergs", ascending=False).head(n - n_neg)]
    negs = index[index["kind"] != "berg"]
    if len(negs):
        picks.append(negs.sample(min(n_neg, len(negs)), random_state=int(rng.integers(1 << 31))))
    sel = pd.concat(picks).head(n)

    cols = 5
    rows_n = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.4 * rows_n),
                             constrained_layout=True, squeeze=False)
    hh_mean, hh_std = tp.norm["HH"]
    g = tp.grid_px()
    for ax, r in zip(axes.flat, sel.itertuples(), strict=False):
        z = np.load(chips_dir / r.path)
        hh = z["x"][tp.bands.index("HH")].astype(np.float32) * hh_std + hh_mean
        hh = np.where(z["valid"], hh, np.nan)
        ax.imshow(hh, cmap="gray", vmin=-30, vmax=-5)
        heat = render_heatmap(z["berg_rc"], (g, g), tp.patch_px, tgt.sigma_m / 10.0)
        ax.imshow(np.ma.masked_less(heat, 0.05), cmap="autumn_r", alpha=0.55, vmin=0, vmax=1,
                  extent=(-0.5, tp.crop_px - 0.5, tp.crop_px - 0.5, -0.5),
                  interpolation="nearest")
        b, s = z["berg_rc"], z["berg_snapped"].astype(bool)
        ax.scatter(b[s, 1], b[s, 0], s=50, facecolors="none", edgecolors="red", linewidths=0.8)
        ax.scatter(b[~s, 1], b[~s, 0], s=50, facecolors="none", edgecolors="yellow",
                   linewidths=0.8)
        u = z["unlabelled_rc"]
        ax.scatter(u[:, 1], u[:, 0], s=50, marker="s", facecolors="none", edgecolors="cyan",
                   linewidths=0.8)
        if z["ice"].any():
            ax.contour(z["ice"].astype(np.float32), levels=[0.5], colors="deepskyblue",
                       linewidths=0.6)
        ax.set_title(f"{r.split} {r.kind}  {str(r.time)[:10]}\n{r.n_bergs} bergs, "
                     f"{r.n_unlabelled} unlabelled", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("THOR chips: HH dB, target heatmap (token grid), labels red = snapped / "
                 "yellow = not, cyan = unlabelled CFAR, blue = pack ice", fontsize=10)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png
