"""Medium bergs as THOR sees them: the Færch EW scene (40 m) next to its 10 m training chips.

One figure (`bench-thor-bergs`), two blocks of rows, each panel `bergs.window_m` wide:
- Færch: verified medium labels, hits and misses of the reference run (`bergs.reference`) at
  its best-F threshold, picked at spread-out lengths. HH / HV (no TNR), HV (TNR), then the
  `bench-thor` heatmap of every run, with the score at the berg and that run's peaks at or
  above its best threshold.
- Training: IIP medium bergs from the head's training chips, picked at evenly spaced quantiles
  of their score as trained. HH / HV and the head at 10 m, then the same chip averaged to 40 m
  (4 x 4 mean of linear intensity) through both 40 m variants: the resolution effect alone, on
  bergs the head is known to find.
Heatmaps share one 0-1 scale, so the compressed scores on the EW scene stay visible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

from iceberg_sar.benchmark.reproduce import scene_paths
from iceberg_sar.benchmark.score import _other_variant, _variant, load_labels, scoring_aoi
from iceberg_sar.benchmark.thor_faerch import (
    ThorFaerchParams,
    ThorRunParams,
    crop_geometry,
)
from iceberg_sar.config import Config
from iceberg_sar.thor.encoder import ThorParams, embed_blocks, load_encoder
from iceberg_sar.thor.features import FeatureParams


@dataclass(frozen=True)
class BergsParams:
    window_m: float
    reference: tuple[str, str]   # (variant, stack)
    n_hits: int
    n_misses: int
    train_size: str
    n_train: int
    multilook: int
    match_m: float
    figure: Path

    @classmethod
    def from_config(cls, cfg: Config) -> BergsParams:
        b = cfg.section("benchmark")["faerch2023"]["thor"]["bergs"]
        return cls(window_m=float(b["window_m"]),
                   reference=(b["reference"]["variant"], b["reference"]["stack"]),
                   n_hits=int(b["n_hits"]), n_misses=int(b["n_misses"]),
                   train_size=str(b["train_size"]), n_train=int(b["n_train"]),
                   multilook=int(b["multilook"]), match_m=float(b["match_m"]),
                   figure=cfg.root / b["figure"])


def spread_pick(values: np.ndarray, n: int) -> np.ndarray:
    """Indices of `n` distinct items nearest the evenly spaced quantiles (1/2n, 3/2n, ...)."""
    if n <= 0 or not len(values):
        return np.empty(0, int)
    order = np.argsort(values, kind="stable")
    ranks = np.arange(len(values))
    picks: list[int] = []
    for q in (np.arange(min(n, len(values))) + 0.5) / n:
        k = q * (len(values) - 1)
        for kk in sorted(ranks, key=lambda j: (abs(j - k), j)):   # nearest free rank
            if int(order[kk]) not in picks:
                picks.append(int(order[kk]))
                break
    return np.array(picks, int)


def max_within(prob: np.ndarray, rc: tuple[float, float], r: float) -> float:
    """Highest finite value of `prob` within `r` (grid units) of `rc` (fractional row, col)."""
    i0, i1 = int(np.floor(rc[0] - r)), int(np.ceil(rc[0] + r)) + 1
    j0, j1 = int(np.floor(rc[1] - r)), int(np.ceil(rc[1] + r)) + 1
    ii, jj = np.mgrid[max(i0, 0):min(i1, prob.shape[0]), max(j0, 0):min(j1, prob.shape[1])]
    near = (ii + 0.5 - rc[0]) ** 2 + (jj + 0.5 - rc[1]) ** 2 <= r ** 2
    vals = prob[ii[near], jj[near]]
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if len(vals) else float("nan")


def window_around(a: np.ndarray, rc: tuple[float, float], half: int) -> np.ndarray:
    """(2 half) square of `a` centred on `rc`, NaN-padded where it leaves the array."""
    out = np.full((2 * half, 2 * half), np.nan, np.float32)
    r0, c0 = int(round(rc[0])) - half, int(round(rc[1])) - half
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + 2 * half, a.shape[0]), min(c0 + 2 * half, a.shape[1])
    if re > rs and ce > cs:
        out[rs - r0:re - r0, cs - c0:ce - c0] = a[rs:re, cs:ce]
    return out


def multilook_db(z: np.ndarray, valid: np.ndarray, mean: float, std: float, f: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Normalised dB band (chip `x`) -> f x f mean of linear intensity, back in dB.
    Returns (dB, valid fraction) on the coarse grid; cells with no valid pixel are NaN."""
    lin = np.where(valid, 10 ** ((z.astype(np.float64) * std + mean) / 10), 0.0)
    h, w = z.shape[0] // f, z.shape[1] // f
    s = lin[:h * f, :w * f].reshape(h, f, w, f).sum((1, 3))
    n = valid[:h * f, :w * f].reshape(h, f, w, f).sum((1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(n > 0, 10 * np.log10(s / np.maximum(n, 1)), np.nan)
    return db.astype(np.float32), n / f ** 2


# --- Færch scene ---------------------------------------------------------------------------------

def faerch_rows(cfg: Config, bp: BergsParams, log: Callable[[str], None]) -> list[dict[str, Any]]:
    """Picked Færch medium bergs: image and heatmap windows per run, scores, hit flags."""
    import geopandas as gpd

    fc = cfg.section("benchmark")["faerch2023"]
    rp = ThorRunParams.from_config(cfg)
    scores = pd.read_csv(rp.out_dir / "thor_scores.csv")
    labels, _ = load_labels(cfg, "faerch")
    aoi = scoring_aoi(cfg, fc["score"]["aoi"])
    inside = gpd.GeoSeries(gpd.points_from_xy(labels.x, labels.y)).within(aoi).to_numpy()
    med = labels[inside & (labels.size_class == "medium") & (labels.status == "verified")]
    med = med.reset_index(drop=True)
    lab = gpd.read_file(cfg.root / fc["labels_faerch"]["out_dir"] / "labels_faerch.geojson")
    length = lab.set_index("label_id").length_m.reindex(med.label_id).to_numpy()

    runs = [r for r in rp.runs if (rp.out_dir / f"{r.tag}.geojson").is_file()]
    best_thr = {}
    peaks = {}
    for r in runs:
        d = scores[(scores["stack"] == r.stack) & (scores.series == r.series)]
        best_thr[r.tag] = float(d.loc[d.f.idxmax(), "threshold"])
        g = gpd.read_file(rp.out_dir / f"{r.tag}.geojson").to_crs(fc["crs"])
        peaks[r.tag] = (np.column_stack([g.geometry.x, g.geometry.y]), g.score.to_numpy())

    xy = med[["x", "y"]].to_numpy(float)

    def hits(tag: str) -> np.ndarray:
        pxy, ps = peaks[tag]
        sel = ps >= best_thr[tag]
        if not sel.any():
            return np.zeros(len(xy), bool)
        d, _ = cKDTree(pxy[sel]).query(xy, distance_upper_bound=bp.match_m)
        return np.isfinite(d)

    ref = next(r for r in runs if (r.variant, r.stack) == bp.reference and r.norm == "iw")
    ref_hit = hits(ref.tag)
    log(f"Færch: {len(med)} verified medium bergs in the scoring area, {int(ref_hit.sum())} "
        f"hit by {ref.tag} at {best_thr[ref.tag]:g}")
    pick = np.concatenate([np.flatnonzero(ref_hit)[spread_pick(length[ref_hit], bp.n_hits)],
                           np.flatnonzero(~ref_hit)[spread_pick(length[~ref_hit], bp.n_misses)]])
    hit_all = {r.tag: hits(r.tag) for r in runs}

    stacks = {_variant(c): scene_paths(c) for c in (cfg, _other_variant(cfg))}
    rows = []
    for i in pick:
        row: dict[str, Any] = {"label_id": int(med.label_id[i]), "length_m": float(length[i]),
                               "images": {}, "heat": {}}
        for sname, sp in stacks.items():
            with rasterio.open(sp.stack) as src:
                rc = ~src.transform * (xy[i, 0], xy[i, 1])
                half = int(round(bp.window_m / 2 / abs(src.transform.a)))
                win = Window(int(round(rc[0])) - half, int(round(rc[1])) - half, 2 * half, 2 * half)
                for k, band in enumerate(src.descriptions):
                    a = src.read(k + 1, window=win, boundless=True, fill_value=np.nan)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        row["images"][f"{band} ({sname})"] = 10 * np.log10(a)
        for r in runs:
            with rasterio.open(rp.out_dir / f"{r.tag}_heatmap.tif") as src:
                prob = src.read(1)
                tok_m = abs(src.transform.a)
                col, rr = ~src.transform * (xy[i, 0], xy[i, 1])
            half = int(round(bp.window_m / 2 / tok_m))
            pxy, ps = peaks[r.tag]
            near = (np.abs(pxy - xy[i]).max(1) <= bp.window_m / 2) & (ps >= best_thr[r.tag])
            row["heat"][r.tag] = {
                "prob": window_around(prob, (rr, col), half),
                "score": max_within(prob, (rr, col), bp.match_m / tok_m),
                "hit": bool(hit_all[r.tag][i]), "thr": best_thr[r.tag],
                # peaks in panel token units (0..2 half), for markers
                "peaks": (pxy[near] - xy[i]) / tok_m * [1, -1] + half,
                "title": f"{r.stack} / {r.series}",
            }
        rows.append(row)
    return rows


# --- training chips ------------------------------------------------------------------------------

def train_rows(cfg: Config, bp: BergsParams, device: str, log: Callable[[str], None]
               ) -> list[dict[str, Any]]:
    """Picked training bergs: 10 m and 40 m windows, head heatmaps at 10 m and 40 m."""
    import torch

    from iceberg_sar.thor.chips import load_index
    from iceberg_sar.thor.features import feature_path
    from iceberg_sar.thor.train import load_head

    p, tp, fp = ThorFaerchParams.from_config(cfg), ThorParams.from_config(cfg), \
        FeatureParams.from_config(cfg)
    base = cfg.root / cfg.section("thor")["training"]["dir"]
    chips_dir, feat_dir = base / "chips", base / "features"
    head = load_head(p.head, device)
    index = load_index(chips_dir)
    index = index[(index.split == "train") & (index.n_bergs > 0)]
    tok10 = tp.patch_px                   # 8 px of 10 m
    half10 = int(round(bp.window_m / 2 / 10))
    cands = []
    for chip in index.itertuples():
        z = np.load(chips_dir / chip.path, allow_pickle=True)   # our own chips (object labels)
        keep = (z["berg_size"].astype(str) == bp.train_size) & z["berg_snapped"].astype(bool)
        if not keep.any():
            continue
        f = feature_path(feat_dir, pd.Series({"scene": chip.scene, "chip_id": chip.chip_id}))
        if not f.is_file():
            continue
        with torch.no_grad():
            feats = torch.from_numpy(np.load(f).astype(np.float32))[None].to(device)
            prob = torch.sigmoid(head(feats))[0, 0].cpu().numpy()
        for rc in z["berg_rc"][keep]:
            if not ((half10 <= rc) & (rc <= tp.crop_px - half10)).all():
                continue          # panel would run off the chip
            cands.append({"chip": chip, "rc": (float(rc[0]), float(rc[1])),
                          "score": max_within(prob, (rc[0] / tok10, rc[1] / tok10),
                                              bp.match_m / (10 * tok10)),
                          "prob": prob})
    log(f"training: {len(cands)} {bp.train_size} bergs in train-split chips, panel inside the chip")
    pick = spread_pick(np.array([c["score"] for c in cands]), bp.n_train)

    variants = {v.name: v for v in p.variants}
    encoders = {}
    for v in variants.values():
        crop_px, _ = crop_geometry(p.crop_m, 10 * bp.multilook, v)
        encoders[v.name] = load_encoder(replace(tp, crop_px=crop_px, patch_px=v.patch_px),
                                        10.0 * bp.multilook, device, v.band_gsd_m)
    rows = []
    for k in pick:
        c = cands[k]
        z = np.load(chips_dir / c["chip"].path, allow_pickle=True)
        x, valid = z["x"].astype(np.float32), z["valid"].astype(bool)
        row: dict[str, Any] = {"chip_id": c["chip"].chip_id, "time": str(c["chip"].time)[:10],
                               "images": {}, "heat": {}}
        db40, x40 = {}, []
        for b, band in enumerate(tp.bands):
            mean, std = tp.norm[band]
            row["images"][f"{band} 10 m"] = window_around(
                np.where(valid, x[b] * std + mean, np.nan), c["rc"], half10)
            db, frac = multilook_db(x[b], valid, mean, std, bp.multilook)
            db40[band] = db
            x40.append(np.where(frac > 0, (np.clip(db, *tp.db_clip) - mean) / std, 0.0))
        rc40 = (c["rc"][0] / bp.multilook, c["rc"][1] / bp.multilook)
        half40 = half10 // bp.multilook
        half_tok = int(round(bp.window_m / 2 / (10 * tok10)))
        row["heat"]["10 m (as trained)"] = {
            "prob": window_around(c["prob"], (c["rc"][0] / tok10, c["rc"][1] / tok10), half_tok),
            "score": c["score"]}
        for band in tp.bands:
            row["images"][f"{band} 40 m"] = window_around(db40[band], rc40, half40)
        x40a = np.stack(x40).astype(np.float32)[None]
        for name, enc in encoders.items():
            feats = embed_blocks(enc, x40a, fp.blocks, device)
            with torch.no_grad():
                prob = torch.sigmoid(head(torch.from_numpy(feats).to(device)))[0, 0].cpu().numpy()
            tok40 = x40a.shape[-1] / prob.shape[0]           # 40 m px per token (2)
            rct = (rc40[0] / tok40, rc40[1] / tok40)
            row["heat"][f"40 m, {name}"] = {
                "prob": window_around(prob, rct, half_tok),
                "score": max_within(prob, rct, bp.match_m / (10 * bp.multilook * tok40))}
        rows.append(row)
    return rows


# --- figure --------------------------------------------------------------------------------------

DB_RANGE = {"HH": (-30.0, -5.0), "HV": (-38.0, -15.0)}   # fixed, so rows compare
RING = "#eb6834"      # categorical slot 2: label ring / peaks
HEAT_CMAP = "Blues"   # one hue, light -> dark (sequential)


def berg_figure(faerch: list[dict[str, Any]], train: list[dict[str, Any]], out: Path,
                window_m: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = max(len(faerch[0]["images"]) - 1 + len(faerch[0]["heat"]) if faerch else 0,
               len(train[0]["images"]) + len(train[0]["heat"]) if train else 0)
    panel = 1.75
    fig = plt.figure(figsize=(panel * ncol + 1.2, panel * (len(faerch) + len(train)) + 1.6),
                     layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[len(faerch) + 0.35, len(train) + 0.35])

    def show_db(ax: Any, a: np.ndarray, band: str) -> None:
        ax.imshow(a, cmap="gray", vmin=DB_RANGE[band][0], vmax=DB_RANGE[band][1],
                  interpolation="nearest")
        n = a.shape[0]
        ax.add_patch(plt.Circle((n / 2 - 0.5, n / 2 - 0.5), n * 0.12, fill=False, ec=RING, lw=1))

    def show_heat(ax: Any, h: dict[str, Any]) -> Any:
        p = h["prob"]
        im = ax.imshow(p, cmap=HEAT_CMAP, vmin=0, vmax=1, interpolation="nearest")
        n = p.shape[0]
        ax.add_patch(plt.Circle((n / 2 - 0.5, n / 2 - 0.5), n * 0.12, fill=False, ec=RING, lw=1))
        if len(h.get("peaks", [])):
            pk = h["peaks"]
            ax.scatter(pk[:, 0] - 0.5, pk[:, 1] - 0.5, s=10, marker="s", c=RING)
        txt = f"{h['score']:.2f}" + ("" if "hit" not in h else (" hit" if h["hit"] else " miss"))
        ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=8, color="#0b0b0b",
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.2})
        return im

    im = None
    cbar_at: tuple[Any, Any] | None = None   # (subfigure, its axes): one colorbar, shared scale
    if faerch:
        axes = top.subplots(len(faerch), ncol, squeeze=False)
        cbar_at = (top, axes)
        img_keys = [k for k in faerch[0]["images"] if k != "HH (TNR)"]
        heat_keys = list(faerch[0]["heat"])
        for r, row in enumerate(faerch):
            for c, key in enumerate(img_keys):
                show_db(axes[r, c], row["images"][key], key.split()[0])
                if r == 0:
                    axes[r, c].set_title(f"{key} dB", fontsize=8)
            for c, key in enumerate(heat_keys, start=len(img_keys)):
                h = row["heat"][key]
                im = show_heat(axes[r, c], h)
                if r == 0:
                    axes[r, c].set_title(f"THOR {h['title']}\n(peaks >= {h['thr']:g})",
                                         fontsize=8)
            axes[r, 0].set_ylabel(f"label {row['label_id']}\n{row['length_m']:.0f} m",
                                  fontsize=8)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        top.suptitle("Færch 2023 scene (EW, 40 m, July): verified medium bergs, hits then misses "
                     "of noTNR / upsample at its best-F threshold", fontsize=10)
    if train:
        axes = bottom.subplots(len(train), ncol, squeeze=False)
        cbar_at = cbar_at or (bottom, axes)
        for r, row in enumerate(train):
            keys = [("img", k) for k in row["images"] if k.endswith("10 m")] + \
                [("heat", "10 m (as trained)")] + \
                [("img", k) for k in row["images"] if k.endswith("40 m")] + \
                [("heat", k) for k in row["heat"] if k.startswith("40 m")]
            for c, (kind, key) in enumerate(keys):
                ax = axes[r, c]
                if kind == "img":
                    show_db(ax, row["images"][key], key.split()[0])
                    title = f"{key} dB"
                else:
                    im = show_heat(ax, row["heat"][key])
                    title = f"THOR {key}"
                if r == 0:
                    ax.set_title(title, fontsize=8)
            for ax in axes[r, len(keys):]:
                ax.set_visible(False)
            axes[r, 0].set_ylabel(f"{row['time']}\n{row['chip_id'][-3:]}", fontsize=8)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        bottom.suptitle("Training chips (IW, 10 m, April-June): IIP medium bergs (61-120 m) at "
                        "spread-out training scores; the same chip averaged to 40 m on the right",
                        fontsize=10)
    if im is not None and cbar_at is not None:
        sub, axs = cbar_at
        cb = sub.colorbar(im, ax=axs[:, -1].tolist(), shrink=0.6, location="right", pad=0.01)
        cb.set_label("THOR head score", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Medium icebergs through THOR (linear head, unchanged). Panels "
                 f"{window_m / 1000:g} km; ring = label; squares = peaks at the run's threshold; "
                 "score = best token within 250 m", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def run_bergs(cfg: Config, device: str | None = None, log: Callable[[str], None] = print
              ) -> Path:
    import torch

    bp = BergsParams.from_config(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    faerch = faerch_rows(cfg, bp, log)
    train = train_rows(cfg, bp, device, log)
    for row in faerch:
        log(f"Færch label {row['label_id']} ({row['length_m']:.0f} m): " + ", ".join(
            f"{k.removeprefix('thor-')} {h['score']:.2f}{'*' if h['hit'] else ''}"
            for k, h in row["heat"].items()))
    for row in train:
        log(f"train {row['chip_id']}: " + ", ".join(
            f"{k} {h['score']:.2f}" for k, h in row["heat"].items()))
    return berg_figure(faerch, train, bp.figure, bp.window_m)
