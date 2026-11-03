"""THOR on the Færch 2023 EW scene (40 m): smoke test before wiring it into `bench-score`.

The head was trained on 10 m HyP3 IW frames: 512 px crops, 64 x 64 tokens of 80 m. THOR
resizes each band to crop / GSD pixels before cutting patches, so a 128 px (5.12 km) crop of
the 40 m stack can give the same token grid in two ways (`benchmark.faerch2023.thor.variants`):
- `upsample`: bands declared 10 m; THOR resamples 128 -> 512 px (bilinear), 8 x 8 px patches.
- `native`: bands declared 40 m (`IW_HH_40`); 2 x 2 px patches, patch embedding resized.
Both run on both stacks (thermal noise removal on / off). SNAP gives sigma0, so there is no
gamma0 conversion; inputs are normalised with the IW stats the head was trained with.

Per crop: a figure (dB bands, then PCA of the tokens and the head's heatmap per stack and
variant, with the Færch labels) and a JSON with the label scores, how far labels sit from
water in feature space (as in 5a), and P / R inside the crop at the viewer threshold.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import box

from iceberg_sar.benchmark.reproduce import scene_paths
from iceberg_sar.benchmark.score import (
    MatchSettings,
    _other_variant,
    _variant,
    apply_match_settings,
    eval_domain,
    labels_in_domain,
    load_labels,
    score_points,
    scoring_aoi,
)
from iceberg_sar.config import Config
from iceberg_sar.preprocess import linear_to_db
from iceberg_sar.thor import PIPELINE, PIPELINE_SHORT
from iceberg_sar.thor.encoder import ThorParams, embed_blocks, load_encoder, pca_rgb, to_model_input
from iceberg_sar.thor.features import FeatureParams
from iceberg_sar.thor.heads import extract_peaks
from iceberg_sar.thor.infer import token_argmax
from iceberg_sar.thor.smoke import token_separation


@dataclass(frozen=True)
class Variant:
    name: str
    band_gsd_m: int
    patch_px: int        # in THOR's input pixels (after its resize)


@dataclass(frozen=True)
class ThorFaerchParams:
    head: Path
    crop_m: float
    variants: tuple[Variant, ...]
    n_crops: int
    min_valid: float
    out_dir: Path
    score_threshold: float   # thor.compare.view_score
    token_min_valid: float   # thor.train.min_valid
    unknown_ignore_m: float

    @classmethod
    def from_config(cls, cfg: Config) -> ThorFaerchParams:
        fc = cfg.section("benchmark")["faerch2023"]
        t, thor = fc["thor"], cfg.section("thor")
        return cls(
            head=cfg.path("models") / "thor" / t["head"] / "final" / "head.pt",
            crop_m=float(t["crop_m"]),
            variants=tuple(Variant(k, int(v["band_gsd_m"]), int(v["patch_px"]))
                           for k, v in t["variants"].items()),
            n_crops=int(t["smoke"]["n_crops"]), min_valid=float(t["smoke"]["min_valid"]),
            out_dir=cfg.root / t["smoke"]["out_dir"],
            score_threshold=float(thor["compare"]["view_score"]),
            token_min_valid=float(thor["train"]["min_valid"]),
            unknown_ignore_m=float(fc["score"]["unknown_ignore_m"]),
        )


def crop_geometry(crop_m: float, pixel_m: float, v: Variant) -> tuple[int, int]:
    """(crop px in the stack, token size in stack px). Fails if the token grid doesn't fit."""
    crop_px = crop_m / pixel_m
    input_px = crop_m / v.band_gsd_m          # THOR's input size after its resize
    if crop_px != int(crop_px) or input_px != int(input_px) or input_px % v.patch_px:
        raise ValueError(f"{v.name}: crop {crop_m:g} m doesn't fit {pixel_m:g} m pixels, "
                         f"{v.band_gsd_m} m GSD and {v.patch_px} px patches")
    grid = int(input_px) // v.patch_px
    if int(crop_px) % grid:
        raise ValueError(f"{v.name}: {int(crop_px)} px crop isn't a whole number of {grid} tokens")
    return int(crop_px), int(crop_px) // grid


def pick_crops(lab_rc: np.ndarray, valid: np.ndarray, crop_px: int, n: int, min_valid: float,
               ok: Callable[[int, int], bool] = lambda r0, c0: True) -> list[tuple[int, int, int]]:
    """Up to `n` non-overlapping crops, (r0, c0, labels inside), most labels first.

    Candidates are centred on each label; a crop needs `min_valid` sea and `ok(r0, c0)`.
    """
    h, w = valid.shape
    cands = []
    for r, c in lab_rc:
        r0 = int(np.clip(round(r) - crop_px // 2, 0, h - crop_px))
        c0 = int(np.clip(round(c) - crop_px // 2, 0, w - crop_px))
        inside = int(((lab_rc[:, 0] >= r0) & (lab_rc[:, 0] < r0 + crop_px)
                      & (lab_rc[:, 1] >= c0) & (lab_rc[:, 1] < c0 + crop_px)).sum())
        cands.append((inside, r0, c0))
    chosen: list[tuple[int, int, int]] = []
    for inside, r0, c0 in sorted(set(cands), key=lambda t: (-t[0], t[1], t[2])):
        if len(chosen) == n:
            break
        if any(abs(r0 - a) < crop_px and abs(c0 - b) < crop_px for a, b, _ in chosen):
            continue
        if valid[r0:r0 + crop_px, c0:c0 + crop_px].mean() < min_valid or not ok(r0, c0):
            continue
        chosen.append((r0, c0, inside))
    return chosen


def read_stack_crop(stack: Path, valid: Path, r0: int, c0: int, crop_px: int
                    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Linear HH / HV with land and nodata as NaN, and the sea (valid) mask, for one crop."""
    win = Window(c0, r0, crop_px, crop_px)
    with rasterio.open(valid) as src:
        sea = src.read(1, window=win).astype(bool)
    with rasterio.open(stack) as src:
        bands = {d: src.read(i + 1, window=win).astype(np.float32)
                 for i, d in enumerate(src.descriptions)}
    for b in bands.values():
        b[~sea | ~np.isfinite(b)] = np.nan
    return bands, sea


def pool(m: np.ndarray, tok_px: int) -> np.ndarray:
    g0, g1 = m.shape[0] // tok_px, m.shape[1] // tok_px
    return m[:g0 * tok_px, :g1 * tok_px].reshape(g0, tok_px, g1, tok_px).mean((1, 3))


def label_scores(prob: np.ndarray, lab_tok: np.ndarray) -> np.ndarray:
    """Head score at each label: max over the 3 x 3 tokens around it (labels ~ 1 token off)."""
    if not len(lab_tok):
        return np.empty(0)
    m = ndimage.maximum_filter(np.nan_to_num(prob, nan=0.0), size=3, mode="constant")
    i = np.clip(lab_tok[:, 0], 0, prob.shape[0] - 1)
    j = np.clip(lab_tok[:, 1], 0, prob.shape[1] - 1)
    return m[i, j]


def water_scores(prob: np.ndarray, tok_valid: np.ndarray, near_tok: np.ndarray, r: int = 3
                 ) -> np.ndarray:
    """Head scores of sea tokens more than `r` tokens from any label or unknown object."""
    near = np.zeros(prob.shape, bool)
    for i, j in near_tok:
        near[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1] = True
    return prob[tok_valid & ~near]


def _rc(transform: rasterio.Affine, xy: np.ndarray) -> np.ndarray:
    if not len(xy):
        return np.empty((0, 2))
    cols, rows = ~transform * (xy[:, 0], xy[:, 1])
    return np.column_stack([rows, cols])


def run_thor_smoke(cfg: Config, device: str | None = None, n_crops: int | None = None,
                   log: Callable[[str], None] = print) -> dict[str, Path]:
    import torch

    from iceberg_sar.thor.train import load_head

    p = ThorFaerchParams.from_config(cfg)
    tp, fp = ThorParams.from_config(cfg), FeatureParams.from_config(cfg)
    if tp.layer not in fp.blocks:
        raise ValueError(f"thor.layer {tp.layer} must be one of thor.features.blocks {fp.blocks}")
    if not p.head.is_file():
        raise FileNotFoundError(f"{p.head} missing; run `thor-fetch-head`")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    stacks = {_variant(c): scene_paths(c) for c in (cfg, _other_variant(cfg))}
    stacks = dict(sorted(stacks.items(), key=lambda kv: kv[0] != "TNR"))   # TNR row first
    for s in stacks.values():
        if not s.stack.is_file():
            raise FileNotFoundError(f"{s.stack} missing; run bench-preprocess (--tnr / --no-tnr)")
    ref = next(iter(stacks.values()))
    with rasterio.open(ref.stack) as src:
        transform, pixel_m = src.transform, abs(src.transform.a)
    with rasterio.open(ref.valid) as src:
        valid_full = src.read(1).astype(bool)

    geo = {v.name: crop_geometry(p.crop_m, pixel_m, v) for v in p.variants}
    crop_px = {g[0] for g in geo.values()}.pop()
    labels, unknown = load_labels(cfg, "faerch")
    lab_xy = labels[["x", "y"]].to_numpy(float)
    lab_rc = _rc(transform, lab_xy)
    unk_rc = _rc(transform, unknown)
    ver = (labels.status == "verified").to_numpy()
    aoi = scoring_aoi(cfg, cfg.section("benchmark")["faerch2023"]["score"]["aoi"])

    def in_aoi(r0: int, c0: int) -> bool:
        x0, y0 = transform * (c0, r0 + crop_px)
        x1, y1 = transform * (c0 + crop_px, r0)
        return box(x0, y0, x1, y1).within(aoi)

    crops = pick_crops(lab_rc[ver], valid_full, crop_px, n_crops or p.n_crops, p.min_valid,
                       in_aoi)
    if not crops:
        raise RuntimeError("No crop with enough sea inside the scoring area")
    del valid_full
    log(f"{len(crops)} crops of {crop_px} px ({p.crop_m / 1000:g} km), device {device}")

    head = load_head(p.head, device)
    tp_v = {v.name: replace(tp, crop_px=crop_px, patch_px=v.patch_px) for v in p.variants}
    encoders = {v.name: load_encoder(tp_v[v.name], pixel_m, device, v.band_gsd_m)
                for v in p.variants}
    hh = tp.bands.index("HH")
    b_layer = fp.blocks.index(tp.layer)
    p.out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    summary: list[dict[str, Any]] = []

    for k, (r0, c0, _) in enumerate(crops):
        def inside(rc: np.ndarray, r0: int = r0, c0: int = c0) -> np.ndarray:
            if not len(rc):
                return np.zeros(0, bool)
            return ((rc[:, 0] >= r0) & (rc[:, 0] < r0 + crop_px)
                    & (rc[:, 1] >= c0) & (rc[:, 1] < c0 + crop_px))

        in_lab = inside(lab_rc)
        crop_labels = labels[in_lab].reset_index(drop=True)
        crop_lab_rc = lab_rc[in_lab] - [r0, c0]
        crop_ver = ver[in_lab]
        crop_unk_rc = unk_rc[inside(unk_rc)] - [r0, c0] if len(unk_rc) else np.empty((0, 2))
        crop_transform = transform * rasterio.Affine.translation(c0, r0)
        panels: dict[str, dict[str, Any]] = {}
        db: dict[str, dict[str, np.ndarray]] = {}
        for sname, sp in stacks.items():
            bands, sea = read_stack_crop(sp.stack, sp.valid, r0, c0, crop_px)
            db[sname] = {b: linear_to_db(bands[b]) for b in tp.bands}
            x = to_model_input(bands, None, tp, radiometry="sigma0")
            for v in p.variants:
                tok_px = geo[v.name][1]
                feats = embed_blocks(encoders[v.name], x[None], fp.blocks, device)
                with torch.no_grad():
                    prob = torch.sigmoid(head(torch.from_numpy(feats).to(device)))[0, 0]
                prob = prob.float().cpu().numpy()
                d = feats.shape[1] // len(fp.blocks)
                layer_feats = feats[0, b_layer * d:(b_layer + 1) * d].astype(np.float32)
                tok_valid = pool(sea.astype(float), tok_px) >= p.token_min_valid
                if tok_valid.shape != prob.shape:
                    raise RuntimeError(f"{v.name}: token grid {prob.shape} != {tok_valid.shape}")
                rgb, _ = pca_rgb(layer_feats, tok_valid)
                sep, _ = token_separation(layer_feats, crop_lab_rc[crop_ver], np.zeros_like(sea),
                                          ~sea, tok_px)
                lab_tok = (crop_lab_rc // tok_px).astype(int)
                unk_tok = (crop_unk_rc // tok_px).astype(int)
                ls = label_scores(prob, lab_tok)
                ws = water_scores(prob, tok_valid, np.vstack([lab_tok, unk_tok]))

                ij, scores = extract_peaks(prob, p.score_threshold, tok_valid)
                ar, ac = token_argmax(bands[tp.bands[hh]], tok_px)
                pk_rc = (np.column_stack([ar[ij[:, 0], ij[:, 1]], ac[ij[:, 0], ij[:, 1]]])
                         if len(ij) else np.empty((0, 2)))
                pk_xy = np.empty((0, 2))
                if len(pk_rc):
                    pk_xy = np.column_stack(
                        crop_transform * (pk_rc[:, 1] + 0.5, pk_rc[:, 0] + 0.5))
                if len(unknown) and len(pk_xy):
                    near = cKDTree(unknown).query(
                        pk_xy, distance_upper_bound=p.unknown_ignore_m)[0] <= p.unknown_ignore_m
                    pk_xy, pk_rc = pk_xy[~near], pk_rc[~near]
                sc = score_points(pk_xy, crop_labels)
                panels[f"{sname}/{v.name}"] = {"rgb": rgb, "prob": prob, "peaks": pk_rc}
                lv, li = ls[crop_ver], ls[~crop_ver]
                row = {
                    "crop": k, "row": r0, "col": c0, "stack": sname, "variant": v.name,
                    "labels": int(in_lab.sum()), "verified": int(crop_ver.sum()),
                    "label_score_median_v": _r(np.median(lv)) if len(lv) else None,
                    "label_score_median_i": _r(np.median(li)) if len(li) else None,
                    "labels_above_thr_v": int((lv >= p.score_threshold).sum()),
                    "water_score_p99": _r(np.percentile(ws, 99)) if len(ws) else None,
                    "water_score_max": _r(ws.max()) if len(ws) else None,
                    "peaks": len(pk_xy),
                    "TP": sc["TP"], "FP": sc["FP"], "FN": sc["FN"], "TP_v": sc["TP_v"],
                    "n_v": sc["n_v"], "recall_v": _r(sc["recall_v"]),
                    "berg_rank_pct": sep.get("berg_rank_pct"),
                    "berg_median_dist": sep.get("berg_median_dist"),
                }
                summary.append(row)
                log(f"crop {k} {sname:5s} {v.name:8s}: verified label score median "
                    f"{row['label_score_median_v']}, water p99 {row['water_score_p99']}, "
                    f"{row['peaks']} peaks >= {p.score_threshold:g}: TP {row['TP']} "
                    f"FP {row['FP']} FN {row['FN']}, verified {row['TP_v']}/{row['n_v']}")
        fig = p.out_dir / f"thor_smoke_crop{k}_r{r0}_c{c0}.png"
        save_smoke_figure(db, panels, crop_lab_rc, crop_ver, crop_unk_rc, p.score_threshold,
                          f"Færch 2023 EW 40 m, crop {k}: rows {r0}:{r0 + crop_px}, cols "
                          f"{c0}:{c0 + crop_px} ({p.crop_m / 1000:g} km), {int(crop_ver.sum())} "
                          f"verified + {int((~crop_ver).sum())} interpolated labels", fig)
        out[f"crop{k}"] = fig

    table = pd.DataFrame(summary)
    csv = p.out_dir / "thor_smoke.csv"
    table.to_csv(csv, index=False)
    meta = {"head": str(p.head), "crop_m": p.crop_m, "crop_px": crop_px, "pixel_m": pixel_m,
            "variants": {v.name: {"band_gsd_m": v.band_gsd_m, "patch_px": v.patch_px,
                                  "token_m": geo[v.name][1] * pixel_m} for v in p.variants},
            "norm": "IW (as trained)", "score_threshold": p.score_threshold, "device": device,
            "crops": [{"row": r, "col": c, "verified": n} for r, c, n in crops]}
    js = p.out_dir / "thor_smoke.json"
    js.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out |= {"table": csv, "meta": js}
    return out


def _r(v: float) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 3)


def save_smoke_figure(db: dict[str, dict[str, np.ndarray]], panels: dict[str, dict[str, Any]],
                      lab_rc: np.ndarray, ver: np.ndarray, unk_rc: np.ndarray, thr: float,
                      title: str, path: Path) -> None:
    """Rows = stacks; columns = HH, HV (dB), then PCA RGB and heatmap per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stacks = list(db)
    variants = list(dict.fromkeys(k.split("/")[1] for k in panels))
    ncol = 2 + 2 * len(variants)
    fig, axes = plt.subplots(len(stacks), ncol, figsize=(3.3 * ncol, 3.6 * len(stacks)),
                             constrained_layout=True, squeeze=False)
    crop = next(iter(db[stacks[0]].values())).shape[0]
    ext = (-0.5, crop - 0.5, crop - 0.5, -0.5)

    def marks(ax: Any) -> None:
        if ver.any():
            ax.scatter(lab_rc[ver, 1], lab_rc[ver, 0], s=70, facecolors="none",
                       edgecolors="lime", linewidths=1.1)
        if (~ver).any():
            ax.scatter(lab_rc[~ver, 1], lab_rc[~ver, 0], s=40, marker="x", c="orange",
                       linewidths=1.1)
        if len(unk_rc):
            ax.scatter(unk_rc[:, 1], unk_rc[:, 0], s=40, marker="+", c="cyan", linewidths=1.1)

    for r, s in enumerate(stacks):
        for c, (b, im) in enumerate(db[s].items()):
            ax = axes[r, c]
            vmin, vmax = np.nanpercentile(im, [2, 99.5])
            ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
            marks(ax)
            ax.set_title(f"{s}: {b} sigma0 dB [{vmin:.0f}, {vmax:.0f}]", fontsize=9)
        for i, v in enumerate(variants):
            pan = panels[f"{s}/{v}"]
            ax = axes[r, 2 + 2 * i]
            ax.imshow(pan["rgb"], extent=ext, interpolation="nearest")
            ax.set_title(f"{s} / {v}: PCA 1-3", fontsize=9)
            ax = axes[r, 3 + 2 * i]
            ax.imshow(pan["prob"], extent=ext, interpolation="nearest", cmap="magma",
                      vmin=0, vmax=1)
            marks(ax)
            pk = pan["peaks"]
            if len(pk):
                ax.scatter(pk[:, 1], pk[:, 0], s=12, marker="s", c="red")
            ax.set_title(f"{s} / {v}: head score (peaks >= {thr:g} red)", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title + "\nlabels: verified (green o), interpolated (orange x), unknown (cyan +)",
                 fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --- bench-thor: whole scoring area, F vs score threshold -------------------------------------

STACKS = ("TNR", "noTNR")
# Categorical slots 1-3 of the dataviz palette (validated all-pairs, light mode).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


@dataclass(frozen=True)
class RunSpec:
    variant: str
    stack: str            # TNR | noTNR
    norm: str = "iw"      # key of ThorRunParams.norms

    @property
    def series(self) -> str:
        """Curve name: the variant, plus the normalisation when it isn't the trained one."""
        return self.variant if self.norm == "iw" else f"{self.variant}, {self.norm} norm"

    @property
    def tag(self) -> str:
        n = "" if self.norm == "iw" else f"-{self.norm}norm"
        return f"thor-{self.variant}{n}_{self.stack}"


@dataclass(frozen=True)
class ThorRunParams:
    out_dir: Path
    overlap_m: float
    score_min: float
    thresholds: tuple[float, ...]
    norms: dict[str, dict[str, tuple[float, float]]]
    runs: tuple[RunSpec, ...]   # every variant x stack with `iw`, then `run.extra`
    figure: Path
    report: Path

    @classmethod
    def from_config(cls, cfg: Config) -> ThorRunParams:
        t = cfg.section("benchmark")["faerch2023"]["thor"]
        r = t["run"]
        th = tuple(sorted(float(x) for x in r["thresholds"]))
        if th[0] < float(r["score_min"]):
            raise ValueError("thor.run.thresholds must be >= thor.run.score_min")
        norms = {"iw": ThorParams.from_config(cfg).norm} | {
            k: {b: (float(m), float(s)) for b, (m, s) in v.items()}
            for k, v in t.get("norms", {}).items()}
        runs = [RunSpec(v, s) for v in t["variants"] for s in STACKS]
        runs += [RunSpec(e["variant"], e["stack"], e.get("norm", "iw")) for e in r.get("extra", [])]
        for s in runs:
            if s.variant not in t["variants"] or s.stack not in STACKS or s.norm not in norms:
                raise ValueError(f"thor.run.extra: unknown variant / stack / norm in {s}")
        return cls(out_dir=cfg.root / r["out_dir"], overlap_m=float(r["overlap_m"]),
                   score_min=float(r["score_min"]), thresholds=th, norms=norms,
                   runs=tuple(dict.fromkeys(runs)),
                   figure=cfg.root / r["figure"], report=cfg.root / r["report"])


def aoi_window(transform: rasterio.Affine, shape: tuple[int, int], aoi: Any, tok_px: int
               ) -> tuple[int, int, int, int]:
    """(row0, col0, height, width) of the scoring area's bounding box, on the token grid."""
    x0, y0, x1, y1 = aoi.bounds
    c0, r0 = ~transform * (x0, y1)
    c1, r1 = ~transform * (x1, y0)
    r0 = max(int(np.floor(r0)) // tok_px * tok_px, 0)
    c0 = max(int(np.floor(c0)) // tok_px * tok_px, 0)
    r1, c1 = min(int(np.ceil(r1)), shape[0]), min(int(np.ceil(c1)), shape[1])
    return r0, c0, r1 - r0, c1 - c0


def scene_peaks(sp: Any, v: Variant, encoder: Any, head: Any, cfg: Config, aoi: Any,
                device: str, log: Callable[[str], None],
                norm: dict[str, tuple[float, float]] | None = None,
                ) -> tuple[pd.DataFrame, np.ndarray, rasterio.Affine]:
    """Sliding-window heatmap over the scoring area's bounding box -> peaks >= score_min.

    As `thor.infer.heatmap_scene`: windows on the token grid, each token from the nearest
    window centre, peak moved to the brightest HH pixel of its token. Returns peaks (row, col
    in stack pixels, score), the token heatmap and its transform.
    """
    import torch
    from shapely.prepared import prep

    from iceberg_sar.thor.infer import InferParams, kept_ranges, window_starts

    p, rp = ThorFaerchParams.from_config(cfg), ThorRunParams.from_config(cfg)
    tp, fp, ip = ThorParams.from_config(cfg), FeatureParams.from_config(cfg), \
        InferParams.from_config(cfg)
    if norm is not None:
        tp = replace(tp, norm=norm)
    with rasterio.open(sp.stack) as src:
        transform, shape = src.transform, (src.height, src.width)
        names = list(src.descriptions)
    pixel_m = abs(transform.a)
    crop_px, tok_px = crop_geometry(p.crop_m, pixel_m, v)
    overlap_px = int(round(rp.overlap_m / pixel_m))
    grid = crop_px // tok_px
    R0, C0, H, W = aoi_window(transform, shape, aoi, tok_px)
    rs = window_starts(H, crop_px, overlap_px, tok_px)
    cs = window_starts(W, crop_px, overlap_px, tok_px)
    rk = kept_ranges([r // tok_px for r in rs], grid)
    ck = kept_ranges([c // tok_px for c in cs], grid)
    gh, gw = rk[-1][1], ck[-1][1]
    prob = np.full((gh, gw), np.nan, np.float32)
    valid = np.zeros((gh, gw), bool)
    peak_r = np.zeros((gh, gw), np.int32)
    peak_c = np.zeros((gh, gw), np.int32)
    area = prep(aoi)

    def in_area(r: int, c: int) -> bool:
        x0, y0 = transform * (C0 + c, R0 + r + crop_px)
        x1, y1 = transform * (C0 + c + crop_px, R0 + r)
        return area.intersects(box(x0, y0, x1, y1))

    jobs = [(i, j) for i in range(len(rs)) for j in range(len(cs)) if in_area(rs[i], cs[j])]
    log(f"  {len(jobs)} windows in the scoring area ({len(rs)} x {len(cs)} in its box)")
    hh = tp.bands.index("HH")
    pending: list[tuple[int, int, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]] = []

    def flush() -> None:
        if not pending:
            return
        x = np.stack([q[2] for q in pending])
        with torch.no_grad():
            f = torch.from_numpy(embed_blocks(encoder, x, fp.blocks, device)).to(device)
            pb = torch.sigmoid(head(f))[:, 0].float().cpu().numpy()
        for (i, j, _, tok_valid, (ar, ac)), pr in zip(pending, pb, strict=True):
            (lo_r, hi_r), (lo_c, hi_c) = rk[i], ck[j]
            t0r, t0c = rs[i] // tok_px, cs[j] // tok_px
            sl = (slice(lo_r - t0r, hi_r - t0r), slice(lo_c - t0c, hi_c - t0c))
            dst = (slice(lo_r, hi_r), slice(lo_c, hi_c))
            prob[dst] = pr[sl]
            valid[dst] = tok_valid[sl]
            peak_r[dst] = ar[sl] + rs[i] + R0
            peak_c[dst] = ac[sl] + cs[j] + C0
        pending.clear()

    with rasterio.open(sp.stack) as stk, rasterio.open(sp.valid) as vld:
        for n, (i, j) in enumerate(jobs):
            win = Window(C0 + cs[j], R0 + rs[i], crop_px, crop_px)
            sea = vld.read(1, window=win).astype(bool)
            if sea.mean() < ip.min_valid:
                continue
            bands = {d: stk.read(k + 1, window=win).astype(np.float32)
                     for k, d in enumerate(names)}
            for b in bands.values():
                b[~sea | ~np.isfinite(b)] = np.nan
            x = to_model_input(bands, None, tp, radiometry="sigma0")
            tok_valid = pool(sea.astype(float), tok_px) >= p.token_min_valid
            pending.append((i, j, x, tok_valid, token_argmax(bands[tp.bands[hh]], tok_px)))
            if len(pending) == ip.batch:
                flush()
            if (n + 1) % 1000 == 0:
                log(f"  {n + 1} / {len(jobs)} windows")
        flush()

    ij, scores = extract_peaks(prob, rp.score_min, valid)
    peaks = pd.DataFrame({"row": peak_r[ij[:, 0], ij[:, 1]], "col": peak_c[ij[:, 0], ij[:, 1]],
                          "score": scores.astype(float)})
    grid_transform = transform * rasterio.Affine.translation(C0, R0) \
        * rasterio.Affine.scale(tok_px)
    return peaks, prob, grid_transform


def run_scenes(cfg: Config, device: str | None = None, force: bool = False,
               log: Callable[[str], None] = print) -> dict[RunSpec, Path]:
    """Peaks GeoJSON (+ heatmap GeoTIFF) per run (`ThorRunParams.runs`); finished ones are kept."""
    import geopandas as gpd
    import torch

    from iceberg_sar.thor.train import load_head

    p, rp = ThorFaerchParams.from_config(cfg), ThorRunParams.from_config(cfg)
    tp = ThorParams.from_config(cfg)
    variants = {v.name: v for v in p.variants}
    fc = cfg.section("benchmark")["faerch2023"]
    if not p.head.is_file():
        raise FileNotFoundError(f"{p.head} missing; run `thor-fetch-head`")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    stacks = {_variant(c): scene_paths(c) for c in (cfg, _other_variant(cfg))}
    aoi = scoring_aoi(cfg, fc["score"]["aoi"])
    rp.out_dir.mkdir(parents=True, exist_ok=True)
    head = None
    encoders: dict[str, Any] = {}
    out: dict[RunSpec, Path] = {}
    for run in rp.runs:
        v, sp, tag = variants[run.variant], stacks[run.stack], run.tag
        gj = rp.out_dir / f"{tag}.geojson"
        out[run] = gj
        if gj.is_file() and not force:
            log(f"{tag}: done ({gj.name})")
            continue
        if not sp.stack.is_file():
            log(f"{tag}: skipped, {sp.stack.name} missing")
            continue
        with rasterio.open(sp.stack) as src:
            pixel_m, crs, transform = abs(src.transform.a), src.crs, src.transform
        if head is None:
            head = load_head(p.head, device)
        if v.name not in encoders:
            crop_px, _ = crop_geometry(p.crop_m, pixel_m, v)
            encoders[v.name] = load_encoder(replace(tp, crop_px=crop_px, patch_px=v.patch_px),
                                            pixel_m, device, v.band_gsd_m)
        log(f"{tag}: {sp.stack.name}, {run.norm} normalisation")
        t0 = pd.Timestamp.now()
        peaks, prob, grid_transform = scene_peaks(sp, v, encoders[v.name], head, cfg, aoi,
                                                  device, log, rp.norms[run.norm])
        xs, ys = transform * (peaks.col.to_numpy() + 0.5, peaks.row.to_numpy() + 0.5)
        gdf = gpd.GeoDataFrame(peaks.assign(detector=tag), crs=crs,
                               geometry=gpd.points_from_xy(xs, ys))
        with rasterio.open(rp.out_dir / f"{tag}_heatmap.tif", "w", driver="GTiff",
                           height=prob.shape[0], width=prob.shape[1], count=1,
                           dtype="float32", crs=crs, transform=grid_transform,
                           nodata=np.nan, compress="deflate") as dst:
            dst.write(prob, 1)
        gj.write_text(gdf.to_json(drop_id=True), encoding="utf-8")
        secs = (pd.Timestamp.now() - t0).total_seconds()
        log(f"{tag}: {len(gdf)} peaks >= {rp.score_min:g} in {secs / 60:.1f} min")
    return out


def score_thresholds(peaks: Any, labels: pd.DataFrame, unknown: np.ndarray, domain: Any,
                     ignore_m: float, thresholds: tuple[float, ...], matcher: str = "greedy"
                     ) -> pd.DataFrame:
    """`score_points` per threshold, on peaks in the scoring domain and away from unknown
    objects; `labels` are taken into the same domain (revision C3)."""
    labels = labels_in_domain(labels, domain)
    xy = np.column_stack([peaks.geometry.x, peaks.geometry.y]).reshape(-1, 2)
    inside = domain.keep(xy)
    peaks, xy = peaks[inside], xy[inside]
    keep = np.ones(len(xy), bool)
    if len(unknown) and len(xy):
        keep = cKDTree(unknown).query(xy, distance_upper_bound=ignore_m)[0] > ignore_m
    s = peaks.score.to_numpy()
    rows = []
    for t in thresholds:
        sel = keep & (s >= t)
        rows.append({"threshold": t, "ignored_unknown": int((~keep & (s >= t)).sum()),
                     "labels_scored": len(labels), **score_points(xy[sel], labels, matcher)})
    return pd.DataFrame(rows)


def run_thor(cfg: Config, device: str | None = None, force: bool = False,
             log: Callable[[str], None] = print) -> dict[str, Path]:
    import geopandas as gpd

    p, rp = ThorFaerchParams.from_config(cfg), ThorRunParams.from_config(cfg)
    fc = cfg.section("benchmark")["faerch2023"]
    runs = run_scenes(cfg, device, force, log)
    labels, unknown = load_labels(cfg, "faerch")
    domain = eval_domain(cfg, fc["score"]["aoi"])
    ms = MatchSettings.from_config(cfg)
    labels = apply_match_settings(labels, ms)
    n_scored = len(labels_in_domain(labels, domain))
    tables = []
    for run, gj in runs.items():
        if not gj.is_file():
            continue
        peaks = gpd.read_file(gj).to_crs(fc["crs"])
        t = score_thresholds(peaks, labels, unknown, domain, p.unknown_ignore_m, rp.thresholds,
                             ms.matcher)
        tables.append(t.assign(detector="thor", variant=run.variant, stack=run.stack,
                               norm=run.norm, series=run.series))
    scores = pd.concat(tables, ignore_index=True)
    csv = rp.out_dir / "thor_scores.csv"
    scores.to_csv(csv, index=False)
    refs = reference_optima(cfg)
    order = list(dict.fromkeys(r.series for r in runs))   # config order -> fixed colour slots
    fig = thor_f_figure(scores, refs, rp.figure, p.score_threshold, order)
    rp.report.write_text(thor_report(cfg, scores, refs, fig, csv, n_scored, order),
                         encoding="utf-8")
    for run in runs:
        d = scores[(scores["stack"] == run.stack) & (scores.series == run.series)]
        if d.empty:
            continue
        b = d.loc[d.f.idxmax()]
        log(f"{run.stack:5s} {run.series:20s}: best F {b.f:.3f} (verified recall "
            f"{b.recall_v:.3f}) at "
            f"{b.threshold:g}, P {b.precision:.2f} R {b.recall:.2f}, {int(b.n_det)} detections")
    return {"scores": csv, "figure": fig, "report": rp.report}


def reference_optima(cfg: Config) -> pd.DataFrame:
    """Best F per CFAR detector and stack, from `bench-score`'s per-run CSVs (empty if none)."""
    from iceberg_sar.benchmark.score import optimum

    base = cfg.root / cfg.section("benchmark")["faerch2023"]["score"]["table_csv"]
    rows = []
    for stack in ("TNR", "noTNR"):
        f = base.with_name(f"{base.stem}_{stack}.csv")
        if f.is_file():
            rows.append(optimum(pd.read_csv(f)).reset_index().assign(stack=stack))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def thor_f_figure(scores: pd.DataFrame, refs: pd.DataFrame, out: Path, view_score: float,
                  series: list[str]) -> Path:
    """One panel per stack: F vs threshold per run series (solid all labels, dotted verified
    only), the best published CFAR and my CA-CFAR on that stack as reference lines. Colours
    follow `series` order (config order), never rank."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iceberg_sar.benchmark.score import DISPLAY, GREY, INK, MINE

    if len(series) > len(SERIES_COLORS):
        raise ValueError(f"{len(series)} THOR series; the palette validates {len(SERIES_COLORS)}")
    colors = dict(zip(series, SERIES_COLORS, strict=False))
    stacks = [s for s in ("noTNR", "TNR") if s in set(scores["stack"])]
    fig, axes = plt.subplots(1, len(stacks), figsize=(6.2 * len(stacks), 4.6), sharey=True,
                             squeeze=False, constrained_layout=True)
    for ax, stack in zip(axes[0], stacks, strict=True):
        for name in series:
            d = scores[(scores["stack"] == stack) & (scores.series == name)]
            if d.empty:
                continue
            d = d.sort_values("threshold")
            c = colors[name]
            ax.plot(d.threshold, d.f, color=c, lw=2, marker="o", ms=4,
                    label=f"{PIPELINE_SHORT} {name}, F (all labels)")
            ax.plot(d.threshold, d.recall_v, color=c, lw=1.5, ls=":",
                    label=f"{PIPELINE_SHORT} {name}, verified-only recall")
        if not refs.empty:
            r = refs[refs["stack"] == stack]
            for sub, color in ((r[~r.detector.isin(list(MINE))], INK),
                               (r[r.detector.isin(list(MINE))], GREY)):
                if sub.empty:
                    continue
                b = sub.loc[sub.f.idxmax()]
                ax.axhline(b.f, color=color, lw=1.2, ls="--",
                           label=f"best {DISPLAY.get(b.detector, b.detector)}: F {b.f:.2f}")
            refs_h = [h for h, lab in zip(*ax.get_legend_handles_labels(), strict=True)
                      if lab.startswith("best ")]
            if refs_h:   # top right stays clear of the curves (F < 0.8 there)
                # an artist of its own, so the shared THOR legend below doesn't replace it
                ax.add_artist(ax.legend(handles=refs_h, fontsize=8, frameon=False,
                                        loc="upper right"))
        ax.axvline(view_score, color=GREY, lw=0.8)
        ax.annotate("viewer threshold", (view_score, 0.02), xytext=(3, 0),
                    textcoords="offset points", fontsize=8, color=INK, rotation=90, va="bottom")
        removal = "with" if stack == "TNR" else "no"
        ax.set_title(f"{stack} stack ({removal} thermal noise removal)", fontsize=10)
        ax.set_xlabel(f"{PIPELINE_SHORT}: head score threshold")
        ax.set_ylim(0, 1)
        ax.grid(color="#e4e4e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel("F (Færch labels, scoring area fig1)")
    handles: dict[str, Any] = {}   # every panel's series (a run may exist on one stack only)
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab.startswith(PIPELINE_SHORT):
                handles.setdefault(lab, h)
    axes[0, -1].legend(handles.values(), handles.keys(), fontsize=8, frameon=False,
                       loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"{PIPELINE} on the Færch 2023 EW scene, 40 m (head unchanged)",
                 fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def thor_report(cfg: Config, scores: pd.DataFrame, refs: pd.DataFrame, fig: Path, csv: Path,
                n_labels: int, series: list[str]) -> str:
    from iceberg_sar.benchmark.score import DISPLAY, MINE

    p = ThorFaerchParams.from_config(cfg)

    def rel(x: Path) -> str:
        return x.relative_to(cfg.root).as_posix()

    def f3(v: float) -> str:
        return "" if pd.isna(v) else f"{v:.3f}"

    head = ("| stack | {} | {} | detections | P | R | F | R verified | R small | R medium | "
            "R large |")
    rule = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [
        f"# {PIPELINE} on the Færch 2023 scene (Sentinel-1 EW, 40 m)", "",
        "Generated by `bench-thor`; don't edit by hand.", "",
        "## Setup", "",
        f"- Head: the final linear head (`{p.head.parent.parent.name}`), trained on 10 m HyP3 IW "
        "frames off Newfoundland, April-June. Nothing retrained or tuned on this scene.",
        f"- Crops of {p.crop_m / 1000:g} km (128 px), 64 x 64 tokens of 80 m, as in training. "
        "`upsample`: bands declared 10 m, THOR resizes the crop to 512 px; `native`: bands "
        "declared 40 m, 2 x 2 px patches.",
        "- SNAP sigma0 (no gamma0 conversion). Normalisation: THOR's IW stats (what the head was "
        "trained with) unless the run says `ew norm` (THOR's own EW HH/HV stats). Land mask "
        "only (no pack-ice mask). Peaks: 3 x 3 token maxima, moved to the brightest HH pixel.",
        f"- Scoring as `bench-score`: {n_labels} Færch labels in the scoring area `fig1` (one "
        "shared evaluation mask for labels and detections), 250 m verified / 1789 m "
        "interpolated, detections near unknown objects ignored. The score threshold is swept "
        "the way the CFARs' PFA is.", "",
        "## Best F per stack and run", "",
        head.format("run", "threshold"), rule,
    ]
    for stack in STACKS:
        for name in series:
            d = scores[(scores["stack"] == stack) & (scores.series == name)]
            if d.empty:
                continue
            b = d.loc[d.f.idxmax()]
            lines.append(
                f"| {stack} | {name} | {b.threshold:g} | {int(b.n_det)} | "
                f"{f3(b.precision)} | {f3(b.recall)} | **{f3(b.f)}** | {f3(b.recall_v)} | "
                f"{f3(b.recall_small)} | {f3(b.recall_medium)} | {f3(b.recall_large)} |")
    if not refs.empty:
        lines += ["", "## CFAR detectors on the same stacks (`bench-score`, best F over PFA)", "",
                  head.format("detector", "PFA"), rule]
        for r in refs.sort_values(["stack", "f"], ascending=[True, False]).itertuples():
            name = DISPLAY.get(r.detector, r.detector) + (" (mine)" if r.detector in MINE else "")
            lines.append(f"| {r.stack} | {name} | {r.pfa:.0e} | {int(r.n_det)} | "
                         f"{f3(r.precision)} | {f3(r.recall)} | {f3(r.f)} | {f3(r.recall_v)} | "
                         f"{f3(r.recall_small)} | {f3(r.recall_medium)} | {f3(r.recall_large)} |")
    lines += ["", f"![F vs threshold](../{rel(fig)})", "",
              "## Caveats", "",
              "- One scene, July 2019, about 58° N; the head never saw EW, 40 m or July.",
              "- The threshold is chosen on this scene (best F), as the CFARs' PFA is in "
              "`bench-score`; neither is a held-out operating point.", "",
              "## Files", "", f"- scores per threshold: `{rel(csv)}` (local)", ""]
    return "\n".join(lines)
