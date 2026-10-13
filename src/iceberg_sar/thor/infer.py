"""Full-scene THOR detection: sliding windows -> stitched token heatmap -> peaks -> GeoJSON.

Windows of `crop_px` overlap by `overlap_px` and start on the token grid. Each scene token is
taken from the window whose centre is nearest (window k keeps the tokens between the midpoints
to its neighbours), so a berg near a window border is scored with context on both sides and
counted once. Peaks are 3x3 local maxima of the stitched heatmap.

Each peak moves from its token centre to the brightest HH pixel in the token (the labels were
snapped to HH peaks the same way), then has to pass the CFAR's own mask (land + buffer, pack
ice + buffer, outside the AOI) so both detectors see the same water. Attributes follow the
CFAR schema; the blob is grown around the peak like `detections.bright_structures`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from iceberg_sar.config import Config, metres_to_px
from iceberg_sar.detections import COPOL, to_db, to_geodataframe
from iceberg_sar.preprocess import find_rtc_bands, product_radiometry
from iceberg_sar.thor.chips import ensure_scene_masks, read_crop
from iceberg_sar.thor.encoder import ThorParams, embed_blocks, load_encoder, to_model_input
from iceberg_sar.thor.features import FeatureParams
from iceberg_sar.thor.heads import extract_peaks
from iceberg_sar.thor.targets import pool_tokens


@dataclass(frozen=True)
class InferParams:
    overlap_px: int
    score_min: float
    min_valid: float          # skip windows with less sea than this
    token_min_valid: float    # tokens with less sea than this can't hold a peak (train.min_valid)
    batch: int
    blob_window_m: float
    grow_db: float

    @classmethod
    def from_config(cls, cfg: Config) -> InferParams:
        t = cfg.section("thor")
        i = t["infer"]
        return cls(overlap_px=int(i["overlap_px"]), score_min=float(i["score_min"]),
                   min_valid=float(i["min_valid"]),
                   token_min_valid=float(t["train"]["min_valid"]), batch=int(i["batch"]),
                   blob_window_m=float(i["blob_window_m"]), grow_db=float(i["grow_db"]))


def window_starts(n_px: int, crop_px: int, overlap_px: int, patch_px: int) -> list[int]:
    """Window starts along one axis: on the token grid, the last one flush with the end
    (rounded down to the grid, so up to patch_px - 1 trailing pixels go unscored)."""
    if n_px < crop_px:
        raise ValueError(f"Scene axis ({n_px} px) is shorter than one window ({crop_px} px)")
    stride = crop_px - overlap_px
    if stride <= 0 or stride % patch_px or overlap_px % patch_px:
        raise ValueError("overlap_px and crop_px - overlap_px must be multiples of patch_px")
    last = (n_px - crop_px) // patch_px * patch_px
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def kept_ranges(starts_tok: list[int], grid: int) -> list[tuple[int, int]]:
    """Per window, the [lo, hi) scene tokens it owns: halfway to each neighbour's centre."""
    pairs = zip(starts_tok, starts_tok[1:], strict=False)
    bounds = [0] + [(a + b + grid) // 2 for a, b in pairs] + [starts_tok[-1] + grid]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def token_argmax(img: np.ndarray, patch_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Per token, (row, col) offset inside the crop of its brightest finite pixel."""
    g0, g1 = img.shape[0] // patch_px, img.shape[1] // patch_px
    t = np.nan_to_num(img[: g0 * patch_px, : g1 * patch_px], nan=-np.inf)
    t = t.reshape(g0, patch_px, g1, patch_px).transpose(0, 2, 1, 3).reshape(g0, g1, -1)
    k = t.argmax(axis=2)
    rows = np.arange(g0)[:, None] * patch_px + k // patch_px
    cols = np.arange(g1)[None, :] * patch_px + k % patch_px
    return rows, cols


def heatmap_scene(
    product_dir: Path, cfg: Config, head, device: str = "cuda", log=print,
) -> dict[str, np.ndarray | rasterio.Affine | object]:
    """Stitched token heatmap and per-token brightest-HH pixel for a whole scene."""
    import torch

    tp, fp, ip = ThorParams.from_config(cfg), FeatureParams.from_config(cfg), \
        InferParams.from_config(cfg)
    name = product_dir.name
    interim = cfg.path("interim") / name
    radiometry = product_radiometry(name)
    ensure_scene_masks(product_dir, cfg)
    with rasterio.open(find_rtc_bands(product_dir)[tp.bands[0]]) as src:
        h, w, transform, crs = src.height, src.width, src.transform, src.crs
        pixel_m = abs(transform.a)
    g = tp.grid_px()
    rs = window_starts(h, tp.crop_px, ip.overlap_px, tp.patch_px)
    cs = window_starts(w, tp.crop_px, ip.overlap_px, tp.patch_px)
    rk = kept_ranges([r // tp.patch_px for r in rs], g)
    ck = kept_ranges([c // tp.patch_px for c in cs], g)
    gh, gw = rk[-1][1], ck[-1][1]
    prob = np.full((gh, gw), np.nan, np.float32)
    valid = np.zeros((gh, gw), bool)
    peak_r = np.zeros((gh, gw), np.int32)
    peak_c = np.zeros((gh, gw), np.int32)

    encoder = load_encoder(tp, pixel_m, device)
    head.eval()
    jobs = [(i, j) for i in range(len(rs)) for j in range(len(cs))]
    log(f"{name}: {len(jobs)} windows ({len(rs)} x {len(cs)}), grid {gh} x {gw} tokens")
    hh = tp.bands.index("HH") if "HH" in tp.bands else 0
    pending: list[tuple[int, int, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]] = []

    def flush() -> None:
        if not pending:
            return
        x = np.stack([p[2] for p in pending])
        with torch.no_grad():
            f = torch.from_numpy(embed_blocks(encoder, x, fp.blocks, device)).to(device)
            pb = torch.sigmoid(head(f))[:, 0].float().cpu().numpy()
        for (i, j, _, tok_valid, (ar, ac)), pr in zip(pending, pb, strict=True):
            (lo_r, hi_r), (lo_c, hi_c) = rk[i], ck[j]
            t0r, t0c = rs[i] // tp.patch_px, cs[j] // tp.patch_px
            sl = (slice(lo_r - t0r, hi_r - t0r), slice(lo_c - t0c, hi_c - t0c))
            dst = (slice(lo_r, hi_r), slice(lo_c, hi_c))
            prob[dst] = pr[sl]
            valid[dst] = tok_valid[sl]
            peak_r[dst] = ar[sl] + rs[i]
            peak_c[dst] = ac[sl] + cs[j]
        pending.clear()

    for n, (i, j) in enumerate(jobs):
        bands, inc_deg, _, land, _ = read_crop(product_dir, interim, rs[i], cs[j], tp)
        if (~land).mean() < ip.min_valid:
            continue
        x = to_model_input(bands, inc_deg, tp, radiometry)
        tok_valid = pool_tokens(~land, tp.patch_px) >= ip.token_min_valid
        pending.append((i, j, x, tok_valid, token_argmax(bands[tp.bands[hh]], tp.patch_px)))
        if len(pending) == ip.batch:
            flush()
        if (n + 1) % 500 == 0:
            log(f"  {n + 1}/{len(jobs)} windows")
    flush()
    return {"prob": prob, "valid": valid, "peak_r": peak_r, "peak_c": peak_c,
            "transform": transform, "crs": crs, "pixel_m": pixel_m}


def peak_attributes(
    rc: np.ndarray, masked: dict[str, Path], cfar_band: str, half_px: int, grow_db: float,
) -> pd.DataFrame:
    """CFAR-schema attributes around each peak pixel, from the masked (CFAR-input) rasters.

    Background = median of the CFAR band in the window; blob = pixels at least `grow_db`
    above it, 8-connected to the peak (at least the peak pixel itself).
    """
    srcs = {pol: rasterio.open(p) for pol, p in masked.items()}
    copol = COPOL.get(cfar_band)
    rows = []
    try:
        for r, c in rc.astype(int):
            win = Window(c - half_px, r - half_px, 2 * half_px + 1, 2 * half_px + 1)
            b = {pol: s.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                 for pol, s in srcs.items()}
            x = b[cfar_band]
            bg = float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
            above = np.isfinite(x) & (x > bg * 10 ** (grow_db / 10))
            above[half_px, half_px] = True
            lab, _ = ndimage.label(above, structure=np.ones((3, 3)))
            blob = lab == lab[half_px, half_px]
            ys, xs = np.nonzero(blob)
            out = {"row": float(r), "col": float(c), "area_px": int(blob.sum()),
                   "extent_px": int(max(np.ptp(ys), np.ptp(xs)) + 1),
                   "background_db": float(to_db(bg))}
            out["structure_px"] = out["extent_px"]
            for pol, arr in b.items():
                vals = arr[blob]
                vals = vals[np.isfinite(vals)]
                out[f"peak_db_{pol}"] = float(to_db(vals.max())) if len(vals) else np.nan
                out[f"mean_db_{pol}"] = float(to_db(vals.mean())) if len(vals) else np.nan
            out["contrast_db"] = out[f"peak_db_{cfar_band}"] - out["background_db"]
            if copol in b and np.isfinite(b[copol]).any():
                out["copol_contrast_db"] = out[f"peak_db_{copol}"] - float(
                    to_db(np.nanmedian(b[copol])))
            rows.append(out)
    finally:
        for s in srcs.values():
            s.close()
    return pd.DataFrame(rows)


def detect_scene(
    product_dir: Path, cfg: Config, head_path: Path, device: str = "cuda", log=print,
    tag: str = "thor",
) -> dict[str, Path]:
    """THOR detections for one preprocessed scene: GeoJSON (CFAR schema + detector, model,
    score), token heatmap GeoTIFF and a summary JSON in <outputs>/detections, named by `tag`
    (e.g. thor-linear) so several heads can sit side by side."""
    from iceberg_sar.thor.train import load_head

    product_dir = Path(product_dir)
    name = product_dir.name
    tp, ip = ThorParams.from_config(cfg), InferParams.from_config(cfg)
    head = load_head(head_path, device)
    res = heatmap_scene(product_dir, cfg, head, device, log)
    prob, valid = res["prob"], res["valid"]

    ij, scores = extract_peaks(prob, ip.score_min, valid)
    rc = np.column_stack([res["peak_r"][ij[:, 0], ij[:, 1]],
                          res["peak_c"][ij[:, 0], ij[:, 1]]]) if len(ij) else np.empty((0, 2))

    interim = cfg.path("interim") / name
    cfar_band = cfg.section("cfar")["band"]
    masked = {pol: interim / f"{name}_{pol}_masked.tif" for pol in ("HH", "HV", "VV", "VH")
              if (interim / f"{name}_{pol}_masked.tif").is_file()}
    if cfar_band not in masked:
        raise FileNotFoundError(f"{name}: no {cfar_band} masked raster; run `preprocess` first")
    with rasterio.open(masked[cfar_band]) as src:
        in_mask = np.array([np.isfinite(v[0]) for v in src.sample(
            [src.xy(r, c) for r, c in rc])], bool) if len(rc) else np.zeros(0, bool)
    rc, scores, ij = rc[in_mask], scores[in_mask], ij[in_mask]

    half = metres_to_px(ip.blob_window_m / 2, res["pixel_m"])
    table = peak_attributes(rc, masked, cfar_band, half, ip.grow_db)
    if not table.empty:
        table["score"] = np.round(scores, 4)
        table["token_row"], table["token_col"] = ij[:, 0], ij[:, 1]
    inc = find_rtc_bands(product_dir).get("inc")
    ice_mask = interim / f"{name}_seaice_mask.tif"
    gdf = to_geodataframe(table, res["transform"], res["crs"], name, inc,
                          ice_mask if ice_mask.is_file() else None)
    if not gdf.empty:
        gdf.insert(5, "detector", "thor")
        gdf.insert(6, "model", tag)
        gdf = gdf.sort_values("score", ascending=False)

    out_dir = cfg.path("outputs") / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson = out_dir / f"{name}_{tag}_detections.geojson"
    if gdf.empty:
        geojson.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    else:
        gdf.to_file(geojson, driver="GeoJSON")

    heat_tif = out_dir / f"{name}_{tag}_heatmap.tif"
    t = res["transform"] * rasterio.Affine.scale(tp.patch_px)
    with rasterio.open(heat_tif, "w", driver="GTiff", height=prob.shape[0],
                       width=prob.shape[1], count=1, dtype="float32", crs=res["crs"],
                       transform=t, nodata=np.nan, compress="deflate") as dst:
        dst.write(np.where(valid, prob, np.nan).astype(np.float32), 1)

    summary = {"product": name, "detector": "thor", "model": tag, "head": str(head_path),
               "score_min": ip.score_min, "peaks": int(len(in_mask)),
               "outside_cfar_mask": int((~in_mask).sum()), "n_detections": len(gdf),
               "n_at_0.5": int((gdf["score"] >= 0.5).sum()) if len(gdf) else 0}
    summary_path = out_dir / f"{name}_{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"detections": geojson, "heatmap": heat_tif, "summary": summary_path}
