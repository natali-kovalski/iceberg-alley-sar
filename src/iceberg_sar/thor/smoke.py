"""Milestone 5a smoke test: does frozen THOR separate bergs and the ice edge from open water?

Embeds one crop, reduces the token features to 3 PCA components and saves them as RGB next
to the dB crops. Bands are read from the raw RTC product (the interim rasters have pack ice
masked out, and the ice edge is part of what we want to see); only land is set to NaN.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window

from iceberg_sar.config import Config
from iceberg_sar.preprocess import find_rtc_bands, linear_to_db, product_radiometry
from iceberg_sar.thor.encoder import (
    ThorParams,
    as_sigma0,
    embed,
    load_encoder,
    pca_rgb,
    to_model_input,
)

ICE_NODATA = 255


def pick_crop(
    rc: np.ndarray,
    ice: np.ndarray,
    shape: tuple[int, int],
    crop_px: int,
    ice_range: tuple[float, float],
) -> tuple[int, int, dict[str, float]]:
    """Top-left (row, col) of the crop with the most detections and some (not all) pack ice.

    `rc` is (N, 2) detection pixel coords; `ice` is the coarse 0/1/255 ice mask covering the
    full scene. Candidate crops are centred on detections. Falls back to the most detections
    if no candidate has an ice fraction in `ice_range`.
    """
    if len(rc) == 0:
        raise ValueError("No detections to centre a crop on")
    fy, fx = shape[0] / ice.shape[0], shape[1] / ice.shape[1]
    best: tuple[float, int, int, dict[str, float]] | None = None
    best_any: tuple[float, int, int, dict[str, float]] | None = None
    for r, c in rc:
        r0 = int(np.clip(round(r) - crop_px // 2, 0, shape[0] - crop_px))
        c0 = int(np.clip(round(c) - crop_px // 2, 0, shape[1] - crop_px))
        inside = ((rc[:, 0] >= r0) & (rc[:, 0] < r0 + crop_px)
                  & (rc[:, 1] >= c0) & (rc[:, 1] < c0 + crop_px))
        cells = ice[int(r0 / fy):int(np.ceil((r0 + crop_px) / fy)),
                    int(c0 / fx):int(np.ceil((c0 + crop_px) / fx))]
        known = cells[cells != ICE_NODATA]
        ice_frac = float((known == 1).mean()) if known.size else 0.0
        info = {"detections": int(inside.sum()), "ice_fraction": round(ice_frac, 3)}
        cand = (float(inside.sum()), r0, c0, info)
        if best_any is None or cand[0] > best_any[0]:
            best_any = cand
        if ice_range[0] <= ice_frac <= ice_range[1] and (best is None or cand[0] > best[0]):
            best = cand
    chosen = best or best_any
    assert chosen is not None
    return chosen[1], chosen[2], chosen[3]


def read_crop(
    product_dir: Path, interim_dir: Path, r0: int, c0: int, params: ThorParams
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, rasterio.Affine]:
    """Raw linear bands with land set to NaN, incidence (deg), ice and land masks for a crop."""
    name = product_dir.name
    paths = find_rtc_bands(product_dir)
    win = Window(c0, r0, params.crop_px, params.crop_px)
    with rasterio.open(paths["inc"]) as src:
        inc = src.read(1, window=win).astype(np.float32)
    inc = np.where(inc > 0, inc, np.nan)
    inc_deg = np.degrees(inc) if np.nanmax(inc) < 2 * np.pi else inc

    with rasterio.open(interim_dir / f"{name}_seaice_mask.tif") as src, \
            rasterio.open(paths[params.bands[0]]) as ref:
        transform = ref.window_transform(win)
        bounds = rasterio.windows.bounds(win, ref.transform)
        ice_win = src.window(*bounds)
        ice = src.read(1, window=ice_win, out_shape=(params.crop_px, params.crop_px),
                       resampling=rasterio.enums.Resampling.nearest, boundless=True,
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


def token_separation(
    feats: np.ndarray, det_rc: np.ndarray, ice: np.ndarray, land: np.ndarray, patch_px: int
) -> tuple[dict[str, float], np.ndarray]:
    """How far berg and ice tokens sit from open water, in units of water's own spread.

    Distance of each token to the open-water mean feature, divided by the median water-token
    distance. ~1 means "looks like water"; well above 1 means the encoder separates it.
    `berg_rank_pct`: share of water tokens closer to the water mean than the median berg token
    (100 = every berg is more anomalous than all water). Also returns the distance map.
    """
    d, h, w = feats.shape

    def pool(m: np.ndarray) -> np.ndarray:
        return m[: h * patch_px, : w * patch_px].reshape(h, patch_px, w, patch_px).mean((1, 3))

    ice_t, land_t = pool(ice.astype(float)) > 0.5, pool(land.astype(float)) > 0.5
    berg_t = np.zeros((h, w), bool)
    tr = np.clip((det_rc // patch_px).astype(int), 0, [h - 1, w - 1])
    berg_t[tr[:, 0], tr[:, 1]] = True
    near_berg = np.zeros_like(berg_t)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            near_berg |= np.roll(np.roll(berg_t, dr, 0), dc, 1)
    water = ~(ice_t | land_t | near_berg)
    flat = feats.reshape(d, -1).T
    mu = flat[water.reshape(-1)].mean(0)
    dist = np.linalg.norm(flat - mu, axis=1).reshape(h, w)
    dist /= float(np.median(dist[water]))
    out: dict[str, float] = {"water_tokens": int(water.sum()), "berg_tokens": int(berg_t.sum()),
                             "ice_tokens": int(ice_t.sum())}
    for key, m in (("berg", berg_t), ("ice", ice_t & ~land_t), ("water", water)):
        out[f"{key}_median_dist"] = round(float(np.median(dist[m])), 2) if m.any() else None
    if berg_t.any():
        out["berg_rank_pct"] = round(
            float((dist[water] < np.median(dist[berg_t])).mean() * 100), 1)
    dist[land_t] = np.nan
    return out, dist


def save_figure(
    bands_db: dict[str, np.ndarray], rgb: np.ndarray, dist: np.ndarray, det_rc: np.ndarray,
    ice: np.ndarray, title: str, path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(bands_db) + 2
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), constrained_layout=True)
    crop = next(iter(bands_db.values())).shape[0]
    for ax, (b, db) in zip(axes, bands_db.items(), strict=False):
        vmin, vmax = np.nanpercentile(db, [2, 99.5])
        ax.imshow(db, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{b} sigma0 (dB) [{vmin:.0f}, {vmax:.0f}]")
    extent = (-0.5, crop - 0.5, crop - 0.5, -0.5)
    axes[-2].imshow(rgb, extent=extent, interpolation="nearest")
    axes[-2].set_title("THOR tokens: PCA 1-3 as RGB")
    axes[-1].imshow(dist, extent=extent, interpolation="nearest", cmap="magma",
                    vmin=0.5, vmax=np.nanpercentile(dist, 99.5))
    if ice.any():
        axes[-1].contour(ice.astype(np.float32), levels=[0.5], colors="cyan", linewidths=1.2)
    axes[-1].scatter(det_rc[:, 1], det_rc[:, 0], s=90, facecolors="none", edgecolors="lime",
                     linewidths=1.2)
    axes[-1].set_title("distance to open water; CFAR (green), ice (cyan)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_smoke(product_dir: Path, cfg: Config, center_rc: tuple[int, int] | None = None,
              device: str | None = None) -> dict[str, Path]:
    import torch

    params = ThorParams.from_config(cfg)
    smoke = cfg.section("thor")["smoke"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = product_dir.name
    interim = cfg.path("interim") / name
    band = cfg.section("cfar")["band"]
    dets = gpd.read_file(cfg.path("outputs") / "detections" / f"{name}_{band}_detections.geojson")
    det_rc_all = dets[["row", "col"]].to_numpy(float)

    with rasterio.open(find_rtc_bands(product_dir)[params.bands[0]]) as src:
        shape, pixel_m = (src.height, src.width), float(src.res[0])
    if center_rc is not None:
        r0 = int(np.clip(center_rc[0] - params.crop_px // 2, 0, shape[0] - params.crop_px))
        c0 = int(np.clip(center_rc[1] - params.crop_px // 2, 0, shape[1] - params.crop_px))
        crop_info: dict[str, float] = {}
    else:
        with rasterio.open(interim / f"{name}_seaice_mask.tif") as src:
            ice_coarse = src.read(1)
        r0, c0, crop_info = pick_crop(det_rc_all, ice_coarse, shape, params.crop_px,
                                      tuple(smoke["ice_fraction_range"]))

    bands, inc_deg, ice, land, _ = read_crop(product_dir, interim, r0, c0, params)
    det_rc = det_rc_all - [r0, c0]
    det_rc = det_rc[((det_rc >= 0) & (det_rc < params.crop_px)).all(1)]

    radiometry = product_radiometry(name)
    x = to_model_input(bands, inc_deg, params, radiometry)
    model = load_encoder(params, pixel_m, device)
    feats = embed(model, x, device, params.layer)[0]
    g = params.grid_px()
    land_tokens = land[: g * params.patch_px, : g * params.patch_px].reshape(
        g, params.patch_px, g, params.patch_px).mean((1, 3)) > 0.5
    rgb, ratio = pca_rgb(feats, ~land_tokens)

    sigma0_db = {b: linear_to_db(as_sigma0(bands[b], inc_deg, params, radiometry))
                 for b in params.bands}
    separation, dist = token_separation(feats, det_rc, ice, land, params.patch_px)
    out_dir = cfg.path("outputs") / "thor"
    stem = f"smoke_{name}_r{r0}_c{c0}"
    fig_path = out_dir / f"{stem}.png"
    save_figure(sigma0_db, rgb, dist, det_rc, ice,
                f"{name}  rows {r0}:{r0 + params.crop_px}  cols {c0}:{c0 + params.crop_px}  "
                f"{params.model} patch {params.patch_px}", fig_path)

    summary = {
        "scene": name, "crop_row_col": [r0, c0], "crop_px": params.crop_px, "pixel_m": pixel_m,
        "device": device, "model": params.model, "patch_px": params.patch_px,
        "feature_shape": list(feats.shape),
        "pca_explained_variance": [round(float(v), 3) for v in ratio],
        "detections_in_crop": len(det_rc), **crop_info,
        "separation": separation,
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"figure": fig_path, "summary": json_path}
