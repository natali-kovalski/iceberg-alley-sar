"""Score a scene's detections: chips -> iceberg_prob per model variant -> GeoJSON + review PNG."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from iceberg_sar.chips import (  # noqa: E402
    ChipParams,
    chip_background_db,
    extract_chips,
    interp_curve,
)
from iceberg_sar.classify.predict import Ensemble, load_ensemble  # noqa: E402
from iceberg_sar.config import Config  # noqa: E402
from iceberg_sar.preprocess import product_radiometry  # noqa: E402


def chip_params(cfg: Config) -> ChipParams:
    c = cfg.section("chips")
    return ChipParams(
        size_px=int(c.get("size_px", 75)),
        to_sigma0=bool(c.get("to_sigma0", True)),
        hv_noise_floor=bool(c.get("hv_noise_floor", True)),
        noise_looks=float(c.get("noise_looks", 4.5)),
        seed=int(cfg.section("classifier").get("seed", 42)),
    )


def load_variants(cfg: Config) -> dict[str, Ensemble]:
    model_dir = cfg.path("models") / "classifier"
    names = list(cfg.section("classifier")["variants"])
    return {v: load_ensemble(model_dir / v) for v in names}


def save_chip_review(
    chips: np.ndarray, probs: np.ndarray, ids: list[str], png: Path, n: int = 8
) -> Path:
    """Most iceberg-like, most ship-like and most uncertain chips (HH top row, HV bottom row)."""
    order = np.argsort(probs)
    groups = {
        "most iceberg-like": order[::-1][:n],
        "most ship-like": order[:n],
        "most uncertain": np.argsort(np.abs(probs - 0.5))[:n],
    }
    fig, axes = plt.subplots(6, n, figsize=(1.6 * n, 10.5), squeeze=False)
    for g, (title, idx) in enumerate(groups.items()):
        for j in range(n):
            for b, (lo, hi) in enumerate(((-30, 5), (-35, -5))):
                ax = axes[2 * g + b, j]
                ax.set_xticks([])
                ax.set_yticks([])
                if j < len(idx):
                    i = idx[j]
                    ax.imshow(chips[i, b], cmap="gray", vmin=lo, vmax=hi)
                    if b == 0:
                        ax.set_title(f"{ids[i][-5:]}  p={probs[i]:.2f}", fontsize=8)
            axes[2 * g, 0].set_ylabel(f"{title}\nHH", fontsize=8)
            axes[2 * g + 1, 0].set_ylabel("HV", fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=80)
    plt.close(fig)
    return png


def classify_product(product_dir: Path, cfg: Config, band: str = "HV") -> dict[str, Path]:
    """Add iceberg_prob (first variant) and iceberg_prob_<variant> to a scene's detections."""
    from iceberg_sar.detections import sample_incidence_deg
    from iceberg_sar.preprocess import find_rtc_bands

    product_dir = Path(product_dir)
    name = product_dir.name
    det_dir = cfg.path("outputs") / "detections"
    det = gpd.read_file(det_dir / f"{name}_{band}_detections.geojson")
    bands = find_rtc_bands(product_dir)
    variants = load_variants(cfg)
    primary = next(iter(variants))
    curve = variants[primary].hv_background_curve

    inc = det["incidence_deg"].to_numpy(dtype=np.float64) if len(det) else np.empty(0)
    if np.isnan(inc).any() and "inc" in bands:
        utm = det.to_crs(_raster_crs(bands["HH"]))
        inc = np.where(np.isnan(inc), sample_incidence_deg(bands["inc"], utm.geometry.x.to_numpy(),
                                                           utm.geometry.y.to_numpy()), inc)
    params = chip_params(cfg)
    if product_radiometry(name) == "sigma0":  # already what Kaggle has
        params = dataclasses.replace(params, to_sigma0=False)
    chips, valid = extract_chips(bands, det["row"].to_numpy(), det["col"].to_numpy(), inc,
                                 params, curve)
    for v, ens in variants.items():
        p = ens.predict(chips, inc) if len(det) else np.empty(0)
        det[f"iceberg_prob_{v}"] = np.round(p, 3)
    det["iceberg_prob"] = det[f"iceberg_prob_{primary}"]
    bg = chip_background_db(chips) if len(det) else np.empty((0, 2))
    det["chip_valid_fraction"] = np.round(valid, 3)
    det["chip_bg_db_HH"] = np.round(bg[:, 0], 2)
    det["chip_bg_db_HV"] = np.round(bg[:, 1], 2)

    out = det_dir / f"{name}_{band}_classified.geojson"
    det.to_file(out, driver="GeoJSON")
    png = save_chip_review(chips, det["iceberg_prob"].to_numpy(), det["id"].tolist(),
                           det_dir / f"{name}_{band}_classified_chips.png")

    probs = {v: det[f"iceberg_prob_{v}"] for v in variants}
    summary = {
        "product": name,
        "n_detections": len(det),
        "primary_variant": primary,
        "mean_prob": {v: round(float(p.mean()), 3) for v, p in probs.items()},
        "n_iceberg_gt_0.5": {v: int((p > 0.5).sum()) for v, p in probs.items()},
        "chip_bg_db_median": {"HH": round(float(np.median(bg[:, 0])), 2),
                              "HV": round(float(np.median(bg[:, 1])), 2)} if len(det) else {},
        "kaggle_hv_bg_db_at_median_inc": round(interp_curve(curve, float(np.nanmedian(inc))), 2)
        if len(det) else None,
    }
    if len(variants) > 1 and len(det):
        a, b = (probs[v] for v in list(variants)[:2])
        summary["variant_agreement"] = round(float(((a > 0.5) == (b > 0.5)).mean()), 3)
        summary["variant_prob_corr"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
    js = det_dir / f"{name}_{band}_classified.json"
    js.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"classified": out, "chips": png, "summary": js}


def _raster_crs(path: Path) -> str:
    import rasterio

    with rasterio.open(path) as src:
        return src.crs.to_string()
