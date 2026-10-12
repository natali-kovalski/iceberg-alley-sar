"""Frozen THOR encoder: our HyP3 RTC bands -> THOR input -> patch-token feature grid.

THOR (FM4CS, Apache 2.0) was pretrained on Sentinel-1 GRD sigma0, thermal-noise corrected,
in dB, with its own HH/HV patch embeddings (no need to put HH/HV in the VV/VH slots).
HyP3 RTC is gamma0; over flat sea sigma0 = gamma0 * cos(incidence).

TerraTorch and the THOR extension are imported lazily, so the rest of the package
(and its tests) work without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iceberg_sar.config import Config

# Our band names -> THOR band names (SARThorBands in thor_terratorch_ext).
THOR_BANDS = {"HH": "IW_HH", "HV": "IW_HV", "VV": "IW_VV", "VH": "IW_VH"}


@dataclass(frozen=True)
class ThorParams:
    model: str
    bands: tuple[str, ...]
    patch_px: int
    crop_px: int
    to_sigma0: bool
    db_clip: tuple[float, float]
    norm: dict[str, tuple[float, float]]  # band -> (mean dB, std dB)
    layer: int

    @classmethod
    def from_config(cls, cfg: Config) -> ThorParams:
        t = cfg.section("thor")
        return cls(
            model=t["model"],
            bands=tuple(t["bands"]),
            patch_px=int(t["patch_px"]),
            crop_px=int(t["crop_px"]),
            to_sigma0=bool(t["to_sigma0"]),
            db_clip=(float(t["db_clip"][0]), float(t["db_clip"][1])),
            norm={b: (float(m), float(s)) for b, (m, s) in t["norm"].items()},
            layer=int(t["layer"]),
        )

    def grid_px(self) -> int:
        """Tokens per side of the feature grid (64 for a 512 px crop at patch 8)."""
        return self.crop_px // self.patch_px


def as_sigma0(
    lin: np.ndarray, inc_deg: np.ndarray | None, params: ThorParams, radiometry: str
) -> np.ndarray:
    """Linear band in the radiometry THOR expects (gamma0 products are converted)."""
    if not (params.to_sigma0 and radiometry == "gamma0"):
        return lin
    if inc_deg is None:
        raise ValueError("gamma0 -> sigma0 needs the incidence-angle map")
    return lin * np.cos(np.radians(inc_deg)).astype(np.float32)


def to_model_input(
    bands_lin: dict[str, np.ndarray],
    inc_deg: np.ndarray | None,
    params: ThorParams,
    radiometry: str = "gamma0",
) -> np.ndarray:
    """Linear RTC bands -> (C, H, W) float32, normalized like THOR's pretraining.

    `radiometry` is the product's ('gamma0' or 'sigma0', see `product_radiometry`).
    NaN (land / nodata) becomes 0 after normalization, i.e. the band mean.
    """
    out = []
    for band in params.bands:
        lin = as_sigma0(np.asarray(bands_lin[band], dtype=np.float32), inc_deg, params,
                        radiometry)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = np.where(lin > 0, 10.0 * np.log10(np.maximum(lin, 1e-12)), params.db_clip[0])
        db = np.clip(db, *params.db_clip)
        mean, std = params.norm[band]
        z = (db - mean) / std
        out.append(np.where(np.isfinite(lin), z, 0.0).astype(np.float32))
    return np.stack(out)


def load_encoder(params: ThorParams, pixel_m: float, device: str = "cpu") -> Any:
    """Build the frozen THOR backbone (weights from the Hugging Face hub, cached)."""
    import importlib.util

    import torch._dynamo

    # THOR decorates kernels with @torch.compile, which needs Triton (not on Windows).
    # Eager mode gives the same numbers, just slower.
    if importlib.util.find_spec("triton") is None:
        torch._dynamo.config.disable = True

    import thor_terratorch_ext  # noqa: F401  (registers the THOR backbones)
    from terratorch import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        params.model,
        pretrained=True,
        model_bands=[THOR_BANDS[b] for b in params.bands],
        ground_cover=int(round(params.crop_px * pixel_m)),
        patch_sizes=params.patch_px,
        merge_method="concat",
    )
    model.eval().requires_grad_(False)
    return model.to(device)


def embed(model: Any, x: np.ndarray, device: str = "cpu", layer: int = -1) -> np.ndarray:
    """(C, H, W) or (B, C, H, W) model input -> (B, D, h, w) features of one encoder block."""
    import torch

    batch = x[None] if x.ndim == 3 else x
    with torch.no_grad():
        feats = model(torch.from_numpy(batch).to(device))
    return feats[layer].float().cpu().numpy()


def embed_blocks(
    model: Any, x: np.ndarray, blocks: tuple[int, ...], device: str = "cpu"
) -> np.ndarray:
    """(B, C, H, W) model input -> (B, len(blocks) * D, h, w) float16, blocks concatenated."""
    import torch

    with torch.no_grad():
        feats = model(torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(device))
        out = torch.cat([feats[b] for b in blocks], dim=1)
    return out.half().cpu().numpy()


def pixel_m_from_name(name: str) -> float:
    """Pixel size from a HyP3 RTC product name (`..._RTC10_...` -> 10 m)."""
    import re

    m = re.search(r"_RTC(\d+)_", name)
    if m is None:
        raise ValueError(f"No RTC pixel size in {name!r}")
    return float(m.group(1))


def pca_rgb(
    feats: np.ndarray, valid: np.ndarray | None = None, pct: tuple[float, float] = (2, 98)
) -> tuple[np.ndarray, np.ndarray]:
    """(D, h, w) features -> (h, w, 3) uint8 image of the first 3 principal components.

    PCA is fitted on valid tokens only, so land doesn't eat a component.
    Returns the image and the explained-variance ratio of the 3 components.
    """
    d, h, w = feats.shape
    flat = feats.reshape(d, -1).T.astype(np.float64)
    fit_mask = np.ones(h * w, bool) if valid is None else valid.reshape(-1)
    fit = flat[fit_mask]
    mu = fit.mean(axis=0)
    _, s, vt = np.linalg.svd(fit - mu, full_matrices=False)
    comps = (flat - mu) @ vt[:3].T
    var = s**2
    ratio = var[:3] / var.sum()

    rgb = np.zeros((h * w, 3), np.uint8)
    for i in range(3):
        lo, hi = np.percentile(comps[fit_mask, i], pct)
        rgb[:, i] = (np.clip((comps[:, i] - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    rgb[~fit_mask] = 0
    return rgb.reshape(h, w, 3), ratio
