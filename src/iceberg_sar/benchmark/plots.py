"""Benchmark figures: detections over the HV dB image."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOUR = "#ff3b30"


def db_stretch(db: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    v = db[np.isfinite(db)]
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def to_db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(x > 0, x, np.nan))


def detection_quicklook(
    hv_db: np.ndarray, gdf: gpd.GeoDataFrame, png: Path, title: str, max_px: int,
    stretch: tuple[float, float],
) -> Path:
    """Full-frame HV dB (decimated to `max_px`) with detections circled."""
    step = max(1, int(np.ceil(max(hv_db.shape) / max_px)))
    img = hv_db[::step, ::step]
    h, w = img.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap="gray", vmin=stretch[0], vmax=stretch[1], interpolation="nearest")
    if len(gdf):
        ax.scatter(gdf["col"] / step, gdf["row"] / step, s=40, facecolors="none",
                   edgecolors=COLOUR, linewidths=0.8)
    ax.text(15, 15, title, color="yellow", fontsize=20, va="top")
    ax.set_axis_off()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, facecolor="black")
    plt.close(fig)
    return png


def densest_window(points: list[gpd.GeoDataFrame], shape: tuple[int, int], size: int,
                   valid: np.ndarray) -> tuple[int, int]:
    """Top-left (row, col) of the `size` px window with the most detections over all
    detectors, among windows that are at least half valid."""
    h, w = shape
    nb_r, nb_c = max(1, h // size), max(1, w // size)
    counts = np.zeros((nb_r, nb_c))
    for g in points:
        if len(g):
            r = np.clip((g["row"].to_numpy() // size).astype(int), 0, nb_r - 1)
            c = np.clip((g["col"].to_numpy() // size).astype(int), 0, nb_c - 1)
            np.add.at(counts, (r, c), 1)
    share = valid[: nb_r * size, : nb_c * size].reshape(nb_r, size, nb_c, size).mean(axis=(1, 3))
    counts[share < 0.5] = -1
    i, j = np.unravel_index(np.argmax(counts), counts.shape)
    return int(i * size), int(j * size)


def side_by_side(
    hv_db: np.ndarray, results: dict[str, gpd.GeoDataFrame], window: tuple[int, int, int],
    png: Path, stretch: tuple[float, float], suptitle: str,
) -> Path:
    """One panel per detector over the same zoom window (row0, col0, size)."""
    r0, c0, size = window
    crop = hv_db[r0:r0 + size, c0:c0 + size]
    n = len(results)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow + 0.6),
                             sharex=True, sharey=True)
    for ax, (name, g) in zip(axes.ravel(), results.items(), strict=False):
        ax.imshow(crop, cmap="gray", vmin=stretch[0], vmax=stretch[1], interpolation="nearest")
        inside = g[(g["row"] >= r0) & (g["row"] < r0 + size)
                   & (g["col"] >= c0) & (g["col"] < c0 + size)] if len(g) else g
        if len(inside):
            ax.scatter(inside["col"] - c0, inside["row"] - r0, s=60, facecolors="none",
                       edgecolors=COLOUR, linewidths=1.0)
        ax.set_title(f"{name}: {len(inside)} in view / {len(g)} total", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png
