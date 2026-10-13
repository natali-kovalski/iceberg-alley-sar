"""Regenerate the figures in docs/img from the current pipeline outputs (`figures` command).

- validation: detections vs. same-pass IIP labels on a map, plus recall by IIP size class.
  The point validation is rerun here, so the figure always matches the numbers.
- detections: the quicklook with CFAR detections that `detect` writes, copied as is.
The Cesium viewer screenshot (viewer.png) is taken by hand.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from iceberg_sar.config import Config  # noqa: E402
from iceberg_sar.detections import scene_timestamp  # noqa: E402
from iceberg_sar.pointval import run_point_validation  # noqa: E402
from iceberg_sar.preprocess import product_radiometry, read_downsampled_db  # noqa: E402

# Reference palette (dataviz skill), light mode. The three series pass the all-pairs
# CVD and normal-vision checks; each also has its own marker shape and a counted legend.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
FOUND = "#2a78d6"     # blue
SAR_ONLY = "#eb6834"  # orange
MISSED = "#1baf7a"    # aqua

IIP_SIZES = {  # IIP size codes -> label (length class)
    "GRL": "Growler\n<5 m", "BB": "Bergy bit\n5-14 m", "SM": "Small\n15-60 m",
    "MED": "Medium\n61-120 m", "LG": "Large\n121-220 m", "VLG": "Very large\n>220 m",
}


def recall_by_size(truth: gpd.GeoDataFrame) -> pd.DataFrame:
    """Matched / compared per IIP size class, in size order; unknown codes go last."""
    c = truth[truth["compared"]]
    t = c.groupby("size")["matched"].agg(found="sum", total="count")
    order = [k for k in IIP_SIZES if k in t.index] + sorted(set(t.index) - set(IIP_SIZES))
    t = t.loc[order]
    t["recall"] = t["found"] / t["total"]
    t["label"] = [IIP_SIZES.get(k, k) for k in t.index]
    return t


def _rounded_hbar(ax: plt.Axes, y: float, width: float, height: float, color: str) -> None:
    """Horizontal bar anchored at 0 with ~4 px rounded corners at the data end only.

    One polygon (no seams). Call after the axes limits are set: the corner radius is
    converted to data units per axis so it is round on screen.
    """
    if width <= 0:
        return
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px = 4 * ax.figure.dpi / 72
    rx = min(px * (x1 - x0) / bbox.width, width / 2)
    ry = min(px * (y1 - y0) / bbox.height, height / 2)
    lo, hi = y - height / 2, y + height / 2
    a = np.linspace(0, np.pi / 2, 12)
    top = np.c_[width - rx + rx * np.sin(a), hi - ry + ry * np.cos(a)]       # top-right corner
    bottom = np.c_[width - rx + rx * np.cos(a), lo + ry - ry * np.sin(a)]    # bottom-right
    pts = np.vstack([[0, hi], top, bottom, [0, lo]])
    ax.add_patch(plt.Polygon(pts, closed=True, linewidth=0, facecolor=color))


def validation_figure(cfg: Config, name: str, truth_kind: str, band: str, png: Path) -> dict:
    """Map of found / missed / SAR-only targets and recall by size. Returns the summary."""
    summary, truth, det = run_point_validation(cfg, name, truth_kind, band, open_water=False)
    masked = cfg.path("interim") / name / f"{name}_{band}_masked.tif"
    db, factor = read_downsampled_db(masked, 1600, "mean")  # smooth backdrop
    with rasterio.open(masked) as src:
        crs, transform = src.crs, src.transform
        pixel_m = float(src.res[0])
    t = truth[truth["compared"]].to_crs(crs)
    t_rows, t_cols = rasterio.transform.rowcol(transform, t.geometry.x, t.geometry.y, op=float)
    t_rc = np.c_[t_rows, t_cols] / factor
    d = det[det["compared"]]
    d_rc = d[["row", "col"]].to_numpy(float) / factor
    found_t, missed_t = t["matched"].to_numpy(), ~t["matched"].to_numpy()
    sar_only = ~d["matched"].to_numpy()

    # Crop the backdrop to its valid pixels so the map fills its panel.
    finite = np.isfinite(db)
    rows_ok, cols_ok = np.flatnonzero(finite.any(1)), np.flatnonzero(finite.any(0))
    r0, c0 = rows_ok[0], cols_ok[0]
    db = db[r0:rows_ok[-1] + 1, c0:cols_ok[-1] + 1]
    t_rc, d_rc = t_rc - [r0, c0], d_rc - [r0, c0]

    fig = plt.figure(figsize=(16, 7.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.14, top=0.86, bottom=0.14,
                          left=0.02, right=0.98)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    # Soft, low-contrast backdrop so the markers carry the figure.
    vmin, vmax = np.nanpercentile(db, [1, 99])
    vmin -= 0.6 * (vmax - vmin)
    ax.imshow(np.ma.masked_invalid(db), cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.scatter(d_rc[sar_only, 1], d_rc[sar_only, 0], s=34, facecolors="none",
               edgecolors=SAR_ONLY, linewidths=1.3, zorder=3)
    ax.scatter(t_rc[found_t, 1], t_rc[found_t, 0], s=30, color=FOUND, edgecolors=SURFACE,
               linewidths=1.0, zorder=4)
    ax.scatter(t_rc[missed_t, 1], t_rc[missed_t, 0], s=70, marker="x", color=MISSED,
               linewidths=2.2, zorder=5)
    ax.set_axis_off()
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=SAR_ONLY,
               markeredgewidth=1.3, markersize=8, label=f"SAR only: {int(sar_only.sum())}"),
        Line2D([], [], marker="o", ls="", color=FOUND, markeredgecolor=SURFACE, markersize=8,
               label=f"Found (SAR + IIP): {int(found_t.sum())}"),
        Line2D([], [], marker="x", ls="", color=MISSED, markeredgewidth=2.2, markersize=9,
               label=f"Missed IIP berg: {int(missed_t.sum())}"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=11,
                    labelcolor=TEXT, facecolor=SURFACE, edgecolor=GRID)
    leg.get_frame().set_linewidth(1)

    bx = fig.add_subplot(gs[1])
    bx.set_facecolor(SURFACE)
    rs = recall_by_size(truth)
    ys = np.arange(len(rs))[::-1].astype(float)
    bx.set_xlim(0, 1.25)
    bx.set_ylim(-0.7, len(rs) - 0.3)
    for y, r in zip(ys, rs.itertuples(), strict=True):
        _rounded_hbar(bx, y, r.recall, 0.4, FOUND)
        bx.text(r.recall + 0.015, y, f"{r.recall:.0%}  ({r.found} / {r.total})", va="center",
                ha="left", color=TEXT, fontsize=12)
    bx.set_yticks(ys, rs["label"], fontsize=12, color=TEXT_2)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"], color=TEXT_2,
                  fontsize=11)
    bx.grid(axis="x", color=GRID, linewidth=1)
    bx.set_axisbelow(True)
    bx.tick_params(length=0)
    for side in ("top", "right", "bottom"):
        bx.spines[side].set_visible(False)
    bx.spines["left"].set_color(GRID)
    found, total = summary["matched"], summary["truth_compared"]
    # Panel titles as figure text on one baseline, so they line up whatever the axes sizes.
    for axis, title, sub in (
        (ax, "Detections vs. IIP analyst labels", "same Sentinel-1 pass, all detections"),
        (bx, "Recall by IIP size class", f"overall {found / total:.0%} ({found} of {total})"),
    ):
        x0 = axis.get_position().x0
        fig.text(x0, 0.955, title, color=TEXT, fontsize=15, va="top")
        fig.text(x0, 0.912, sub, color=TEXT_2, fontsize=12, va="top")
    when = pd.Timestamp(scene_timestamp(name)).strftime("%Y-%m-%d %H:%M UTC")
    bx.text(0, -0.08,
            f"{when}, Sentinel-{name[1:3]} IW, {band}, {pixel_m:.0f} m, "
            f"{product_radiometry(name)}.\n"
            "IIP = International Ice Patrol sightings DB (NSIDC G00807), positions at 0.1' "
            "cell centres.\n"
            f"Median match distance {summary['median_match_dist_m']} m; match radius "
            f"{cfg.section('validation')['max_dist_m']} m.",
            transform=bx.transAxes, va="top", color=TEXT_2, fontsize=10.5)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return summary


def copy_detections_figure(cfg: Config, name: str, band: str, png: Path) -> Path:
    """The quicklook with detections that `detect` wrote for this scene."""
    src = cfg.path("outputs") / "detections" / f"{name}_{band}_detections.png"
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found; run `detect` first")
    png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, png)
    return png


def make_figures(cfg: Config) -> dict[str, Path]:
    """Every scripted figure listed in the config's `figures` section."""
    f = cfg.section("figures")
    out = cfg.root / f["out_dir"]
    v, d = f["validation"], f["detections"]
    written = {}
    png = out / v["file"]
    validation_figure(cfg, v["scene"], v["truth"], v["band"], png)
    written["validation"] = png
    written["detections"] = copy_detections_figure(cfg, d["scene"], d["band"], out / d["file"])
    return written
