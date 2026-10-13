"""The 5d figure: recall vs. detections per scene, CFAR (PFA sweep) and THOR (score sweep)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iceberg_sar.figures import FOUND, GRID, MISSED, SAR_ONLY, SURFACE, TEXT, TEXT_2  # noqa: E402

# Validated three-series palette from figures.py, fixed per detector; marker shape is the
# second encoding, and every series is direct-labelled at its end.
STYLE = {
    "cfar": {"color": SAR_ONLY, "marker": "s", "label": "CA-CFAR (PFA sweep)"},
    "thor-linear": {"color": FOUND, "marker": "o", "label": "THOR + linear head"},
    "thor-centernet": {"color": MISSED, "marker": "^", "label": "THOR + CenterNet ensemble"},
}


def recall_figure(sweep: pd.DataFrame, p, meta: dict, png: Path) -> Path:
    """One panel per scene (pooled first, if present); x = detections in open water (log),
    y = recall."""
    from iceberg_sar.thor.compare import POOLED

    scenes = list(dict.fromkeys(sweep["scene"]))
    if POOLED in scenes:
        scenes = [POOLED] + [s for s in scenes if s != POOLED]
    ncols = min(3, len(scenes))
    nrows = -(-len(scenes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows + 0.4),
                             squeeze=False, facecolor=SURFACE)
    for ax in axes.flat[len(scenes):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, scenes, strict=False):
        s = sweep[sweep["scene"] == name]
        ax.set_facecolor(SURFACE)
        for det, g in s.groupby("detector", sort=False):
            st = STYLE.get(det, {"color": TEXT_2, "marker": "x", "label": det})
            g = g.sort_values("detections")
            ax.plot(g["detections"], g["recall"], color=st["color"], lw=2, marker=st["marker"],
                    ms=5 if det != "cfar" else 7, markeredgecolor=SURFACE, markeredgewidth=1,
                    label=st["label"], zorder=3)
            if det == "cfar":
                for _, r in g.iterrows():
                    ax.annotate(f"{r['setting']:.0e}".replace("e-0", "e-"),
                                (r["detections"], r["recall"]), textcoords="offset points",
                                xytext=(4, -11), fontsize=7, color=TEXT_2)
                ref = g[g["setting"] == p.cfar_pfa]
                ax.plot(ref["detections"], ref["recall"], "s", ms=12, mfc="none",
                        mec=TEXT, mew=1.2, zorder=4)
            else:
                ref = g.iloc[(g["setting"] - p.view_score).abs().argsort()[:1]]
                ax.plot(ref["detections"], ref["recall"], "o", ms=11, mfc="none", mec=TEXT,
                        mew=1.2, zorder=4)
        n_truth = int(s["truth"].iloc[0])
        xs = np.append(np.geomspace(1, n_truth, 60), 1e5)
        ax.plot(xs, np.minimum(xs / n_truth, 1), color=TEXT_2, lw=1, ls=":", zorder=1)
        ax.annotate("ideal: every detection\nis a label", (n_truth * 0.42, 0.42),
                    fontsize=7, color=TEXT_2, ha="right")
        m = meta[name]
        if name == POOLED:
            title = (f"All {len(m['scenes'])} scored frames: {m['labels_compared']} IIP labels "
                     "(pooled)")
        else:
            when = name.split("_")[2]
            title = (f"{when[:4]}-{when[4:6]}-{when[6:8]} {name[:3]} {when[9:15]}: "
                     f"{m['labels_compared']} IIP labels")
            title += " (same pass)" if m["truth"] == "iip-satellite" else \
                f" ({m['truth']}, drift {'applied' if m['drift']['applied'] else 'unresolved'})"
        ax.set_title(title, loc="left", fontsize=10, color=TEXT,
                     fontweight="bold" if name == POOLED else "normal")
        ax.set_xscale("log")
        ax.set_xlim(10, max(3000, s["detections"].max() * 1.2))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Detections in open water" + (", summed over frames" if name == POOLED
                                                     else " per frame"), color=TEXT_2)
        ax.set_ylabel(f"Recall (match ≤ {meta['radius_m']:.0f} m)", color=TEXT_2)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=8)
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT)
    fig.text(0.01, 0.01, f"Open square: CFAR at PFA {p.cfar_pfa:g}; open circles: THOR at score "
             f"{p.view_score:g} (the configured operating points).\nCFAR points are labelled "
             "with their PFA. Detections within "
             f"{meta['near_ice_km']:g} km of pack ice are dropped for both.",
             fontsize=7.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.4 / fig.get_figheight(), 1, 1))
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png
