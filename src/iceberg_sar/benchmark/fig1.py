"""Outlines from Fig. 1 of Færch 2023: the Sentinel-2 area (pink) and the ALOS-2 scene (blue).

The paper gives no coordinates for its AOI, only Fig. 1. Its icebergs lie where Sentinel-1,
Sentinel-2 and ALOS-2 overlap, and Table 1's five S2 tiles cover only about half of that.
So the figure is georeferenced: pixel positions read off the published figure (2067 x 1479 px,
`fig1_points.csv`) for five control points with known coordinates (three corners of the
Sentinel-1 slice, the two weather stations), an affine fit from UTM to figure pixels, and the
outline vertices mapped back. Accuracy: a few km (the fit residuals), enough to choose tiles
and to cut the AOI, not for anything at the scale of a berg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon


def fit_affine(geo: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine geo (x, y) -> pixel; returns the 3x2 matrix and residuals in px."""
    a = np.column_stack([geo, np.ones(len(geo))])
    m, *_ = np.linalg.lstsq(a, px, rcond=None)
    return m, np.hypot(*(a @ m - px).T)


def pixel_to_geo(m: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Inverse of the affine fit: pixel -> geo (x, y)."""
    return np.linalg.solve(m[:2].T, (np.asarray(px, float) - m[2]).T).T


def outlines(points_csv: Path, crs: str) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Polygons for every non-control `kind` in the CSV (EPSG:4326), and the fit quality."""
    pts = pd.read_csv(points_csv)
    to_m = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    ctrl = pts[pts.kind == "control"]
    geo = np.column_stack(to_m.transform(ctrl.lon.to_numpy(), ctrl.lat.to_numpy()))
    m, res = fit_affine(geo, ctrl[["px_x", "px_y"]].to_numpy(float))
    m_per_px = 1 / np.mean(np.linalg.norm(m[:2], axis=0))
    rows = []
    for kind, g in pts[pts.kind != "control"].groupby("kind", sort=False):
        xy = pixel_to_geo(m, g[["px_x", "px_y"]].to_numpy(float))
        lon, lat = to_ll.transform(xy[:, 0], xy[:, 1])
        rows.append({"name": kind, "geometry": Polygon(zip(lon, lat, strict=True))})
    info = {"control_points": len(ctrl), "m_per_px": round(float(m_per_px), 1),
            "residual_px": [round(float(r), 1) for r in res],
            "rms_km": round(float(np.sqrt(np.mean(res**2)) * m_per_px / 1000), 1)}
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf.assign(source=f"georeferenced from Færch 2023 Fig. 1 (rms {info['rms_km']} km)"), info
