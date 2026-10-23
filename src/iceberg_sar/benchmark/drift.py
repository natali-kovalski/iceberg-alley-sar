"""Milestone 1d, Path B: move the Sentinel-2 labels to where the bergs were at the S1 pass.

Sentinel-2 imaged the scene about 5 h after Sentinel-1, so the bergs drifted in between. The
paper (sec. 3.4) matched them by hand:

  "icebergs observed in the Sentinel-2 image were manually matched to objects in the SAR
   images ... the process was aided by the fact that, on a large scale, icebergs arranged in
   clusters often drift in similar directions and over a similar distance ... it was possible
   to create verified drift paths for ... 270 of the icebergs in the Sentinel-1 image ...
   A linear interpolation method was used to predict the expected drift paths of these
   icebergs ... tested on a subset of the dataset (10 %) and shown to give an average distance
   error of ... 1789 m for Sentinel-1."

Automated version (parameters in `labels.drift`):
1. SAR targets: bright objects in the 40 m stack by one global threshold on incidence-normalized
   intensity (each band divided by its median at that incidence angle), HH AND HV. Deliberately not one
   of the benchmarked CFARs.
2. Drift field: for nodes on a grid, the shift (S2 label - SAR target) that lines up the most
   label-SAR pairs within `verify_radius_m`, using labels within `neighbourhood_m` of the
   node; kept if at least `min_support` pairs agree. A cluster that drifted together lines up
   at one shift; zero drift is allowed and wins ties.
3. Verified: each label is moved back by its node's drift and matched one-to-one to SAR
   targets within `verify_radius_m`; the matched SAR target is the berg's S1 position.
4. Interpolated: the other labels get a drift linearly interpolated (Delaunay) from the
   verified drift vectors, nearest verified vector outside their hull.
5. Checks as in the paper: number verified, mean drift, and the interpolation error on a
   random 10 % of the verified labels.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

from iceberg_sar.matching import match_points


def normalized_intensity(stack: np.ndarray, inc: np.ndarray, valid: np.ndarray,
                         bin_deg: float, sample_step: int = 4) -> np.ndarray:
    """Each band divided by its median over valid pixels in the same incidence-angle bin;
    returns the smaller of the two normalized bands (1 = typical clutter at that angle), so a
    threshold on it requires HH AND HV to be bright (HyP3-style HV speckle alone can't pass).

    The medians come from every `sample_step`-th pixel in both directions (1/16 of a scene by
    default) and are applied through a lookup table, so no full-scene index arrays are built.
    """
    ok = valid & np.isfinite(inc)
    bins = np.zeros(inc.shape, np.int16)
    bins[ok] = np.floor(inc[ok] / bin_deg).astype(np.int16)
    s = (slice(None, None, sample_step), slice(None, None, sample_step))
    ok_s, bins_s = ok[s], bins[s][ok[s]]
    out = np.full(inc.shape, np.inf, np.float32)
    for band in stack:
        med = pd.Series(band[s][ok_s]).groupby(bins_s).median()
        lut = np.full(int(bins.max()) + 1, np.nan, np.float32)
        lut[med.index.to_numpy()] = med.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.where(ok, band / lut[bins], 0.0).astype(np.float32)
        np.minimum(out, np.nan_to_num(norm), out=out)
    out[~ok] = 0.0
    return out


def sar_targets(stack_path: Any, inc_path: Any, valid_path: Any, p: dict[str, Any]
                ) -> gpd.GeoDataFrame:
    """Bright SAR objects: normalized intensity over 10^(threshold_db/10), 8-connected,
    `max_area_px` (paper: 500) or less; intensity-weighted centroid, scene CRS."""
    with rasterio.open(stack_path) as s, rasterio.open(inc_path) as a, \
            rasterio.open(valid_path) as v:
        stack, inc, valid = s.read(), a.read(1), v.read(1) > 0
        transform, crs = s.transform, s.crs
    norm = normalized_intensity(stack, inc, valid, p["incidence_bin_deg"])
    det = valid & (norm > 10 ** (p["sar_threshold_db"] / 10))
    labels, n = ndimage.label(det, structure=np.ones((3, 3), bool))
    idx = np.arange(1, n + 1)
    area = ndimage.sum_labels(det, labels, idx)
    keep = idx[area <= p["max_area_px"]]
    com = np.array(ndimage.center_of_mass(norm, labels, keep)).reshape(-1, 2)
    peak = ndimage.maximum(norm, labels, keep)
    xs, ys = rasterio.transform.xy(transform, com[:, 0], com[:, 1])
    return gpd.GeoDataFrame({"area_px": area[keep - 1].astype(int),
                             "peak_db": 10 * np.log10(np.asarray(peak))},
                            geometry=gpd.points_from_xy(xs, ys), crs=crs)


def histogram_peak(disp: np.ndarray, search_m: float, bin_m: float, radius_m: float
                   ) -> tuple[np.ndarray, int]:
    """Shift (dx, dy) supported by the most label-SAR pairs, and that number of pairs.

    Support of a shift = pairs whose displacement lies within `radius_m` of it (the 2-D
    histogram summed over a disk). Ties go to the smallest drift: zero drift is common
    (slow or grounded bergs near the coast).
    """
    edges = np.arange(-search_m, search_m + bin_m, bin_m)
    h, _, _ = np.histogram2d(disp[:, 0], disp[:, 1], bins=[edges, edges])
    r = max(1, round(radius_m / bin_m))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    support = ndimage.convolve(h, (x * x + y * y <= r * r).astype(float), mode="constant")
    centres = (edges[:-1] + edges[1:]) / 2
    best = support.max()
    cand = np.argwhere(support == best)
    k = np.argmin(np.hypot(centres[cand[:, 0]], centres[cand[:, 1]]))
    i, j = cand[k]
    return np.array([centres[i], centres[j]]), int(round(best))


def pair_displacements(lab: np.ndarray, sar: np.ndarray, search_m: float) -> np.ndarray:
    """All (label - sar) vectors with both components within +-search_m."""
    tree = cKDTree(sar)
    out = []
    for k, nbrs in enumerate(tree.query_ball_point(lab, search_m * np.sqrt(2))):
        if nbrs:
            d = lab[k] - sar[nbrs]
            out.append(d[(np.abs(d) <= search_m).all(axis=1)])
    return np.concatenate(out) if out else np.empty((0, 2))


def drift_nodes(lab: np.ndarray, sar: np.ndarray, p: dict[str, Any]) -> pd.DataFrame:
    """Local drift at grid nodes: histogram peak of displacements from nearby labels.

    Nodes with fewer than `min_node_labels` labels nearby, or whose best shift is supported by
    fewer than `min_support` label-SAR pairs within `verify_radius_m`, are dropped (no common
    drift that chance alignments can't explain).
    """
    step = p["node_spacing_m"]
    x0, y0 = lab.min(axis=0)
    x1, y1 = lab.max(axis=0)
    gx, gy = np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step))
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    tree = cKDTree(lab)
    rows = []
    for node, near in zip(nodes, tree.query_ball_point(nodes, p["neighbourhood_m"]),
                          strict=True):
        if len(near) < p["min_node_labels"]:
            continue
        disp = pair_displacements(lab[near], sar, p["search_m"])
        if len(disp) == 0:
            continue
        peak, support = histogram_peak(disp, p["search_m"], p["bin_m"], p["verify_radius_m"])
        if support >= p["min_support"]:
            rows.append({"x": node[0], "y": node[1], "dx": peak[0], "dy": peak[1],
                         "n_labels": len(near), "support": support})
    return pd.DataFrame(rows)


def node_drift_at(xy: np.ndarray, nodes: pd.DataFrame) -> np.ndarray:
    """Drift of the nearest node for each point."""
    _, k = cKDTree(nodes[["x", "y"]].to_numpy()).query(xy)
    return nodes[["dx", "dy"]].to_numpy()[k]


def interpolate_drift(at: np.ndarray, known_xy: np.ndarray, known_d: np.ndarray) -> np.ndarray:
    """Linear (Delaunay) interpolation of drift vectors; nearest outside the convex hull."""
    lin = LinearNDInterpolator(known_xy, known_d)(at)
    near = NearestNDInterpolator(known_xy, known_d)(at)
    return np.where(np.isfinite(lin), lin, near)


def rebuild(labels_s2: gpd.GeoDataFrame, sar: gpd.GeoDataFrame, p: dict[str, Any],
            seed: int = 0) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Labels at their S1-time position with status verified / interpolated, and checks."""
    lab = np.column_stack([labels_s2.geometry.x, labels_s2.geometry.y])
    sxy = np.column_stack([sar.geometry.x, sar.geometry.y])
    nodes = drift_nodes(lab, sxy, p)
    if nodes.empty:
        raise ValueError("No drift node with a clear peak; check the SAR targets / parameters")
    expected = lab - node_drift_at(lab, nodes)
    m = match_points(expected, sxy, p["verify_radius_m"])
    li, si = m.pairs[:, 0], m.pairs[:, 1]
    pos = np.full_like(lab, np.nan)
    pos[li] = sxy[si]
    verified = np.zeros(len(lab), bool)
    verified[li] = True
    drift = lab - pos  # S1 -> S2 displacement, known for verified labels
    others = ~verified
    if verified.sum() >= 3:
        drift[others] = interpolate_drift(lab[others], lab[verified], drift[verified])
    else:
        drift[others] = node_drift_at(lab[others], nodes)
    pos[others] = lab[others] - drift[others]
    out = labels_s2.copy()
    out["s2_x"], out["s2_y"] = lab[:, 0], lab[:, 1]
    out["status"] = np.where(verified, "verified", "interpolated")
    out["radius_m"] = np.where(verified, p["radius_verified_m"], p["radius_interpolated_m"])
    out["drift_dx_m"], out["drift_dy_m"] = drift[:, 0], drift[:, 1]
    out["drift_m"] = np.hypot(drift[:, 0], drift[:, 1])
    out["label_source"] = "reconstructed (Path B)"
    out = out.set_geometry(gpd.points_from_xy(pos[:, 0], pos[:, 1]), crs=labels_s2.crs)
    checks = {"n_labels": len(out), "n_verified": int(verified.sum()),
              "n_drift_nodes": len(nodes), "n_sar_targets": len(sar),
              "mean_drift_verified_m": float(out.loc[verified, "drift_m"].mean())
              if verified.any() else float("nan"),
              "median_match_dist_m": float(np.median(m.distances_m)) if len(li) else float("nan")}
    checks.update(holdout_error(lab[verified], drift[verified], p["holdout_fraction"], seed))
    return out, {"checks": checks, "nodes": nodes}


def holdout_error(xy: np.ndarray, d: np.ndarray, fraction: float, seed: int) -> dict[str, Any]:
    """The paper's interpolation test: hide a random share of the verified drift vectors,
    interpolate them from the rest, report the mean position error."""
    n = len(xy)
    k = max(1, round(fraction * n))
    if n < k + 3:
        return {"holdout_n": 0, "holdout_mean_error_m": float("nan")}
    rng = np.random.default_rng(seed)
    test = np.zeros(n, bool)
    test[rng.choice(n, k, replace=False)] = True
    pred = interpolate_drift(xy[test], xy[~test], d[~test])
    err = np.hypot(*(pred - d[test]).T)
    return {"holdout_n": int(k), "holdout_mean_error_m": float(err.mean()),
            "holdout_median_error_m": float(np.median(err))}
