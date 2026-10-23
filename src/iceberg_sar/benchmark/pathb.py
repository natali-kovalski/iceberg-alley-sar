"""Milestone 1d, Path B: run the label rebuild end to end and document it.

Outputs (all named and tagged "reconstructed"), under `labels.dir`:
  s2_targets_reconstructed.geojson   Sentinel-2 objects at their S2 position
  labels_reconstructed.geojson       the same objects at their S1 position (verified /
                                     interpolated, search radius per label) -> scorer in 1f
  drift_paths_reconstructed.geojson  S2 -> S1 lines; drift_nodes_reconstructed.geojson
  sar_targets_for_matching.geojson   bright SAR objects used for matching (not a detector run)
  aoi_reconstructed.geojson          S1 swath x usable S2 area (land, clouds, nodata removed)
and `labels.report` (Markdown, committed): steps, quotes, counts next to the paper's.
The paper's numbers are read from the XML at run time, never typed in.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import LineString

from iceberg_sar.benchmark import plots
from iceberg_sar.benchmark.drift import rebuild, sar_targets
from iceberg_sar.benchmark.labels import SIZE_CLASSES, s2_targets
from iceberg_sar.benchmark.reproduce import scene_paths
from iceberg_sar.benchmark.scenes import TABLE_NS, _text, parse_product
from iceberg_sar.config import Config

MARK = "RECONSTRUCTED LABELS"


def paper_numbers(xml_path: Path) -> dict[str, Any]:
    """Validation-set figures from the paper's text and Table 2 (sec. 3.4), for Sentinel-1."""
    root = ET.parse(xml_path).getroot()
    text = " ".join(_text(p) for p in root.iter("p"))
    out: dict[str, Any] = {}
    patterns = {
        "total": r"A total of (\d+) icebergs were detected",
        "mean_length_m": r"average iceberg length was (\d+) m",
        "verified_s1": r"and (\d+) of the icebergs in the Sentinel-1 image",
        "mean_drift_s1_m": r"and (\d+) m between the Sentinel-2 and Sentinel-1",
        "interp_error_s1_m": r"(\d+) m for Sentinel-1",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m is None:
            raise ValueError(f"Paper figure '{k}' not found in {xml_path} ({pat!r})")
        out[k] = int(m[1])
    for tw in root.iter("table-wrap"):
        if _text(tw.find("label")) == "Table 2":
            rows = [[_text(c) for c in r.iter(TABLE_NS + "entry")]
                    for r in tw.iter(TABLE_NS + "row")][1:]
            out["classes"] = {r[0].split()[0].lower(): int(r[1]) for r in rows}
    return out


def _flags(labels: gpd.GeoDataFrame, land_path: Path, swath_path: Path) -> gpd.GeoDataFrame:
    """in_sar_land_buffer / in_swath at the label's S1 position (sampled from the 40 m masks)."""
    xy = list(zip(labels.geometry.x, labels.geometry.y, strict=True))
    with rasterio.open(land_path) as l_src, rasterio.open(swath_path) as s_src:
        land = np.array([v[0] for v in l_src.sample(xy)]) > 0
        swath = np.array([v[0] for v in s_src.sample(xy)]) > 0
    return labels.assign(in_sar_land_buffer=land, in_swath=swath)


def _swath_polygon(swath_path: Path, factor: int = 5) -> gpd.GeoDataFrame:
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(swath_path) as src:
        h, w = src.height // factor, src.width // factor
        m = src.read(1, out_shape=(h, w)) > 0
        t = src.transform * src.transform.scale(src.width / w, src.height / h)
        geoms = [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=t) if v]
        return gpd.GeoDataFrame(geometry=[gpd.GeoSeries(geoms).union_all()], crs=src.crs)


def run_pathb(cfg: Config, force_s2: bool = False, log: Any = print) -> dict[str, Any]:
    fc = cfg.section("benchmark")["faerch2023"]
    p = fc["labels"]
    sp = scene_paths(cfg)
    for f in (sp.stack, sp.valid, sp.land):
        if not f.is_file():
            raise FileNotFoundError(f"{f} missing; run `bench-preprocess` and `bench-landmask`")
    out_dir = cfg.root / p["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = out_dir / "s2_targets_cache_reconstructed.gpkg"  # all tiles, before the AOI cut
    if cache.is_file() and not force_s2:
        targets = gpd.read_file(cache, layer="targets")
        s2_area = gpd.read_file(cache, layer="area")
        log(f"Sentinel-2 targets from cache ({len(targets)}); --force-s2 to redo")
    else:
        log("Sentinel-2 targets (4 sigma):")
        targets, s2_area = s2_targets(cfg, log)
        cache.unlink(missing_ok=True)
        targets.to_file(cache, layer="targets", driver="GPKG")
        s2_area.to_file(cache, layer="area", driver="GPKG")
    aoi = s2_area.overlay(_swath_polygon(sp.swath), how="intersection")
    what = "S1 swath x usable S2 area"
    if p.get("aoi_alos2"):
        outl = cfg.root / fc["fig1_outlines"]
        if not outl.is_file():
            raise FileNotFoundError(f"{outl} missing; run `bench-fig1` first")
        alos = gpd.read_file(outl).query("name == 'alos2'").to_crs(fc["crs"])
        aoi = aoi.overlay(alos[["geometry"]], how="intersection")
        what += " x ALOS-2 (Fig. 1)"
    targets = targets[targets.within(aoi.union_all())].reset_index(drop=True)
    targets.insert(0, "label_id", np.arange(len(targets)))
    log(f"{len(targets)} S2 targets inside the AOI ({what}), {aoi.area.sum() / 1e6:,.0f} km2")

    sar = sar_targets(sp.stack, sp.inc, sp.valid, p["drift"])
    log(f"{len(sar)} bright SAR objects for matching")
    labels, info = rebuild(targets, sar, p["drift"])
    labels = _flags(labels, sp.land, sp.swath)
    c = info["checks"]
    log(f"verified {c['n_verified']} / {c['n_labels']}, mean drift (verified) "
        f"{c['mean_drift_verified_m']:.0f} m, holdout error {c['holdout_mean_error_m']:.0f} m")

    files = _write(out_dir, targets, labels, sar, info["nodes"], aoi, fc["crs"])
    paper = paper_numbers(cfg.root / fc["paper_xml"])
    pngs = _quicklooks(sp, labels, sar, out_dir.parent / "quicklooks", int(fc["quicklook_max_px"]))
    report = cfg.root / p["report"]
    report.write_text(_report(cfg, labels, info, paper, files, pngs, aoi), encoding="utf-8")
    return {"files": files, "pngs": pngs, "report": report, "checks": c}


def _write(out_dir: Path, targets: gpd.GeoDataFrame, labels: gpd.GeoDataFrame,
           sar: gpd.GeoDataFrame, nodes: Any, aoi: gpd.GeoDataFrame, crs: str
           ) -> dict[str, Path]:
    tag = {"label_source": "reconstructed (Path B)"}
    paths = {}

    def save(name: str, gdf: gpd.GeoDataFrame) -> None:
        path = out_dir / name
        gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
        paths[name] = path

    save("s2_targets_reconstructed.geojson", targets.assign(**tag))
    save("labels_reconstructed.geojson", labels)
    lines = [LineString([(a, b), (g.x, g.y)]) for a, b, g in
             zip(labels.s2_x, labels.s2_y, labels.geometry, strict=True)]
    save("drift_paths_reconstructed.geojson", gpd.GeoDataFrame(
        labels[["label_id", "status", "drift_m"]].assign(**tag), geometry=lines, crs=crs))
    save("drift_nodes_reconstructed.geojson", gpd.GeoDataFrame(
        nodes.assign(**tag), geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=crs))
    save("sar_targets_for_matching.geojson", sar)
    save("aoi_reconstructed.geojson", aoi.assign(**tag))
    return paths


def _quicklooks(sp: Any, labels: gpd.GeoDataFrame, sar: gpd.GeoDataFrame, qdir: Path,
                max_px: int) -> dict[str, Path]:
    """Full scene with labels, and a zoom on the densest 20 km with drift paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(sp.stack) as src:
        db = plots.to_db(src.read(1))
        inv = ~src.transform
    stretch = plots.db_stretch(db)
    rc = lambda x, y: (np.array(inv * (np.asarray(x), np.asarray(y)))[::-1])  # noqa: E731
    lr, lc = rc(labels.geometry.x, labels.geometry.y)
    sr, sc = rc(labels.s2_x, labels.s2_y)
    ver = (labels.status == "verified").to_numpy()
    colours = {"verified": "#30d158", "interpolated": "#ff9f0a"}
    out = {}

    step = max(1, int(np.ceil(max(db.shape) / max_px)))
    fig = plt.figure(figsize=(db.shape[1] / step / 100, db.shape[0] / step / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(db[::step, ::step], cmap="gray", vmin=stretch[0], vmax=stretch[1])
    for st, m in (("interpolated", ~ver), ("verified", ver)):
        ax.scatter(lc[m] / step, lr[m] / step, s=30, facecolors="none", edgecolors=colours[st],
                   linewidths=0.8, label=f"{st} ({m.sum()})")
    ax.text(15, 15, f"{MARK}: Sentinel-2 4-sigma, moved to the S1 pass. HH dB, 40 m",
            color="yellow", fontsize=18, va="top")
    ax.legend(loc="lower left", fontsize=14)
    ax.set_axis_off()
    out["full"] = qdir / "labels_reconstructed_full.png"
    qdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out["full"], facecolor="black")
    plt.close(fig)

    size = 500  # 20 km at 40 m
    r0, c0 = plots.densest_window([labels.assign(row=lr, col=lc)], db.shape, size,
                                  np.isfinite(db))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(db[r0:r0 + size, c0:c0 + size], cmap="gray", vmin=stretch[0], vmax=stretch[1],
              extent=(c0, c0 + size, r0 + size, r0))
    srr, scc = rc(sar.geometry.x, sar.geometry.y)
    ax.scatter(scc, srr, s=8, c="#64d2ff", marker="+", linewidths=0.6, label="SAR bright objects")
    for st, m in (("interpolated", ~ver), ("verified", ver)):
        ax.plot(np.vstack([sc[m], lc[m]]), np.vstack([sr[m], lr[m]]), color=colours[st], lw=0.6)
        ax.scatter(lc[m], lr[m], s=40, facecolors="none", edgecolors=colours[st], linewidths=1,
                   label=f"{st}: S1 position")
        ax.scatter(sc[m], sr[m], s=6, c=colours[st], label=f"{st}: S2 position")
    ax.set_xlim(c0, c0 + size)
    ax.set_ylim(r0 + size, r0)
    ax.set_title(f"{MARK}: drift S2 (dot) -> S1 (ring), 20 km zoom")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    out["zoom"] = qdir / "labels_reconstructed_zoom.png"
    fig.savefig(out["zoom"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def _report(cfg: Config, labels: gpd.GeoDataFrame, info: dict[str, Any], paper: dict[str, Any],
            files: dict[str, Path], pngs: dict[str, Path], aoi: gpd.GeoDataFrame) -> str:
    fc = cfg.section("benchmark")["faerch2023"]
    p = fc["labels"]
    d = p["drift"]
    c = info["checks"]
    s2 = [s for s in fc["scenes"] if s["sensor"] == "Sentinel-2"]
    tiles = lambda basis: "/".join(parse_product(s["product_id"])["tile"]  # noqa: E731
                                   for s in s2 if s.get("basis", "table1") == basis) or "none"
    s1 = scene_paths(cfg).product
    rel = lambda f: f.relative_to(cfg.root).as_posix()  # noqa: E731
    sea = labels[~labels.in_sar_land_buffer]
    counts = labels["size_class"].value_counts()
    counts_sea = sea["size_class"].value_counts()
    rows = [(name, paper["classes"].get(name, float("nan")), int(counts.get(name, 0)),
             int(counts_sea.get(name, 0))) for name, _, _ in SIZE_CLASSES]
    lines = [
        f"# {MARK}: Færch 2023 validation set rebuilt from Sentinel-2 (Path B)",
        "",
        "Generated by `python -m iceberg_sar.cli bench-labels`. The paper's 492-berg set with",
        "drift paths was not available, so it is rebuilt here. **Everything below and every file",
        "it lists is reconstructed labels, not the paper's validation data.** Scores against",
        "these labels are not directly comparable with Tables 3 and 5.",
        "",
        "## What the paper did (sec. 3.4, quoted from `faerch2023.xml`)",
        "",
        "> In the Sentinel-2 image icebergs were identified by applying an N-Sigma CFAR detector",
        "> to the sum of the high-resolution bands: B2, B3, B4, and B8. Pixels brighter than the",
        "> mean background plus 4 times the standard deviations were marked as possible icebergs.",
        "> The results were then manually checked to remove artifacts from clouds and land, and a",
        "> few icebergs missed by the automatic detection were added.",
        "",
        "> To correct for this drift, icebergs observed in the Sentinel-2 image were manually",
        "> matched to objects in the SAR images. [...] on a large scale, icebergs arranged in",
        "> clusters often drift in similar directions and over a similar distance. [...] A linear",
        "> interpolation method was used to predict the expected drift paths of these icebergs.",
        "",
        "## Steps used here",
        "",
        f"1. **Data:** Sentinel-2B L1C, 2019-07-04 15:36, tiles {tiles('table1')} (Table 1) and",
        f"   {tiles('fig1')} (not in Table 1, but inside the S2 outline of the paper's Fig. 1;",
        "   Table 1's five tiles cover about half of the paper's iceberg area). Original processing",
        "   baseline N0207 (Google public bucket); TOA reflectance = DN / 10000.",
        f"   SAR: `{s1}`, 40 m sigma0 from SNAP.",
        f"2. **N-sigma CFAR:** sum of {', '.join(p['bands'])}. Background mean and standard",
        f"   deviation over a square ring, {p['guard_m']} m guard / {p['outer_m']} m outer (the",
        "   paper gives no window for S2; this is its SAR window). A pixel is a target if it is",
        f"   above mean + {p['n_sigma']} sigma and at least {p['min_ring_fraction']:.0%} of its",
        "   ring is valid.",
        "3. **Instead of the manual check** (automatic rules, an approximation):",
        f"   - land: OSM land polygons + {p['land_buffer_m']} m excluded;",
        f"   - clouds: L1C cloud-mask polygons over {p['cloud_min_area_m2'] / 1e6:g} km² excluded"
        f" (+{p['cloud_buffer_m']} m); objects with mean {p['swir_band']} TOA reflectance over "
        f"{p['max_swir']} dropped (clouds are bright in SWIR, ice is dark);",
        f"   - objects of {p['min_area_px']} px or more and at most {p['max_area_m2']:g} m²; "
        "8-connected;",
        "   - the paper's manually added missed bergs cannot be reproduced.",
        "4. **Size:** length = major axis of the object (second moments), Table 2 classes.",
        "5. **AOI:** the paper's AOI is only drawn in its Fig. 1, where all icebergs lie inside",
        "   S1 x S2 x ALOS-2. Used here: S1 swath x usable S2 area (no land, large clouds or",
        "   nodata)" + (" x the ALOS-2 outline georeferenced from Fig. 1 (`bench-fig1`, a few km"
                      " accuracy)" if p.get("aoi_alos2") else "") +
        f", {aoi.area.sum() / 1e6:,.0f} km².",
        "6. **Drift, automated** (`drift.py`):",
        "   - bright SAR objects for matching: one global threshold on intensity normalized by "
        f"its median at the same incidence angle ({d['sar_threshold_db']} dB in both HH and HV,"
        f" {d['incidence_bin_deg']}° bins). Not one of the benchmarked detectors;",
        "   - local drift at nodes every "
        f"{d['node_spacing_m'] / 1000:g} km: the shift (within ±{d['search_m'] / 1000:g} km) "
        f"that lines up the most S2 labels within {d['neighbourhood_m'] / 1000:g} km of the "
        f"node with a SAR object ({d['verify_radius_m']} m), kept if at least "
        f"{d['min_support']} labels agree; ties go to the smaller drift "
        f"({c['n_drift_nodes']} nodes kept);",
        f"   - **verified:** label moved back by its node's drift, matched one-to-one to a SAR "
        f"object within {d['verify_radius_m']} m; its position is that SAR object's;",
        "   - **interpolated:** drift linearly interpolated (Delaunay) from the verified drift",
        "     vectors, nearest vector outside their hull;",
        f"   - search radii for scoring (1f): {d['radius_verified_m']} m verified, "
        f"{d['radius_interpolated_m']} m interpolated (paper sec. 3.4).",
        "",
        "## Counts next to the paper",
        "",
        "| | Paper | Rebuilt (all) | Rebuilt, outside the SAR land buffer |",
        "|---|---|---|---|",
    ]
    lines += [f"| {name} | {pv} | {a} | {b} |" for name, pv, a, b in rows]
    lines += [
        f"| **total** | {paper['total']} | {len(labels)} | {len(sea)} |",
        f"| verified drift paths (S1) | {paper['verified_s1']} | {c['n_verified']} | "
        f"{int((sea.status == 'verified').sum())} |",
        f"| mean drift, verified (m) | {paper['mean_drift_s1_m']} | "
        f"{c['mean_drift_verified_m']:.0f} | |",
        f"| interpolation error, 10 % holdout (m) | {paper['interp_error_s1_m']} | "
        f"{c['holdout_mean_error_m']:.0f} (median {c.get('holdout_median_error_m', float('nan')):.0f},"
        f" n={c['holdout_n']}) | |",
        f"| mean length (m) | {paper['mean_length_m']} | {labels.length_m.mean():.0f} | |",
        "",
        f"Bright SAR objects used for matching: {c['n_sar_targets']}. Median distance of a",
        f"verified match (moved-back label to SAR object): {c['median_match_dist_m']:.0f} m.",
        "",
        "## Known differences from the paper's labels",
        "",
        f"- **Drift matching is the weak step: {c['n_verified']} verified paths vs the paper's "
        f"{paper['verified_s1']}.** Most S2 bergs are small and do not stand out from speckle in",
        "  40 m EW HH/HV, so few line up with a SAR object. Looser settings (more SAR objects,",
        "  fewer labels required to agree) verified more, but with a mean drift near 7 km, which is",
        "  what random pairs in a ±10 km search give: chance alignments, not drift. The verified",
        "  paths that remain are mostly slow or grounded bergs near the coast. The other labels",
        "  keep a drift interpolated from these few vectors, so their positions are uncertain by",
        f"  more than the {d['radius_interpolated_m']} m radius suggests. The holdout error "
        f"(n={c['holdout_n']}) is too small a sample to mean much.",
        "- Cloud and land artifacts are removed by rules, not by eye; missed bergs are not added.",
        "  A gallery of 30 random large labels (2026-10-23) had 2 artifacts: a rocky islet missing",
        "  from OSM, and an aircraft streak (moving objects smear across the S2 bands).",
        f"- Sizes: mean length {labels.length_m.mean():.0f} m vs the paper's "
        f"{paper['mean_length_m']} m. The 4-sigma region",
        "  includes half-bright edge pixels (about +10 m per side). Shortening every length by",
        "  20 m matches the paper's mean but not its class split (more small, still more large),",
        "  so lengths are kept as measured and the size classes are compared with that caveat.",
        "- Verified paths require a SAR object found by one global threshold, so verified labels",
        "  lean towards bergs that are bright in SAR. The paper's manual matching has the same",
        "  lean (it notes small bergs were hard to match), but not necessarily the same amount.",
        "- The AOI is approximated (see step 5). Labels inside the SAR land buffer are kept and",
        "  flagged (`in_sar_land_buffer`); whether the paper counted them is not stated.",
        "- Ship removal and removal of bergs drifting in from outside the AOI were not done.",
        "",
        "## Files (local, gitignored)",
        "",
    ]
    lines += [f"- `{rel(f)}`" for f in files.values()]
    lines += [f"- `{rel(f)}`" for f in pngs.values()]
    lines += ["", "Checks (JSON):", "", "```json", json.dumps(c, indent=2), "```", ""]
    return "\n".join(lines)
