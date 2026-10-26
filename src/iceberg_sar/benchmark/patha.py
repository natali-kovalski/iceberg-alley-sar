"""Milestone 1d, Path A: the paper's own validation set, from Laust Færch (by email, 2026-10-24).

Private data: kept under `data/` (gitignored), never committed or redistributed.

Input (`labels_faerch.dir`, EPSG:5938 polar stereographic, the paper's working grid):
  matching-sentinel2-sentinel1-interpolated.shp  492 lines, S2 position -> S1 position.
      `method` is "manual" (verified drift path) or "interpolate" (expected drift path);
      `length`, `width`, `area` are in Sentinel-2 pixels.
  sentinel1-unknown-objects.shp  bright SAR objects without an optical match (paper sec. 3.4:
      "A few bright objects in the SAR images that were covered by clouds in the optical data
      were masked out"). Some rows have no geometry; those are dropped.

The paper's lengths and drifts are EPSG:5938 metres, which are ~7.5 % longer than true metres
at 58° N (scale factor 1.068-1.078 here). Both are kept: `length_m` / `drift_m` are true metres,
`length_m_paper` / `drift_m_paper` are the paper's. Size classes use the paper's metres, which
reproduces Table 2 exactly.

Outputs under `labels_faerch.out_dir`, same point schema as `labels_reconstructed.geojson`
(the 1f scorer reads either): labels_faerch.geojson, drift_paths_faerch.geojson,
unknown_objects_faerch.geojson, and `labels_faerch.report` (Markdown, committed: counts only).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import Proj
from shapely.geometry import LineString

from iceberg_sar.benchmark.labels import SIZE_CLASSES, size_class
from iceberg_sar.benchmark.pathb import _flags, _quicklooks, paper_numbers
from iceberg_sar.benchmark.reproduce import scene_paths
from iceberg_sar.config import Config

LABEL_SOURCE = "Færch 2023 (Path A, author's validation set)"


def faerch_labels(path: Path, crs: str, s2_pixel_m: float, verified_methods: Sequence[str],
                  radius_verified_m: float, radius_interpolated_m: float) -> gpd.GeoDataFrame:
    """Drift lines -> one point per berg at its S1 position, in `crs`."""
    src = gpd.read_file(path)
    if src.crs is None:
        raise ValueError(f"{path} has no CRS (.prj missing?)")
    if not (src.geom_type == "LineString").all():
        raise ValueError(f"{path}: expected LineStrings (S2 -> S1), got {src.geom_type.unique()}")
    ll = src.to_crs("EPSG:4326")
    lon = np.array([g.coords[0][0] for g in ll.geometry])
    lat = np.array([g.coords[0][1] for g in ll.geometry])
    k = Proj(src.crs).get_factors(lon, lat).meridional_scale  # paper metres per true metre

    out = src.to_crs(crs)
    s2 = np.array([g.coords[0] for g in out.geometry])
    s1 = np.array([g.coords[-1] for g in out.geometry])
    verified = src["method"].isin(verified_methods).to_numpy()
    length_m = src["length"].to_numpy(float) * s2_pixel_m
    labels = gpd.GeoDataFrame({
        "label_id": src["id"].str.extract(r"(\d+)$")[0].astype(int),
        "s2_id": src["id"],
        "method": src["method"],
        "status": np.where(verified, "verified", "interpolated"),
        "radius_m": np.where(verified, radius_verified_m, radius_interpolated_m),
        "s2_x": s2[:, 0], "s2_y": s2[:, 1],
        "drift_dx_m": s1[:, 0] - s2[:, 0], "drift_dy_m": s1[:, 1] - s2[:, 1],
        "drift_m": np.hypot(*(s1 - s2).T),
        "drift_m_paper": src.geometry.length.to_numpy(),
        "length_m": length_m,
        "length_m_paper": length_m * k,
        "width_m": src["width"].to_numpy(float) * s2_pixel_m,
        "area_m2": src["area"].to_numpy(float) * s2_pixel_m**2,
        "shape": src["shape"],
        "size_class": size_class(length_m * k),
        "label_source": LABEL_SOURCE,
    }, geometry=gpd.points_from_xy(s1[:, 0], s1[:, 1]), crs=crs)
    if not labels.label_id.is_unique:
        raise ValueError(f"{path}: duplicate ids")
    return labels


def faerch_unknown(path: Path, crs: str) -> tuple[gpd.GeoDataFrame, int]:
    """Unknown SAR objects with a geometry, and how many rows had none."""
    src = gpd.read_file(path)
    keep = src.geometry.notna() & ~src.geometry.is_empty
    return src[keep].to_crs(crs)[["geometry"]].assign(label_source=LABEL_SOURCE), int((~keep).sum())


def compare_pathb(labels: gpd.GeoDataFrame, rebuilt: gpd.GeoDataFrame, max_m: float
                  ) -> dict[str, Any]:
    """How the Path B rebuild compares: S2 detection, and S1 position where both have the berg."""
    s2 = gpd.GeoDataFrame({"x1": labels.geometry.x, "y1": labels.geometry.y},
                          geometry=gpd.points_from_xy(labels.s2_x, labels.s2_y), crs=labels.crs)
    rb = gpd.GeoDataFrame({"rb_status": rebuilt.status.to_numpy(),
                           "rx1": rebuilt.geometry.x.to_numpy(),
                           "ry1": rebuilt.geometry.y.to_numpy()},
                          geometry=gpd.points_from_xy(rebuilt.s2_x, rebuilt.s2_y), crs=labels.crs)
    j = gpd.sjoin_nearest(s2, rb, max_distance=max_m, distance_col="d").sort_values("d")
    j = j[~j.index.duplicated()].drop_duplicates("index_right")  # one-to-one, nearest first
    j["s1_err_m"] = np.hypot(j.x1 - j.rx1, j.y1 - j.ry1)
    by = {st: {"n": int(len(g)), "median_s1_err_m": float(g.s1_err_m.median()),
               "within_250_m": int((g.s1_err_m <= 250).sum())}
          for st, g in j.groupby("rb_status")}
    return {"n_faerch": len(labels), "n_rebuilt": len(rebuilt), "s2_matched": int(len(j)),
            "max_m": max_m, "by_rebuilt_status": by}


def run_patha(cfg: Config, log: Any = print) -> dict[str, Any]:
    fc = cfg.section("benchmark")["faerch2023"]
    p = fc["labels_faerch"]
    sp = scene_paths(cfg)
    src_dir = cfg.root / p["dir"]
    for f in (src_dir / p["drift_paths"], src_dir / p["unknown_objects"], sp.stack, sp.land,
              sp.swath):
        if not f.is_file():
            raise FileNotFoundError(f"{f} missing")
    crs = fc["crs"]
    labels = faerch_labels(src_dir / p["drift_paths"], crs, float(p["s2_pixel_m"]),
                           p["verified_methods"], float(p["radius_verified_m"]),
                           float(p["radius_interpolated_m"]))
    labels = _flags(labels, sp.land, sp.swath)
    unknown, n_unknown_empty = faerch_unknown(src_dir / p["unknown_objects"], crs)
    log(f"{len(labels)} labels ({(labels.status == 'verified').sum()} verified), "
        f"{len(unknown)} unknown SAR objects ({n_unknown_empty} rows without geometry dropped)")

    out_dir = cfg.root / p["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    def save(name: str, gdf: gpd.GeoDataFrame) -> None:
        files[name] = out_dir / name
        gdf.to_crs("EPSG:4326").to_file(files[name], driver="GeoJSON")

    save("labels_faerch.geojson", labels)
    lines = [LineString([(a, b), (g.x, g.y)]) for a, b, g in
             zip(labels.s2_x, labels.s2_y, labels.geometry, strict=True)]
    save("drift_paths_faerch.geojson", gpd.GeoDataFrame(
        labels[["label_id", "status", "drift_m", "label_source"]], geometry=lines, crs=crs))
    save("unknown_objects_faerch.geojson", unknown)

    rebuilt_path = cfg.root / fc["labels"]["dir"] / "labels_reconstructed.geojson"
    cmp = None
    if rebuilt_path.is_file():
        cmp = compare_pathb(labels, gpd.read_file(rebuilt_path).to_crs(crs), float(p["compare_m"]))
        log(f"Path B check: {cmp['s2_matched']} / {len(labels)} S2 positions found by the rebuild")

    pngs = _quicklooks(sp, labels, None, out_dir.parent / "quicklooks", int(fc["quicklook_max_px"]),
                       stem="labels_faerch",
                       caption="Færch 2023 validation set (Path A) at the S1 pass. HH dB, 40 m",
                       zoom_title="Færch 2023 labels: drift S2 (dot) -> S1 (ring), 20 km zoom",
                       extra=unknown)
    paper = paper_numbers(cfg.root / fc["paper_xml"])
    report = cfg.root / p["report"]
    report.write_text(_report(cfg, labels, n_unknown_empty, unknown, paper, cmp, files, pngs),
                      encoding="utf-8")
    return {"files": files, "pngs": pngs, "report": report}


def _report(cfg: Config, labels: gpd.GeoDataFrame, n_unknown_empty: int,
            unknown: gpd.GeoDataFrame, paper: dict[str, Any], cmp: dict[str, Any] | None,
            files: dict[str, Path], pngs: dict[str, Path]) -> str:
    fc = cfg.section("benchmark")["faerch2023"]
    p = fc["labels_faerch"]
    rel = lambda f: f.relative_to(cfg.root).as_posix()  # noqa: E731
    ver = labels.status == "verified"
    sea = ~labels.in_sar_land_buffer
    counts = labels.size_class.value_counts()
    counts_true = labels.assign(c=size_class(labels.length_m.to_numpy())).c.value_counts()
    big_ver = int((ver & (labels.length_m_paper >= 50)).sum())
    lines = [
        "# Færch 2023 validation set (Path A)",
        "",
        "Generated by `python -m iceberg_sar.cli bench-labels --source faerch`. The paper's own",
        "labels, sent by Laust Færch on 2026-10-24. **Private: the data stays in `data/` and is",
        "never committed.** This file holds counts only.",
        "",
        f"Source: `{p['drift_paths']}` (one line per berg, S2 position -> S1 position) and",
        f"`{p['unknown_objects']}`, CRS EPSG:5938 (polar stereographic, the paper's grid).",
        f"Loaded into the scene grid ({fc['crs']}); the S1 end of each line is the label.",
        "",
        "## Counts next to the paper",
        "",
        "| | Paper | Færch file | Outside the SAR land buffer |",
        "|---|---|---|---|",
    ]
    lines += [f"| {n} (Table 2) | {paper['classes'].get(n)} | {int(counts.get(n, 0))} | "
              f"{int(labels[sea].size_class.value_counts().get(n, 0))} |"
              for n, _, _ in SIZE_CLASSES]
    lines += [
        f"| **total** | {paper['total']} | {len(labels)} | {int(sea.sum())} |",
        f"| verified drift paths (`method` in {p['verified_methods']}) | {paper['verified_s1']} | "
        f"{int(ver.sum())} | {int((ver & sea).sum())} |",
        f"| ...of which >= 50 m long | | {big_ver} | |",
        f"| mean drift, verified (paper m) | {paper['mean_drift_s1_m']} | "
        f"{labels[ver].drift_m_paper.mean():.0f} | |",
        f"| mean length (paper m) | {paper['mean_length_m']} | "
        f"{labels.length_m_paper.mean():.1f} | |",
        f"| in the S1 swath | | {int(labels.in_swath.sum())} | |",
        "",
        "## Notes",
        "",
        f"- **Units.** `length` is in S2 pixels ({p['s2_pixel_m']} m). The paper measured",
        "  length",
        "  and drift in EPSG:5938 metres, which are "
        f"{(labels.length_m_paper / labels.length_m).median():.3f}x true metres here. With that",
        "  factor the size classes match Table 2 exactly; in true metres they are "
        + "/".join(str(int(counts_true.get(n, 0))) for n, _, _ in SIZE_CLASSES) + ".",
        f"  Mean verified drift in true metres: {labels[ver].drift_m.mean():.0f} m.",
        f"- **Verified count.** {int(ver.sum())} paths are `manual`, the paper says"
        f" {paper['verified_s1']}.",
        f"  {big_ver} of them are bergs of at least 50 m; the paper says",
        "  bergs under 50 m were interpolated (sec. 5.1), so its 270 probably counts only those.",
        "  The scorer uses the file's `method` (250 m radius for all manual paths); ask Laust if",
        "  the small manual ones should take the 1789 m radius.",
        "- **Search radii** (250 m / 1789 m, sec. 3.4) were probably set in EPSG:5938 metres too,",
        f"  i.e. ~{250 / (labels.length_m_paper / labels.length_m).median():.0f} m /"
        f" {1789 / (labels.length_m_paper / labels.length_m).median():.0f} m true. Small"
        " difference; the scorer keeps the paper's values.",
        f"- **Unknown SAR objects:** {len(unknown)} with a position, {n_unknown_empty} rows",
        "  without geometry (dropped). The paper masked bright SAR objects under clouds out of",
        "  the scoring.",
        f"- {int((labels.area_m2 == 0).sum())} labels have area 0 and no width ("
        + ", ".join(f"{n} {m}" for m, n in
                    labels[labels.area_m2 == 0].method.value_counts().items())
        + "): probably the bergs the paper added by hand after the S2 detection. They keep",
        "  their `length`.",
    ]
    if cmp:
        b = cmp["by_rebuilt_status"]
        lines += [
            "",
            "## How the Path B rebuild compares",
            "",
            f"S2 detection: {cmp['s2_matched']} of the {cmp['n_faerch']} Færch bergs have a rebuilt"
            f" S2 target within {cmp['max_m']:g} m (the rebuild has {cmp['n_rebuilt']}).",
            "For those, the distance between the two S1 positions:",
            "",
            "| Path B status | n | median S1 difference (m) | within 250 m |",
            "|---|---|---|---|",
        ]
        lines += [f"| {st} | {v['n']} | {v['median_s1_err_m']:.0f} | {v['within_250_m']} |"
                  for st, v in b.items()]
    lines += ["", "## Files (local, gitignored)", ""]
    lines += [f"- `{rel(f)}`" for f in [*files.values(), *pngs.values()]]
    if cmp:
        lines += ["", "Path B check (JSON):", "", "```json", json.dumps(cmp, indent=2), "```"]
    return "\n".join(lines) + "\n"
