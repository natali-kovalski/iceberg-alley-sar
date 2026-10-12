"""Run CFAR over a masked scene and turn detected pixels into point targets with attributes.

The scene is processed in row blocks with a halo, so memory stays bounded on 12k x 12k
scenes. Each block reads `halo` extra rows above and below:
- CFAR needs `background_px` rows of context, and `edge_buffer_px` more for the
  near-invalid test, so detections within that margin of the block's edge are untrusted.
- A target is kept by the block whose *core* rows contain its topmost pixel, and only if
  it ends inside the trusted region. Targets taller than `max_extent_px` are dropped.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from iceberg_sar.cfar import CfarParams, ring_mean
from iceberg_sar.config import Config, metres_to_px
from iceberg_sar.seaice import distance_to_ice_km

SCENE_TIME = re.compile(r"_(\d{8}T\d{6})_")
COPOL = {"HV": "HH", "VH": "VV"}  # co-pol partner of each cross-pol CFAR band
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class DetectionParams:
    min_area_px: int
    max_area_px: int
    edge_buffer_px: int  # drop targets this close to any masked/nodata pixel
    max_extent_px: int = 64  # tallest target a block can hold; bigger ones are dropped
    grow_db: float = 4.0  # region-growing level above background for `bright_structures`
    max_structure_px: int = 15  # reject targets on bright structures longer than this
    copol_min_contrast_db: float | None = None  # co-pol peak over co-pol background; None = off
    merge_by_structure: bool = True  # CFAR cores in one grown blob = one target
    block_rows: int = 1024


def scene_timestamp(name: str) -> str:
    """ISO 8601 UTC acquisition start from a Sentinel-1 / HyP3 product name."""
    m = SCENE_TIME.search(f"_{name}_")
    if not m:
        raise ValueError(f"No acquisition time found in '{name}'")
    t = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    return t.isoformat().replace("+00:00", "Z")


def to_db(x: np.ndarray | float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(np.asarray(x) > 0, x, np.nan))


def near_invalid(valid: np.ndarray, buffer_px: int) -> np.ndarray:
    """True within `buffer_px` (square) of any invalid pixel or the array edge."""
    if buffer_px <= 0:
        return ~valid
    size = 2 * buffer_px + 1
    frac = ndimage.uniform_filter((~valid).astype(np.float32), size=size, mode="constant", cval=1.0)
    return frac > 1e-6


def bright_structures(x: np.ndarray, background: np.ndarray, grow_db: float) -> np.ndarray:
    """3x3-smoothed intensity more than `grow_db` above the CFAR background.

    CFAR only flags a target's brightest pixels. Growing to this lower level shows what the
    target is part of: a compact blob (berg, ship) or a long filament (sea-ice strip, where
    CFAR catches just a few bright knots).
    """
    valid = np.isfinite(x)
    s = ndimage.uniform_filter(np.where(valid, x, 0.0).astype(np.float32), size=3)
    c = ndimage.uniform_filter(valid.astype(np.float32), size=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = s / c
        return valid & (smooth > background * 10 ** (grow_db / 10))


def _extent(slices: list[tuple[slice, slice] | None]) -> np.ndarray:
    return np.array([0 if s is None else max(s[0].stop - s[0].start, s[1].stop - s[1].start)
                     for s in slices])


class _LabelStats:
    """Per-label sum / max / mean over the labelled pixels only.

    Same results as `ndimage.sum_labels` etc., but those scan the whole block on every call;
    on a 10 m scene (40 M px per block, targets a tiny fraction of it) that dominated the run.
    """

    def __init__(self, labels: np.ndarray, n: int) -> None:
        self.flat = np.flatnonzero(labels)
        self.lab = labels.ravel()[self.flat]
        self.n = n
        self.rows, self.cols = np.divmod(self.flat, labels.shape[1])

    def _vals(self, a: np.ndarray) -> np.ndarray:
        return a.ravel()[self.flat].astype(np.float64)

    def sum(self, a: np.ndarray | None = None, vals: np.ndarray | None = None) -> np.ndarray:
        v = self._vals(a) if vals is None else vals
        return np.bincount(self.lab, weights=v, minlength=self.n + 1)[1:]

    def count(self) -> np.ndarray:
        return np.bincount(self.lab, minlength=self.n + 1)[1:].astype(np.float64)

    def max(self, a: np.ndarray) -> np.ndarray:
        out = np.full(self.n + 1, -np.inf)
        np.maximum.at(out, self.lab, self._vals(a))
        return out[1:]

    def mean(self, a: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sum(a) / self.count()

    def centre_of_mass(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self._vals(weights)
        total = self.sum(vals=w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sum(vals=w * self.rows) / total, self.sum(vals=w * self.cols) / total


def component_table(
    det: np.ndarray,
    bands: dict[str, np.ndarray],
    cfar_band: str,
    background: np.ndarray,
    near_edge: np.ndarray,
    structures: np.ndarray,
    row_offset: int,
    keep_rows: tuple[int, int],
    trusted_stop: int,
    p: DetectionParams,
    copol: tuple[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-target stats for connected components of `det` in one block (pixel coords global).

    `keep_rows` = [start, stop) of this block's core rows (global); `trusted_stop` = first
    untrusted global row at the bottom of the block. `structures` is the grown mask from
    `bright_structures`; a target's structure extent is that of the grown blob containing it.
    `copol` = (band, ring-mean background) of the co-pol band, for the co-pol check.
    With `p.merge_by_structure`, CFAR cores in the same grown blob form one target: a ship
    (or berg) often breaks into two cores tens of metres apart in azimuth.
    """
    grown, _ = ndimage.label(structures | det, structure=EIGHT_CONNECTED)
    if p.merge_by_structure:
        blob_ids = np.unique(grown[det])
        n = len(blob_ids)
        labels = np.zeros(det.shape, dtype=np.int32)
        labels[det] = np.searchsorted(blob_ids, grown[det]) + 1
    else:
        labels, n = ndimage.label(det, structure=EIGHT_CONNECTED)
    if n == 0:
        return pd.DataFrame()
    slices = ndimage.find_objects(labels)
    top = np.array([s[0].start for s in slices]) + row_offset
    bottom = np.array([s[0].stop for s in slices]) + row_offset
    height = np.array([s[0].stop - s[0].start for s in slices])
    width = np.array([s[1].stop - s[1].start for s in slices])
    st = _LabelStats(labels, n)
    area = st.count().astype(int)   # labels cover exactly the det pixels

    owned = (top >= keep_rows[0]) & (top < keep_rows[1]) & (bottom <= trusted_stop)
    touches_edge = st.max(near_edge) > 0
    blob = st.max(grown).astype(int)  # each target lies in one blob
    structure = _extent(ndimage.find_objects(grown))[blob - 1]
    keep = (
        owned
        & ~touches_edge
        & (area >= p.min_area_px)
        & (area <= p.max_area_px)
        & (height <= p.max_extent_px)
        & (structure <= p.max_structure_px)
    )
    if not keep.any():
        return pd.DataFrame()
    com_r, com_c = st.centre_of_mass(np.nan_to_num(bands[cfar_band], nan=0.0))
    out = pd.DataFrame({
        "row": com_r[keep] + row_offset,
        "col": com_c[keep],
        "area_px": area[keep],
        "extent_px": np.maximum(height, width)[keep],
        "structure_px": structure[keep],
        "background_db": to_db(st.mean(background)[keep]),
    })
    for pol, arr in bands.items():
        finite = np.isfinite(arr)
        vals = np.where(finite, arr, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = st.sum(vals) / st.sum(finite)
        out[f"peak_db_{pol}"] = to_db(st.max(vals)[keep])
        out[f"mean_db_{pol}"] = to_db(mean[keep])
    out["contrast_db"] = out[f"peak_db_{cfar_band}"] - out["background_db"]
    if copol is not None:
        pol, bg = copol
        out["copol_contrast_db"] = out[f"peak_db_{pol}"] - to_db(st.mean(bg)[keep])
        if p.copol_min_contrast_db is not None:
            # Icebergs stand out in co- and cross-pol; HV speckle spikes have no HH support.
            # NaN (no co-pol background) fails the test, like a masked CFAR ring.
            out = out[out["copol_contrast_db"] >= p.copol_min_contrast_db]
    return out


def detect_raster(
    paths: dict[str, Path],
    cfar_band: str,
    cfar: CfarParams,
    p: DetectionParams,
) -> tuple[pd.DataFrame, rasterio.Affine, rasterio.crs.CRS]:
    """Block-wise CFAR + component extraction over co-registered masked rasters."""
    tables, transform, crs = detect_raster_multi(paths, cfar_band, [cfar], p)
    return tables[0], transform, crs


def detect_raster_multi(
    paths: dict[str, Path],
    cfar_band: str,
    cfars: list[CfarParams],
    p: DetectionParams,
) -> tuple[list[pd.DataFrame], rasterio.Affine, rasterio.crs.CRS]:
    """`detect_raster` at several thresholds (e.g. a PFA sweep) in one pass over the scene.

    The CFAR windows must match; only the threshold differs. Background, grown structures and
    the co-pol background are computed once per block, so each extra threshold only costs its
    component table.
    """
    cfar = cfars[0]
    if any((c.guard_px, c.background_px, c.min_background_fraction)
           != (cfar.guard_px, cfar.background_px, cfar.min_background_fraction) for c in cfars):
        raise ValueError("All CFAR settings must share the windows and background fraction")
    halo = cfar.background_px + p.edge_buffer_px + p.max_extent_px
    copol_band = COPOL.get(cfar_band, "")
    trusted_margin = cfar.background_px + p.edge_buffer_px
    srcs = {pol: rasterio.open(path) for pol, path in paths.items()}
    try:
        ref = srcs[cfar_band]
        h, w = ref.height, ref.width
        for pol, s in srcs.items():
            if (s.height, s.width) != (h, w) or s.transform != ref.transform:
                raise ValueError(f"Band {pol} is not on the same grid as {cfar_band}")
        tables: list[list[pd.DataFrame]] = [[] for _ in cfars]
        for r0 in range(0, h, p.block_rows):
            r1 = min(r0 + p.block_rows, h)
            w0, w1 = max(0, r0 - halo), min(h, r1 + halo)
            win = Window(0, w0, w, w1 - w0)
            bands = {pol: s.read(1, window=win).astype(np.float32) for pol, s in srcs.items()}
            x = bands[cfar_band]
            if not np.isfinite(x).any():
                continue
            bg, n = ring_mean(x, cfar.guard_px, cfar.background_px)
            candidate = np.isfinite(x) & (n >= cfar.min_background_fraction * cfar.ring_pixels)
            edge = near_invalid(np.isfinite(x), p.edge_buffer_px)
            grown = bright_structures(x, bg, p.grow_db)
            trusted_stop = h if w1 == h else w1 - trusted_margin
            copol = None
            for i, c in enumerate(cfars):
                with np.errstate(invalid="ignore"):
                    det = candidate & (x > c.alpha * bg)   # = ca_cfar(x, c)
                if copol is None and copol_band in bands and det.any():
                    copol_bg, _ = ring_mean(bands[copol_band], cfar.guard_px, cfar.background_px)
                    copol = (copol_band, copol_bg)
                t = component_table(det, bands, cfar_band, bg, edge, grown, w0, (r0, r1),
                                    trusted_stop, p, copol)
                if not t.empty:
                    tables[i].append(t)
        return ([pd.concat(t, ignore_index=True) if t else pd.DataFrame() for t in tables],
                ref.transform, ref.crs)
    finally:
        for s in srcs.values():
            s.close()


def sample_incidence_deg(inc_path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Incidence angle (degrees) at map coordinates. HyP3 inc_map is in radians; 0 = nodata."""
    with rasterio.open(inc_path) as src:
        vals = np.array([v[0] for v in src.sample(zip(xs, ys, strict=True))], dtype=np.float64)
        nodata = src.nodata
    bad = (vals == 0) | ~np.isfinite(vals)
    if nodata is not None:
        bad |= vals == nodata
    radians = np.nanmax(np.where(bad, np.nan, vals), initial=0) < 2 * np.pi
    deg = np.degrees(vals) if radians else vals
    return np.where(bad, np.nan, deg)


def to_geodataframe(
    table: pd.DataFrame,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    scene_id: str,
    inc_path: Path | None,
    ice_mask: Path | None = None,
) -> gpd.GeoDataFrame:
    """Pixel-space table -> EPSG:4326 points with the attributes used downstream.

    `ice_mask` is the Milestone 1 pack-ice mask (same CRS as the scene); without it
    `distance_to_ice_km` is NaN.
    """
    timestamp = scene_timestamp(scene_id)
    if table.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    table = table.sort_values(["row", "col"]).reset_index(drop=True)
    xs, ys = rasterio.transform.xy(transform, table["row"].to_numpy(), table["col"].to_numpy(),
                                   offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    px_m2 = abs(transform.a * transform.e)
    gdf = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(xs, ys), crs=crs).to_crs("EPSG:4326")

    stamp = timestamp[:19].replace("-", "").replace(":", "")
    gdf.insert(0, "id", [f"{stamp}_{i:05d}" for i in range(len(gdf))])
    gdf.insert(1, "scene_id", scene_id)
    gdf.insert(2, "timestamp", timestamp)
    gdf.insert(3, "lon", gdf.geometry.x.round(6))
    gdf.insert(4, "lat", gdf.geometry.y.round(6))
    gdf["area_m2"] = gdf["area_px"] * px_m2
    gdf["extent_m"] = gdf["extent_px"] * abs(transform.a)
    gdf["structure_m"] = gdf["structure_px"] * abs(transform.a)
    gdf["incidence_deg"] = sample_incidence_deg(inc_path, xs, ys) if inc_path else np.nan
    gdf["distance_to_ice_km"] = distance_to_ice_km(ice_mask, xs, ys) if ice_mask else np.nan
    gdf["row"] = gdf["row"].round(2)
    gdf["col"] = gdf["col"].round(2)
    num = gdf.select_dtypes("float").columns.difference(["lon", "lat", "row", "col"])
    gdf[num] = gdf[num].round(2)
    return gdf


def resolve_enl(enl_cfg: float | str, cfar_path: Path, cell_px: int) -> float:
    """Config `enl`: a number, or "auto" to estimate it from the scene's water pixels."""
    if enl_cfg != "auto":
        return float(enl_cfg)
    from iceberg_sar.cfar import estimate_enl
    from iceberg_sar.seaice import coarse_stats

    return estimate_enl(coarse_stats(cfar_path, cell_px, min_valid_frac=1.0).cv)


def save_overlay(
    db: np.ndarray,
    factor: int,
    vmin: float,
    vmax: float,
    gdf: gpd.GeoDataFrame,
    png_path: Path,
    title: str,
) -> Path:
    """dB quicklook with detections circled (pixel coords scaled to the quicklook)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = db.shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(db, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if not gdf.empty:
        ax.scatter(gdf["col"] / factor, gdf["row"] / factor, s=120, facecolors="none",
                   edgecolors="#ff3b30", linewidths=1.2)
    ax.text(20, 50, title, color="yellow", fontsize=28, va="top")
    ax.set_axis_off()
    fig.savefig(png_path, dpi=dpi, facecolor="black")
    plt.close(fig)
    return png_path


def _product_setup(
    product_dir: Path, cfg: Config, band: str | None,
) -> tuple[dict[str, Path], str, float, CfarParams, DetectionParams]:
    """Masked rasters, CFAR band, pixel size, CFAR and detection parameters from the config."""
    name = product_dir.name
    interim = cfg.path("interim") / name
    masked = {
        pol: interim / f"{name}_{pol}_masked.tif"
        for pol in ("HH", "HV", "VV", "VH")
        if (interim / f"{name}_{pol}_masked.tif").is_file()
    }
    if not masked:
        raise FileNotFoundError(f"No masked rasters in {interim}; run `preprocess` first")
    c, d = cfg.section("cfar"), cfg.section("detections")
    if c.get("variant", "CA") != "CA":
        raise NotImplementedError(f"CFAR variant {c['variant']} (only CA so far)")
    fallback = {"HV": "VH", "VH": "HV", "HH": "VV", "VV": "HH"}
    want = (band or c["band"]).upper()
    band = want if want in masked else fallback[want]
    if band not in masked:
        raise ValueError(f"CFAR band {want} (or {band}) not in {sorted(masked)}")

    with rasterio.open(masked[band]) as src:
        pixel_m = abs(src.transform.a)

    def px(metres: float) -> int:
        return metres_to_px(float(metres), pixel_m)

    enl = resolve_enl(c.get("enl", "auto"), masked[band], px(c.get("enl_cell_m", 500)))
    cfar = CfarParams(
        guard_px=px(c["guard_m"]), background_px=px(c["background_m"]),
        pfa=float(c["pfa"]), enl=enl,
        min_background_fraction=float(c.get("min_background_fraction", 0.5)),
    )
    params = DetectionParams(
        min_area_px=int(d["min_area_px"]),
        max_area_px=metres_to_px(float(d["max_area_m2"]), pixel_m**2),
        edge_buffer_px=px(d["edge_buffer_m"]), max_extent_px=px(d.get("max_extent_m", 1280)),
        grow_db=float(d.get("grow_db", 4.0)),
        max_structure_px=px(d.get("max_structure_m", 300)),
        copol_min_contrast_db=(None if d.get("copol_min_contrast_db") is None
                               else float(d["copol_min_contrast_db"])),
        merge_by_structure=bool(d.get("merge_by_structure", True)),
        block_rows=int(d.get("block_rows", 1024)),
    )
    return masked, band, pixel_m, cfar, params


def detect_product(product_dir: Path, cfg: Config, band: str | None = None) -> dict[str, Path]:
    """CFAR detection on a preprocessed HyP3 product; writes GeoJSON, overlay PNG, summary JSON.

    `band` overrides `cfar.band` from the config (e.g. to compare HH vs HV).
    """
    import json

    from iceberg_sar.preprocess import db_stretch, find_rtc_bands, read_downsampled_db

    product_dir = Path(product_dir)
    name = product_dir.name
    interim = cfg.path("interim") / name
    masked, band, pixel_m, cfar, params = _product_setup(product_dir, cfg, band)
    table, transform, crs = detect_raster(masked, band, cfar, params)
    inc = find_rtc_bands(product_dir).get("inc") if product_dir.is_dir() else None
    ice_mask = interim / f"{name}_seaice_mask.tif"
    gdf = to_geodataframe(table, transform, crs, name, inc,
                          ice_mask if ice_mask.is_file() else None)

    out_dir = cfg.path("outputs") / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson = out_dir / f"{name}_{band}_detections.geojson"
    if gdf.empty:
        geojson.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    else:
        gdf.to_file(geojson, driver="GeoJSON")

    p = cfg.section("preprocess")
    db, factor = read_downsampled_db(masked[band], p["quicklook_max_px"], p["quicklook_pool"])
    vmin, vmax = db_stretch(db, p.get("quicklook_db_range"))
    title = f"{name}  {band}  CA-CFAR pfa={cfar.pfa:g}  n={len(gdf)}"
    overlay = save_overlay(db, factor, vmin, vmax, gdf, out_dir / f"{name}_{band}_detections.png",
                           title)

    summary = {
        "product": name,
        "timestamp": scene_timestamp(name),
        "cfar_band": band,
        "pixel_m": pixel_m,
        "guard_px": cfar.guard_px,
        "background_px": cfar.background_px,
        "pfa": cfar.pfa,
        "enl": round(cfar.enl, 3),
        "alpha": round(cfar.alpha, 3),
        "alpha_db": round(float(to_db(cfar.alpha)), 2),
        "n_detections": len(gdf),
        "area_px_quantiles": (gdf["area_px"].quantile([0.1, 0.5, 0.9]).tolist()
                              if len(gdf) else []),
    }
    summary_path = out_dir / f"{name}_{band}_detections.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"detections": geojson, "overlay": overlay, "summary": summary_path}


def detect_product_sweep(
    product_dir: Path, cfg: Config, pfas: list[float], out_dir_for: Callable[[float], Path],
    band: str | None = None,
) -> dict[float, Path]:
    """`detect_product` at several PFAs in one pass (no overlay PNG). Each PFA's GeoJSON and
    summary JSON go to `out_dir_for(pfa)` under the usual file names."""
    import json

    from iceberg_sar.preprocess import find_rtc_bands

    product_dir = Path(product_dir)
    name = product_dir.name
    interim = cfg.path("interim") / name
    masked, band, pixel_m, base, params = _product_setup(product_dir, cfg, band)
    cfars = [replace(base, pfa=float(pfa)) for pfa in pfas]
    tables, transform, crs = detect_raster_multi(masked, band, cfars, params)
    inc = find_rtc_bands(product_dir).get("inc") if product_dir.is_dir() else None
    ice_mask = interim / f"{name}_seaice_mask.tif"
    # One GeoDataFrame for the whole sweep, so incidence and distance to ice (a full-scene
    # distance transform) are computed once, not per PFA.
    tagged = [t.assign(sweep_k=k) for k, t in enumerate(tables) if not t.empty]
    allg = to_geodataframe(pd.concat(tagged, ignore_index=True) if tagged else pd.DataFrame(),
                           transform, crs, name, inc, ice_mask if ice_mask.is_file() else None)
    out = {}
    for k, c in enumerate(cfars):
        gdf = allg[allg["sweep_k"] == k].drop(columns="sweep_k") if len(allg) else allg
        d = Path(out_dir_for(c.pfa))
        d.mkdir(parents=True, exist_ok=True)
        geojson = d / f"{name}_{band}_detections.geojson"
        if gdf.empty:
            geojson.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
        else:
            gdf.to_file(geojson, driver="GeoJSON")
        summary = {"product": name, "timestamp": scene_timestamp(name), "cfar_band": band,
                   "pixel_m": pixel_m, "guard_px": c.guard_px, "background_px": c.background_px,
                   "pfa": c.pfa, "enl": round(c.enl, 3), "alpha": round(c.alpha, 3),
                   "alpha_db": round(float(to_db(c.alpha)), 2), "n_detections": len(gdf)}
        (d / f"{name}_{band}_detections.json").write_text(json.dumps(summary, indent=2),
                                                           encoding="utf-8")
        out[c.pfa] = geojson
    return out
