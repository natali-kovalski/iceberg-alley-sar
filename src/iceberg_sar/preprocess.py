"""Land/AOI/nodata masking, dB conversion, and quicklook PNGs for HyP3 RTC products.

HyP3 RTC products are UTM GeoTIFFs named like `<product>_HH.tif`, `<product>_HV.tif`,
`<product>_inc_map.tif`, with 0 as nodata, in linear power (gamma0) when scale=power.
Masked outputs stay in linear power, because CFAR (Milestone 2) runs on linear intensity.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.geometry.base import BaseGeometry  # noqa: E402

from iceberg_sar.config import Config, metres_to_px  # noqa: E402
from iceberg_sar.seaice import (  # noqa: E402
    apply_coarse_mask,
    coarse_stats,
    detect_ice,
    ice_polygons,
    write_mask,
)
from iceberg_sar.search import load_aoi  # noqa: E402

POL_PATTERN = re.compile(r"_(HH|HV|VV|VH)\.tif$", re.IGNORECASE)
NODATA = np.float32(np.nan)
DB_FLOOR = 1e-10  # avoids log10(0)


def linear_to_db(x: np.ndarray) -> np.ndarray:
    """Backscatter power (linear) to decibels: 10*log10(x). Non-positive values become NaN."""
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, 10.0 * np.log10(np.maximum(x, DB_FLOOR)), np.nan).astype(np.float32)


RADIOMETRY_PATTERN = re.compile(r"_RTC\d+_[A-Z]_([gs])[a-z]{5}_", re.IGNORECASE)


def product_radiometry(name: str) -> str:
    """'gamma0' or 'sigma0' from a HyP3 RTC product name (`..._RTC10_G_gpuned_...`: g / s)."""
    m = RADIOMETRY_PATTERN.search(name)
    if not m:
        raise ValueError(f"Not a HyP3 RTC product name: {name}")
    return "gamma0" if m.group(1).lower() == "g" else "sigma0"


def find_rtc_bands(product_dir: Path) -> dict[str, Path]:
    """Map polarization (and 'inc' for the incidence-angle map) to file paths."""
    bands: dict[str, Path] = {}
    for tif in sorted(Path(product_dir).glob("*.tif")):
        m = POL_PATTERN.search(tif.name)
        if m:
            bands[m.group(1).upper()] = tif
        elif tif.name.endswith("_inc_map.tif"):
            bands["inc"] = tif
    if not any(k in bands for k in ("HH", "HV", "VV", "VH")):
        raise FileNotFoundError(f"No polarization GeoTIFFs found in {product_dir}")
    return bands


def ocean_geometries(
    land: gpd.GeoDataFrame,
    raster_crs: rasterio.crs.CRS,
    raster_bounds: tuple[float, float, float, float],
    buffer_m: float,
) -> list[BaseGeometry]:
    """Buffered land polygons in the raster CRS, clipped to the raster extent.

    The raster CRS must be metric (HyP3 RTC outputs are UTM), so the buffer is in metres.
    """
    extent = box(*raster_bounds).buffer(buffer_m * 2)
    land_r = land.to_crs(raster_crs)
    land_r = land_r[land_r.intersects(extent)]
    if land_r.empty:
        return []
    clipped = land_r.geometry.intersection(extent)
    buffered = clipped.buffer(buffer_m) if buffer_m > 0 else clipped
    return [g for g in buffered if not g.is_empty]


def build_valid_mask(
    shape: tuple[int, int],
    transform: rasterio.Affine,
    land_geoms: list[BaseGeometry],
    aoi_geoms: list[BaseGeometry] | None,
) -> np.ndarray:
    """True where a pixel is open water we want to keep (not land, inside AOI)."""
    valid = np.ones(shape, dtype=bool)
    if land_geoms:
        valid &= geometry_mask(land_geoms, out_shape=shape, transform=transform)
    if aoi_geoms:
        valid &= ~geometry_mask(aoi_geoms, out_shape=shape, transform=transform)
    return valid


def mask_band(
    src_path: Path,
    dst_path: Path,
    land_geoms: list[BaseGeometry],
    aoi_geoms: list[BaseGeometry] | None,
    block_rows: int = 1024,
) -> dict[str, float]:
    """Write a float32 copy of `src_path` with land, out-of-AOI, and nodata set to NaN.

    Processes row blocks to keep memory bounded on large scenes. Returns pixel stats.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = n_valid = 0
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="float32", nodata=NODATA, compress="deflate", predictor=3,
            tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for row in range(0, src.height, block_rows):
                win = Window(0, row, src.width, min(block_rows, src.height - row))
                data = src.read(1, window=win).astype(np.float32)
                valid = build_valid_mask(
                    (int(win.height), int(win.width)),
                    src.window_transform(win), land_geoms, aoi_geoms,
                )
                valid &= np.isfinite(data) & (data > 0)
                if src.nodata is not None and not np.isnan(src.nodata):
                    valid &= data != src.nodata
                data[~valid] = NODATA
                dst.write(data, 1, window=win)
                n_total += data.size
                n_valid += int(valid.sum())
    return {"pixels": n_total, "valid_pixels": n_valid, "valid_fraction": n_valid / max(n_total, 1)}


def pool_downsample(arr: np.ndarray, factor: int, how: str = "max") -> np.ndarray:
    """Downsample by an integer factor with NaN-aware max or mean pooling."""
    if factor <= 1:
        return arr
    h, w = (arr.shape[0] // factor) * factor, (arr.shape[1] // factor) * factor
    blocks = arr[:h, :w].reshape(h // factor, factor, w // factor, factor)
    # Fully masked blocks trigger "All-NaN slice" warnings; they correctly become NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "max":
            return np.nanmax(blocks, axis=(1, 3))
        if how == "mean":
            return np.nanmean(blocks, axis=(1, 3))
    raise ValueError(f"Unknown pooling '{how}'")


def read_downsampled_db(path: Path, max_px: int, how: str) -> tuple[np.ndarray, int]:
    """Read a linear-power raster block-wise, pool it to <= max_px, return dB array and factor."""
    with rasterio.open(path) as src:
        factor = max(1, int(np.ceil(max(src.height, src.width) / max_px)))
        rows = []
        step = factor * max(1, 1024 // factor)
        for row in range(0, (src.height // factor) * factor, step):
            nrows = min(step, (src.height // factor) * factor - row)
            win = Window(0, row, src.width, nrows)
            rows.append(pool_downsample(src.read(1, window=win).astype(np.float32), factor, how))
    # Pool in linear power, then convert: averaging dB values would bias the result.
    return linear_to_db(np.vstack(rows)), factor


def db_stretch(db: np.ndarray, db_range: list[float] | None) -> tuple[float, float]:
    if db_range:
        return float(db_range[0]), float(db_range[1])
    finite = db[np.isfinite(db)]
    if finite.size == 0:
        return -30.0, 0.0
    lo, hi = np.percentile(finite, [2, 99.9])
    return float(lo), float(hi)


def save_quicklook(db: np.ndarray, png_path: Path, vmin: float, vmax: float) -> Path:
    """Grayscale PNG with masked (NaN) pixels transparent."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.clip((db - vmin) / (vmax - vmin), 0, 1)
    rgba = plt.get_cmap("gray")(np.nan_to_num(scaled, nan=0.0))
    rgba[..., 3] = np.isfinite(db).astype(float)
    plt.imsave(png_path, rgba)
    return png_path


def mask_sea_ice(
    masked: dict[str, Path],
    ice_cfg: dict,
    interim: Path,
    outputs_dir: Path,
    name: str,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Detect pack ice on the cross-pol band and blank it (NaN) in every masked band."""
    band = next((b for b in (ice_cfg.get("band", "HV"), "HV", "VH") if b in masked), None)
    if band is None:
        raise ValueError("Sea-ice mask needs a cross-pol band (HV or VH)")
    with rasterio.open(masked[band]) as src:
        factor = metres_to_px(float(ice_cfg["cell_m"]), abs(src.transform.a))
    st = coarse_stats(masked[band], factor)
    cell_m = abs(st.transform.a)
    ice, water_db = detect_ice(
        st.mean_lin, st.cv, cell_m=cell_m,
        min_excess_db=ice_cfg["min_excess_db"], max_cv=ice_cfg["max_cv"],
        min_area_km2=ice_cfg["min_area_km2"], strip_min_km2=ice_cfg["strip_min_km2"],
        buffer_m=ice_cfg["buffer_m"],
    )
    mask_tif = write_mask(ice, st.transform, st.crs, interim / f"{name}_seaice_mask.tif")
    polys = ice_polygons(ice, st.transform, st.crs)
    geojson = outputs_dir / "seaice" / f"{name}_seaice.geojson"
    geojson.parent.mkdir(parents=True, exist_ok=True)
    polys.to_file(geojson, driver="GeoJSON")

    newly = {pol: apply_coarse_mask(path, ice, factor) for pol, path in masked.items()}
    meta: dict[str, object] = {
        "band": band,
        "cell_m": cell_m,
        "water_ref_db": round(water_db, 2),
        "ice_area_km2": round(float(ice.sum()) * cell_m**2 / 1e6, 1),
        "n_polygons": len(polys),
        "pixels_masked": newly,
    }
    return meta, {"seaice mask": mask_tif, "seaice polys": geojson}


def preprocess_product(product_dir: Path, land: gpd.GeoDataFrame, cfg: Config) -> dict[str, Path]:
    """Mask every polarization in a HyP3 RTC product and write quicklooks + a metadata JSON."""
    p = cfg.section("preprocess")
    product_dir = Path(product_dir)
    bands = find_rtc_bands(product_dir)
    name = product_dir.name
    interim = cfg.path("interim") / name
    ql_dir = cfg.path("outputs") / "quicklooks"

    first = next(v for k, v in bands.items() if k != "inc")
    with rasterio.open(first) as src:
        crs, bounds = src.crs, src.bounds
    if not crs.is_projected:
        raise ValueError(f"Expected a projected (metric) CRS, got {crs}")

    land_geoms = ocean_geometries(land, crs, tuple(bounds), p["land_buffer_m"])
    aoi_geoms = None
    if p.get("mask_outside_aoi", True):
        aoi = gpd.GeoSeries([load_aoi(cfg.path("aoi"))], crs="EPSG:4326").to_crs(crs)
        aoi_geoms = list(aoi)

    outputs: dict[str, Path] = {}
    meta: dict[str, object] = {
        "product": name,
        "radiometry": product_radiometry(name),  # sigma0 dB is ~1-1.5 dB below gamma0 at sea
        "crs": crs.to_string(),
        "bounds_wgs84": list(transform_bounds(crs, "EPSG:4326", *bounds)),
        "land_buffer_m": p["land_buffer_m"],
        "bands": {},
    }
    masked: dict[str, Path] = {}
    for pol, src_path in bands.items():
        if pol == "inc":
            continue
        masked[pol] = interim / f"{name}_{pol}_masked.tif"
        meta["bands"][pol] = mask_band(src_path, masked[pol], land_geoms, aoi_geoms)  # type: ignore[index]
        outputs[f"{pol} masked"] = masked[pol]

    ice_cfg = cfg.raw.get("seaice", {})
    if ice_cfg.get("enabled", False):
        ice_meta, ice_outputs = mask_sea_ice(masked, ice_cfg, interim, cfg.path("outputs"), name)
        meta["seaice"] = ice_meta
        outputs.update(ice_outputs)
        for pol, n in ice_meta["pixels_masked"].items():  # type: ignore[union-attr]
            b = meta["bands"][pol]  # type: ignore[index]
            b["valid_pixels"] -= n
            b["valid_fraction"] = b["valid_pixels"] / max(b["pixels"], 1)

    for pol, path in masked.items():
        db, factor = read_downsampled_db(path, p["quicklook_max_px"], p["quicklook_pool"])
        vmin, vmax = db_stretch(db, p.get("quicklook_db_range"))
        png = ql_dir / f"{name}_{pol}_db.png"
        outputs[f"{pol} quicklook"] = save_quicklook(db, png, vmin, vmax)
        meta["bands"][pol].update(  # type: ignore[index]
            db_stretch=[round(vmin, 2), round(vmax, 2)], downsample_factor=factor,
        )
    if "inc" in bands:
        outputs["inc"] = bands["inc"]

    meta_path = interim / f"{name}_preprocess.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    outputs["metadata"] = meta_path
    return outputs
