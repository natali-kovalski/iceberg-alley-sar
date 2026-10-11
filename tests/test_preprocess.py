"""Preprocessing tests on a small synthetic 'HyP3-like' UTM product (no network)."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from iceberg_sar.preprocess import (
    find_rtc_bands,
    linear_to_db,
    mask_band,
    ocean_geometries,
    pool_downsample,
)

CRS = "EPSG:32621"
PIX = 20.0
X0, Y0 = 600_000.0, 5_500_000.0  # UTM 21N, off NE Newfoundland
H = W = 100


def _write(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs=CRS, transform=from_origin(X0, Y0, PIX, PIX), nodata=0,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


@pytest.fixture
def product(tmp_path: Path) -> Path:
    d = tmp_path / "S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_ABCD"
    d.mkdir()
    sea = np.full((H, W), 0.01, dtype=np.float32)  # -20 dB sea clutter
    sea[50, 80] = 1.0                              # 0 dB bright target
    sea[:, :5] = 0                                 # nodata strip
    _write(d / f"{d.name}_HH.tif", sea)
    _write(d / f"{d.name}_HV.tif", sea * 0.1)
    _write(d / f"{d.name}_inc_map.tif", np.full((H, W), 0.6))
    return d


def test_linear_to_db() -> None:
    out = linear_to_db(np.array([1.0, 0.1, 0.01, 0.0, -1.0]))
    np.testing.assert_allclose(out[:3], [0.0, -10.0, -20.0], atol=1e-5)
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_find_rtc_bands(product: Path) -> None:
    assert set(find_rtc_bands(product)) == {"HH", "HV", "inc"}


def test_pool_keeps_single_bright_pixel_with_max() -> None:
    a = np.full((8, 8), 0.01, dtype=np.float32)
    a[3, 5] = 1.0
    assert pool_downsample(a, 4, "max").max() == pytest.approx(1.0)
    assert pool_downsample(a, 4, "mean").max() < 0.1


def test_mask_band_removes_land_buffer_and_nodata(product: Path, tmp_path: Path) -> None:
    # Land occupies the left 30 px (600 m) of the scene; 100 m buffer adds 5 px.
    land_ll = gpd.GeoDataFrame(
        geometry=[box(X0 - 5000, Y0 - H * PIX, X0 + 30 * PIX, Y0)], crs=CRS
    ).to_crs("EPSG:4326")
    with rasterio.open(product / f"{product.name}_HH.tif") as src:
        geoms = ocean_geometries(land_ll, src.crs, tuple(src.bounds), buffer_m=100)
    out = tmp_path / "masked.tif"
    stats = mask_band(product / f"{product.name}_HH.tif", out, geoms, None, block_rows=16)

    with rasterio.open(out) as src:
        data = src.read(1)
    assert np.isnan(data[:, :35]).all()          # land + ~100 m buffer
    assert np.isfinite(data[:, 40:]).all()       # open water kept
    assert data[50, 80] == pytest.approx(1.0)    # target untouched, still linear
    assert 0.5 < stats["valid_fraction"] < 0.7


def test_overpass_query_and_point_discs() -> None:
    from iceberg_sar.landmask import overpass_query, point_features, points_to_discs

    q = overpass_query((-56.0, 48.0, -52.0, 52.0), {"natural": ["rock", "islet"]})
    assert 'node(48.0,-56.0,52.0,-52.0)["natural"~"^(rock|islet)$"];' in q
    pts = point_features([{"type": "node", "id": 1, "lat": 49.79, "lon": -54.25,
                           "tags": {"natural": "rock"}}])
    assert pts.iloc[0]["kind"] == "natural=rock"
    discs = points_to_discs(pts, 20, "EPSG:32621")
    area = gpd.GeoSeries(discs, crs="EPSG:4326").to_crs("EPSG:32621").area.iloc[0]
    assert abs(area - np.pi * 20**2) < 0.05 * np.pi * 20**2


def test_product_radiometry_from_hyp3_name():
    from iceberg_sar.preprocess import product_radiometry

    assert product_radiometry("S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1") == "gamma0"
    assert product_radiometry("S1B_IW_20190517T094027_DHP_RTC10_G_spuned_A1B2_HH.tif") == "sigma0"
    with pytest.raises(ValueError):
        product_radiometry("S1A_IW_GRDH_1SDH_20190511T094108")
