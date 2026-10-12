from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from iceberg_sar.seaice import (
    apply_coarse_mask,
    coarse_stats,
    detect_ice,
    distance_to_ice_km,
    write_mask,
)

WATER, ICE = 10 ** (-31 / 10), 10 ** (-20 / 10)
KW = dict(cell_m=200, min_excess_db=6, max_cv=0.8, min_area_km2=2, strip_min_km2=0.5, buffer_m=0)


def _scene() -> tuple[np.ndarray, np.ndarray]:
    mean = np.full((100, 100), WATER)
    cv = np.full((100, 100), 1.1)
    return mean, cv


def test_uniform_pack_ice_is_masked() -> None:
    mean, cv = _scene()
    mean[10:30, 10:30], cv[10:30, 10:30] = ICE, 0.5  # 16 km² floe field
    ice, water_db = detect_ice(mean, cv, **KW)
    assert water_db == np.float32(-31) or abs(water_db + 31) < 0.01
    assert ice[10:30, 10:30].all()
    assert ice.sum() == 400


def test_isolated_iceberg_cells_are_kept() -> None:
    mean, cv = _scene()
    mean[50, 50], cv[50, 50] = ICE, 3.0            # one spiky bright cell
    mean[70:72, 70:72], cv[70:72, 70:72] = ICE, 0.6  # 2x2 bright, uniform (big berg)
    ice, _ = detect_ice(mean, cv, **KW)
    assert not ice.any()


def test_thin_spiky_ice_strip_is_masked() -> None:
    mean, cv = _scene()
    mean[60, 20:40], cv[60, 20:40] = ICE, 1.5  # 20 cells = 0.8 km² ribbon
    ice, _ = detect_ice(mean, cv, **KW)
    assert ice[60, 20:40].all()


def test_buffer_grows_mask() -> None:
    mean, cv = _scene()
    mean[40:60, 40:60], cv[40:60, 40:60] = ICE, 0.5
    ice, _ = detect_ice(mean, cv, **{**KW, "buffer_m": 1000})
    assert ice[40:60, 35].all() and not ice[40:60, 30].any()


def _write(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs="EPSG:32621", transform=from_origin(600000, 5500000, 20, 20),
        nodata=float("nan"),
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def test_coarse_stats_and_apply_mask_on_uneven_grid(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    data = rng.exponential(WATER, (35, 47)).astype(np.float32)  # not a multiple of 10
    data[:10, :10] = ICE
    path = _write(tmp_path / "hv.tif", data)

    st = coarse_stats(path, factor=10)
    assert st.mean_lin.shape == (4, 5)
    assert abs(st.mean_lin[0, 0] - ICE) < 1e-6 and st.cv[0, 0] < 1e-3
    assert 0.5 < st.cv[1, 1] < 1.5  # exponential speckle: CV ~ 1
    assert abs(st.transform.a) == 200

    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 0] = mask[3, 4] = True  # includes the partial bottom-right cell
    n = apply_coarse_mask(path, mask, factor=10, block_rows=10)
    with rasterio.open(path) as src:
        out = src.read(1)
    assert np.isnan(out[:10, :10]).all() and np.isnan(out[30:, 40:]).all()
    assert np.isfinite(out[10:30, :]).all()
    assert n == 100 + 5 * 7


def test_distance_to_ice_km(tmp_path: Path) -> None:
    ice = np.zeros((20, 20), dtype=bool)
    ice[:, :5] = True  # ice in the 5 western columns, 200 m cells
    path = write_mask(ice, from_origin(0, 4000, 200, 200), "EPSG:32621", tmp_path / "m.tif")
    xs = np.array([100.0, 1100.0, 3100.0, 99_999.0])  # in ice, 1 cell out, 11 out, off grid
    d = distance_to_ice_km(path, xs, np.full(4, 2000.0))
    np.testing.assert_allclose(d[:3], [0.0, 0.2, 2.2])
    assert np.isnan(d[3])


def test_distance_to_ice_km_without_ice_is_nan(tmp_path: Path) -> None:
    path = write_mask(np.zeros((5, 5), bool), from_origin(0, 1000, 200, 200), "EPSG:32621",
                      tmp_path / "m.tif")
    assert np.isnan(distance_to_ice_km(path, np.array([100.0]), np.array([900.0]))).all()


def test_ice_core_undoes_the_buffer():
    from scipy import ndimage

    from iceberg_sar.seaice import _disk, ice_core

    core = np.zeros((40, 40), bool)
    core[10:20, 10:20] = True                     # pack-ice core
    core[0:6, 30:40] = True                       # core touching the array border
    buffered = ndimage.binary_dilation(core, structure=_disk(5))
    np.testing.assert_array_equal(ice_core(buffered, cell_m=200, buffer_m=1000), core)
    assert not ice_core(np.zeros((5, 5), bool), 200, 1000).any()


def test_min_db_drops_faint_coastal_water_but_keeps_ice() -> None:
    # Calm day: noise-floor-removed water at -40 dB, a band of slightly brighter water at
    # -31 dB (passes +6 dB alone), and real pack ice at -22 dB.
    mean = np.full((100, 100), 10 ** (-40 / 10))
    cv = np.full((100, 100), 1.1)
    mean[:, 0:4], cv[:, 0:4] = 10 ** (-31 / 10), 0.5      # coastal band, 80 km2
    mean[40:60, 40:60], cv[40:60, 40:60] = 10 ** (-22 / 10), 0.5
    loose, _ = detect_ice(mean, cv, **KW)
    strict, _ = detect_ice(mean, cv, **{**KW, "min_db": -30})
    assert loose[:, 0:4].all() and loose[40:60, 40:60].all()
    assert not strict[:, 0:4].any() and strict[40:60, 40:60].all()
