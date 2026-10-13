"""End-to-end detection on a small synthetic masked scene (no network)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from iceberg_sar.cfar import CfarParams
from iceberg_sar.detections import (
    DetectionParams,
    detect_raster,
    detect_raster_multi,
    scene_timestamp,
    to_geodataframe,
)

CRS = "EPSG:32621"
PIX = 20.0
X0, Y0 = 600_000.0, 5_600_000.0
H = W = 300
NAME = "S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_ABCD"

COMPACT = (100, 100)          # 2x2 bright target
ON_BLOCK_EDGE = (49, 200)     # 2x2 target straddling rows 49/50 (block_rows=50)
NEAR_MASK = (244, 270)        # 6 px from the masked patch below it


def _write(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs=CRS, transform=from_origin(X0, Y0, PIX, PIX), nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    hv = rng.gamma(4.0, 0.001 / 4.0, size=(H, W))
    for r, c in (COMPACT, ON_BLOCK_EDGE, NEAR_MASK):
        hv[r : r + 2, c : c + 2] = 0.1                # +20 dB
    hv[160, 30:110] = 0.03                            # 1.6 km ice strip, +15 dB
    hv[160, [50, 51, 80, 81]] = 0.1                   # bright knots on the strip
    hv[250:, 250:] = np.nan                           # masked (land / pack ice)
    return {
        "HV": _write(tmp_path / f"{NAME}_HV.tif", hv),
        "HH": _write(tmp_path / f"{NAME}_HH.tif", hv * 10),
    }


def _run(scene: dict[str, Path], block_rows: int) -> tuple:
    cfar = CfarParams(guard_px=3, background_px=10, pfa=1e-6, enl=4.0)
    p = DetectionParams(min_area_px=2, max_area_px=500, edge_buffer_px=10, max_extent_px=20,
                        max_structure_px=15, block_rows=block_rows)
    return detect_raster(scene, "HV", cfar, p)


@pytest.mark.parametrize("block_rows", [50, 1024])
def test_detects_compact_targets_only(scene: dict[str, Path], block_rows: int) -> None:
    table, _, _ = _run(scene, block_rows)
    rows, cols = table["row"] - 0.5, table["col"] - 0.5   # centroid of a 2x2 block
    found = {(round(r), round(c)) for r, c in zip(rows, cols, strict=True)}
    assert found == {COMPACT, ON_BLOCK_EDGE}   # strip and near-mask target rejected
    assert (table["area_px"] == 4).all()
    assert table["peak_db_HV"].to_numpy() == pytest.approx(-10.0, abs=0.01)
    assert table["peak_db_HH"].to_numpy() == pytest.approx(0.0, abs=0.01)
    assert (table["contrast_db"] > 15).all()


def test_copol_check_drops_cross_pol_only_target(scene: dict[str, Path]) -> None:
    with rasterio.open(scene["HH"]) as src:
        hh = src.read(1)
    hh[COMPACT[0] : COMPACT[0] + 2, COMPACT[1] : COMPACT[1] + 2] = 0.01  # sea level in HH
    scene = {**scene, "HH": _write(scene["HH"].with_name("hh_flat.tif"), hh)}
    cfar = CfarParams(guard_px=3, background_px=10, pfa=1e-6, enl=4.0)

    def found(min_db: float | None) -> set[tuple[int, int]]:
        p = DetectionParams(min_area_px=2, max_area_px=500, edge_buffer_px=10,
                            max_extent_px=20, copol_min_contrast_db=min_db)
        table, _, _ = detect_raster(scene, "HV", cfar, p)
        return {(round(r - 0.5), round(c - 0.5)) for r, c in zip(table.row, table.col,
                                                                    strict=True)}

    assert found(None) == {COMPACT, ON_BLOCK_EDGE}
    assert found(6.0) == {ON_BLOCK_EDGE}


@pytest.mark.parametrize("merge", [False, True])
def test_split_target_merged_by_structure(scene: dict[str, Path], merge: bool) -> None:
    # One ship-like target: two CFAR cores 5 px apart in azimuth, joined by a dimmer
    # bridge (+6 dB, below the +7.3 dB CFAR threshold but above grow_db after smoothing).
    split = (200, 100)
    with rasterio.open(scene["HV"]) as src:
        hv = src.read(1)
    r, c = split
    hv[r : r + 2, c : c + 2] = 0.1
    hv[r + 5 : r + 7, c : c + 2] = 0.1
    hv[r + 2 : r + 5, c : c + 2] = 0.004
    scene = {"HV": _write(scene["HV"].with_name("hv_split.tif"), hv),
             "HH": _write(scene["HH"].with_name("hh_split.tif"), hv * 10)}
    cfar = CfarParams(guard_px=3, background_px=10, pfa=1e-6, enl=4.0)
    p = DetectionParams(min_area_px=2, max_area_px=500, edge_buffer_px=10, max_extent_px=20,
                        max_structure_px=15, merge_by_structure=merge)
    table, _, _ = detect_raster(scene, "HV", cfar, p)
    at_split = table[(table.row - r).between(-1, 8) & (table.col - c).between(-1, 3)]
    if merge:
        assert len(at_split) == 1
        assert at_split.area_px.item() == 8
        assert at_split.row.item() == pytest.approx(r + 3.0, abs=0.01)  # mean of rows r..r+1, r+5..r+6
    else:
        assert len(at_split) == 2
    assert len(table) == len(at_split) + 2   # COMPACT and ON_BLOCK_EDGE unchanged


def test_geodataframe_attributes(scene: dict[str, Path]) -> None:
    table, transform, crs = _run(scene, 1024)
    gdf = to_geodataframe(table, transform, crs, NAME, inc_path=None)
    assert gdf.crs.to_epsg() == 4326
    assert list(gdf["id"]) == ["20250508T094925_00000", "20250508T094925_00001"]
    assert (gdf["timestamp"] == "2025-05-08T09:49:25Z").all()
    assert (gdf["area_m2"] == 4 * PIX * PIX).all()
    assert gdf["lon"].between(-58, -52).all() and gdf["lat"].between(49, 52).all()
    assert gdf["distance_to_ice_km"].isna().all()  # no ice mask given


def test_distance_to_ice_attribute(scene: dict[str, Path], tmp_path: Path) -> None:
    from iceberg_sar.seaice import write_mask

    ice = np.zeros((H // 10, W // 10), dtype=bool)
    ice[25:, 25:] = True  # 200 m cells over the masked patch at rows/cols 250+
    mask = write_mask(ice, from_origin(X0, Y0, 10 * PIX, 10 * PIX), CRS, tmp_path / "ice.tif")
    table, transform, crs = _run(scene, 1024)
    gdf = to_geodataframe(table, transform, crs, NAME, inc_path=None, ice_mask=mask)
    # ON_BLOCK_EDGE centre (row 50, col 200.5) -> cell (5, 20); COMPACT -> cell (10, 10)
    expected = [0.2 * np.hypot(20, 5), 0.2 * np.hypot(15, 15)]
    assert gdf["distance_to_ice_km"].to_numpy() == pytest.approx(expected, abs=0.01)


def test_scene_timestamp() -> None:
    assert scene_timestamp(NAME) == "2025-05-08T09:49:25Z"
    with pytest.raises(ValueError):
        scene_timestamp("no_time_here")


def test_multi_threshold_equals_separate_runs(scene: dict[str, Path]) -> None:
    """A PFA sweep in one pass gives exactly what separate runs give."""
    p = DetectionParams(min_area_px=2, max_area_px=500, edge_buffer_px=10, max_extent_px=20,
                        max_structure_px=15, copol_min_contrast_db=6, block_rows=50)
    cfars = [CfarParams(guard_px=3, background_px=10, pfa=pfa, enl=4.0)
             for pfa in (1e-3, 1e-6, 1e-9)]
    multi, _, _ = detect_raster_multi(scene, "HV", cfars, p)
    for c, table in zip(cfars, multi, strict=True):
        single, _, _ = detect_raster(scene, "HV", c, p)
        assert len(single) > 0
        pd.testing.assert_frame_equal(table.reset_index(drop=True),
                                      single.reset_index(drop=True))
    assert len(multi[0]) >= len(multi[2])


def test_multi_threshold_needs_same_windows(scene: dict[str, Path]) -> None:
    p = DetectionParams(min_area_px=2, max_area_px=500, edge_buffer_px=10)
    with pytest.raises(ValueError):
        detect_raster_multi(scene, "HV", [CfarParams(3, 10, 1e-6, 4.0),
                                          CfarParams(3, 12, 1e-6, 4.0)], p)
