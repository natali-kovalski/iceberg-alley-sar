"""CZML export: per-scene time windows, near-ice flag, footprint and pack-ice packets."""

import json
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from iceberg_sar.config import Config
from iceberg_sar.export_czml import (
    CFAR_RGBA,
    ViewerParams,
    build_czml,
    export_czml,
    find_scenes,
    near_ice,
    pass_coverage,
    point_px,
    scene_intervals,
)

SCENES = {
    "S1C_IW_20250502T094802_DHP_RTC20_G_gpuned_570A": "2025-05-02T09:48:02Z",
    "S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_FDE3": "2025-05-08T09:49:25Z",
}


def _detections(name: str, ts: str, distances: list[float]) -> gpd.GeoDataFrame:
    n = len(distances)
    return gpd.GeoDataFrame(
        {
            "id": [f"{ts[:10]}_{i:05d}" for i in range(n)],
            "scene_id": name,
            "timestamp": ts,
            "lon": -53.0 - 0.1 * np.arange(n),
            "lat": 50.5,
            "area_px": 3,
            "structure_px": 6,
            "background_db": -32.0,
            "peak_db_HH": -10.0,
            "mean_db_HH": -12.0,
            "peak_db_HV": -15.0,
            "mean_db_HV": np.nan,
            "contrast_db": 18.0,
            "area_m2": 1200.0,
            "structure_m": 120.0,
            "incidence_deg": 38.0,
            "distance_to_ice_km": distances,
        },
        geometry=[Point(-53.0 - 0.1 * i, 50.5) for i in range(n)],
        crs="EPSG:4326",
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    det_dir = tmp_path / "data" / "outputs" / "detections"
    det_dir.mkdir(parents=True)
    distances = {0: [0.5, 12.0, np.nan], 1: [30.0]}
    for k, (name, ts) in enumerate(SCENES.items()):
        _detections(name, ts, distances[k]).to_file(
            det_dir / f"{name}_HV_detections.geojson", driver="GeoJSON")
        (det_dir / f"{name}_HV_detections.json").write_text(
            json.dumps({"product": name, "timestamp": ts}), encoding="utf-8")
    first = next(iter(SCENES))
    ice = tmp_path / "data" / "outputs" / "seaice"
    ice.mkdir(parents=True)
    gpd.GeoDataFrame({"area_km2": [40.0]}, geometry=[box(-54, 50, -53.8, 50.2)],
                     crs="EPSG:4326").to_file(ice / f"{first}_seaice.geojson", driver="GeoJSON")
    raw = tmp_path / "data" / "raw" / first
    raw.mkdir(parents=True)
    gpd.GeoDataFrame({"value": [1, 1]},
                     geometry=[box(600_000, 5_590_000, 600_040, 5_590_040),
                               box(500_000, 5_500_000, 700_000, 5_700_000)],
                     crs="EPSG:32621").to_file(raw / f"{first}_shape.shp")
    return Config(root=tmp_path, raw={"paths": {
        "raw": "data/raw", "outputs": "data/outputs", "czml": "viewer/public/data/out.czml",
    }, "hyp3": {"resolution": 20}})


def test_scene_intervals_chain_and_last_window() -> None:
    a, b = datetime(2025, 5, 2, 9, 48), datetime(2025, 5, 8, 9, 49)
    assert scene_intervals([a, b], 7) == [(a, b), (b, datetime(2025, 5, 15, 9, 49))]
    assert scene_intervals([], 7) == []


def test_scene_intervals_same_pass_and_season_gap() -> None:
    a1, a2 = datetime(2025, 5, 14, 9, 48, 2), datetime(2025, 5, 14, 9, 48, 27)   # adjacent frames
    c = datetime(2026, 10, 1, 9, 48)
    week = datetime(2025, 5, 21, 9, 48, 2)
    assert scene_intervals([a1, a2, c], 7) == [
        (a1, week), (a1, week), (c, datetime(2026, 10, 8, 9, 48))]   # a2 shows from a1


def test_near_ice_treats_nan_as_open_water() -> None:
    np.testing.assert_array_equal(near_ice(np.array([0.5, 5.0, 12.0, np.nan]), 5.0),
                                  [True, False, False, False])


def test_find_scenes_only_configured_resolution(cfg: Config) -> None:
    det_dir = cfg.path("outputs") / "detections"
    name = "S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_1234"
    _detections(name, "2025-05-08T09:49:25Z", [1.0]).to_file(
        det_dir / f"{name}_HV_detections.geojson", driver="GeoJSON")
    (det_dir / f"{name}_HV_detections.json").write_text(
        json.dumps({"product": name, "timestamp": "2025-05-08T09:49:25Z"}), encoding="utf-8")
    assert [s.name for s in find_scenes(cfg, "HV")] == list(SCENES)
    cfg.raw["hyp3"]["resolution"] = 10
    assert [s.name for s in find_scenes(cfg, "HV")] == [name]


def test_find_scenes_date_range_is_inclusive(cfg: Config) -> None:
    assert [s.start.day for s in find_scenes(cfg, "HV", start=date(2025, 5, 8))] == [8]
    assert [s.start.day for s in find_scenes(cfg, "HV", end=date(2025, 5, 2))] == [2]
    assert find_scenes(cfg, "HV", date(2025, 5, 3), date(2025, 5, 7)) == []


def test_point_size_grows_with_structure() -> None:
    p = ViewerParams()
    small, medium, large = p.size_px
    assert point_px(0, p) == point_px(179, p) == point_px(float("nan"), p) == small
    assert point_px(180, p) == point_px(269, p) == medium
    assert point_px(270, p) == point_px(1e6, p) == large


def test_find_scenes_sorted_with_optional_layers(cfg: Config) -> None:
    scenes = find_scenes(cfg, "HV")
    assert [s.start.day for s in scenes] == [2, 8]
    assert scenes[0].footprint is not None and scenes[0].seaice is not None
    assert scenes[1].footprint is None and scenes[1].seaice is None


def test_build_czml_packets(cfg: Config) -> None:
    packets = build_czml(find_scenes(cfg, "HV"), ViewerParams(near_ice_km=5))
    by_id = {p["id"]: p for p in packets}
    assert packets[0]["id"] == "document"
    assert by_id["document"]["clock"]["interval"] == "2025-05-02T09:48:02Z/2025-05-15T09:49:25Z"
    assert by_id["legend"]["properties"]["near_ice_km"] == 5

    s1 = by_id["scene/S1C_IW_20250502T094802_DHP_RTC20_G_gpuned_570A"]["properties"]
    assert (s1["n_detections"], s1["n_near_ice"], s1["n_open_water"]) == (3, 1, 2)
    assert s1["ice_area_km2"] == 40.0

    dets = [p for p in packets if p["id"].startswith("det/")]
    assert len(dets) == 4
    first = dets[0]
    assert first["availability"] == "2025-05-02T09:48:02Z/2025-05-08T09:49:24Z"   # no overlap
    assert dets[-1]["availability"] == "2025-05-08T09:49:25Z/2025-05-15T09:49:25Z"
    assert first["properties"]["near_ice"] is True
    assert first["point"]["color"]["rgba"][:3] == [150, 150, 150]
    assert "mean_db_HV" not in first["properties"]                # NaN -> left out
    assert all(v is not None for d in dets for v in d["properties"].values())  # Cesium rejects null
    assert dets[2]["properties"]["near_ice"] is False              # NaN distance
    assert "no pack ice mapped" in dets[2]["description"]

    footprints = [p for p in packets if p["id"].startswith("footprint/")]
    assert len(footprints) == 1                                    # largest polygon only
    lons = footprints[0]["polygon"]["positions"]["cartographicDegrees"][::3]
    assert -60 < min(lons) < max(lons) < -50                       # reprojected to 4326
    assert len([p for p in packets if p["id"].startswith("seaice/")]) == 1


def test_build_czml_ice_free_scene(cfg: Config) -> None:
    """Summer scene: empty pack-ice layer, all distances NaN."""
    first = next(iter(SCENES))
    gpd.GeoDataFrame({"area_km2": []}, geometry=[], crs="EPSG:4326").to_file(
        cfg.path("outputs") / "seaice" / f"{first}_seaice.geojson", driver="GeoJSON")
    packets = build_czml(find_scenes(cfg, "HV"), ViewerParams())
    assert not [p for p in packets if p["id"].startswith("seaice/")]
    s1 = next(p for p in packets if p["id"] == f"scene/{first}")["properties"]
    assert s1["ice_area_km2"] == 0.0


def test_export_writes_valid_json(cfg: Config) -> None:
    out, packets = export_czml(cfg)
    assert out == cfg.path("czml")
    assert json.loads(out.read_text(encoding="utf-8")) == packets


def test_thor_layer_threshold_and_counts(cfg: Config) -> None:
    """THOR detections: own `thor/` entities above the score threshold, counted per scene."""
    name, ts = next(iter(SCENES.items()))
    thor = _detections(name, ts, [0.5, 12.0, np.nan]).assign(
        id=lambda d: "t" + d["id"], score=[0.9, 0.5, 0.2], detector="thor")
    thor.to_file(cfg.path("outputs") / "detections" / f"{name}_thor-linear_detections.geojson",
                 driver="GeoJSON")
    scenes = find_scenes(cfg, "HV")
    assert scenes[0].thor is not None and scenes[1].thor is None
    packets = build_czml(scenes, ViewerParams(near_ice_km=5, thor_score=0.35))
    rings = [p for p in packets if p["id"].startswith("thor/")]
    assert len(rings) == 2                                         # score 0.2 dropped
    assert rings[0]["point"]["outlineColor"]["rgba"][:3] == [150, 150, 150]   # near ice
    assert rings[1]["properties"]["score"] == 0.5
    by_id = {p["id"]: p for p in packets}
    assert by_id[f"scene/{name}"]["properties"]["n_thor"] == 1     # open water only
    # Same size scale for both detectors: a THOR ring just encloses the CFAR dot of equal extent.
    dot = next(p for p in packets if p["id"] == f"det/{thor['id'].iloc[1][1:]}")
    assert dot["point"]["color"]["rgba"] == CFAR_RGBA
    assert rings[1]["point"]["pixelSize"] == dot["point"]["pixelSize"] + 4
    assert "n_thor" not in by_id[f"scene/{next(reversed(SCENES))}"]["properties"]


def test_pass_coverage_fraction_and_gap() -> None:
    aoi = box(0, 0, 100_000, 100_000)
    frames = [box(0, 0, 50_000, 100_000), box(40_000, 0, 60_000, 50_000)]
    gap, km2, frac = pass_coverage(aoi, frames)
    assert km2 == pytest.approx(5_000 + 500)          # overlap between frames counted once
    assert frac == pytest.approx(0.55)
    assert gap.area == pytest.approx(aoi.area * 0.45)
    gap, km2, frac = pass_coverage(aoi, [])
    assert (km2, frac) == (0.0, 0.0) and gap.equals(aoi)


def test_build_czml_aoi_and_pass_coverage(cfg: Config) -> None:
    """AOI outline always on; per pass: coverage stats and the unobserved AOI part."""
    first, second = SCENES
    aoi = gpd.GeoDataFrame(geometry=[box(-58, 49.5, -52, 52)], crs="EPSG:4326")
    packets = build_czml(find_scenes(cfg, "HV"), ViewerParams(), aoi=aoi)
    by_id = {p["id"]: p for p in packets}
    assert "availability" not in by_id["aoi/0"]
    assert by_id["aoi/0"]["polygon"]["fill"] is False

    p1 = by_id["pass/2025-05-02"]
    assert p1["properties"]["frames"] == first
    assert p1["availability"] == "2025-05-02T09:48:02Z/2025-05-08T09:49:24Z"
    assert 0 < p1["properties"]["aoi_coverage_pct"] < 100
    assert 0 < p1["properties"]["observed_km2"] <= 200 * 200
    gap = [p for p in packets if p["id"].startswith("unobserved/2025-05-02/")]
    assert gap and gap[0]["availability"] == p1["availability"]

    # Second scene has no footprint: coverage unknown, so no stats and no dimming.
    assert "aoi_coverage_pct" not in by_id["pass/2025-05-08"]["properties"]
    assert not [p for p in packets if p["id"].startswith("unobserved/2025-05-08")]


def test_build_czml_frames_of_one_pass_share_coverage(cfg: Config) -> None:
    first = next(iter(SCENES))
    name = first.replace("094802", "094827").replace("570A", "570B")
    ts = "2025-05-02T09:48:27Z"
    det_dir = cfg.path("outputs") / "detections"
    _detections(name, ts, [20.0]).to_file(det_dir / f"{name}_HV_detections.geojson",
                                          driver="GeoJSON")
    (det_dir / f"{name}_HV_detections.json").write_text(
        json.dumps({"product": name, "timestamp": ts}), encoding="utf-8")
    raw = cfg.path("raw") / name
    raw.mkdir(parents=True)
    gpd.GeoDataFrame({"value": [1]}, geometry=[box(500_000, 5_700_000, 700_000, 5_900_000)],
                     crs="EPSG:32621").to_file(raw / f"{name}_shape.shp")
    aoi = gpd.GeoDataFrame(geometry=[box(-58, 49.5, -52, 52)], crs="EPSG:4326")
    scenes = find_scenes(cfg, "HV", end=date(2025, 5, 2))

    def props(frames: list) -> dict:
        packets = build_czml(frames, ViewerParams(), aoi=aoi)
        return next(p for p in packets if p["id"] == "pass/2025-05-02")["properties"]

    p1, p2 = props(scenes[:1]), props(scenes)
    assert p2["frames"].split() == [first, name]
    assert p2["aoi_coverage_pct"] > p1["aoi_coverage_pct"]


def test_pass_coverage_closes_seam_between_frames() -> None:
    """A 20 m staircase gap between adjacent frames must not leave slivers in the gap polygon."""
    aoi = box(0, 0, 100_000, 100_000)
    left = box(0, 0, 50_000, 100_000)
    right = box(50_020, 0, 100_000, 50_000)
    gap, _, frac = pass_coverage(aoi, [left, right])
    assert gap.geom_type == "Polygon"
    assert frac == pytest.approx(0.75, abs=1e-3)
