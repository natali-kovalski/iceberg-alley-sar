"""Path B label rebuild: S2 N-sigma detection and the automated drift matching."""

import geopandas as gpd
import numpy as np

from iceberg_sar.benchmark.drift import histogram_peak, holdout_error, rebuild
from iceberg_sar.benchmark.labels import nsigma_detect, object_table, ring_stats, size_class


def test_ring_stats_constant_and_hole() -> None:
    x = np.full((40, 40), 2.0)
    valid = np.ones_like(x, bool)
    x[20, 20] = 100.0  # inside the guard window of the centre pixel: not in its ring
    mean, std, frac = ring_stats(x, valid, guard_px=5, outer_px=11, block_rows=7)
    assert np.isclose(mean[20, 20], 2.0) and np.isclose(std[20, 20], 0.0)
    assert np.isclose(frac[20, 20], 1.0)
    assert np.isclose(frac[0, 0], (6 * 6 - 3 * 3) / (11**2 - 5**2))  # corner: ring cut off


def test_ring_stats_ignores_invalid() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 0.1, (60, 60))
    valid = np.ones_like(x, bool)
    x[:, :25] = 50.0  # bright, but masked
    valid[:, :25] = False
    mean, _, _ = ring_stats(x, valid, 5, 11)
    assert abs(mean[30, 30] - 1.0) < 0.05


def test_nsigma_detect_finds_bright_object() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 0.01, (200, 200)).astype(np.float32)
    x[100:103, 100:104] = 0.5
    det, _ = nsigma_detect(x, np.ones_like(x, bool),
                        {"guard_m": 90, "outer_m": 130, "n_sigma": 4, "min_ring_fraction": 0.5},
                        pixel_m=10.0)
    t = object_table(det, x, np.zeros_like(x), 10.0)
    big = t[t.area_px >= 2]
    assert len(big) == 1 and big.iloc[0].area_px == 12
    assert np.isclose(big.iloc[0].row, 101) and np.isclose(big.iloc[0].col, 101.5)


def test_size_class_edges() -> None:
    assert list(size_class(np.array([10, 59.9, 60, 120, 121]))) == [
        "small", "small", "medium", "large", "large"]


def test_histogram_peak() -> None:
    rng = np.random.default_rng(2)
    true = np.array([3000.0, -1500.0])
    disp = np.vstack([true + rng.normal(0, 60, (80, 2)), rng.uniform(-8000, 8000, (2000, 2))])
    peak, support = histogram_peak(disp, 8000, 100, 250)
    assert np.hypot(*(peak - true)) < 150 and support >= 70


def test_histogram_peak_prefers_small_drift_on_ties() -> None:
    disp = np.array([[0.0, 0.0], [0.0, 0.0], [5000.0, 0.0], [5000.0, 0.0]])
    peak, support = histogram_peak(disp, 8000, 100, 250)
    assert support == 2 and np.hypot(*peak) < 100


def _synthetic(n: int = 200, seed: int = 3) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                                                     np.ndarray]:
    """Bergs drifting by a smooth field; SAR sees 70 % of them plus clutter."""
    rng = np.random.default_rng(seed)
    s1 = rng.uniform(0, 60000, (n, 2))
    drift = np.column_stack([3000 + 0.02 * s1[:, 1], -2000 + 0.01 * s1[:, 0]])
    s2 = s1 + drift
    seen = rng.random(n) < 0.7
    sar = np.vstack([s1[seen] + rng.normal(0, 30, (seen.sum(), 2)),
                     rng.uniform(0, 60000, (100, 2))])
    lab = gpd.GeoDataFrame({"size_class": "small"}, index=range(n),
                           geometry=gpd.points_from_xy(s2[:, 0], s2[:, 1]), crs="EPSG:32620")
    sar_g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sar[:, 0], sar[:, 1]), crs="EPSG:32620")
    return lab, sar_g, s1


P = {"search_m": 6000, "bin_m": 100, "node_spacing_m": 5000,
     "neighbourhood_m": 15000, "min_node_labels": 5, "min_support": 4,
     "verify_radius_m": 300, "radius_verified_m": 250, "radius_interpolated_m": 1789,
     "holdout_fraction": 0.1}


def test_rebuild_recovers_drift() -> None:
    lab, sar, s1 = _synthetic()
    out, info = rebuild(lab, sar, P)
    pos = np.column_stack([out.geometry.x, out.geometry.y])
    err = np.hypot(*(pos - s1).T)
    ver = (out.status == "verified").to_numpy()
    assert ver.mean() > 0.5
    assert np.median(err[ver]) < 100          # verified = the matched SAR object
    assert np.median(err[~ver]) < 500          # interpolated from neighbours
    assert info["checks"]["holdout_mean_error_m"] < 500
    assert set(out.radius_m) == {250, 1789}


def test_holdout_error_too_few() -> None:
    assert holdout_error(np.zeros((3, 2)), np.zeros((3, 2)), 0.1, 0)["holdout_n"] == 0


def test_normalized_intensity_flattens_incidence_trend() -> None:
    from iceberg_sar.benchmark.drift import normalized_intensity

    rng = np.random.default_rng(4)
    inc = np.tile(np.linspace(20, 45, 200, dtype=np.float32), (200, 1))
    trend = 10 ** (-(inc - 20) / 25)  # 10 dB darker at far range
    stack = np.stack([trend * rng.gamma(9, 1 / 9, inc.shape), trend * rng.gamma(9, 1 / 9, inc.shape)])
    stack[:, 100, 150] = 50 * trend[100, 150]  # one bright target in both bands, far range
    valid = np.ones(inc.shape, bool)
    valid[:5] = False
    n = normalized_intensity(stack.astype(np.float32), inc, valid, 0.5)
    assert n[100, 150] > 20 and np.median(n[valid]) < 1.1 and (n[~valid] == 0).all()


def _faerch_shapefile(tmp_path):  # type: ignore[no-untyped-def]
    """Two drift lines in EPSG:5938 near 58.5 N, like Færch's file (sizes in S2 pixels)."""
    from shapely.geometry import LineString

    pts = gpd.GeoSeries(gpd.points_from_xy([-60.5, -60.4], [58.5, 58.6]), crs=4326).to_crs(5938)
    lines = [LineString([(p.x, p.y), (p.x + 1000, p.y - 2000)]) for p in pts]
    gdf = gpd.GeoDataFrame({"method": ["manual", "interpolate"], "id": ["S2_7", "S2_12"],
                            "area": [40.0, 0.0], "width": [5.0, np.nan], "length": [10.0, 5.5],
                            "shape": [0.5, np.nan]}, geometry=lines, crs=5938)
    path = tmp_path / "drift.shp"
    gdf.to_file(path)
    return path, pts


def test_faerch_labels_s1_end_status_and_paper_metres(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from iceberg_sar.benchmark.patha import faerch_labels

    path, pts = _faerch_shapefile(tmp_path)
    lab = faerch_labels(path, "EPSG:32620", 10.0, ["manual"], 250, 1789)
    assert lab.label_id.tolist() == [7, 12]
    assert lab.status.tolist() == ["verified", "interpolated"]
    assert lab.radius_m.tolist() == [250, 1789]
    # label = S1 end of the line, S2 position = start
    s2 = pts.to_crs(32620)
    assert np.allclose(lab.s2_x, s2.x, atol=0.01) and np.allclose(lab.s2_y, s2.y, atol=0.01)
    s1 = gpd.GeoSeries(gpd.points_from_xy(pts.x + 1000, pts.y - 2000), crs=5938).to_crs(32620)
    assert np.allclose(lab.geometry.x, s1.x, atol=0.01)
    # EPSG:5938 metres are ~7 % longer than true ones at 58.5 N
    k = lab.length_m_paper / lab.length_m
    assert np.allclose(lab.length_m, [100.0, 55.0]) and ((k > 1.06) & (k < 1.08)).all()
    assert np.allclose(lab.drift_m_paper, np.hypot(1000, 2000))
    assert (lab.drift_m < lab.drift_m_paper).all()
    # classes use paper metres: 100 m -> ~107 m (medium), 55 m -> ~59 m (still small)
    assert lab.size_class.tolist() == ["medium", "small"]
    assert np.isclose(lab.area_m2.iloc[0], 4000.0) and np.isnan(lab.width_m.iloc[1])


def test_faerch_unknown_drops_empty_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from iceberg_sar.benchmark.patha import faerch_unknown

    pts = gpd.GeoSeries(gpd.points_from_xy([-60.5], [58.5]), crs=4326).to_crs(5938)
    gdf = gpd.GeoDataFrame({"id": [np.nan, np.nan]}, geometry=[pts.iloc[0], None], crs=5938)
    gdf.to_file(tmp_path / "u.shp")
    unk, n_empty = faerch_unknown(tmp_path / "u.shp", "EPSG:32620")
    assert len(unk) == 1 and n_empty == 1 and unk.crs.to_epsg() == 32620
