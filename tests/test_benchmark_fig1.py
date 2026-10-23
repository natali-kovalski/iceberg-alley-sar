"""Georeferencing Fig. 1: the affine fit and its inverse."""

from pathlib import Path

import numpy as np

from iceberg_sar.benchmark.fig1 import fit_affine, outlines, pixel_to_geo

ROOT = Path(__file__).resolve().parents[1]


def test_affine_roundtrip() -> None:
    rng = np.random.default_rng(0)
    geo = rng.uniform(0, 500_000, (6, 2))
    true = np.array([[0.002, 0.0003], [0.0004, -0.0019], [10.0, 1200.0]])  # rotation + scale + flip
    px = np.column_stack([geo, np.ones(6)]) @ true
    m, res = fit_affine(geo, px)
    assert np.allclose(m, true) and res.max() < 1e-6
    assert np.allclose(pixel_to_geo(m, px), geo)


def test_committed_points_fit_within_a_few_km() -> None:
    gdf, info = outlines(ROOT / "benchmarks" / "faerch2023" / "fig1_points.csv", "EPSG:32620")
    assert set(gdf.name) == {"pink_s2", "alos2"}
    assert info["rms_km"] < 10
    alos = gdf.set_index("name").geometry["alos2"]
    assert alos.contains(gdf.set_index("name").geometry["pink_s2"].centroid)
