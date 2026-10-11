import numpy as np
import pandas as pd

from iceberg_sar.iip import cell_centres
from iceberg_sar.thor.chips import (
    ChipParams,
    labelled_area_grid,
    negative_kind,
    plan_positive_crops,
    scene_split,
    to_local,
    unlabelled_detections,
)
from iceberg_sar.thor.targets import TargetParams, render_heatmap, snap_points


def chip_params(**kw) -> ChipParams:
    ts = pd.Timestamp
    base = dict(margin_px=10, min_sea_fraction=0.5, negatives_per_positive=1.0,
                negative_mix={"water": 0.5, "ice_edge": 0.25, "edge": 0.25},
                ice_edge_range=(0.05, 0.6), edge_range=(0.05, 0.5), labelled_area_buffer_m=100,
                unlabelled_radius_m=300, coarse_factor=4, seed=0,
                val=((ts("2019-02-01"), ts("2019-03-25")),),
                holdout=((ts("2019-03-25"), ts("2019-05-07")),))
    base.update(kw)
    return ChipParams(**base)


def test_cell_centres_undo_truncation_both_hemispheres():
    step = 0.1 / 60
    lat, lon = cell_centres(np.array([49.5]), np.array([-53.2]))
    np.testing.assert_allclose(lat, 49.5 + step / 2)
    np.testing.assert_allclose(lon, -53.2 - step / 2)  # west: further from Greenwich


def test_scene_split():
    p = chip_params()
    assert scene_split(pd.Timestamp("2019-03-06 09:40"), p) == "val"
    assert scene_split(pd.Timestamp("2019-04-29 09:41"), p) == "test"
    assert scene_split(pd.Timestamp("2019-05-06 23:00"), p) == "test"
    assert scene_split(pd.Timestamp("2020-05-05 09:41"), p) == "train"


def test_positive_crops_cover_every_point_with_margin():
    rng = np.random.default_rng(0)
    rc = rng.uniform(0, 1000, size=(40, 2))
    crops = plan_positive_crops(rc, (1000, 1000), 128, 10, rng)
    for r, c in rc:
        assert any(10 <= r - r0 < 118 and 10 <= c - c0 < 118 for r0, c0 in crops) or \
            r < 10 or c < 10 or r >= 990 or c >= 990  # scene border can't give a full margin
    assert all(0 <= r0 <= 872 and 0 <= c0 <= 872 for r0, c0 in crops)
    assert len(crops) < len(rc)  # nearby points share crops


def test_negative_kind():
    p = chip_params()
    assert negative_kind(1.0, 0.0, p) == "water"
    assert negative_kind(0.9, 0.2, p) == "ice_edge"
    assert negative_kind(0.8, 0.0, p) == "edge"
    assert negative_kind(0.97, 0.0, p) is None   # a sliver of land: neither water nor edge
    assert negative_kind(1.0, 0.9, p) is None    # deep in the pack


def test_labelled_area_and_unlabelled():
    rc = np.array([[40.0, 40.0], [40.0, 80.0], [80.0, 60.0]])
    area = labelled_area_grid(rc, (40, 40), buffer_px=8, factor=4)
    assert area[15, 15] and not area[0, 0] and not area[39, 39]
    det = np.array([[41.0, 41.0], [200.0, 200.0]])
    np.testing.assert_array_equal(unlabelled_detections(det, rc, 30), [[200.0, 200.0]])


def test_to_local_keeps_points_inside():
    rc = np.array([[100.0, 100.0], [50.0, 300.0], [700.0, 100.0]])
    np.testing.assert_array_equal(to_local(rc, 0, 0, 512), [[100, 100], [50, 300]])
    assert to_local(np.empty((0, 2)), 0, 0, 512).shape == (0, 2)


def test_snap_points_moves_to_bright_peak_only_if_it_stands_out():
    p = TargetParams(snap_m=100, snap_background_m=200, snap_min_db=6, sigma_m=60)
    rng = np.random.default_rng(0)
    db = 10 * np.log10(rng.exponential(1e-3, size=(101, 101)))
    db[45:48, 57:60] = 0.0                        # bright berg 10 px east of the label
    pts, moved, con = snap_points(db, np.array([[50.0, 50.0]]), p, pixel_m=10)
    assert moved[0] and con[0] > 20
    assert abs(pts[0, 0] - 46) <= 1 and abs(pts[0, 1] - 58) <= 1
    flat = 10 * np.log10(rng.exponential(1e-3, size=(101, 101)))
    _, moved, _ = snap_points(flat, np.array([[50.0, 50.0]]), p, pixel_m=10)
    assert not moved[0]


def test_render_heatmap_peaks_at_one_in_the_points_cell():
    heat = render_heatmap(np.array([[20.0, 44.0]]), (8, 8), stride_px=8, sigma_px=6)
    assert heat[2, 5] == 1.0 and heat.max() == 1.0
    assert 0 < heat[2, 4] < 1 and heat[7, 0] < 1e-3
    assert render_heatmap(np.empty((0, 2)), (4, 4), 8, 6).sum() == 0
