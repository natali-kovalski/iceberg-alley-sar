"""Revision phase 3 helpers: random shifts and the bootstrap resampling."""

import numpy as np
import pytest

from iceberg_sar.benchmark import robustness as rb


def test_shift_points_moves_all_by_one_vector() -> None:
    xy = np.array([[0.0, 0.0], [10.0, 5.0]])
    out = rb.shift_points(xy, 5000, np.random.default_rng(1))
    d = out - xy
    assert np.allclose(d[0], d[1]) and np.hypot(*d[0]) == pytest.approx(5000)


def test_boot_index_within_groups_keeps_group_sizes() -> None:
    groups = np.array([0, 0, 0, 1, 1])
    idx = rb.boot_index(groups, 200, np.random.default_rng(0), "label")
    assert idx.shape == (200, 5)
    assert np.isin(idx[:, :3], [0, 1, 2]).all() and np.isin(idx[:, 3:], [3, 4]).all()


def test_boot_index_by_group_takes_whole_groups() -> None:
    groups = np.array([0, 0, 1])
    idx = rb.boot_index(groups, 50, np.random.default_rng(0), "group")
    for row in idx:
        r = row[row >= 0].tolist()
        assert r.count(0) == r.count(1)    # group 0 always comes whole
    hit = np.array([True, True, False])
    rec = rb.boot_recall(hit, idx)
    assert set(np.round(rec, 3)) <= {0.0, 0.5, 0.667, 1.0}


def _labels(xy: list[tuple[float, float]], radius: float) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame({"x": [p[0] for p in xy], "y": [p[1] for p in xy],
                         "radius_m": radius, "status": "verified", "size_class": "small"})


def test_best_threshold_reoptimises_under_new_matching() -> None:
    # Two labels; peak A (score 0.9) on label 1, peak B (score 0.5) 300 m from label 2, and a
    # false alarm C (score 0.6). At 500 m the loosest threshold wins (B hits); at 250 m B can't
    # hit, so the stricter threshold that drops C and B wins.
    cand = np.array([[0.0, 0.0], [10_300.0, 0.0], [50_000.0, 0.0]])
    run = rb.EwRun("thor-pipeline", 0.5, cand, cand_xy=cand,
                   cand_score=np.array([0.9, 0.5, 0.6]), thresholds=(0.5, 0.7))
    lab = [(0.0, 0.0), (10_000.0, 0.0)]
    thr, r = rb.best_threshold(run, _labels(lab, 500), "greedy")
    assert thr == 0.5 and r["TP"] == 2
    thr, r = rb.best_threshold(run, _labels(lab, 250), "greedy")
    assert thr == 0.7 and r["TP"] == 1 and r["FP"] == 0


def test_best_threshold_needs_candidates() -> None:
    with pytest.raises(ValueError):
        rb.best_threshold(rb.EwRun("x", 0.1, np.empty((0, 2))), _labels([], 250), "greedy")
