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
