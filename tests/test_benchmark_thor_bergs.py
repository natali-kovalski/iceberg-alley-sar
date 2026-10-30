import numpy as np

from iceberg_sar.benchmark.thor_bergs import max_within, multilook_db, spread_pick, window_around


def test_spread_pick_quantiles_distinct():
    v = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0])   # value = rank
    p = spread_pick(v, 4)
    assert len(set(p.tolist())) == 4
    assert sorted(v[p].tolist()) == [1.0, 3.0, 6.0, 8.0]              # ~12/38/62/88 %
    assert spread_pick(v, 0).size == 0
    assert sorted(spread_pick(v[:2], 3).tolist()) == [0, 1]            # fewer items than asked


def test_max_within_radius_and_nan():
    a = np.zeros((10, 10), np.float32)
    a[5, 8] = 0.9          # 3 cells right of the centre cell (5, 5)
    a[5, 6] = np.nan
    assert max_within(a, (5.5, 5.5), 2.0) == 0.0
    assert max_within(a, (5.5, 5.5), 3.0) == np.float32(0.9)


def test_window_around_pads_with_nan():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    w = window_around(a, (0, 0), 2)
    assert w.shape == (4, 4)
    assert np.isnan(w[:2]).all() and np.isnan(w[:, :2]).all()
    np.testing.assert_array_equal(w[2:, 2:], a[:2, :2])


def test_multilook_db_means_linear_and_skips_invalid():
    mean, std = -20.0, 5.0
    db = np.array([[-10.0, -20.0], [-20.0, -20.0]])                   # 0.1, 0.01 x 3
    z = (db - mean) / std
    out, frac = multilook_db(z, np.ones((2, 2), bool), mean, std, 2)
    np.testing.assert_allclose(out[0, 0], 10 * np.log10((0.1 + 0.03) / 4), atol=1e-5)
    valid = np.array([[True, False], [False, False]])
    out, frac = multilook_db(z, valid, mean, std, 2)
    np.testing.assert_allclose(out[0, 0], -10.0, atol=1e-5)
    assert frac[0, 0] == 0.25
