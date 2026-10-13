import numpy as np
import pytest

from iceberg_sar.thor.encoder import ThorParams, pca_rgb, to_model_input
from iceberg_sar.thor.smoke import pick_crop, token_separation


def params(**kw) -> ThorParams:
    base = dict(model="thor_v1_tiny", bands=("HH", "HV"), patch_px=8, crop_px=64,
                to_sigma0=True, db_clip=(-40.0, 10.0),
                norm={"HH": (-14.0, 7.0), "HV": (-22.0, 7.0)}, layer=5)
    base.update(kw)
    return ThorParams(**base)


def test_to_model_input_sigma0_and_normalization():
    p = params()
    hh = np.full((4, 4), 10 ** (-1.4), np.float32)  # -14 dB gamma0
    hv = np.full((4, 4), 10 ** (-2.2), np.float32)  # -22 dB
    inc = np.full((4, 4), 60.0, np.float32)          # cos 60 = 0.5 -> -3.01 dB
    x = to_model_input({"HH": hh, "HV": hv}, inc, p)
    assert x.shape == (2, 4, 4) and x.dtype == np.float32
    np.testing.assert_allclose(x, -3.0103 / 7.0, atol=1e-4)


def test_to_model_input_nan_becomes_mean_and_floor_clips():
    p = params(bands=("HV",), to_sigma0=False)
    hv = np.array([[np.nan, 0.0, 1e-9, 10 ** (-2.2)]], np.float32)
    x = to_model_input({"HV": hv}, None, p)[0, 0]
    assert x[0] == 0.0                                   # land / nodata -> band mean
    np.testing.assert_allclose(x[1:3], (-40 + 22) / 7.0)  # zero and -90 dB pinned to floor
    np.testing.assert_allclose(x[3], 0.0, atol=1e-5)


def test_to_model_input_needs_incidence_for_sigma0():
    with pytest.raises(ValueError):
        to_model_input({"HH": np.ones((2, 2)), "HV": np.ones((2, 2))}, None, params())


def test_pca_rgb_ignores_invalid_tokens():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(16, 8, 8))
    valid = np.ones((8, 8), bool)
    valid[0] = False
    feats[:, 0] = 1e6  # huge invalid tokens must not drive the PCA
    rgb, ratio = pca_rgb(feats, valid)
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    assert (rgb[0] == 0).all() and rgb[1:].std() > 20
    assert ratio.shape == (3,) and 0 < ratio.sum() < 1


def test_pick_crop_prefers_ice_edge_then_most_detections():
    shape, crop = (1000, 1000), 100
    rc = np.array([[50, 50], [60, 60], [70, 70], [500, 500], [520, 520]], float)
    ice = np.zeros((10, 10), np.uint8)  # 100 px cells
    ice[5, 5] = 1                       # ice only around (500, 500)
    r0, c0, info = pick_crop(rc, ice, shape, crop, (0.1, 0.9))
    assert info["detections"] == 2 and info["ice_fraction"] > 0
    assert r0 <= 500 < r0 + crop
    r0, c0, info = pick_crop(rc, ice, shape, crop, (0.95, 1.0))  # nothing qualifies
    assert info["detections"] == 3 and (r0, c0) == (0, 0)  # ties keep the first candidate


def test_token_separation_flags_odd_tokens():
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(8, 8, 8))
    feats[:, 2, 3] += 10  # berg token
    det_rc = np.array([[2 * 4 + 1, 3 * 4 + 2]], float)
    no = np.zeros((32, 32), bool)
    s, dist = token_separation(feats, det_rc, no, no, patch_px=4)
    assert s["berg_tokens"] == 1 and s["berg_rank_pct"] == 100.0
    assert s["berg_median_dist"] > 3 and s["water_median_dist"] == 1.0
    assert dist.shape == (8, 8)


def test_to_model_input_leaves_sigma0_products_alone():
    p = params(bands=("HH",))
    hh = np.full((2, 2), 10 ** (-1.4), np.float32)
    inc = np.full((2, 2), 60.0, np.float32)
    np.testing.assert_allclose(to_model_input({"HH": hh}, inc, p, "sigma0"), 0.0, atol=1e-5)
    np.testing.assert_allclose(to_model_input({"HH": hh}, None, p, "sigma0"), 0.0, atol=1e-5)
