import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from iceberg_sar.benchmark.thor_faerch import (
    RunSpec,
    ThorRunParams,
    Variant,
    aoi_window,
    crop_geometry,
    label_scores,
    pick_crops,
    score_thresholds,
    water_scores,
)
from iceberg_sar.thor.encoder import model_band_names


def test_model_band_names_suffix_only_off_default_gsd():
    assert model_band_names(("HH", "HV")) == ["IW_HH", "IW_HV"]
    assert model_band_names(("HH", "HV"), 10) == ["IW_HH", "IW_HV"]
    assert model_band_names(("HH", "HV"), 40) == ["IW_HH_40", "IW_HV_40"]


def test_crop_geometry_both_variants_give_80m_tokens():
    assert crop_geometry(5120, 40, Variant("upsample", 10, 8)) == (128, 2)
    assert crop_geometry(5120, 40, Variant("native", 40, 2)) == (128, 2)
    assert crop_geometry(5120, 10, Variant("iw", 10, 8)) == (512, 8)   # the training setup


def test_crop_geometry_rejects_uneven_grid():
    with pytest.raises(ValueError):
        crop_geometry(5000, 40, Variant("bad", 40, 2))
    with pytest.raises(ValueError):
        crop_geometry(5120, 40, Variant("bad", 40, 3))


def test_pick_crops_most_labels_first_no_overlap_and_sea():
    valid = np.ones((100, 100), bool)
    valid[:, 80:] = False                                   # land on the right
    lab = np.array([[10, 10], [12, 14], [15, 12],            # cluster of 3
                    [60, 60], [62, 58],                      # cluster of 2
                    [50, 90]], float)                        # on land
    crops = pick_crops(lab, valid, 20, 5, 0.9)
    assert [n for _, _, n in crops][:2] == [3, 2]
    for i, (r0, c0, _) in enumerate(crops):
        assert valid[r0:r0 + 20, c0:c0 + 20].mean() >= 0.9
        for r1, c1, _ in crops[i + 1:]:
            assert abs(r0 - r1) >= 20 or abs(c0 - c1) >= 20
    assert pick_crops(lab, valid, 20, 5, 0.9, ok=lambda r0, c0: False) == []


def test_aoi_window_snaps_to_token_grid_and_clips():
    t = rasterio.Affine(40, 0, 0, 0, -40, 4000)              # 100 x 100 px, origin (0, 4000)
    r0, c0, h, w = aoi_window(t, (100, 100), box(410, 1000, 2000, 3950), 2)
    assert (r0, c0) == (0, 10) and r0 % 2 == 0 and c0 % 2 == 0
    assert (h, w) == (75, 40)
    assert aoi_window(t, (100, 100), box(-500, -500, 9000, 9000), 2) == (0, 0, 100, 100)


def test_score_thresholds_filters_area_unknown_and_score():
    import geopandas as gpd
    import pandas as pd

    peaks = gpd.GeoDataFrame({"score": [0.9, 0.5, 0.1, 0.9]},
                             geometry=gpd.points_from_xy([0, 1000, 2000, 50_000], [0, 0, 0, 0]))
    labels = pd.DataFrame({"x": [0.0, 2000.0], "y": [0.0, 0.0], "radius_m": [250.0, 250.0],
                           "status": ["verified", "verified"], "size_class": ["small", "large"]})
    unknown = np.array([[1000.0, 10.0]])                     # the 0.5 peak is ignored
    t = score_thresholds(peaks, labels, unknown, box(-100, -100, 3000, 100), 250, (0.05, 0.3))
    assert t.n_det.tolist() == [2, 1]                        # 50 km peak is outside the area
    assert t.TP.tolist() == [2, 1] and t.FP.tolist() == [0, 0]
    assert t.ignored_unknown.tolist() == [1, 1]


def test_run_specs_keep_old_tags_and_add_extras():
    assert RunSpec("upsample", "TNR").tag == "thor-upsample_TNR"   # runs before norms existed
    ew = RunSpec("upsample", "noTNR", "ew")
    assert ew.tag == "thor-upsample-ewnorm_noTNR" and ew.series == "upsample, ew norm"


def test_run_params_from_config(tmp_path):
    import copy

    from iceberg_sar.config import Config, load_config

    cfg = load_config()
    raw = copy.deepcopy(cfg.raw)
    t = raw["benchmark"]["faerch2023"]["thor"]
    t["variants"] = {"upsample": {"band_gsd_m": 10, "patch_px": 8}}
    t["norms"] = {"ew": {"HH": [-12.0, 6.0], "HV": [-22.0, 7.0]}}
    t["run"]["extra"] = [{"variant": "upsample", "stack": "noTNR", "norm": "ew"},
                         {"variant": "upsample", "stack": "TNR"}]          # duplicate of a base run
    rp = ThorRunParams.from_config(Config(root=tmp_path, raw=raw))
    assert [r.tag for r in rp.runs] == ["thor-upsample_TNR", "thor-upsample_noTNR",
                                        "thor-upsample-ewnorm_noTNR"]
    assert rp.norms["ew"]["HH"] == (-12.0, 6.0) and set(rp.norms) == {"iw", "ew"}
    t["run"]["extra"] = [{"variant": "upsample", "stack": "noTNR", "norm": "sar"}]
    with pytest.raises(ValueError):
        ThorRunParams.from_config(Config(root=tmp_path, raw=raw))


def test_label_and_water_scores():
    prob = np.zeros((10, 10), np.float32)
    prob[2, 3] = 0.8                                         # one token off the label
    prob[8, 8] = 0.3                                         # water far from labels
    s = label_scores(prob, np.array([[2, 2], [6, 0]]))
    np.testing.assert_allclose(s, [0.8, 0.0])
    w = water_scores(prob, np.ones_like(prob, bool), np.array([[2, 2]]), r=3)
    assert w.max() == pytest.approx(0.3) and 0.8 not in w
