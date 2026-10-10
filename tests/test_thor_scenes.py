import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from iceberg_sar.thor.scenes import SceneParams, group_passes, in_holdout, select_frames


def params(**kw) -> SceneParams:
    april = (pd.Timestamp("2019-04-01"), pd.Timestamp("2019-05-01"))
    base = dict(seasons=(2019,), holdout=(april,), pol="HH+HV", pass_gap_min=30, match_min=5,
                min_sightings=1, max_scenes=2, credits_per_scene=60, max_credits=1000)
    base.update(kw)
    return SceneParams(**base)


def test_in_holdout_is_half_open():
    t = pd.Series(pd.to_datetime(["2019-03-31 23:59", "2019-04-01 00:00", "2019-04-30 23:59",
                                  "2019-05-01 00:00"]))
    assert in_holdout(t, params().holdout).tolist() == [False, True, True, False]


def test_group_passes_splits_on_time_gap():
    t = pd.to_datetime(["2019-05-05 09:39", "2019-05-05 09:41", "2019-05-05 21:23",
                        "2019-05-06 09:31"])
    g = gpd.GeoDataFrame({"time": t}, geometry=[Point(0, 0)] * 4, crs="EPSG:4326")
    assert group_passes(g, 30)["pass_id"].tolist() == [0, 0, 1, 2]


def test_select_frames_filters_pol_ranks_and_caps():
    f = gpd.GeoDataFrame({
        "scene": ["a", "b", "c", "d"],
        "start_time": ["2019-01-04", "2019-01-01", "2019-01-02", "2019-01-03"],
        "polarization": ["HH+HV", "HH+HV", "VV+VH", "HH+HV"],
        "aoi_sightings": [5, 30, 99, 20],
    }, geometry=[box(0, 0, 1, 1)] * 4, crs="EPSG:4326")
    assert select_frames(f, params())["scene"].tolist() == ["b", "d"]  # top 2, by time
    assert len(select_frames(f, params(max_credits=60))) == 1          # credit cap wins
