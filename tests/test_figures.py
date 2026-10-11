import geopandas as gpd
from shapely.geometry import Point

from iceberg_sar.figures import recall_by_size


def test_recall_by_size_orders_classes_and_skips_uncompared():
    t = gpd.GeoDataFrame({
        "size": ["MED", "SM", "MED", "LG", "GEN", "SM"],
        "compared": [True, True, True, True, True, False],
        "matched": [True, False, False, False, True, True],
    }, geometry=[Point(0, 0)] * 6)
    r = recall_by_size(t)
    assert list(r.index) == ["SM", "MED", "LG", "GEN"]   # IIP size order, unknown codes last
    assert r.loc["SM", "total"] == 1 and r.loc["SM", "found"] == 0
    assert r.loc["MED", "recall"] == 0.5
    assert r.loc["GEN", "label"] == "GEN"
