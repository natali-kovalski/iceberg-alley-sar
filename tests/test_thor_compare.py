import geopandas as gpd
import numpy as np
import pandas as pd

from iceberg_sar.pointval import PointValParams
from iceberg_sar.thor.compare import (
    CompareParams,
    label_bins,
    open_water,
    pfa_label,
    pooled,
    score_points,
    shared_drift,
)


def test_pooled_sums_counts_and_recomputes_recall() -> None:
    sweep = pd.DataFrame({"scene": ["a", "a", "b", "b"], "detector": ["cfar", "thor"] * 2,
                          "setting": [1e-6, 0.5] * 2, "detections": [10, 12, 30, 28],
                          "matched": [5, 8, 20, 24], "truth": [10, 10, 40, 40]})
    out = pooled(sweep, ["detector", "setting"], ["detections", "matched", "truth"], "truth")
    cfar = out[out["detector"] == "cfar"].iloc[0]
    assert (cfar["detections"], cfar["matched"], cfar["truth"]) == (40, 25, 50)
    assert cfar["recall"] == 0.5 and (out["scene"] == "all scored scenes").all()


def test_open_water_drops_near_ice_keeps_nan() -> None:
    det = gpd.GeoDataFrame({"distance_to_ice_km": [1.0, 7.0, np.nan]},
                           geometry=gpd.points_from_xy([0, 1, 2], [0, 0, 0]))
    assert open_water(det, 5).index.tolist() == [1, 2]


def test_score_points_flags_matched_labels() -> None:
    truth = np.array([[0.0, 0.0], [1000.0, 0.0], [5000.0, 0.0]])
    det = np.array([[50.0, 0.0], [1100.0, 0.0], [9000.0, 0.0]])
    tp, hit = score_points(det, truth, 200)
    assert tp == 2 and hit.tolist() == [True, True, False]


def test_shared_drift_pools_detectors_and_needs_time_gap() -> None:
    rng = np.random.default_rng(0)
    bergs = rng.uniform(0, 20_000, (30, 2))
    shift = np.array([3000.0, -1500.0])
    vp = PointValParams()
    # Each detector sees half the bergs; pooled they line up with all the drifted truth.
    d = shared_drift([bergs[:15], bergs[15:]], bergs + shift, gap_s=5 * 3600, vp=vp)
    assert d.applied and abs(d.dx_m - 3000) < 50 and abs(d.dy_m + 1500) < 50
    same_pass = shared_drift([bergs], bergs, gap_s=60, vp=vp)
    assert not same_pass.applied and same_pass.dx_m == 0


def test_shared_drift_rejects_chance_alignment() -> None:
    rng = np.random.default_rng(1)
    d = shared_drift([rng.uniform(0, 50_000, (5, 2))], rng.uniform(0, 50_000, (5, 2)),
                     gap_s=5 * 3600, vp=PointValParams())
    assert not d.applied


def test_label_bins() -> None:
    p = CompareParams(out_dir=None, scenes={}, cfar_pfas=[], cfar_pfa=1e-6, thor_models=[],
                      view_score=0.35, radii_m=[], ice_bins_km=[0, 5, 20, 1000],
                      incidence_bins_deg=[29, 35, 40, 47], near_ice_km=5)
    truth = pd.DataFrame({"size": ["SM", "MED", "LG", "VLG", "GEN"],
                          "distance_to_ice_km": [2.0, 10.0, 50.0, np.nan, 30.0],
                          "incidence_deg": [30.0, 36.0, 41.0, 44.0, 33.0]})
    b = label_bins(truth, p)
    assert b["size"].tolist() == ["small 15-60 m", "medium 61-120 m", "large >120 m",
                                  "large >120 m", "other / unknown"]
    assert b["ice_km"].tolist()[:2] == ["0-5 km", "5-20 km"]
    assert b["ice_km"].iloc[2] == ">20 km" and b["ice_km"].iloc[3] == "no ice in scene"
    assert b["incidence_deg"].iloc[0] == "29-35°"


def test_pfa_label() -> None:
    assert pfa_label(1e-6) == "pfa_1e-06"
