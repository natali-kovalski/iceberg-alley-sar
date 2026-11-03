"""1f scoring: greedy one-to-one matching, all-labels vs verified-only, reference parsing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iceberg_sar.benchmark import score

ROOT = Path(__file__).resolve().parents[1]


def _labels(rows: list[tuple[float, float, str]]) -> pd.DataFrame:
    return pd.DataFrame({
        "x": [r[0] for r in rows], "y": [r[1] for r in rows], "status": [r[2] for r in rows],
        "radius_m": [250.0 if r[2] == "verified" else 1789.0 for r in rows],
        "size_class": ["small"] * len(rows)})


def test_greedy_match_closest_first_one_to_one() -> None:
    det = np.array([[0.0, 0.0], [100.0, 0.0], [5000.0, 0.0]])
    lab = np.array([[90.0, 0.0], [10.0, 0.0]])
    d, lab_i = score.greedy_match(det, lab, np.array([250.0, 250.0]))
    assert sorted(zip(d.tolist(), lab_i.tolist(), strict=True)) == [(0, 1), (1, 0)]


def test_greedy_match_label_radius() -> None:
    det = np.array([[1000.0, 0.0]])
    lab = np.array([[0.0, 0.0], [0.0, 0.0]])
    d, lab_i = score.greedy_match(det, lab, np.array([250.0, 1789.0]))
    assert lab_i.tolist() == [1]


def test_score_points_all_and_verified() -> None:
    labels = _labels([(0, 0, "verified"), (10_000, 0, "interpolated")])
    det = np.array([[50.0, 0.0],        # verified label: TP in both
                    [11_000.0, 0.0],    # interpolated label: TP all-labels, ignored verified-only
                    [10_500.0, 0.0],    # second det near the interpolated label: FP / ignored
                    [50_000.0, 0.0]])   # far from everything: FP in both
    s = score.score_points(det, labels)
    assert (s["TP"], s["FP"], s["FN"]) == (2, 2, 0)
    assert s["f"] == pytest.approx(2 * 2 / (2 * 2 + 2))
    # revision C5: verified-only recall from the same matching, no verified-only F
    assert (s["n_v"], s["TP_v"], s["recall_v"]) == (1, 1, 1.0)
    assert "f_v" not in s and "precision_v" not in s


def test_score_points_no_detections() -> None:
    s = score.score_points(np.empty((0, 2)), _labels([(0, 0, "verified")]))
    assert (s["TP"], s["FP"], s["FN"]) == (0, 0, 1) and s["f"] == 0


def test_parse_reference_tables() -> None:
    ref = score.parse_reference(ROOT / "benchmarks/faerch2023/faerch2023.xml")
    t3, t5 = ref["table3"], ref["table5"]
    assert set(t3) == set(t5) == set(score.PAPER_NAMES.values())
    for e in t3.values():   # every berg is a TP or an FN
        assert e["TP"] + e["FN"] == 492
    for e in t5.values():
        assert sum(e["n"].values()) == 492
    assert t3["gamma"]["derived"]["f"] == pytest.approx(
        2 * t3["gamma"]["TP"] / (2 * t3["gamma"]["TP"] + t3["gamma"]["FP"] + t3["gamma"]["FN"]),
        abs=1e-4)


def test_optimum_lowest_pfa_on_ties() -> None:
    df = pd.DataFrame({"detector": ["a"] * 3, "pfa": [1e-3, 1e-9, 1e-5], "f": [0.5, 0.6, 0.6]})
    assert score.optimum(df).loc["a"].pfa == pytest.approx(1e-9)


def test_compare_and_size_tables_add_my_cacfar_without_paper_values() -> None:
    ref = score.parse_reference(ROOT / "benchmarks/faerch2023/faerch2023.xml")
    rows = []
    for det in ["gamma", "cacfar"]:
        for pfa, f in [(1e-9, 0.5), (1e-5, 0.6)]:
            rows.append({"detector": det, "pfa": pfa, "f": f, "recall_v": f, "TP": 10, "FP": 5,
                         "FN": 3, "recall_small": 0.1, "recall_medium": 0.5,
                         "recall_large": 0.9})
    cmp = score.compare_table(pd.DataFrame(rows), ref, 0.05, None)
    mine = cmp.set_index("detector").loc["cacfar"]
    assert mine.pfa == pytest.approx(1e-5) and np.isnan(mine.paper_f) and mine.within is None
    sizes = score.size_table(pd.DataFrame(rows), ref)
    assert sizes[sizes.detector == "cacfar"].pct_at_my_opt.tolist() == pytest.approx([10, 50, 90])


def test_size_table_selects_the_exact_pfa_row() -> None:
    """Revision C1: np.isclose's default atol made every PFA < 1e-8 match the first row."""
    ref = score.parse_reference(ROOT / "benchmarks/faerch2023/faerch2023.xml")
    rows = [{"detector": "gamma", "pfa": 10.0 ** -k, "f": 0.1 * (k == 9), "recall_v": 0.0,
             "TP": 1, "FP": 1, "FN": 1, "recall_small": k / 100, "recall_medium": k / 100,
             "recall_large": k / 100} for k in range(21, 0, -1)]
    sizes = score.size_table(pd.DataFrame(rows), ref)
    paper_pfa = ref["table5"]["gamma"]["pfa"]
    k = round(-np.log10(paper_pfa))
    assert sizes[sizes.detector == "gamma"].pct_at_paper_pfa.tolist() == pytest.approx([k] * 3)
    assert score.same_pfa([1e-19, 1e-12, 1e-21], 1e-19).tolist() == [True, False, False]
    assert score.same_pfa([1e-19, 1e-12, 1e-21], 1e-12).tolist() == [False, True, False]


def test_idpolrad_tiny_pfas_flagged_and_never_optimal() -> None:
    """Revision C2: 1 - pfa is exactly 1.0 in float64 for pfa <= 1e-17."""
    sc = {"unreliable": {"pde_detectors": ["idpolrad"], "rel_err": 0.5}}
    assert score.is_unreliable("idpolrad", 1e-17, sc)
    assert not score.is_unreliable("idpolrad", 1e-16, sc)   # 11 % off: kept
    assert not score.is_unreliable("gamma", 1e-21, sc)
    df = pd.DataFrame({"detector": ["idpolrad"] * 2, "pfa": [1e-19, 1e-9], "f": [0.9, 0.4],
                       "unreliable": [True, False]})
    assert score.optimum(df).loc["idpolrad"].pfa == pytest.approx(1e-9)


def test_optimal_match_beats_greedy_on_a_chain() -> None:
    """Revision C4: greedy takes the closest pair and can lose a match; optimal can't."""
    det = np.array([[0.0, 0.0], [300.0, 0.0]])
    lab = np.array([[100.0, 0.0], [-150.0, 0.0]])
    r = np.array([250.0, 250.0])
    assert len(score.greedy_match(det, lab, r)[0]) == 1      # det0-lab0 (100 m) blocks lab1
    d, l_ = score.optimal_match(det, lab, r)
    assert sorted(zip(d.tolist(), l_.tolist(), strict=True)) == [(0, 1), (1, 0)]


def test_match_settings_radii() -> None:
    labels = _labels([(0, 0, "verified"), (0, 0, "interpolated")]).assign(scale_5938=[1.07, 1.07])
    ms = score.MatchSettings(radius_units="epsg5938", radius_interpolated_m=1000.0)
    r = score.apply_match_settings(labels, ms).radius_m.tolist()
    assert r == pytest.approx([250 / 1.07, 1000 / 1.07])
    with pytest.raises(ValueError):
        score.MatchSettings(matcher="hungarian").check()
