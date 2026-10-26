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
    assert (s["TP_v"], s["FP_v"], s["FN_v"], s["ignored_v"]) == (1, 1, 0, 2)


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
