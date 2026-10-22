"""Total <-> per-channel PFA (Færch et al. 2023, sec. 2.1.1), against hand-worked values.

Hand-worked examples (checked by hand, not computed by the code under test):

AND, total 1e-6:  PFA_ch = sqrt(1e-6) = 1e-3                       (the paper's own example)
AND, total 1e-9:  PFA_ch = sqrt(1e-9) = 10^-4.5 = 3.16228e-5       (Milestone 0c setting)
OR,  total 1e-6:  PFA_ch = 1 - sqrt(1 - 1e-6). With sqrt(1-x) = 1 - x/2 - x²/8 - ...:
                  PFA_ch = x/2 + x²/8 = 5e-7 + 1.25e-13 = 5.00000125e-7  ("≈ 0.5e-6" in the paper)
                  check: 2·PFA_ch - PFA_ch² = 1.00000025e-6 - 2.5e-13 = 1.0e-6
OR,  total 1e-20: PFA_ch = 5e-21 (x²/8 term is 1.25e-41, negligible).
                  Written as 1 - sqrt(1 - 1e-20) in float64 this is exactly 0.
"""

import math

import pytest

from iceberg_sar.benchmark.pfa import combined_pfa, detector_channel_pfa, per_channel_pfa


@pytest.mark.parametrize(
    ("total", "merge", "expected"),
    [
        (1e-6, "AND", 1e-3),
        (1e-9, "AND", 3.16228e-5),
        (1e-6, "OR", 5.00000125e-7),
        (1e-20, "OR", 5e-21),
    ],
)
def test_hand_worked(total: float, merge: str, expected: float) -> None:
    assert per_channel_pfa(total, merge) == pytest.approx(expected, rel=1e-6)


def test_naive_or_form_underflows_but_ours_does_not() -> None:
    assert 1 - math.sqrt(1 - 1e-20) == 0.0
    assert per_channel_pfa(1e-20, "OR") > 0


@pytest.mark.parametrize("merge", ["AND", "OR"])
@pytest.mark.parametrize("total", [1e-21, 1e-12, 1e-6, 1e-3, 1e-1])
def test_round_trip(total: float, merge: str) -> None:
    assert combined_pfa(per_channel_pfa(total, merge), merge) == pytest.approx(total, rel=1e-9)


def test_and_is_stricter_per_channel_than_or() -> None:
    assert per_channel_pfa(1e-9, "AND") > 1e-9 > per_channel_pfa(1e-9, "OR")


def test_single_statistic_detectors_use_the_total() -> None:
    assert detector_channel_pfa("gamma", 1e-6, "AND") == pytest.approx(1e-3)
    for d in ("nis", "wishart", "idpolrad"):
        assert detector_channel_pfa(d, 1e-6, "AND") == 1e-6


@pytest.mark.parametrize("bad", [0.0, 1.0, -1e-3, 2.0])
def test_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        per_channel_pfa(bad, "AND")


def test_rejects_unknown_merge() -> None:
    with pytest.raises(ValueError):
        per_channel_pfa(1e-6, "XOR")  # type: ignore[arg-type]
