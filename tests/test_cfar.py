"""CA-CFAR tests on synthetic gamma-distributed clutter."""

import numpy as np
import pytest

from iceberg_sar.cfar import CfarParams, ca_cfar, estimate_enl, ring_mean, threshold_factor


def test_threshold_factor_matches_exponential_limit() -> None:
    # Single-look (ENL=1) clutter with a near-perfect background estimate: alpha -> -ln(pfa).
    assert threshold_factor(1e-6, 1.0, 10**7) == pytest.approx(-np.log(1e-6), rel=1e-3)


def test_threshold_factor_decreases_with_looks() -> None:
    # More looks -> less speckle -> a lower threshold gives the same false-alarm rate.
    assert threshold_factor(1e-6, 5.0, 1560) < threshold_factor(1e-6, 1.0, 1560)


def test_ring_mean_ignores_nan_and_guard() -> None:
    x = np.full((61, 61), 2.0, dtype=np.float32)
    x[30, 30] = 1000.0            # target inside the guard window
    x[5:15, 5:15] = np.nan        # masked patch
    mean, n = ring_mean(x, guard_px=2, background_px=6)
    assert mean[30, 30] == pytest.approx(2.0)
    assert n[30, 30] == 13 * 13 - 5 * 5
    assert n[10, 20] < 13 * 13 - 5 * 5  # ring overlaps the masked patch


def test_empirical_false_alarm_rate() -> None:
    rng = np.random.default_rng(0)
    enl, pfa = 4.0, 1e-3
    x = rng.gamma(enl, 0.01 / enl, size=(600, 600)).astype(np.float32)
    det, _ = ca_cfar(x, CfarParams(guard_px=2, background_px=8, pfa=pfa, enl=enl))
    inner = det[8:-8, 8:-8]
    assert 0.5 * pfa < inner.mean() < 2 * pfa


def test_detects_bright_target_and_skips_mostly_masked_ring() -> None:
    rng = np.random.default_rng(1)
    x = rng.gamma(4.0, 0.01 / 4.0, size=(100, 100)).astype(np.float32)
    x[50, 50] = 1.0                   # +20 dB
    x[:, 80:] = np.nan
    x[40, 78] = 1.0                   # bright, but its ring is > 50 % masked
    params = CfarParams(guard_px=2, background_px=8, pfa=1e-6, enl=4.0,
                        min_background_fraction=0.6)
    det, _ = ca_cfar(x, params)
    assert det[50, 50]
    assert not det[40, 78]
    assert not det[:, 80:].any()


def test_estimate_enl_recovers_gamma_looks() -> None:
    rng = np.random.default_rng(2)
    blocks = rng.gamma(5.0, 1 / 5.0, size=(400, 25 * 25))
    cv = blocks.std(axis=1) / blocks.mean(axis=1)
    assert estimate_enl(cv) == pytest.approx(5.0, rel=0.15)


def test_threshold_factor_finite_and_exact_at_tiny_pfa() -> None:
    from scipy import special

    # Færch sweep range: scipy's F isf returns inf below ~1e-16
    for pfa in (1e-21, 1e-16, 1e-6):
        a = threshold_factor(pfa, 13.4, 88)
        d1, d2 = 2 * 13.4, 2 * 13.4 * 88
        assert np.isfinite(a)
        assert special.betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * a)) == pytest.approx(pfa, rel=1e-6)
    assert threshold_factor(1e-21, 13.4, 88) > threshold_factor(1e-16, 13.4, 88)
