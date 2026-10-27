"""Cell-averaging CFAR (CA-CFAR) on linear SAR intensity.

For each pixel, the local clutter level is the mean of the *background ring*: a square
window of half-width `background_px` minus an inner guard window of half-width `guard_px`
(the guard keeps a target's own energy out of its clutter estimate). A pixel is a detection
when `intensity > alpha * ring_mean`.

Threshold factor `alpha` from the false-alarm probability (Pfa): speckled sea clutter is
modelled as gamma-distributed intensity with ENL L (equivalent number of looks; L=1 is
single-look exponential speckle, multilooked GRD products have L > 1). The ratio of one
pixel to the mean of N independent background pixels then follows an F(2L, 2NL)
distribution, so `alpha = F.isf(pfa, 2L, 2NL)`. Real sea clutter has a heavier tail than
gamma, so the realised false-alarm rate is higher than `pfa`. Estimating L from the scene
itself (see `estimate_enl`) partly absorbs this.

Implementation: window sums come from box filters (`uniform_filter`, O(1) per pixel for
any window size). NaN (masked land / ice / nodata) pixels are excluded by filtering values
and valid-counts separately, so the ring mean uses only real water pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, special


@dataclass(frozen=True)
class CfarParams:
    guard_px: int
    background_px: int
    pfa: float
    enl: float
    min_background_fraction: float = 0.5  # of the full ring; fewer valid pixels -> no decision

    def __post_init__(self) -> None:
        if not 0 <= self.guard_px < self.background_px:
            raise ValueError("Need 0 <= guard_px < background_px")
        if not 0 < self.pfa < 1:
            raise ValueError("pfa must be in (0, 1)")
        if self.enl <= 0:
            raise ValueError("enl must be > 0")

    @property
    def ring_pixels(self) -> int:
        return (2 * self.background_px + 1) ** 2 - (2 * self.guard_px + 1) ** 2

    @property
    def alpha(self) -> float:
        return threshold_factor(self.pfa, self.enl, self.ring_pixels)


def threshold_factor(pfa: float, enl: float, n_background: int) -> float:
    """CA-CFAR multiplier on the background mean for gamma(L) clutter and N ring pixels.

    `stats.f.isf` works through 1 - pfa and returns inf below ~1e-16 (the Færch sweep goes
    to 1e-21), so invert the F survival function directly: P(F > x) = I_w(d2/2, d1/2) with
    w = d2 / (d2 + d1 x), the regularized incomplete beta function.
    """
    d1, d2 = 2 * enl, 2 * enl * n_background
    w = special.betaincinv(d2 / 2, d1 / 2, pfa)
    return float(d2 * (1 - w) / (d1 * w))


def _box_sum(a: np.ndarray, half: int) -> np.ndarray:
    size = 2 * half + 1
    return ndimage.uniform_filter(a, size=size, mode="constant", cval=0.0) * (size * size)


def ring_mean(
    intensity: np.ndarray, guard_px: int, background_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of valid (finite) pixels in the background ring, and the valid-pixel count.

    Pixels outside the array count as invalid, so image edges are handled like masks.
    """
    valid = np.isfinite(intensity)
    vals = np.where(valid, intensity, 0.0).astype(np.float64)
    ones = valid.astype(np.float64)
    s = _box_sum(vals, background_px) - _box_sum(vals, guard_px)
    n = _box_sum(ones, background_px) - _box_sum(ones, guard_px)
    n = np.rint(n)  # box filters leave tiny float residue on integer counts
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
    return mean.astype(np.float32), n.astype(np.int32)


def ca_cfar(intensity: np.ndarray, params: CfarParams) -> tuple[np.ndarray, np.ndarray]:
    """Detection mask and background mean (linear) for a linear-intensity array (NaN = masked)."""
    bg, n = ring_mean(intensity, params.guard_px, params.background_px)
    enough = n >= params.min_background_fraction * params.ring_pixels
    with np.errstate(invalid="ignore"):
        det = np.isfinite(intensity) & enough & (intensity > params.alpha * bg)
    return det, bg


def estimate_enl(cv: np.ndarray) -> float:
    """ENL from per-cell coefficients of variation over water: L = 1 / median(CV)^2.

    The median is robust to the few cells holding targets or ice. Sea-state gradients
    inside cells inflate CV, so L comes out slightly low, which makes CFAR conservative.
    """
    finite = cv[np.isfinite(cv) & (cv > 0)]
    if finite.size == 0:
        raise ValueError("No valid cells to estimate ENL from")
    return float(1.0 / np.median(finite) ** 2)
