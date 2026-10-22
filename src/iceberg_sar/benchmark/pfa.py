"""Total vs per-channel probability of false alarm for HH/HV Boolean merging.

Færch et al. 2023, sec. 2.1.1: with independent HH and HV noise and the same PFA on both
channels,

  AND: PFA_total = PFA_ch²               ->  PFA_ch = sqrt(PFA_total)
  OR : PFA_total = 2 PFA_ch - PFA_ch²    ->  PFA_ch = 1 - sqrt(1 - PFA_total)  (Šidák)

The OR form is computed as -expm1(log1p(-p) / 2): written as 1 - sqrt(1 - p) it cancels to 0
in float64 for p below ~1e-16, inside the paper's 1e-21 ... 1e-1 sweep.
"""

from __future__ import annotations

import math
from typing import Literal

Merge = Literal["AND", "OR"]

# Detectors that run per channel and merge with AND/OR; the others test one statistic.
BOOLEAN_MERGED = frozenset({"gamma", "lognorm", "k"})


def _check(p: float) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError(f"PFA must be in (0, 1), got {p}")


def per_channel_pfa(total: float, merge: Merge) -> float:
    """PFA to apply on each of HH and HV so the merged result has PFA `total`."""
    _check(total)
    if merge == "AND":
        return math.sqrt(total)
    if merge == "OR":
        return -math.expm1(0.5 * math.log1p(-total))
    raise ValueError(f"merge must be 'AND' or 'OR', got {merge!r}")


def combined_pfa(per_channel: float, merge: Merge) -> float:
    """Merged PFA from the same per-channel PFA on HH and HV (inverse of `per_channel_pfa`)."""
    _check(per_channel)
    if merge == "AND":
        return per_channel * per_channel
    if merge == "OR":
        return -math.expm1(2.0 * math.log1p(-per_channel))
    raise ValueError(f"merge must be 'AND' or 'OR', got {merge!r}")


def detector_channel_pfa(detector: str, total: float, merge: Merge) -> float:
    """PFA each test of `detector` runs at: per channel if Boolean-merged, else the total."""
    return per_channel_pfa(total, merge) if detector in BOOLEAN_MERGED else total
