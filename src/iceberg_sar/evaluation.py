"""The shared evaluation mask (revision C3): one scoring domain for labels and detections.

Every score goes through `EvalDomain.keep`. A label (at its SAR-time position) and a detection
are kept or dropped by the same test, so no zone is excluded for one side only: excluding
detections from a zone while its labels stay in the denominator caps recall at labels the
detector is never allowed to find.

- `area`: polygon in the scoring CRS (Færch EW scene: the Fig. 1 scoring area). None: no limit.
- `near_ice_km`: distance to the buffered pack-ice mask (`seaice.distance_to_ice_km`, the same
  function and mask for both sides). `zone="open_water"` keeps points at >= `near_ice_km`
  (IW frames: 5 km); `zone="near_ice"` keeps only those closer (revision C8). A NaN distance
  (no ice in the scene, off the mask grid) counts as open water.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import shapely

Zone = Literal["open_water", "near_ice"]


@dataclass(frozen=True)
class EvalDomain:
    area: Any = None
    near_ice_km: float | None = None
    zone: Zone = "open_water"

    def keep(self, xy: np.ndarray, ice_km: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask of the points (n, 2) inside the domain; `ice_km` per point if the
        domain uses the pack-ice distance."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        ok = np.ones(len(xy), bool)
        if self.area is not None and len(xy):
            ok &= shapely.contains_xy(self.area, xy[:, 0], xy[:, 1])
        if self.near_ice_km is not None:
            if ice_km is None:
                raise ValueError("this domain needs the distance to pack ice of every point")
            d = np.asarray(ice_km, float)
            near = d < self.near_ice_km          # NaN -> False: open water
            ok &= near if self.zone == "near_ice" else ~near
        return ok

    def describe(self) -> str:
        parts = []
        if self.area is not None:
            parts.append("inside the scoring area")
        if self.near_ice_km is not None:
            parts.append(f"{'within' if self.zone == 'near_ice' else 'at least'} "
                         f"{self.near_ice_km:g} km {'of' if self.zone == 'near_ice' else 'from'} "
                         "pack ice")
        return ", ".join(parts) or "everywhere"
