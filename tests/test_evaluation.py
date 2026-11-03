"""Revision C3: one evaluation mask for labels and detections."""

import numpy as np
import pytest
from shapely.geometry import box

from iceberg_sar.evaluation import EvalDomain


def test_area_keeps_points_inside() -> None:
    d = EvalDomain(area=box(0, 0, 10, 10))
    assert d.keep(np.array([[5, 5], [15, 5]])).tolist() == [True, False]


def test_open_water_and_near_ice_are_complements() -> None:
    xy = np.zeros((4, 2))
    ice = np.array([1.0, 5.0, 12.0, np.nan])     # NaN: no ice in the scene -> open water
    ow = EvalDomain(near_ice_km=5).keep(xy, ice)
    near = EvalDomain(near_ice_km=5, zone="near_ice").keep(xy, ice)
    assert ow.tolist() == [False, True, True, True]
    assert (ow ^ near).all()


def test_ice_domain_needs_distances() -> None:
    with pytest.raises(ValueError):
        EvalDomain(near_ice_km=5).keep(np.zeros((1, 2)))


def test_same_test_for_labels_and_detections() -> None:
    """The bug this fixes: detections near ice dropped, their labels kept as misses."""
    d = EvalDomain(near_ice_km=5)
    lab_ice = np.array([2.0, 30.0])
    det_ice = np.array([2.1, 29.0])
    assert d.keep(np.zeros((2, 2)), lab_ice).tolist() == d.keep(np.zeros((2, 2)), det_ice).tolist()
