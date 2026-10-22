"""Land polygons from OSM coastline ways (fallback when the land-polygon server is down)."""

from shapely.geometry import Point

from iceberg_sar.benchmark.landmask import coastline_land

BBOX = (0.0, 0.0, 10.0, 10.0)
# Mainland coast running north along x=4 (land on the left = west), crossing the bbox.
MAINLAND = [(4.0, -1.0), (4.0, 5.0), (4.0, 11.0)]
# Island: closed ring, counter-clockwise = land inside.
ISLAND = [(6.0, 6.0), (8.0, 6.0), (8.0, 8.0), (6.0, 8.0), (6.0, 6.0)]
# Lake-like ring inside the island would be clockwise; here a clockwise ring in the sea is an
# inland-water hole drawn the wrong way round, so test only the documented cases.


def test_mainland_and_island() -> None:
    land = coastline_land([MAINLAND, ISLAND], BBOX).union_all()
    assert land.contains(Point(2, 5))        # west of the coast
    assert land.contains(Point(7, 7))        # island
    assert not land.contains(Point(5, 2))    # sea
    assert not land.contains(Point(9, 9))    # sea beside the island
    assert abs(land.area - (4 * 10 + 2 * 2)) < 1e-9


def test_reversed_direction_flips_land() -> None:
    land = coastline_land([MAINLAND[::-1]], BBOX).union_all()
    assert land.contains(Point(6, 5)) and not land.contains(Point(2, 5))


def test_island_only_sea_has_hole() -> None:
    gdf = coastline_land([ISLAND], BBOX)
    assert len(gdf) == 1 and abs(gdf.area.sum() - 4) < 1e-9


def test_loose_end_is_closed_to_the_frame() -> None:
    # Coast enters from the south and stops inside the bbox (an extract cut at a border).
    stub = [(4.0, -1.0), (4.0, 5.0), (4.0, 7.0)]
    gdf = coastline_land([stub], BBOX)
    assert gdf.attrs["loose_ends"] == 1
    land = gdf.union_all()
    assert land.contains(Point(2, 3)) and not land.contains(Point(6, 3))


def test_no_coastline_is_all_sea() -> None:
    assert coastline_land([], BBOX).empty
