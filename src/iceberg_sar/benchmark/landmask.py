"""Milestone 1c: land mask for the Færch 2023 scene, as in the paper (sec. 3.1).

"A land mask was created from OpenStreetMap land polygons ... A buffer of 500 m was added to
the mask to avoid any issues with SAR layover or bad geocoding."

OSM land polygons (split, EPSG:4326) are read for the scene's bbox, reprojected to the scene
CRS, buffered by `land.buffer_m`, and rasterized onto the scene grid (1 = land). Unlike the
Stage 1 mask, point-mapped rocks are not added: the paper used the land polygons only.
"""

from __future__ import annotations

import re

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

from iceberg_sar.config import Config
from iceberg_sar.landmask import download


def land_polygons(cfg: Config, bbox_4326: tuple[float, float, float, float], crs: str,
                  margin_deg: float = 0.2) -> gpd.GeoDataFrame:
    """OSM land polygons intersecting the bbox (plus margin), in `crs`.

    Source: the osmdata.openstreetmap.de land polygons. If that server can't be reached, the
    same polygons are built from the OSM coastline ways for the bbox (`coastline_land`): from
    the local extract `benchmark.faerch2023.land.osm_pbf` if set, else via Overpass. Cached
    under data/land/. Column `source` says which one was used.
    """
    from urllib.error import URLError

    p = cfg.section("preprocess")
    pbf = cfg.section("benchmark")["faerch2023"]["land"].get("osm_pbf")
    w, s, e, n = bbox_4326
    bbox = (w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg)
    land_dir = cfg.path("land_mask").parent
    try:
        zip_path = download(p["land_polygons_url"], land_dir)
    except (URLError, OSError) as err:
        cache = land_dir / ("osm_coastline_land_" + "_".join(f"{v:.2f}" for v in bbox) + ".gpkg")
        if not cache.is_file():
            if pbf:
                print(f"land polygons server unreachable ({err}); building land from the "
                      f"coastline ways in {Path(pbf).name}")
                ways = coastline_from_pbf(cfg.root / pbf, bbox, land_dir)
                src = f"OSM coastline ways ({Path(pbf).name}), polygonized"
            else:
                print(f"land polygons server unreachable ({err}); building land from OSM "
                      "coastline ways via Overpass")
                ways = fetch_coastline(p["overpass_urls"], bbox)
                src = "OSM coastline ways (Overpass), polygonized"
            coastline_land(ways, bbox).assign(source=src).to_file(cache, driver="GPKG")
        return gpd.read_file(cache).to_crs(crs)
    land = gpd.read_file(f"zip://{zip_path}!{p['land_polygons_member']}", bbox=bbox)
    return land.assign(source="osmdata land-polygons-split-4326").to_crs(crs)


def coastline_from_pbf(pbf: Path, bbox: tuple[float, float, float, float], work_dir: Path
                       ) -> list[list]:
    """natural=coastline ways in the bbox from an OSM .pbf extract, as lists of (lon, lat).

    GDAL's OSM driver turns closed ways with area tags (natural=*, place=island, ...) into
    polygons, which loses the way direction (land is on its left) and drops most islands from
    the lines layer. A copy of its osmconf.ini keeps every closed way as a line and exposes
    `natural` as a field of the lines layer.
    """
    import pyogrio

    default = Path(pyogrio.__file__).parent / "gdal_data" / "osmconf.ini"
    conf = work_dir / "osmconf_coastline.ini"
    text = default.read_text(encoding="utf-8")
    text = re.sub(r"^closed_ways_are_polygons=.*$", "closed_ways_are_polygons=no_such_tag", text,
                  flags=re.M)
    text = re.sub(r"(\[lines\][\s\S]*?^attributes=)", r"\1natural,", text, count=1, flags=re.M)
    work_dir.mkdir(parents=True, exist_ok=True)
    conf.write_text(text, encoding="utf-8")
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(conf)})
    try:
        gdf = pyogrio.read_dataframe(pbf, layer="lines", where="natural = 'coastline'", bbox=bbox)
    finally:
        pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": None})
    ways = []
    for g in gdf.geometry:
        for part in getattr(g, "geoms", [g]):
            ways.append(list(part.coords))
    return ways


def fetch_coastline(urls: list[str], bbox: tuple[float, float, float, float]) -> list[list]:
    """natural=coastline ways in the bbox from Overpass, as lists of (lon, lat)."""
    import json
    import urllib.parse
    import urllib.request

    w, s, e, n = bbox
    q = f'[out:json][timeout:600];way["natural"="coastline"]({s},{w},{n},{e});out geom;'
    data = urllib.parse.urlencode({"data": q}).encode()
    errors = []
    for url in urls:
        req = urllib.request.Request(url, data=data, headers={"User-Agent": "iceberg-alley-sar"})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                ways = json.load(resp)["elements"]
            return [[(pt["lon"], pt["lat"]) for pt in wy["geometry"]] for wy in ways]
        except OSError as err:
            errors.append(f"{url}: {err}")
    raise RuntimeError("All Overpass endpoints failed:\n" + "\n".join(errors))


def coastline_land(ways: list[list], bbox: tuple[float, float, float, float]
                   ) -> gpd.GeoDataFrame:
    """Land polygons inside `bbox` from OSM coastline ways (EPSG:4326).

    OSM convention: land is on the left of a coastline way. The ways and the bbox outline are
    noded and polygonized into faces; with each face oriented so that its interior is on the
    left of every ring (exterior counter-clockwise, holes clockwise), a face is land when a
    coastline segment runs along its boundary in the same direction, and sea when one runs
    the opposite way. This is what osmcoastline does to build the land polygons, for one bbox.

    Face edges are matched to coastline segments by geometry (the edge midpoint lies on the
    segment), since clipping and noding cut segments into pieces.

    A coastline that ends inside the bbox (an extract cut at a land border) leaves a face
    open; each such loose end is joined straight to the nearest point of the bbox outline,
    so land is only uncertain beyond that connector. The count is in `attrs["loose_ends"]`.
    """
    import shapely
    from shapely.geometry import LineString, box
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union

    frame = box(*bbox)
    if not any(len(c) >= 2 for c in ways):  # open sea: no coastline, no land
        out = gpd.GeoDataFrame({"source": []}, geometry=[], crs="EPSG:4326")
        out.attrs["loose_ends"] = 0
        return out
    seg = np.array([(a, b) for c in ways for a, b in zip(c[:-1], c[1:], strict=True)
                    if a != b], dtype=float)                       # (n, 2 points, 2 coords)
    tree = shapely.STRtree(shapely.linestrings(seg))
    seg_dir = seg[:, 1] - seg[:, 0]
    lines = [LineString(c) for c in ways if len(c) >= 2]
    ends: dict[tuple, int] = {}
    for c in ways:
        if len(c) >= 2 and tuple(c[0]) != tuple(c[-1]):
            for p in (tuple(c[0]), tuple(c[-1])):
                ends[p] = ends.get(p, 0) + 1
    inner = frame.buffer(-1e-9)
    loose = [p for p, k in ends.items() if k == 1 and inner.contains(shapely.Point(p))]
    edge = frame.exterior
    lines += [LineString([p, edge.interpolate(edge.project(shapely.Point(p))).coords[0]])
              for p in loose]
    clipped = [g for g in (ln.intersection(frame) for ln in lines) if not g.is_empty]
    noded = unary_union([*clipped, frame.boundary])
    land = []
    for face in shapely.polygonize(list(getattr(noded, "geoms", [noded]))).geoms:
        f = orient(face, 1.0)
        votes = 0.0
        for ring in (f.exterior, *f.interiors):
            c = np.asarray(ring.coords)
            mid = shapely.points((c[:-1] + c[1:]) / 2)
            edge, hit = tree.query(mid, predicate="dwithin", distance=1e-9)
            d = c[1:] - c[:-1]
            votes += float(np.sign((d[edge] * seg_dir[hit]).sum(axis=1)).sum())
        if votes > 0:
            land.append(face)
    out = gpd.GeoDataFrame({"source": "OSM coastline ways (Overpass), polygonized"},
                           index=range(len(land)), geometry=land, crs="EPSG:4326")
    out.attrs["loose_ends"] = len(loose)
    if loose:
        print(f"coastline: {len(loose)} loose ends inside the bbox joined to its outline")
    return out


def land_mask(cfg: Config, grid: Path, out: Path, buffer_m: float) -> Path:
    """uint8 raster on `grid`'s pixels: 1 = land or within `buffer_m` of it."""
    with rasterio.open(grid) as src:
        profile, crs, transform, shape = src.profile, src.crs, src.transform, src.shape
        bbox = transform_bounds(crs, "EPSG:4326", *src.bounds)
    land = land_polygons(cfg, bbox, crs.to_string())
    geoms = land.geometry.buffer(buffer_m) if buffer_m else land.geometry
    mask = rasterize(((g, 1) for g in geoms if not g.is_empty), out_shape=shape,
                     transform=transform, fill=0, dtype="uint8", all_touched=True)
    profile.update(count=1, dtype="uint8", nodata=None, compress="deflate", predictor=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.update_tags(buffer_m=str(buffer_m), source="; ".join(sorted(set(land["source"]))))
    return out


def mask_overlay(hh_db: np.ndarray, mask: np.ndarray, png: Path, title: str,
                 stretch: tuple[float, float], max_px: int = 2400) -> Path:
    """HH dB with the land mask in translucent purple (the paper's Fig. 3 convention)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(1, int(np.ceil(max(hh_db.shape) / max_px)))
    img, m = hh_db[::step, ::step], mask[::step, ::step]
    h, w = img.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap="gray", vmin=stretch[0], vmax=stretch[1], interpolation="nearest")
    ax.imshow(np.ma.masked_where(m == 0, m), cmap="Purples", vmin=0, vmax=1.5, alpha=0.55,
              interpolation="nearest")
    ax.text(15, 15, title, color="yellow", fontsize=18, va="top")
    ax.set_axis_off()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, facecolor="black")
    plt.close(fig)
    return png
