"""Milestone 1a: download the Færch 2023 benchmark products and map their footprints.

- Sentinel-1 EW GRD: zipped SAFE from CDSE (MD5 from the catalogue). Without CDSE
  credentials in `.env`, the identical product comes from ASF (MD5 from ASF).
- Sentinel-2 L1C: the original 2019 processing (N0207) only exists in Google's public bucket,
  so only the files in `s2_files` are fetched from there, each checked against its MD5.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from rasterio.features import shapes
from shapely.geometry import Polygon, shape

from iceberg_sar.benchmark.scenes import parse_product
from iceberg_sar.config import Config
from iceberg_sar.search import asf_download, cdse_download, cdse_lookup, stream_download


def raw_dir(cfg: Config) -> Path:
    return cfg.root / cfg.section("benchmark")["faerch2023"]["dir"] / "raw"


def s2_prefix(product_id: str) -> str:
    """Bucket prefix: tiles/<zone>/<band>/<square>/<product>.SAFE/"""
    tile = parse_product(product_id)["tile"]  # e.g. T20VNJ
    return f"tiles/{tile[1:3]}/{tile[3]}/{tile[4:6]}/{product_id}.SAFE/"


def s2_listing(product_id: str, bucket_url: str, wanted: list[str]) -> list[dict[str, Any]]:
    """Objects of one S2 product whose name ends with one of `wanted` (name, size, md5, url)."""
    items: list[dict[str, Any]] = []
    params: dict[str, str] = {"prefix": s2_prefix(product_id),
                              "fields": "nextPageToken,items(name,size,md5Hash,mediaLink)"}
    while True:
        r = requests.get(bucket_url, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()
        items += body.get("items", [])
        if "nextPageToken" not in body:
            break
        params["pageToken"] = body["nextPageToken"]
    picked = [i for i in items if any(i["name"].endswith(w) for w in wanted)]
    missing = [w for w in wanted if not any(i["name"].endswith(w) for i in picked)]
    if missing:
        raise LookupError(f"{product_id}: not in the bucket: {missing}")
    return picked


def download_s2(product_id: str, dest_dir: Path, bucket_url: str, wanted: list[str],
                log: Any = print) -> Path:
    """Wanted files into dest_dir/<product>.SAFE/... (same layout as the SAFE)."""
    safe = dest_dir / f"{product_id}.SAFE"
    for item in s2_listing(product_id, bucket_url, wanted):
        rel = item["name"].split(".SAFE/", 1)[1]
        dest = safe / rel
        if not dest.is_file():
            log(f"  {rel} ({int(item['size']) / 2**20:.0f} MB)")
        stream_download(item["mediaLink"], dest, item["md5Hash"])
    return safe


def download_s1(product_id: str, dest_dir: Path, cdse: dict[str, str], log: Any = print) -> Path:
    load_dotenv()
    if os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD"):
        product = cdse_lookup([product_id], cdse["catalogue_url"]).iloc[0].to_dict()
        log(f"  CDSE {product['size_mb']} MB")
        return cdse_download(product, dest_dir, cdse["download_url"], cdse["token_url"])
    log("  no CDSE credentials in .env: downloading the same product from ASF")
    return asf_download(product_id, dest_dir)


def download_all(cfg: Config, log: Any = print) -> dict[str, Path]:
    fc = cfg.section("benchmark")["faerch2023"]
    out = raw_dir(cfg)
    paths = {}
    for s in fc["scenes"]:
        log(f"{s['product_id']} ({s['source']})")
        if s["sensor"] == "Sentinel-1":
            paths[s["product_id"]] = download_s1(s["product_id"], out, fc["cdse"], log)
        else:
            paths[s["product_id"]] = download_s2(s["product_id"], out, fc["s2_bucket_url"],
                                                 fc["s2_files"], log)
    return paths


def s2_band(safe: Path, band: str) -> Path:
    """Path of one band (e.g. 'B08') inside a downloaded S2 SAFE."""
    hits = sorted(safe.glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2"))
    if not hits:
        raise FileNotFoundError(f"{band} not found under {safe}")
    return hits[0]


def s2_data_footprint(safe: Path, band: str = "B08", factor: int = 20) -> gpd.GeoDataFrame:
    """Polygon of the tile's valid (non-zero) pixels at `factor`x coarser resolution."""
    with rasterio.open(s2_band(safe, band)) as src:
        h, w = src.height // factor, src.width // factor
        data = src.read(1, out_shape=(h, w))
        t = src.transform * src.transform.scale(src.width / w, src.height / h)
        # Exteriors only: stray zero pixels in the decimated read would show up as holes.
        geoms = [Polygon(shape(g).exterior) for g, v in
                 shapes((data > 0).astype(np.uint8), transform=t) if v]
        return gpd.GeoDataFrame(geometry=[gpd.GeoSeries(geoms).union_all()], crs=src.crs)


def footprints(cfg: Config) -> gpd.GeoDataFrame:
    """S1 footprint (CDSE catalogue) and S2 data footprints, EPSG:4326, one row per product."""
    fc = cfg.section("benchmark")["faerch2023"]
    rows = []
    for s in fc["scenes"]:
        if s["sensor"] == "Sentinel-1":
            g = cdse_lookup([s["product_id"]], fc["cdse"]["catalogue_url"]).geometry.iloc[0]
        else:
            fp = s2_data_footprint(raw_dir(cfg) / f"{s['product_id']}.SAFE")
            g = fp.to_crs("EPSG:4326").geometry.iloc[0]
        rows.append({"sensor": s["sensor"], "product_id": s["product_id"], "geometry": g})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def footprint_map(fp: gpd.GeoDataFrame, png: Path, land: gpd.GeoDataFrame | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s1 = fp[fp.sensor == "Sentinel-1"]
    s2 = fp[fp.sensor == "Sentinel-2"]
    overlap = s1.union_all().intersection(s2.union_all())
    fig, ax = plt.subplots(figsize=(8, 8))
    if land is not None and len(land):
        land.plot(ax=ax, color="#d9d4c7", edgecolor="none")
    s1.boundary.plot(ax=ax, color="#1f77b4", linewidth=2, label="Sentinel-1 EW")
    s2.boundary.plot(ax=ax, color="#d62728", linewidth=1, label="Sentinel-2 L1C tiles (data)")
    gpd.GeoSeries([overlap], crs=fp.crs).plot(ax=ax, color="#d62728", alpha=0.15)
    for _, r in s2.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(parse_product(r.product_id)["tile"], (c.x, c.y), fontsize=8, ha="center")
    share = overlap.area / s2.union_all().area
    ax.set_title(f"Færch 2023 benchmark scenes, 2019-07-04\n"
                 f"S2 data area inside the S1 footprint: {share:.0%}")
    minx, miny, maxx, maxy = fp.total_bounds
    ax.set_xlim(minx - 0.5, maxx + 0.5)
    ax.set_ylim(miny - 0.3, maxy + 0.3)
    ax.set_aspect(1 / np.cos(np.radians((miny + maxy) / 2)))
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(loc="lower right")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png
