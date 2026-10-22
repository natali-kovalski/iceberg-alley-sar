"""Find Sentinel-1 GRD scenes over the AOI with asf_search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asf_search as asf
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from iceberg_sar.config import Config

SCENE_COLUMNS = [
    "scene",
    "start_time",
    "beam_mode",
    "polarization",
    "processing_level",
    "flight_direction",
    "path",
    "frame",
    "aoi_overlap",
]


def load_aoi(path: str | Path) -> BaseGeometry:
    """AOI as a single EPSG:4326 geometry (union of all features)."""
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    return gdf.geometry.union_all()


def _record(product: asf.ASFProduct, aoi: BaseGeometry) -> dict[str, Any]:
    p = product.properties
    footprint = shape(product.geometry)
    overlap = footprint.intersection(aoi).area / aoi.area if aoi.area else 0.0
    return {
        "scene": p["sceneName"],
        "start_time": p["startTime"],
        "beam_mode": p.get("beamModeType"),
        "polarization": p.get("polarization"),
        "processing_level": p.get("processingLevel"),
        "flight_direction": p.get("flightDirection"),
        "path": p.get("pathNumber"),
        "frame": p.get("frameNumber"),
        "aoi_overlap": round(overlap, 3),
        "geometry": footprint,
    }


def search_scenes(
    cfg: Config, start: str | None = None, end: str | None = None
) -> gpd.GeoDataFrame:
    """Search ASF for GRD scenes intersecting the AOI; dates default to the config's range."""
    s = cfg.section("search")
    aoi = load_aoi(cfg.path("aoi"))
    results = asf.geo_search(
        intersectsWith=aoi.wkt,
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=s["processing_levels"],
        beamMode=s["beam_modes"],
        start=start or s["start_date"],
        end=end or s["end_date"],
        maxResults=s.get("max_results"),
    )
    records = [_record(r, aoi) for r in results]
    if not records:
        return gpd.GeoDataFrame(columns=[*SCENE_COLUMNS, "geometry"], geometry="geometry",
                                crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf.sort_values("start_time").reset_index(drop=True)


def save_scenes(gdf: gpd.GeoDataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


# --- Copernicus Data Space (CDSE): look up and download products by name -------------------
# The OData catalogue is public; downloads need a CDSE account (CDSE_USERNAME / CDSE_PASSWORD).

def cdse_lookup(product_ids: list[str], catalogue_url: str) -> gpd.GeoDataFrame:
    """One row per product name (with or without .SAFE): CDSE id, size, MD5, online, footprint."""
    import requests

    rows = []
    for pid in product_ids:
        name = pid if pid.endswith(".SAFE") else f"{pid}.SAFE"
        r = requests.get(catalogue_url, params={"$filter": f"Name eq '{name}'"}, timeout=60)
        r.raise_for_status()
        found = r.json()["value"]
        if not found:
            raise LookupError(f"Not in the CDSE catalogue: {name}")
        p = found[0]
        md5 = next((c["Value"] for c in p.get("Checksum", []) if c["Algorithm"] == "MD5"), None)
        rows.append({"product_id": pid.removesuffix(".SAFE"), "cdse_id": p["Id"],
                     "size_mb": round(p["ContentLength"] / 2**20), "online": p["Online"],
                     "md5": md5, "start_time": p["ContentDate"]["Start"],
                     "geometry": shape(p["GeoFootprint"])})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def cdse_token(token_url: str) -> str:
    import os

    import requests
    from dotenv import load_dotenv

    load_dotenv()
    user, password = os.environ.get("CDSE_USERNAME"), os.environ.get("CDSE_PASSWORD")
    if not user or not password:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD in .env (dataspace.copernicus.eu)")
    r = requests.post(token_url, timeout=60, data={
        "client_id": "cdse-public", "grant_type": "password", "username": user, "password": password})
    r.raise_for_status()
    return r.json()["access_token"]


def stream_download(url: str, dest: Path, md5: str | None, headers: dict[str, str] | None = None,
                    follow_auth: bool = False, session: Any = None) -> Path:
    """Download to `dest` via a .part file, checking the MD5 on the fly; skip if already there.

    `md5` may be hex or base64 (Google Cloud Storage). `follow_auth` re-sends `headers` on
    redirects (requests drops Authorization across hosts, and CDSE redirects to its download
    nodes). `session` is an already authenticated requests session (e.g. ASF's).
    """
    import base64
    import hashlib

    import requests

    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    r = s.get(url, headers=headers, stream=True, timeout=120, allow_redirects=not follow_auth)
    while follow_auth and r.is_redirect:
        r = s.get(r.headers["Location"], headers=headers, stream=True, timeout=120,
                  allow_redirects=False)
    r.raise_for_status()
    h = hashlib.md5()
    tmp = dest.with_name(dest.name + ".part")
    with r, tmp.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if md5 and len(md5) != 32:  # base64 digest -> hex
        md5 = base64.b64decode(md5).hex()
    if md5 and h.hexdigest().lower() != md5.lower():
        tmp.unlink()
        raise OSError(f"MD5 mismatch for {dest.name}: got {h.hexdigest()}, expected {md5}")
    tmp.replace(dest)
    return dest


def cdse_download(product: dict[str, Any], dest_dir: Path, download_url: str,
                  token_url: str) -> Path:
    """Zipped SAFE from CDSE, MD5-checked against the catalogue."""
    dest = dest_dir / f"{product['product_id']}.zip"
    if dest.is_file():
        return dest
    headers = {"Authorization": f"Bearer {cdse_token(token_url)}"}
    return stream_download(download_url.format(id=product["cdse_id"]), dest, product["md5"],
                           headers, follow_auth=True)


def asf_download(product_id: str, dest_dir: Path) -> Path:
    """The same Sentinel-1 product from ASF (Earthdata login), MD5-checked against ASF's value."""
    import os

    from dotenv import load_dotenv

    dest = dest_dir / f"{product_id}.zip"
    if dest.is_file():
        return dest
    load_dotenv()
    session = asf.ASFSession().auth_with_creds(os.environ["EARTHDATA_USERNAME"],
                                              os.environ["EARTHDATA_PASSWORD"])
    grd = [p for p in asf.granule_search([product_id])
           if p.properties["processingLevel"].startswith("GRD")]
    if not grd:
        raise LookupError(f"Not on ASF: {product_id}")
    p = grd[0].properties
    return stream_download(p["url"], dest, p["md5sum"], session=session)
