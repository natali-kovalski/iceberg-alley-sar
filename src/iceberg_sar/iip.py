"""International Ice Patrol (IIP) Iceberg Sightings Database (NSIDC G00807).

One row per sighting: position, UTC time, method, size class and source, per ice season
(Oct-Sep). Ends with the 2021 season. Useful sources off Newfoundland:
- aircraft / ship: METHOD `R/V` (radar + visual), `VIS`, `RAD`; SOURCE is a call sign.
- satellite: METHOD `SAT-HIGH` / `SAT-LOW` (analyst confidence). SOURCE `SN1A`/`SN1B` is
  Sentinel-1A/B; `SNL1` has Sentinel-1 pass times too (09:39 UTC descending).
Satellite sightings from the *same* Sentinel-1 pass are simultaneous with our scene, so they
need no drift correction (but come from the same sensor, so they are not independent).
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

IIP_URL = "https://noaadata.apps.nsidc.org/NOAA/G00807/IIP_{season}IcebergSeason.csv"
COLUMNS = ["season", "number", "date", "time", "lat", "lon", "method", "size", "shape", "source"]
SENTINEL1_SOURCES = {"SN1A", "SN1B", "SNL1"}


def download_season(season: int, out_dir: Path, url_pattern: str = IIP_URL) -> Path:
    """Download one season's sightings CSV (skips if present). 2019 = Oct 2018 - Sep 2019."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"IIP_{season}IcebergSeason.csv"
    if not out.is_file():
        with urlopen(url_pattern.format(season=season), timeout=120) as r:
            out.write_bytes(r.read())
    return out


def load_sightings(csv_path: Path) -> gpd.GeoDataFrame:
    """Sightings as EPSG:4326 points with a UTC `time` column.

    Header text and date formats vary between seasons, so columns are taken by position.
    """
    df = pd.read_csv(csv_path, header=0, names=COLUMNS, usecols=range(len(COLUMNS)))
    df = df.dropna(subset=["date", "time", "lat", "lon"])
    hhmm = df["time"].astype(int)
    df["time"] = (pd.to_datetime(df["date"], format="mixed")
                  + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m"))
    df = df.drop(columns="date")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")


POSITION_STEP_DEG = 0.1 / 60  # positions are stored on a 0.1 arc-minute grid


def cell_centres(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undo IIP's truncation of positions to 0.1 arc-minutes (toward the equator / Greenwich).

    Measured on 2019-04-29 (271 same-pass labels): bergs sit on average 101 m north and 67 m
    west of their IIP positions, i.e. half a grid step (93 m / 60 m at 49.5 N). Moving each
    position to the centre of its cell cuts the median label error from 135 m to 67 m.
    All satellite sightings are on the grid; 72-92% of aircraft / ship ones are, the rest carry
    finer positions and are left alone.
    """
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    half = POSITION_STEP_DEG / 2
    on_grid = (np.abs(lat / POSITION_STEP_DEG - np.round(lat / POSITION_STEP_DEG)) < 0.03) & \
              (np.abs(lon / POSITION_STEP_DEG - np.round(lon / POSITION_STEP_DEG)) < 0.03)
    return (np.where(on_grid, lat + np.sign(lat) * half, lat),
            np.where(on_grid, lon + np.sign(lon) * half, lon))


def select_sightings(
    sightings: gpd.GeoDataFrame,
    area: BaseGeometry,
    start: pd.Timestamp,
    end: pd.Timestamp,
    satellite: bool | None = None,
) -> gpd.GeoDataFrame:
    """Sightings inside `area` (EPSG:4326) between `start` and `end` (UTC, naive).

    `satellite`: True = Sentinel-1 sightings only, False = everything else, None = all.
    """
    s = sightings[(sightings.time >= start) & (sightings.time <= end)]
    s = s[s.within(area)]
    if satellite is not None:
        s = s[s.source.isin(SENTINEL1_SOURCES) == satellite]
    return s.copy()
