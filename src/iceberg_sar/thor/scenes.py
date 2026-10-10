"""Training scenes for the THOR head: Sentinel-1 frames with same-pass IIP analyst labels.

IIP satellite sightings (SOURCE SN1A/SN1B/SNL1) are picked by an analyst from the same
Sentinel-1 pass, so they are point labels with no drift between label and image. We group
them into passes, find the IW HH+HV GRD frames on ASF that cover them, and count labelled
bergs per frame inside the AOI (where the land and pack-ice masks exist).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import asf_search as asf
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from iceberg_sar.config import Config
from iceberg_sar.iip import SENTINEL1_SOURCES, download_season, load_sightings
from iceberg_sar.search import load_aoi


@dataclass(frozen=True)
class SceneParams:
    seasons: tuple[int, ...]
    holdout: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    pol: str
    pass_gap_min: int
    match_min: int
    min_sightings: int
    max_scenes: int
    credits_per_scene: int
    max_credits: int

    @classmethod
    def from_config(cls, cfg: Config) -> SceneParams:
        t = cfg.section("thor")["training"]
        return cls(
            seasons=tuple(t["seasons"]),
            holdout=tuple((pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1))
                          for a, b in t["holdout"]),
            pol=t["pol"],
            pass_gap_min=int(t["pass_gap_min"]),
            match_min=int(t["match_min"]),
            min_sightings=int(t["min_sightings"]),
            max_scenes=int(t["max_scenes"]),
            credits_per_scene=int(t["credits_per_scene"]),
            max_credits=int(t["max_credits"]),
        )


def satellite_sightings(cfg: Config, seasons: tuple[int, ...]) -> gpd.GeoDataFrame:
    """All Sentinel-1 analyst sightings for the given IIP seasons."""
    out_dir = cfg.root / cfg.section("validation")["iip_dir"]
    frames = [load_sightings(download_season(s, out_dir)) for s in seasons]
    g = pd.concat(frames, ignore_index=True)
    g = g[g["source"].isin(SENTINEL1_SOURCES)]
    return gpd.GeoDataFrame(g, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)


def in_holdout(
    times: pd.Series, holdout: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
) -> pd.Series:
    mask = pd.Series(False, index=times.index)
    for start, end in holdout:
        mask |= (times >= start) & (times < end)
    return mask


def group_passes(sightings: gpd.GeoDataFrame, gap_min: int) -> gpd.GeoDataFrame:
    """Add a `pass_id`: sightings less than `gap_min` minutes apart belong to one pass."""
    s = sightings.sort_values("time").copy()
    new_pass = s["time"].diff() > pd.Timedelta(minutes=gap_min)
    s["pass_id"] = new_pass.cumsum().astype(int)
    return s


def find_frames(
    passes: gpd.GeoDataFrame, aoi: BaseGeometry, params: SceneParams
) -> gpd.GeoDataFrame:
    """ASF IW GRD frames covering each pass, with labelled-berg counts inside the AOI.

    Only passes with at least `min_sightings` AOI sightings are searched.
    """
    rows = []
    for pid, grp in passes.groupby("pass_id"):
        inside = grp[grp.within(aoi)]
        if len(inside) < params.min_sightings:
            continue
        t0, t1 = grp["time"].min(), grp["time"].max()
        pad = pd.Timedelta(minutes=params.match_min)
        results = asf.geo_search(
            intersectsWith=inside.union_all().convex_hull.buffer(0.01).wkt,
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel="GRD_HD",
            beamMode="IW",
            start=(t0 - pad).isoformat() + "Z",
            end=(t1 + pad).isoformat() + "Z",
        )
        for r in results:
            p = r.properties
            fp = shape(r.geometry)
            n = int(inside.within(fp).sum())
            if n == 0:
                continue
            rows.append({
                "scene": p["sceneName"],
                "start_time": p["startTime"],
                "polarization": p.get("polarization"),
                "pass_id": int(pid),
                "aoi_sightings": n,
                "pass_sightings": len(grp),
                "geometry": fp,
            })
    if not rows:
        return gpd.GeoDataFrame(columns=["scene", "geometry"], geometry="geometry", crs="EPSG:4326")
    frames = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return frames.drop_duplicates("scene").sort_values("start_time").reset_index(drop=True)


def select_frames(frames: gpd.GeoDataFrame, params: SceneParams) -> gpd.GeoDataFrame:
    """Frames with the right polarization, most labelled bergs first, within the credit budget."""
    ok = frames[frames["polarization"] == params.pol]
    ok = ok.sort_values("aoi_sightings", ascending=False)
    n = min(params.max_scenes, params.max_credits // params.credits_per_scene)
    return ok.head(n).sort_values("start_time").reset_index(drop=True)


def training_scenes(cfg: Config) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """(all candidate frames, selected frames). Held-out dates are dropped before searching."""
    params = SceneParams.from_config(cfg)
    s = satellite_sightings(cfg, params.seasons)
    s = s[~in_holdout(s["time"], params.holdout)]
    passes = group_passes(s, params.pass_gap_min)
    frames = find_frames(passes, load_aoi(cfg.path("aoi")), params)
    return frames, select_frames(frames, params)


def save_frames(frames: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames.to_file(path, driver="GeoJSON")
    return path
