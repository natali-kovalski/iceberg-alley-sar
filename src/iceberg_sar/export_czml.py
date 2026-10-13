"""Detections, scene footprints and pack ice -> one time-dynamic CZML document for Cesium.

Each scene is shown from its acquisition time until the next scene starts, so scrubbing
the Cesium timeline steps through the dates. Detections keep all their attributes as CZML
`properties`; `near_ice` flags candidates close to mapped pack ice (likely floes), which
the viewer greys out or hides instead of deleting them. Static targets (seen at the same
spot on several dates, see `static_targets.py`) are left out.

THOR detections (Milestone 5), where a scene has them, are a second layer of `thor/`
entities: open rings above the score threshold. Both detectors are drawn on one footing: one
colour per detector, size from the grown target extent (`structure_m`, measured the same way
for both). A THOR ring is drawn just outside a CFAR dot of the same extent, so a berg both
detectors found reads as a dot inside a ring.

One Sentinel-1 frame covers only ~15-40% of the AOI. Frames of one pass (same day) are grouped
into a `pass/` entity with the AOI share they observe, and the AOI outside them is drawn as an
`unobserved/` polygon, so an unimaged sea never reads as an empty one.

THOR training frames (Milestone 5b) carry their split (`train`, or `val` for the pack-ice set
that was never trained on) and their same-pass IIP analyst labels as `truth/` crosses, at the
HH peak they were snapped to: the positions the THOR head was trained on.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely import make_valid, union_all
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from iceberg_sar.config import Config
from iceberg_sar.preprocess import find_rtc_bands
from iceberg_sar.static_targets import StaticParams, find_static, is_static, static_params

NEAR_ICE_RGBA = [150, 150, 150, 200]
CFAR_RGBA = [251, 162, 56, 255]  # orange filled dot
FOOTPRINT_RGBA = [255, 255, 0, 255]   # bright yellow: stands out on imagery and water
ICE_RGBA = [120, 200, 255, 90]
THOR_RGBA = [0, 229, 255, 255]   # cyan ring: distinct from the CFAR orange, readable on dark water
THOR_RING_PX = 2                 # ring width; the ring sits just outside a same-size CFAR dot
AOI_RGBA = [255, 255, 255, 220]
UNOBSERVED_RGBA = [8, 12, 20, 90]   # lightly dims the basemap where no frame of the pass looked
SEAM_CLOSE_M = 500.0               # closes slivers between frames of one pass
TRUTH_HEX = "#1e4fd8"              # dark blue X: a label, not a detection (dots and rings are)
TRUTH_PX = 15
_CROSS = "M2 2L13 13M13 2L2 13"
TRUTH_IMAGE = "data:image/svg+xml;base64," + base64.b64encode(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{TRUTH_PX}" height="{TRUTH_PX}">'
    f'<path d="{_CROSS}" stroke="{TRUTH_HEX}" stroke-width="2.5" stroke-linecap="round"/></svg>'
    .encode()).decode()
# Columns of a chip scene's labels.csv kept as CZML properties.
TRUTH_PROPS = ("season", "number", "time", "method", "size", "shape", "source", "snapped",
               "snap_m")


@dataclass(frozen=True)
class ViewerParams:
    near_ice_km: float = 5.0
    last_interval_days: float = 7.0   # max display time per scene (S1A+S1C revisit is 6 d)
    # Size classes by grown target extent, shared by CFAR and THOR: small / medium / large.
    size_breaks_m: tuple[float, float] = (180.0, 270.0)   # pooled terciles of both detectors
    size_px: tuple[float, float, float] = (6.0, 13.0, 22.0)
    simplify_m: float = 100.0         # polygon simplification before reprojecting
    thor_model: str = "thor-linear"   # THOR detection tag shown as its own layer
    thor_score: float = 0.35          # THOR operating point (thor.compare.view_score)
    exclude: tuple[str, ...] = ()     # products left out of the viewer


@dataclass(frozen=True)
class Scene:
    name: str
    start: datetime
    detections: Path
    footprint: Path | None
    seaice: Path | None
    thor: Path | None = None
    split: str | None = None     # THOR chips split ("train" / "val"); None = not a training frame
    labels: Path | None = None   # THOR chips labels.csv (snapped same-pass IIP labels)
    raster: Path | None = None   # HH band: pixel grid of the labels' row / col


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")


def scene_intervals(starts: list[datetime], max_days: float) -> list[tuple[datetime, datetime]]:
    """Display interval per scene (`starts` sorted): until the next day's pass, at most `max_days`.

    Frames of one pass (same day, seconds apart) share the pass's first start and end, so
    jumping to a pass shows all its frames at once. The cap keeps a scene from lingering
    across the months or years between seasons.
    """
    out = []
    for t in starts:
        t0 = next(s for s in starts if s.date() == t.date())
        cap = t0 + timedelta(days=max_days)
        nxt = next((s for s in starts if s.date() > t.date()), None)
        out.append((t0, min(nxt, cap) if nxt else cap))
    return out


def pass_coverage(aoi: BaseGeometry, footprints: list[BaseGeometry]
                  ) -> tuple[BaseGeometry, float, float]:
    """(AOI part outside all `footprints`, observed km², observed fraction of the AOI).

    All geometries in one metric CRS.
    """
    aoi = make_valid(aoi)   # HyP3 footprints can self-touch; overlays then throw
    union = make_valid(union_all([make_valid(f) for f in footprints]))
    # Adjacent frames of a pass meet along pixel staircases; close the seam's slivers.
    union = union.buffer(SEAM_CLOSE_M, join_style="mitre").buffer(-SEAM_CLOSE_M,
                                                                  join_style="mitre")
    observed = aoi.intersection(union)
    return aoi.difference(observed), observed.area / 1e6, observed.area / aoi.area


def _footprint(path: Path) -> gpd.GeoDataFrame:
    """Largest valid-data polygon of a HyP3 `_shape.shp` (the rest are specks), native UTM."""
    fp = gpd.read_file(path)
    return fp.iloc[[int(fp.area.argmax())]]


def near_ice(distance_km: np.ndarray, threshold_km: float) -> np.ndarray:
    """True where a detection is closer than `threshold_km` to pack ice (NaN = no ice mapped)."""
    d = np.asarray(distance_km, dtype=float)
    return np.nan_to_num(d, nan=np.inf) < threshold_km


def point_px(structure_m: float, p: ViewerParams) -> float:
    """Marker diameter from the size class of the grown target extent; same for CFAR and THOR.

    Three discrete classes instead of a continuous ramp: most targets fall within ~100-330 m,
    where a ramp gives sizes too close to tell apart. Missing extent counts as small.
    """
    if not math.isfinite(structure_m):
        return p.size_px[0]
    return p.size_px[int(np.searchsorted(p.size_breaks_m, structure_m, side="right"))]


def _ring(coords: Any) -> list[float]:
    out: list[float] = []
    for x, y in coords:
        out += [round(x, 5), round(y, 5), 0.0]
    return out


def _polygons(geom: BaseGeometry) -> list[Polygon]:
    """Polygon parts only: `make_valid` can leave stray lines and points in a collection."""
    if geom.is_empty:
        return []
    if hasattr(geom, "geoms"):
        return [q for g in geom.geoms for q in _polygons(g)]
    return [geom] if isinstance(geom, Polygon) else []


def _to_wgs84(gdf: gpd.GeoDataFrame, simplify_m: float) -> gpd.GeoDataFrame:
    if gdf.empty:   # e.g. no pack ice in summer: bounds are NaN, so no UTM zone to estimate
        return gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    metric =gdf if gdf.crs and gdf.crs.is_projected else gdf.to_crs(gdf.estimate_utm_crs())
    metric = metric.assign(geometry=metric.geometry.simplify(simplify_m))
    return metric.to_crs("EPSG:4326")


def _num(value: Any, fmt: str = ".1f") -> str:
    """Formatted number, or an en dash for missing values (None / NaN)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return format(value, fmt)


def _distance_text(km: Any, flag: bool) -> str:
    if km is None or not math.isfinite(km):
        return "no pack ice mapped"
    return f"{km:.1f} km" + (" (near ice: likely floe)" if flag else "")


def _description(props: dict[str, Any]) -> str:
    g = props.get
    rows = [
        ("Scene", props["scene_id"]),
        ("Time (UTC)", props["timestamp"]),
        ("Lat / lon", f"{_num(g('lat'), '.4f')}, {_num(g('lon'), '.4f')}"),
        ("Area", f"{_num(g('area_m2'), '.0f')} m² ({_num(g('area_px'), 'd')} px)"),
        ("Structure", f"{_num(g('structure_m'), '.0f')} m"),
        ("Peak HV / HH", f"{_num(g('peak_db_HV'))} / {_num(g('peak_db_HH'))} dB"),
        ("Mean HV / HH", f"{_num(g('mean_db_HV'))} / {_num(g('mean_db_HH'))} dB"),
        ("Contrast", f"{_num(g('contrast_db'))} dB over {_num(g('background_db'))} dB"),
        ("Incidence", f"{_num(g('incidence_deg'))}°"),
        ("Distance to ice", _distance_text(g("distance_to_ice_km"), props["near_ice"])),
    ]
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="cesium-infoBox-defaultTable"><tbody>{body}</tbody></table>'


def _clean(value: Any) -> Any:
    """JSON-safe scalar: NaN -> None, numpy -> Python."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def detection_packets(gdf: gpd.GeoDataFrame, availability: str, p: ViewerParams
                      ) -> list[dict[str, Any]]:
    if gdf.empty:
        return []
    flags = near_ice(gdf["distance_to_ice_km"].to_numpy(), p.near_ice_km)
    packets = []
    for (_, row), flag in zip(gdf.iterrows(), flags, strict=True):
        raw = row.drop(labels="geometry").to_dict()
        raw["timestamp"] = str(row["timestamp"])[:19].replace(" ", "T") + "Z"
        raw["near_ice"] = bool(flag)
        # Cesium's CZML loader throws on null property values, so missing ones are left out.
        props = {k: c for k, v in raw.items() if (c := _clean(v)) is not None}
        rgba = NEAR_ICE_RGBA if flag else CFAR_RGBA
        packets.append({
            "id": f"det/{row['id']}",
            "name": f"CA-CFAR {row['id']}",
            "parent": f"scene/{row['scene_id']}",
            "availability": availability,
            "description": _description(raw),
            "position": {"cartographicDegrees": [float(row["lon"]), float(row["lat"]), 0.0]},
            "point": {
                "pixelSize": point_px(float(row["structure_m"]), p),
                "color": {"rgba": rgba},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 1,
            },
            "properties": props,
        })
    return packets


def thor_packets(gdf: gpd.GeoDataFrame, availability: str, p: ViewerParams
                 ) -> list[dict[str, Any]]:
    """THOR detections as open rings sized like the CFAR dots; `near_ice` flagged the same way."""
    if gdf.empty:
        return []
    flags = near_ice(gdf["distance_to_ice_km"].to_numpy(), p.near_ice_km)
    packets = []
    for (_, row), flag in zip(gdf.iterrows(), flags, strict=True):
        raw = row.drop(labels="geometry").to_dict()
        raw["timestamp"] = str(row["timestamp"])[:19].replace(" ", "T") + "Z"
        raw["near_ice"] = bool(flag)
        props = {k: c for k, v in raw.items() if (c := _clean(v)) is not None}
        rgba = NEAR_ICE_RGBA if flag else THOR_RGBA
        packets.append({
            "id": f"thor/{row['id']}",
            "name": f"THOR {row['id']} (score {float(row['score']):.2f})",
            "parent": f"scene/{row['scene_id']}",
            "availability": availability,
            "description": _description(raw),
            "position": {"cartographicDegrees": [float(row["lon"]), float(row["lat"]), 0.0]},
            "point": {
                # Inner diameter = the CFAR dot (plus its 1 px outline) for the same extent.
                "pixelSize": point_px(float(row["structure_m"]), p) + 2 + THOR_RING_PX,
                "color": {"rgba": [0, 0, 0, 1]},   # near-transparent fill keeps it clickable
                "outlineColor": {"rgba": rgba},
                "outlineWidth": THOR_RING_PX,
            },
            "properties": props,
        })
    return packets


def read_truth(labels: Path, raster: Path) -> pd.DataFrame:
    """Visible IIP labels of a THOR training frame in lon/lat, at their snapped pixel.

    Labels outside the AOI or within the land buffer (`visible` False) were never targets.
    """
    df = pd.read_csv(labels)
    df = df[df["visible"].astype(bool)].reset_index(drop=True)
    if df.empty:
        return df.assign(lon=[], lat=[])
    with rasterio.open(raster) as src:
        x, y = rasterio.transform.xy(src.transform, df["row"].to_numpy(), df["col"].to_numpy())
        crs = src.crs
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    return df.assign(lon=np.asarray(lon), lat=np.asarray(lat))


def _truth_description(props: dict[str, Any], scene_id: str) -> str:
    g = props.get
    snap = (f"to the HH peak, {_num(g('snap_m'), '.0f')} m" if g("snapped")
            else "no (0.1' cell centre)")
    rows = [
        ("Scene", scene_id),
        ("IIP sighting", f"{g('season')} #{g('number')}"),
        ("Time (UTC)", g("time")),
        ("Source", f"{g('source')} / {g('method')}"),
        ("Size / shape", f"{g('size')} / {g('shape')}"),
        ("Lat / lon", f"{_num(g('lat'), '.4f')}, {_num(g('lon'), '.4f')}"),
        ("Snapped", snap),
    ]
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="cesium-infoBox-defaultTable"><tbody>{body}</tbody></table>'


def truth_packets(truth: pd.DataFrame, scene_id: str, availability: str
                  ) -> list[dict[str, Any]]:
    """IIP labels as green crosses, drawn in front of the detector dots and rings."""
    packets = []
    for i, row in truth.iterrows():
        raw = {k: row[k] for k in (*TRUTH_PROPS, "lat", "lon") if k in row}
        props = {k: c for k, v in raw.items() if (c := _clean(v)) is not None}
        packets.append({
            "id": f"truth/{scene_id}/{i}",
            "name": f"IIP label {row['season']} #{row['number']}",
            "parent": f"scene/{scene_id}",
            "availability": availability,
            "description": _truth_description(props, scene_id),
            "position": {"cartographicDegrees": [float(row["lon"]), float(row["lat"]), 0.0]},
            "billboard": {
                "image": TRUTH_IMAGE,
                "width": TRUTH_PX,
                "height": TRUTH_PX,
                # Toward the camera, so a cross is never hidden under a same-place dot or ring.
                "eyeOffset": {"cartesian": [0.0, 0.0, -500.0]},
            },
            "properties": props,
        })
    return packets


def polygon_packets(geom: BaseGeometry, id_prefix: str, availability: str,
                    fill_rgba: list[int] | None, outline_rgba: list[int]
                    ) -> list[dict[str, Any]]:
    packets = []
    for i, poly in enumerate(_polygons(geom)):
        polygon: dict[str, Any] = {
            "positions": {"cartographicDegrees": _ring(poly.exterior.coords)},
            "height": 0,
            "fill": fill_rgba is not None,
            "outline": True,
            "outlineColor": {"rgba": outline_rgba},
        }
        if fill_rgba is not None:
            polygon["material"] = {"solidColor": {"color": {"rgba": fill_rgba}}}
        if poly.interiors:
            polygon["holes"] = {"cartographicDegrees": [_ring(r.coords) for r in poly.interiors]}
        packet: dict[str, Any] = {"id": f"{id_prefix}/{i}", "polygon": polygon}
        if availability:   # "" = always shown
            packet["availability"] = availability
        packets.append(packet)
    return packets


def find_scenes(cfg: Config, band: str, start: date | None = None,
                end: date | None = None, thor_model: str = "thor-linear",
                chips_dir: Path | None = None) -> list[Scene]:
    """Scenes with a detections GeoJSON for `band` at the configured HyP3 resolution, by time.

    Filtering on the product's RTC resolution (e.g. `_RTC10_`) keeps a date from showing
    twice when it was processed at both 20 m and 10 m. `start` / `end` (inclusive) keep only
    scenes acquired in that date range. With `chips_dir` (THOR chips), training frames get
    their split and labels.
    """
    det_dir = cfg.path("outputs") / "detections"
    rtc = f"_RTC{int(cfg.section('hyp3')['resolution'])}_"
    scenes = []
    for f in det_dir.glob(f"*{rtc}*_{band}_detections.geojson"):
        name = f.name.removesuffix(f"_{band}_detections.geojson")
        summary = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        day = _parse(summary["timestamp"]).date()
        if (start and day < start) or (end and day > end):
            continue
        shp = cfg.path("raw") / name / f"{name}_shape.shp"
        ice = cfg.path("outputs") / "seaice" / f"{name}_seaice.geojson"
        thor = det_dir / f"{name}_{thor_model}_detections.geojson"
        split = labels = raster = None
        meta = chips_dir / name / "scene.json" if chips_dir else None
        if meta and meta.is_file():
            split = json.loads(meta.read_text(encoding="utf-8"))["split"]
            if meta.with_name("labels.csv").is_file() and (cfg.path("raw") / name).is_dir():
                labels = meta.with_name("labels.csv")
                raster = find_rtc_bands(cfg.path("raw") / name)["HH"]
        scenes.append(Scene(name, _parse(summary["timestamp"]), f,
                            shp if shp.is_file() else None, ice if ice.is_file() else None,
                            thor if thor.is_file() else None, split, labels, raster))
    return sorted(scenes, key=lambda s: s.start)


def build_czml(scenes: list[Scene], p: ViewerParams, static: gpd.GeoDataFrame | None = None,
               sp: StaticParams | None = None, aoi: gpd.GeoDataFrame | None = None
               ) -> list[dict[str, Any]]:
    if not scenes:
        raise ValueError("No scenes with detections; run `detect` first")
    intervals = scene_intervals([s.start for s in scenes], p.last_interval_days)
    t0, t1 = intervals[0][0], intervals[-1][1]
    doc: dict[str, Any] = {
        "id": "document",
        "name": "Iceberg Alley SAR detections",
        "version": "1.0",
        "clock": {
            "interval": f"{_iso(t0)}/{_iso(t1)}",
            "currentTime": _iso(t0),
            "multiplier": 3600,
            "range": "CLAMPED",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }
    legend = {
        "id": "legend",
        "properties": {
            "near_ice_km": p.near_ice_km,
            # CZML reads a bare list as time samples ([t, v, ...] -> v); "array" keeps it a list.
            "size_breaks_m": {"array": list(p.size_breaks_m)},
            "size_px": {"array": list(p.size_px)},
        },
    }
    packets: list[dict[str, Any]] = [doc, legend]
    metric = aoi.estimate_utm_crs() if aoi is not None else None
    if aoi is not None:   # no availability: the frame of reference stays put on every date
        packets += polygon_packets(_to_wgs84(aoi, p.simplify_m).union_all(), "aoi", "", None,
                                   AOI_RGBA)
    passes: dict[date, dict[str, Any]] = {}
    for i, (scene, (start, end)) in enumerate(zip(scenes, intervals, strict=True)):
        # CZML intervals include their end; stop 1 s early so two scenes never show at once.
        shown_until = end - timedelta(seconds=1) if i < len(scenes) - 1 else end
        avail = f"{_iso(start)}/{_iso(shown_until)}"
        gdf = gpd.read_file(scene.detections)
        n_static = 0
        if static is not None and sp is not None:
            drop = is_static(gdf, static, sp)
            gdf, n_static = gdf[~drop], int(drop.sum())
        flags = (near_ice(gdf["distance_to_ice_km"].to_numpy(), p.near_ice_km) if len(gdf)
                 else np.zeros(0, dtype=bool))
        thor = gpd.GeoDataFrame()
        if scene.thor:
            thor = gpd.read_file(scene.thor)
            if len(thor):
                thor = thor[thor["score"] >= p.thor_score]
            if len(thor) and static is not None and sp is not None:
                thor = thor[~is_static(thor, static, sp)]
        ice_km2 = 0.0
        if scene.seaice:
            ice = gpd.read_file(scene.seaice)
            ice_km2 = float(ice["area_km2"].sum()) if "area_km2" in ice else 0.0
        truth = (read_truth(scene.labels, scene.raster) if scene.labels and scene.raster
                 else None)
        packets.append({
            "id": f"scene/{scene.name}",
            "name": scene.name,
            "availability": avail,
            "properties": {
                "scene_id": scene.name,
                "platform": scene.name[:3],
                "start": _iso(start),
                "end": _iso(end),
                "n_detections": len(gdf),
                "n_near_ice": int(np.sum(flags)),
                "n_open_water": int(len(gdf) - np.sum(flags)),
                "ice_area_km2": round(ice_km2, 1),
                "n_static_removed": n_static,
                **({"n_thor": int((~near_ice(thor["distance_to_ice_km"].to_numpy(),
                                              p.near_ice_km)).sum()) if len(thor) else 0,
                    "thor_score": p.thor_score} if scene.thor else {}),
                **({"thor_split": scene.split} if scene.split else {}),
                **({"n_truth": len(truth)} if truth is not None else {}),
            },
        })
        pas = passes.setdefault(scene.start.date(), {"start": start, "until": shown_until,
                                                     "frames": [], "footprints": []})
        pas["frames"].append(scene.name)
        if scene.footprint:
            fp = _footprint(scene.footprint)
            if metric is not None:
                # Simplified: raw HyP3 outlines are pixel staircases, slow to buffer.
                pas["footprints"].append(fp.to_crs(metric).geometry.iloc[0]
                                         .simplify(p.simplify_m))
            packets += polygon_packets(_to_wgs84(fp, p.simplify_m).geometry.iloc[0],
                                       f"footprint/{scene.name}", avail, None, FOOTPRINT_RGBA)
        if scene.seaice:
            ice_geom = _to_wgs84(gpd.read_file(scene.seaice), p.simplify_m).union_all()
            packets += polygon_packets(ice_geom, f"seaice/{scene.name}", avail, ICE_RGBA,
                                       ICE_RGBA[:3] + [200])
        packets += detection_packets(gdf, avail, p)
        packets += thor_packets(thor, avail, p)
        if truth is not None:
            packets += truth_packets(truth, scene.name, avail)
    for day, pas in passes.items():
        avail = f"{_iso(pas['start'])}/{_iso(pas['until'])}"
        props: dict[str, Any] = {"date": day.isoformat(), "frames": " ".join(pas["frames"])}
        # Coverage only when every frame has a footprint; a missing one would read as unobserved.
        if aoi is not None and len(pas["footprints"]) == len(pas["frames"]):
            gap, km2, frac = pass_coverage(aoi.to_crs(metric).union_all(), pas["footprints"])
            props |= {"observed_km2": round(km2), "aoi_coverage_pct": round(100 * frac, 1)}
            gap_wgs = _to_wgs84(gpd.GeoDataFrame(geometry=[gap], crs=metric), p.simplify_m)
            # simplify can leave the gap invalid
            packets += polygon_packets(make_valid(gap_wgs.geometry.iloc[0]),
                                       f"unobserved/{day}", avail, UNOBSERVED_RGBA,
                                       UNOBSERVED_RGBA[:3] + [0])
        packets.append({"id": f"pass/{day}", "name": f"Pass {day}", "availability": avail,
                        "properties": props})
    return packets


def viewer_params(cfg: Config) -> ViewerParams:
    v = cfg.raw.get("viewer", {})
    d = ViewerParams()
    return ViewerParams(
        near_ice_km=float(v.get("near_ice_km", d.near_ice_km)),
        last_interval_days=float(v.get("last_interval_days", d.last_interval_days)),
        size_breaks_m=tuple(v.get("size_breaks_m", d.size_breaks_m)),  # type: ignore[arg-type]
        size_px=tuple(v.get("size_px", d.size_px)),  # type: ignore[arg-type]
        simplify_m=float(v.get("simplify_m", d.simplify_m)),
        thor_model=str(v.get("thor_model", d.thor_model)),
        thor_score=float(cfg.raw.get("thor", {}).get("compare", {}).get("view_score",
                                                                         d.thor_score)),
        exclude=tuple(v.get("exclude") or ()),
    )


def build_static(cfg: Config, band: str) -> tuple[gpd.GeoDataFrame, StaticParams]:
    """Static targets from every processed scene; written next to the detections."""
    sp = static_params(cfg)
    dets = [(s.start.date(), gpd.read_file(s.detections)) for s in find_scenes(cfg, band)]
    static = find_static(dets, sp)
    out = cfg.path("outputs") / f"static_targets_{band}.geojson"
    static.to_file(out, driver="GeoJSON")
    return static, sp


def export_czml(cfg: Config, band: str = "HV", start: date | None = None,
                end: date | None = None) -> tuple[Path, list[dict[str, Any]]]:
    static, sp = build_static(cfg, band)
    p = viewer_params(cfg)
    aoi_path = cfg.path("aoi") if "aoi" in cfg.section("paths") else None
    aoi = gpd.read_file(aoi_path) if aoi_path and aoi_path.is_file() else None
    thor = cfg.raw.get("thor", {}).get("training", {})
    chips_dir = cfg.root / thor["dir"] / "chips" if "dir" in thor else None
    scenes = [s for s in find_scenes(cfg, band, start, end, p.thor_model, chips_dir)
              if s.name not in p.exclude]
    packets = build_czml(scenes, p, static, sp, aoi)
    out = cfg.path("czml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(packets, separators=(",", ":")), encoding="utf-8")
    return out, packets
