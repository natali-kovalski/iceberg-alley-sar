"""End-to-end run for a date range: search -> HyP3 RTC -> masks -> THOR + CFAR -> CZML.

Every step skips work that is already on disk (downloaded products, masked rasters,
detections), so a re-run only does what is missing and costs no HyP3 credits for scenes
that were processed before.

THOR (the detector) needs a CUDA GPU, TerraTorch and the trained head; the CA-CFAR baseline
needs neither. Without them THOR is skipped with one notice and the run finishes with the CFAR.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd

from iceberg_sar.config import Config

# HyP3 RTC_GAMMA credit cost per scene by pixel spacing (ASF credit table, 2025).
RTC_CREDITS = {10: 60, 20: 15, 30: 5}
DETECTORS = ("thor", "cfar")   # canonical order: THOR is the detector, the CFAR the baseline
THOR_MODULES = ("torch", "terratorch", "thor_terratorch_ext")


@dataclass(frozen=True)
class RunParams:
    pol: str | None = "HH+HV"       # None = any polarization
    min_aoi_overlap: float = 0.05   # fraction of the AOI a scene must cover
    max_scenes: int | None = 3      # cap on scenes per run (HyP3 credits); None = all
    band: str = "HV"                # detections exported to CZML
    detectors: tuple[str, ...] = DETECTORS
    thor_device: str = "cuda"


class ThorStatus(NamedTuple):
    ready: bool
    reason: str      # GPU and head when ready, otherwise why THOR is skipped
    head: Path       # trained head checkpoint
    tag: str         # detections tag, e.g. thor-linear (the viewer's THOR layer)


@dataclass(frozen=True)
class PlannedScene:
    granule: str
    start_time: str
    aoi_overlap: float
    product_dir: Path | None        # already downloaded HyP3 product, if any


def parse_detectors(value: str | Iterable[str]) -> tuple[str, ...]:
    """'thor,cfar' or a list -> known detectors in canonical order (THOR first)."""
    names = value.split(",") if isinstance(value, str) else list(value)
    chosen = {str(n).strip().lower() for n in names} - {""}
    if not chosen or chosen - set(DETECTORS):
        raise ValueError(f"detectors: expected some of {list(DETECTORS)}, got {value!r}")
    return tuple(d for d in DETECTORS if d in chosen)


def run_params(cfg: Config, detectors: str | Iterable[str] | None = None) -> RunParams:
    """`run:` settings; `detectors` (e.g. from the CLI) overrides `run.detectors`."""
    r = cfg.raw.get("run", {})
    d = RunParams()
    return RunParams(
        pol=r.get("pol", d.pol),
        min_aoi_overlap=float(r.get("min_aoi_overlap", d.min_aoi_overlap)),
        max_scenes=r.get("max_scenes", d.max_scenes),
        band=str(cfg.section("cfar").get("band", d.band)).upper(),
        detectors=parse_detectors(detectors if detectors is not None
                                  else r.get("detectors", d.detectors)),
        thor_device=str(r.get("thor_device", d.thor_device)),
    )


def thor_head(cfg: Config) -> tuple[Path, str]:
    """(head checkpoint, detections tag) for the THOR layer the viewer shows (viewer.thor_model).

    Tags follow `thor-detect`: thor-<head> for the final model, thor-<head>-<model> otherwise.
    """
    from iceberg_sar.export_czml import viewer_params

    tag = viewer_params(cfg).thor_model
    head, _, model = tag.removeprefix("thor-").partition("-")
    return cfg.path("models") / "thor" / head / (model or "final") / "head.pt", tag


def missing_modules() -> list[str]:
    return [m for m in THOR_MODULES if importlib.util.find_spec(m) is None]


def cuda_device_name() -> str | None:
    """Name of the first CUDA GPU, or None. Any CUDA error counts as no GPU: with
    CUDA_VISIBLE_DEVICES="" is_available() can still say True and the device lookup throws."""
    import torch

    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        pass
    return None


def thor_status(cfg: Config, device: str = "cuda") -> ThorStatus:
    """Whether THOR can run here; the reason says what is missing, cheapest check first."""
    head, tag = thor_head(cfg)
    if not head.is_file():
        return ThorStatus(False, f"no trained head at {head} (run `thor-fetch-head`)", head, tag)
    if missing := missing_modules():
        return ThorStatus(False, f"{', '.join(missing)} not installed", head, tag)
    if not device.startswith("cuda"):
        return ThorStatus(True, f"{device} (much slower than a GPU)", head, tag)
    if (gpu := cuda_device_name()) is None:
        return ThorStatus(False, "no CUDA GPU", head, tag)
    return ThorStatus(True, f"CUDA ({gpu}), head {head}", head, tag)


def select_scenes(scenes: gpd.GeoDataFrame, p: RunParams) -> gpd.GeoDataFrame:
    """Filter by polarization and AOI overlap; keep the `max_scenes` best-covering, by time."""
    if scenes.empty:
        return scenes
    keep = scenes[scenes["aoi_overlap"] >= p.min_aoi_overlap]
    if p.pol:
        keep = keep[keep["polarization"] == p.pol]
    keep = keep.drop_duplicates("scene")
    if p.max_scenes is not None:
        keep = keep.sort_values("aoi_overlap", ascending=False, kind="stable").head(p.max_scenes)
    return keep.sort_values("start_time").reset_index(drop=True)


def find_product(raw_dir: Path, granule: str, resolution: int) -> Path | None:
    """Unzipped HyP3 RTC product for a granule, matched on platform, beam mode and start time.

    Granule `S1A_IW_GRDH_1SDH_20250508T094925_...`
    -> product `S1A_IW_20250508T094925_DHP_RTC10_...`.
    """
    parts = granule.split("_")
    start = next((s for s in parts if len(s) == 15 and s[8] == "T"), None)
    if start is None:
        raise ValueError(f"No start time in granule name '{granule}'")
    pattern = f"{parts[0]}_{parts[1]}_{start}_*_RTC{resolution}_*"
    matches = sorted(d for d in Path(raw_dir).glob(pattern) if d.is_dir())
    return matches[-1] if matches else None


def plan(scenes: gpd.GeoDataFrame, raw_dir: Path, resolution: int) -> list[PlannedScene]:
    return [
        PlannedScene(r.scene, str(r.start_time), float(r.aoi_overlap),
                     find_product(raw_dir, r.scene, resolution))
        for r in scenes.itertuples()
    ]


def order_and_download(granules: list[str], cfg: Config, log: Callable[[str], None]) -> None:
    """Submit RTC jobs (reusing existing ones), wait for them, download and unzip."""
    from iceberg_sar.hyp3 import download_jobs, get_client, submit_rtc

    client = get_client()
    log(f"HyP3 credits remaining: {client.check_credits()}")
    batches = [submit_rtc(client, g, cfg) for g in granules]
    batch = sum(batches[1:], batches[0])
    log(f"waiting for {len(batch)} HyP3 job(s); this often takes 30-60 min")
    batch = client.watch(batch)
    for d in download_jobs(batch, cfg.path("raw")):
        log(f"downloaded  {d.name}")


def process_product(product_dir: Path, land: gpd.GeoDataFrame, cfg: Config, band: str,
                    force: bool, log: Callable[[str], None], cfar: bool = True) -> Path | None:
    """Preprocess and run the CFAR on one product unless its outputs exist.

    Returns the CFAR detections, or None with `cfar` False (preprocessing only, for THOR).
    """
    from iceberg_sar.detections import detect_product
    from iceberg_sar.preprocess import preprocess_product

    name = product_dir.name
    if force or not (cfg.path("interim") / name / f"{name}_preprocess.json").is_file():
        log(f"preprocess  {name}")
        preprocess_product(product_dir, land, cfg)
    if not cfar:
        return None
    detections = cfg.path("outputs") / "detections" / f"{name}_{band}_detections.geojson"
    if force or not detections.is_file():
        log(f"detect      {name}")
        detections = detect_product(product_dir, cfg, band=band)["detections"]
    return detections


def detect_thor(product_dir: Path, cfg: Config, status: ThorStatus, device: str, force: bool,
                log: Callable[[str], None]) -> Path:
    """THOR detections for one preprocessed product unless they exist."""
    from iceberg_sar.thor.infer import detect_scene

    name = product_dir.name
    out = cfg.path("outputs") / "detections" / f"{name}_{status.tag}_detections.geojson"
    if force or not out.is_file():
        log(f"thor        {name}")
        out = detect_scene(product_dir, cfg, status.head, device, log=log,
                           tag=status.tag)["detections"]
    return out


def run_pipeline(cfg: Config, start: date, end: date, *, dry_run: bool = False,
                 force: bool = False, log: Callable[[str], None] = print,
                 detectors: str | Iterable[str] | None = None) -> Path | None:
    """Run the whole pipeline for scenes acquired between `start` and `end` (inclusive).

    `detectors` overrides `run.detectors`. Returns the CZML path, or None for a dry run.
    """
    from iceberg_sar.export_czml import export_czml
    from iceberg_sar.landmask import build_land_mask, load_land
    from iceberg_sar.search import search_scenes

    p = run_params(cfg, detectors)
    res = int(cfg.section("hyp3")["resolution"])
    # The end date is inclusive: search up to the end of that day, not its first second.
    found = search_scenes(cfg, start=f"{start}T00:00:00Z", end=f"{end}T23:59:59Z")
    scenes = plan(select_scenes(found, p), cfg.path("raw"), res)
    if not scenes:
        raise RuntimeError(f"No {p.pol or ''} scenes over the AOI between {start} and {end}")
    for s in scenes:
        state = "downloaded" if s.product_dir else "to order"
        log(f"{s.start_time[:19]}  {s.granule}  aoi={s.aoi_overlap:.0%}  {state}")
    missing = [s.granule for s in scenes if s.product_dir is None]
    log(f"{len(scenes)} of {len(found)} scenes selected; {len(missing)} to order "
        f"(~{len(missing) * RTC_CREDITS.get(res, 0)} HyP3 credits at {res} m)")
    thor = thor_status(cfg, p.thor_device) if "thor" in p.detectors else None
    if thor is not None:
        log(f"THOR        will run: {thor.reason}" if thor.ready else
            f"THOR        skipped: {thor.reason}"
            + ("; the CFAR baseline still runs" if "cfar" in p.detectors else ""))
    if "cfar" not in p.detectors:
        log("note        the viewer lists scenes by their CFAR detections; scenes without "
            "them are left out of the CZML")
    if dry_run:
        return None

    if missing:
        order_and_download(missing, cfg, log)
        scenes = plan(select_scenes(found, p), cfg.path("raw"), res)
    land = load_land(build_land_mask(cfg))
    for s in scenes:
        if s.product_dir is None:
            log(f"skipped     {s.granule} (HyP3 job did not succeed)")
            continue
        process_product(s.product_dir, land, cfg, p.band, force, log, cfar="cfar" in p.detectors)
        if thor is not None and thor.ready:
            detect_thor(s.product_dir, cfg, thor, p.thor_device, force, log)

    # The viewer shows every processed scene, not just this run's date range.
    out, packets = export_czml(cfg, p.band)
    n = sum(1 for k in packets if k["id"].startswith("scene/"))
    log(f"czml        {n} scene(s) -> {out}")
    return out
