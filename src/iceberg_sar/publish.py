"""Publish derived viewer data (CZML, recall figures, manifest) for static hosting.

`czml` writes one combined CZML to `data/outputs/`. `publish` splits it into a small
`base.czml` (clock, legend, AOI) and one CZML per scene and detector, copies the THOR vs CFAR
figures, and writes `manifest.json`, which the viewer reads at startup. The output folder is
rebuilt from scratch on every run and committed, so GitHub Pages serves it as is.

Only files under `paths.outputs` are ever read; raw/interim rasters, Kaggle chips, model weights
and cached features are never published.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from iceberg_sar.config import Config

Detector = Literal["cfar", "thor"]
MANIFEST = "manifest.json"
BASE_CZML = "base.czml"
COMPARE_DIR = "compare"
# Never published, whatever the config says (CLAUDE.md: no raw data, chips or weights).
FORBIDDEN_PATHS = ("raw", "interim", "kaggle", "models")
FORBIDDEN_DIRS = ("data/thor",)          # THOR chips, cached features, heads
# Sentinel-1 product type letters 1-2: polarization mode.
POLARIZATIONS = {"DH": "HH+HV", "DV": "VV+VH", "SH": "HH", "SV": "VV"}
# Packets of one scene, by id prefix: `<prefix>/<scene_id>/...` or `<prefix>/<scene_id>`.
SCENE_PREFIXES = ("scene", "footprint", "seaice", "truth")
DATE_PREFIXES = ("pass", "unobserved")    # `<prefix>/<YYYY-MM-DD>...`: go with the date's 1st scene
BASE_PREFIXES = ("legend", "aoi")
STUB_DOCUMENT = {"id": "document", "version": "1.0"}   # every CZML stream must start with one
LOCK_RETRIES = 5
LOCK_WAIT_S = 0.5


class SceneEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    date: date
    polarization: str
    detector: Detector
    czml: str                                   # relative to the manifest
    quicklook: str | None = None                # not published yet (Milestone 6 decision)
    quicklook_bounds: tuple[float, float, float, float] | None = None   # W, S, E, N (EPSG:4326)
    n_detections: int = Field(ge=0)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated: datetime
    base_czml: str
    compare_index: str | None = None            # THOR vs CFAR figures (`compare/index.json`)
    scenes: list[SceneEntry]


@dataclass(frozen=True)
class PublishParams:
    out_dir: Path
    scenes: tuple[str, ...] | None              # None = all
    detectors: tuple[Detector, ...]
    max_file_mb: float
    max_total_mb: float

    @classmethod
    def from_config(cls, cfg: Config) -> PublishParams:
        p = cfg.section("publish")
        scenes = p.get("scenes", "all")
        detectors = tuple(p.get("detectors", ["cfar"]))
        if bad := set(detectors) - {"cfar", "thor"}:
            raise ValueError(f"publish.detectors: unknown {sorted(bad)}")
        return cls(
            out_dir=(cfg.root / p["out_dir"]).resolve(),
            scenes=None if scenes == "all" else tuple(scenes),
            detectors=detectors,  # type: ignore[arg-type]
            max_file_mb=float(p["max_file_mb"]),
            max_total_mb=float(p["max_total_mb"]),
        )


class PublishError(RuntimeError):
    pass


def polarization(scene_id: str) -> str:
    """'S1A_IW_20190429T094108_DHP_...' -> 'HH+HV'."""
    mode = scene_id.split("_")[3][:2]
    return POLARIZATIONS.get(mode, mode)


def _scene_of(packet_id: str, scenes: set[str]) -> str | None:
    prefix, _, rest = packet_id.partition("/")
    if prefix in SCENE_PREFIXES:
        name = rest.split("/")[0]
        return name if name in scenes else None
    return None


def split_czml(packets: list[dict[str, Any]], p: PublishParams
               ) -> tuple[list[dict[str, Any]], dict[tuple[str, Detector], list[dict[str, Any]]]]:
    """Combined CZML -> base packets and per (scene, detector) packets.

    Every packet must land somewhere: an unknown id prefix fails instead of silently vanishing.
    """
    if not packets or packets[0].get("id") != "document":
        raise PublishError("CZML must start with a document packet")
    all_scenes = {pk["properties"]["scene_id"] for pk in packets if pk["id"].startswith("scene/")}
    keep = all_scenes if p.scenes is None else all_scenes & set(p.scenes)
    if p.scenes is not None and (missing := set(p.scenes) - all_scenes):
        raise PublishError(f"publish.scenes not in the CZML: {sorted(missing)}")
    # A date's pass / unobserved packets go with its first kept scene. A pass with only some of
    # its frames kept loses its coverage, which would describe the frames left out.
    frames_by_day: dict[str, list[str]] = defaultdict(list)
    for name in sorted(all_scenes):
        frames_by_day[_day(name)].append(name)
    day_owner = {d: next((f for f in fs if f in keep), None) for d, fs in frames_by_day.items()}
    partial = {d for d, fs in frames_by_day.items() if 0 < len(set(fs) & keep) < len(fs)}

    base = [packets[0]]
    per: dict[tuple[str, Detector], list[dict[str, Any]]] = defaultdict(list)
    for pk in packets[1:]:
        pid = pk["id"]
        prefix = pid.split("/")[0]
        if prefix in BASE_PREFIXES:
            base.append(pk)
        elif prefix in ("det", "thor"):
            name = pk["properties"]["scene_id"]
            det: Detector = "cfar" if prefix == "det" else "thor"
            if name in keep and det in p.detectors:
                per[(name, det)].append(pk)
        elif prefix in SCENE_PREFIXES:
            if name := _scene_of(pid, keep):
                if prefix == "scene" and "thor" not in p.detectors:
                    pk = _without_thor(pk)
                per[(name, "cfar")].append(pk)
        elif prefix in DATE_PREFIXES:
            day = pid.split("/")[1]
            if (owner := day_owner.get(day)) is None:
                continue
            if day in partial:
                if prefix == "unobserved":
                    continue
                pk = {**pk, "properties": {k: v for k, v in pk["properties"].items()
                                           if k not in ("observed_km2", "aoi_coverage_pct")}}
            per[(owner, "cfar")].append(pk)
        else:
            raise PublishError(f"Unknown CZML packet id: {pid}")
    # A scene with THOR run but nothing above the score still gets an (empty) THOR file.
    for pk in packets:
        props = pk.get("properties", {})
        if pk["id"].startswith("scene/") and "n_thor" in props and "thor" in p.detectors:
            if props["scene_id"] in keep:
                per.setdefault((props["scene_id"], "thor"), [])
    return base, dict(per)


def _day(scene_id: str) -> str:
    t = scene_id.split("_")[2]
    return f"{t[:4]}-{t[4:6]}-{t[6:8]}"


def _without_thor(packet: dict[str, Any]) -> dict[str, Any]:
    props = {k: v for k, v in packet["properties"].items() if k not in ("n_thor", "thor_score")}
    return {**packet, "properties": props}


def _write_czml(path: Path, packets: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(packets, separators=(",", ":")), encoding="utf-8")


def check_source(path: Path, cfg: Config) -> Path:
    """Only derived products under `paths.outputs` may be published."""
    path = path.resolve()
    forbidden = [cfg.path(k) for k in FORBIDDEN_PATHS if k in cfg.section("paths")]
    forbidden += [(cfg.root / d).resolve() for d in FORBIDDEN_DIRS]
    if any(path.is_relative_to(f) for f in forbidden):
        raise PublishError(f"Refusing to publish from a raw/private location: {path}")
    if not path.is_relative_to(cfg.path("outputs")):
        raise PublishError(f"Only files under paths.outputs are published: {path}")
    return path


def check_out_dir(out_dir: Path, cfg: Config) -> None:
    """The output folder is deleted and rebuilt, so keep it inside the repo and out of data/."""
    root = cfg.root.resolve()
    if out_dir == root or not out_dir.is_relative_to(root):
        raise PublishError(f"publish.out_dir must be a folder inside the repo: {out_dir}")
    if out_dir.is_relative_to(cfg.path("data_dir")):
        raise PublishError(f"publish.out_dir must not be inside the data folder: {out_dir}")


def check_sizes(folder: Path, max_file_mb: float, max_total_mb: float) -> float:
    """Total size in MB; fails if one file or the total is over its limit."""
    files = [f for f in folder.rglob("*") if f.is_file()]
    big = [(f, f.stat().st_size / 1e6) for f in files if f.stat().st_size / 1e6 > max_file_mb]
    if big:
        listing = ", ".join(f"{f.name} ({mb:.1f} MB)" for f, mb in big)
        raise PublishError(f"Files over publish.max_file_mb={max_file_mb:g}: {listing}")
    total = sum(f.stat().st_size for f in files) / 1e6
    if total > max_total_mb:
        raise PublishError(f"Published data is {total:.1f} MB, over "
                           f"publish.max_total_mb={max_total_mb:g}")
    return total


def build(cfg: Config, p: PublishParams, staging: Path) -> Manifest:
    """Write everything into an empty `staging` folder and return the manifest."""
    czml_path = check_source(cfg.path("czml"), cfg)
    if not czml_path.is_file():
        raise PublishError(f"No CZML at {czml_path}; run `czml` first")
    packets = json.loads(czml_path.read_text(encoding="utf-8"))
    base, per = split_czml(packets, p)
    _write_czml(staging / BASE_CZML, base)

    starts = {pk["properties"]["scene_id"]: pk["properties"]["start"]
              for pk in packets if pk["id"].startswith("scene/")}
    entries = []
    for (name, det), pks in sorted(per.items(), key=lambda kv: (starts[kv[0][0]], kv[0][1])):
        fname = f"{name}_{det}.czml"
        _write_czml(staging / fname, [STUB_DOCUMENT, *pks])
        prefix = "det/" if det == "cfar" else "thor/"
        entries.append(SceneEntry(
            scene_id=name, date=date.fromisoformat(starts[name][:10]),
            polarization=polarization(name), detector=det, czml=fname,
            n_detections=sum(pk["id"].startswith(prefix) for pk in pks)))

    compare_index = None
    if "compare_figures" in cfg.section("paths"):
        src = check_source(cfg.path("compare_figures"), cfg)
        if (src / "index.json").is_file():
            shutil.copytree(src, staging / COMPARE_DIR)
            compare_index = f"{COMPARE_DIR}/index.json"
    manifest = Manifest(generated=datetime.now(timezone.utc).replace(microsecond=0),
                        base_czml=BASE_CZML, compare_index=compare_index, scenes=entries)
    (staging / MANIFEST).write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    return manifest


def _retry(action: Callable[[], object], what: Path) -> None:
    """Windows: a file just closed by a watcher or virus scanner can stay locked briefly."""
    for attempt in range(LOCK_RETRIES):
        try:
            action()
            return
        except PermissionError:
            if attempt == LOCK_RETRIES - 1:
                raise PublishError(f"{what} is locked by another program (a running `npm run "
                                   f"dev` / preview, or an Explorer window). Close it and "
                                   f"re-run `publish`.") from None
            time.sleep(LOCK_WAIT_S)


def _replace_contents(src: Path, dst: Path) -> None:
    """Empty `dst` and move `src`'s entries into it. `dst` itself is kept: a dev server
    watching viewer/public/ holds a handle on it, and Windows can't rename or delete a
    folder with an open handle."""
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        _retry((lambda o=old: shutil.rmtree(o)) if old.is_dir() else old.unlink, old)
    for new in src.iterdir():
        _retry(lambda n=new: shutil.move(n, dst / n.name), dst / new.name)


def publish(cfg: Config) -> tuple[Path, Manifest, float]:
    """Rebuild `publish.out_dir`. Everything is written and checked in a staging folder under
    paths.outputs first, so a failed build or size check leaves the previous publish in place."""
    p = PublishParams.from_config(cfg)
    check_out_dir(p.out_dir, cfg)
    cfg.path("outputs").mkdir(parents=True, exist_ok=True)
    # Same volume as out_dir (cheap moves), and outside the folder the dev server watches.
    with tempfile.TemporaryDirectory(prefix="publish-", dir=cfg.path("outputs")) as tmp:
        staging = Path(tmp)
        manifest = build(cfg, p, staging)
        total_mb = check_sizes(staging, p.max_file_mb, p.max_total_mb)
        _replace_contents(staging, p.out_dir)
    return p.out_dir, manifest, total_mb


def read_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
