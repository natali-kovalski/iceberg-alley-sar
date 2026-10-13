"""`publish`: CZML split per scene and detector, manifest schema, size limits, source guard."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from iceberg_sar.config import Config, load_config
from iceberg_sar.publish import (
    Manifest,
    PublishError,
    PublishParams,
    check_sizes,
    polarization,
    publish,
    read_manifest,
    split_czml,
)

A = "S1A_IW_20190429T094108_DHP_RTC10_G_gpuned_671B"   # pass 2019-04-29, frame 1 of 2
B = "S1A_IW_20190429T094133_DHP_RTC10_G_gpuned_AAAA"   # pass 2019-04-29, frame 2 of 2
C = "S1C_IW_20250502T094802_DHP_RTC10_G_gpuned_F77A"   # pass 2025-05-02, no THOR run


def _scene(name: str, start: str, thor: bool) -> dict[str, Any]:
    props: dict[str, Any] = {"scene_id": name, "start": start, "n_detections": 2}
    if thor:
        props |= {"n_thor": 1, "thor_score": 0.35}
    return {"id": f"scene/{name}", "availability": "x", "properties": props}


def _point(prefix: str, name: str, i: int) -> dict[str, Any]:
    return {"id": f"{prefix}/{name[7:22]}_{i:05d}", "parent": f"scene/{name}",
            "properties": {"scene_id": name}}


def _packets() -> list[dict[str, Any]]:
    return [
        {"id": "document", "version": "1.0", "clock": {"interval": "a/b"}},
        {"id": "legend", "properties": {"near_ice_km": 5}},
        {"id": "aoi/0", "polygon": {}},
        _scene(A, "2019-04-29T09:41:08Z", thor=True),
        {"id": f"footprint/{A}/0", "polygon": {}},
        {"id": f"seaice/{A}/0", "polygon": {}},
        _point("det", A, 0), _point("det", A, 1), _point("thor", A, 0),
        {"id": f"truth/{A}/0", "parent": f"scene/{A}"},
        _scene(B, "2019-04-29T09:41:33Z", thor=True),
        _point("det", B, 0),
        _scene(C, "2025-05-02T09:48:02Z", thor=False),
        _point("det", C, 0), _point("det", C, 1),
        {"id": "unobserved/2019-04-29/0", "polygon": {}},
        {"id": "pass/2019-04-29", "properties": {"date": "2019-04-29", "frames": f"{A} {B}",
                                                 "observed_km2": 9, "aoi_coverage_pct": 30.0}},
        {"id": "pass/2025-05-02", "properties": {"date": "2025-05-02", "frames": C}},
    ]


def _params(**kw: Any) -> PublishParams:
    d: dict[str, Any] = dict(out_dir=Path("out"), scenes=None, detectors=("cfar", "thor"),
                             max_file_mb=25, max_total_mb=300)
    return PublishParams(**(d | kw))


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    outputs = tmp_path / "data" / "outputs"
    (outputs / "compare").mkdir(parents=True)
    (outputs / "iceberg_alley.czml").write_text(json.dumps(_packets()), encoding="utf-8")
    (outputs / "compare" / "index.json").write_text(
        json.dumps({"2019-04-29": {"figure": "2019-04-29.png", "labels": 3}}), encoding="utf-8")
    (outputs / "compare" / "2019-04-29.png").write_bytes(b"png")
    raw = {
        "paths": {"data_dir": "data", "raw": "data/raw", "interim": "data/interim",
                  "outputs": "data/outputs", "kaggle": "data/kaggle/train.json",
                  "models": "data/models", "czml": "data/outputs/iceberg_alley.czml",
                  "compare_figures": "data/outputs/compare"},
        "publish": {"out_dir": "viewer/public/data", "scenes": "all",
                    "detectors": ["cfar", "thor"], "max_file_mb": 25, "max_total_mb": 300},
        **{s: {} for s in ("search", "hyp3", "preprocess", "cfar", "detections")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def _set(cfg: Config, section: str, **kw: Any) -> Config:
    raw = {**cfg.raw, section: {**cfg.raw[section], **kw}}
    return Config(root=cfg.root, raw=raw)


def test_polarization_from_product_name() -> None:
    assert polarization(A) == "HH+HV"
    assert polarization("S1A_IW_20250508T094925_DVP_RTC10_G_gpuned_02C1") == "VV+VH"


def test_split_routes_every_packet() -> None:
    base, per = split_czml(_packets(), _params())
    assert [p["id"] for p in base] == ["document", "legend", "aoi/0"]
    assert set(per) == {(A, "cfar"), (A, "thor"), (B, "cfar"), (B, "thor"), (C, "cfar")}
    a = [p["id"] for p in per[(A, "cfar")]]
    assert a[0] == f"scene/{A}"                       # before its children (`parent`)
    assert {f"footprint/{A}/0", f"seaice/{A}/0", f"truth/{A}/0"} <= set(a)
    # The pass and its unobserved area go with the date's first frame only.
    assert {"pass/2019-04-29", "unobserved/2019-04-29/0"} <= set(a)
    assert not [p for p in per[(B, "cfar")] if p["id"].startswith(("pass/", "unobserved/"))]
    assert [p["id"] for p in per[(A, "thor")]] == [_point("thor", A, 0)["id"]]
    assert per[(B, "thor")] == []                     # THOR ran, nothing above the score
    total = len(base) + sum(len(v) for v in per.values())
    assert total == len(_packets())


def test_split_without_thor_drops_layer_and_counts() -> None:
    _, per = split_czml(_packets(), _params(detectors=("cfar",)))
    assert {d for _, d in per} == {"cfar"}
    scene_a = next(p for p in per[(A, "cfar")] if p["id"] == f"scene/{A}")
    assert "n_thor" not in scene_a["properties"]      # the viewer hides the THOR toggle


def test_split_scene_subset_drops_coverage_of_partial_pass() -> None:
    _, per = split_czml(_packets(), _params(scenes=(B,)))
    assert {s for s, _ in per} == {B}
    ids = [p["id"] for p in per[(B, "cfar")]]
    assert "pass/2019-04-29" in ids and "unobserved/2019-04-29/0" not in ids
    pas = next(p for p in per[(B, "cfar")] if p["id"] == "pass/2019-04-29")
    assert "aoi_coverage_pct" not in pas["properties"]
    with pytest.raises(PublishError, match="not in the CZML"):
        split_czml(_packets(), _params(scenes=("S1A_IW_20990101T000000_DHP_X",)))


def test_split_rejects_unknown_packet() -> None:
    with pytest.raises(PublishError, match="Unknown"):
        split_czml([*_packets(), {"id": "mystery/1"}], _params())


def test_publish_writes_manifest_and_files(cfg: Config) -> None:
    out, manifest, _ = publish(cfg)
    assert out == cfg.root / "viewer" / "public" / "data"
    assert read_manifest(out / "manifest.json") == manifest
    assert [(e.scene_id, e.detector, e.n_detections) for e in manifest.scenes] == [
        (A, "cfar", 2), (A, "thor", 1), (B, "cfar", 1), (B, "thor", 0), (C, "cfar", 2)]
    assert manifest.compare_index == "compare/index.json"
    assert (out / "compare" / "2019-04-29.png").is_file()
    for e in manifest.scenes:
        packets = json.loads((out / e.czml).read_text(encoding="utf-8"))
        assert packets[0] == {"id": "document", "version": "1.0"}
        assert e.quicklook is None and e.polarization == "HH+HV"
    base = json.loads((out / manifest.base_czml).read_text(encoding="utf-8"))
    assert "clock" in base[0]


def test_publish_is_idempotent_and_rebuilds(cfg: Config) -> None:
    out, _, _ = publish(cfg)
    (out / "stale.czml").write_text("[]", encoding="utf-8")
    publish(cfg)
    assert not (out / "stale.czml").exists()
    names = sorted(p.name for p in out.iterdir())
    publish(cfg)
    assert sorted(p.name for p in out.iterdir()) == names


def test_manifest_schema_round_trip_and_validation() -> None:
    m = Manifest.model_validate({
        "generated": "2026-10-13T12:00:00Z", "base_czml": "base.czml", "scenes": [{
            "scene_id": A, "date": "2019-04-29", "polarization": "HH+HV", "detector": "thor",
            "czml": "a.czml", "quicklook": None, "quicklook_bounds": [-55, 49, -52, 51],
            "n_detections": 4}]})
    assert Manifest.model_validate_json(m.model_dump_json()) == m
    bad = m.model_dump(mode="json")
    bad["scenes"][0]["detector"] = "yolo"
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)
    bad = m.model_dump(mode="json")
    bad["scenes"][0]["extra"] = 1
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


def test_size_limits_fire(tmp_path: Path) -> None:
    (tmp_path / "a.czml").write_bytes(b"x" * 2_000_000)
    (tmp_path / "b.czml").write_bytes(b"x" * 2_000_000)
    assert check_sizes(tmp_path, 3, 5) == pytest.approx(4.0)
    with pytest.raises(PublishError, match="max_file_mb"):
        check_sizes(tmp_path, 1, 5)
    with pytest.raises(PublishError, match="max_total_mb"):
        check_sizes(tmp_path, 3, 3)


def test_failed_publish_keeps_previous_output(cfg: Config) -> None:
    out, _, _ = publish(cfg)
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(PublishError, match="max_total_mb"):
        publish(_set(cfg, "publish", max_total_mb=0.0000001))
    assert sorted(p.name for p in out.iterdir()) == before
    assert not list(cfg.path("outputs").glob("publish-*"))    # staging folder cleaned up


def test_locked_file_fails_with_a_clear_message(cfg: Config, monkeypatch: pytest.MonkeyPatch
                                                ) -> None:
    out, _, _ = publish(cfg)

    def locked(self: Path, *a: Any, **kw: Any) -> None:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr("iceberg_sar.publish.LOCK_WAIT_S", 0)
    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(PublishError, match="locked by another program"):
        publish(cfg)
    assert out.is_dir()                                  # the folder itself is never removed


@pytest.mark.parametrize("source", ["data/raw/x.czml", "data/interim/x.czml",
                                    "data/kaggle/train.json", "data/models/x.czml",
                                    "data/thor/features/x.czml", "elsewhere/x.czml"])
def test_publish_never_reads_private_locations(cfg: Config, source: str) -> None:
    src = cfg.root / source
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_packets()), encoding="utf-8")
    with pytest.raises(PublishError, match="raw/private|paths.outputs"):
        publish(_set(cfg, "paths", czml=source))
    with pytest.raises(PublishError, match="raw/private|paths.outputs"):
        publish(_set(cfg, "paths", compare_figures=str(Path(source).parent)))


@pytest.mark.parametrize("out_dir", [".", "..", "data/outputs/web"])
def test_out_dir_must_be_inside_repo_and_outside_data(cfg: Config, out_dir: str) -> None:
    with pytest.raises(PublishError, match="out_dir"):
        publish(_set(cfg, "publish", out_dir=out_dir))
