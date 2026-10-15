"""End-to-end run: scene selection, matching granules to downloaded products, dry runs."""

from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from iceberg_sar import pipeline
from iceberg_sar.config import Config
from iceberg_sar.pipeline import RunParams, find_product, run_pipeline, select_scenes

G_0502 = "S1C_IW_GRDH_1SDH_20250502T094802_20250502T094827_002150_00490B_014B"
G_0508 = "S1A_IW_GRDH_1SDH_20250508T094925_20250508T094950_059101_075526_EB40"
G_0514 = "S1C_IW_GRDH_1SDH_20250514T094802_20250514T094827_002325_004E8A_DEAF"


def _scenes() -> gpd.GeoDataFrame:
    rows = [
        (G_0514, "2025-05-14T09:48:02Z", "HH+HV", 0.41),
        (G_0502, "2025-05-02T09:48:02Z", "HH+HV", 0.41),
        ("S1A_IW_GRDH_1SDV_20250505T000000_x", "2025-05-05T00:00:00Z", "VV+VH", 0.60),
        ("S1A_IW_GRDH_1SDH_20250506T000000_x", "2025-05-06T00:00:00Z", "HH+HV", 0.01),
        (G_0508, "2025-05-08T09:49:25Z", "HH+HV", 0.42),
    ]
    return gpd.GeoDataFrame(
        [dict(zip(["scene", "start_time", "polarization", "aoi_overlap"], r, strict=True))
         for r in rows],
        geometry=[box(-55, 49, -53, 51)] * len(rows), crs="EPSG:4326")


def test_select_filters_pol_and_overlap_and_sorts_by_time() -> None:
    out = select_scenes(_scenes(), RunParams(max_scenes=None))
    assert list(out.scene) == [G_0502, G_0508, G_0514]


def test_select_keeps_best_covering_when_capped() -> None:
    out = select_scenes(_scenes(), RunParams(max_scenes=1))
    assert list(out.scene) == [G_0508]


def test_select_any_pol() -> None:
    out = select_scenes(_scenes(), RunParams(pol=None, max_scenes=None))
    assert len(out) == 4


def test_find_product_matches_platform_time_and_resolution(tmp_path: Path) -> None:
    (tmp_path / "S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_FDE3").mkdir()
    (tmp_path / "S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1").mkdir()
    (tmp_path / "S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1.zip").touch()
    assert find_product(tmp_path, G_0508, 10).name.endswith("RTC10_G_gpuned_02C1")
    assert find_product(tmp_path, G_0508, 20).name.endswith("RTC20_G_gpuned_FDE3")
    assert find_product(tmp_path, G_0502, 10) is None


def _cfg(tmp_path: Path) -> Config:
    return Config(root=tmp_path, raw={
        "paths": {"raw": "raw", "interim": "interim", "outputs": "outputs", "models": "models"},
        "hyp3": {"resolution": 10},
        "cfar": {"band": "HV"},
        "run": {"max_scenes": 2},
    })


def test_dry_run_orders_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "raw" / "S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1").mkdir(parents=True)
    monkeypatch.setattr("iceberg_sar.search.search_scenes", lambda cfg, start, end: _scenes())
    monkeypatch.setattr(pipeline, "order_and_download",
                        lambda *a: pytest.fail("dry run must not order"))
    lines: list[str] = []
    out = run_pipeline(_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), dry_run=True,
                       log=lines.append)
    assert out is None
    assert "downloaded" in lines[0] and G_0508 in lines[0]
    assert "to order" in lines[1] and G_0514 in lines[1]
    assert any("1 to order (~60 HyP3 credits at 10 m)" in line for line in lines)


def test_no_scenes_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iceberg_sar.search.search_scenes",
                        lambda cfg, start, end: _scenes().iloc[:0])
    with pytest.raises(RuntimeError, match="No HH\\+HV scenes"):
        run_pipeline(_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), dry_run=True)


# --- Milestone 7b: THOR in the run ---------------------------------------------------------

def _thor_cfg(tmp_path: Path, **run: object) -> Config:
    cfg = _cfg(tmp_path)
    cfg.raw["run"] |= run
    cfg.raw["viewer"] = {"thor_model": "thor-linear"}
    return cfg


def _head(cfg: Config) -> Path:
    head = cfg.path("models") / "thor" / "linear" / "final" / "head.pt"
    head.parent.mkdir(parents=True)
    head.touch()
    return head


def test_detectors_from_config_and_override(tmp_path: Path) -> None:
    assert pipeline.run_params(_cfg(tmp_path)).detectors == ("thor", "cfar")   # default
    assert pipeline.run_params(_thor_cfg(tmp_path, detectors=["cfar"])).detectors == ("cfar",)
    # The CLI override wins; order is canonical (THOR first), duplicates and case don't matter.
    p = pipeline.run_params(_thor_cfg(tmp_path, detectors=["cfar"]), detectors="CFAR, thor,cfar")
    assert p.detectors == ("thor", "cfar")


@pytest.mark.parametrize("bad", ["", "thor,sift", ["yolo"]])
def test_detectors_rejects_unknown_or_empty(bad: object) -> None:
    with pytest.raises(ValueError, match="detectors"):
        pipeline.parse_detectors(bad)


def test_thor_head_follows_viewer_model(tmp_path: Path) -> None:
    cfg = _thor_cfg(tmp_path)
    head, tag = pipeline.thor_head(cfg)
    assert (tag, head) == ("thor-linear", cfg.path("models") / "thor/linear/final/head.pt")
    cfg.raw["viewer"]["thor_model"] = "thor-centernet-fold_a"
    assert pipeline.thor_head(cfg)[0].parts[-3:] == ("centernet", "fold_a", "head.pt")


def test_thor_status_needs_head_and_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _thor_cfg(tmp_path)
    monkeypatch.setattr(pipeline, "missing_modules", lambda: [])
    monkeypatch.setattr(pipeline, "cuda_device_name", lambda: None)
    s = pipeline.thor_status(cfg)
    assert not s.ready and "no trained head" in s.reason and "thor-fetch-head" in s.reason
    _head(cfg)
    s = pipeline.thor_status(cfg)
    assert not s.ready and "no CUDA GPU" in s.reason
    monkeypatch.setattr(pipeline, "cuda_device_name", lambda: "RTX Test")
    s = pipeline.thor_status(cfg)
    assert s.ready and "RTX Test" in s.reason and s.tag == "thor-linear"
    monkeypatch.setattr(pipeline, "missing_modules", lambda: ["terratorch"])
    s = pipeline.thor_status(cfg)
    assert not s.ready and "terratorch" in s.reason


def test_dry_run_reports_thor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iceberg_sar.search.search_scenes", lambda cfg, start, end: _scenes())
    status = pipeline.ThorStatus(False, "no CUDA GPU", Path("h.pt"), "thor-linear")
    monkeypatch.setattr(pipeline, "thor_status", lambda cfg, device="cuda": status)
    lines: list[str] = []
    run_pipeline(_thor_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), dry_run=True,
                 log=lines.append)
    assert any(line.startswith("THOR") and "skipped: no CUDA GPU" in line for line in lines)
    lines.clear()
    monkeypatch.setattr(pipeline, "thor_status",
                        lambda cfg, device="cuda": status.__class__(True, "CUDA (X)", *status[2:]))
    run_pipeline(_thor_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), dry_run=True,
                 log=lines.append)
    assert any(line.startswith("THOR") and "will run: CUDA (X)" in line for line in lines)


def _stub_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ready: bool) -> dict[str, list]:
    """Full (non-dry) run with every external step stubbed; records what was called."""
    for name in ("S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1",
                 "S1C_IW_20250514T094802_DHP_RTC10_G_gpuned_AAAA"):
        (tmp_path / "raw" / name).mkdir(parents=True)
    calls: dict[str, list] = {"cfar": [], "thor": [], "czml": []}
    monkeypatch.setattr("iceberg_sar.search.search_scenes", lambda cfg, start, end: _scenes())
    monkeypatch.setattr("iceberg_sar.landmask.build_land_mask", lambda cfg: None)
    monkeypatch.setattr("iceberg_sar.landmask.load_land", lambda path: None)
    monkeypatch.setattr(pipeline, "process_product",
                        lambda d, land, cfg, band, force, log, cfar=True:
                        calls["cfar"].append((d.name, cfar)))
    monkeypatch.setattr(pipeline, "detect_thor",
                        lambda d, cfg, status, device, force, log: calls["thor"].append(d.name))
    monkeypatch.setattr("iceberg_sar.export_czml.export_czml",
                        lambda cfg, band: calls["czml"].append(band) or (tmp_path / "x.czml", []))
    status = pipeline.ThorStatus(ready, "CUDA (X)" if ready else "no CUDA GPU", Path("h.pt"),
                                 "thor-linear")
    monkeypatch.setattr(pipeline, "thor_status", lambda cfg, device="cuda": status)
    return calls


def test_run_with_gpu_writes_thor_and_cfar(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run(tmp_path, monkeypatch, ready=True)
    out = run_pipeline(_thor_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), log=print)
    assert out == tmp_path / "x.czml"
    assert [c[1] for c in calls["cfar"]] == [True, True]
    assert len(calls["thor"]) == 2 and calls["czml"] == ["HV"]


def test_run_without_gpu_skips_thor_and_finishes(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run(tmp_path, monkeypatch, ready=False)
    lines: list[str] = []
    out = run_pipeline(_thor_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15),
                       log=lines.append)
    assert out is not None and calls["thor"] == [] and len(calls["cfar"]) == 2
    assert sum("skipped: no CUDA GPU" in line for line in lines) == 1   # one notice, not per scene


def test_run_cfar_only_never_checks_thor(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run(tmp_path, monkeypatch, ready=True)
    monkeypatch.setattr(pipeline, "thor_status", lambda *a, **k: pytest.fail("not requested"))
    run_pipeline(_thor_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), log=print,
                 detectors="cfar")
    assert calls["thor"] == [] and len(calls["cfar"]) == 2


def test_run_thor_only_still_preprocesses(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run(tmp_path, monkeypatch, ready=True)
    run_pipeline(_thor_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), log=print,
                 detectors="thor")
    assert [c[1] for c in calls["cfar"]] == [False, False] and len(calls["thor"]) == 2
