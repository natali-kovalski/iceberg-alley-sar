from pathlib import Path

import pytest
from typer.testing import CliRunner

from iceberg_sar.cli import app
from iceberg_sar.config import REQUIRED_SECTIONS, load_config, metres_to_px

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config.yaml"


def test_repo_config_loads() -> None:
    cfg = load_config(CONFIG)
    assert cfg.section("cfar")["variant"] == "CA"
    assert cfg.path("aoi").is_file()
    assert cfg.path("raw") == (REPO_ROOT / "data" / "raw").resolve()


def test_missing_section_raises(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("paths: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing sections"):
        load_config(bad)


def test_repo_config_includes_benchmark() -> None:
    cfg = load_config(CONFIG)
    assert cfg.files == (CONFIG.resolve(), (REPO_ROOT / "benchmarks" / "config.yaml").resolve())
    assert "include" not in cfg.raw
    assert cfg.section("benchmark")["out_dir"] == "data/benchmark"


_BASE = "".join(f"{s}: {{}}\n" for s in REQUIRED_SECTIONS)


def test_include_merges_sections(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "extra.yaml").write_text("extra: {a: 1}\n", encoding="utf-8")
    main = tmp_path / "config.yaml"
    main.write_text(_BASE + "include: [sub/extra.yaml]\n", encoding="utf-8")
    cfg = load_config(main)
    assert cfg.section("extra") == {"a": 1}
    assert cfg.root == tmp_path.resolve()  # paths still resolve against the main config


def test_include_rejects_duplicate_section(tmp_path: Path) -> None:
    (tmp_path / "extra.yaml").write_text("cfar: {variant: OS}\n", encoding="utf-8")
    main = tmp_path / "config.yaml"
    main.write_text(_BASE + "include: [extra.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="defined twice.*cfar"):
        load_config(main)


def test_missing_include_raises(tmp_path: Path) -> None:
    main = tmp_path / "config.yaml"
    main.write_text(_BASE + "include: [nope.yaml]\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(main)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_cli_help_and_info() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["info", "--config", str(CONFIG)])
    assert result.exit_code == 0, result.output
    assert "aoi:" in result.output


def test_metres_to_px_scales_with_resolution() -> None:
    assert metres_to_px(400, 20) == 20
    assert metres_to_px(400, 10) == 40
    assert metres_to_px(5, 20) == 1  # never below one pixel
