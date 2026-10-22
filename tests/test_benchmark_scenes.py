"""scenes.csv vs Table 1 of the paper: product-name parsing and the mismatch report."""

from pathlib import Path

import pytest

from iceberg_sar.benchmark.scenes import (
    CONFIG_BEGIN,
    CONFIG_END,
    check_scenes,
    load_scenes,
    parse_product,
    table1,
    write_config_block,
)

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "benchmarks" / "faerch2023" / "faerch2023.xml"
CSV = ROOT / "benchmarks" / "faerch2023" / "scenes.csv"

TABLE = [
    {"Sensor": "Sentinel-1", "Format": "EW GRDH", "Acquisition date/time": "4 July 2019 10:27:30",
     "Tile(s)/orbit identifier": "–/027966", "Pixel spacing": "40 m", "Bands": "HH, HV",
     "ENL": "10.7a"},
]


def test_parse_s1_and_s2() -> None:
    s1 = parse_product("S1A_EW_GRDH_1SDH_20190704T102730_20190704T102831_027966_032866_DEF7.SAFE")
    assert s1["format"] == "EW GRDH" and s1["abs_orbit"] == "027966"
    s2 = parse_product("S2B_MSIL1C_20190704T153609_N0207_R111_T20VNJ_20190704T190623")
    assert s2["tile"] == "T20VNJ" and s2["rel_orbit"] == "111" and s2["baseline"] == "0207"
    with pytest.raises(ValueError):
        parse_product("not_a_product")


def _row(**kw: str) -> dict[str, str]:
    base = {"sensor": "Sentinel-1", "format": "EW GRDH", "acquisition": "2019-07-04T10:27:30",
            "tile_orbit": "–/027966", "pixel_spacing_m": "40", "bands": "HH, HV", "enl": "10.7",
            "product_id": "S1A_EW_GRDH_1SDH_20190704T102730_20190704T102831_027966_032866_DEF7",
            "source": "cdse"}
    return base | kw


def test_check_agrees_and_flags() -> None:
    assert all(c.ok for c in check_scenes([_row()], TABLE))
    bad = {c.field for c in check_scenes(
        [_row(enl="3.0", product_id="S1A_EW_GRDM_1SDH_20190704T102830_20190704T102931_"
              "027967_032866_AAAA")], TABLE) if not c.ok}
    assert bad == {"csv.enl", "product.format", "product.time", "product.abs_orbit"}


@pytest.mark.skipif(not XML.is_file(), reason="paper XML not present")
def test_table1_from_xml_and_committed_csv() -> None:
    t = table1(XML)
    assert [r["Sensor"] for r in t] == ["Sentinel-2", "Sentinel-1", "ALOS-2"]
    checks = check_scenes(load_scenes(CSV), t)
    # The CSV transcribes Table 1 faithfully; only the orbit identifiers disagree with the
    # products (documented mismatches, see benchmarks/faerch2023/scenes.csv).
    assert all(c.ok for c in checks if c.field.startswith("csv."))
    assert {c.field for c in checks if not c.ok} == {"product.abs_orbit", "product.rel_orbit"}


def test_write_config_block(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"a: 1\n{CONFIG_BEGIN}\n    old: x\n{CONFIG_END}\nb: 2\n", encoding="utf-8")
    write_config_block(cfg, [_row()])
    text = cfg.read_text(encoding="utf-8")
    assert "old: x" not in text and "product_id: S1A_EW_GRDH" in text
    assert text.startswith("a: 1\n") and text.endswith("b: 2\n")
