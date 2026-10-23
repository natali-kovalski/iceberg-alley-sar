"""Færch 2023 benchmark scenes: `scenes.csv` checked against Table 1 of the paper's JATS XML.

`scenes.csv` has two parts per row: the paper's Table 1 columns as transcribed (sensor ...
enl), and the product we use for it (`product_id`, `source`). `basis` says why a product is
there: `table1`; `table1_note_format` (the GRDM, kept because Table 1's ENL fits it, not
its "GRDH"); `fig1` (S2 tiles not in Table 1 but inside the paper's Fig. 1 S2 outline; their
tile column copies Table 1's pattern, so the check flags them). `check_scenes` compares both
parts with Table 1 as parsed from `faerch2023.xml`, so a typo in the CSV and a product that
differs from what the paper names both show up as mismatches.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TABLE_NS = "{http://docs.oasis-open.org/ns/oasis-exchange/table}"
PAPER_COLUMNS = ("sensor", "format", "acquisition", "tile_orbit", "pixel_spacing_m", "bands",
                 "enl")
CONFIG_BEGIN = "    # >>> scenes: generated from scenes.csv by `bench-scenes --write-config`"
CONFIG_END = "    # <<< scenes"


@dataclass(frozen=True)
class Check:
    product: str
    field: str
    paper: str
    ours: str
    ok: bool


def _text(e: ET.Element) -> str:
    return " ".join("".join(e.itertext()).replace(" ", " ").replace("\xa0", " ").split())


def table1(xml_path: Path) -> list[dict[str, str]]:
    """Rows of Table 1 as {header: cell text}; footnote letters stay on the cell (e.g. 10.7a)."""
    root = ET.parse(xml_path).getroot()
    for tw in root.iter("table-wrap"):
        label = tw.find("label")
        if label is not None and _text(label) == "Table 1":
            rows = [[_text(c) for c in r.iter(TABLE_NS + "entry")] for r in tw.iter(TABLE_NS + "row")]
            return [dict(zip(rows[0], r, strict=True)) for r in rows[1:]]
    raise ValueError(f"Table 1 not found in {xml_path}")


def load_scenes(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_product(name: str) -> dict[str, Any]:
    """Fields of a Sentinel-1 GRD or Sentinel-2 L1C product name."""
    name = name.removesuffix(".SAFE")
    m = re.fullmatch(r"(S1[ABC])_(\w\w)_(GRD[HM])_1S(\w\w)_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{6})_"
                     r"(\w{6})_(\w{4})", name)
    if m:
        return {"mission": m[1], "mode": m[2], "format": f"{m[2]} {m[3]}",
                "start": datetime.strptime(m[5], "%Y%m%dT%H%M%S"),
                "stop": datetime.strptime(m[6], "%Y%m%dT%H%M%S"), "abs_orbit": m[7]}
    m = re.fullmatch(r"(S2[ABC])_MSIL1C_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\w{5})_(\d{8}T\d{6})",
                     name)
    if m:
        return {"mission": m[1], "format": "L1C",
                "start": datetime.strptime(m[2], "%Y%m%dT%H%M%S"), "baseline": m[3],
                "rel_orbit": m[4], "tile": f"T{m[5]}"}
    raise ValueError(f"Unrecognised product name: {name}")


def _paper_time(s: str) -> datetime:
    return datetime.strptime(s, "%d %B %Y %H:%M:%S")


def _strip_note(s: str) -> str:
    """'10.7a' -> '10.7': drop a table footnote letter."""
    return re.sub(r"(?<=\d)[a-z]$", "", s)


def check_scenes(scenes: list[dict[str, str]], table: list[dict[str, str]],
                 time_tol_s: float = 60.0) -> list[Check]:
    """CSV transcription vs Table 1, then each product vs the paper's row."""
    by_sensor = {r["Sensor"]: r for r in table}
    checks: list[Check] = []
    for s in scenes:
        pid = s["product_id"]
        paper = by_sensor.get(s["sensor"])
        if paper is None:
            checks.append(Check(pid, "sensor", "(not in Table 1)", s["sensor"], False))
            continue
        tiles = [t.split("/") for t in paper["Tile(s)/orbit identifier"].split()]
        # 1. Transcription: the CSV's paper columns against Table 1.
        tile, orbit = s["tile_orbit"].split("/")
        transcribed = {
            "format": (paper["Format"], s["format"]),
            "acquisition": (_paper_time(paper["Acquisition date/time"]).isoformat(),
                            s["acquisition"]),
            "tile_orbit": (" ".join("/".join(t) for t in tiles), s["tile_orbit"]),
            "pixel_spacing_m": (paper["Pixel spacing"].removesuffix(" m"), s["pixel_spacing_m"]),
            "bands": (paper["Bands"], s["bands"]),
            "enl": (_strip_note(paper["ENL"]).replace("n/a", ""), s["enl"]),
        }
        for field, (p, o) in transcribed.items():
            ok = [tile, orbit] in tiles if field == "tile_orbit" else p == o
            checks.append(Check(pid, f"csv.{field}", p, o, ok))
        # 2. Product: what the product name says against the paper's row.
        prod = parse_product(pid)
        t = _paper_time(paper["Acquisition date/time"])
        if prod["format"].startswith("EW"):
            inside = prod["start"] <= t <= prod["stop"]
            checks.append(Check(pid, "product.format", paper["Format"], prod["format"],
                                prod["format"] == paper["Format"]))
            checks.append(Check(pid, "product.time", t.isoformat(),
                                f"{prod['start'].isoformat()} - {prod['stop'].time()}", inside))
            checks.append(Check(pid, "product.abs_orbit", orbit, prod["abs_orbit"],
                                orbit.lstrip("0") == prod["abs_orbit"].lstrip("0")))
        else:
            dt = abs(prod["start"] - t)
            checks.append(Check(pid, "product.tile", tile, prod["tile"], prod["tile"] == tile))
            checks.append(Check(pid, "product.time", t.isoformat(), prod["start"].isoformat(),
                                dt <= timedelta(seconds=time_tol_s)))
            checks.append(Check(pid, "product.rel_orbit", orbit, f"R{prod['rel_orbit']}",
                                orbit == f"R{prod['rel_orbit']}"))
    return checks


def config_block(scenes: list[dict[str, str]]) -> str:
    """YAML lines for `benchmark.faerch2023.scenes`, between the generated-block markers."""
    lines = [CONFIG_BEGIN, "    scenes:"]
    for s in scenes:
        prod = parse_product(s["product_id"])
        lines.append(f"      - {{sensor: {s['sensor']}, product_id: {s['product_id']}, "
                     f"source: {s['source']}, basis: {s.get('basis', 'table1')}, "
                     f"start: '{prod['start'].isoformat()}'}}")
    lines.append(CONFIG_END)
    return "\n".join(lines)


def write_config_block(config_path: Path, scenes: list[dict[str, str]]) -> None:
    """Replace the generated block in config.yaml in place (comments elsewhere are kept)."""
    text = config_path.read_text(encoding="utf-8")
    i, j = text.find(CONFIG_BEGIN), text.find(CONFIG_END)
    if i < 0 or j < i:
        raise ValueError(f"Markers for the generated scenes block not found in {config_path}")
    text = text[:i] + config_block(scenes) + text[j + len(CONFIG_END):]
    config_path.write_text(text, encoding="utf-8")
