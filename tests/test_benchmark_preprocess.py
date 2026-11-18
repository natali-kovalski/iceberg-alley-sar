"""Multilook stack and ENL estimate for the benchmark runner input."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from iceberg_sar.benchmark.enl import box_table, combine, estimate_enl, moment_enl
from iceberg_sar.benchmark.preprocess import block_mean, multilook_stack


def test_block_mean_ignores_nan_and_reports_valid_share() -> None:
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    x[0, 0] = np.nan
    mean, frac = block_mean(x, 2)
    assert mean[0, 0] == pytest.approx((1 + 4 + 5) / 3)
    assert mean[1, 1] == pytest.approx((10 + 11 + 14 + 15) / 4)
    assert frac[0, 0] == 0.75 and frac[1, 1] == 1.0


def test_block_mean_crops_partial_blocks() -> None:
    mean, _ = block_mean(np.ones((9, 10), np.float32), 4)
    assert mean.shape == (2, 2)


def _write(path: Path, data: np.ndarray, pix: float = 10.0) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs="EPSG:32622", nodata=np.nan,
                       transform=from_origin(600_000, 5_600_000, pix, pix)) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def test_multilook_stack(tmp_path: Path) -> None:
    hh = np.full((64, 48), 0.02, np.float32)
    hv = np.full((64, 48), 0.002, np.float32)
    hh[0, 0] = np.nan                       # one bad pixel -> its 4x4 block is invalid
    paths = {"HH": _write(tmp_path / "hh.tif", hh), "HV": _write(tmp_path / "hv.tif", hv)}
    raster, mask = multilook_stack(paths, 4, tmp_path / "s.tif", tmp_path / "m.tif",
                                   block_rows=16)
    with rasterio.open(raster) as src:
        assert (src.count, src.shape, src.res) == (2, (16, 12), (40.0, 40.0))
        assert src.descriptions == ("HH", "HV")
        assert src.transform.c == 600_000 and src.transform.f == 5_600_000
        data = src.read()
    with rasterio.open(mask) as src:
        valid = src.read(1)
    assert valid[0, 0] == 0 and valid.sum() == 16 * 12 - 1
    assert np.isnan(data[0, 0, 0]) and data[1, 5, 5] == pytest.approx(0.002)


def test_moment_enl_recovers_gamma_looks() -> None:
    rng = np.random.default_rng(1)
    for looks in (1.0, 4.0, 10.7):
        x = rng.gamma(looks, 1 / looks, 200_000)
        assert moment_enl(x) == pytest.approx(looks, rel=0.03)


def test_box_table_and_auto_estimate(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    stack = np.stack([rng.gamma(6, 0.02 / 6, (100, 100)),
                      rng.gamma(2, 0.002 / 2, (100, 100))]).astype(np.float32)
    valid = np.ones((100, 100), bool)
    valid[:20, :20] = False                 # boxes touching this aren't used
    stack[:, 70, 70] *= 1000                # a berg lowers its own box's ENL
    t = box_table(stack, valid, 20)
    assert len(t) == 24                     # 25 boxes minus the masked one
    assert t["enl_HH"].median() == pytest.approx(6, rel=0.15)

    with rasterio.open(tmp_path / "st.tif", "w", driver="GTiff", height=100, width=100, count=2,
                       dtype="float32", crs="EPSG:32622",
                       transform=from_origin(600_000, 5_600_000, 40, 40)) as dst:
        dst.write(stack)
    with rasterio.open(tmp_path / "va.tif", "w", driver="GTiff", height=100, width=100, count=1,
                       dtype="uint8", crs="EPSG:32622",
                       transform=from_origin(600_000, 5_600_000, 40, 40)) as dst:
        dst.write(valid.astype(np.uint8), 1)
    r = estimate_enl(tmp_path / "st.tif", tmp_path / "va.tif", 800, "auto", "min",
                     sidecar=tmp_path / "enl.json")
    assert r["box_px"] == [20, 20] and r["box_rowcol"] != [60, 60]
    assert r["enl"]["HH"] == pytest.approx(6, rel=0.2)
    assert r["enl"]["HV"] == pytest.approx(2, rel=0.2)
    assert r["enl_used"] == r["enl"]["HV"]
    assert (tmp_path / "enl.json").is_file()


def test_combine() -> None:
    e = {"HH": 6.0, "HV": 2.0}
    assert (combine(e, "min"), combine(e, "mean"), combine(e, "HH")) == (2.0, 4.0, 6.0)
    with pytest.raises(ValueError):
        combine(e, "max")


def test_graph_variant_drops_thermal_noise_node(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    from iceberg_sar.benchmark.preprocess import graph_variant

    graph = Path(__file__).resolve().parents[1] / "benchmarks/faerch2023/s1_ew_preprocess.xml"
    assert graph_variant({"remove_thermal_noise": True}, graph, tmp_path) == graph
    root = ET.parse(graph_variant({"remove_thermal_noise": False}, graph, tmp_path)).getroot()
    ops = {n.get("id"): n for n in root.findall("node")}
    assert "ThermalNoiseRemoval" not in ops
    calib_src = ops["Calibration"].find("sources/sourceProduct")
    assert calib_src.get("refid") == "Remove-GRD-Border-Noise"


def test_graph_variant_snap_noclip_keeps_node_and_turns_clipping_off(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    from iceberg_sar.benchmark.preprocess import SNAP_NOCLIP, graph_variant

    graph = Path(__file__).resolve().parents[1] / "benchmarks/faerch2023/s1_ew_preprocess.xml"
    out = graph_variant({"remove_thermal_noise": True, "tnr_variant": SNAP_NOCLIP}, graph, tmp_path)
    assert out != graph
    root = ET.parse(out).getroot()
    tnr = next(n for n in root.findall("node") if n.findtext("operator") == "ThermalNoiseRemoval")
    assert tnr.findtext("parameters/removeThermalNoise") == "true"
    assert tnr.findtext("parameters/clipNegativeValues") == "false"
    calib_src = next(n for n in root.findall("node") if n.get("id") == "Calibration")
    assert calib_src.find("sources/sourceProduct").get("refid") == tnr.get("id")


def test_split_snap_output_keeps_negative_pixels_inside_swath(tmp_path: Path) -> None:
    from iceberg_sar.benchmark.preprocess import split_snap_output

    hh = np.full((4, 4), 0.01, np.float32)
    hh[:, 0] = 0.0                     # outside the swath (SNAP writes 0)
    hh[1, 1] = -1e-4                   # below the noise floor, kept negative
    hv = np.full((4, 4), -2e-5, np.float32)
    ang = np.full((4, 4), 30.0, np.float32)
    src = tmp_path / "snap.tif"
    with rasterio.open(src, "w", driver="GTiff", height=4, width=4, count=3, dtype="float32",
                       crs="EPSG:32620", transform=from_origin(0, 160, 40, 40)) as dst:
        dst.write(np.stack([hh, hv, ang]))
    info = split_snap_output(src, tmp_path / "s.tif", tmp_path / "i.tif", tmp_path / "m.tif")
    assert info["valid_px"] == 12
    with rasterio.open(tmp_path / "s.tif") as f:
        assert f.read(1)[1, 1] == pytest.approx(-1e-4)
    assert info["hh_negative_in_swath"] == pytest.approx(1 / 12)
    assert info["hv_negative_in_swath"] == 1.0
