"""Benchmark adapter: window conversion, object table, and (if `cfar-bench` exists) the runner."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from iceberg_sar.benchmark import adapter
from iceberg_sar.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
BENCH = CFG.section("benchmark")
SCENE = "S1A_IW_20190429T094108_DHP_RTC40_G_gpuned_TEST"


@pytest.mark.parametrize(
    ("metres", "pixel_m", "expected"),
    [(360, 40, 9), (520, 40, 13), (360, 10, 37), (520, 10, 53), (120, 40, 3), (2280, 40, 57),
     (400, 40, 11), (350, 40, 9), (10, 40, 1)],
)
def test_odd_px(metres: float, pixel_m: float, expected: int) -> None:
    assert adapter.odd_px(metres, pixel_m) == expected


def test_paper_windows_at_40m() -> None:
    s = adapter.detector_settings("gamma", 40.0, BENCH)
    assert (s.wi_px, s.wo_px, s.allow_window_crop) == (9, 13, False)


def test_window_over_limit_fails_loudly_at_10m() -> None:
    with pytest.raises(adapter.WindowLimitError, match="53 px"):
        adapter.detector_settings("gamma", 10.0, BENCH)


def test_idpolrad_crop_is_opt_in() -> None:
    s = adapter.detector_settings("idpolrad", 40.0, BENCH)
    assert (s.wi_px, s.wo_px, s.allow_window_crop) == (3, 57, True)
    strict = BENCH | {"detectors": {"idpolrad": {"inner_m": 120, "outer_m": 2280}}}
    with pytest.raises(adapter.WindowLimitError):
        adapter.detector_settings("idpolrad", 40.0, strict)


def test_unknown_detector() -> None:
    with pytest.raises(ValueError, match="Unknown detector"):
        adapter.detector_settings("ca", 40.0, BENCH)


def test_object_table_area_length_and_size_filter() -> None:
    det = np.zeros((40, 40), bool)
    det[5, 5:10] = True           # horizontal line, 5 px
    det[20:22, 20:22] = True      # 2x2 square
    det[30, 30] = det[31, 31] = True  # diagonal pair: two objects with 4-connectivity
    hh = np.full(det.shape, 0.01, np.float32)
    hh[20, 20] = 1.0
    bands = {"HH": hh, "HV": np.full(det.shape, 0.001, np.float32)}
    t = adapter.object_table(det, bands, connectivity=4, max_object_px=500)
    assert sorted(t["area_px"]) == [1, 1, 4, 5]
    line = t[t["area_px"] == 5].iloc[0]
    assert line["length_px"] == pytest.approx(5.0)
    assert (line["row"], line["col"]) == (5.0, 7.0)
    sq = t[t["area_px"] == 4].iloc[0]
    assert sq["peak_db_HH"] == pytest.approx(0.0)
    assert sq["mean_db_HH"] == pytest.approx(10 * np.log10((1.0 + 3 * 0.01) / 4), abs=1e-4)
    assert t["mean_db_HV"].to_numpy() == pytest.approx(-30.0)
    assert t["background_db"].isna().all() and t["contrast_db"].isna().all()

    t8 = adapter.object_table(det, bands, connectivity=8, max_object_px=500)
    assert sorted(t8["area_px"]) == [2, 4, 5]
    assert sorted(adapter.object_table(det, bands, 4, max_object_px=4)["area_px"]) == [1, 1, 4]


def test_runner_command_flags(tmp_path: Path) -> None:
    s = adapter.DetectorSettings("idpolrad", 3, 57, True)
    cmd = adapter.runner_command(BENCH, ROOT, s, tmp_path / "i.tif", tmp_path / "m.tif",
                                 [1e-9, 1e-3], 4.25, tmp_path)
    assert cmd[0].endswith("python.exe") and cmd[1].endswith("run_cfar.py")
    assert cmd.count("--pfa") == 2 and "--allow-window-crop" in cmd
    assert cmd[cmd.index("--wo") + 1] == "57" and cmd[cmd.index("--method") + 1] == "AND"
    assert "--enl-hh" not in cmd


def test_per_band_enl_flags_only_for_gamma_and_k(tmp_path: Path) -> None:
    bands = {"HH": 5.964, "HV": 1.823}
    for name, expected in [("gamma", True), ("k", True), ("wishart", False), ("nis", False)]:
        s = adapter.DetectorSettings(name, 9, 13)
        cmd = adapter.runner_command(BENCH, ROOT, s, tmp_path / "i.tif", tmp_path / "m.tif",
                                     [1e-9], 1.823, tmp_path, enl_bands=bands)
        assert ("--enl-hh" in cmd) is expected
        if expected:
            assert cmd[cmd.index("--enl-hh") + 1] == "5.964"
            assert cmd[cmd.index("--enl-hv") + 1] == "1.823"


# --- end to end through the external runner ------------------------------------------------

def _write(path: Path, data: np.ndarray, dtype: str) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[-2], width=data.shape[-1],
                       count=data.shape[0], dtype=dtype, crs="EPSG:32622",
                       transform=from_origin(600_000, 5_600_000, 40, 40)) as dst:
        dst.write(data.astype(dtype))
    return path


needs_env = pytest.mark.skipif(not (ROOT / BENCH["cfar_env_python"]).is_file(),
                               reason="cfar-bench env not installed")


def _scene(tmp_path: Path, enl: float = 8.0, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    img = rng.gamma(enl, 1 / enl, size=(2, 160, 160)).astype(np.float32)
    img[0] *= 0.02   # HH water ~ -17 dB
    img[1] *= 0.002  # HV water ~ -27 dB
    img[:, 80:82, 60:63] *= 300   # 2x3 px berg, ~25 dB above water in both bands
    valid = np.ones((1, 160, 160), np.uint8)
    valid[0, :, 120:] = 0         # masked "land" strip
    img[:, :, 120:] = 50.0        # very bright land: must not leak into rings or detections
    return (_write(tmp_path / "img.tif", img, "float32"),
            _write(tmp_path / "mask.tif", valid, "uint8"))


@needs_env
@pytest.mark.parametrize("name", ["gamma", "k"])
def test_per_band_path_matches_library_when_enls_are_equal(tmp_path: Path, name: str) -> None:
    """The runner's per-band path re-implements detector.run; with HH ENL = HV ENL it must
    give the library's raster exactly (dense speckle at a high PFA, so there is much to compare)."""
    raster, mask = _scene(tmp_path, enl=4.0, seed=3)
    single = adapter.run(name, raster, mask, [1e-3], enl=4.0, cfg=CFG,
                         out_dir=tmp_path / "single", scene_id=SCENE)
    banded = adapter.run(name, raster, mask, [1e-3], enl=4.0, enl_bands={"HH": 4.0, "HV": 4.0},
                         cfg=CFG, out_dir=tmp_path / "banded", scene_id=SCENE)
    tif = f"{name}_pfa1e-03.tif"
    with rasterio.open(tmp_path / "single" / tif) as a:
        da = a.read(1)
    with rasterio.open(tmp_path / "banded" / tif) as b:
        db = b.read(1)
    assert da.sum() > 5 and np.array_equal(da, db)
    assert len(single[1e-3]) == len(banded[1e-3])
    meta = (tmp_path / "banded" / f"{name}_run.json").read_text(encoding="utf-8")
    assert '"enl_mode": "per_band"' in meta


@needs_env
def test_run_gamma_end_to_end(tmp_path: Path) -> None:
    enl = 8.0
    raster, mask = _scene(tmp_path, enl)
    out = adapter.run("gamma", raster, mask, [1e-9], enl=enl, cfg=CFG,
                      out_dir=tmp_path / "out", scene_id=SCENE)
    gdf = out[1e-9]
    assert len(gdf) == 1
    berg = gdf.iloc[0]
    assert (round(berg["row"]), round(berg["col"])) == (80, 61)
    assert berg["area_px"] == 6 and berg["area_m2"] == 6 * 1600
    assert berg["detector"] == "gamma" and berg["pfa"] == 1e-9
    assert berg["pfa_per_channel"] == pytest.approx(10**-4.5)
    assert (tmp_path / "out" / "gamma_pfa1e-09.geojson").is_file()
    meta = (tmp_path / "out" / "gamma_run.json").read_text(encoding="utf-8")
    assert '"wo_px": 13' in meta and '"window_cropped": false' in meta
