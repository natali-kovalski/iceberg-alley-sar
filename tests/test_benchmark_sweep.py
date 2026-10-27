"""2b sweep: config parsing, band split for the CA-CFAR, and the cached-run check."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from iceberg_sar.benchmark import sweep
from iceberg_sar.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")


def test_params_from_config_sorted_and_tagged() -> None:
    p = sweep.SweepParams.from_config(CFG)
    assert p.pfas == sorted(p.pfas) and len(p.pfas) == 11
    assert p.pfas[0] == pytest.approx(1e-21) and p.pfas[-1] == pytest.approx(1e-1)
    assert p.cacfar_tags == {"cacfar-nolength": False, "cacfar": True}
    assert replace(p, cacfar_length_rules=[False]).cacfar_tags == {"cacfar-nolength": False}


def test_split_bands_masks_both(tmp_path: Path) -> None:
    img = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) + 1
    valid = np.ones((4, 5), np.uint8)
    valid[1, 2] = 0
    common = dict(driver="GTiff", height=4, width=5, crs="EPSG:32622",
                  transform=from_origin(600_000, 5_600_000, 40, 40))
    with rasterio.open(tmp_path / "s.tif", "w", count=2, dtype="float32", **common) as d:
        d.write(img)
    with rasterio.open(tmp_path / "m.tif", "w", count=1, dtype="uint8", **common) as d:
        d.write(valid, 1)
    out = sweep.split_bands(tmp_path / "s.tif", tmp_path / "m.tif",
                            {"HH": tmp_path / "hh.tif", "HV": tmp_path / "hv.tif"})
    for i, pol in enumerate(("HH", "HV")):
        with rasterio.open(out[pol]) as src:
            b = src.read(1)
            assert src.transform == common["transform"]
        assert np.isnan(b[1, 2])
        assert np.array_equal(b[valid > 0], img[i][valid > 0])


def test_reference_done_needs_same_settings_and_files(tmp_path: Path) -> None:
    p = replace(sweep.SweepParams.from_config(CFG), pfas=[1e-9, 1e-5])
    meta = {"pfas_requested": [1e-5, 1e-9], "max_objects": p.max_objects,
            "runs": [{"pfa": 1e-9}, {"pfa": 1e-5}]}
    (tmp_path / "gamma_run.json").write_text(json.dumps(meta), encoding="utf-8")
    assert sweep._reference_done(tmp_path, "gamma", p) is None   # GeoJSONs missing
    for tag in ("1e-09", "1e-05"):
        (tmp_path / f"gamma_pfa{tag}.geojson").write_text("{}", encoding="utf-8")
    assert set(sweep._reference_done(tmp_path, "gamma", p)) == {1e-9, 1e-5}
    assert sweep._reference_done(tmp_path, "gamma", replace(p, max_objects=10)) is None
    assert sweep._reference_done(tmp_path, "gamma", replace(p, pfas=[1e-9])) is None


# --- scoring (iwcompare) ---------------------------------------------------------------------

def test_recall_at_interpolates_in_log_counts_without_extrapolating() -> None:
    import pandas as pd

    from iceberg_sar.benchmark.iwcompare import recall_at

    curve = pd.DataFrame({"detections": [100, 1000, 0], "recall": [0.4, 0.8, 0.0]})
    r = recall_at(curve, [50, 100, 316, 1000, 2000])
    assert np.isnan(r[0]) and np.isnan(r[-1])
    assert r[1] == pytest.approx(0.4) and r[3] == pytest.approx(0.8)
    assert r[2] == pytest.approx(0.6, abs=1e-3)   # half way in log10


def test_recall_at_takes_best_recall_for_repeated_counts() -> None:
    import pandas as pd

    from iceberg_sar.benchmark.iwcompare import recall_at

    curve = pd.DataFrame({"detections": [100, 100, 400], "recall": [0.3, 0.5, 0.7]})
    assert recall_at(curve, [100])[0] == pytest.approx(0.5)


def test_faerch_params_21_levels_and_table1_enl() -> None:
    p = sweep.FaerchSweepParams.from_config(CFG)
    assert len(p.pfas) == 21 and p.pfas[0] == 1e-21 and p.pfas[-1] == 1e-1
    assert p.enl == pytest.approx(10.7)


def test_pfa_status_resumes_and_stops_at_cap(tmp_path: Path) -> None:
    from iceberg_sar.benchmark.adapter import pfa_tag

    pfas = [1e-9, 1e-7, 1e-5, 1e-3]

    def done(pfa: float, n: int) -> None:
        (tmp_path / f"g_pfa{pfa_tag(pfa)}.json").write_text(
            json.dumps({"pfa": pfa, "objects": n, "outlier_px": n, "runtime_s": 1.0,
                        "raster": f"g_pfa{pfa_tag(pfa)}.tif"}), encoding="utf-8")

    done(1e-7, 10)
    finished, todo = sweep.pfa_status(tmp_path, "g", pfas, max_objects=100)
    assert list(finished) == [1e-7] and todo == [1e-9, 1e-5, 1e-3]
    done(1e-5, 500)   # over the cap: nothing above it runs
    finished, todo = sweep.pfa_status(tmp_path, "g", pfas, max_objects=100)
    assert sorted(finished) == [1e-7, 1e-5] and todo == [1e-9]


def test_cap_runs_keeps_the_first_pfa_over_the_cap() -> None:
    assert sweep.cap_runs([10, 50, 200, 900], max_objects=100) == 3
    assert sweep.cap_runs([10, 50], max_objects=100) == 2


def test_faerch_cacfar_settings_paper_windows_and_length_rule(tmp_path: Path) -> None:
    from iceberg_sar.detections import NO_STRUCTURE_LIMIT

    rng = np.random.default_rng(0)
    hv = tmp_path / "hv.tif"
    with rasterio.open(hv, "w", driver="GTiff", width=120, height=120, count=1,
                       dtype="float32", crs="EPSG:32620",
                       transform=from_origin(500_000, 6_700_000, 40, 40)) as dst:
        dst.write(rng.gamma(10.0, 0.1, (120, 120)).astype(np.float32), 1)
    pixel_m, cfar, params = sweep.faerch_cacfar_settings(CFG, hv, length_rule=True)
    assert pixel_m == 40 and (cfar.guard_px, cfar.background_px) == (4, 6)   # 9 / 13 px
    assert params.max_structure_px == 8                                      # 300 m at 40 m
    assert 7 < cfar.enl < 13                                                 # gamma(L=10) speckle
    _, _, off = sweep.faerch_cacfar_settings(CFG, hv, length_rule=False)
    assert off.max_structure_px == NO_STRUCTURE_LIMIT
    assert sweep.faerch_cacfar_tags(CFG) == {"cacfar": True, "cacfar-nolength": False}
