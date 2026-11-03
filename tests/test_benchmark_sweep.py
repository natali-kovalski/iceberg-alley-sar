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
    V = sweep.CacfarVariant
    assert p.cacfar_tags == {"cacfar-nolength": V(False, "paper"), "cacfar": V(True, "paper"),
                             "cacfar-nolength-ownwin": V(False, "own"),
                             "cacfar-ownwin": V(True, "own"),
                             "cacfar-nolength-and": V(False, "paper", "and")}
    assert p.cacfar_windows == {"inner_m": 360, "outer_m": 520}
    assert list(replace(p, cacfar_variants=[V(False)]).cacfar_tags) == ["cacfar-nolength"]


def test_10m_tags_only_for_configured_pfas() -> None:
    p = sweep.SweepParams.from_config(CFG)
    tags = p.cacfar_10m_tags
    assert list(tags) == [sweep.CACFAR_10M, sweep.CACFAR_10M_AND]
    assert tags[sweep.CACFAR_10M_AND] == sorted(tags[sweep.CACFAR_10M_AND])
    assert list(replace(p, cacfar_10m_and_pfas=[]).cacfar_10m_tags) == [sweep.CACFAR_10M]
    assert replace(p, cacfar_10m_pfas=[], cacfar_10m_and_pfas=[]).cacfar_10m_tags == {}


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
    V = sweep.CacfarVariant
    assert sweep.faerch_cacfar_tags(CFG) == {
        "cacfar": V(True), "cacfar-nolength": V(False),
        "cacfar-and": V(True, copol="and"), "cacfar-nolength-and": V(False, copol="and")}


def _hv_raster(path: Path) -> Path:
    rng = np.random.default_rng(1)
    with rasterio.open(path, "w", driver="GTiff", width=120, height=120, count=1,
                       dtype="float32", crs="EPSG:32622",
                       transform=from_origin(600_000, 5_600_000, 40, 40)) as dst:
        dst.write(rng.gamma(2.0, 0.005, (120, 120)).astype(np.float32), 1)
    return path


def test_cacfar_settings_paper_or_own_windows(tmp_path: Path) -> None:
    hv = _hv_raster(tmp_path / "hv.tif")
    _, own, _ = sweep.cacfar_settings(CFG, hv, length_rule=True)
    _, paper, _ = sweep.cacfar_settings(CFG, hv, length_rule=True,
                                        windows={"inner_m": 360, "outer_m": 520})
    assert (paper.guard_px, paper.background_px) == (4, 6)   # 9 / 13 px at 40 m
    assert (own.guard_px, own.background_px) == (2, 10)      # cfar.guard_m / background_m
    assert paper.enl == own.enl


def test_run_cacfar_redoes_a_variant_whose_windows_changed(tmp_path: Path, monkeypatch) -> None:
    """A run.json from other windows must not be reused (2b's tags changed meaning 2026-10-28)."""
    import geopandas as gpd

    hv = _hv_raster(tmp_path / "hv.tif")
    calls = []
    monkeypatch.setattr(sweep, "band_paths", lambda prep: {"HH": hv, "HV": hv})

    def fake_detect(paths, band, cfar, params):
        calls.append(cfar.pfa)
        return None, None, None

    monkeypatch.setattr(sweep, "detect_raster", fake_detect)
    monkeypatch.setattr(sweep, "to_geodataframe",
                        lambda *a: gpd.GeoDataFrame({"x": [1]}, crs="EPSG:32622",
                                                    geometry=gpd.points_from_xy([0], [0])))
    p = replace(sweep.SweepParams.from_config(CFG), pfas=[1e-9, 1e-5])
    prep = {"work": tmp_path, "scene": "S", "inc": None}
    v = sweep.CacfarVariant(False, "paper")
    sweep.run_cacfar(CFG, prep, p, v, None, False, lambda m: None)
    assert len(calls) == 2
    sweep.run_cacfar(CFG, prep, p, v, None, False, lambda m: None)   # same settings: cached
    assert len(calls) == 2
    meta_f = tmp_path / v.tag / f"{v.tag}_run.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    assert (meta["guard_px"], meta["background_px"], meta["windows"]) == (4, 6, "paper")
    meta["guard_px"], meta["background_px"] = 2, 10                 # as the old own-window run
    meta_f.write_text(json.dumps(meta), encoding="utf-8")
    sweep.run_cacfar(CFG, prep, p, v, None, False, lambda m: None)
    assert len(calls) == 4


def test_pool_sweep_sums_frames_and_drops_settings_one_frame_lacks() -> None:
    import pandas as pd

    from iceberg_sar.benchmark.iwcompare import pool_sweep

    t = pd.DataFrame({"scene": ["a", "a", "b"], "detector": ["g", "g", "g"],
                      "setting": [1e-9, 1e-5, 1e-9], "detections": [100, 300, 50],
                      "matched": [40, 60, 10], "truth": [80, 80, 20]})
    out = pool_sweep(t, 2)
    assert out[["setting", "detections", "matched", "truth"]].values.tolist() == [
        [1e-9, 150, 50, 100]]                       # 1e-5 only ran in frame a
    assert out["recall"].iloc[0] == pytest.approx(0.5)


def test_summary_table_reads_each_frame_at_its_own_label_multiple() -> None:
    import pandas as pd

    from iceberg_sar.benchmark.iwcompare import POOLED, pool_sweep, summary_table

    t = pd.DataFrame({"scene": ["a", "a", "b", "b"], "detector": ["g"] * 4,
                      "setting": [1e-9, 1e-5] * 2, "detections": [100, 400, 10, 40],
                      "matched": [40, 80, 5, 10], "truth": [100, 100, 10, 10]})
    t["recall"] = t["matched"] / t["truth"]
    s = summary_table(t, pool_sweep(t, 2), ["a", "b"], per_label=2)
    r = s.iloc[0]
    assert r["a"] == pytest.approx(0.6, abs=1e-3)   # 200 detections: half way 100..400 in log
    assert r["b"] == pytest.approx(0.75, abs=1e-3)  # 20 detections: half way 10..40
    assert r[POOLED] == pytest.approx((45 / 110 + 90 / 110) / 2, abs=1e-3)   # 220 of 110..440


def test_and_cfars_per_channel_pfa_and_own_enl(tmp_path: Path) -> None:
    hv = _hv_raster(tmp_path / "hv.tif")
    hh = tmp_path / "hh.tif"
    rng = np.random.default_rng(2)
    with rasterio.open(hv) as src:
        profile = src.profile
    with rasterio.open(hh, "w", **profile) as dst:
        dst.write(rng.gamma(8.0, 0.01, (120, 120)).astype(np.float32), 1)
    _, base, _ = sweep.cacfar_settings(CFG, hv, length_rule=False)
    hv_c, hh_c, enl_hh = sweep.and_cfars(CFG, hh, base, [1e-10, 1e-4])
    assert [c.pfa for c in hv_c] == pytest.approx([1e-5, 1e-2])   # sqrt: AND merge
    assert [c.pfa for c in hh_c] == pytest.approx([1e-5, 1e-2])
    assert all(c.enl == base.enl for c in hv_c) and all(c.enl == enl_hh for c in hh_c)
    assert 5 < enl_hh < 11 and enl_hh > 2 * base.enl              # gamma(8) vs gamma(2)


def test_old_run_json_without_copol_stays_cached(tmp_path: Path, monkeypatch) -> None:
    """run.json files from before 2c have no `copol` key; they ran the 6 dB check (db)."""
    import geopandas as gpd

    hv = _hv_raster(tmp_path / "hv.tif")
    calls = []
    monkeypatch.setattr(sweep, "band_paths", lambda prep: {"HH": hv, "HV": hv})
    monkeypatch.setattr(sweep, "detect_raster",
                        lambda paths, band, cfar, params: calls.append(1) or (None, None, None))
    monkeypatch.setattr(sweep, "to_geodataframe",
                        lambda *a: gpd.GeoDataFrame({"x": [1]}, crs="EPSG:32622",
                                                    geometry=gpd.points_from_xy([0], [0])))
    p = replace(sweep.SweepParams.from_config(CFG), pfas=[1e-9])
    prep = {"work": tmp_path, "scene": "S", "inc": None}
    v = sweep.CacfarVariant(False, "paper")
    sweep.run_cacfar(CFG, prep, p, v, None, False, lambda m: None)
    meta_f = tmp_path / v.tag / f"{v.tag}_run.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    del meta["copol"], meta["enl_copol"]
    meta_f.write_text(json.dumps(meta), encoding="utf-8")
    sweep.run_cacfar(CFG, prep, p, v, None, False, lambda m: None)
    assert len(calls) == 1
