"""Revision C11b: step my CA-CFAR towards the library's gamma detector, one change at a time.

On the Færch EW scene (noTNR stack, the reproduction run), starting from my HH AND HV CA-CFAR
without the length rule (2c's `cacfar-nolength-and`, the paper's 9 / 13 px windows):
- `enl`: ENL fixed at Table 1's 10.7 for both bands (gamma takes it as given), instead of
  measured per band;
- `ring`: the library's circular clutter ring (offsets 4 <= r <= 6 px, 68 px) instead of my
  square ring (88 px);
- `objects`: the library's object handling: 4-connected components of 2-500 px, no merging of
  cores by grown structure, no edge buffer.
Each change alone and cumulatively; every run swept over the 21 PFAs, scored like `bench-score`
(same domain, labels, matching), best F next to gamma's. What remains after all three is the
CA-CFAR vs gamma statistics themselves (the library fits the gamma threshold numerically) and
smaller details listed in `docs/cacfar-validation.md`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from iceberg_sar.benchmark import score as sc
from iceberg_sar.config import Config

STEPS = {   # name -> changes applied
    "mine": (),
    "enl": ("enl",),
    "ring": ("ring",),
    "objects": ("objects",),
    "enl+ring": ("enl", "ring"),
    "enl+ring+objects": ("enl", "ring", "objects"),
}


@dataclass(frozen=True)
class AblationParams:
    out_dir: Path
    report: Path
    max_object_px: int
    min_object_px: int

    @classmethod
    def from_config(cls, cfg: Config) -> AblationParams:
        a = cfg.section("benchmark")["ablation"]
        b = cfg.section("benchmark")
        return cls(out_dir=cfg.root / a["out_dir"], report=cfg.root / a["report"],
                   max_object_px=int(b["max_object_px"]), min_object_px=int(b["minsize_px"]))


def step_settings(cfg: Config, changes: tuple[str, ...], ap: AblationParams):
    """(HV cfars, HH cfars, detection params, ENLs) for one step, per total PFA of the sweep."""
    from iceberg_sar.benchmark import adapter
    from iceberg_sar.benchmark.reproduce import scene_paths
    from iceberg_sar.benchmark.sweep import (
        BANDS,
        FaerchSweepParams,
        and_cfars,
        faerch_cacfar_settings,
        split_bands,
    )

    fp = FaerchSweepParams.from_config(cfg)
    sp = scene_paths(cfg)
    stem = f"{sp.stem}_{adapter.pixel_size_m(sp.stack):g}m"
    paths = {pol: sp.stack.parent / f"{stem}_{pol}_masked.tif" for pol in BANDS}
    if not all(f.is_file() for f in paths.values()):
        split_bands(sp.stack, sp.valid, paths)
    _, base, params = faerch_cacfar_settings(cfg, paths["HV"], length_rule=False)
    if "ring" in changes:
        base = replace(base, ring_shape="circle")
    hv, hh, enl_hh = and_cfars(cfg, paths["HH"], base, fp.pfas)
    enls = {"HV": base.enl, "HH": enl_hh}
    if "enl" in changes:
        hv = [replace(c, enl=fp.enl) for c in hv]
        hh = [replace(c, enl=fp.enl) for c in hh]
        enls = {"HV": fp.enl, "HH": fp.enl}
    if "ring" in changes:
        hh = [replace(c, ring_shape="circle") for c in hh]
    params = replace(params, copol_min_contrast_db=None)
    if "objects" in changes:
        params = replace(params, merge_by_structure=False, connectivity=4, edge_buffer_px=0,
                         min_area_px=ap.min_object_px, max_area_px=ap.max_object_px)
    return paths, hv, hh, params, enls, fp, sp


def run_step(cfg: Config, name: str, ap: AblationParams, labels: pd.DataFrame,
             unknown: np.ndarray, domain, log: Callable[[str], None]) -> pd.DataFrame:
    """Sweep one step (cached as a CSV of scores) and score every PFA."""
    from iceberg_sar.detections import detect_raster_multi, to_geodataframe

    out = ap.out_dir / f"{name}.csv"
    if out.is_file():
        log(f"{name}: cached")
        return pd.read_csv(out)
    paths, hv, hh, params, enls, fp, sp = step_settings(cfg, STEPS[name], ap)
    fc = cfg.section("benchmark")["faerch2023"]
    tables, transform, crs = detect_raster_multi(paths, "HV", hv, params, hh)
    rows = []
    for pfa, t in zip(fp.pfas, tables, strict=True):
        row = {"step": name, "pfa": pfa, "objects": len(t), "over_cap": len(t) > fp.max_objects,
               "enl_hv": enls["HV"], "enl_hh": enls["HH"], "ring_px": hv[0].ring_pixels}
        if not row["over_cap"]:
            if len(t):
                g = to_geodataframe(t, transform, crs, sp.product, sp.inc).to_crs(fc["crs"])
                xy, _ = sc.detections_xy_from(g, domain, unknown,
                                              float(fc["score"]["unknown_ignore_m"]))
            else:
                xy = np.empty((0, 2))
            row |= sc.score_points(xy, labels)
        rows.append(row)
    df = pd.DataFrame(rows)
    ap.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    b = sc.optimum(df.assign(detector=name))
    log(f"{name}: best F {b.f.iloc[0]:.3f} at {b.pfa.iloc[0]:.0e} (ENL HV {enls['HV']:.2f}, "
        f"HH {enls['HH']:.2f}, ring {hv[0].ring_pixels} px)")
    return df


def run_ablation(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Path]:
    ap = AblationParams.from_config(cfg)
    fc = cfg.section("benchmark")["faerch2023"]
    labels, unknown = sc.load_labels(cfg, fc["labels_source"])
    domain = sc.eval_domain(cfg, fc["score"]["aoi"])
    lab = sc.apply_match_settings(sc.labels_in_domain(labels, domain), sc.MatchSettings())
    parts = [run_step(cfg, name, ap, lab, unknown, domain, log) for name in STEPS]
    allr = pd.concat(parts, ignore_index=True)
    base = cfg.root / fc["score"]["table_csv"]
    ref = pd.read_csv(base.with_name(f"{base.stem}_{sc._variant(cfg)}.csv"))
    opt_ref = sc.optimum(ref)
    gamma = opt_ref.loc["gamma"]
    sweeps = pd.concat([ref[ref.detector.isin([*PRIOR_STEPS, "gamma"])],
                        allr.rename(columns={"step": "detector"})], ignore_index=True)
    band = float(cfg.section("benchmark")["optimum_band_f"])
    spread = sc.optimum_spread(sweeps, band)
    best = sc.optimum(allr.rename(columns={"step": "detector"})).reindex(list(STEPS))
    # 2a's 6 dB-check rows (HV CFAR only, same windows / ENL as "mine"), read from bench-score
    prior = opt_ref.reindex(list(PRIOR_STEPS)).assign(
        enl_hv=best.enl_hv.loc["mine"], enl_hh=np.nan, ring_px=best.ring_px.loc["mine"])
    best = pd.concat([prior, best])
    t = pd.DataFrame({"step": best.index, "pfa": best.pfa,
                      "pfa_lo": spread.pfa_lo.reindex(best.index),
                      "pfa_hi": spread.pfa_hi.reindex(best.index),
                      "edge": spread.edge.reindex(best.index), "TP": best.TP, "FP": best.FP,
                      "precision": best.precision, "recall": best.recall, "f": best.f,
                      "gap_to_gamma": best.f - gamma.f, "enl_hv": best.enl_hv,
                      "enl_hh": best.enl_hh, "ring_px": best.ring_px})
    km2 = domain.area.area / 1e6
    t["fp_per_1000km2"] = t.FP / km2 * 1000
    t.to_csv(ap.out_dir / "ablation_summary.csv", index=False)
    fa = false_alarm_table(sweeps, km2, _pixel_m(cfg))
    fa.to_csv(ap.out_dir / "false_alarms_vs_pfa.csv", index=False)
    d = sweeps[sweeps.detector == "cacfar-nolength"]
    pl = spread.loc["cacfar-nolength"]
    fp_at = {k: int(d[sc.same_pfa(d.pfa, pl[k])].FP.iloc[0]) for k in ("pfa_lo", "pfa_hi")}
    plateau = {**pl.to_dict(), "fp_lo": fp_at["pfa_lo"], "fp_hi": fp_at["pfa_hi"]}
    ap.report.write_text(report(t, gamma, len(lab), band, fa, km2, plateau), encoding="utf-8")
    (ap.out_dir / "gamma.json").write_text(json.dumps({"f": gamma.f, "pfa": gamma.pfa}),
                                           encoding="utf-8")
    return {"summary": ap.out_dir / "ablation_summary.csv", "report": ap.report}


PRIOR_STEPS = ("cacfar", "cacfar-nolength")   # bench-score detector tags (Milestone 2a)
FA_PFAS = (1e-9, 1e-7, 1e-5, 1e-3, 1e-1)        # columns of the false-alarm table
FA_ROWS = {"cacfar-nolength": "HV + 6 dB HH check (row 2)",
           "mine": "HH AND HV (row 3)", "gamma": "Gamma (library)"}


def _pixel_m(cfg: Config) -> float:
    from iceberg_sar.benchmark import adapter
    from iceberg_sar.benchmark.reproduce import scene_paths

    return float(adapter.pixel_size_m(scene_paths(cfg).stack))


def false_alarm_table(sweeps: pd.DataFrame, area_km2: float, pixel_m: float) -> pd.DataFrame:
    """Unmatched detections per 1,000 km² of scoring area at a few total PFAs, for the rows in
    `FA_ROWS`, plus the nominal count (PFA x pixels in 1,000 km²). NaN where the PFA wasn't
    swept, inf where the run was over the object cap or the sweep stopped there."""
    rows = []
    for det, name in FA_ROWS.items():
        d = sweeps[sweeps.detector == det]
        row: dict[str, object] = {"row": name}
        for pfa in FA_PFAS:
            r = d[sc.same_pfa(d.pfa, pfa)]
            capped = d.over_cap.astype(bool) & (d.pfa < pfa)   # the sweep stopped below
            row[pfa] = (np.inf if capped.any() or (len(r) and bool(r.over_cap.iloc[0]))
                        else np.nan if r.empty else float(r.FP.iloc[0]) / area_km2 * 1000)
        rows.append(row)
    px = 1e9 / pixel_m ** 2
    rows.append({"row": f"Nominal: PFA x {px:,.0f} pixels", **{p: p * px for p in FA_PFAS}})
    return pd.DataFrame(rows)


def _fa_cell(v: float) -> str:
    if np.isnan(v):
        return "–"
    if np.isinf(v):
        return "over cap"
    return f"{v:.2g}" if v >= 0.01 else f"{v:.0e}".replace("e-0", "e-")

LABELS = {"cacfar": "My CA-CFAR as in 2a: HV + 6 dB HH check, length rule on",
          "cacfar-nolength": "+ length rule off",
          "mine": "+ HH AND HV instead of the 6 dB check (start of the sweeps below)",
          "enl": "+ fixed ENL 10.7 (both bands)", "ring": "+ circular ring (68 px)",
          "objects": "+ library object handling", "enl+ring": "+ fixed ENL + circular ring",
          "enl+ring+objects": "+ all three"}


def report(t: pd.DataFrame, gamma: pd.Series, n_labels: int, band: float,
           fa: pd.DataFrame, area_km2: float, plateau: dict) -> str:
    """`plateau`: row 2's (`cacfar-nolength`) optimum spread plus its FP at both ends."""
    e = lambda p: f"{p:.0e}".replace("e-0", "e-")  # noqa: E731
    L = ["# My CA-CFAR stepped towards gamma (revision C11b)", "",
         "Generated by `python -m iceberg_sar.cli bench-ablation`. Færch EW scene, noTNR stack, "
         f"{n_labels} labels in the scoring area, scored like `bench-score`. Each row is one "
         "21-PFA sweep; F at its best PFA. The first two rows are Milestone 2a's runs, read "
         "from the `bench-score` table (`cacfar`, `cacfar-nolength`); the HH test there is the "
         "fixed 6 dB contrast check, so it has no HH ENL. From the third row (= 2c's "
         "`cacfar-nolength-and`) the rows are this command's sweeps: single changes first, "
         "then cumulative.", "",
         f"Gamma (library, HH AND HV, ENL 10.7): **F {gamma.f:.3f}** at {gamma.pfa:.0e}.", "",
         f"| Step | ENL HV / HH | Ring px | Best PFA | PFA range, F within {band:g} | TP / FP | "
         "FP / 1,000 km² | P | R | F | F − gamma |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        enl_hh = "–" if pd.isna(r.enl_hh) else f"{r.enl_hh:.1f}"
        L.append(f"| {LABELS[r.step]} | {r.enl_hv:.1f} / {enl_hh} | {int(r.ring_px)} | "
                 f"{r.pfa:.0e} | {sc.spread_text(pd.Series(r._asdict()))} | "
                 f"{int(r.TP)} / {int(r.FP)} | {r.fp_per_1000km2:.1f} | {r.precision:.3f} | "
                 f"{r.recall:.3f} | {r.f:.3f} | {r.gap_to_gamma:+.3f} |")
    L += ["", f"The PFA range shows how sharp each optimum is: F stays within {band:g} of its "
          "best over that range, so a wide range means the exact best PFA is noise. FP / "
          f"1,000 km² is the unmatched detections at the best PFA over the {area_km2:,.0f} km² "
          "scoring area; it is an upper bound on false alarms, since unlabelled bergs and other "
          "real targets count as FP.", "",
          f"Row 2's best PFA ({e(plateau['pfa'])}) is nominal only. With the 6 dB HH check, PFA "
          f"has stopped controlling the detector: a real false-alarm rate of {plateau['pfa']:g} "
          f"would flag {plateau['pfa']:.0%} of the water pixels, yet from {e(plateau['pfa_lo'])} "
          f"to {e(plateau['pfa_hi'])} FP grows only from {plateau['fp_lo']} to "
          f"{plateau['fp_hi']} and F stays within {band:g}"
          + (", up to the last PFA swept" if plateau["edge"] == "high" else "") + ". The "
          "fixed HH contrast threshold limits detections there, not the HV CFAR (2c: it caps "
          "recall). Any PFA on that plateau gives the same F, so the step from row 2 to row 3 "
          "does not depend on which PFA is called best.", "",
          "Unmatched detections per 1,000 km² across the sweep (total PFA; for the AND rows each "
          "band runs at its square root). How FP grows with PFA shows whether PFA is still "
          "the setting that limits detections; the absolute levels are not comparable with the "
          "nominal row, because FP includes real unlabelled targets and the sea clutter is not "
          "exactly gamma-distributed:", "",
          "| Row | " + " | ".join(f"{p:.0e}".replace("e-0", "e-") for p in FA_PFAS) + " |",
          "|---|" + "---|" * len(FA_PFAS)]
    for r in fa.itertuples(index=False):
        L.append(f"| {r[0]} | " + " | ".join(_fa_cell(v) for v in r[1:]) + " |")
    L += ["", "Replacing the 6 dB HH check with HH AND HV (row 2 to row 3) "
          "is the only change larger than the noise. Of the remaining "
          "pipeline differences, ENL (fixed vs measured), ring shape and object handling move "
          "F by at most ±0.013 each, and fixing the ENL at 10.7 moves the best PFA one decade, "
          "onto gamma's optimum (a statement about nominal PFAs: both detectors then assume "
          "the same clutter statistics). What is left (the library fits gamma's threshold "
          "numerically; my CA-CFAR uses the closed-form F-distribution factor) is not "
          "separated here.",
          "", "Bootstrap intervals for F on this scene are about ±0.04 (`docs/revision-"
          "robustness.md`), and paired differences between detectors about ±0.02; read gaps "
          "below that as not distinguishable.", ""]
    return "\n".join(L)
