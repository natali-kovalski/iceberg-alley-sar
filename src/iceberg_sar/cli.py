"""Command-line entry point: `python -m iceberg_sar.cli --help`."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer

from iceberg_sar import __version__
from iceberg_sar.config import DEFAULT_CONFIG, load_config

app = typer.Typer(
    help="Iceberg Alley SAR: detect icebergs in Sentinel-1 imagery.",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Path to config.yaml.")
]


@app.command()
def info(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Show version, config location, AOI, and data directories."""
    cfg = load_config(config)
    search = cfg.section("search")
    typer.echo(f"iceberg-sar {__version__}")
    typer.echo(f"config:     {cfg.root / Path(config).name}")
    aoi = cfg.path("aoi")
    typer.echo(f"aoi:        {aoi} ({'found' if aoi.is_file() else 'MISSING'})")
    typer.echo(f"dates:      {search['start_date']} to {search['end_date']}")
    for d in cfg.data_dirs:
        typer.echo(f"data dir:   {d} ({'exists' if d.is_dir() else 'not created'})")


@app.command("init-dirs")
def init_dirs(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Create the data/ directory tree from config.yaml."""
    cfg = load_config(config)
    for d in cfg.data_dirs:
        d.mkdir(parents=True, exist_ok=True)
        typer.echo(f"ok  {d}")


@app.command()
def search(
    config: ConfigOption = DEFAULT_CONFIG,
    pol: Annotated[str | None, typer.Option(help="Filter, e.g. 'HH+HV' or 'VV+VH'.")] = None,
) -> None:
    """List Sentinel-1 GRD scenes over the AOI and save them as GeoJSON footprints."""
    from iceberg_sar.search import save_scenes, search_scenes

    cfg = load_config(config)
    gdf = search_scenes(cfg)
    if pol:
        gdf = gdf[gdf.polarization == pol]
    if gdf.empty:
        typer.echo("No scenes found.")
        raise typer.Exit(1)
    s = cfg.section("search")
    out_name = f"scenes_{s['start_date']}_{s['end_date']}.geojson"
    out = save_scenes(gdf, cfg.path("outputs") / out_name)
    cols = ["scene", "beam_mode", "polarization", "flight_direction", "aoi_overlap"]
    typer.echo(gdf[cols].to_string())
    typer.echo(f"\n{len(gdf)} scenes -> {out}")


@app.command()
def order(granule: str, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Submit a HyP3 RTC job for one Sentinel-1 granule (skips if already ordered)."""
    from iceberg_sar.hyp3 import get_client, submit_rtc

    cfg = load_config(config)
    client = get_client()
    typer.echo(f"Credits remaining: {client.check_credits()}")
    batch = submit_rtc(client, granule, cfg)
    for job in batch:
        typer.echo(f"job {job.job_id}  status={job.status_code}  granule={granule}")


@app.command()
def download(
    config: ConfigOption = DEFAULT_CONFIG,
    wait: Annotated[bool, typer.Option(help="Wait for running jobs to finish.")] = True,
) -> None:
    """Download finished HyP3 RTC jobs into data/raw and unzip them."""
    from iceberg_sar.hyp3 import JOB_NAME, download_jobs, get_client

    cfg = load_config(config)
    client = get_client()
    batch = client.find_jobs(name=JOB_NAME)
    if wait and not batch.complete():
        batch = client.watch(batch)
    for d in download_jobs(batch, cfg.path("raw")):
        typer.echo(f"ok  {d}")


@app.command("land-mask")
def land_mask(
    config: ConfigOption = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option(help="Rebuild even if the GeoPackage exists.")] = False,
) -> None:
    """Download OSM land polygons and point rocks/islets (once), build the AOI land mask."""
    from iceberg_sar.landmask import build_land_mask

    cfg = load_config(config)
    typer.echo(f"ok  {build_land_mask(cfg, force=force)}")


@app.command()
def preprocess(
    product_dir: Annotated[Path, typer.Argument(help="Unzipped HyP3 RTC product directory.")],
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Mask land and nodata, write masked linear rasters and dB quicklook PNGs."""
    from iceberg_sar.landmask import build_land_mask, load_land
    from iceberg_sar.preprocess import preprocess_product

    cfg = load_config(config)
    land = load_land(build_land_mask(cfg))
    for kind, path in preprocess_product(product_dir, land, cfg).items():
        typer.echo(f"{kind:<12} {path}")


@app.command()
def detect(
    product_dir: Annotated[Path, typer.Argument(help="HyP3 RTC product directory (in data/raw).")],
    config: ConfigOption = DEFAULT_CONFIG,
    band: Annotated[str | None, typer.Option(help="Override cfar.band, e.g. HH or HV.")] = None,
) -> None:
    """Run CA-CFAR on a preprocessed scene; write detections GeoJSON and an overlay PNG."""
    import json

    from iceberg_sar.detections import detect_product

    cfg = load_config(config)
    outputs = detect_product(product_dir, cfg, band=band)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    typer.echo(
        f"band={summary['cfar_band']}  ENL={summary['enl']}  "
        f"threshold={summary['alpha']}x background ({summary['alpha_db']} dB)  "
        f"detections={summary['n_detections']}"
    )
    for kind, path in outputs.items():
        typer.echo(f"{kind:<12} {path}")


@app.command("ground-truth")
def ground_truth(
    day: Annotated[str, typer.Argument(help="Date, YYYY-MM-DD (e.g. the scene date).")],
    days_around: Annotated[int, typer.Option(help="Also fetch this many days before/after.")] = 1,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Download NAIS iceberg chart GIFs (IIP + Canadian Ice Service) around a date."""
    from datetime import date, timedelta

    from iceberg_sar.groundtruth import download_nais_chart

    cfg = load_config(config)
    url = cfg.section("ground_truth")["nais_url"]
    d0 = date.fromisoformat(day)
    out_dir = cfg.path("ground_truth")
    for k in range(-days_around, days_around + 1):
        typer.echo(f"ok  {download_nais_chart(d0 + timedelta(days=k), out_dir, url)}")


@app.command()
def validate(
    product_dir: Annotated[Path, typer.Argument(help="HyP3 RTC product directory (in data/raw).")],
    counts: Annotated[Path, typer.Option(help="Transcribed NAIS counts CSV.")],
    band: Annotated[str, typer.Option(help="Which detections to validate.")] = "HV",
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Compare SAR detections per 1-degree square with transcribed NAIS chart counts."""
    import geopandas as gpd

    from iceberg_sar.groundtruth import (
        compare_counts,
        load_degree_counts,
        valid_fraction_per_square,
    )

    cfg = load_config(config)
    name = Path(product_dir).name
    det = gpd.read_file(cfg.path("outputs") / "detections" / f"{name}_{band}_detections.geojson")
    masked = cfg.path("interim") / name / f"{name}_{band}_masked.tif"
    min_frac = cfg.section("ground_truth")["min_valid_fraction"]
    cmp = compare_counts(det, load_degree_counts(counts), valid_fraction_per_square(masked),
                         min_frac)
    shown = cmp[cmp.valid_fraction > 0].drop(columns="geometry")
    typer.echo(shown.to_string(index=False))
    used = cmp[cmp.compared]
    typer.echo()
    typer.echo(f"squares with >= {min_frac:.0%} coverage: SAR {used.sar_detections.sum()} "
               f"vs chart (coverage-scaled) {used.expected.sum():.1f}; "
               f"detections outside listed squares: {len(det) - cmp.sar_detections.sum()}")
    out = cfg.path("outputs") / "validation" / f"{name}_{band}_nais.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmp.to_file(out, driver="GeoJSON")
    typer.echo(f"ok  {out}")


DateOption = Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])]


@app.command()
def czml(
    config: ConfigOption = DEFAULT_CONFIG,
    band: Annotated[str, typer.Option(help="Which detections to export.")] = "HV",
    start: DateOption = None,
    end: DateOption = None,
) -> None:
    """Export scenes' detections, footprints and pack ice to one CZML for the viewer.

    Without --start/--end, every scene with detections at the configured resolution is used.
    """
    from iceberg_sar.export_czml import export_czml

    cfg = load_config(config)
    out, packets = export_czml(cfg, band, start and start.date(), end and end.date())
    for p in packets:
        if p["id"].startswith("scene/"):
            s = p["properties"]
            typer.echo(f"{s['start'][:10]}  {s['scene_id']}  detections={s['n_detections']}  "
                       f"open water={s['n_open_water']}  near ice={s['n_near_ice']}  "
                       f"static removed={s['n_static_removed']}")
    typer.echo(f"ok  {out}")


@app.command()
def run(
    config: ConfigOption = DEFAULT_CONFIG,
    start: DateOption = None,
    end: DateOption = None,
    dry_run: Annotated[bool, typer.Option(help="Search and show the plan; order nothing.")] = False,
    force: Annotated[bool, typer.Option(help="Redo preprocessing and detection.")] = False,
) -> None:
    """End to end for a date range: search, HyP3 RTC, masks, CFAR, CZML for the viewer.

    Dates default to `search.start_date` / `end_date`; scene selection is set in `run:`.
    The CZML includes every scene processed so far, not only this date range.
    Already downloaded products and existing outputs are reused.
    """
    from iceberg_sar.pipeline import run_pipeline

    cfg = load_config(config)
    s = cfg.section("search")
    d0 = start.date() if start else date.fromisoformat(str(s["start_date"]))
    d1 = end.date() if end else date.fromisoformat(str(s["end_date"]))
    out = run_pipeline(cfg, d0, d1, dry_run=dry_run, force=force, log=typer.echo)
    if out:
        typer.echo("view it:  cd viewer; npm install; npm run dev")


@app.command("train-classifier")
def train_classifier(
    config: ConfigOption = DEFAULT_CONFIG,
    variant: Annotated[str | None, typer.Option(help="Train only this variant, e.g. hh.")] = None,
) -> None:
    """Train the iceberg/ship CNN on the Kaggle chips (stratified k-fold CV)."""
    from iceberg_sar.classify.dataset import load_kaggle
    from iceberg_sar.classify.train import TrainParams, train_cv

    cfg = load_config(config)
    c = cfg.section("classifier")
    data = load_kaggle(cfg.path("kaggle"))
    names = [variant] if variant else list(c["variants"])
    for name in names:
        params = TrainParams(
            bands=tuple(c["variants"][name]), use_inc=bool(c["use_incidence_angle"]),
            folds=int(c["folds"]), epochs=int(c["epochs"]), batch_size=int(c["batch_size"]),
            learning_rate=float(c["learning_rate"]), weight_decay=float(c["weight_decay"]),
            width=int(c["width"]), dropout=float(c["dropout"]), seed=int(c["seed"]),
        )
        out = cfg.path("models") / "classifier" / name
        m = train_cv(data, params, out)
        typer.echo(f"{name:<6} bands={'+'.join(params.bands)}  OOF log loss={m['oof_log_loss']}  "
                   f"accuracy={m['oof_accuracy']}  (prior log loss {m['prior_log_loss']}, "
                   f"{m['train_seconds']} s)")
        typer.echo(f"ok  {out}")


@app.command()
def classify(
    product_dir: Annotated[Path, typer.Argument(help="HyP3 RTC product directory (in data/raw).")],
    config: ConfigOption = DEFAULT_CONFIG,
    band: Annotated[str, typer.Option(help="Which detections to classify.")] = "HV",
) -> None:
    """Add iceberg_prob to a scene's detections (experimental; see README on trusting it)."""
    import json

    from iceberg_sar.classify.apply import classify_product

    cfg = load_config(config)
    outputs = classify_product(product_dir, cfg, band)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    typer.echo(json.dumps(summary, indent=2))
    for kind, path in outputs.items():
        typer.echo(f"{kind:<12} {path}")


@app.command("iip-sightings")
def iip_sightings(
    seasons: Annotated[list[int], typer.Argument(help="Seasons, e.g. 2019 (Oct 2018-Sep 2019).")],
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Download IIP Iceberg Sightings Database seasons (NSIDC G00807, up to 2021)."""
    from iceberg_sar.iip import download_season

    cfg = load_config(config)
    for season in seasons:
        out_dir = cfg.root / cfg.section("validation")["iip_dir"]
        typer.echo(f"ok  {download_season(season, out_dir)}")


@app.command("s2-targets")
def s2_targets(
    product_dir: Annotated[Path, typer.Argument(help="HyP3 RTC product directory (in data/raw).")],
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Extract Sentinel-2 optical targets on the scene's date over its footprint."""
    from iceberg_sar.detections import scene_timestamp
    from iceberg_sar.pointval import scene_footprint
    from iceberg_sar.sentinel2 import OpticalParams, extract_for_area

    cfg = load_config(config)
    name = Path(product_dir).name
    v = cfg.section("validation")
    targets, clear, log = extract_for_area(
        scene_footprint(cfg.path("raw"), name).iloc[0], scene_timestamp(name)[:10],
        cfg.root / v["s2_dir"], float(v["s2_max_cloud"]), OpticalParams())
    for line in log:
        typer.echo(line)
    typer.echo(f"ok  {targets}")
    typer.echo(f"ok  {clear}")


@app.command("validate-points")
def validate_points_cmd(
    product_dir: Annotated[Path, typer.Argument(help="HyP3 RTC product directory (in data/raw).")],
    truth: Annotated[str, typer.Option(help="iip-satellite, iip-aircraft or s2.")] = "s2",
    band: Annotated[str, typer.Option(help="Which detections to validate.")] = "HV",
    open_water: Annotated[bool, typer.Option(help="Only detections away from pack ice.")] = True,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Match detections to truth points (drift-corrected); report precision and recall."""
    import json

    import pandas as pd

    from iceberg_sar.pointval import run_point_validation

    cfg = load_config(config)
    name = Path(product_dir).name
    summary, t, d = run_point_validation(cfg, name, truth, band, open_water)
    typer.echo(json.dumps(summary, indent=2))
    size_col = "size" if "size" in t else None
    if size_col is None and "area_m2" in t:
        t["size"] = pd.cut(t["area_m2"], [0, 400, 1600, 6400, float("inf")],
                           labels=["<400 m2", "400-1600 m2", "1600-6400 m2", ">6400 m2"])
    c = t[t.compared]
    if len(c):
        typer.echo("recall by size class:")
        typer.echo(c.groupby("size", observed=True)["matched"].agg(["sum", "count"]).to_string())
    out = cfg.path("outputs") / "validation"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{name}_{band}_{truth}"
    t.assign(time=t["time"].astype(str)).to_file(f"{stem}_truth.geojson", driver="GeoJSON")
    d.to_file(f"{stem}_detections.geojson", driver="GeoJSON")
    (out / f"{stem.name}_summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    typer.echo(f"ok  {stem}_*.geojson")


@app.command()
def figures(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Regenerate the README / docs figures in docs/img from current outputs."""
    from iceberg_sar.figures import make_figures

    for kind, path in make_figures(load_config(config)).items():
        typer.echo(f"ok  {kind:<11} {path}")


@app.command("thor-smoke")
def thor_smoke(
    product_dir: Annotated[
        Path | None, typer.Argument(help="HyP3 RTC product directory (default: thor.smoke.scene).")
    ] = None,
    row: Annotated[int | None, typer.Option(help="Crop centre row (default: auto).")] = None,
    col: Annotated[int | None, typer.Option(help="Crop centre column (default: auto).")] = None,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Milestone 5a: embed one crop with frozen THOR, save PCA-of-tokens RGB next to the dB crop."""
    from iceberg_sar.thor.smoke import run_smoke

    cfg = load_config(config)
    if product_dir is None:
        product_dir = cfg.path("raw") / cfg.section("thor")["smoke"]["scene"]
    center = (row, col) if row is not None and col is not None else None
    outputs = run_smoke(Path(product_dir), cfg, center)
    typer.echo(outputs["summary"].read_text(encoding="utf-8"))
    for kind, path in outputs.items():
        typer.echo(f"{kind:<8} {path}")


@app.command("thor-scenes")
def thor_scenes(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Milestone 5b: find IW HH+HV frames with same-pass IIP labels; pick the training set."""
    from iceberg_sar.thor.scenes import save_frames, training_scenes

    cfg = load_config(config)
    out = cfg.root / cfg.section("thor")["training"]["dir"]
    frames, selected = training_scenes(cfg)
    save_frames(frames, out / "candidate_frames.geojson")
    path = save_frames(selected, out / "training_frames.geojson")
    typer.echo(f"{len(frames)} candidate frames ({frames['aoi_sightings'].sum()} AOI sightings)")
    typer.echo(f"{len(selected)} selected ({selected['aoi_sightings'].sum()} AOI sightings)")
    for r in selected.itertuples():
        typer.echo(f"  {r.scene}  {r.aoi_sightings:4d}")
    typer.echo(f"ok  {path}")


@app.command("thor-order")
def thor_order(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Order sigma0 HyP3 RTC for the THOR training frames (skips ones already ordered).

    Frames HyP3 rejects (e.g. no DEM coverage over open ocean) are replaced by the next-ranked
    candidates; the frames actually ordered are written back to training_frames.geojson.
    """
    import dataclasses

    import geopandas as gpd
    from hyp3_sdk.exceptions import HyP3Error

    from iceberg_sar.hyp3 import get_client, submit_rtc
    from iceberg_sar.thor.scenes import SceneParams, save_frames, select_frames

    cfg = load_config(config)
    t = cfg.section("thor")["training"]
    params = SceneParams.from_config(cfg)
    candidates = gpd.read_file(cfg.root / t["dir"] / "candidate_frames.geojson")
    unlimited = dataclasses.replace(params, max_scenes=len(candidates), max_credits=10**9)
    ranked = select_frames(candidates, unlimited).sort_values("aoi_sightings", ascending=False)
    target = len(select_frames(candidates, params))
    client = get_client()
    typer.echo(f"Credits remaining: {client.check_credits()}")
    ordered = []
    for r in ranked.itertuples():
        if len(ordered) == target:
            break
        try:
            batch = submit_rtc(client, r.scene, cfg, name=THOR_JOB_NAME,
                               overrides={"radiometry": t["radiometry"]})
        except HyP3Error as e:
            typer.echo(f"skip {r.scene}: {e}")
            continue
        ordered.append(r.Index)
        for job in batch:
            typer.echo(f"job {job.job_id}  status={job.status_code}  {r.scene}")
    save_frames(ranked.loc[ordered].sort_values("start_time"),
                cfg.root / t["dir"] / "training_frames.geojson")
    typer.echo(f"{len(ordered)} frames ordered. Credits remaining: {client.check_credits()}")


@app.command("thor-download")
def thor_download(
    config: ConfigOption = DEFAULT_CONFIG,
    wait: Annotated[bool, typer.Option(help="Wait for running jobs to finish.")] = True,
) -> None:
    """Download the THOR training scenes into data/raw; zips are deleted after extraction."""
    from iceberg_sar.hyp3 import download_jobs, get_client

    cfg = load_config(config)
    client = get_client()
    batch = client.find_jobs(name=THOR_JOB_NAME)
    if wait and not batch.complete():
        batch = client.watch(batch)
    failed = [j for j in batch if j.failed()]
    for d in download_jobs(batch, cfg.path("raw"), delete_zip=True):
        typer.echo(f"ok  {d}")
    for j in failed:
        typer.echo(f"FAILED  {j.job_parameters.get('granules')}  {j.job_id}")


@app.command("thor-chips")
def thor_chips(
    product_dirs: Annotated[
        list[Path] | None,
        typer.Argument(help="Preprocessed RTC products (default: downloaded training frames)."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Chips directory (default: <thor dir>/chips).")]
    = None,
    sheet: Annotated[int, typer.Option(help="Chips in the contact sheet.")] = 20,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Milestone 5b: cut 512 px chips with point labels; write a summary and contact sheet."""
    import geopandas as gpd
    import pandas as pd

    from iceberg_sar.thor.chips import chip_scene, contact_sheet, load_index, summary_table
    from iceberg_sar.thor.scenes import find_product

    cfg = load_config(config)
    thor_dir = cfg.root / cfg.section("thor")["training"]["dir"]
    out = out or thor_dir / "chips"
    if not product_dirs:
        frames = gpd.read_file(thor_dir / "training_frames.geojson")
        product_dirs = [p for g in frames["scene"]
                        if (p := find_product(cfg.path("raw"), g, "sigma0")) is not None]
        typer.echo(f"{len(product_dirs)} of {len(frames)} training frames downloaded")
    for d in product_dirs:
        if not (cfg.path("interim") / d.name / f"{d.name}_seaice_mask.tif").is_file():
            typer.echo(f"skip {d.name}: run `preprocess` first")
            continue
        index = chip_scene(d, cfg, out)
        typer.echo(f"ok  {d.name}: {index['kind'].value_counts().to_dict()}")
    index = load_index(out)
    if index.empty:
        return
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        table = summary_table(index)
        typer.echo(table.to_string())
    typer.echo(f"ok  {contact_sheet(index, out, out / 'contact_sheet.png', cfg, n=sheet)}")
    try:
        table.to_csv(out / "summary.csv")
    except PermissionError:  # e.g. open in Excel, which locks it on Windows
        typer.echo(f"warning: could not write {out / 'summary.csv'} (is it open?)")


@app.command("thor-sheet")
def thor_sheet(
    kind: Annotated[str | None, typer.Option(help="berg, water, edge or ice_edge.")] = None,
    split: Annotated[str | None, typer.Option(help="train, val or test.")] = None,
    scene: Annotated[str | None, typer.Option(help="Substring of the scene name.")] = None,
    n: Annotated[int, typer.Option(help="Max chips (5 per row).")] = 40,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Contact sheet of a subset of chips (no re-cutting), e.g. --kind ice_edge --split train."""
    from iceberg_sar.thor.chips import contact_sheet, load_index

    cfg = load_config(config)
    chips = cfg.root / cfg.section("thor")["training"]["dir"] / "chips"
    index = load_index(chips)
    for col, val in (("kind", kind), ("split", split)):
        if val:
            index = index[index[col] == val]
    if scene:
        index = index[index["scene"].str.contains(scene, regex=False)]
    if index.empty:
        typer.echo("no chips match")
        raise typer.Exit(1)
    index = index.sort_values(["time", "chip_id"])
    parts = [p for p in (split, kind, scene) if p]
    name = "contact_sheet_" + ("_".join(parts) if parts else "all") + ".png"
    title = " ".join(parts) + f" ({min(len(index), n)} of {len(index)} chips)"
    typer.echo(f"ok  {contact_sheet(index, chips, chips / name, cfg, n=n, title=title)}")


@app.command("thor-features")
def thor_features(
    device: Annotated[str, typer.Option(help="torch device (cuda or cpu).")] = "cuda",
    overwrite: Annotated[bool, typer.Option(help="Re-embed chips already cached.")] = False,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Milestone 5c: cache frozen THOR features for every chip (<thor dir>/features)."""
    from iceberg_sar.thor.features import cache_features

    cfg = load_config(config)
    thor_dir = cfg.root / cfg.section("thor")["training"]["dir"]
    done, cached = cache_features(thor_dir / "chips", thor_dir / "features", cfg, device,
                                  overwrite)
    typer.echo(f"ok  {done} chips embedded, {cached} already cached -> {thor_dir / 'features'}")


@app.command("thor-train")
def thor_train(
    head: Annotated[str, typer.Option(help="linear or centernet.")] = "linear",
    fold: Annotated[str, typer.Option(
        help="A fold from thor.train.folds, 'final' (thor.train.final) or 'all' (both).",
    )] = "all",
    device: Annotated[str, typer.Option(help="torch device (cuda or cpu).")] = "cuda",
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Milestone 5c: train a head on the cached features (data/models/thor/<head>/<fold>).

    Folds early-stop on the other year and write curves + val overlays; 'final' builds the 5d
    model from both years (a fixed-epoch refit or an ensemble of the folds, per the config).
    """
    from iceberg_sar.thor.train import overlay_sheet, train_final, train_head

    cfg = load_config(config)
    head_dir = cfg.path("models") / "thor" / head
    folds = list(cfg.section("thor")["train"]["folds"])
    steps = [*folds, "final"] if fold == "all" else [fold]
    for step in steps:
        typer.echo(f"== {head} {step}")
        if step == "final":
            outputs = train_final(head, cfg, head_dir, device, log=typer.echo)
        else:
            out = head_dir / f"fold_{step}"
            outputs = train_head(head, cfg, out, device, log=typer.echo, fold=step)
            outputs["overlays"] = overlay_sheet(out, cfg, out / "val_overlays.png")
        typer.echo(outputs["summary"].read_text(encoding="utf-8"))
        for kind, path in outputs.items():
            typer.echo(f"{kind:<8} {path}")


@app.command("thor-overlays")
def thor_overlays(
    head: Annotated[str, typer.Option(help="linear or centernet.")] = "linear",
    fold: Annotated[str, typer.Option(help="Fold whose val chips to show.")] = "a",
    threshold: Annotated[float | None, typer.Option(help="Peak threshold (default: best F1).")]
    = None,
    n: Annotated[int, typer.Option(help="Chips (5 per row).")] = 20,
    seed: Annotated[int, typer.Option(help="Negative-chip sample.")] = 0,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Heatmap overlays of a trained fold head on its val chips (no retraining)."""
    from iceberg_sar.thor.train import overlay_sheet

    cfg = load_config(config)
    out = cfg.path("models") / "thor" / head / f"fold_{fold}"
    name = ("val_overlays.png" if threshold is None and seed == 0
            else f"val_overlays_t{threshold}_s{seed}.png")
    typer.echo(f"ok  {overlay_sheet(out, cfg, out / name, n, seed, threshold)}")


@app.command("thor-detect")
def thor_detect(
    product_dirs: Annotated[list[Path], typer.Argument(help="Preprocessed RTC products.")],
    head: Annotated[str, typer.Option(help="linear or centernet.")] = "linear",
    model: Annotated[str, typer.Option(help="final, or fold_<name> for a fold model.")]
    = "final",
    device: Annotated[str, typer.Option(help="torch device (cuda or cpu).")] = "cuda",
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Milestone 5c: THOR detections for whole scenes (CFAR schema + detector, model, score).

    Writes <scene>_thor-<head>[-<model>]_detections.geojson (model omitted for 'final').
    """
    from iceberg_sar.thor.infer import detect_scene

    cfg = load_config(config)
    weights = cfg.path("models") / "thor" / head / model / "head.pt"
    tag = f"thor-{head}" if model == "final" else f"thor-{head}-{model}"
    for d in product_dirs:
        outputs = detect_scene(Path(d), cfg, weights, device, log=typer.echo, tag=tag)
        typer.echo(outputs["summary"].read_text(encoding="utf-8"))
        for kind, path in outputs.items():
            typer.echo(f"{kind:<10} {path}")


THOR_JOB_NAME = "iceberg-alley-thor"


if __name__ == "__main__":
    app()
