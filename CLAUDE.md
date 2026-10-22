# Iceberg Alley SAR: Iceberg Detection off Newfoundland

## What this project is
Detect icebergs off the coast of Newfoundland & Labrador in Sentinel-1 SAR imagery with **THOR**, a SAR foundation model (frozen encoder + a small trained head), benchmarked against a classical **CA-CFAR** detector, and display the results on an interactive **CesiumJS** globe with a timeline.

**Main story (since Milestone 5):** THOR is the detector, the CA-CFAR is the baseline. The classical SAR pipeline (HyP3 calibration, land mask, HV pack-ice mask) feeds both, and THOR depends on it. Headline: on four held-out April 2019 passes (512 same-pass IIP labels), THOR finds 0.84 vs. CFAR 0.74 at the same number of detections, and 11 vs. 4 of 14 large bergs. Don't use the old CFAR 78% as a headline: it counts near-ice detections and isn't comparable (the CFAR's comparable figure is 0.74). Keep the caveats with the headline: one month of one year, THOR learned from IIP analyst labels, precision unknown without AIS.

This is a portfolio project. It should show hands-on SAR and maritime-domain skills to remote sensing / ocean-tech employers in St. John's, NL. Clarity, reproducibility, and a polished demo matter more than state-of-the-art accuracy.

## About me (the developer)
- Geospatial engineer, 15+ years: photogrammetry (DEM, orthophoto, stereo), 3D reconstruction, geospatial AI.
- Strong in: Python (rasterio, Shapely, GeoPandas, FastAPI), PostGIS, PyTorch, CesiumJS, Vue, Docker, AWS.
- **New to SAR.** Explain SAR-specific concepts briefly when they come up (backscatter, speckle, polarization, incidence angle, CFAR, dB vs. linear). Don't explain general Python or GIS basics.
- I prefer small, hands-on steps that produce a visible result quickly over long setup phases.

## How to work with me (instructions for Claude Code)
- Work one milestone at a time. Before starting a milestone, confirm the plan in a few lines. After finishing, show how to run it and what to look at.
- Keep modules small and testable. Use type hints. Put parameters in `config.yaml` instead of hardcoding them.
- Never commit raw data, credentials, or model weights. Use `.env` for secrets and keep `data/` in `.gitignore`.
- Default CRS for outputs is EPSG:4326 (GeoJSON / CZML). For metric processing, use the scene's native projection or UTM 21N/22N (EPSG:32621 / 32622).
- When a choice has a well-known trade-off (e.g. CFAR variant, window sizes), pick a sensible default, say why in one line, and move on.

## Tech stack
- **Python 3.11+**: numpy, scipy, rasterio, shapely, geopandas, scikit-image, pyyaml, typer (CLI)
- **Data access**: `asf_search` (find scenes), `hyp3_sdk` (order RTC processing)
- **Viz**: CesiumJS (optionally in a small Vue app), CZML for time-dynamic detections
- **Optional**: FastAPI to serve results, Docker for reproducibility

## Data sources
1. **Sentinel-1 SAR imagery** via ASF Vertex (https://search.asf.alaska.edu). Requires a free NASA Earthdata login.
   - Product: GRD, processed with **HyP3 RTC** into analysis-ready GeoTIFFs (gamma0). Request **power (linear)** scale and convert to dB where needed. Note the HyP3 monthly processing credit limit.
   - **Check the polarization of each scene.** Depending on area and date, it may be IW (VV/VH) or EW (HH/HV). Prefer HH/HV: the CFAR and sea-ice mask are tuned on HV.
   - Fallback source: Copernicus Data Space Ecosystem.
2. **Area of interest**: "Iceberg Alley", the NE Newfoundland coast, roughly Bonavista to St. Anthony, extending offshore. Store the AOI as `aoi.geojson`.
3. **Season**: April–June has the most icebergs. Start with 2–3 scenes from one spring.
4. **Ground truth / validation**: Canadian Ice Service iceberg charts and the International Ice Patrol iceberg data for matching dates. These are for visual and approximate validation, not pixel-exact labels.
5. **Land mask**: OSM land polygons or GSHHG coastline, buffered ~500 m to remove coastal clutter.
6. **Point-level validation**: IIP Iceberg Sightings Database (NSIDC G00807, seasons up to 2021) and Sentinel-2 L2A optical targets (Earth Search COGs). Because IIP sightings end in 2021, per-berg validation uses spring 2019 scenes.
7. **Experimental classifier data**: Kaggle Statoil/C-CORE Iceberg Classifier Challenge chips (`data/kaggle/train.json`, don't redistribute).

## Repository structure
```
iceberg-alley-sar/
├── CLAUDE.md
├── README.md
├── config.yaml
├── aoi.geojson
├── data/                 # gitignored
│   ├── raw/              # HyP3 downloads
│   ├── interim/          # masked / calibrated rasters
│   ├── outputs/          # detections (GeoJSON), quicklooks, CZML, validation
│   ├── land/             # OSM land polygons clipped to the AOI
│   ├── ground_truth/     # NAIS chart GIFs, IIP sightings, Sentinel-2 targets
│   ├── kaggle/           # Statoil/C-CORE train.json (experimental classifier)
│   └── models/           # trained classifier weights
├── docs/                 # findings write-ups and figures
├── validation/           # hand-transcribed NAIS chart counts (CSV)
├── src/iceberg_sar/
│   ├── config.py         # config.yaml loading, metre -> pixel conversion
│   ├── search.py         # find scenes with asf_search
│   ├── hyp3.py           # order + download RTC products
│   ├── landmask.py       # OSM land polygons -> AOI land mask
│   ├── seaice.py         # HV pack-ice mask, distance to ice
│   ├── preprocess.py     # land mask, dB conversion, nodata handling
│   ├── cfar.py           # CFAR detector
│   ├── detections.py     # connected components -> GeoJSON with attributes
│   ├── groundtruth.py    # NAIS chart download, per-square count comparison
│   ├── iip.py            # IIP iceberg sightings (NSIDC G00807)
│   ├── sentinel2.py      # Sentinel-2 optical targets
│   ├── matching.py       # one-to-one matching, drift-offset estimation
│   ├── pointval.py       # point-level precision / recall
│   ├── export_czml.py    # detections -> CZML for Cesium
│   ├── pipeline.py       # end-to-end `run` for a date range
│   ├── chips.py          # Kaggle-matched chips (experimental classifier)
│   ├── classify/         # experimental iceberg vs. ship CNN
│   └── cli.py            # typer CLI tying it together
├── viewer/               # Vue 3 + Vite + CesiumJS front end
├── notebooks/            # exploration only; logic lives in src/
└── tests/
```

## Current status (as of 2026-10-13)
- **Milestones 0–6: done.** The live viewer is on GitHub Pages with THOR and CFAR layers, an About ("?") dialog and per-pass THOR vs CFAR charts.
- **Docs reframed THOR-first (2026-10-13):** README, About dialog, `docs/validation-findings.md` (note on 78% vs 74%), and the presentations in `docs/presentation/` (Project Brief v2 deck + one-pager, CA-CFAR vs THOR brief as the technical deep-dive).
- **Milestone 7 (branch `milestone7-thor-first`, 2026-10-15):** viewer is THOR-first (7a), `run` runs THOR +
  CFAR with a skip path (7b), `thor-fetch-head` pulls the head from a GitHub release pinned by SHA-256 (7c),
  screenshots retaken (7d). Head checkpoints now load with `weights_only=True`.
- **Stage 2 / Milestone 0 (branch `Establishing-benchmark`, 2026-10-21): 0a–0c done** (0d is mine).
  `cfar-bench` is a uv venv (no conda), Python 3.10.9, library at commit `4136b4c`
  (`benchmarks/ENVIRONMENT.md`). API quirks (dB input, NaN masking, 41 px stencil, K LUT index)
  in `benchmarks/cfar_api_notes.md`. `bench-smoke` runs all six detectors on 2019-04-29 at 40 m
  (`benchmarks/m0_smoke.md`). 40 m HH/HV ENLs are 6.0 / 1.8, so gamma/K run with per-band ENL (`benchmark.enl_mode`, decided 2026-10-22).
- **Stage 2 / Milestone 1 (branch `Establishing-benchmark`, 2026-10-22): 1a–1d done (1d as Path B), 1e–1f not started.**
  Commands: `bench-scenes`, `bench-download`, `bench-preprocess`, `bench-landmask`, `bench-labels`.
  - 1a: `scenes.csv` checked against Table 1. Open mismatches: S1 orbit (paper 028068, product
    027966 / rel. 069) and S2 rel. orbit (paper R068, product R111). S1 EW GRDH (slice 10:27:30) came
    from ASF (no CDSE login in `.env`; CDSE download is used once `CDSE_USERNAME/PASSWORD` exist).
    S2 is the original N0207 product from Google's public bucket (CDSE only has N0500). The S1
    slice is the last of the pass and covers 71 % of the S2 area.
  - 1b: SNAP 14 at `%LOCALAPPDATA%\Programs\esa-snap` (installed 2026-10-22). Ellipsoid-Correction-RD
    (GG has no pixel spacing), nearest neighbour, 40 m, UTM 20N; `nodataValueAtSea=false` is required.
  - 1c: osmdata.openstreetmap.de was unreachable, so land comes from the coastline ways in
    `data/land/newfoundland-and-labrador-261020.osm.pbf` (`benchmark.faerch2023.land.osm_pbf`).
  - ENL at 40 m on GRDH: HH 2.45 / HV 0.52, vs the paper's 10.7. Suggests the paper used GRDM;
    open question whether to process GRDM too before 1e.
  - 1d (RECONSTRUCTED LABELS, `benchmarks/faerch2023/labels_rebuilt.md`): 394 S2 bergs in the overlap
    vs 492 (large 135 vs 136). Two rules replace the paper's manual check: contrast >= 2x the water
    and B11 <= 0.035. Weak step: drift. Only 18 verified paths vs 270; most S2 bergs don't stand out
    in 40 m EW, and looser settings verify chance pairs (mean drift ~7 km = random). Open: wait for
    Færch's labels, match by hand, or score all-labels only.
- **Milestones 0–3: done.** Search, HyP3 RTC (10 m), land + pack-ice masks, CA-CFAR on HV with strip rejection and an HH co-pol check, GeoJSON detections, and the Vue + Cesium viewer.
- **Scenes:** spring 2025 (2025-05-02, 05-08, 05-14) in the viewer, checked against NAIS charts. Spring 2019 (2019-04-15, 04-29) and 2020-06-10 for point-level validation.
- **Validation (beyond the plan):** 78% recall (207/267) against same-pass IIP analyst labels on 2019-04-29, median offset 67 m (IIP positions are truncated to 0.1'; corrected to cell centres). Precision is not measurable without AIS. Large bergs (>120 m) are missed: their grown blobs exceed `max_structure_m` (the ice-strip rule). See `docs/validation-findings.md`.
- **Experimental:** iceberg vs. ship CNN trained on Kaggle chips (`iceberg_prob`). Use it as a review hint only; see the README.
- **Milestone 4: done.** Portfolio README with screenshots, `cli.py run` end to end for a date range (`pipeline.py`, scene selection in `run:` in `config.yaml`), and two LinkedIn post drafts in `docs/linkedin-posts.md` (local only, excluded via `.git/info/exclude`, never commit). Not done: Dockerfile, deployed static viewer.
- **Stretch not done:** OS-CFAR / K-distribution (`cfar.variant` has a slot for it).
- **Milestone 5 (branch `milestone5-thor`):** 5a and 5b done (`thor-smoke`, `thor-scenes/order/download/chips`, `src/iceberg_sar/thor/`). 20 sigma0 training frames, 723 train / 154 val chips, 432 / 78 visible labelled bergs. THOR takes HH/HV natively; frozen features separate bergs and ice strips from water. See `docs/thor-vs-cfar.md`. Local GPU (CUDA) works; no Triton on Windows, so dynamo is disabled.
  5c done: cached features (blocks 2/5/8/11), linear + CenterNet heads (`thor-features`,
  `thor-train`, `thor-overlays`, `thor-detect`). Val moved to two year folds (2019 <-> 2020; the
  Feb-Mar 2019 chips are pack ice, kept as a separate stress set). Linear F1 0.72 / 0.83 on the
  folds, CenterNet no better (overfits). Final 5d models: linear refit on both years (150 epochs),
  CenterNet = ensemble of the two fold models.
  5d done (`thor-compare [--sweep]`, `thor/compare.py`): 4 held-out same-pass frames (2019-04-17,
  04-23, 04-24, 04-29; 512 labels). Pooled at matched detection counts THOR linear 0.84 vs CFAR
  0.74 (04-29 alone: 0.89 vs 0.74); CenterNet equal; small bergs a tie. HyP3 rejects open-ocean
  frames (no DEM), so richer April frames couldn't be added (`data/thor/test_candidates.csv`).
  CFAR misses large bergs (4/14 vs THOR 11/14): grown blobs (340-470 m) exceed `max_structure_m`.
  THOR needs the CFAR's pack-ice mask (47-59% of its peaks on 04-15 fall in it). 2019-04-15 is
  unscored (aircraft truth, drift unresolved). THOR is a toggleable viewer layer.
  PFA sweep runs in one pass (`detect_raster_multi`); `component_table` stats now reduce over
  labelled pixels only (same output, much faster). Not done: 4x4 patches, AIS precision.

#### Stage 1
## Milestones

### Milestone 0: Setup (short)
- Repo, virtual env, `pyproject.toml`, `.gitignore`, `config.yaml`, `aoi.geojson`.
- **Done when:** `python -m iceberg_sar.cli --help` runs.

### Milestone 1: First image
- Search Sentinel-1 scenes over the AOI for a spring date range. Order one HyP3 RTC job, download it, and open it with rasterio.
- Apply the land mask. Save a quicklook PNG in dB with a sensible stretch.
- **Done when:** I can see a masked SAR scene of the NL coast and spot bright targets by eye.

### Milestone 2: CFAR detection
- Implement **CA-CFAR** (cell-averaging) on linear intensity: target cell, guard window, background window, threshold factor from a chosen false-alarm rate. Use a vectorized or convolution-based implementation for speed.
- Post-process: connected components, min/max size filter, remove edge and nodata artifacts.
- Output GeoJSON points with attributes: id, scene id, timestamp, lat/lon, area (px and m²), peak and mean backscatter (dB) per band, incidence angle.
- Compare visually against Canadian Ice Service charts for the same date.
- **Done when:** a GeoJSON of candidate targets overlays sensibly on the quicklook, with few obvious false alarms at sea.
- Stretch: try a K-distribution or OS-CFAR and compare.

### Milestone 3: Cesium viewer
- Export detections to CZML with time intervals per scene, so the Cesium timeline steps through dates.
- Style points by detection attributes (e.g. size by `structure_m`, color by contrast or distance to pack ice). Show the scene footprint. Info box shows attributes.
- Optional: SAR quicklook as an imagery layer (static PNG with bounds, or COG tiles via TiTiler).
- **Done when:** I can open the viewer, scrub the timeline, and click icebergs.

### Milestone 4: Polish for the portfolio
- README with a short write-up: problem, data, method, results, limitations, screenshots or GIF.
- Reproducible run: `cli.py run --config config.yaml` goes end to end for a given date range.
- Optional: Dockerfile, deployed static viewer, short blog or LinkedIn post.
- **Done when:** someone else can clone the repo, follow the README, and reproduce the demo.

## Known pitfalls to watch for
- Speckle and sea state: rough seas raise clutter and false alarms. Log wind conditions where possible.
- Coastline and islands produce bright returns, so the land mask buffer matters. OSM land polygons miss rocks/islets mapped as single nodes (thousands off NL); the land mask adds them from Overpass, and repeat detections at one spot are dropped as static targets.
- Near-range vs. far-range incidence angle changes backscatter. Keep incidence angle as an attribute of each detection.
- Sea ice (pack ice) looks very different from icebergs in open water. Early in the season, pack ice can cover parts of the AOI. Mask or exclude it rather than detecting it as icebergs.
- Keep dB vs. linear consistent: CFAR on linear intensity, dB for display and reported attributes.
- 10 m IW GRD is oversampled (~20 m true resolution). HV speckle grains span ~2×2 px and pass a 2 px minimum, so the HH co-pol check is needed.
- HyP3 removes the HV thermal noise floor, so open-water HV sits near −30 to −40 dB with ENL < 1. Estimate ENL from the scene instead of assuming it.
- Keep size parameters in metres in `config.yaml`. They are converted to pixels per scene, so tuning carries over between 20 m and 10 m.

## Decisions made
- First dates: spring 2025 (2025-05-02, 05-08, 05-14). Spring 2019 for IIP point validation, since that database ends in 2021.
- Viewer: Vue 3 + Vite + CesiumJS, Esri imagery basemap (no ion token needed).

## Open questions (decide with me as we go)
- Whether to add FastAPI + PostGIS for storing detections across many scenes, or keep flat files (flat files so far).
- Whether to keep the experimental classifier in the portfolio write-up, given that it can't be validated without AIS.


### Milestone 5: THOR foundation-model detector (comparison with CFAR)

Goal: run the THOR foundation model (frozen encoder + small trained head) on the
same HH/HV scenes as the CFAR, and compare both against the same IIP labels.
Reference: Forgaard et al., "Efficient iceberg detection in Sentinel-1 imagery
using the THOR foundation model" (IGARSS 2026). Weights: FM4CS/THOR-1.0-tiny and
-base on Hugging Face (Apache 2.0), loaded via TerraTorch.

New code lives in `src/iceberg_sar/thor/` (`encoder.py`, `chips.py`, `targets.py`,
`heads.py`, `train.py`, `infer.py`). Parameters go in a `thor:` block in `config.yaml`.

#### 5a: Smoke test (no training)
- Install TerraTorch + the THOR extension. Load THOR Tiny.
- Check the Sentinel-1 band list in the THOR code: are HH/HV supported, or only VV/VH?
  If only VV/VH: feed HH→VV slot, HV→VH slot, and note it as a known domain shift.
- Check what input THOR expects: sigma0, linear vs dB, and normalization stats.
- Run the frozen encoder on one 512×512 crop from 2025-05-08 at patch size 8×8.
  Reduce the embeddings to 3 PCA components and save them as an RGB PNG next to the dB quicklook.
- **Done when:** the PCA image shows bergs (and the ice edge) as visibly distinct
  from open water. If it doesn't, stop and reconsider before building anything.

#### 5b: Training data from IIP labels
- Select HH/HV IW scenes 2019–2021 that have same-pass IIP Sentinel-1 analyst labels.
  **Hold out all of April 2019** (covers the 2019-04-15 and 04-29 validation scenes
  and the same bergs on nearby dates). Split train/val by scene, never by crop.
- Order HyP3 RTC with **sigma0 radiometry** for these scenes (THOR was pretrained
  on sigma0). Budget against the monthly HyP3 credit limit before ordering.
- Extract 512×512 crops: crops around labelled bergs, plus negative crops sampled
  from open water, the pack-ice edge and coastal areas. Apply the land/ice masks as
  for the CFAR.
- Targets: same-pass labels need no drift; move IIP positions to their 0.1' cell centre, snap
  to the HH peak (done: `targets.py`), then render each sighting as
  a Gaussian on a heatmap (sigma in metres in config). Point labels, no masks.
- Save the chips under `data/thor/chips/`.
- **Done when:** I can see a contact sheet of ~20 crops with target Gaussians
  overlaid, and a summary table of crops/bergs per scene and per split.

#### 5c: Train the heads
- Precompute and cache frozen-encoder features for all chips (`data/thor/features/`).
  THOR Tiny at 8×8 on 512² is 64×64×192 per crop, a few MB each. After that,
  head training is fast and can be iterated on cheaply.
- Heads: (1) linear baseline, (2) CenterNet-style heatmap head (MLP + sigmoid,
  focal loss), which is the primary head because it matches point labels.
- Inference: sliding window with overlap over a full scene, peak extraction
  (3×3 max-pool NMS + threshold) → detections GeoJSON with the **same schema as
  the CFAR output**, plus `detector: "thor"` and `score`.
- **Done when:** validation loss curves are saved, and heatmap overlays on held-out
  validation crops look sensible.

#### 5d: Head-to-head with CFAR
- Run both detectors on 2019-04-15 and 2019-04-29 with identical masks, identical
  IIP labels and identical matching radius (`pointval.py`).
- Since precision isn't measurable without AIS, compare **recall at a matched
  number of detections**. Sweep the CFAR PFA and the THOR score threshold, and plot
  recall vs. detections per scene for both.
- Break down misses and detections by berg size (esp. >120 m, which the CFAR misses),
  distance to pack ice and incidence angle.
- Run THOR on the spring 2025 scenes and add it as a toggleable CZML layer in the viewer.
- Write up in `docs/thor-vs-cfar.md`.
- **Done when:** one figure (recall vs. detection count, both methods) and the
  write-up answer: where does THOR beat the CFAR, where doesn't it, and why.

#### Milestone 5 decisions
- Start with THOR Tiny at 8×8 patches. The paper reports Tiny ≈ Base, and 8×8 is close to
  4×4 at a quarter of the tokens. Try 4×4 only if small bergs are being missed.
- Encoder stays frozen. No fine-tuning of THOR itself.
- CenterNet head is primary, linear head is the sanity baseline. (Revised in 5c: linear matched or
  beat CenterNet on both folds, so linear is the main THOR curve, CenterNet the second.)
- Head selection on year folds, never on April 2019. Final models train without a val set
  (linear: fixed-epoch refit; CenterNet: fold ensemble), see `docs/thor-vs-cfar.md`. Nothing is
  tuned on the 5d test scenes; if something must change after 5d, change it on the folds and say so.

#### Milestone 5 pitfalls
- Radiometry: gamma0 RTC ≠ sigma0. Use sigma0 products or convert with incidence angle.
- Leakage: the same berg appears across consecutive passes. Hold out by date range, not by scene.
- Tile borders: stitch overlapping windows and keep only centre regions, or peaks double-count.
- Compute: 5a/5b run on CPU. Feature caching and 4×4 inference need a GPU
  (e.g. an AWS spot g4dn/g5 instance). Don't commit weights or cached features.

#### Milestone 5 open questions
- If there are too few HH/HV scenes with IIP labels: add VV/VH crops to training
  (the paper's setup) and accept the cross-polarization shift?
- Whether a 4×4 run is worth the GPU cost after seeing the 8×8 results.


### Milestone 6: Live viewer on GitHub Pages

Goal: a public, stable URL for the Cesium viewer, updated by
`cli.py run` → `cli.py publish` → `git push`. Static hosting only, no backend.

#### 6a: Data-driven viewer + publish command (local)
- Make the viewer data-driven. At startup it fetches `data/manifest.json` and loads
  every CZML and quicklook listed there. No scene lists hardcoded in Vue code.
- Build all data URLs from `import.meta.env.BASE_URL`, so the viewer works both
  at `/` locally and under `/iceberg-alley-sar/` on Pages.
- Manifest schema (keep it small, validate it with a pydantic model or a JSON Schema test):
  - `generated` (ISO timestamp), `scenes`: list of
    `{scene_id, date, polarization, detector, czml, quicklook, quicklook_bounds [W,S,E,N], n_detections}`.
  - `detector` is `"cfar"` or `"thor"`, so the THOR layer from Milestone 5 slots in.
- New command `cli.py publish`, a new module `publish.py`:
  - Copies CZML from `data/outputs/` into `viewer/public/data/`.
  - Writes web quicklooks: PNG or WebP, longest side ≤ `publish.quicklook_max_px`,
    EPSG:4326 bounds recorded in the manifest.
  - Writes `manifest.json`.
  - Fails loudly if any single file > `publish.max_file_mb` or the folder total > `publish.max_total_mb`.
  - Publishes derived products only: never raw/interim rasters, Kaggle chips, model weights or cached features.
  - Is idempotent: it rebuilds `viewer/public/data/` from scratch on each run.
- `.gitignore`: keep `data/` ignored, and explicitly allow `viewer/public/data/`.
- config.yaml:
```yaml
  publish:
    out_dir: viewer/public/data
    scenes: all            # or a list of scene_ids
    detectors: [cfar]      # add thor after Milestone 5
    quicklook_max_px: 4096
    quicklook_format: webp
    max_file_mb: 25
    max_total_mb: 300
```
- Tests: manifest round-trip and schema, the size-limit check fires, and publish
  never copies anything from `data/raw`, `data/interim`, `data/kaggle` or `data/models`.
- **Done when:** `cli.py publish` followed by `npm run build && npm run preview` (with the
  production base path) shows all published scenes, with no hardcoded scene names left in the viewer.

#### 6b: GitHub Actions → Pages
- Vite: set `base: '/iceberg-alley-sar/'` for production builds (env-driven, `/` in dev).
  Make sure Cesium's static assets (Workers, Assets, Widgets, ThirdParty) are copied
  into the build and `CESIUM_BASE_URL` points under the base path.
  Check this in `npm run preview` before pushing.
- Workflow `.github/workflows/pages.yml`:
  - Triggers: push to `main` touching `viewer/**` or the workflow file; plus `workflow_dispatch`.
  - Permissions: `contents: read`, `pages: write`, `id-token: write`.
  - Concurrency: group `pages`, cancel in progress.
  - Job `build`: checkout, setup-node (pinned LTS, npm cache on `viewer/package-lock.json`),
    `npm ci`, `npm run build` in `viewer/`, `actions/configure-pages`,
    `actions/upload-pages-artifact` with `viewer/dist`.
  - Job `deploy`: `actions/deploy-pages`, environment `github-pages`, output the page URL.
  - Pin every action to a major version tag.
  - Python does not run in CI. Data is produced locally by `publish` and committed.
- Repo settings (manual, I do this): Settings → Pages → Source: GitHub Actions.
- **Done when:** a push to `main` deploys, and the Pages URL loads the globe, the
  timeline steps through scenes, and clicking a detection shows its attributes.

#### 6c: Make it shareable
- Before the repo goes public: scan the full git history for `.env`, credentials,
  data files and the LinkedIn drafts (`gitleaks` or `git log --all -- <path>`).
  Fix this before flipping the visibility, not after.
- Check that the Esri basemap loads on the deployed domain. Check the browser
  console for CORS or 404 errors on Cesium workers.
- Add a short "About" panel or overlay to the viewer: what the points are, data
  sources (Sentinel-1/Copernicus, IIP), method (CA-CFAR on HV), a link to the repo.
- README: "Live demo" link at the top, and a "Updating the live viewer" section
  with the three commands.
- **Done when:** the link works in a fresh incognito window on desktop and on a phone.

#### Milestone 6 decisions
- GitHub Pages: free, static, and lives next to the portfolio repo. Railway is unnecessary (no backend).
- Published data is committed to `viewer/public/data/`. This is fine while it stays under the size limits.

#### Milestone 6 open questions
- If quicklooks outgrow the limits: move `viewer/public/data/` to a Cloudflare R2
  or S3 bucket with CORS, and have `publish` upload there (no redeploy needed).
- Whether to show the experimental `iceberg_prob` in the public viewer or hide it.

### Milestone 7: THOR-first product

Goal: make the code match the story. The docs already present THOR as the detector and the
CA-CFAR as the baseline; the viewer and `cli.py run` still treat the CFAR as primary.

#### 7a: Viewer
- THOR is the primary layer: listed first in the legend and layer list, drawn on top, and its
  open-water count shown first per scene. The CFAR becomes a "CA-CFAR baseline" layer
  (keep it on by default or off: decide when you see it).
- Subtitle: "Sentinel-1 HH/HV · THOR detections, CA-CFAR baseline" (now "CA-CFAR candidates + THOR").
- Info box for a THOR detection: score first, then the shared attributes. Today the score is
  only in the entity name (info box title); `_description()` in `export_czml.py` is shared with
  the CFAR and has no score row. Add a "THOR score" row at the top of the table for THOR
  packets, then re-run `czml` + `publish` to rebuild the CZMLs.
- Check the About dialog still matches what the viewer shows (its text was updated in the docs pass).
- **Done when:** a first-time visitor reads the viewer as a THOR result with a CFAR comparison,
  on desktop and phone.

#### 7b: THOR in the pipeline
- `cli.py run` runs THOR after preprocessing, controlled by `run.detectors: [thor, cfar]` in
  `config.yaml`, then `czml` (and optionally `publish`), so one command gives both layers.
- If CUDA or the trained head is missing, skip THOR with one clear message and still finish the
  CFAR part (don't fail the run). `--detectors` on the CLI overrides the config.
- `--dry-run` reports whether THOR will run (GPU found, head found).
- Tests: detector selection, the skip path (no GPU / no head) with a stubbed THOR call.
- **Done when:** `run --start ... --end ...` on a GPU machine writes THOR and CFAR detections and
  a CZML with both layers, and on a CPU-only machine finishes with the CFAR and a skip notice.

#### 7c: Reproducible THOR without retraining
- The trained linear head is tiny (769 parameters) but the rule is "never commit weights".
  Options: a GitHub release asset, or a Hugging Face model repo, plus a `thor-fetch-head`
  command that downloads it to `data/models/thor/linear/final/`. Decide with me first.
- README Quickstart then becomes: install, fetch head, `run`.

#### 7d: Visuals
- Done 2026-10-14 (ahead of 7a-c): `docs/img/viewer.png` (2025-05-08, THOR rings + CFAR dots) and
  `docs/img/viewer_compare.png` (2019-04-29 with its recall chart), taken from `npm run preview`
  with Playwright + Edge; swapped into the Project Brief v2 deck, the one-pager and the CA-CFAR vs
  THOR brief. Retake them after 7a, since the viewer's subtitle and legend order will change.

#### Milestone 7 decisions (2026-10-15)
- Head weights: GitHub release asset (tag `thor-head-v1`, asset `head.pt`), pinned in
  `thor.release` by SHA-256. `thor-fetch-head` downloads it to `data/models/thor/linear/final/head.pt`.
- The CA-CFAR baseline layer stays on by default: a dot inside a ring reads as "found by both".
- `run` takes its THOR head from `viewer.thor_model`, so the run and the viewer layer can't drift apart.
  A `--detectors thor` run still preprocesses, but the CZML lists scenes by their CFAR output.
- 7d: only the Project Brief v2 deck (slide 11) holds the full screenshot; the other decks use a
  globe-only crop that 7a didn't change. The deck's PDF was re-exported with PowerPoint.


## Stage 2: Establishing Benchmark

Goal: validate my detectors against published, peer-reviewed CFAR detectors on a
scene with independent optical ground truth, before choosing a research direction.
Reference: Færch et al. 2023, "Iceberg detection in Sentinel-1 and ALOS-2 SAR
imagery using CFAR detectors" (The Cryosphere 17, 5335). Code:
github.com/LaustFaerch/cfar-object-detection (GPL-3.0, Zenodo 10.5281/zenodo.11504039).
Milestone numbering restarts at 0 for Stage 2. Stage 1 milestones stay as they are.

### Stage 2 conventions

- New code lives in `src/iceberg_sar/benchmark/` (`adapter.py`, `pfa.py`,
  `preprocess.py`, `landmask.py`, `labels.py`, `sweep.py`, `score.py`, `plots.py`).
  Parameters go in a `benchmark:` block in `config.yaml`. Nothing hardcoded.
- The GPL-3.0 CFAR library is NEVER imported into `src/iceberg_sar/` and never
  copied into this repo. It runs in its own environment, called through a small
  runner script in `external/cfar_runner/` (own LICENSE file, GPL-3.0), which
  talks to the main package through files only: GeoTIFF in, GeoTIFF/GeoJSON out.
- Benchmark inputs (committed to the repo):
  - `benchmarks/faerch2023/scenes.csv`: scene info for the benchmark (Sentinel-1 EW
    and Sentinel-2 scenes from Table 1 of the paper).
  - `benchmarks/faerch2023/faerch2023.xml`: the paper as full-text JATS XML
    (Færch et al. 2023, The Cryosphere, CC BY 4.0). This is the source of truth
    for processing steps, parameters and published results. Read it with an XML
    parser, don't rely on memory of the paper, and don't fetch it from the web.
- Generated outputs go under `data/benchmark/faerch2023/` (gitignored). Never
  commit scenes, masks, labels or detection rasters.
- Published results go into `benchmarks/faerch2023/reference.yaml`, parsed from
  Tables 3 and 5 in `faerch2023.xml`. Each entry records its table number and row.
  Show me the result before using it; I check it against the paper.
  Never fill in a value from memory. If something isn't in the XML, stop and ask.
- Fig. 6 curves come only from `benchmarks/faerch2023/reference_fig6.csv`, which I
  digitize myself. If the file doesn't exist, skip the curve overlay and compare
  against the tables only.
- Explain SAR-specific concepts briefly when they first come up (ENL, PFA
  correction, multilooking). One or two sentences, not a lecture.
- I'm on Windows 11 (RTX 3070 Laptop 8 GB, 16 GB RAM). Paths via pathlib,
  no bash-only scripts.

### Milestone 0: Benchmark environment

#### 0a: Run the reference code as-is
- Clone cfar-object-detection to `external/cfar-object-detection/` (gitignored).
- Create a separate conda env `cfar-bench` from the repo's requirements.
  Do not install it into the main project env (Numba/NumPy pins may clash).
- Run `examples.ipynb` end to end. Record the exact repo commit hash and package
  versions in `benchmarks/ENVIRONMENT.md`.
- Read the README and code, and write `benchmarks/cfar_api_notes.md`: the detector
  names and function signatures, expected input (linear intensity vs dB, band order,
  dtype), how the mask is passed, the units of `wi`/`wo`, the 41 px window limit,
  and how ENL is supplied or estimated. Cross-check the detector list against the
  detectors named in `faerch2023.xml`.
- **Done when:** the notebook runs cleanly and the API notes list every
  detector the paper uses with its call signature.

#### 0b: Thin adapter
- `external/cfar_runner/run_cfar.py` (runs in `cfar-bench`): CLI that takes an
  input GeoTIFF (HH, HV bands, linear intensity), a mask GeoTIFF, a detector name,
  a list of PFA values, `wi`, `wo`, ENL, and writes one binary detection GeoTIFF per
  PFA, preserving georeferencing.
- `src/iceberg_sar/benchmark/adapter.py` (main env): builds the inputs, calls the
  runner with `subprocess` using `benchmark.cfar_env_python` from config, then
  converts each detection raster into objects (connected components → centroid,
  area, length) as GeoJSON with the **same schema as the CA-CFAR output**, plus
  `detector`, `pfa`, `pfa_per_channel`.
- `pfa.py`: function that converts a target total PFA into per-channel PFA for
  HH AND HV and for HH OR HV, following sec. 2.1.1 of `faerch2023.xml`. Unit-test
  it against a hand-worked example that I check.
- Window sizes are given in metres in config and converted to odd pixel counts
  from the raster's pixel spacing. Fail loudly if a window exceeds the library limit.
- **Done when:** one call to `adapter.run(detector, raster, mask, pfas)` returns
  GeoJSON for any detector, and the PFA unit tests pass.

#### 0c: All detectors on one of my frames
- Pick one April 2019 IW frame (HyP3 RTC, HH/HV).
- Window problem: 360 m / 520 m at 10 m is 37 / 53 px, over the 41 px limit.
  Multilook the frame 4×4 to 40 m (mean of linear intensity) for this run, and
  note that multilooking raises the ENL.
- Estimate ENL from a homogeneous open-water box (mean²/variance in linear
  intensity), store it per scene in a sidecar JSON.
- Run every detector at one mid-range PFA (e.g. 1e-9 total) with the existing land mask.
- Output: quicklook PNG per detector with detections over the HV dB image, and a
  table of detection counts per detector.
- **Done when:** I can see all detectors' detections side by side on the same frame
  and the counts table is in `benchmarks/m0_smoke.md`.

#### 0d: (me, not Claude Code) Ask Færch for the labels
- Email Laust Færch for the 492-berg validation set with drift paths.
  Milestone 1 can start without them; 1d has a fallback.

### Milestone 1: Reproduce Færch 2023 on C-band

#### 1a: Download the benchmark scenes
- Read `benchmarks/faerch2023/scenes.csv`, print the parsed scene list, and check
  it against Table 1 in `faerch2023.xml`. Report any mismatch before downloading.
- Load the scenes into `benchmark.faerch2023.scenes` in config, generated from
  the CSV rather than typed by hand.
- `search.py` gains a `--product-ids` mode for the Copernicus Data Space
  (CDSE OData API, credentials from `.env`).
- Download the Sentinel-1 EW GRDM and Sentinel-2 L1C products to
  `data/benchmark/faerch2023/raw/`, and verify checksums.
- **Done when:** both products are on disk with checksums verified, and a
  footprint map PNG shows they overlap.

#### 1b: SNAP preprocessing
- Write a SNAP GPT graph `benchmarks/faerch2023/s1_ew_preprocess.xml`:
  Apply-Orbit-File → Remove-GRD-Border-Noise → ThermalNoiseRemoval → Calibration
  (sigma0, HH and HV) → Ellipsoid-Correction at 40 m. Take the operator choice and
  settings from the processing description in `faerch2023.xml` and quote the
  relevant passage in a comment in the graph file.
- `preprocess.py` calls `gpt` with `-c` cache set from config (~10 GB) and tiles if
  memory runs out. Output: GeoTIFF, linear sigma0, HH/HV bands, in a UTM CRS.
- Keep it linear: the CFARs run on linear intensity, dB is for quicklooks only.
- **Done when:** a quicklook of HH and HV in dB shows the scene with no border-noise
  stripes, and pixel spacing is 40 m.

#### 1c: Land mask
- `landmask.py`: OSM land polygons (osmdata.openstreetmap.de, split, EPSG:4326),
  clipped to the scene, reprojected to the scene's UTM CRS, buffered 500 m
  (buffer in metres, from config), rasterized to the scene grid.
- **Done when:** mask overlay quicklook shows the coast fully covered with the buffer.

#### 1d: Ground truth
- Path A (labels arrive): `labels.py` loads them as-is into GeoJSON, keeps the
  verified vs interpolated flag and drift info, reports counts (expect 492 total,
  ~270 confident).
- Path B (no labels): rebuild from the Sentinel-2 scene with the method as described
  in `faerch2023.xml` (4-sigma threshold on the optical band). Write the exact
  steps used, and quote the section followed, in
  `benchmarks/faerch2023/labels_rebuilt.md`. Every output file and plot from
  Path B is marked "reconstructed labels".
- **Done when:** a quicklook shows labels over the SAR image, and the counts
  table matches the paper (Path A) or is documented with its differences (Path B).

#### 1e: Detector sweep
- `sweep.py`: every detector × 21 PFA levels (log-spaced 1e-21 to 1e-1), with the
  paper's windows (360 m guard, 520 m outer) and HH AND HV merging via `pfa.py`.
- Cache detections per (detector, pfa) as GeoJSON; skip finished runs on restart.
  The sweep must be resumable, I run it overnight.
- **Done when:** all detector/PFA combinations exist on disk and a log reports
  runtime per detector.

#### 1f: Score and compare
- `score.py`: nearest-object matching between detection objects and labels, one
  label per detection. Radii from config: 250 m for verified positions, 1789 m for
  interpolated ones. Recall, precision, F1 per (detector, pfa).
- Report verified-only and all-labels results separately.
- Table of F1 at optimum PFA next to `reference.yaml`, with the difference column.
- Plot F1 vs PFA per detector. If `reference_fig6.csv` exists, overlay the
  digitized Fig. 6 curves; otherwise plot my curves only.
- Tolerance from config (`benchmark.repro_tolerance_f1`, start at 0.05).
- **Done when:** the table and figure are in `docs/faerch2023-reproduction.md`,
  with each detector marked within or outside tolerance and a short explanation
  for any that are outside.

### Milestone 2: Validate my CA-CFAR

#### 2a: CA-CFAR on the Færch scene
- Run my CA-CFAR on the 40 m EW scene under the Milestone 1 protocol: same land mask,
  same windows (converted to pixels), same labels, same scorer.
- Map my threshold to a PFA scale so it can sweep the same 21 levels. If my detector
  isn't parameterized by PFA, sweep its own threshold and plot F1 vs detection count
  for all detectors instead (both views go in the note).
- Run two variants: as it is (HV only, with the length rule) and without the
  length rule.
- Add both as extra curves to the 1f figure.
- **Done when:** the 1f figure and table include my CA-CFAR as two extra rows.

#### 2b: Reference detectors on my IW frames
- Run the gamma and NIS detectors (via the adapter) on all four April 2019 IW frames,
  multilooked to 40 m as in 0c, with my land/ice masks.
- Also run my CA-CFAR on the same 40 m frames, so resolution isn't a variable.
- Compare against IIP labels with `pointval.py` and the existing matching radius.
  Precision isn't measurable here, so compare **recall at matched detection counts**:
  sweep PFA / threshold, plot recall vs detections per frame for all three.
- **Done when:** one recall-vs-detections figure per frame, plus a combined one.

#### 2c: Validation note
- `docs/cacfar-validation.md`: where my CA-CFAR sits relative to the published
  detectors on both datasets, and every place it behaves differently (length rule,
  single-channel HV vs dual-pol merge, window sizes, ENL handling).
- **Done when:** the note answers, in one page with figures: does my baseline
  behave like a standard gamma CA-CFAR, and where doesn't it, and why.

### Stage 2 decisions
- CFAR library stays external, behind a file-based subprocess boundary (GPL-3.0).
- The paper is read from local JATS XML, never from memory or a web fetch.
- C-band only: the ALOS-2 scene from the paper is out of scope.
- Benchmark resolution is 40 m. IW frames are multilooked to 40 m when compared
  with the reference detectors.
- Reproduction is judged on F1 at optimum PFA and on curve shape, not single points.
- ENL per band for gamma / K on our IW frames (`benchmark.enl_mode: per_band`, 2026-10-22):
  HyP3 HV water has a much lower ENL than HH (1.8 vs 6.0 at 40 m). Wishart keeps one ENL (min).
  For the Færch EW scene the paper used one ENL (10.7) for both; use `single` there.

### Stage 2 pitfalls
- Per-channel PFA ≠ final PFA. Always log both.
- The 1789 m radius makes speckle count as hits at high PFA. Never report
  all-labels recall without the verified-only number next to it.
- EW GRDM and IW RTC differ in resolution, ENL, radiometry (sigma0 vs gamma0)
  and processing. Don't mix them in one table without saying so.
- Multilooking changes the ENL: re-estimate it after resampling, never reuse the 10 m value.
- SNAP on 16 GB: set the cache explicitly, close other apps, tile if needed.
- Numba compiles on first call; don't count that in runtime.

### Stage 2 open questions
- Did the labels arrive (Path A) or am I rebuilding them (Path B)?
- If a detector misses tolerance in 1f: is it preprocessing, ENL, or the labels?
