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
- **Docs reframed THOR-first (2026-10-13):** README, About dialog, `docs/validation-findings.md` (note on 78% vs 74%), and the presentations in `docs/presentation/` (Project Brief v2 deck + one-pager, CA-CFAR vs THOR brief as the technical deep-dive). The code still treats the CFAR as primary in places; that's Milestone 7.
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

#### Milestone 7 open questions
- Distribution of the head weights (7c).
- Whether the CFAR layer stays visible by default in the public viewer.
