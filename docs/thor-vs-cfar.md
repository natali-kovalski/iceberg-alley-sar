# THOR vs. CFAR (Milestone 5)

Frozen THOR encoder (FM4CS, Apache 2.0) + a small trained head, compared with the CA-CFAR
on the same HH/HV scenes and IIP labels. Reference: Forgaard et al., IGARSS 2026.

## 5a: Smoke test (frozen encoder, no training)

`python -m iceberg_sar.cli thor-smoke [PRODUCT_DIR] [--row R --col C]` embeds one 512×512
crop and writes `data/outputs/thor/smoke_<scene>_r<row>_c<col>.png` + `.json`.

### What THOR expects
- **Bands:** THOR has native `IW_HH` / `IW_HV` patch embeddings (its own band group), so
  HH/HV go in as themselves. No VV/VH slot swap and no domain-shift caveat for that.
- **Radiometry:** pretrained on Sentinel-1 GRD **sigma0**, thermal-noise corrected,
  Range-Doppler geocoded (THOR paper, appendix). HyP3 RTC is gamma0, so we convert with
  sigma0 = gamma0 · cos(incidence) from the HyP3 incidence map.
- **Scale and normalization:** dB, standardized with THOR's 10 m IW stats
  (HH −13.95 ± 6.80 dB, HV −22.46 ± 6.91 dB). Normalization is the caller's job, not the
  backbone's. NaN (land, nodata) → 0 after normalization, i.e. the band mean.
- **Geometry:** `ground_cover` = crop size in metres (512 px × 10 m = 5120 m). At patch 8
  THOR Tiny gives 64×64 tokens × 192 features per block, 12 blocks.
- **Windows:** THOR wraps kernels in `torch.compile`, which needs Triton (not available on
  Windows). `load_encoder` turns dynamo off when Triton is missing; results are identical,
  just un-fused. One 512² crop on the local GPU takes well under a second after model load.

### Result on 2025-05-08 (S1A, 10 m)

| Crop | CFAR bergs | Berg-token distance to water mean (× water median) | Bergs more anomalous than water tokens |
|---|---|---|---|
| Ice edge (rows 782, cols 8439) | 3 | 2.75 | 100% |
| Open water (rows 8554, cols 7072) | 4 | 2.84 | 100% |

(Block 5, HH+HV.) The figure's last panel maps each token's distance from the open-water mean:
- **Open water:** the 4 CFAR bergs are the 4 brightest tokens. A dozen other tokens are
  also bright; they could be small bergs below the CFAR threshold or speckle. A trained head
  has to sort those out.
- **Ice edge:** ice strips form a distinct class in both the PCA and the distance map.
- **PCA RGB is a weak test for bergs.** Water fills >95% of tokens, so the top 3 components
  model speckle (≈12% variance each) and 3–4 berg tokens never get a component of their own.
  The distance map is the more honest view.

### Sweep (same two crops)
- **Block:** mid blocks separate bergs best (block 3: 2.9×, block 5: 2.8×, block 11: 1.8–2.2×).
  Deeper blocks trade local detail for context. Default for the smoke test: block 5. Heads in
  5c will see several blocks (the paper's detector uses blocks 2, 5, 8, 11).
- **HV floor:** clipping HV at −40, −32 or −28 dB changes little. About half of the open-water HV
  pixels are below −40 dB (the HyP3 noise floor is removed), so HV is mostly noise at sea.
- **HH only:** bergs separate somewhat more (3.2–4.3× at block 3). HV stays in, since the
  paper and the CFAR use it and it is what separates pack ice from bergs.

### Things to carry into 5b/5c
- The swath edge lights up in the distance map (NaN filled with the band mean makes a hard
  edge). Drop or down-weight crops with nodata, or mask those tokens out of the loss.
- `seaice_mask` includes a 1 km buffer, so "ice tokens" are mostly water near ice. Use the
  unbuffered ice cores if ice tokens need their own statistics.

**Verdict:** pass. Bergs and ice strips are distinct from open water in the frozen
features. Go ahead with 5b.

## 5b: Training data

`thor-scenes` → `thor-order` → `thor-download` → `preprocess` + `detect` per scene →
`thor-chips`. Output in `data/thor/chips/` (`summary.csv`, `contact_sheet.png`, one folder per
scene with `.npz` chips, `labels.csv`, `index.csv`, `scene.json`).

### Scenes
20 IW HH+HV frames with same-pass IIP Sentinel-1 analyst labels, HyP3 RTC **sigma0** at 10 m
(1,200 credits). Six top-ranked frames were rejected by HyP3 (no DEM over open ocean) and
replaced by the next-ranked ones.

- **Holdout:** 2019-03-25 to 2019-05-06, i.e. April plus a week each side. The frames richest in
  labels (2019-05-04/05/06) came 5-7 days after the 04-29 test scene and would show the same
  bergs, which drift only 10-20 km/day.
- **Split by date:** val = Feb-Mar 2019 (3 frames), at least 8 weeks from any training frame.
  Train = 17 frames, May-Jun 2019 and Apr-May 2020.

### Labels
- **IIP positions are truncated to 0.1 arc-minute.** Moving them to the cell centre halves the
  label error (median 135 m → 67 m). Then each label snaps to the brightest HH peak within 200 m
  if it is ≥ 6 dB above its surroundings: 99% of training labels snap.
- **Labels the model can't see are not targets.** 75 of 515 training labels (15%) fall on
  masked pixels: 82 of the 110 such labels overall are outside the AOI (the land mask exists
  only inside it), 28 are within the 500 m land buffer of islands and islets, where bergs
  often ground. They are kept as `masked_rc` but never rendered as targets.
- **Unlabelled CFAR detections** (no label within 300 m) are stored per chip so the loss can
  ignore them instead of treating them as water: 267 in train, 449 in val.

| Split | Frames | Labels | On masked pixels | Visible, in chips | Crops (berg / water / edge / ice edge) |
|---|---|---|---|---|---|
| train | 17 | 515 | 75 | 432 | 713 (398 / 198 / 99 / 18) |
| val | 3 | 114 | 35 | 78 | 154 (76 / 38 / 20 / 20) |

### Masks in the chips
- **Validity comes from a separate sea mask** (`<scene>_sea.tif`: 1 = sea incl. pack ice,
  0 = land + 500 m buffer, outside the AOI, nodata), written by `preprocess` and backfilled for
  older scenes. The masked bands blank land and pack ice alike. A first version un-masked
  "anything inside the ice mask" to show the model the ice, and near the coast that let land
  through (5% of one ice-edge chip was land, 11% land buffer). Fixed; chips re-cut.
- **The chip `ice` layer is pack ice without the CFAR's 1 km buffer** (`<scene>_seaice_core.tif`,
  the buffered mask eroded by the same disk). The buffer keeps floes away from the CFAR; THOR
  sees the real pixels and should learn where the ice actually is. Ice-edge chips are sampled
  on this core, so they contain real ice (median 12% of the chip), not just buffer water.

### Caveats
- **The ice detector may flag coastal water.** On 2020-05-16 it marks a thin band along the
  coast that looks like open water in HH. With the HV noise floor removed the open-water
  reference is very low (-36 dB on that scene), so +6 dB is easy to exceed. CFAR-side issue;
  not fixed here.
- **The pack-ice mask fails in early-season scenes.** It defines open water from the darkest
  quarter of the scene; when ice covers most of it (Feb-Mar 2019) the reference is ice and the
  pack goes unmasked. The CFAR then fires on floes: 326 of 329 detections on 2019-03-06 have no
  label. This affects val more than train, and matters for the 5d comparison too.
- One berg can appear in two adjacent frames of the same pass (both in train); that is the
  same image twice, not leakage across splits.
