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
