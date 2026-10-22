# cfar-object-detection: API notes

Notes from reading the library at commit `4136b4c` (see `ENVIRONMENT.md`) and checking its
behaviour in the `cfar-bench` env. Items marked **(measured)** were checked with small
synthetic inputs, not only by reading the code.

## Detectors: paper vs library

`faerch2023.xml`, sec. 3.2, names six detectors, and the library has all six:

| Paper (sec. 3.2) | Strategy (sec. 2.1.1) | `detector=` | Module | Input it works on | Uses `enl` | Uses `method` |
|---|---|---|---|---|---|---|
| Log-normal | 3: per channel + Boolean | `'lognorm'` | `lognormal.py` | **dB** | no | yes |
| Gamma | 3: per channel + Boolean | `'gamma'` | `gamma.py` | linear | yes | yes |
| K | 3: per channel + Boolean | `'k'` | `kdistribution.py` | linear | yes | yes |
| NIS | 1: combined channel | `'nis'` | `normsum.py` + `gamma.py` | linear | no (estimates its own) | no |
| iDPolRAD | 1: combined channel | `'idpolrad'` | `dpolrad.py` | linear | no | no |
| Wishart | 2: multivariate | `'wishart'` | `wishart.py` | linear | yes | no |

The paper merges the per-channel detectors with **AND** (sec. 3.3). The library also has a
`utils.multiscale` decorator, which the paper doesn't use.

## Entry point

```python
cfar_filters.detector.run(image, mask, detector='gamma', method='AND', pfa=1e-9,
                          enl=10.7, minsize=2, wi=9, wo=15) -> np.ndarray[bool]  # (X, Y)
```

| Argument | Expected |
|---|---|
| `image` | `np.ndarray`, **float32**, shape **(2, X, Y)**, band order **[HH, HV]**, in **dB**. `run` converts to linear itself (except for log-normal). It raises `TypeError` on other dtypes and `ValueError` on other shapes. A random-sample "smells like" test only warns if the data don't look like dB. |
| `mask` | `np.ndarray`, **bool**, (X, Y), **True = valid**. Raises `TypeError` for other dtypes. An all-False mask is treated as "no mask" (all valid). |
| `detector` | one of the six names above. An unknown name warns and returns `0`, it doesn't raise. |
| `method` | `'AND'` / `'OR'`. Only used by gamma, lognorm and k. |
| `pfa` | **total** PFA in (0, 1). For AND/OR, `run` converts it to per-channel PFA itself (see below). |
| `enl` | one float, used for **both** channels. The paper uses 10.7 for S1 EW, from the product spec (Table 1), not estimated. |
| `minsize` | removes objects **smaller than** `minsize` (skimage `remove_small_objects`, 4-connectivity). The docstring says "this size or smaller", which is wrong. **(measured)** With `minsize=2`, 1-px objects and diagonal-only pairs are removed. This matches the paper's "objects covering only a single pixel were removed". |
| `wi`, `wo` | **diameters in pixels** of the guard area and the outer clutter area. For iDPolRAD they are the test and training windows instead. |

The per-detector functions (`gamma.detector(image2d, mask, pfa, enl, wi, wo)` etc.) take a
single band, in linear intensity (log-normal: dB). They also apply no PFA correction and no
small-object removal.

Returns a bool outlier raster of shape (X, Y). Every detector sets a **20 px border** to
False (`mask_edges`), which is 800 m at 40 m.

## PFA per channel (sec. 2.1.1)

`run` passes per-channel PFA to the single-band detectors:

- AND: `sqrt(pfa)`
- OR: `1 - sqrt(1 - pfa)`

These match the paper's equations. The OR form cancels catastrophically in float64: for
total PFA < ~1e-16 it returns 0. The paper's Sentinel-1 runs use AND, so this doesn't affect
the reproduction. `iceberg_sar.benchmark.pfa` uses the stable form `-expm1(0.5*log1p(-pfa))`
for logging. NIS, iDPolRAD and Wishart apply the total PFA to their single test statistic.

## Window geometry **(measured)**

`fast_edge_mean` / `fast_edge_std` are Numba stencils with a fixed neighbourhood of
±20 px. A pixel at offset (i, j) enters the clutter estimate when

`wi // 2 <= sqrt(i² + j²) <= wo // 2`, and it is not NaN.

| `wi`, `wo` | Clutter px | Outer extent | Guard (excluded) span on axis |
|---|---|---|---|
| 9, 13 (paper at 40 m) | 68 | 13 px | **7 px** (offsets −3…3) |
| 9, 15 (notebook) | 104 | 15 px | 7 px |
| 10, 20 (README example) | 248 | 21 px | 9 px |
| 0, 3 (iDPolRAD test) | 5 | 3 px | none: **plus-shape incl. centre** |
| 0, 57 (iDPolRAD train) | 1677 | **41 px** (cropped square) | none |

Consequences:

- The guard is **narrower than `wi`**: clutter starts at `wi // 2` px. With wi = 9 at 40 m,
  clutter starts 160 m from the centre, not the paper's nominal 180 m. The README drawing
  shows a guard of radius `wi // 2`, which doesn't match the code.
- An even `wo` gives an odd extent `2·(wo // 2) + 1`.
- **Size limit:** a `wo // 2` above 20 can't fit the stencil, so the window is silently
  cropped to a 41 × 41 square. The effective limit is **`wo ≤ 41`**. `run` warns from
  `wo > 40` (off by one).
- **iDPolRAD in the paper (57 × 57 px training window) exceeds the limit.** It must have run
  with the cropped 41 × 41 square (1677 px), unless the authors edited the stencil. Our adapter
  only allows this through `allow_window_crop: true` for iDPolRAD, and records it in the output.
- Its "3 × 3" test window is really the centre plus its 4 neighbours (5 px).
- Wishart computes its number of looks `m` from the nominal ring area
  `π((wo−1)²−(wi−1)²)/4 · enl` (62.8 · enl for 9/13), not the true pixel count (68).

## Masking **(measured)**

- `mask` is only checked at the **centre** pixel. Neighbours enter the clutter estimate unless
  they are **NaN**. In the notebook, land is not NaN, so coastal rings include land pixels.
- If every ring pixel is NaN, `fast_edge_mean` returns **0, not NaN**. Then gamma and K flag
  every positive pixel, and Wishart's ε substitution makes the same pixel bright too.
  `fast_edge_std` returns NaN, so log-normal is safe.
- The runner (`external/cfar_runner/run_cfar.py`) therefore sets masked pixels to NaN
  (`masked_as_nan`). It also removes pixels whose ring is less than `min_ring_fraction` valid
  from `mask` (0.5, as in our CA-CFAR's `min_background_fraction`; 0 = library as-is).
- Log-normal standardises with the global mean/std before thresholding. That is an affine
  transform and doesn't change its detections.
- iDPolRAD fits its generalised-gamma threshold **globally**, on 10 000 random pixels with
  0 < iDPolRAD < 50·mean (seed fixed at 0; this also resets numpy's global RNG).

## ENL

- One `enl` for both channels (gamma, K, Wishart). Log-normal and iDPolRAD don't use it.
- The runner's `--enl-hh` / `--enl-hv` (adapter `enl_bands`, config `enl_mode: per_band`)
  run gamma and K per band with their own ENL. They follow `detector.run`'s steps
  (dB → linear, per-channel PFA, Boolean merge, `remove_small_objects`), and with equal ENLs
  the raster is identical to the library's (tested). Wishart still gets the single `--enl`.
- NIS estimates its own ENL from the NIS image: mean²/var over pixels below 2 × median
  (paper, sec. 3.3).
- `utils.calc_enl(samples)` = nanmean² / nanvar, the same moment estimator we use for the
  open-water box in 0c.

## K detector

- Thresholds come from a LUT over ν = linspace(1, 20, 40) for the given PFA and ENL, computed
  once per call (~20 s per the paper's Table 7; per channel, so twice for AND).
- ν is estimated per pixel by method of moments from the ring mean/variance and clipped to [1, 20].
- **LUT index (measured):** `v_idx = round(ν · 20 / 40) = round(ν / 2)` is not the
  position of ν in the LUT. Estimated ν = 2, 5, 10, 20 picks LUT ν = 1.49, 1.97, 3.44, 5.87.
  K therefore uses thresholds for heavier-textured clutter than it estimated, so it is more
  conservative (fewer detections) than intended. The same line is in v1, v2 and HEAD with
  N = 40, so the published K results include it. We run it as-is and don't patch it.

## Other behaviour to keep in mind

- gamma: the threshold multiplier comes from a Nelder–Mead fit starting at 3. The library
  warns "might not have converged" if the result equals the start value. At very small PFA,
  check the warnings the runner logs.
- Wishart is two-sided. The library keeps only outliers brighter than the clutter mean in
  **both** bands.
- The code uses all CPU cores (Numba `parallel=True`).
- Numba compiles each kernel on first call (several seconds). Don't count that in runtime.
- `examples.ipynb`, cell 8, plots `lognormal_outliers` from the single-band cell, not the
  dual-band `lognorm_outliers`. This only affects the plot.
