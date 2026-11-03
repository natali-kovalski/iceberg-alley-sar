**Verdict: partial agreement under modified preprocessing (revised 2026-11-02).** With the
paper's own labels (Path A) and **without** thermal noise removal (the paper describes it on),
five of six detectors come within 0.05 of the paper's best F: gamma +0.015, Wishart -0.002,
NIS -0.004, log-normal -0.013, K -0.015 (shared evaluation mask, 487 labels in the scoring
area; 2026-10-27 over all 492: +0.011 / -0.006 / -0.008 / -0.017 / -0.019). iDPolRAD is 0.058
below. Gamma and Wishart reach their F 13-15 decades of PFA away from the paper's optimum, so
this is agreement in F level, not a full reproduction. The shapes agree with the paper's sec. 4: "The K detector and
the log-normal detector achieve their maximum accuracy at a very narrow interval of PFA levels,
whereas the gamma, iDPolRAD, NIS, and Wishart generally exhibit higher F score across a wide
range of PFA levels." Fig. 6 is not digitized, so the comparison of shapes is by eye.

**Deviation: no thermal noise removal.** The paper lists thermal noise removal among its SNAP
steps. After SNAP 14 thermal noise removal (subtraction, then clipping of negatives) all
detectors degrade (grey curves, last table column: every detector stays at F 0.54 or below,
Wishart produces >20,000 objects even at 1e-21). Revision C12 (`docs/revision-tnr-ablation.md`)
subtracts the same noise field three ways: gamma with Table 1's ENL gets F 0.640 without
subtraction, 0.430 with SNAP's clipped subtraction, and 0.640 with the subtraction floored at the
noise level. On this scene the drop comes from how the pixels below the noise floor are handled,
not from the subtraction; no software defect is claimed:

| ENL on the 40 m GRDM stack | HH | HV |
|---|---|---|
| Table 1 (product definition) | 10.7 | 10.7 |
| With thermal noise removal, median 2 km open-water box | 9.5 | 0.98 |
| Without, same boxes | 10.74 | 11.45 |
| With, 1 km boxes around gamma's false positives (1e-19) | 0.51 | 0.43 |

SNAP 14 subtracts the noise floor and clips negative values to ~3.7e-12 (-115 dB), not 0: 23.7 % of
HV and 0.75 % of HH open-water pixels sit there. Dark, calm water then fluctuates far more than an
ENL of 10.7 allows, and the false positives cluster in one such patch near the coast (median HH
-35.6 dB). SNAP 14 also ignores `removeThermalNoise=false`, so the run without it drops the node
from the graph (`preprocess.graph_variant`). Without noise removal the measured ENL matches
Table 1, which is what the detectors assume. Which SNAP version the paper used, and what its
noise removal did, is an open question for the authors.

**Optimum PFA.** NIS, K, log-normal and iDPolRAD peak within one decade of the paper's optimum.
Gamma (1e-6 vs 1e-19) and Wishart (1e-5 vs 1e-20) reach the paper's F, TP and FP about 13-15
decades higher: at the paper's PFA they detect fewer objects (gamma 1e-19: TP / FP 156 / 8 vs
238 / 32). These are the two detectors that take the ENL as given (NIS estimates its own; K's
look-up table also uses it, but K matches). A different effective ENL in the paper's gamma and
Wishart runs would shift the PFA axis without changing the F level; not checked. Tables 3 and 5
also disagree on gamma's optimum PFA (1e-19 vs 1e-9) while describing the same detections.

**iDPolRAD (outside tolerance, -0.058).** Same number of false positives as the paper (70 vs 74),
fewer true positives (162 vs 193), mostly medium bergs (31 % vs 41 %). iDPolRAD's threshold
comes from one generalized-gamma fit over the whole image (10,000 random pixels, `cfar_api_notes.md`),
so it depends on what the image contains: our land mask (OSM coastline + 500 m), swath and
noise-floor handling differ from the paper's. Its 57 px training window is cropped to the
library's 41 px, as it must have been in the paper. The flat start of the curve (1e-21 to 1e-17,
identical detections) is numerical: the library computes `1 - pfa`, which is exactly 1.0 in
float64 for PFA <= 1e-17. Those runs are flagged and left out of every optimum (revision C2).

**What else holds.**
- Labels are not the issue: Path A reproduces Table 2 exactly (181 / 175 / 136) and the mean
  verified drift (3954 vs 3953 m).
- Scoring area: the cloud-free Path B area changes F by at most 0.011.
- Verified-only recall (revision C5; the old verified-only F mixed a different label set with
  its own false-positive accounting) is 0.70-0.91 at each detector's optimum (iDPolRAD 0.56),
  against all-labels recall 0.41-0.55 (iDPolRAD 0.33): most misses are interpolated labels.
  Hits on interpolated labels are at chance level: shifting every detection 5 km in a random
  direction still matches 11-20 of them on average, more than the real runs do (3-14;
  `docs/revision-robustness.md`).
- Size breakdown at my optimum follows Table 5 (large 87-93 %, small 4-10 %), except iDPolRAD.
  The "at the paper's PFA" column was wrong until 2026-11-02 (revision C1): the lookup matched
  every PFA below 1e-8 to the 1e-21 run.
- Matching (revision C4): greedy vs optimal one-to-one, true vs EPSG:5938-metre radii, and an
  interpolated radius of 1000 or 500 m change best F by at most 0.03 and swap at most adjacent
  ranks. Differences under ~0.02 F between detectors are not distinguishable with this sample:
  the 95% bootstrap interval of F is about ±0.04, of paired differences about ±0.02.
- GRDH instead of GRDM (gamma 1e-19): more false positives (575 vs 341) and higher recall, so
  the paper's 40 m, ENL 10.7 product is GRDM; Table 1's "GRDH" is a typo.
- Sweep coverage (1e-1 to 1e-21): 117 of 126 (detector, PFA) runs scored; the other 9 are above
  the 20,000-object cap, where precision is below 0.1.

**Open questions for the authors:** SNAP version and whether thermal noise removal was applied;
the ENL passed to gamma and Wishart; the radius for the 15 small `manual` labels (250 m used);
Table 3 vs Table 5 optimum PFAs.

### My CA-CFAR on this scene (Stage 2, Milestone 2a, 2026-10-27)

Stage 1's CA-CFAR as it is, on the same noTNR stack, mask, labels, scorer and scoring area, with
the paper's windows (9 / 13 px). It has no paper value, so there is no tolerance; the reference
for it is gamma, the closest of the six (a CA-type CFAR on a gamma clutter model), at F 0.636.

| | Best F (PFA) | TP / FP / FN | Precision | Small / medium / large found |
|---|---|---|---|---|
| Gamma (paper's code) | 0.636 (1e-6) | 255 / 55 / 237 | 0.82 | 9.9 / 64.6 / 91.2 % |
| My CA-CFAR, length rule on | 0.567 (1e-5) | 200 / 14 / 292 | 0.93 | 3.3 / 47.4 / 81.6 % |
| My CA-CFAR, length rule off | 0.587 (1e-1) | 220 / 38 / 272 | 0.85 | 6.1 / 52.0 / 86.8 % |

**It is precise but finds less.** Precision stays at 0.85-0.97 over all 21 PFAs; recall never
passes 0.45 (gamma: 0.52 at its optimum). The curve rises slowly and does not collapse at high PFA
like the reference detectors: from 1e-5 to 1e-1 it adds only 37 detections.

**Why: the HH co-pol check caps it.** A one-off diagnostic (length rule off, one rule switched
off at a time; `data/benchmark/faerch2023/score/cacfar_ablation*.csv`, not part of `bench-sweep`):

| Variant (length rule off) | Best F (PFA) | Detections at 1e-5 / 1e-3 |
|---|---|---|
| As run | 0.587 (1e-1) | 221 / 235 |
| No co-pol check | 0.609 (1e-7) | 795 / 16,981 |
| No 200 m edge buffer | 0.600 (1e-3) | 229 / 248 |
| No merging by grown structure | 0.586 (1e-1) | 221 / 235 |
| None of the three | 0.613 (1e-7) | 817 / 17,258 |

The check keeps a target only if its HH peak is 6 dB over the HH background. That is a second,
fixed threshold that doesn't move with the PFA, so at high PFA the HV test passes almost
everything and HH decides. It was added for 10 m HyP3 HV, where speckle grains pass a 2 px
minimum; at 40 m on this stack it costs 0.02 F at the optimum and flattens the curve. The
remaining gap to gamma (0.609-0.613 vs 0.636) is about what one channel (HV) costs against
gamma's HH AND HV test, at the same windows. The edge buffer costs ~0.01; merging nothing.
Revision C11b (2026-11-02, `docs/revision-ablation.md`): with HH AND HV instead of the check,
F 0.644 vs gamma 0.640 (paired difference +0.004 [-0.012, +0.018]). From there, fixed ENL 10.7,
the library's circular ring and its object handling each move F by at most ±0.013; in this
ablation the co-pol check accounts for the gap, and the other pipeline differences don't
measurably add to it on this scene.

**Length rule.** Off is better by 0.02 F, all of it in medium and large bergs (large 87 vs
82 %). This scene has no pack-ice mask; the rule only removes bergs here, as in 5d and 2b. At
1e-1 it costs more (F 0.43 vs 0.59): the low threshold joins CFAR pixels into longer grown
structures, which the rule then drops.

**PFA axis.** My ENL is measured on HV as 1 / median(CV)² over 500 m cells: 13.4, against the
10.7 the reference detectors take from Table 1 (2 km box: HV 11.5). A higher ENL lowers the
threshold at a given PFA, so my curve sits to the left of where gamma's would be with the same
ENL. F levels are comparable, PFAs only roughly.

**Bug found on the way.** `cfar.threshold_factor` used `scipy.stats.f.isf`, which returns inf
below PFA ~1e-16 (it works through 1 - PFA), so 1e-21 to 1e-17 gave no detections. It now
inverts the F tail through the incomplete beta function; Stage 1 PFAs (>= 1e-12) are unchanged
to 6 digits. 2b's 40 m CA-CFAR runs were redone; their matched-count recall was not affected.

**With thermal noise removal (the "as described" stack) the order changes.** Best F, TNR stack:
K 0.533, my CA-CFAR 0.494 (no rule) / 0.468, gamma 0.428, iDPolRAD 0.322, NIS 0.233,
log-normal 0.145, Wishart none (> 20,000 objects). My CA-CFAR loses ~0.09, gamma 0.21, NIS 0.42.
It measures its ENL on the scene (HV 1.05 here, 13.4 without TNR), so its threshold rises with
the rougher statistics; gamma and Wishart keep Table 1's 10.7. Gamma, NIS and iDPolRAD peak at
1e-21, the end of the sweep, so their TNR optimum may lie lower still. My remaining loss is the
clipped -115 dB pixels: false positives 81 vs 14, true positives 175 vs 200 at the optimum.

### THOR on this scene (2026-10-28)

`bench-thor`, report in `docs/faerch2023-thor.md`. The final linear head from Milestone 5,
unchanged: trained on 10 m HyP3 IW frames off Newfoundland (April-June), never on EW, 40 m or
July, and nothing tuned here except the score threshold, which is swept like the PFA. Crops of
5.12 km (128 px) give the training token grid (64 x 64 tokens of 80 m) in two ways: `upsample`
declares the bands 10 m and THOR resizes the crop to 512 px; `native` declares them 40 m and
cuts 2 x 2 px patches. This is the first precision number for THOR: the IIP frames have none.

**Result (revised 2026-11-02, shared evaluation mask).** Best F of the THOR pipeline (frozen
encoder + IW-trained head) without TNR, upsample: 0.572 at threshold 0.12, precision 0.83,
recall 0.44, verified-only recall 0.72. Like with like: my CA-CFAR with the length rule has
0.571 at precision 0.94, without it 0.591 at precision 0.85; Wishart 0.657, NIS 0.654 and gamma
0.640 are higher. On F the pipeline and my 6 dB CA-CFAR are not distinguishable (paired with
gamma: -0.068 [-0.097, -0.041] vs -0.069 [-0.097, -0.045]). With TNR, upsample 0.520, behind K
(0.537) and ahead of my CA-CFAR (0.497) and gamma (0.430). The current pipeline does not
transfer to EW: THOR itself was pretrained on IW and EW, but the head was trained on 10 m IW
only, and a head trained or adapted for EW is untested.

**Where it loses: medium bergs.** Recall at the optimum, small / medium / large: THOR pipeline
0.06 / 0.49 / 0.86, gamma 0.10 / 0.65 / 0.91, Wishart 0.09 / 0.72 / 0.93. The best run misses
47 of the 128 verified medium bergs in the scoring area; other runs (TNR, native, EW
normalisation) recover 13 of them, mostly the brighter ones (by HH contrast: < 4 dB 0 of 4,
4-8 dB 0.50 → 0.57 with any run, > 8 dB 0.73 → 0.85; `medium_contrast.csv`). Precision is as good as the
CFARs'. Small bergs are lost by everyone at 40 m.

**Upsample beats native** on both stacks (0.568 vs 0.512, 0.516 vs 0.482), mostly through
precision (0.83 vs 0.65 at similar recall): THOR does better seeing the crop at the 10 m grid
it was trained on than with a resized 2 x 2 px patch embedding.

**Thermal noise removal costs THOR 0.05 F**, against gamma's 0.21: it has no ENL to break. The
3-crop smoke test pointed the other way (TNR scored bergs higher); over the scoring area the
clipped HV floor gives 4x as many weak peaks (330k vs 84k above 0.02), so no TNR wins, as for
the CFARs.

**Normalisation.** THOR's own EW stats (HH -12.8 / 6.6 dB, HV -22.7 / 7.2) instead of the IW
stats the head was trained with: F 0.503 vs 0.568, medium recall 0.40 vs 0.49. The head is tied
to the inputs it learned on, so the medium-berg gap is not a normalisation offset.

**Calibration.** Scores are compressed: the best threshold is 0.12 (no TNR) or 0.25 (TNR),
against 0.35 in the viewer, so a THOR operating point does not transfer between products.

**HH AND HV instead of the 6 dB check (2c, 2026-10-30).** Tags `cacfar-and` / `cacfar-nolength-and`:
a CA-CFAR on HH as well, per pixel, both bands at sqrt(PFA) (the AND merge of sec. 2.1.1), each with
its own measured ENL (HV 13.4, HH 12.7). Best F without TNR: 0.639 at 1e-7 without the length rule
(TP / FP 251 / 42, precision 0.86; small / medium / large 9.4 / 63.4 / 90.4 %), 0.627 with it.
That equals gamma (0.636) and is within 0.015 of Wishart and NIS. With TNR: 0.495 / 0.466, the
same as the HV-only runs, so it keeps the robustness of the measured ENL. The curve now
follows gamma's shape, about a decade further left (higher ENL). Summary:
`docs/cacfar-validation.md`.
