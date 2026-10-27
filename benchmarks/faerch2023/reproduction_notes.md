**Verdict (2026-10-27).** With the paper's own labels (Path A), five of six detectors reproduce
the paper's best F within 0.05: gamma +0.011, Wishart -0.006, NIS -0.008, log-normal -0.017,
K -0.019. iDPolRAD is 0.061 below. The shapes agree with the paper's sec. 4: "The K detector and
the log-normal detector achieve their maximum accuracy at a very narrow interval of PFA levels,
whereas the gamma, iDPolRAD, NIS, and Wishart generally exhibit higher F score across a wide
range of PFA levels." Fig. 6 is not digitized, so the comparison of shapes is by eye.

**Deviation: no thermal noise removal.** The paper lists thermal noise removal among its SNAP
steps. With SNAP 14 that step breaks the detectors' assumptions (grey curves, last table column:
every detector stays at F 0.53 or below, Wishart produces >20,000 objects even at 1e-21):

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

**iDPolRAD (outside tolerance, -0.061).** Same number of false positives as the paper (70 vs 74),
fewer true positives (162 vs 193), mostly medium bergs (31 % vs 41 %). iDPolRAD's threshold
comes from one generalized-gamma fit over the whole image (10,000 random pixels, `cfar_api_notes.md`),
so it depends on what the image contains: our land mask (OSM coastline + 500 m), swath and
noise-floor handling differ from the paper's. Its 57 px training window is cropped to the
library's 41 px, as it must have been in the paper. The flat start of the curve (1e-21 to 1e-17,
identical detections) is the global threshold saturating.

**What else holds.**
- Labels are not the issue: Path A reproduces Table 2 exactly (181 / 175 / 136) and the mean
  verified drift (3954 vs 3953 m).
- Scoring area: the cloud-free Path B area changes F by at most 0.011.
- Verified-only F is 0.81-0.87 at each optimum: most misses are interpolated (1789 m) labels,
  i.e. small bergs whose position is uncertain.
- Size breakdown at my optimum follows Table 5 (large 87-93 %, small 4-10 %), except iDPolRAD.
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
