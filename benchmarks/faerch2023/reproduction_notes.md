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
