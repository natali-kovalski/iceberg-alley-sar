# Validation: how accurate is the detector?

> **Note (Milestone 5).** This page validates the CA-CFAR, which is now the project's baseline.
> The main detector is THOR; see [thor-vs-cfar.md](thor-vs-cfar.md). The 78% below counts CFAR
> detections near pack ice too. The THOR comparison drops detections within 5 km of pack ice for
> both detectors, which gives the CFAR **74%** (197 / 267) on this pass and 74% pooled over four
> passes, against THOR's 89% and 84%. Compare THOR with the 74%, not the 78%.

**Headline (CFAR, before THOR):** on a Sentinel-1 pass where International Ice Patrol (IIP) analysts marked every
iceberg they identified, the detector found **78% of them (207 of 267)**, with a median
position difference of **67 m**. Small bergs (15-60 m) are found 65% of the time, medium
bergs (61-120 m) 81%.

![Detections vs. IIP labels, 2019-04-29](img/validation_20190429.png)

## The ground-truth problem

Iceberg ground truth off Newfoundland is scarce, and each source has a catch. We tried four:

| Source | What it gives | Usable for | Outcome |
|---|---|---|---|
| NAIS daily iceberg chart | Estimated count per 1° square (~7,800 km²), 00 UTC | Order-of-magnitude check | Totals within 2-4x, but errors cancel (see below) |
| IIP sightings, **same Sentinel-1 pass** | Berg positions marked by IIP analysts on the scene we process | Per-berg recall, no drift | **Headline result** |
| IIP sightings, **aircraft** | Positions from reconnaissance flights | Independent recall | Inconclusive: 5-7 h later, drift unresolved |
| Sentinel-2 optical | Bright objects in open water, 10 m | Independent precision and recall | Spring: loose sea ice dominates. June: targets almost all inshore, inside the land mask |

The IIP sightings database (NSIDC G00807) ends with the 2021 season, so per-berg validation
uses spring 2019 scenes rather than the 2025 scenes shown in the viewer. Same pipeline, same
parameters.

## Headline result: 2019-04-29

Scene: Sentinel-1A IW, 2019-04-29 09:41 UTC, HH/HV, HyP3 RTC 10 m. IIP lists 276 sightings
from this pass (source `SNL1`, all high confidence); 267 fall on unmasked water and are compared.
A detection matches a sighting within 500 m, one-to-one. CFAR cores that belong to one bright
blob are merged into a single target (`merge_by_structure`); before this, ships and some bergs
split into two detections ~50 m apart, which inflated the count from 534 to 687 for one lost match.

| | Value |
|---|---|
| Recall, all detections | **78%** (207 / 267) |
| Recall, open-water detections only (> 5 km from pack ice) | 74% (197 / 267) |
| Median match distance | 67 m (137 m before correcting IIP positions, see method notes) |
| Detections without an IIP label (in the labelled area) | 324 of 531 |

**Recall by distance from the coast is stable** (73-84% from 0 to 100 km), so the detector is
not only working in easy offshore water.

**Large bergs are missed (0 of 4 over 120 m).** Each has a detection 1-4 km away but none on
the berg. Likely causes are the 300 m structure filter (tuned to reject sea-ice strips) and the
CFAR guard window being smaller than a large berg, which puts part of the berg into its own
background estimate. Not yet fixed.

**Precision cannot be measured with this source.** 324 detections have no IIP label. They are
strong targets (median contrast 16.9 dB vs. 18.7 dB for matched ones), similar in size, mostly
away from pack ice, and concentrated in the western part of the scene among labelled bergs.
They are a mix of bergs IIP did not record, fishing vessels, and false alarms, in unknown
proportions. **39% (207 / 531) is a lower bound on precision, not an estimate.** Separating
them needs AIS vessel positions for the same pass.

**This check is not independent.** IIP analysts looked at the same Sentinel-1 image. It tells us
the detector agrees with expert interpretation of the radar data, not that it agrees with the
ocean.

## What did not work, and why

**NAIS chart counts (2025 scenes).** Expected counts (chart count scaled by scene coverage)
vs. open-water detections: 26 vs. 106, 31 vs. 98, 71 vs. 137. The totals hide two opposite
errors: inshore of 54°W SAR finds 3-8x more than the chart; offshore it finds fewer on 05-02 and
05-14, where the HV sea background is 3-4 dB higher (rougher sea), so small bergs lose contrast.
Charts are per 1° square, partly modelled ("reduced reconnaissance" on all three dates), and
include growlers too small for SAR. They can rule out gross errors, not measure accuracy.

**Sentinel-2 on 2019-04-15.** The clear tiles were full of loose floes and brash ice: up to
37,800 isolated bright objects per tile. At 10 m, optical imagery cannot tell a small berg from
an ice floe, so this is not iceberg truth. June scenes (pack ice gone, bergs still present) avoid
this: on 2020-06-10, 30 isolated targets were extracted and visually confirmed as bergs.

**Sentinel-2 on 2020-06-10, against the matching radar frame.** Not usable for precision. The
30 optical bergs sit in coves and among islands of inshore Bonavista and Notre Dame Bay: 18 fall
on OSM land polygons (coastline error, or bergs grounded against the shore) and 8 more within
the 500 m land buffer. Only 3 are on valid radar pixels, against 4 detections in the clear
optical area. With a 5 h gap and so few points the drift cannot be estimated ("drift
unresolved"), and none match. The radar scene itself is quiet: 9 detections over the whole
frame (531 on the same frame on 2019-04-29), and IIP analysts marked only 2 bergs on this pass.
One of those is on valid water and was missed. CFAR does fire on it (HV 17 dB over background,
HH 22 dB), but it is a single 10 m pixel in HV and `min_area_px: 2` removes it. One case is not
enough to change the filter, but it is the same small-berg weakness seen in the size table.

**Aircraft sightings on 2019-04-15.** 25 sightings fall on valid radar pixels, 5.5-7.5 h after
the pass, all stamped with the same time. Bergs drift a few km in that time, and with only 12
open-water detections there were too few pairs to estimate the drift. Disabling each detection
filter in turn did not change the result, so the filters are not the cause. The validation code
now reports "drift unresolved" instead of scoring against a chance alignment.

## Method notes

- `validate-points` compares only targets both sources could see: truth points must sit on
  valid radar pixels (away from land, pack ice and nodata); with optical truth, detections must
  fall in the clear part of the optical image.
- When truth is not simultaneous, a common drift offset is estimated (the shift that aligns the
  most points) and compared with what a random shift achieves. Weak peaks are rejected.
- Matching is one-to-one (Hungarian assignment), so one detection cannot claim two bergs.
- **IIP positions are truncated, not rounded,** to 0.1 arc-minute (all satellite sightings and
  72-92% of aircraft/ship ones lie on that grid). On 2019-04-29 the bergs sat on average 101 m
  north and 67 m west of their IIP positions: half a grid step (93 m / 60 m at 49.5 N).
  `load_truth` moves on-grid positions to their cell centre, which halves the median match
  distance (137 m to 67 m). Recall does not change: the 500 m match radius already absorbed it.

## Reproduce

The figure above is regenerated from current outputs with `python -m iceberg_sar.cli figures`.

```powershell
python -m iceberg_sar.cli order S1A_IW_GRDH_1SDH_20190429T094108_20190429T094133_027003_030A4E_8BD8
python -m iceberg_sar.cli download
python -m iceberg_sar.cli preprocess data\raw\<product_dir>
python -m iceberg_sar.cli detect data\raw\<product_dir>
python -m iceberg_sar.cli iip-sightings 2019
python -m iceberg_sar.cli validate-points data\raw\<product_dir> --truth iip-satellite --no-open-water
```

`<product_dir>` is the unzipped HyP3 product, e.g. `S1A_IW_20190429T094108_DHP_RTC10_G_gpuned_671B`
(the suffix differs per order). Outputs go to `data/outputs/validation/`.

## Next steps

1. Fix large-berg recall (structure filter and guard window scaled to target size).
2. AIS vessel positions for the same pass, to turn the precision lower bound into a number.
3. Independent precision still needs a June Sentinel-2 date with bergs offshore of the land
   buffer (2020-06-10 had them almost all inshore).
