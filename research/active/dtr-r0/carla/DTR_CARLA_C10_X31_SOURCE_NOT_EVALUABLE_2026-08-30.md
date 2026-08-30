# DTR CARLA C10 X31 source result — 2026-08-30

## Decision

`DTR_CARLA_C10_X31_OPAQUE_SINGLE_PASS_SOURCE_NOT_EVALUABLE_RASTER_PERMEABLE_OCCLUDER`

C10 fixed C9's occluder/wearer route intersection and completed all four
frozen sensor shards, but it did not reach model admission.  The joined source
failed exactly `track_then_complete_physical_occlusion_contract_met`: the
side-facing Fusorosa bus satisfied the preregistered 3-D OBB containment
windows while target instance labels remained visible through its rendered
silhouette.  No X24 index, detector run, X24 prediction, X31 prediction, or
score was started.

## Frozen identity

- Git commit: `f6ac7728d87cdf9d8e565b5e9b394f99b536f4d8`
- Run ID: `c10-x31-opaque-20260830-203500`
- Cohort: `DTR_CARLA_C10_X31_OPAQUE_SINGLE_PASS_SOURCE_DISJOINT_V1`
- Capture seed: `130365`
- Protocol SHA-256:
  `255402227C86133B95706F7B49957F9AB463A146EDC3D266853381794BC5078B`
- X31 predictor SHA-256:
  `28E777021FC6AA39129480B069B544B562F559FD447CCE164706B6DCAC3A9F18`
- Source root:
  `E:\\linnan\\CARLA\\experiments\\dtr-carla-c10-x31-opaque-single-pass\\evidence\\c10-x31-opaque-20260830-203500`

## What C10 established

All four `instance`, `wearable`, `depth`, and `witness` shards completed.  Each
contains `728` payloads at `1280x720`, for `2,912` raw payloads total.  All
expected CONTACT/SAFE outcomes and responsible sets matched, all scripted
poses matched their authoritative transforms, all RGB/depth alignments passed,
all blueprints and bounding boxes passed, and no blueprint fallback occurred.
The C9 collision-contamination failure is therefore causally removed by the
C10 route geometry.

The joined source materialized `728` truth-blind model observation frames,
sealed `1,484` model-package files and `6,041` evidence files, and passed every
source check except physical complete occlusion.  Those packages are source
artifacts only; no algorithm model inference or metric evaluation ran.

## Source-admission failure

The frozen complete-occlusion threshold is exactly zero target instance pixels
for one `6..13` frame run, with at least ten trackable frames before and eight
after.  Target pixels inside each preregistered OBB-containment window were:

| Episode | OBB-contained frames | Target pixels in those frames | Eligible run |
| --- | --- | --- | --- |
| `ep_01` | `12..18` | `347, 451, 387, 434, 582, 652, 543` | none |
| `ep_02` | `11..18` | `466, 0, 368, 407, 435, 0, 477, 1537` | isolated `12`, `16` |
| `ep_03` | `12..18` | `114, 237, 390, 455, 408, 392, 663` | none |
| `ep_04` | `11..17` | `91, 167, 378, 495, 195, 374, 525` | none |
| `ep_05` | `12..18` | `2713, 130, 311, 429, 328, 318, 869` | none |
| `ep_06` | `13..19` | `4285, 39, 274, 739, 655, 422, 1641` | none |
| `ep_07` | `12..18` | `20, 122, 243, 320, 254, 292, 312` | none |
| `ep_08` | `11..17` | `0, 41, 186, 154, 70, 259, 375` | isolated `11` |

The bus itself remained strongly present during the crossings, with roughly
`212k..512k` instance pixels.  Thus this is not bus absence, pose timing,
route clearance, collision contamination, or a threshold defect.  Persistent
target labels inside the projected bus OBB prove a source-representation
opacity failure.  Windows/openings are the likely mesh mechanism, but that
mechanism remains an inference because no PNG was inspected.

The frozen report records no runs for `ep_01`, `ep_03` through `ep_07`; `ep_02`
has two `0.1 s` runs with pre/post counts `11/3` and `3/21`; `ep_08` has one
`0.1 s` run with pre/post `11/0`.  All eight selected runs are null.

## Evidence and cleanup

- Joined source result SHA-256:
  `CD8F1C6043B4687044CF568C626ADCF54359DE63B35EA6A5317007C4721A363B`
- Sealed evidence manifest SHA-256:
  `6BD470306F0A811A3BDD6B1C4AF8102F6C740777D6951E6A3FB1150ADCE1D06F`
- Sealed source-model manifest SHA-256:
  `E541420FE969B7159AF05460F09F5EB56FB8FD51EAEC9AA6582059154D9E2372`
- Join stderr SHA-256:
  `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`
- Captured tree: `6,063` files, `6,476,817,590` bytes

The runner cleaned its owned CARLA processes and ports `2000..2002`; both
process and listener counts were zero after failure.  The C10 source tree is
retained as terminal failure evidence and must not be deleted, resumed,
replayed, reseeded, partially selected, modeled, predicted, or scored.

## Next legal route

Use a fresh source identity and seed.  Make one representation change: replace
the raster-permeable bus with the already observed solid-body firetruck while
keeping targets, aliases, wearer, map, weather, collision policy, X31,
detector, route thresholds, and score gates frozen.  Match the single lateral
pass to the frozen `0.6..1.3 s` dwell analytically; formal fresh instance pixels
remain the authority.

## Claim boundary

C10 is `NOT_EVALUABLE`.  It proves the route-collision fix and isolates a
raster-opacity source defect, but supplies no Development metric,
confirmation, generalization, product, deployment, or safety evidence for X31.
