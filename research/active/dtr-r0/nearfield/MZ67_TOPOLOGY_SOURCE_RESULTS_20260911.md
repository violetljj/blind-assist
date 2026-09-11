# MZ67: completed controlled topology source

The [registered source protocol](MZ67_TOPOLOGY_SOURCE_20260911.md) supplies
4096 retained frames, 1024 exact camera-relative geometry
IDs and 2048 invariant event-cell support pairs.
Retain this as COMPONENT_OR_CHALLENGER, mode COMPONENT, for source use only.
It provides no model-effect result: 0 fits/steps and
0 learned-model prediction frames were executed.

The original 64 canary cases remain included in the
4096 budget; main added 4032.
Both canaries were admitted before main. No outcome-selected replacement or
budget extension is authorized. The four target families are open bike stand,
tapered cone, hollow concrete pot and irregular concrete chunk. Their prototype
meshes and sites were already consumed as background. Controlled target placement,
scale/yaw/tilt and nominal topology do not establish natural installation,
unseen categories, through-hole sensing or hardware performance.

## Actual source and supervision

All 4096 frame IDs are unique. Source-valid frames:
4096/4096; floor-accepted:
4096/4096; native intent matches:
4096/4096. Capture/render checks cover
4096 rows. The explicit source-health status
is `ALL_FRAMES_HEALTHY`. UNKNOWN remains
0 query bits and 12370308 full-frame
cells. Missing or invalid packets are not clearance evidence. All actual rows,
extra range positives [0, 0, 0, 0], intent mismatches and native counts
are retained; failed observation is not relabeled as healthy.

| Role / family | Frames | Geometry IDs | Known BN/BF/HN/HF | Positive BN/BF/HN/HF |
|---|---|---|---|---|
| TRAIN_CANDIDATE/hollow_concrete_pot | 512 | 128 | [512, 512, 512, 512] | [128, 128, 128, 128] |
| TRAIN_CANDIDATE/irregular_concrete_chunk | 512 | 128 | [512, 512, 512, 512] | [128, 128, 128, 128] |
| TRAIN_CANDIDATE/open_bike_stand | 512 | 128 | [512, 512, 512, 512] | [128, 128, 128, 128] |
| TRAIN_CANDIDATE/tapered_cone | 512 | 128 | [512, 512, 512, 512] | [128, 128, 128, 128] |
| CALIBRATION/hollow_concrete_pot | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |
| CALIBRATION/irregular_concrete_chunk | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |
| CALIBRATION/open_bike_stand | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |
| CALIBRATION/tapered_cone | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |
| HELDOUT_GEOMETRY/hollow_concrete_pot | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |
| HELDOUT_GEOMETRY/irregular_concrete_chunk | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |
| HELDOUT_GEOMETRY/open_bike_stand | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |
| HELDOUT_GEOMETRY/tapered_cone | 256 | 64 | [256, 256, 256, 256] | [64, 64, 64, 64] |

Roles remain TRAIN/CALIBRATION/HELDOUT_GEOMETRY from the frozen joint recipe,
with 0 geometry-ID intersections. Packet range
and validity plus original RGB references are predictor inputs. Native depth,
full-frame counts/known cells, roles, family and actor metadata are loss/evaluator
information. Pair count tensors match exactly; valid background changes in
2048 pairs. This does not make background returns
direct target measurements.

## Actual tensor overlap

| Tensor | Query | Held configs | Positive | Any variant matches TRAIN | All variants match TRAIN | All-zero configs |
|---|---|---|---|---|---|---|
| counts | whole | 256 | 192 | 0 | 0 | 64 |
| counts | BODY_NEAR | 256 | 64 | 0 | 0 | 192 |
| counts | BODY_FAR | 256 | 64 | 0 | 0 | 192 |
| counts | HEAD_NEAR | 256 | 64 | 0 | 0 | 192 |
| counts | HEAD_FAR | 256 | 64 | 0 | 0 | 192 |
| presence | whole | 256 | 192 | 36 | 36 | 64 |
| presence | BODY_NEAR | 256 | 64 | 4 | 4 | 192 |
| presence | BODY_FAR | 256 | 64 | 12 | 12 | 192 |
| presence | HEAD_NEAR | 256 | 64 | 27 | 27 | 192 |
| presence | HEAD_FAR | 256 | 64 | 31 | 31 | 192 |

These are positive-only TRAIN references. An exact count match and a boolean
presence match are different statements. Whole tensors and each query are
reported separately; all-zero configurations do not establish positive novelty.
Support/site replicas collapse by geometry ID, retaining every label variant.
The measured generalization boundary is `NOT_ESTABLISHED_BY_SOURCE_AUDIT`;
disjoint metadata IDs do not prove disjoint native labels.

| Tensor | Role | Positive configs | Any match old TRAIN | All match old TRAIN | Has a novel variant |
|---|---|---|---|---|---|
| counts | TRAIN_CANDIDATE | 384 | 0 | 0 | 384 |
| counts | CALIBRATION | 192 | 0 | 0 | 192 |
| counts | HELDOUT_GEOMETRY | 192 | 0 | 0 | 192 |
| presence | TRAIN_CANDIDATE | 384 | 1 | 1 | 383 |
| presence | CALIBRATION | 192 | 0 | 0 | 192 |
| presence | HELDOUT_GEOMETRY | 192 | 0 | 0 | 192 |

The old reference is the sealed positive TRAIN pool from MZ55
(1600 frames) and MZ61
(2048 frames). These comparisons
are descriptive: no source row was removed, no role changed, and no new admission
rule was introduced. Full family/configuration tables remain in
[within-source collision evidence](../../../../artifacts.local/work/mz67-topology-source-20260911/combined-v1/tensor-collision-audit.json) and
[old-TRAIN collision evidence](../../../../artifacts.local/work/mz67-topology-source-20260911/combined-v1/old-train-collision-audit.json).

Root actually reviewed 80 reserved RGB/native cases,
separately from the 80 owner-local native audits
and 40 independent raw mask/depth pair checks. Image
agreement, opening/backface/material checks and specific visual notes are bound
in [the final visual review](../../../../artifacts.local/work/mz67-topology-source-20260911/visual-review-final-v1.json). Native audits
use the original fixed rows, not a post-outcome sample.

## Storage and next use

Reported original capture data: 7547614895 logical bytes;
compact training archives: 1703344513 bytes; original RGB:
1698086934 bytes. These representations overlap in content and are not
additive savings. Native raw remains owner-local. Transparent NTFS compression
separately saved 3206631424 allocated bytes across
8192 exact native/support files, with 5663358976 logical
bytes preserved. See [execution and storage evidence](MZ67_EXECUTION_20260911.md).

The existing [weak-input census](../../../../artifacts.local/work/mz67-topology-source-20260911/weak-input-availability-v1/report.md) uses
old MZ61 TRAIN observations; the
[MZ66 saved-output diagnostic](../../../../artifacts.local/work/mz66-replay-projection-20260911/calibration-diagnostic-v1/report.md)
identifies calibration-margin/retention questions. They motivate separately
declared model mechanisms and do not show an effect from MZ67. The separately
registered MZ68 uses old MZ61/MZ48, without MZ67 data; this source record supplies
no MZ68 effect result. Preserve prior sources/models and
register any future use independently. No natural-scene, Android/default-app,
VL53L8CX calibration or safety promotion follows from source admission.
