# DTR CARLA C11 X31 source result — 2026-08-30

## Decision

`DTR_CARLA_C11_X31_SOLID_BODY_SOURCE_NOT_EVALUABLE_PARTIAL_RASTER_OCCLUSION_1_OF_8`

C11 completed all four frozen CARLA sensor shards and removed most of the C10
bus-permeability failure, but it did not reach model admission.  The joined
source failed exactly `track_then_complete_physical_occlusion_contract_met`:
the firetruck produced complete target disappearance in several central poses,
yet only `ep_08` supplied one `0.6..1.3 s` zero-pixel run with the required
trackable context before and after.  No X24 index, detector run, X24 prediction,
X31 prediction, or score was started.

## Frozen identity

- Git commit: `289676fe8f80e002fbdfb943b46d088129fb4c47`
- Run ID: `c11-x31-solid-20260830-211827`
- Cohort: `DTR_CARLA_C11_X31_SOLID_BODY_SOURCE_DISJOINT_V1`
- Capture seed: `131365`
- Protocol SHA-256:
  `744320C57ABBCD25835155760C15E83A421D66FC5DEDC35FB07F5F30BE0E5ACD`
- X31 predictor SHA-256:
  `28E777021FC6AA39129480B069B544B562F559FD447CCE164706B6DCAC3A9F18`
- Source root:
  `E:\\linnan\\CARLA\\experiments\\dtr-carla-c11-x31-solid-body\\evidence\\c11-x31-solid-20260830-211827`

## What C11 established

All four `instance`, `wearable`, `depth`, and `witness` shards completed.  Each
contains `728` payloads at `1280x720`, for `2,912` raw payloads total and `75`
unique actual blueprints.  CONTACT/SAFE outcomes and responsible sets matched,
scripted poses and collision policy matched, RGB/depth alignment passed, all
payload inventories passed, and no blueprint fallback occurred.  Every joined
source check except physical complete occlusion passed.

The joined source materialized `728` truth-blind observation frames, sealed
`1,484` model-package files and `6,041` evidence files.  Those are source
packages only; no algorithm inference or score exists.

## Source-admission result

The unchanged authority requires exactly zero target instance pixels for one
contiguous `6..13` frame run, with at least ten consecutive trackable frames
before and eight after.  At `1280x720`, the frozen trackable fraction `0.0002`
corresponds to at least `184.32` target pixels.

| Episode | Predicted OBB-contained frames | Target pixels in those frames | Formal zero-pixel run | Pre/post trackable frames | Result |
| --- | --- | --- | --- | --- | --- |
| `ep_01` | `14..19` | `437, 97, 62, 72, 88, 379` | none | — | fail |
| `ep_02` | `13..19` | `254, 0, 66, 60, 72, 90, 0` | isolated `14`, `19` | `13/0`, `0/18` | fail |
| `ep_03` | `14..19` | `549, 225, 0, 0, 0, 296` | `16..18` (`0.3 s`) | `15/25` | duration fail |
| `ep_04` | `13..19` | `0, 0, 0, 0, 0, 0, 0` | `13..19` (`0.7 s`) | `0/21` | pre-track fail |
| `ep_05` | `14..19` | `122, 362, 444, 1, 23, 448` | none | — | fail |
| `ep_06` | `16..21` | `170, 0, 26, 212, 4192, 7058` | isolated `17` | `0/0` | fail |
| `ep_07` | `14..19` | `338, 181, 0, 0, 0, 261` | `16..18` (`0.3 s`) | `0/26` | fail |
| `ep_08` | `12..19` | `0, 0, 0, 0, 0, 0, 0, 0` | `12..19` (`0.8 s`) | `12/21` | pass |

The firetruck itself remained strongly visible during the predicted windows,
with about `197k..502k` instance pixels.  C11 therefore improved complete
occlusion coverage from C10's `0/8` to `1/8`, and produced central zero-pixel
runs in four additional episodes, but it did not meet the frozen source gate.

`ep_04` shows why OBB containment still overstates raster authority: frame 12
retained only `63` target pixels, below the `184.32` trackable threshold, before
the valid seven-frame disappearance.  The contact sheet was inspected after
the terminal was sealed.  It shows a close foreground firetruck whose rendered
silhouette has wheel, cab, front/rear, and edge structure rather than an opaque
rectangle.  This agrees with the instance evidence: central body poses can hide
the target completely, while entry/exit poses leak small target fragments.

## Evidence and cleanup

- Joined source result SHA-256:
  `8B730BA1A0D5BE7BFF87750775FA4193E5BA3BB00F44A6D271C2073AFE499AAE`
- Physical-occlusion report SHA-256:
  `64F5361F68A9C142C674711DBB9576E10069FFD95418C51EB59854E803D0C210`
- Sealed evidence manifest SHA-256:
  `3E9ABBDD03E1391EBFDFE1D36F61BDFBF97F61FECBE3B46008490907D1F30DB1`
- Sealed source-model manifest SHA-256:
  `D9A69FA0013CA365AD31A47842849E6D2E2030A25463251A0D6CCD806FD67CB0`
- Captured tree: `6,063` files, `6,513,176,445` bytes

The runner released CARLA and ports `2000..2002`.  The C11 source tree is
terminal evidence and must not be deleted, resumed, replayed, reseeded,
partially selected, predicted, or scored.

## Next legal route

Do not tune the zero-pixel or trackability thresholds.  Use a fresh source
identity and one raster-representation change whose prelaunch check is based on
the rendered instance mask rather than a 3-D OBB.  The preferred direction is a
CARLA-native kinematic opaque prop or an explicitly modeled opaque shell; the
exact asset must be reachability-probed before freezing a C12 cohort.

Once an evaluable source exists, the next algorithmic structural candidate is
a per-surface-cell causal flow memory: transport authorized RGB-D surface cells
with local flow, prune them with current free-space rays, and keep any
never-observed hidden hypothesis in a separate provenance state.

## Claim boundary

C11 is `NOT_EVALUABLE`.  It proves that the firetruck representation is a real
opacity improvement over the bus and isolates the remaining raster-silhouette
defect.  It supplies no X24/X31 Development metric, confirmation,
generalization, product, deployment, user-benefit, reliability, or safety
evidence.
