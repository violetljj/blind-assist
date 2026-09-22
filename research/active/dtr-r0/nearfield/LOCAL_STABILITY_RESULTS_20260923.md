# LOCAL retains partial gains, but fails composite/trajectory stability

**LOCAL_COMPOSITE_TRAJECTORY_STABILITY_NEGATIVE.** The frozen LOCAL component
still recovers useful detections on the new source, but its added false-alert
costs exceed the predeclared budget and early benefit is uneven across paths.
Retain the old LOCAL component in its previous scope; this broader transfer is
a scoped NEGATIVE_CONTROL, not evidence for a reliable general enhancement.

The [fixed protocol](LOCAL_STABILITY_PROTOCOL_20260923.md) was applied without
training, model/threshold/feature changes, scene exclusions or a rescue run.
All five governed stages succeeded and the independent audit passed. This is
a valid negative stability result, not a capture or label-admission failure.

## Full result

576 new controlled frames, 48 clips, eight composite geometries: four families
with small/large sizes, three lateral placements and two posed trajectories.
Both trajectories have 112 positive/176 negative frames and 16 positive events;
the whole source has 224 positive/352 negative frames and 32 events. All 3456
query labels are valid. Ground truth is the union of each rendered component
cube, never an enclosing box that fills a composite shape's holes.

| Frozen arm | TP / FP / FN | Recall | Precision | FPR | Events | False segments / sampled duration |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A current | 107 / 12 / 117 | 47.77% | 89.92% | 3.41% | 22/32 | 12 / 2.4 s |
| A OR RAW | 109 / 16 / 115 | 48.66% | 87.20% | 4.55% | 23/32 | 13 / 3.2 s |
| A OR LOCAL | **119 / 20 / 105** | **53.13%** | 85.61% | 5.68% | **27/32** | **18 / 4.0 s** |
| RAW standalone | 18 / 4 / 206 | 8.04% | 81.82% | 1.14% | 9/32 | 2 / 0.8 s |
| LOCAL standalone | 103 / 13 / 121 | 45.98% | 88.79% | 3.69% | 27/32 | 11 / 2.6 s |

LOCAL versus A gains 12 true frames and five previously missed events. Three
already detected events advance by 0.2, 0.2 and 0.4 s. It also adds eight false
frames and six false segments. All A alerts and onsets are retained by the OR
rule; this is structural retention, not a separately learned capability.

LOCAL versus RAW gains ten true frames net, but loses two RAW-only true frames
on large T-bar INSIDE cases, frame 04, one per trajectory. Net recall gain is
4.46 points, below the fixed five-point floor. Five of eight groups gain and
both BODY/HEAD recall noninferiority checks pass, but LOCAL adds four false
frames and five false segments over RAW, above the +1/+1 comparison costs.

## Earlier detection is real but uneven

| Trajectory, 288 frames each | A TP / FP / FN; events | LOCAL TP / FP / FN; events | Benefits versus A | Added false frames / segments |
| --- | --- | --- | --- | --- |
| Approach then return | 67 / 9 / 45; 16/16 | 69 / 14 / 43; 16/16 | 2 earlier | +5 / +3 |
| Approach, dwell, return | 40 / 3 / 72; 6/16 | 50 / 6 / 62; 11/16 | 5 recovered, 1 earlier | +3 / +3 |

Each trajectory has 11 events where A does not alert at entry, so the timing
test has sufficient opportunity. The total eight benefits meets its aggregate
floor, but the approach-return path supplies only two, below the required three
per trajectory. The timing clause therefore fails, rather than being declared
NOT_EVALUABLE. Total and both per-trajectory false-alert clauses also fail.
The complete three-clause stability decision is **FAIL**.

All five recovered events are in the dwell schedule. Four are BOUNDARY cases
(small/large L plate, small depth step, small inverted L); the fifth is small
inverted-L INSIDE. Their first alerts occur at 0.6--1.4 s, compared with a true
entry at 0.6 s. Thus event recovery does not imply uniformly early detection.
The three advances are depth-step BOUNDARY cases. Full per-event records remain
in metrics.json, including missed events and every first-alert timestamp.

Mean positive-event coverage increases from A's 47.77% to LOCAL's 53.13%, but
minimum coverage remains zero because five events are still missed. Internal
gaps between a first and last positive alert are A 5 segments/8 frames and
LOCAL 7/12. These counts do not mean LOCAL removed A alerts: extra detections
can expand the span in which a silent sample is called an internal gap. Coverage,
leading/trailing misses and complete event records accompany the gap counts.

UNKNOWN remains 545/576 for all arms. LOCAL alerts on 108 UNKNOWN frames and
is silent on 437, including 105 known-positive and 332 known-negative frames.
The latter remain abstentions, not measured free space. No mask is predicted;
mask IoU is not an output metric for this frozen scalar classifier.

## Where gains and costs occur

| Shape, 144 frames / 56 positive each | A TP / FP / FN | LOCAL TP / FP / FN |
| --- | --- | --- |
| L plate | 27 / 4 / 29 | 29 / 4 / 27 |
| Depth step | 43 / 7 / 13 | 50 / 13 / 6 |
| Inverted L plate | 24 / 1 / 32 | 27 / 3 / 29 |
| T bar | 13 / 0 / 43 | 13 / 0 / 43 |

| Stratum | Frames | A TP / FP / FN | LOCAL TP / FP / FN |
| --- | ---: | --- | --- |
| Small size | 288 | 43 / 2 / 69 | 52 / 4 / 60 |
| Large size | 288 | 64 / 10 / 48 | 67 / 16 / 45 |
| BODY | 288 | 70 / 11 / 42 | 79 / 17 / 33 |
| HEAD | 288 | 37 / 1 / 75 | 40 / 3 / 72 |
| INSIDE | 192 | 77 / 4 / 35 | 79 / 8 / 33 |
| BOUNDARY | 192 | 30 / 2 / 82 | 40 / 4 / 72 |
| OUTSIDE | 192 | 0 / 6 / 0 | 0 / 8 / 0 |

The six added depth-step false frames all occur at frame 02, with the target's
foremost surface at **3.15 m**, before the fixed 3 m query entrance. Two are
small INSIDE (both paths); four are large INSIDE/BOUNDARY (both paths). The other
two added false frames are large inverted-L OUTSIDE on approach-return, frames
06/07. This distinguishes range-entry errors from lateral OUTSIDE errors; all
eight remain false positives under the frozen task definition. It neither
changes the cost gate nor establishes what feature caused the HGB score.

The small-size arm preserves a larger gain (+9 TP/+2 FP) than the large-size
arm (+3/+6). This descriptive contrast is not a newly selected operating scope.
All eight base groups, RAW controls and individual losses remain reported; no
favorable subgroup replaces the failed full test.

## Interpretation and evidence boundary

The previous strong transfer was over solid cuboids. Here both A and the
learned branches struggle more, while LOCAL still recovers five events. This
narrows the claimed capability: useful on some controlled layouts does not
establish stable early detection over new geometric compositions and paths.
The result does not isolate whether composition, visible area, size or depth
schedule causes the change, because those source conditions jointly differ.
The four shapes and eight geometries are selected procedural tests, not a
sample of natural occurrence frequencies or a general failure-rate estimate.

All geometry is still made from opaque axis-aligned cubes using the same map,
renderer, material assets and hypothetical single-return sensor law. The target
and backdrop materials stay fixed within each family across sizes/paths. All
samples are settled static poses, not continuous video, real velocity, motion
blur, physical ToF dwell behavior or endpoint latency. Nominal 0.2 s onset
improvements therefore remain simulated sampling evidence.

Retain the previous LOCAL COMPONENT and A/UNKNOWN, and record this exact broader
stability test as NEGATIVE_CONTROL. Preserve both the event-recovery benefit
and its range-entry/lateral costs. Further work needs a justified different
mechanism or source question; this result does not authorize threshold changes,
source pruning or refitting a rescue on the consumed cohort.

## Execution, audit and artifacts

Artifact root: `artifacts.local/evidence/ba-local-stability-20260923`, with plan,
source-only comparison to three prior specs, fixed stage source snapshots and
governed RunSpecs. Sibling `-capture`, `-prepared`, `-predictions`, `-evaluated`
and `-audit` contain all receipts, hashes, predictions and measurements.

Capture: 576/576 frames, 342.64 s on the logged D3D12 adapter 0, NVIDIA GeForce
RTX 5060 Laptop GPU. Map/source unchanged and every task-created actor/process
released. Preflight found no other Unreal process, 6688 MiB free GPU memory and
about 108.5 GB free on F:. No secondary or paid worker was allocated.

Materialization: 16.34 s, all 3456 query labels valid and all 1728 rendered
component bounds within 2 mm. Fixed-model prediction/sealing: 13.76 s overall,
including 8.99 s for both feature branches and 1.32 s for A. Actual CPU backend
records TASK_NOT_GPU_SUITABLE for the retained HGB implementation. These are
host batch timings, not a RAW-versus-LOCAL speed comparison or application latency.

Six focused unittest cases pass, including ten public-identity/extraction checks;
the independent auditor's seven synthetic fixtures also pass. The governed
audit passes **173,032 checks**, independently reproducing all 576 rendered
cuboid-union labels, all five arms' saved readouts, full and subgroup metrics,
event onsets, costs and the three decision clauses. It verifies 48 native
depth frames, one fixed entry frame per clip including OUTSIDE controls;
the other native validity flags are tied to sealed visibility accounting.
It verifies old model, selection and feature-code identities without repeating
model inference or proving the feature/model numerical implementation anew.
Coverage/gap summaries use the primary tested routine; they are descriptive,
not an independently replayed temporal model. All five science stages succeeded
on their first attempts, with zero fits or cutoff selections and no resources
left running for this task.
