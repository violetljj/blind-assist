# MZ77 fixed-model simulated approach result

Decision: `MZ77_APPROACH_NO_DYNAMIC_GAIN_PARTIAL_GAP_INTERRUPTION`.
Original DIVERSE supplies no earlier correct warning or extra true query bits over OLD_NEG in this cohort. It adds five HEAD_FAR false bits in the head-bar clip. Preserve both frozen models and cuts; retain this sequence harness as a diagnostic component, with no model promotion or automatic successor.

## Scope and execution

[Frozen protocol](MZ77_APPROACH_SEQUENCE_20260911.md): four clips (head bar, thin pole, wall, no-added-target control), 40 settled poses each, front distance 4.5 to 0.6 m. The nominal 10 Hz / 1 m/s coordinates describe sampled approach positions, not measured walking, contact, latency or a real-time sensor stream. Targets use simple cube geometry and default material; the bar is a geometric fixture without a physical mount claim.

All 160 frames passed source readiness, file hashes, native depth structure and bounds checks (120 target frames, 40 controls). Twelve montage views at indices 0, 20 and 39 were visually inspected. The initial launcher failed its project-path preflight with zero frames; the corrected launcher used the same frozen specification and collector. Capture took 601.617 s including initial shader compilation. Worker exit was zero, with no retained task process or listener.

Generic simulated 8x8 two-return packets were sealed before predictor execution. CENTER_GAP removes both returns in central columns 2..5 at indices 20..22 and 30..32; 406 originally valid return slots were removed across the cohort. A float32 packet correction was recorded before inference in the v2 freeze receipt. These are synthetic availability interventions, not calibrated VL53L8CX physics. RGB and packet values are the predictor inputs; native depth, target bounds and truth are evaluator-only.

Frozen RGB, ToF, MZ5, MZ37, OLD_NEG and original MZ70 DIVERSE were evaluated with original cuts and explicit RGB NCHW. A 16-frame historical parity check passed before new inference. CUDA sensor generation took 5.950 s; full inference including parity took 11.973 s. CUDA native-label scoring plus scalar metrics and plotting took 10.715 s (scalar metrics 0.118 s). These batch timings are not deployment latency. Eight sequence-metric tests passed; an independent scalar loop checked all 7,680 original four-query decisions.

## Paired outcomes

Each condition has 130 positive and 510 negative native query bits over 160 frames. Counts below are TP / FP / FN / TN, with the original four BODY/HEAD near/far queries kept separate in the machine-readable result.

| Method | CLEAN | CENTER_GAP |
| --- | --- | --- |
| RGB | 22 / 64 / 108 / 446 | 22 / 64 / 108 / 446 |
| ToF | 52 / 49 / 78 / 461 | 44 / 44 / 86 / 466 |
| MZ5 | 40 / 50 / 90 / 460 | 32 / 50 / 98 / 460 |
| MZ37 | 63 / 26 / 67 / 484 | 47 / 23 / 83 / 487 |
| OLD_NEG | 79 / 28 / 51 / 482 | 60 / 25 / 70 / 485 |
| DIVERSE | 79 / 33 / 51 / 477 | 60 / 30 / 70 / 480 |

For the following union diagnostics, each relevant BODY_ANY or HEAD_ANY episode has 26 positive samples, indices 14..39. Near/far union coverage is not correct distance-bin classification. DIVERSE and OLD_NEG have identical true-hit indices in both conditions.

| Clip / union | First true distance CLEAN -> GAP | Positive hits CLEAN -> GAP |
| --- | --- | --- |
| Head bar / HEAD_ANY | 3.1 -> 3.1 m | 26/26 -> 21/26 |
| Thin pole / BODY_ANY | 3.1 -> 3.1 m | 23/26 -> 17/26 |
| Thin pole / HEAD_ANY | 3.1 -> 3.1 m | 26/26 -> 20/26 |
| Wall / BODY_ANY | 2.4 -> 2.2 m | 9/26 -> 7/26 |
| Wall / HEAD_ANY | 1.6 -> 1.6 m | 11/26 -> 8/26 |

Head bar: DIVERSE's extra five HEAD_FAR false bits precede native range eligibility. CLEAN HEAD_ANY on/off transitions are 3/2 versus OLD_NEG 1/0; GAP is 5/4 versus 3/2. Gaps suppress true alerts at 20..22 and 30..31; alerting returns at 32 even before the second packet restoration. Both models alert at the first restored samples 23 and 33. The longest positive silence spans three nominal samples (0.3 s by sample-occupancy convention).

Thin pole: six union hits disappear at the two gaps, with restoration at 23 and 33. CLEAN HEAD_ANY has six pre-onset false alerts; HEAD_NEAR has 21 false bits and HEAD_FAR misses all 15 far positives. Thus apparently complete CLEAN head-union coverage hides incorrect near/far assignment. BODY_ANY has one false alert. Missing central packets leave no valid ToF returns during pole gaps; bar and wall retain peripheral returns.

Wall: CLEAN BODY_ANY already misses 17/26 positives and first alerts only at index 21; GAP delays that to 23, a nominal 0.2 s. CLEAN HEAD_FAR misses all 15 far positives. Both models lose two true BODY_ANY samples and three HEAD_ANY samples under gaps. Stable advance warning is not established.

The 40 no-added-target control frames have no OLD_NEG or DIVERSE alerts. Their ToF packets are already entirely invalid, so the gap intervention is NOT_EVALUABLE there. Native evaluator knownness and sensor availability are different quantities: absence of an alert or return is not clearance. Other natural invalid returns and UNKNOWN remain recorded. Intended-target overlap labels are an evaluator subset, not evidence that the model attributed an alert to that target.

## Evidence and disposition

Canonical evidence: `artifacts.local/work/mz77-approach-sequence-20260911/`:

- `primary-definition.json`, both predictor/evaluator freeze receipts and source/input-contract checks;
- `returned-v1/capture-v1/source-admission.json`, native capture and `returned-v1/source-validation-v1/`;
- `sensor-v1/`, `inference-v1/` and `score-v1/` receipts, predictions, evaluator labels, `result.json` and `timeline.png`;
- worker execution and final-release receipts. Returned archive SHA256: `659eb9513774bc05d1a85f489436d9d902fd935a67002837b655ea9e55f1eb18`.

This bounded controlled simulation is complete. It exposes interrupted warnings and weak wall/far coverage; static MZ67 gains did not translate to extra true sequence detections here. No training, threshold rescue, temporal smoothing, real-hardware validation or deployment claim was added. Any later mechanism change requires its own bounded comparison preserving these failures and OLD_NEG; this result does not start one.
