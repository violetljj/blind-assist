# MZ98: most A gains do not require strict future-entry admission

Decision: `A_MECHANISM_AUDIT_COMPLETE`. Retain this consumed diagnostic as a
mechanism component, not promotion of an exclusion variant. The claim that A's
future-only recall improvement chiefly demonstrates useful trajectory prediction
is not supported. Current route eligibility, distance and closing-gate interaction
explain much of the effect; strict future entry adds modest TP at substantial FP.

[Frozen audit](MZ98_A_MECHANISM_AUDIT_20260912.md), code
`cf9a244391762c6c819500f0d684e1ca30a509ff`, uses only already-consumed MZ96/MZ97
test panels. No new training, collection, threshold tuning or production change.
Original observations, models, predictions and receipts are untouched. Original
baseline and full A predictions reproduce exactly on both panels.

## Three different mechanisms inside A

- **R, raw current route occupancy:** current observed z>.2m, r<3.6m and
  |bearing|<=12deg. No trajectory fit or closing-speed test is needed.
- **F, fitted-current correction:** R fails, but the eligible fitted position
  already lies in the route wedge at t=0 and meets A's range/depth constraints.
- **P, strict future entry:** R fails and the fitted position starts outside
  the wedge, but its fitted CV trajectory enters during the next second.

Return labels are exclusive; frames can contain multiple classes. Identical
matching and histories feed all replays; excluded admissions remain observations
for tracking. Hysteresis, ToF temporal geometry/selection and outer hold are
unchanged. R-only below means **Radar admission is current-only**, not that the
whole sensor system has no temporal or predictive processing.

## Fixed exclusion results

| Panel / Radar admission | TP | FP | FN | F1 | Future-only TP | False segments | Fragments |
|---|---:|---:|---:|---:|---:|---:|---:|
| MZ96 baseline | 306 | 200 | 103 | .6689 | 41/117 | 48 | 35 |
| MZ96 R only | 337 | 156 | 72 | .7472 | 67/117 | 43 | 39 |
| MZ96 R+F | 338 | 158 | 71 | .7470 | 67/117 | 43 | 39 |
| MZ96 full A=R+F+P | 342 | 176 | 67 | .7379 | 70/117 | 47 | 39 |
| MZ97 baseline | 231 | 167 | 74 | .6572 | 21/56 | 40 | 34 |
| MZ97 R only | 241 | 128 | 64 | .7151 | 35/56 | 30 | 34 |
| MZ97 R+F | 250 | 141 | 55 | .7184 | 35/56 | 32 | 35 |
| MZ97 full A=R+F+P | 254 | 169 | 51 | .6978 | 36/56 | 38 | 33 |

All listed methods miss0 complete truth events on these panels. This does not
establish equal first-alert timing between exclusions and full A or general
event robustness. All-frame final-output denominators are unchanged.

Descriptive sum only,2,560 frames /714 positives:

| Admission | TP | FP | FN | F1 | Future-only TP |
|---|---:|---:|---:|---:|---:|
| baseline | 537 | 367 | 177 | .6638 | 62/173 |
| R only | 578 | 284 | 136 | .7335 | 102/173 |
| R+F | 588 | 299 | 126 | .7345 | 102/173 |
| full A | 596 | 345 | 118 | .7202 | 106/173 |

Adding F to R adds10 TP/15 FP and0 future-only TP. Adding P to R+F adds8 TP/46 FP,
including4 future-only TP. Those P increments are4TP/18FP on MZ96 and4TP/28FP
on MZ97. These are replay dependencies after state processing; current-frame
branch labels alone cannot establish them. Increments depend on this order and
other branches, not an independent additive physical-cause decomposition.

R alone retains102/106 of full A's future-only TP but only578/596 overall TP.
R+F retains588/596 overall TP but still102/106 future-only TP. Neither reaches
the prior98% future-only retention floor relative to A. Higher consumed F1 does
not automatically justify selecting or promoting either variant.

## The GT label explains a large part of the apparent future benefit

Native evaluator bounds split the173 `future_only` positive frames into:

| Evaluator-side category | Positive frames | Baseline TP | R TP | R+F TP | Full A TP |
|---|---:|---:|---:|---:|---:|
| Already inside extended current wedge, outside direct3.18m criterion | 120 | 36 | 75 | 75 | 76 |
| No real current extended-wedge hazard; enters later | 53 | 26 | 27 | 27 | 30 |

GT future includes t=0. Therefore120/173=69.4% of these future-only frames do not
require physical future entry. Of the net44 additional future-only TP from full A
versus baseline,40 occur in this current extended-wedge category and4 in the
strict future-entry category. P adds3 TP on the stricter53-frame slice, not44.
R predictions on strict-future GT can involve the unchanged ToF temporal branch,
state persistence or noisy measured geometry; they do not imply R predicts motion.

Thus the supported story is **task-aligned route admission**, including current
extended occupancy. The stronger claim that predictive crossing is the main
validated primitive needs revision. The GT itself is horizontal point-center
route occupancy, not finite-body collision.

## What the current gate controls reveal

All four controls below use raw |bearing|<=12deg and z>.2m, with identical ToF/state:

| Range / closing requirement | TP | FP | Future-only TP |
|---|---:|---:|---:|
| r<3.18, vr<=-.35 | 516 | 217 | 57 |
| r<3.6, vr<=-.35 | 534 | 250 | 70 |
| r<3.18, no closing gate | 546 | 245 | 75 |
| r<3.6, no closing gate (R) | 578 | 284 | 102 |

At the expanded current range, dropping the closing gate adds44 TP/34 FP and32
future-only TP. With no closing gate, expanding current range adds32 TP/39 FP and27
future-only TP. Effects overlap and interact; do not add these marginal counts.
The old-range/old-closing narrow current control removes150 FP but also21 TP
relative to baseline. That comparison also adds A's z>.2 condition, so the stored
`narrow_bearing_vs_baseline` diagnostic key is not a pure bearing-only ablation.

The earlier range-only control retained the old20deg angle and closing gates.
Its failure cannot establish that distance is irrelevant; the useful current
range change is coupled to route geometry and motion eligibility.

## Final FP paths, with their accompanying TP

Full A descriptive aggregate:

| Selected sensor / evidence stage | TP | FP |
|---|---:|---:|
| Radar, current admitted support | 237 | 191 |
| Radar, hysteresis without current support | 45 | 61 |
| Radar, outer hold | 6 | 21 |
| ToF, direct geometry | 266 | 30 |
| ToF, temporal geometry | 21 | 33 |
| ToF, outer hold | 21 | 9 |

A adds64 TP/101 FP and removes5 TP/123 FP versus baseline. Its added alerts break
down into current Radar52TP/68FP, Radar hysteresis9TP/22FP, and Radar outer hold
3TP/11FP. None of these counts says removing all history is beneficial; each path
also carries correct alerts, and a branch can affect later state on another frame.

Strict-future exclusion removes46 of345 total FP, so P is costly but not the only
remaining problem. Most FP still have current Radar support or other sensor/state
paths. `branch_mix` in frame records describes current/latest supporting returns,
not unique causal objects or the original two-of-three activation seed. `origin_frame`
is this support pointer, not a complete causal history.

Individual raw-return physical identity was not retained by the capture format.
This audit therefore does not invent real-object/ghost/multipath attribution.
Ghost-scene/family counts remain contextual. Doppler measures relative motion and
there is no translational ego estimate, so target versus wearer contributions
cannot be separated. Shared gyro yaw compensation is rotational and also used by
baseline; it is not a new A-specific contribution.

## Delivery and next decision

Three focused tests passed (current extended admission without history, strict
entry classification, instrumentation parity/restoration). Independent review
verified11 output hashes, exact original prediction parity and nested final masks
R subset R+F subset A. CPU scalar diagnostic only; no capture/training process or
paid allocation was started. Original artifact receipts and dependency seals were
verified before replay. Data are consumed; aggregate numbers are descriptive,
not a new confirmatory test or unseen-family result.

Evidence: `artifacts.local/work/mz98-a-mechanism-audit-20260912/run-v1/` contains
per-panel result.json, frame-level predictions/source paths, per-return histories
and branch tags, diagnostic predictions, summary and hashed receipt. Preserve
these task-owned files for reproduction; the original MZ96/MZ97 payloads remain
unchanged. No production/default-App change or new model was made.

The next useful candidate, if authorized, would explicitly test current route
eligibility and fitted-current correction against full A with a fresh evaluation
boundary. Strict future-entry prediction should earn its additional authority
through measured benefit rather than serve as the headline mechanism by default.
This audit does not launch that successor or select a variant from consumed F1.
