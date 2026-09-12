# MZ89 joint temporal geometry results

Decision: `JOINT_GEOMETRY_FULL_GATE_NOT_MET`.

The joint temporal angular state demonstrates a useful covariance correction,
but its full conservative geometry/spatial-Radar recipe does not replace hard
MZ85 or simple hold. Stop this recipe without fitting margins, matching gates,
Radar compatibility, lifetime or another source on these outcomes.

## Frozen comparison

[Protocol](MZ89_JOINT_GEOMETRY_20260912.md), predictor and eight focused tests were
committed as `1ca53100` before source materialization and scoring. One run used
36 new constructed episodes / 1,080 frames (423 positive, 657 negative).
All six arms received identical serialized observations, excluding evaluator
truth/family/variant. Source/observation/evaluator hashes precede prediction;
prediction hash precedes evaluation. No training, parameter search or outcome retry.

| Arm | TP | FP | FN | F1 | UNKNOWN | Positive UNKNOWN | False segments | Missed events | Within-event fragments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard MZ85 | 382 | 101 | 41 | .8433 | 7 | 2 | 11 | 1 | 24 |
| Hard + one-frame missing-ToF hold | 383 | 101 | 40 | .8445 | 6 | 1 | 11 | 1 | 23 |
| MZ88 full negative control | 338 | 156 | 85 | .7372 | 178 | 85 | 17 | 0 | 27 |
| Independent marginals + spatial Radar | 256 | 23 | 167 | .7293 | 403 | 167 | 5 | 3 | 25 |
| Joint covariance + coarse Radar | 353 | 157 | 70 | .7567 | 118 | 67 | 17 | 0 | 26 |
| Joint covariance + spatial Radar | 302 | 23 | 121 | .8075 | 303 | 118 | 5 | 3 | 25 |

Full candidate versus hard: TP -80, FP -78, FN +80. All FP are in the stressed
boundary/head families and fall101 to23; false segments fall11 to5. However,
nominal variants lose20 baseline TP, total fragments rise24 to25, and two events
detected by hard are entirely missed. Maximum added delay among jointly detected
baseline events is0.2 s; this does not compensate for the two misses. UNKNOWN
remains counted in the full denominator:118 of121 candidate FN are UNKNOWN.
No conditional-known F1 or early false alert is presented as a benefit.

## Attribution

1. **Temporal covariance has a controlled task effect.** Holding causal matching,
   all marginal error loadings, spatial Radar and readout fixed, joint covariance
   changes256/23/167 to302/23/121. All46 added TP are lateral crossing; no TP is
   lost and no false frame changes. Crossing improves25 to71/98, but hard still
   reaches85/98. This supports the cross-frame covariance correction in this
   narrow proxy, not physical occupancy reconstruction or overall promotion.
2. **Spatial Radar removes a large false-confirmation mechanism at a recall cost.**
   Relative to joint+coarse, joint+spatial removes134 FP but loses51 TP:41 inside
   boundary and10 multi-target-gap frames. Its declared seven-degree Radar envelope
   plus twelve-degree corridor admits only the quantized central azimuth bin.
   Seventeen UNCERTAIN frames receive spatial Radar confirmation. This is a strict
   compatibility tradeoff, not evidence that coarse Radar has become precise.
3. **Unresolved losses are broader than crossing.** Relative to hard, the candidate
   loses23 inside-boundary,14 crossing,10 multi-target-gap and33 weak-Radar TP.
   Shared error is no longer repeatedly counted in velocity, but the current
   conservative support and persistent missing-motion uncertainty still abstain
   too often. No evidence here identifies a threshold that would recover these
   without restoring false alerts.
4. **Simple memory remains a credible control.** A single non-reseeding hold adds
   one TP with no new FP and reduces one fragment. It remains the best aggregate
   arm in this run; this small Development gain does not reopen MZ86 or promote a
   runtime policy.

The F1, TP retention, crossing retention, nominal invariance, missed-event/timing,
and fragment gates fail. FP reduction and authority integrity pass. Preserve hard
geometry as the reference; retain the full MZ89 recipe as a negative control and
its isolated covariance delta as explanatory evidence only. No automatic MZ90,
training, dense occupancy, hardware collection or physical collision head.

## Evidence and validation

Canonical durable evidence:
`artifacts.local/work/mz89-joint-geometry-20260912/run-v1/`.
Prediction SHA-256:
`4cb7cdec0a5f6ed6491f6ae15000c2d6e6e1d4c1cf1a00bf44f72af1ce9c61d9`.

Eight focused unit tests pass: covariance cancellation/PSD, persistent missing
increment uncertainty, first-frame bias loading, actual gap duration, mutual
matching/permutation/ambiguity, spatial Radar rejection, missing versus known-empty
ToF/height authority, causal prefixes/episode reset/schema isolation and contiguous
event accounting (some tests cover several invariants). All13 sealed file hashes
were verified; `validation.json` records paired attribution and nominal losses.
The pre-commit knowledge checks also passed. Independent code review preceded
scoring; its first-frame bias-loading correction was made before freezing.
An initial registration fingerprint was repaired after that pre-freeze edit;
no experiment outputs existed at that point.

The CPU scalar predictor took0.1433 s for1,080 frames; complete materialization,
prediction, evaluation and evidence writing took0.3527 s. These are local batch
timings, not device latency. No worker, paid allocation or persistent process was
started. The run directory is retained as decision evidence, owned by MZ89.

Inputs remain exact analytic object-return geometry, synthetic lateral/height
hints and already-stabilized Radar, with constructed IMU disturbances. New-state
arms correct stale-history elapsed time and use mutual matching; covariance and
Radar ablations hold these changes fixed. Legacy controls preserve their original
first-return and timing behavior. The new head-motion family uses explicit
outside-corridor returns and independent central Radar clutter, not MZ88 replay.
Yaw-only covariance with deterministic range gates omits range/pitch/translation
uncertainty. This is source-aware constructed Development, not8x8 ToF validation,
full BODY/HEAD swept geometry, thin-pole recovery, physical alert benefit, novelty,
deployment, user benefit or safety evidence. UNKNOWN is not CLEAR.
