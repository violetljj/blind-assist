# MZ100: causal angle-anchor feasibility fails coverage and reliability

Decision: `CAUSAL_ANGLE_ANCHOR_FEASIBILITY_FAILED`. Keep the frozen configuration
as a NEGATIVE_CONTROL, not an active calibration or fusion component. Baseline R
and the historical core remain unchanged. The first-stage stopping condition was
reached, so no downstream candidate alerts or F1 were generated/scored. This does
not establish that every possible observable angle estimator must fail.

## One bounded hypothesis and execution

[Frozen protocol](MZ100_CAUSAL_ANGLE_ANCHOR_PROTOCOL_20260912.md), code `5981475b8ff52ebb459dcf003cd9eef82a5a93b7`,
used only consumed MZ99 seed99013 observations:128 episodes,5120 frames. This is
posthoc Development feasibility, not fresh validation. No new capture, training,
threshold sweep, fixed-width readout change, ghost classifier or successor.
One CPU run completed in0.691s. There was no failed/retried execution.

The estimator uses all8 horizontal ToF slots in the unchanged15-key raw schema.
Historical narrative references to16 keys were counting errors; the actual input
contract is unchanged. Adjacent range-compatible zones are grouped; only mutually
unique range/angle-compatible Radar/ToF pairs contribute nominal bias intervals.
Three contributing frames over>=0.2s within a1s window and interval width<=10deg
are needed. Contradictions revoke calibration; ambiguity does not update it.
Only a strictly earlier cached estimate may correct current Radar angles during
ToF absence. Cache age is tied to its last contributing anchor, never renewed
by empty frames, and expires after1s. Each episode resets. No scene bias, actor
identity, evaluator bearing or future packet enters estimation/application.

These intervals are nominal tolerances, not confidence intervals or guaranteed
object-center enclosures. ToF group ranges/angles can differ from Radar surface
range/center angle, and several targets can be merged or ambiguously associated.

## Frozen first-stage results

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Coverage of Radar-available / ToF-missing frames | >=10% | 98/2875 =3.41% | FAIL |
| Applied estimates within5deg of scene bias | >=90% | 80/98 =81.63% | FAIL |
| Real-return angle MAE at actual application slots | Decrease | 7.3679 to3.9260deg,114 slots | PASS |

Calibration was applied in19 of128 episodes; median anchor age was0.3s. The
114 real-return slots are paired before/after measurements at the same actual
application locations. No anchor-frame or ToF-present measurements dilute this
error comparison. Applications also affected10 persistent-ghost and12 transient
returns: the predictor has no identity-based exemption.

Frame statuses:4119 no-pair,684 contributing but not validated anchor,
218 ambiguous,2 temporal conflicts,97 validated anchor frames. There were814
mutually unique pair records (807 real,5 persistent ghost,2 transient); the103
pairs in validated frames were all real-return tagged. A real Radar tag alone
does not prove that its ToF partner is the same object. The observed calibration
errors cannot be attributed solely to ghost contamination.

## Which errors improve and which worsen

The following strata use simulator bias only after estimates were sealed. They
are descriptive, not a per-frame truth filter or selectable calibration policy.

| True offset | Applied frames | Active episodes | Offset estimates within5deg | Real slots | Raw angle MAE | Corrected angle MAE |
|---|---:|---:|---:|---:|---:|---:|
| -10deg | 22 | 4 | 18/22 | 26 | 8.7221 | 3.7889 |
| 0deg | 36 | 7 | 22/36 | 43 | 2.9587 | 4.3378 |
| +10deg | 40 | 8 | 40/40 | 45 | 10.7986 | 3.6116 |

The aggregate local error reduction is real, but it hides harm in the zero-bias
subset.14 of18 offset errors exceeding5deg occur there. A narrow temporal
intersection can incorrectly infer a nonzero calibration. In ue99-0082, all12
application frames have6.25deg offset error; ue99-0110 contributes2 application
frames with8.4375deg error. ue99-0112 contributes the other4 failing frames.
These are failure examples, not grounds to blacklist episodes or refit thresholds.

Of2777 eligible fallback frames without application,2707 had never had a validated
anchor earlier in that episode;70 had a prior validated anchor but none usable
at the current frame (expired/revoked). This is a descriptive prior-anchor count,
not a counterfactual test of a longer cache. Merely extending lifetime cannot
address the dominant never-established-anchor group under this frozen estimator.

## Interpretation and inheritance

MZ99 established value from exact evaluator angles; MZ100 does not establish a
usable observable substitute. Unique nominal associations plus temporal interval
intersection provide some useful local offsets, but insufficient availability
and imperfect reliability in the actual Radar fallback windows. Neither adding
history nor seeing a narrower interval guarantees a correct common reference.

Do not report F1 improvement or preservation: the frozen stage1 gate failed and
stage2 was not executed. Keep R and existing outputs unchanged. Do not relax the
coverage/reliability gate, expand tolerances or lengthen persistence based on this
outcome. Revisit only under an explicitly changed information/association premise
and appropriate new evaluation; this result does not rule out all calibration,
all ToF/Radar fusion, or a separately task-matched spatial readout. The independent
fixed-width control and contact-specific readout were not part of this bounded
angle-anchor run and were not launched automatically.

## Validation and durable evidence

Six unit tests PASS: mutual uniqueness/ambiguity, strict prior use and expiration,
episode reset, all-prefix causality, angle-only mutation, same-frame conflict,
invalid packets, adjacent cluster boundaries and temporal-conflict revocation.
Independent pre-outcome review found no mechanical blocker and clarified that
ghosts cannot be excluded by identity. Estimates, corrected inputs and anchor
records were sealed before opening scene bias/provenance. Input/capture receipts,
9 output payload hashes and19 implementation hashes verified; no predictions.npz
exists, consistent with the first-stage stop. No persistent process or paid
resource was created.

Artifacts: `artifacts.local/work/mz100-causal-angle-anchor-20260912/run-v1/`.
`stage1-diagnostic.json` includes all128 episode rows; `anchor-estimates.npz`,
`anchor-records.json` and `corrected-raw.npz` preserve the implemented intervention.
`estimator-seal.json`, `result.json`, source snapshots and `receipt.json` preserve
replay and stopping evidence. All original MZ99 inputs and results are untouched.

Receipt SHA256: `2d64687a7f178fa21576bb0176825a55e1dcbca05cab438c61499b8f1d43ffe2`.
