# MZ65 actor reuse: faster capture, appearance preservation not established

The [fixed48-frame A/B/R comparison](MZ65_ACTOR_POOL_20260911.md) completed once.
All captures, native world checks and scoring exited0. The speed conditions pass,
but RGB residual limits fail and the exact packet baseline check is unstable.
Keep the original collector. Retain this recipe as a NEGATIVE_CONTROL for its
claimed appearance-preserving role under the frozen settling policy; the result
does not reject every form or future use of actor pooling.

| Measured time,16 frames | A original | B reuse | R original repeat |
| --- | ---: | ---: | ---: |
| Capture receipt minus map load, seconds | 34.719 | 31.125 | 33.203 |
| Actor preparation sum, seconds | 1.547 | 0.531 | 1.608 |
| Capture command including startup/exit, seconds | 97.005 | 87.032 | 89.715 |

B reduces map-excluded elapsed time by2.078 seconds,6.26%, relative to the faster
baseline R. This exceeds the1.516-second A/R drift, and preparation time is below
both baselines. B creates16 actor slots and reuses96 instead of112 creations in
each baseline. Pre-readiness settling time also changes; the entire elapsed-time
difference cannot be attributed solely to the measured actor-preparation sum.
These are observed engineering timings, not accepted quality-preserving speedups.

Native depth values are exactly identical in every A/B/R comparison. Query bits,
event masks/counts, native validity, world-support, UNKNOWN, full-frame counts,
selected bins and packet validity/counts also match exactly. Every readiness
check is READY, actual render counts match, active/inactive flag checks pass,
and paired target depth is unchanged. No extra renderer frame or model run was
introduced to rescue a comparison.

The exact-array gate fails only on float64 `range_m`, including A versus R.
A saved-array diagnostic finds maximum absolute difference2.220446049250313e-15
metres; all values become identical on float32 conversion. The frozen CUDA
scatter reduction is a plausible numerical explanation, not a rerun-confirmed
cause. The original exact gate remains FAIL and its baseline classification
remains NOT_EVALUABLE_BASELINE_INSTABILITY. This is not a changed native depth
or an estimated physical sensor error.

RGB independently fails the frozen repeated-capture envelope:12/16 full images,
13/16 A-native query regions and13/16 boundary bands exceed at least one of the
five residual limits. Thus the packet numerical explanation does not rescue
appearance preservation. For example, full-image case0 has mean absolute RGB
difference0.278 for B-A versus0.169 for R-A, in8-bit channel units. Complete
per-case mean, quantiles, maxima and changed-value counts are retained, including
cases with better individual metrics. No tolerance was widened after capture.

Root actually viewed all16 A/B/R panels. Expected awning, sign, grille and rod
placement, the two support contexts, and absence of visible inactive objects
passed the gross visual check. This does not establish fine pixel equality and
does not override the numerical failure. All48 captured views and overlays are
bound in `root-visual-review.json` and the returned score receipt.

The result separates an engineering opportunity from its current limitation:
actor preparation can be reduced, while the existing RGB preservation evidence
does not support adopting this variant for the current collector. Changes to
appearance handling or evaluation precision require a separately declared check.
Do not rerun these48 frames, select favorable cases or reinterpret this as a
model, new training source, real hardware or safety result.

[Execution](MZ65_EXECUTION_20260911.md) records the startup-only mechanical repair,
complete return and resource release. Full result:
`artifacts.local/work/mz65-actor-pool-20260911/returned-v1/evidence/score-v1/result.json`,
SHA `465087b526cda02c963ee148b0f4fe5184b120a706cff2e82a633ab55629a860`.
