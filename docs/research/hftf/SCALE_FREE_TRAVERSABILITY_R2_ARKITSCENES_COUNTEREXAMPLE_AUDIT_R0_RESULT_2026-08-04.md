# ARKitScenes visit 484248 scale-free counterexample audit

Date: 2026-08-04

Decision: `DISAGREEMENT_DETECTOR_ONLY_WITH_ORIENTATION_RECEIPT`; unrectified
input is `DISABLE`.

The failure is not a threshold problem and is not explained by one global DA
scale error. It is a stored-orientation, close-planar-scene, and local spatial
ordering failure. The immutable parent result remains
`SCALE_FREE_TRAVERSABILITY_R2_NOT_EVALUABLE_SOURCE_SUPPORT`.

## Mechanism answer

All 150 frames have the official ARKitScenes `left` orientation and require the
repository's clockwise 90-degree rectification before stored columns can mean
physical left/center/right. The original R2 evaluator did not apply that
rectification. The median optical-axis tilt was also 56.36 degrees, so this is a
handheld scan rather than a fixed forward navigation camera.

The bathroom is dominated by close planes: 132/148 auditable frames contain a
large plane, 119 contain a near plane, 123 have a vertical dominant plane, and
72 have one plane supported in all three stored bands. The representative rows
show doors, walls, counters, floor, and fixtures crossing bands that do not
correspond to physical left/right while the image is sideways.

Source depth is not the main failure. Confidence-2 source coverage has median
95.49%; nearest reconstruction borrows across a band boundary for only 0.57%
of filled pixels at the median; and global-nearest versus band-local-nearest
score ordering agrees on every auditable frame. The two zero-source startup
frames remain explicit source failures.

DA and sensor depth have a median fitted global scale of 0.806 and median
post-scale AbsRel of 8.00%, but only 47.97% of frames have the same full
left/center/right band ordering. Since the frozen operator is exactly invariant
to multiplying all DA depths by one scalar, the directional failure is local
geometry/spatial rank, not global scale.

`AMBIGUOUS` is mixed rather than uniformly correct: 39 decisions are reasonable
refusals against the reconstructed reference, 45 are wrong-refusal proxies, and
43 directional candidate outputs are over-answer proxies. This consumed
reference is not an actionability or safety label.

For mechanism attribution only, applying the official 90-degree orientation to
both candidate and reference raises recommendation coverage from 22.41% to
82.61% and directional agreement from 38.46% to 89.47%. This post-result
counterfactual never replaces the R2 ledger, terminal, cohort, or gates.

## Route decision

Scale-free is not admitted as an auxiliary user output or fallback. It may be
retained only as a Development disagreement detector/diagnostic after an
explicit orientation and coordinate receipt. Any unrectified or identity-unknown
input must disable it. A new upright, fixed-camera evaluation would be required
before considering a broader role.

Full ignored evidence is under
`artifacts.local/evidence/hftf/scale-free-traversability-r2-484248-counterexample-audit-r0-20260804/`.
The full result SHA-256 is
`C59A601AB3F347CA66DB609599312FF14DC36FB748FCE9591D65598C99263D9A`.
The evidence remains consumed Development mechanism analysis only; it does not
authorize clearance, alerts, safety, App integration, or production behavior.
