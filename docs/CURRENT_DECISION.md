# Current research decisions

Updated: 2026-08-31

Status: `L10_R0_ACTIVE / DTR_R2_DYNAMIC_RETAINED`

This file owns only decisions that affect what may run or be claimed now. Full
result chains remain in the two route ledgers and result files; Git preserves
the superseded long-form decision history.

## L10-R0

### Controller and active observation

- Keep seek, guide, reacquire, deficit-specific observation actions, and the
  causal handoff guard.
- PanoLab's `4/4` result establishes entrance-ray recovery mechanics only. It
  does not establish pixel-portal identity, arrival, or handoff readiness.
- Actions remain tied to the information gap: `APPROACH`, `SIDESTEP/PAN`,
  `SWEEP`, or `HOLD`.

### SEVN address-door backend

Decision:
`L10_SEVN_V2_ADDRESS_PANORAMA_DISJOINT_SAME_SOURCE_CONFIRMATION_GATE_NOT_MET`.

The frozen V2 stack improved the consumed 24-address panel to `14/2/8`
correct/wrong/`UNKNOWN` binding and `19/22` visible-number OCR. On 21 new
addresses and 28 new frames with zero reference overlap, OCR was `12/20` and
binding was `9/0/12`. Precision among emitted bindings stayed high, but recall
was insufficient.

Next admissible action: change the OCR observation representation and use fresh
addresses/frames. Do not tune tiling, thresholds, ranking, or abstention on the
opened confirmation panel.

### Metric portal and endpoint

3RScan registered extent established a Development ceiling. The latest
source-distinct spatial mask reached `0.5403` complete IoU and `0.422 m`
centroid error, below the `60%` ceiling-retention gate, and confused an
overlapping doorframe.

Next admissible action: add exact-instance or portal-set authority before
another endpoint-mask successor. A geometrically plausible nearby frame cannot
be counted as the target entrance.

Generic Panoramax pixel-portal mining and the consumed SceneFun3D ordinal source
remain closed.

## DTR-R2

### Public/JRDB line

Decision: `DTR_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_GATE_MET` for Development
only.

X21 reached `5/6` CONTACT, 11 false segments, 45.45% Event F1, `3.061 s`
median lead, and `8/18` dropout recovery on six consumed sequences. It may
transport only an already authorized component row while its anchor remains in
the same live track; it cannot absorb a new current cell.

Next admissible action: one frozen, genuinely source-disjoint confirmation of
unchanged X21. Do not tune or resample the six opened sequences.

### CARLA algorithm and occlusion-source line

- X24 remains the same-source C2 Development reference.
- X26 and X30 missed their frozen gates; their consumed cohorts are closed.
- C8-C11 did not admit an evaluable X31 occlusion source. C11 reached only
  `1/8` valid complete-occlusion episodes and ran no X31 prediction or metric.
- X31 remains a candidate representation, not a result.

Next admissible action: admit a new raster-observable occlusion source before
inference. Do not tune C8-C11 source thresholds, select favorable episodes, or
convert `NOT_EVALUABLE` into an algorithm failure/success.

### CARLA native-dynamics line

N3 materialized Town01, Town04, and Town05 with `12/12` authored long-tail
effects. The sole frozen N4 invocation completed Town01, then stopped before
Town04 pixels when free memory was below the frozen floor; Town05 never ran.

Decision: `DTR_CARLA_N4_REPLAY_ATTEMPT_CONSUMED_INCOMPLETE`. A complete replay
requires a new versioned authority. N4 v1 cannot be resumed, retried, or
reported as a three-town result.

## Cross-route boundaries

- L10 and DTR do not wait for, modify, or validate one another.
- Proposal, selection, referent, affordance, waypoint, arrival, and handoff are
  distinct authorities.
- `UNKNOWN` is not `CLEAR`; `NOT_EVALUABLE` is not a negative result.
- Curated, synthetic, replay, registered-source, and device evidence keep their
  actual claim ceilings.
- Local uncommitted candidates are not current authority until a scoped result
  delivery updates the owning route current.

## Stop here

- Do not rescue consumed evidence with threshold, seed, cohort, backbone,
  aggregation, or narrative changes.
- Do not add governance or tests unless they protect interpretation, prevent a
  material irreversible failure, or change the next decision.
- Do not infer Android readiness, natural-distribution performance, user
  benefit, navigation reliability, or safety from the current research results.

Route authority:
[L10-R0 current](../research/active/l10-r0/CURRENT.md) and
[DTR-R2 current](../research/active/dtr-r0/CURRENT.md).
