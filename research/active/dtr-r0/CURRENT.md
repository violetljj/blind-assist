# DTR-R2 current

Updated: 2026-09-01

Status: `DTR_R2_DYNAMIC_RETAINED`

## Question

Can BlindAssist emit stable `ONSET / HOLD / ESCALATE / CLEAR` events when
future obstacle occupancy intersects the wearer's route, while preserving
`UNKNOWN` instead of turning missing evidence into safety?

## Current decisions

- **Public/JRDB line:** X21 transports only a component already authorized by
  raw X13 birth and the same live track. Its six-sequence replay reached `5/6`
  CONTACT, 11 false segments, 45.45% Event F1, `3.061 s` median lead, and
  `8/18` dropout recovery. This is a same-source Development pass only.
- **CARLA algorithm line:** X65 pooled `+15 TP / +0 FP / +1.44 pp F1` over X64
  across consumed C26/C27/C28/C32 Development. C33 then terminated as frozen
  source-not-evaluable before any prediction. The sole C34 scored invocation
  completed on genuinely new pixels at `83.01/73.84/78.15` percent
  precision/recall/F1 with all authority invariants zero and acceptable safe
  segments. It improved X54 by `-9 FP / +2.11 pp F1`, but tied X64 exactly.
  Although C34 contained 17 selected contact-loss ambiguity frames and 16
  pre-conflict joint-credential frames, X65 recorded zero ancestry
  synchronization and zero handback frames. C34 is therefore
  mechanism-not-exercised, not incremental X65 confirmation; its 83.01%
  precision also missed the frozen 85% floor. Consumed diagnosis then produced
  X67, which separates existence from route-risk authority only after a dormant
  track was reactivated and lost again beyond the inherited measurement hold
  horizon with receding direction-only motion. X68 then preserves each surface
  footprint but uses a same-direction, lateral-nonexpanding object-local metric
  velocity to remove lattice-quantized near-miss motion. Across
  C26/C27/C28/C32/C34, X68 improved four cohorts and was classification-neutral
  on one: pooled `636 TP / 63 FP / 227 FN`, or `90.99/73.70/81.43%`, for
  `0 TP / -15 FP / +0.77 pp F1` over X67. C34 reached
  `88.19/73.84/80.38%`. X69 then allows a current X25 object-local rigid
  footprint to falsify only mature, measured cross-route surface ambiguity
  after the inherited 1.0 s history window. It improved every cohort, removing
  another 12 false positives with no true-positive loss. Pooled X69 is
  `636 TP / 51 FP / 227 FN` at `92.58/73.70/82.06%`; C34 is
  `90.71/73.84/81.41%`. X70 then gives an X25 rigid identity a collision
  credential only when current X69 surface, X25 rigid-footprint, and X24
  metric-point risk spatially agree. That identity may hand risk back across a
  current surface dropout, while X69 explicit contradiction release retains
  precedence. X70 recovered four true positives with no false-positive cost:
  pooled `640 TP / 51 FP / 223 FN` at `92.62/74.16/82.37%`; C34 is
  `90.78/74.42/81.79%`. This is cross-cohort non-regressing Development, not
  fresh X70 confirmation.
- **CARLA occlusion-source line:** C8 through C11 did not admit an evaluable X31
  source. C11 improved full disappearance coverage to `1/8`, but failed the
  frozen physical-occlusion source gate; no X31 prediction or metric was run.
- **CARLA native-dynamics line:** N3 materialized three towns and all `12/12`
  authored long-tail effects. The sole N4 replay attempt completed Town01, then
  stopped before Town04 pixels because free memory was below the frozen floor;
  Town05 never started. N4 v1 is a consumed incomplete Development attempt.

## Next admissible work

1. Run one genuinely source-disjoint confirmation of unchanged X21 before any
   promotion claim.
2. For X31, admit a new source using raster-observable occlusion authority
   before model inference; do not tune C8-C11 source thresholds or select
   favorable episodes.
3. A new N4 replay requires a new versioned authority. The consumed incomplete
   invocation cannot be resumed or reported as a three-town result.
4. Do not rerun or tune C34 as confirmation. X70 preserves X69's cross-route
   contradiction release, then recovers four surface-dropout contact frames
   through a three-representation object credential with no false-positive
   cost. Continue consumed Development against the remaining pooled
   `223 FN / 51 FP` (C34 `44 FN / 13 FP`). Most opened misses have no surface
   route candidate or jointly agreeing metric candidate, so prioritize
   observation reach or object-local occupancy birth rather than relaxing
   credential birth. Require another visible cross-cohort effect before
   freezing a new confirmation source.

Local uncommitted candidates and outputs are work in progress, not route
authority. This page changes only in the scoped delivery that accepts or closes
their result.

## Stop and claim boundary

- Do not tune the consumed JRDB sequences, CARLA cohorts, route tube,
  lifecycle, association, or source gates against opened outcomes.
- `UNKNOWN` and `NOT_EVALUABLE` are not `CLEAR`, negative evidence, or safety.
- Component identity is diagnostic; wearer-global route conflict owns event
  correctness.
- Public replay and CARLA evidence are Development/mechanism evidence, not
  Android, natural-distribution, user-benefit, deployment, or safety evidence.

## Detail and evidence

- Detailed route ledger and reproduction commands: [README.md](README.md)
- X21 result:
  [X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md](X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md)
- X24 result:
  [DTR_CARLA_X24_PLAN_ADHERENT_DEVELOPMENT_RESULT_2026-08-30.md](carla/DTR_CARLA_X24_PLAN_ADHERENT_DEVELOPMENT_RESULT_2026-08-30.md)
- C11 source terminal:
  [DTR_CARLA_C11_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md](carla/DTR_CARLA_C11_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md)
- N3/N4 result:
  [DTR_CARLA_N3_N4_MULTITOWN_NATIVE_FROZEN_REPLAY_RESULT_2026-08-31.md](carla/DTR_CARLA_N3_N4_MULTITOWN_NATIVE_FROZEN_REPLAY_RESULT_2026-08-31.md)
- X64 consumed transfer Development:
  [DTR_CARLA_X64_CONSUMED_TRANSFER_DEVELOPMENT_20260831.md](carla/DTR_CARLA_X64_CONSUMED_TRANSFER_DEVELOPMENT_20260831.md)
- X64 C29-C32 fresh confirmation:
  [DTR_CARLA_C29_C32_X64_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C29_C32_X64_FRESH_CONFIRMATION_20260901.md)
- X65 consumed cross-cohort Development:
  [DTR_CARLA_X65_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X65_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- C33 terminal source result:
  [DTR_CARLA_C33_SOURCE_NOT_EVALUABLE_20260901.md](carla/DTR_CARLA_C33_SOURCE_NOT_EVALUABLE_20260901.md)
- Frozen C34 X65 protocol:
  [dtr_carla_c34_x65_fresh_source_protocol.json](carla/dtr_carla_c34_x65_fresh_source_protocol.json)
- C34 X65 fresh confirmation:
  [DTR_CARLA_C34_X65_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C34_X65_FRESH_CONFIRMATION_20260901.md)
- X67 consumed cross-cohort Development:
  [DTR_CARLA_X67_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X67_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X68 consumed cross-cohort Development:
  [DTR_CARLA_X68_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X68_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X69 consumed cross-cohort Development:
  [DTR_CARLA_X69_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X69_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X70 consumed cross-cohort Development:
  [DTR_CARLA_X70_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X70_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
