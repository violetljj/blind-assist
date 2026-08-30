# DTR-R2 current

Updated: 2026-08-31

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
- **CARLA algorithm line:** X24 remains the same-source C2 Development
  reference. X26 and X30 did not meet their frozen gates. X31 remains a
  representation candidate, not a promoted result.
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
