# Current decision: GRAIL geometry-observable owner orientation

Status: `STOP_G0_POSE_TRANSPORT / DEVELOPMENT_GATE_NOT_MET / NO_FINAL_TEST`

## Authorized question

Does source-native reference-to-query camera yaw provide the missing information
needed to identify the owner-canonical PRESERVE versus FLIP permutation?

This is a synthetic ProcTHOR Development mechanism probe. It does not establish
a deployable cross-session pose source, active multiview behavior, natural-scene
orientation, Android/device capability, navigation, product, or safety behavior.

## Frozen surface

- Protocol and code: `research/active/grail-r1cg/`
- Source: the pinned ProcTHOR-10K train revision used by R1C-L
- Roster: 24 fresh houses excluding all 180 R1C-L train/validation houses
- Baseline: always PRESERVE
- Challenger: PRESERVE iff the cosine of source-native query-minus-reference
  camera yaw is non-negative; otherwise FLIP
- Fixed boundary: 90 degrees, with no Development threshold selection
- Primary evidence: discriminative, FLIP-only, PRESERVE-only, Drawer, Doorway
- Advancement: at least +8pp overall, at least 65% FLIP-only accuracy, and
  positive uplift on both object types

Owner yaw, owner position, object coordinates, depth, model training, threshold
sweeps, R1C-L final data, and protected test data were not used.

## Development result

The complete roster produced 1,157 views and 2,562 ordered pairs with zero
runtime timeout. The discriminative subset contained 2,094 pairs: 514 FLIP-only
and 1,580 PRESERVE-only; 468 additional pairs accepted both modes.

| Arm | Discriminative | FLIP-only | PRESERVE-only |
| --- | ---: | ---: | ---: |
| Always PRESERVE | 75.45% | 0.00% | 100.00% |
| G0 pose transport | 75.26% | 15.95% | 94.56% |

G0 was `-0.19pp` overall versus the prior. Doorway uplift was `0.00pp`; Drawer
uplift was `-0.25pp`. Recovering 82 FLIP-only pairs cost 86 PRESERVE-only pairs.
The route therefore missed all advancement conditions and stops without any
threshold sweep, training, final access, or downstream referent/complete claim.

Exact frozen roster and result hashes are in
`research/active/grail-r1cg/grail_r1c_g0_manifest_v1.json` and
`research/active/grail-r1cg/grail_r1c_g0_development_result_v1.json`.

## Decision

Source-native relative camera yaw alone is not the missing owner-orientation
observable in this bounded setup. Do not rescue G0 by tuning the yaw boundary,
adding a learned pose head, or fusing it into the consumed R1C-L cohort.

This negative does not test active multiview appearance: extra reference frames
could still reveal new asymmetric evidence that a scalar relative yaw cannot
create. Any such G1 must be a new versioned experiment with actual additional
views and a separately declared product-observable pose contract; it is not an
automatic continuation of G0.

## Preserved prior terminals

- GRAIL R1C-L remains
  `STOP_R1C_L_WITHOUT_FINAL_TEST / DEVELOPMENT_GATE_NOT_MET / FINAL_UNOPENED`.
- The unseen-location Router remains
  `MSLS_SOURCE_ADMITTED / ROUTER_DEVELOPMENT_GATE_NOT_MET / TEST_UNOPENED`.

Neither prior cohort was reopened, tuned, rerun, or fused into R1C-G0.
