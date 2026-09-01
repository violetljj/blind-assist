# DTR CARLA C43 / X96 source result — 2026-09-01

## Terminal decision

`DTR_CARLA_C43_X96_SOURCE_NOT_EVALUABLE`

C43 completed all four sensor shards, but only `2 / 8` preregistered local
track-then-complete-occlusion contracts passed. The source gate therefore
failed before candidate materialization or X96 scoring. C43 has no arm metric,
dropout-recovery result, or X96-vs-X94 conclusion and must not be reopened for
scoring.

## Frozen source

C43 was a new cohort after C42's zero-frame terminal. It preserved the frozen
X96 algorithm and `2 / 3 / 6` frame intervention matrix while changing to seed
`431096`, new weather assignments, new plan receipts and pixels, and the
previously unused `c17_*` actor trajectories.

- Protocol SHA-256:
  `3804A3F47122136A7AFFD842CD922108AA3BD13744318322E24274613410AED2`
- Source root:
  `E:\linnan\CARLA\experiments\dtr-carla-c43-x96-dropout-survival\evidence\c43-x96-source-20260901-222000`
- Source result SHA-256:
  `8BF25B77F1C3029BE9776DE337B61F06A1BE3A3121FD561E68B417CE476B74D7`

The first depth-server attempt exited with zero depth frames. The protocol's
single zero-frame retry resumed the unchanged run and completed. The joined
source contained `728` frames per sensor, `2,912` raw sensor payloads, four
layouts, eight episodes, and `73` unique actual blueprints. Contact/safe
outcomes, RGB-depth alignment, payload inventories, and truth-blind model-root
checks passed.

## Failure boundary

Only `ep_01` and `ep_07` satisfied the local physical-occlusion contract;
the other six episodes did not provide the required track-then-complete visual
loss. This is a source-reachability failure of the `c17_*` trajectories, not a
measured failure of X96. Moving the frozen intervention, relaxing the source
gate, or scoring the two passing episodes would consume truth post hoc and is
not authorized.

Keep X94 as the cumulative main arm and X73 as the latest complete
source-disjoint authority. C43 is consumed `SOURCE_NOT_EVALUABLE` synthetic
Development evidence, not real-sensor, deployment, reliability, user-benefit,
or safety evidence.
