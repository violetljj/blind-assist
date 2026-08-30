# DTR CARLA C8 X31 source result — 2026-08-30

## Decision

`DTR_CARLA_C8_X31_TRANSPORT_CONE_SOURCE_NOT_EVALUABLE_SCRIPTED_POSE_DRIFT`

C8 did not reach source admission and carries no X24/X31 metric result.  The
single frozen capture attempt stopped in the first `instance` shard, episode
`ep_01`, after payload indices `000000..000042`.  No model replay or scorer was
run, and no source `result.json`, model package, or evaluator package exists.

## Frozen identity

- Git commit: `9479e89173769fe326c4f2afc894af9b2d7feaeb`
- Run ID: `c8-x31-transport-20260830-193417`
- Cohort: `DTR_CARLA_C8_X31_TRANSPORT_CONE_SOURCE_DISJOINT_V1`
- Protocol SHA-256:
  `180A9FF243FB6D07999AD936481F77429BE2759B0A9038D13E6B9821BC35ACF0`
- X31 predictor SHA-256:
  `28E777021FC6AA39129480B069B544B562F559FD447CCE164706B6DCAC3A9F18`
- Source root:
  `E:\linnan\CARLA\experiments\dtr-carla-c8-x31-transport-cone\evidence\c8-x31-transport-20260830-193417`

## Failure evidence

The authoritative scripted-pose gate rejected `c8_l01_alias` at captured
sample index `42`.  The expected transform had roll `0.0` degrees and surface
height `0.8000000119 m`; the CARLA snapshot reported roll
`-2.0482177734` degrees and height `0.9297086596 m`.  Planar position residual
was `0.0321352 m` and full 3D residual was `0.1336301 m`.  This is a source
materialization failure, not an obstacle-risk false negative or false positive.

- Captured payload frames: `43`
- Partial source bytes: `3,346,596`
- Client stdout SHA-256:
  `8784154039DC1C04BFDCB50E2961FD0F35D08C78DC88A7B35ED863D05D0FB785`
- Client stderr SHA-256:
  `11D5274C9030DA68AED09B3A277815F2729864051C7EF1FD713EAA7160E01D35`
- Source `result.json`: absent
- Model package: absent
- Evaluator package: absent

The partial source is retained as failure evidence.  It must not be deleted,
resumed, reseeded in place, or scored.  The runner cleaned its owned CARLA
processes and ports `2000..2002`; both process and listener counts were zero
after failure.

## Next legal route

Use a new C9 source identity and seed.  Keep X31, the detector, route, and all
metric gates frozen.  Change only the source representation: disable CARLA
physics collisions for every actor whose transform is authoritatively scripted,
while retaining its rendered geometry, bounding box, physical-occlusion role,
and evaluator `collision_relevant` contract.  This separates scripted pose
authority from engine collision response instead of loosening the pose gate or
replaying C8.

## Claim boundary

C8 is `NOT_EVALUABLE`.  It supplies no Development, confirmation,
generalization, product, deployment, or safety evidence for X31.
