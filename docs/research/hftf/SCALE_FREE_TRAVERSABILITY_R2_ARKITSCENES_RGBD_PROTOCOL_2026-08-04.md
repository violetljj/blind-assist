# Scale-Free Traversability R2 ARKitScenes RGB-D Protocol

Date: 2026-08-04

Status: `FROZEN_BEFORE_ARKITSCENES_CANDIDATE_OUTPUT_EXECUTION`

Bonn R1 failed closed because one sequence supplied only 48.45% valid truth
scores against its pre-frozen 50% gate. R2 does not lower or reinterpret that
gate. It adds a separately frozen public RGB-D replication on the 20 consumed,
parent-disjoint ARKitScenes development visits already present locally.

The R0 candidate remains byte- and parameter-identical. The data role is
`PROJECT_CONSUMED_DEVELOPMENT`; because the scale-free operator was frozen from
phone RGB without ARKitScenes output selection, the narrower description
`OPERATOR_UNSEEN_EXTERNAL_REPLICATION` is also permitted. Neither description
means globally fresh, sealed, Confirmation, safety, or production evidence.

## Frozen cohort and truth

The cohort is bound to roster SHA-256
`7CE2D9931723EF7517531F7389FF1DFA0E4BF9BD4C8291A9E72A5BBFF7102EEC`:
16 train-role and four validation-role visits, one video per unique visit, and
all 150 already locked matched RGB/depth/confidence frames per video in timestamp
order. The visit is the independent unit (`n=20`).

RGB alone enters Depth Anything V2. After candidate inference, source depth is
decoded in millimetres. Only finite 0.25–6.0 m returns with source confidence 2
are valid. A frame requires at least 50% such returns. To make the sparse sensor
map usable by the already frozen dense spatial operator, invalid pixels are
filled from the nearest confidence-2 return in image coordinates; this derived
dense truth is then scored by the exact R0 scale-free and causal mechanics.
Frames below 50% source-valid support remain unknown and are never imputed.

## Frozen gates

Each visit requires truth score coverage at least 0.80, candidate execution
coverage at least 0.95, and at least 20 non-ambiguous truth directions. On truth
directional frames, candidate recommendation coverage must be at least 0.50 per
visit. Exact directional accuracy must be at least 0.60 in every visit and 0.75
as a visit-macro mean. Visit-macro left-versus-right opposite error must not
exceed 0.05. Exact decision agreement including `AMBIGUOUS` is diagnostic only.

The independent validator must recompute summaries and the terminal directly
from the immutable frame ledger without importing the evaluator. No source,
frame, truth reconstruction, operator, threshold, or gate may change after
ARKitScenes candidate outputs are read.

The only terminals are:

- `SCALE_FREE_TRAVERSABILITY_R2_NOT_EVALUABLE_SOURCE_SUPPORT`;
- `SCALE_FREE_TRAVERSABILITY_R2_EXTERNAL_RGBD_REPLICATION_NOT_SUPPORTED`;
- `SCALE_FREE_TRAVERSABILITY_R2_EXTERNAL_RGBD_REPLICATION_SUPPORTED_DEVELOPMENT_ONLY`.

None authorizes clearance, metric distance, alerts, safety, or production use.
