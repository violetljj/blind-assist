# Quality-gated clearance fusion R0 source admission

Status:

```text
FROZEN_BEFORE_SOURCE_ACQUISITION_OR_LABEL_READ
```

The local ancestry audit found zero immediately reusable, fully unconsumed
ARKit/TUM/Bonn parents. This protocol therefore freezes a finite public-source
admission universe before any new acquisition or label access.

Only identity, file integrity, timestamp continuity, intrinsics, trajectory,
metric-depth capability, license and ancestry may be inspected. Clearance,
geometry state, transitions, labels, teacher outputs, student outputs and
candidate metrics are forbidden until identity lock is complete.

The candidate universe is limited to three unconsumed TUM Freiburg-3 sitting
sequences and previously unconsumed official ARKitScenes visits/scenes. Bonn,
the consumed P3 parents, the eight locked R0.2.1 validation parents, the R0.1
attempted holdout visits and legacy P1 ancestry are permanently excluded.

This is an admission protocol, not permission to train or evaluate. If fewer
than three new parents with at least eight four-frame clips each survive the
identity and integrity audit, the route terminates as
`QUALITY_GATED_CLEARANCE_FUSION_R0_DATA_NOT_READY`.
