# Metric clearance field A0 UniDepth source comparison

Date: 2026-08-03

Status: `FROZEN_BEFORE_UNIDEPTH_DENSE_FIELD_OUTCOME`

## Question

Is the corrected A0 failure specific to Metric3D dense geometry, or does the
same deterministic clearance construction also fail with the faster
UniDepthV2-S metric-depth source?

The fixed-world-floor Development and consumed `walking_rpy` reference reports,
RGB frames, bands, ground RANSAC, obstacle heights, 2nd percentile, horizons,
unknown handling, and seven A0 gates are unchanged. The only changed variable
is the depth source:

- `lpiccinelli/unidepth-v2-vits14`;
- official local implementation and cached weights;
- published intrinsics supplied;
- resolution level 0;
- CUDA inference.

UniDepth confidence is not used by this deterministic source comparison. A
pass on all seven A0 gates may authorize a fresh `walking_halfsphere` source
confirmation. A failure closes this exact deterministic UniDepth field, not
future use of its uncertainty output.
