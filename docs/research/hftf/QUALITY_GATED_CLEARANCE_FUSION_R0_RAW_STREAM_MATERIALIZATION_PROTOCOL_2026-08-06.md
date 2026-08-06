# Quality-gated clearance fusion R0 raw-stream materialization

This producer creates the missing, parent-disjoint replay stream for the
quality-gated clearance filter. A2-392 is the only learned model allowed to be
loaded. It is used only for per-frame raw depth geometry and frozen disagreement
against the independently registered RGB-D metric depth of the same source
frame. No P3 student, optimizer or training process is constructed.

The `tof_valid` field is retained for compatibility with the frozen filter but
means `independent_metric_sensor_valid` in this route. `teacher_age_s=0.0` is
used only for source-associated RGB/metric-depth pairs; no teacher cadence is
inferred or searched. Invalid metric depth or invalid A2 geometry is fail-closed
and remains visible in the output validity fields.

The output is development-only evidence. It cannot open a P1, prove
generalization, or authorize deployment.
