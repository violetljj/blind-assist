# Quality-gated clearance fusion R0 development protocol

This is a separate deterministic post-processing route after the P3 temporal
lightweight-head route was closed as mixed and not promotable. It does not
reopen P3, change its loss, or authorize A5S, QNN/HTP, Android, cadence search,
or any new model training.

The replay compares the existing raw per-frame geometry stream with the frozen
quality-gated filter. The filter consumes precomputed clearance, geometry
validity, ToF validity, teacher age and frozen A2 disagreement. It does not
re-infer RGB, load a checkpoint, construct an optimizer or read holdout
outcomes.

The three newly admitted parents are fixed by the source-admission amendment.
Each must provide at least eight non-overlapping four-frame clips with source
timestamps and adjacent gaps no larger than 500 ms. Parent is the statistical
unit; frames are not independent samples.

The claim ceiling is `DEVELOPMENT_SIGNAL_ONLY`. A positive result can justify a
separately frozen confirmatory evaluation; it cannot be called a P1 pass,
generalization evidence, safety evidence or deployment authorization.
