# MZ56 execution, evidence and remaining bottleneck

[The measured-context comparison](MZ56_GLOBAL_ANCHOR_RESULTS_20260911.md)
passes its fixed primary check. GLOBAL recovers 26 new nonfit BODY_NEAR events
with native winners outside the sensor field, versus 11 for LOCAL. Its 410 total
nonfit TP still trail the retained OLD_NEG union's 463; keep GLOBAL as a
component/challenger. All source and evaluation remain consumed Development.

The successful CUDA run took 150.116 seconds: LOCAL fitting 18.995, GLOBAL
fitting 16.203, streamed evaluation 96.627, and the remainder for binding,
initialization and output work. Each arm used 600 steps and 11,020 trainable
parameters. CPU saved-output scoring took 2.700 seconds. Eight focused synthetic
model checks had passed before execution. The two arms have exactly identical
initial parameters; actual GPU logits match the original stated numerical
tolerance, not bit for bit.

The first attempt stopped during initialization before any fit or feature
extraction. An added bitwise GPU check rejected a 0.000005722 difference from
the inherited reduction arithmetic. Its directory, original runner and failure
receipts remain intact. The successful run-v2 uses the already declared
absolute 0.00002 / relative 0.000001 tolerance for numerical forward equivalence;
weights, schedules, masks, labels, cutoffs and saved decisions retain exact
checks. This is a mechanical correction, not an additional training budget.
The 150.116-second figure covers run-v2 only, not all preparation and failed work.

Training used the existing read-only 4,408,012,928-byte feature file with its
original frame index, hash and sole NTFS link. No training feature was recomputed
and no duplicate physical dense allocation was made. Only 5,734 uncached full
RGB images were encoded for evaluation, reading 2,315,820,182 RGB bytes. The
retained 790 prior arrays needed no new baseline model inference.

After model exit and scoring PASS, root verified the unchanged cache hash,
sole link, closed handles and no owning process, then removed the exact MZ56
scratch tree. Checkpoints, predictions and original sources remain. The primary
GPU was handed to the separately allocated MZ55 capture; this is a resource
handoff, not concurrent rendering with model fitting.

A separate saved-output MZ54 diagnostic explains why wider RGB alone was
insufficient. Among 185 baseline-missed nonfit BODY_NEAR positives, FULL has
110 native winners versus CROP's 10, but 108 of those FULL native winners remain
below its unchanged cutoff. The 16 lost HEAD_NEAR cases have correct native
winners, while broader HEAD localization also deteriorates. Both score-tail
separation and localization matter. Existing negative replay already penalizes
the spatial maximum; any future ranking change must add a distinct mechanism
and must not move the responsible calibration negatives into training.

Evidence lives under `artifacts.local/work/mz56-global-anchor-20260911/` in
`run-v2`, `score-v1`, `preflight-failure-v1` and `resource-release.json`;
the independent diagnostic is under the MZ54 task's `diagnostic-ranking-v1`.
No result here supplies real optical calibration, clearance or default promotion.
