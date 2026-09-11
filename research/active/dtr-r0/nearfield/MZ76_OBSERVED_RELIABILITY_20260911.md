# MZ76: matched continuation with observed neighbor geometry

Consumed Development experiment, defined before new model training. The question
is whether local consistency between reported ToF zones helps an existing RGB+
ToF head under ordinary and partially available returns. Existing MZ61/MZ67 model
packets contain only range/valid and frame identity: no measured signal, sigma,
ambient or status quality. This experiment does not synthesize those fields or
claim calibrated sensor confidence.

Start both arms from frozen MZ70 DIVERSE. Expand its first head layer from49 to52
inputs, preserve the original49 columns and initialize the new3 columns to zero.
BASE receives three zero channels. RELIABILITY receives: centre nearest distance
minus the four-connected valid-neighbor lower median, neighbor lower-median
absolute deviation, and valid-neighbor fraction of in-bounds neighbors. Distance
features divide by4. Missing centres or no valid neighbors produce zero channels;
outside the original ToF field the new channels are zero. No return is filtered.
Both arms have11,116 parameters; BASE's zero-input columns have no effective
capacity, so equal parameter count does not isolate information from effective
capacity. Original frozen DIVERSE is retained as a third comparison.

Use sorted2,048 original TRAIN rows from each MZ61/MZ67 source. Seed176 chooses
one independent permutation per source, partitioned into256 batches of8. Each
step uses8 frames from each source, shared by both arms. Every TRAIN frame appears
once per arm. Profiles cycle IDEAL, CLOSEST_REPORTED_PROXY, FARTHEST_REPORTED_PROXY
(86/85/85 steps), then pass through the explicit canonical model adapter. No
ALL_INVALID training or profile is included. Both arms use256 Adam steps at
learning rate0.0001, fresh optimizer state, original native loss_for, and all
head parameters trainable. There is no old-negative replay in this bounded
two-source test: it cannot establish broad legacy retention or isolate forgetting.

Keep original visual encoders, normalization, query geometry, MZ37/OLD_NEG and
all cutoffs frozen. Both new heads use the original DIVERSE cutoff vector. Before
fitting,32 original TRAIN frames check original saved IDEAL outputs and expanded
heads against the frozen head, including declared float tolerance2e-5/1e-6 and
exact decision signs. Fresh comparisons also cover all three canonical profiles.
Encode one full RGB view per training batch and share it between arms; no full
dataset feature cache. Evaluation shares the existing three visual views across
profiles and heads. The original sources and previous runs remain unchanged.

After256 steps per arm, evaluate exactly the existing1,024 HELDOUT_GEOMETRY rows
per source under all three profiles. No outcome-based selection, extra fitting,
new cutoff or automatic successor experiment. Independent CPU scoring after
sealed inference verifies original identities, packet transformations, fixed
candidate/OR arithmetic, unchanged frozen-head comparisons, and scalar metrics.
Report each query's TP/FP/FN/TN and UNKNOWN for BASE, RELIABILITY and frozen
DIVERSE; include exact gained/lost positive and added/removed false event IDs.
Primary comparison is RELIABILITY versus matched BASE, with both compared to
the frozen original. All source/profile tradeoffs remain visible; no single
aggregate chooses a promoted model. No improvement, or improvement purchased
with worse false alarms/old-source errors, limits this mechanism's value.

This is a small controlled continuation, not fresh confirmation, real-device
validation, quality-aware hardware learning or an application model replacement.
Native TRAIN labels are loss-only; HELD labels are opened only by the scorer.
New source-contact diagnostic frames do not enter this experiment.

Artifact root: artifacts.local/work/mz76-observed-reliability-20260911.
