# Clearance-Student Mobile S1.1 mechanism-correction protocol

Status: `ONE_MECHANISM_CORRECTION_EXPERIMENT_AUTHORIZED / E0_ONLY_UNTIL_PREFLIGHT_PASS /
S1_B_NOT_AUTHORIZED / CONSUMED_120_FRAME_COHORT_FORBIDDEN_FOR_TUNING /
NO_QNN_QAT_ANDROID_OR_SAFETY_AUTHORITY`.

S1.1 is the only authorized correction of S1.  It tests whether S1 failed from
mobile capacity or because pretrained representation, valid supervision, and
feature alignment were not actually implemented.  It is not a third backbone
arm and does not reopen the failed S1 checkpoint.

## Frozen mechanism corrections

1. Load torchvision MobileNetV3-Large ImageNet weights and record their SHA256.
   Encoder initialization must remain byte-identical after decoder/head
   initialization; `_initialize()` may touch only newly owned modules.
2. Metric truth is valid only where `confidence == 2` and depth is in
   `[0.25m, 6m]`.  Horizontal/vertical gradient terms require both adjacent
   pixels to be valid; scale uses valid pixels only.
3. Before model construction, materialize ground plane/normal, camera height,
   left/center/right clearance and occupancy targets from the training-only
   metric stream.  Report frame and per-target coverage.  A head with
   insufficient frozen coverage must be removed from the protocol rather than
   left present but untrained.
4. Remove the 960-to-2048 context expansion.  Context is capped at 512 channels;
   capacity is assigned preferentially to 1/4 and 1/8 decoder refinement.
5. Feature targets must be normalized.  Prefer teacher DPT/refinement features;
   otherwise use per-channel normalized encoder features and cosine or
   normalized-L1 loss.  Uniformly enlarging one 8x8 target to every student
   scale is forbidden.

## E0 training-side sanity screen

E0 uses only a newly versioned training/validation split derived without the
consumed 120-frame cohort.  Before training it must bind encoder-weight SHA,
encoder pre/post initialization equality, target coverage, cohort identities,
seed, epochs, checkpoint rule, and all loss weights.

Required outputs are scale-aligned AbsRel, ground-recovery coverage, depth
distribution quantiles/saturation, finite geometry-output coverage, and the
encoder binding receipt.  Any undefined metric or zero ground recovery closes
S1.1 without E1.

## E1 new development gate

E1 requires a new versioned development cohort.  It compares Canonical, frozen
S0, and the single frozen S1.1 checkpoint.  Every geometry metric must be
defined, known collision decisions must be positive, scale-aligned AbsRel must
be strictly below S0's `0.1190`, and low false-clear cannot be obtained through
all-OCCUPIED or all-UNKNOWN collapse.

If E1 fails, close the MobileNetV3 self-developed student branch.  Passing E1
still does not authorize QNN, QAT, Android, production replacement, or safety
claims; those require a separate decision.
