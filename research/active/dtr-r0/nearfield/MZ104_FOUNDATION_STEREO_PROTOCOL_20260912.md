# MZ104: frozen learned stereo with separate spatial and alert evaluation

2026-09-12; EXPLORE, consumed Development. New user authorization after MZ103.
MZ103's frozen no-more-final-FN gate remains failed. Its exact native diagnostic
showed that final frame metrics mix depth errors with two-frame confirmation
and exit holding. This experiment explicitly evaluates those stages separately.

Hypothesis: one public pretrained FoundationStereo frontend can remove false
SGBM spatial support without rejecting genuine thin or small surfaces. No
training, tuning, point-span filter, new capture, or lifecycle changes.

Source: all 576 existing MZ101/MZ102 stereo pairs, 640x360, baseline 0.10 m,
horizontal FOV 70 degrees. Known camera/body pose, full64-zone ideal ToF, common
FOV, depth range0.5--4m, original BODY/HEAD boxes and two-on/two-off state remain
unchanged. Preserve UNKNOWN as absent support. Baselines are original ToF and
SGBM+ToF; score learned stereo alone and learned stereo+ToF separately.

Producer receives only RGB paths, IDs, calibration and hashes; no native depth,
ToF, pose, scene actors, labels or original predictions. Depth outputs and their
configuration/input hashes are sealed before evaluation. A first-pair unlabeled
engineering canary may check shape, finite output, CUDA compatibility and cost.
Freeze exact weights/source/license/backend/iterations/preprocessing before the
full replay. Default is official11-33-40 small, original resolution with padding,
20 iterations, documented mixed precision. If the original public distribution
is unavailable, an official NVIDIA adapted small model is eligible as a disclosed
engineering substitution before predictions; it is not the identical checkpoint.
No anonymous mirror or silent model/configuration switch. Save mechanical failure
logs; repairs cannot select on task scores. Positive disparity maps to axial
metric depth using the original f*baseline/disparity, undoing any padding/resize.

Primary component decision, **on each panel**: learned+ToF raw spatial FP is
lower than SGBM+ToF, raw FN does not increase, and at least95% of baseline raw
TP on thin_left/thin_right/small_head/occluded_thin is retained (each present
slice separately). These compare current geometric evidence, not final alerts.
Record exact gained/lost TP and FP, stereo-only counts, BODY/HEAD and appearance.

Separate lifecycle check: retain all baseline-union detected events and add no
entirely false sessions; report first-correct delays, FP segments/duration,
fragments, final TP/FP/FN/F1 and current/held error paths. More than0.25s extra
paired delay or event loss prevents claiming an end-to-end improvement even if
the spatial component improves. Final FN alone does not fail the spatial claim;
all such costs are reported. Measure actual model latency and peak VRAM; no
real-time claim without measured total processing time within the frame budget.

Budget: one checkpoint/configuration after engineering feasibility, one576-frame
replay. No parameter/model sweep, training, event-state candidate or automatic
fresh source. If runtime or official distribution remains unavailable, preserve
the runnable implementation/provenance and report NOT_EVALUABLE rather than a
negative algorithm result. If component gates fail retain the old baseline; if
they pass retain a consumed-Development challenger only, not the default App or
hardware/future-contact evidence. Finish scoped verification/delivery and release
task-owned processes. The unrelated historical ASE registry failure is recorded
if still present; never rewrite old evidence or manually append the ledger.

## Engineering configuration amendment before candidate outcomes

Original11-33-40 config and checkpoint downloads remained quota-blocked. Select
the official NVIDIA TAO adapted small dynamic v2.0 FP32 ONNX graph, SHA256
`a001a7bc0512a0bc3b3218194e924784e58b20656c6f1ea2c151024e555cfd64`.
Refinement iterations are baked into this graph, not the original requested20.
The 384x640 padded first-pair canary initialized CUDA but failed before producing
depth: a Softmax requested2,658,140,160bytes when the6GiB arena had only
1,749,155,072bytes available. Preserve that failure as engineering evidence.

One bounded memory recovery uses the same graph and6GiB cap at **512x288**:
uniform0.8 RGB scaling via OpenCV INTER_AREA before ImageNet normalization,
no padding needed; INTER_LINEAR disparity upsampling back to640x360 followed
by division by0.8 restores original pixel disparity for original-focal metric
depth. Keep original input hashes. This trades spatial resolution for memory;
critical-slice retention gates remain unchanged. A revised unlabeled canary
must establish numerical validity and actual CUDA execution before the full
replay. No subsequent resolution/precision/checkpoint sweep is included.
Record kernel time and complete pair-processing time separately. Right-image
left-boundary visibility is an explicit geometric validity mask, not matching
confidence; within the common FOV and0.5--4m range it cannot remove query support.
