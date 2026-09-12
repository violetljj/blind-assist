# MZ97: constrained residual rejection and fresh rule replication

EXPLORE, one new UE source and one fixed model configuration. MZ96 full learned
replacement failed event/recall criteria. Hypothesis: conditioning training on A
alerts and constraining rejection on validation may remove false alerts without
that loss. This changes the learned task, not the frozen physical rules.

Reuse MZ96 collector unchanged, fresh seed97013,128 independent scene instances
x40 frames, predetermined80/16/32 train/validation/test scenes. Spec SHA256
c236a8d102f2cd7fd3e3219746cae4cc4ba14538d767c2a436760c6f2e933411.
Same template/sensor-proxy distribution, not unseen-family or physical RF/body
collision validation. No MZ96 samples or labels enter fitting/selection/test.
All previous source/measurement/UNKNOWN caveats remain as in the MZ96 protocol.
This is a fresh synthetic Development replication, not a protected final test.

Freeze MZ94 A, MZ95 B and MZ96 features. Report matched_hold, range-only, A, B,
A+B on all new test frames. Compare A to matched_hold and A+B to A with paired
TP/FP changes, event loss/delay, fragments and episode-bootstrap F1 differences.
Do not choose a new baseline after test. A is the suppressor's fixed reference.

Train on train-split A-positive frames only. Target1 means that A alert is false
under declared current-or-1s route GT. Same34 observable causal features as MZ96,
no source identities, truth geometry, ghost labels, absolute time or future data.
XGBoost300trees depth3 learning_rate.05 min_child_weight5 subsample.8
colsample_bytree.8 lambda5 hist seed97013, no hyperparameter search. Measure one
equivalent CPU and CUDA fit on training candidates via research_backend; select
by runtime only. Retain selected fit. No full decision replacement model is fit.

Placement is explicit: post-process A's final causal alert (including A state).
Output = A AND (predicted false-alert score < rejection threshold). No extra
hysteresis or hold follows; rejected means unsupported/UNKNOWN, not clearance.
This guarantees output is a subset of A and cannot feed back into A's state.
It is a post-alert residual filter, not a raw-candidate or per-object detector.
Fragmentation is assessed on final output because frame rejection can cause holes.

On validation only evaluate thresholds0.00..1.00 in.01 steps plus DISABLED.
Eligible active thresholds must preserve>=98% of A TP, separately>=98% of
future-only A TP, lose no A-detected truth event, add<=.2s first-alert delay,
and not increase false segments or truth-event fragments. Among eligible choose
fewest FP, then most TP, then highest threshold. If none strictly reduces FP,
select DISABLED and exactly reproduce A. Zero denominators are reported and
treated as vacuous; never report a measured recall from zero opportunities.

Seal model, selected threshold/disabled state, rule and residual test predictions,
source/split and local dependency hashes before test scoring. Mechanical label
routing can write separate split payloads but cannot inspect held-out outcomes.
One fresh source, one fit configuration, no test-driven adjustment or rerun.

Test reports all-frame TP/FP/FN/F1, future-only recall and A-TP retention, paired
removed FP/lost TP, events/delay/fragments/false segments, common no-ToF/no-alert
UNKNOWN marker, ghost-present/absent scene slices (not per-return attribution).
Useful residual gain requires active filtering, strictly fewer FP and higher F1,
both98% retention floors, no lost A event, delay<=.2s and no increased segments
or fragments. A/B replication is reported separately even if residual is disabled.
Use1000 fixed seed episode-bootstrap draws for F1 differences; no model selection
from bootstrap. Feature importance is descriptive and optional, not causal proof.

Stop after capture and this comparison. Do not promote default App/core on these
results. A useful filter is only a Development challenger; a no-op or failed gate
is a negative control. Preserve source, packets, labels, model, scores, receipts
and mechanical failures. Release owned UE processes within1800s capture timeout.
All outputs under artifacts.local/work/mz97-residual-suppression-20260912/;
reuse MZ96 artifact-local XGBoost dependency. Complete report, inheritance and
scoped normal commit/push without another experiment.
