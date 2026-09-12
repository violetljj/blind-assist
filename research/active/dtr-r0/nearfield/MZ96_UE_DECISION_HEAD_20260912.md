# MZ96 fresh UE geometry decision-head comparison

EXPLORE, one fresh source and one fixed small XGBoost configuration. Prior MZ90
through MZ95 observations/results are consumed Development and never training or
test samples here. A and B are separately evaluated before this comparison.

## Data and source boundary

Fresh source-v3 seed96013,128 independent scene episodes x40 frames at0.1s,
predetermined80train/16validation/32test scenes (3200/640/1280 frames). Split is
by full scene, not shuffled frames or repeated variants. Templates are shared
across splits, so this is scene-instance independence, not unseen-family transfer.
Spec SHA256 a87173e43287eb6f5cb18cf3adf101337d6b7a52a7817a198734faab3266fb06.
Earlier2episode engineering canary uses seed96012 and is excluded.

UE5.8 native collision rays and engine transforms supply geometry; ToF uses8x8
rays aggregated into8 horizontal bins. Radar uses actor-directed native occlusion
checks followed by a hypothetical noisy/quantized return model.32 scenes also
contain a synthetic persistent ghost, independent of family, with the same0.8
detection/noise law as real Radar proxies; ghosts are not native hits or GT.
All are explicitly UE_NATIVE_COLLISION_GEOMETRY_PLUS_HYPOTHETICAL_RADAR_NOT_RF.
No RGB detector, RF physics, realistic material optics or device trial is implied.
GT is the declared horizontal point-center route wedge, current-or-1s with3.6m
radial cap, not finite human-body collision. Engine IDs, positions/velocities,
family, ghost flags, split and absolute frame time are not model features.

Native raw/evaluator/geometry streams remain separate. Materialization routes
labels into separate split files without inspecting test outcome statistics.
Model training reads train labels; validation selects only threshold and reference
rule. Test labels are opened only after model, threshold and all test predictions
are saved and hashed. No iterative regeneration or test-driven model revision.

## Same-data baselines and model

All methods receive identical causal raw packets/history: original matched_hold,
range-only3.6 diagnostic (strong simple challenger), A horizon admission,
B pointwise coverage authority, and A+B composition. Keep the original A/B
parameters frozen. B's pointwise compatibility is not full uncertainty coverage.
Do not assume A+B is best; choose the reference rule by validation F1 (tie: fewer
FP, then listed order), and report every rule on test.

Frame-level structured features only: nearest Radar ranges/bearings/radial speeds,
valid/qualified/horizon counts and track age; ToF nearest ranges/bearings, valid
count, direct/temporal geometry; range/bearing disagreement, pointwise coverage,
sensor conflict; previous1–3 rule-support/availability values and causal age since
support. No truth attributes, source family, actor IDs, calibration oracle, future
packets, test statistics or learned detector. This is a frame hazard head, not a
per-object localization model.

XGBoost binary logistic,300 trees, depth3, learning_rate0.05, min_child_weight5,
subsample0.8, colsample_bytree0.8, reg_lambda5, seed96013, hist. No architecture or
hyperparameter search. Benchmark one equivalent CPU and CUDA training fit on
train only via research_backend, zero warmups/one repeat; select by measured time,
not predictive performance, and retain its fit. Record actual backend config.
Validation chooses threshold among0.10..0.90 in0.05 steps by final F1 after the
same2-of-3/2-empty hysteresis (ties: fewer FP then higher threshold). No outer
sensor-priority hold on learned head; report this fixed readout difference.

Test outcome: frame TP/FP/FN/F1, UNKNOWN using the common no-ToF/no-alert marker,
event misses/first alert delay/fragments/false segments, future-only recall,
ghost-present versus absent scene slices, all-rule comparison. Threshold selection
and model remain sealed before test labels. Report gain importance and mean
absolute TreeSHAP as associations, not causal explanations. Optional paired
episode bootstrap uses1000 fixed seed draws versus validation-selected rule.

Useful result gate: test F1 at least reference+0.02, TP no lower, FP no higher,
no more missed reference events, added shared-event delay<=0.2s and no increased
false segments/fragments. Report partial gains honestly if any gate fails.
This only supports a fresh synthetic Development challenger, never automatic
core/default-App promotion, real-world accuracy, edge latency or safety claims.

Stop after one full capture and this fixed comparison. Mechanical recovery may
repair execution with logs/input identity preserved; no retries for outcomes.
Native capture timeout1800s, release its owned editor/Zen processes. Dependency
and evidence roots stay under artifacts.local/work/mz96-ue-decision-20260912/.
Persist source, split manifests, observations, model, scores, receipts and failure
diagnostics; no paid worker allocations. User-authorized A/B/learning delivery
continues to documentation and normal scoped commit/push regardless of outcome.
