# MZ43: matched training with restricted ToF observations

2026-09-11 EXPLORE. MZ40 exposed degradation of ideal-trained models when close
returns are merged or missing. MZ41's binary missing-query guard recovered four
true bits at four new false bits. MZ2 only stressed frozen checkpoints; it did
not train on these interventions. Hypothesis: exposure to the same restricted
observations during training improves the missed-obstacle/false-alert tradeoff
without requiring a larger model or extra visual inference.

Reuse MZ1's exact 5,000 cached observations and roles (2,500 TRAIN, 1,000 DEV,
1,500 EVAL), original 1028->128->4 MLP, initial state, 300 x 128 sample schedule,
AdamW settings, BCE and zero logit threshold. All are consumed Development.
Reproduce the ideal TOF_ONLY and FUSION fits and require exact final tensor and
loss parity with MZ1 before interpreting the comparison. Fit one MIXED version
of each from the same initialization. The only change is the training packet:
each sampled exposure cycles IDEAL, MERGE_CLOSE, DROP_CLOSE in equal numbers.
Use the frozen MZ40 600 mm close-pair operator; no arm ID, ambiguity flag, native
depth, material identity or query labels are predictor features. Keep explicit
validity, original frozen RGB features and original B alerts. Only TRAIN labels
enter fitting. No model/seed/threshold selection or continuation steps.

After all fits end, evaluate every fixed checkpoint on all three observation
conditions. Compare TOF alone, joint FUSION, and equal-logit ensemble of the new
ToF head with the unchanged MZ1 RGB head. Reuse old ideal checkpoints and MZ5;
report frozen MZ28/MZ37 on MZ36 as contextual baselines with their different
training/decision histories, not a matched causal comparison. MZ36's 400
attempts, 380 admitted frames and 80 UNKNOWN bits remain in the denominator.
No MZ36 or new-object labels enter fitting, calibration or selection. MZ42 and
the independently collected far block are not fitted or used to choose a model.

Primary effect: MIXED ensemble versus IDEAL ensemble on MZ36, separately for
merge and missing inputs: FP/FN, paired recovered/lost positives, added/removed
false positives, exact frames, and per-query coverage. Keep as a useful candidate
if restricted-condition pooled FN falls without pooled FP increasing, and ideal
MZ36 plus old EVAL show no FP/FN increase. Apply the same stated comparison to
joint FUSION separately. Report tradeoffs even if this criterion fails; never
select a threshold after viewing errors. Training reproduction failure means
INVALID_FOR_REQUESTED_COMPARISON, not evidence against restricted training.

Budget: two 300-step ideal reproductions and two 300-step mixed fits, one final
cached evaluation, and one focused independent packet/scoring/isolation check.
No backbone inference, native recapture or permanent dense cache. Model compute
is CUDA; scalar packing and scoring use CPU. Record actual device and timings;
release task-owned model memory/processes. Exact source hashes and a recoverable
compact checkpoint remain under `artifacts.local/work/mz43-restricted-training-20260911`.
This tests a declared sensor restriction, not calibrated VL53L8CX failure rates,
real-world safety, novelty, or natural-scene robustness.
