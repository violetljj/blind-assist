# MZ74: frozen model response to a surviving close return

EXPLORE, consumed controlled Development. The user prioritizes RGB plus ordinary
or partially available ToF; deliberately all-invalid inputs are secondary.

Question: with an unresolved pair of nearby geometric returns, how does keeping
one existing return affect the trained model's body/head detections and false
positives? Hypothesis: deletion of both returns hides useful geometric evidence,
but assuming the near target survives can overstate recovery on small objects.
The paired nearest/farthest scenarios expose that dependence without assuming
which return is strongest. They are uncalibrated sensitivity inputs, not hardware
detection probabilities or guaranteed bounds on sensor performance.

Use both original MZ70 CONTROL/DIVERSE checkpoints, original cutoffs, and the
complete unchanged MZ37 plus OLD_NEG OR pipeline. No fit, threshold search or
new calibration. Compare the existing IDEAL, MERGE_CLOSE and DROP_CLOSE results
against CLOSEST_REPORTED_PROXY and FARTHEST_REPORTED_PROXY from
[tof_return_sensitivity.py](tof_return_sensitivity.py). That implementation keeps
the selected distance byte-exact in its original slot when both are valid and
their separation is strictly below 0.600 m. All other bytes are unchanged;
never use labels to choose the endpoint or feed profile identity to the model.

One run covers all 4,096 MZ61 and 4,096 MZ67 frames. Primary reporting is each
source's original HELD geometry, all four queries separately, with TP/FP/FN/TN,
UNKNOWN, candidate/final OR, and actual bit additions/losses versus the old
profiles. TRAIN, CAL and all-source slices are descriptive diagnostics. The
native cell evidence may explain winners in the scorer, never in inference.
All groups are already consumed Development; no fresh confirmation claim.

The run encodes RGB once per frame for both profiles and heads. First replay
the first 16 sorted original TRAIN IDs per source under all three original
profiles: 32 additional RGB loads, frozen raw-output tolerance 2e-5 absolute /
1e-6 relative, exact final decision signs and discrete outputs. Winner ties may
change without changing pooled logits; record this separately. Stop on a
checkpoint, input hash, arithmetic or decision parity mismatch. Do not silently
change the reference. Mechanical failures retain their log and require an
evidence-based correction before a separately recorded resume.

The budget is 8,192 new unique-frame inferences, two new profiles per frame,
32 initial parity frames, zero training and zero new cutoffs. Independent scoring
starts after a successful sealed run. No model/profile is automatically promoted:
report gain and cost for both endpoints. If benefits depend on always retaining
the nearest return, prioritize real return/quality evidence; if both preserve
useful detections, retain that observation for a later explicitly tested training
or fusion change. If neither helps, reject preservation alone for the tested
role and diagnose the score/geometry conflict. Either outcome leaves original
MZ70 models and their recorded challenger dispositions intact.

CPU handles compact I/O and small packet arrays; model inference uses the
available primary GPU. No dense feature cache is persisted. Worker-only asset
metadata extraction is an independent source-repair engineering task, with no
new source admission or labels entering this comparison. Close archive/image
handles and verify process exit; retain predictions and diagnostic evidence.
