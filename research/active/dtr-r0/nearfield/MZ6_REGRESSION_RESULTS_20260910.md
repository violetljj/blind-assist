# MZ6: fixed-ensemble regressions are mainly conflicting readouts

2026-09-10, zero-fit consumed Development diagnostic. **The 27 MZ5 exact
regressions contain 32 wrong event bits: 26 have a correct unimodal branch
overruled by the other branch; six have both branches wrong.** This supports
preserving useful modality decisions, and does not establish missing sensory
information as the dominant cause. No thresholds, weights or alerts changed.

## Reproduced evidence and denominator distinctions

[Diagnostic code](mz6_regression_diagnostic.py) verifies saved prediction, source
index and native-depth receipts and reconstructs the original52 gains/27 losses.
All27 original RGB images were inspected in three contact sheets; their RGB and
native-depth file hashes match the index. Four logits for RGB, ToF, MZ1 and MZ5,
full/cropped visible support, all64 two-return zones and validity bits are saved
for every regression. Observations are the same already consumed controlled UE
source, not new confirmation.

The27 rows are23 far and4 near;10 BOTH,8 BODY_ONLY and9 HEAD_ONLY. Their32 errors
are31 false negatives and1 false positive. Frame3841 has a new BODY_NEAR false
positive alongside a missed BODY_FAR; it is not a new wrong-far or cross-body
error under those specific definitions.

| Wrong bits in the27 exact regressions | Count /32 |
| --- | ---: |
| RGB correct, ToF overrules |17|
| ToF correct, RGB overrules |9|
| Both branches wrong |6|

Absolute ensemble error margins have min/median/max0.0071/0.5284/2.4523.
Only5/32 are within0.25 of zero,15/32 within0.5, and24/32 within1. These are
descriptive bins, not searched decision thresholds or calibrated confidence.
Some losses are near cancellation, others substantial disagreement. “Both
branches wrong” describes classifier outputs, not absence of information.

## All-event recall transitions, including already-inexact rows

| Event | Old TP lost | New TP recovered | Net TP |
| --- | ---: | ---: | ---: |
| BODY_NEAR |2|2|0|
| BODY_FAR |17|2|-15|
| HEAD_NEAR |4|3|-1|
| HEAD_FAR |15|0|-15|

The32 far TP losses in this table are a different selection from the32 errors
inside the27 exact-regression rows. Across all far losses,17 are RGB-correct
overruled,8 ToF-correct overruled and7 both-wrong. All32 have>=3 native visible
pixels inside the ToF angular crop, and29/32 have zone-center approximate event
support. Thus the clean simulated source usually carries supporting geometry;
the present evidence does not justify saying new observations are always needed.
Zone-center support is an approximation and supplies no actual return identity.

Inside the27-row selection,31/32 error bits have full native positive support,
30/32 crop support and26/32 zone-center support. The false positive has no native
support; the one missed positive outside the crop remains a coverage distinction.
Visual inspection shows bars/support fixtures against buildings and vegetation;
this is contextual observation, not proof that background caused the errors.

## Corrected baseline interpretation and next check

MZ5 exact improves1353->1378/1500, while negative controls improve544->578/600.
The other900 rows decrease809->800: the net25 gain is34 negative-control gains
minus9 elsewhere. Near recall is essentially unchanged. Wrong-far12->4 means
fewer far activations in near-only conditions, not eight recovered near hazards.

Retain the compact132872-parameter **readout** baseline for its controlled error
reduction, with far recall and its measured readout-only cost explicit. The frozen
RGB encoder means this comparison does not diagnose backbone suppression during
joint training. The local MLP recipe remains a weak mechanism control, not a
reason to reject all spatial fusion.

The separate [bounded short-sequence pilot](MZ6_SHORT_SEQUENCE_PROTOCOL_20260910.md)
tests a genuinely different opportunity: actual recent raw returns when a small
foreground return is artificially absent now. It compares current MZ5, trailing
three-logit mean and locally matched compatible packet completion. It does not
retune this consumed cohort or claim to have solved the27 regressions.

## Artifacts and verification

- [Result](../../../../artifacts.local/work/mz6-regression-20260910/run-v1/result.json),
  [all27 records](../../../../artifacts.local/work/mz6-regression-20260910/run-v1/regressions.json),
  [receipt](../../../../artifacts.local/work/mz6-regression-20260910/run-v1/receipt.json).
- [RGB sheet1](../../../../artifacts.local/work/mz6-regression-20260910/run-v1/regressions-1.jpg),
  [sheet2](../../../../artifacts.local/work/mz6-regression-20260910/run-v1/regressions-2.jpg),
  [sheet3](../../../../artifacts.local/work/mz6-regression-20260910/run-v1/regressions-3.jpg).

CPU scalar/metadata workload, zero model forwards/training. Separately expressed
scalar reconstruction agrees on all6000 event signs and52/27 paired transitions;
1500 original alert rows and all hashed inputs remain unchanged. This is a
descriptive decision diagnostic, not a newly selected fusion policy or safety
claim. Reproduce with `mz6_regression_diagnostic.py --output` pointing to a new
child of `artifacts.local/work/mz6-regression-20260910/`.
