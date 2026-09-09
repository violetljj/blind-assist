# Frozen joint decoder retains large near/far gains on paired translation

2026-09-09 EXPLORE. JOINT passes all3 predeclared spatial-transfer criteria on
the existing750-pair EVAL diagnostic. This is a substantial controlled-Development
advance in observed HEAD range attribution, not full-model alert replacement,
new-source confirmation, calibrated metric depth, or product/safety evidence.
[Protocol](BODY_QUERY_CONTEXT_DISTANCE_PROTOCOL_20260909.md),
[inference](body_query_context_distance.py),
[independent probability analysis](body_query_context_distance_analysis.py).

## Main effect

All3 methods see identical RGB; decoders and TRAIN normalization remain unchanged
after the [relation-source comparison](BODY_QUERY_CONTEXT_DECODER_RESULTS_20260909.md).
No additional fitting, threshold selection or source capture occurred. Native
labels are evaluator-only. Primary denominator is750 rigid near/far pairs with
1500 accepted HEAD-positive endpoints and2250 positive HEAD-near query cells.

| Primary distance EVAL | Original10k B | LOCAL | JOINT |
| --- | ---: | ---: | ---: |
| HEAD-near query TP /2250 |1 (0.044%)|1889 (83.96%)|2085 (92.67%)|
| HEAD-near query FP /2250 |1|114|36|
| Wrong-far FP on near endpoints /750 |730|205|45|
| Exact two-range HEAD decisions /1500 |638 (42.53%)|1057 (70.47%)|1350 (90.00%)|
| Far-score direction correct /750 |574|673|694|
| Mean far-minus-near far-score |−0.06815|0.56948|0.83266|
| Count-derived HEAD alert hits /1500 |1402|1380|1325|
| Count-derived BODY false alerts /1500 |113|92|120|

Metric audit (2026-09-09): 1/2250 is 0.044444%, so the percentage above is
correct. The 90% figure measures individual frames, not complete near/far pairs.
Requiring both endpoints to have the exact two-range state gives JOINT 618/750
(82.4%) and original B 0/750. The frozen predictions are unchanged; the audit
receipt is `artifacts.local/work/body-query-fresh-size-20260909/historical-metric-audit.json`.

JOINT exceeds the frozen near recall50%, wrong-far<=150 and exact range>=1200
criteria. LOCAL passes only near recall, so it remains a useful simpler comparator.
Unlike direction ranking alone, the joint result improves absolute correct range
assignment and removes most opposite-range activation. Ties retain the frozen
1e-6 rule. Both range scores are independently computed by3-cell count convolution.

Across all2500 correlated pairs, JOINT near query hits7010/7500, wrong-far161/2500
and exact range4554/5000. Region variability remains: all-source big07 exact108/160
and near recall76.25%; dense02 exact190/230 and near82.90%. These are descriptive
region slices, not additional independent confirmations. Primary EVAL retains
68,304,102 native UNKNOWN pixels; these are not converted to free space.

## Retained alert and range evidence interface

The new count-derived alerts regress, as in the relation-source evaluation.
Do not promote JOINT as a replacement BodyQuery B model. The delivered
[ContextEvidence](body_query_context_evidence.py) returns unchanged baseline alert
logits/support and a separate JOINT range branch from one shared frozen backbone
pass. No asserted range is UNKNOWN; disagreement is exposed, never silently used
to suppress an alert or declare CLEAR. BASE alert scores on distance frames match
the historical predictions exactly, retaining1402 hits and113 BODY false alerts.

[Interface parity check](body_query_context_evidence_check.py) also passes all3000
relation EVAL frames:0 alert decision changes, exactly0 maximum alert/support
probability difference, and7.75e-7 maximum geometry-count probability difference.
BODY/HEAD alert-versus-range disagreement counts are55/84; no calibration or
safety meaning is attached to the absence of a range assertion. This preserves
508/600 original complete alert groups by construction and verified parity; it
does not manufacture a passed477->508 count-derived replacement result.

Warm CUDA batch1 forward median changes16.86ms baseline to19.29ms two-output;
p95 changes23.77ms to27.60ms,30 samples each. These are measured research runtime
costs, excluding RGB preprocessing/I/O, not Android latency or sustained rate.

## Evidence and limits

The independent analyzer verifies all primary/all/region/family/partition counts,
native count-event identities, normalized probabilities, paired memberships,
historical metadata/native/prediction hashes, final decoder/normalization/threshold
bindings, and retained alert parity. Inference separately rehashes every RGB.
The unchanged baseline forward has exactly0 probability difference. Validation
PASS and spatial gate PASS coexist with the failed full alert replacement gate.

All frames were previously admitted and consumed, with shared sites/assets.
No HEAD-negative endpoints exist here, so HEAD alert false-positive rate is not
estimable. This does not establish natural camera robustness, metric3D geometry,
continuous approach timing, user benefit or safety. A new independent source and
unchanged-method test are needed before broadening the claim. Current gains also
do not isolate context from decoder capacity, direct raw features or optimization.

Durable outputs: `artifacts.local/work/body-query-context-distance-20260909/run-v1/`
and `artifacts.local/work/body-query-context-decoder-20260909/interface-check-v1/`.
All checkpoints, features, predictions, selections, UNKNOWN evidence and validation
attempts are preserved. No additional fits; processes exited and no persistent
worker/service remains. Retain JOINT as a geometry component alongside original B.

Delivery validation uses a staged-source snapshot to exclude unrelated WIP. Source
syntax/report links,10 knowledge unit tests and library validation pass. The
decision engine history-recall check remains0.70 against0.80; an unchanged HEAD
snapshot reproduces the same three failed cases. This unrelated existing gap is
retained in `artifacts.local/work/body-query-breakthrough-delivery-20260909/`.
Initial snapshot path/environment setup failures and a corrected one-terminal-per-
experiment association are preserved in its attempt logs. Executed training source
bytes are preserved, including harmless trailing blank lines; other whitespace
checks pass. No model/evaluation outputs were changed for delivery.
