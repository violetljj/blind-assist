# HEAD-P1 appearance and relation pairing

EXPLORE, authorized after the completed HEAD-S1 analysis. Current Coverage1198
is the primary frozen diagnostic; F1134 is a historical comparator. S1 did not
recover HEAD from max/top6/LSE, but did not identify a backbone-only cause.

Question: does a geometry-preserving rendered appearance intervention damage
correct HEAD relations or visible positive support? The diagnostic separates
lighting and opaque fixture material changes rather than editing mask exteriors.
No threshold, pooling, architecture, Dice or backbone update in this phase.

## Fixed source and prediction

Use the first eight units of each accepted coverage64 TRAIN/EVAL source: four
fixture families twice per role, sixteen units total. Preserve those roles;
both are consumed same-world Development, not independent worlds. Select CLEAR
and HEAD_ONLY per unit. Render A (original), L (lighting-only), M (opaque fixture
material-only): 96 frames, two 2x2 contrasts sharing A. No outcomes select units.
The source generator freezes exact appearance settings before capture. A small
render/native canary admits the source; retain any failed canary and its reason.

For each appearance pair verify unchanged camera/wearer and object geometry,
native positive/UNKNOWN masks, target labels and actual RGB change. Actor
coordinates alone are insufficient. Full native-mask equality is primary;
any unequal pair is reported as NOT_EVALUABLE for strict invariance rather
than relabelled, silently dropped or forced equal. Preserve complete counts.
Geometry transitions receive recomputed native labels. No model sees depth,
geometry, masks, group names or source roles as inputs.

Freeze models separately. Coverage1198 checkpoint SHA256:
`d60c871fd0a292bfd5b30c625e5a4189632a7c42576fd6d951c71e40e6d8b7bc`;
F1134: `d911a3a717efb5a93fadb3268ba962db94455565805effc7077f75215670cf46`.
Use original Torch2.9.1+cu128 worker, historical model-source bytes, batch32,
RGB BOX-resized144x256, fixed historical DEV cutoffs. Save near/support raw
logits as well as probabilities before evaluator labels are loaded. Verify
one original DEV batch against each model's existing cached prediction.

## Metrics and decision before outcomes

Report per model/role/appearance/body head: TP/FP denominators, AUC/AP when
both classes exist, support IoU, known-positive recall, known-negative activation
and peaks including UNKNOWN. Pair rows retain probability and raw-logit signed
HEAD differences z(HEAD_ONLY)-z(CLEAR), their direction, four-bit correctness,
appearance-induced alert flips, and support-recall differences. Never take
absolute positive-negative separation as evidence of correct relations. No RSI
ratio or appearance invariance alone can certify success; constant/reversed
outputs fail relation correctness. Correlated units remain descriptive evidence.

The primary Coverage1198 supplies a useful appearance-failure signal if either
factor, among strict native-equal pairs, causes damage in at least two distinct
units: an A-correct HEAD decision becomes wrong, an A-positive signed relation
margin becomes nonpositive, or HEAD_ONLY known-positive support recall drops
by at least0.20. Report which condition fired and role/family breadth; this
is an exploratory continuation rule, not statistical significance. Raw-logit
changes without task damage remain sensitivity evidence only.

If a signal exists and TRAIN pairs admit exact known-mask correspondence,
continue the user-authorized matched ERM versus weak-consistency comparison,
freezing its concrete protocol before optimizer steps. Use identical paired
TRAIN images, exposure, initialization, batch sequence and total step budget;
the EVAL source never enters fitting. Include original DEV and coverage EVAL,
retain pixel recall/false activation and signed pair ordering, and calibrate
only on original DEV. The diagnostic's observed EVAL is explicitly consumed.
Do not stack MIL, Dice or a new fine-tuning policy into this comparison.

If no useful damage is observed, or strict invariance is not evaluable, stop
the appearance-training branch and retain the scoped result. Do not increase
randomization, recapture by model outcome, repeat S1, tune thresholds or start
other architectures. Complete reporting and scoped delivery either way.

Raw native source/capture and receipts remain in the task-owned worker
`artifacts/work/head-paired-20260908`; thin evidence lives under the matching
primary `artifacts.local/work/head-paired-20260908`. Release task-owned editor,
model processes and scheduled jobs, preserving durable evidence. No App or
baseline promotion follows automatically from this controlled diagnostic.
