# HEAD-P1 results: appearance diagnosis and matched consistency

Completed:96-frame diagnostic plus two matched2000-step fits. The fixed weak
consistency recipe does not improve original EVAL64 over matched ERM. Retain
Coverage1198; neither new fit is a replacement. Initial protocols remain
immutable in HEAD_PAIRED_20260908.md and HEAD_PAIRED_FIT_20260908.md.

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

## Pre-model source recovery

The12-frame v1 canary preserved every positive native support mask and correct
BODY/HEAD label, but failed all8 appearance comparisons because318..2411 pixels
per head changed between UNKNOWN and known0 near tree/sky boundaries. Full
invariance admission stayed failed; no model was evaluated on those frames.

A separate4-frame original/original source probe disabled background mesh world
position offset in the task-owned editor memory, before fixture creation. Both
CLEAR and HEAD_ONLY repeats then had zero native support/UNKNOWN differences.
`tools/head_paired_static_control.py` and `tools/head_paired_capture.py` preserve
this explicit static-background source contract without saving map/material
assets or altering shared renderer files. The subsequent12-frame v2 canary and
full acquisition use this same control in every appearance. A retains the
original lighting/material/geometry settings but is a static-background rerender,
not a claim of exact old dynamic RGB equality. Fixed sample/metric definitions
and the strict native-equality criterion are unchanged. The initial protocol is
retained as `artifacts.local/work/head-paired-20260908/protocol-p1-v1.md`.

## Completed frozen diagnostic

Full96 passed:64/64 appearance comparisons preserve exact native positive and
UNKNOWN masks, geometry and actual RGB change. Native depth also matches exactly
in all72 canary/full appearance comparisons. BODY0/96 and HEAD48/96 match the
intended relations. Controller verified all130 transferred evidence files.
Both original DEV32 prediction checks passed bit-for-bit. Frozen inference took
3.63 seconds on worker CUDA with zero optimizer steps. Result SHA256:
`48a167394842477129a7234c494d70fec4aaf03d867f3e063255ab627422c744`.

| Primary Coverage1198 factor | TRAIN damaged | EVAL damaged | Total | Trigger |
|---|---:|---:|---:|---|
| L, reduced illumination | 1/8 | 2/8 | 3/16 | yes |
| M, opaque material swap | 0/8 | 0/8 | 0/16 | no |

The three L cases are different failures, not three new missed alerts:

- TRAIN hanging_sign unit07: positive near probability .99708→.44551 changes
  TP→FN at the original DEV cutoff .99026066. Signed relation remains positive
  (11.6778→8.7063), and positive support stays12/12. The alert loss cannot be
  explained by missing positive support alone.
- EVAL cabinet unit01: signed relation +.10988→−.07881 reverses. Near remains
  FN and support stays0/16; this was already a substantial A failure.
- EVAL oblique_rod unit06: positive support5/6→3/6 loses two cells. Signed
  relation instead improves −.98350→+.32574. Near remains FN.

EVAL A/L/M all yield1/8 TP and0/8 FP. HEAD AUC is .84375/.765625/.859375;
known-positive pixel recall is57.5%/56.25%/57.5%. TRAIN A already has6/8 TP
despite100% positive support. A has15/16 positive relation margins overall;
L also has15/16, concealing one lost and one recovered ordering. Report these
as separate end points rather than equating support, ranking and alerting.

Historical F1134 damages2/16 units under L and1/16 under M; none adds an alert
miss. These are relation/support failures, with EVAL AUC improving while support
recall falls. Neither comparator isolates a single backbone-only cause.

The primary predeclared trigger is met, so execute the fixed matched fits in
[HEAD_PAIRED_FIT_20260908.md](HEAD_PAIRED_FIT_20260908.md). Illumination sensitivity
is present, but the major EVAL deficit already exists in A. This diagnosis does
not establish appearance shift as its dominant cause or predict that consistency
will fix it. No threshold was retuned for this diagnostic.

Evidence: `artifacts.local/work/head-paired-20260908/inference-v1/`,
`evidence-all-v2/`, `native-supplement/native-depth-pair-diagnostic.json`, and
`capture-handoff.json`. Retain the failed dynamic canary. Four-frame A/A RGB
MAE remains2.10–2.29/255 despite exact native masks; temporal rendering residuals
limit material attribution, especially where M RGB change is similarly small.

## Matched fit results

Both fits completed exactly2000 steps, total591.74 seconds including evaluation,
on the original Torch2.9.1+cu128/RTX3060 worker. Initialization digests match,
BN buffers remain unchanged, and the saved training sequence regenerates
exactly from seed17. Each arm consumes40000 original TRAIN draws and24000
paired TRAIN draws; diagnostic EVAL fitting draws are0. No intermediate
checkpoint selection, retry fit, extra seed or loss-weight search occurred.

Original EVAL64 contains32 positives and32 negatives per head. Each model uses
its own original DEV128 cutoff; the original Coverage row is contextual because
it used a different sampling sequence and exposure. ERM versus CONSISTENCY is
the matched comparison.

| Model | HEAD TP /32 | HEAD FP /32 | HEAD AUC | HEAD positive-pixel recall | BODY TP /32 | BODY FP /32 |
|---|---:|---:|---:|---:|---:|---:|
| Original Coverage1198 | 12 | 1 | .708984 | 55.94% | 16 | 1 |
| Matched paired ERM | 17 | 8 | .702148 | 58.39% (167/286) | 10 | 1 |
| Matched paired CONSISTENCY | 17 | 8 | .702148 | 57.69% (165/286) | 10 | 1 |

CONSISTENCY adds no HEAD operating-point or AUC benefit over ERM. Its positive
support loses2/286 cells; known-negative support activation changes only
3.975%→3.960%. BODY AUC changes .720703→.722656 while positive support recall
falls62.22%→61.21%. Original EVAL HEAD false-positive rate is25% for both new
fits despite DEV selection at6/64 false positives (9.375%). DEV calibration
therefore does not establish EVAL false-positive control.

Both new fits have DEV HEAD TP52/64 and FP6/64. HEAD cutoffs are ERM .98700106,
CONSISTENCY .98392510 (Coverage .99026066). Paired EVAL remains small: eight
HEAD-positive and eight CLEAR images per appearance, with no positive BODY.

| Paired EVAL appearance | ERM HEAD TP/FP | CONSISTENCY HEAD TP/FP | ERM AUC | CONSISTENCY AUC | ERM support recall | CONSISTENCY support recall |
|---|---:|---:|---:|---:|---:|---:|
| A | 1/0 | 2/0 | .906250 | .921875 | 26/80 (32.50%) | 25/80 (31.25%) |
| L | 3/0 | 3/0 | .796875 | .796875 | 38/80 (47.50%) | 36/80 (45.00%) |
| M | 2/0 | 2/0 | .953125 | .953125 | 39/80 (48.75%) | 38/80 (47.50%) |

The small A operating-point gain does not reproduce in original EVAL64 and
coexists with lower support recall in every paired EVAL appearance. Both fits
reduce primary L-damaged units3→1/16; both retain the EVAL cabinet failure.
Thus most reduced sensitivity already appears in matched ERM. The consistency
term does not earn credit for that shared improvement. Reduced pair damage
also does not offset the substantial miss/false-alarm tradeoff or BODY loss.

The original TRAIN hanging_sign miss is repaired by both arms: A/L positive
probabilities approach .9998 and support remains12/12. Both paired TRAIN sets
have8/8 TP,0/8 FP and100% positive support in every appearance. EVAL cabinet
remains unsupported (0/16) and missed in A/L; the L margin stays negative
(-.10864 ERM, -.04049 CONSISTENCY).

Crucially, EVAL oblique_rod loses its damage flag by becoming worse in both
appearances: original A/L support5/6→3/6 becomes1/6→1/6 in both fits, with both
alerts still missed. Equality is not recovery. The extra CONSISTENCY A TP is
crossbar unit04: probability .976794→.985043 plus cutoff .987001→.983925.
At the ERM cutoff both arms still have1/8 A TP. This is a joint score/cutoff
change, not independent evidence of broad robustness.

Compared with original Coverage, paired EVAL A support recall falls57.5% to
32.5%/31.25%. Higher paired AUC and selected TP cannot conceal that evidence
loss. Because new ERM also changes sampling/exposure, its differences from
Coverage cannot isolate augmentation from oversampling or schedule effects.

## Supplementary A/A noise reference

The pre-model four-frame probe uses one TRAIN crossbar unit, CLEAR twice and
HEAD_ONLY twice. Frozen Coverage HEAD probability changes by .00319/.00589
without alert flips; corresponding raw-logit changes are2.0083/.2612. HEAD
support-map maximum changes approach .196/.193 despite mean changes below .009.
F1134 likewise has no alert flips. These two A/A pairs cannot bound noise in
all16 units. Small signed-margin reversals and local support-cell changes must
not be attributed entirely to pure illumination/material causality. The main
result is observed sensitivity under the appearance-capture intervention.
This supplementary forward-only result does not revise the predeclared trigger
or choose either fitted model.

## Disposition and validation

Retain the exact paired source and diagnostic as reusable components. Archive
this fixed .1 probability-consistency recipe as a scoped negative control;
it is not evidence that all consistency methods or appearance augmentation
fail. Keep Coverage1198 and stop this appearance-training branch at its budget.
No MIL, Dice, backbone change, plaza reopening or App promotion follows.

Validation: four evaluator edge-case tests and seven trainer tests passed;
capture/native admission passed; original DEV32 predictions match exactly for
both frozen checkpoints;23 transferred fit files passed SHA checks; controller
regenerated the full training schedule, validated all48 paired TRAIN native
masks and recomputed both fitted diagnostic results exactly. Worker additionally
revalidated immutable inputs and both checkpoint/optimizer hashes. All task
capture/model processes and scheduled tasks are released; transport ZIPs were
removed after verified transfer. Durable raw data, model/optimizer/RNG files
remain on the worker as project evidence, without reserved live compute.

Fit result SHA256:
`bda8126e16c5adeb95019d7793b8fe9de35270f64f7188d883745c54a4a22b05`.
Primary evidence under `artifacts.local/work/head-paired-20260908/`:
`fit-v1/result.json`, `fit-v1/controller-support.json`,
`controller-verification.json`, `noise-v1/result.json`, `model-release.json`.
Frozen protocols, raw prediction caches, source hashes, exposure and terminal
job receipts are retained. This is one-seed consumed same-world Development,
not independent-world or real-device evidence.
