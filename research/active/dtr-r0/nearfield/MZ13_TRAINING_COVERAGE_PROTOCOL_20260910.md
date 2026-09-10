# MZ13: existing-source training coverage for selective addition

User authorized addressing the MZ12 hanging-sign attribution errors with existing
data. Seventeen of18 added FP bits were signs, including erroneous BODY additions
on HEAD_ONLY distance endpoints. Their SOURCE scores/extent overlap useful
recoveries. Test training coverage before introducing a new architecture.

One bounded EXPLORE fit; no acquisition or new geometric mechanism. Keep the
MZ11 four-input per-query linear gate (20 parameters), input scales, positive-only
addition, zero initialization, seed111, AdamW lr.03/weight_decay.001 and600 steps.
Keep MZ5, MZ9 SOURCE, ContextEvidence and all upstream thresholds frozen. Add the
original relation TRAIN_ONLY5000 and distance train2500, all families, to old
TRAIN2500 plus consumed MZ6 clean200. Do not select only the observed17 errors.
No dataset/family/site/identity input enters the gate.

Apply the same mean BCE rule to nonempty cohort/query/class groups across four
training cohorts. The expanded mix changes relative cohort weights as a necessary
part of this coverage intervention; it is not an equal-presentation experiment
or proof that any single negative example caused a gain. One fit from zero, no
seed/step/learning-rate sweep or continuation. Log effective eligible positive
and negative counts; total source size is not the gate's effective sample count.

The new7500 source frames are the original training partitions. MZ12's3000
DEV frames remain excluded from weight fitting. Old MZ11 DEV1000 alone selects
per-query cutoffs strictly above every false-addition score, unchanged calibration
procedure. No cutoff fitting on MZ12 or sequence. All involved evaluation sets
are already consumed Development; even a successful result is not independent
confirmation. Upstream relation encoder training is explicitly acknowledged.

Freeze original MZ11 as comparator. Report old DEV, clean/stress sequence and
the unchanged MZ12 relation2000/distance1000, including signs, other families,
complete groups/sites and accepted/rejected TP/FP bits. Wrong RGB correspondence
uses existing MZ11 wrong SOURCE inputs with the newly fixed gate and cutoffs.
No new feature or observation extraction is needed for these evaluated cohorts.

Retain a favorable Development challenger only if sign-added FP on the3000
falls below17, total added FP per query does not rise in either MZ12 cohort,
far TP sum does not decline in either cohort, old DEV near and far added-TP sums
do not decline from MZ11, and thin recovery is at least46/49 clean and44/49 stress.
Every original MZ5-positive bit must remain unchanged. These comparison gates
do not relax or override MZ11's separate48/49 confirmation-admission failure.
Report individual lost/recovered bits even if sums pass. A trivial reject-all
gate cannot pass recall checks. Stop after this one fit irrespective of result.

Extraction uses the unchanged generic two-return geometric simulation and frozen
RGB pipeline, with native depth providing sensor preprocessing and evaluator
truth only. Invalid evidence is not CLEAR. CUDA for extraction/fitting; CPU for
metadata and scalar verification. Keep hash-bound inputs, outputs and failures
under artifacts.local; no App, deployment, hardware or safety claim.
