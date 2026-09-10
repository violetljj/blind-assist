# MZ60: objective diagnosis and bounded native-query test

The [registered contrast](MZ60_NATIVE_QUERY_20260911.md) changes only the
training maximum used for positive frame-query BCE. It uses known native
witnesses instead of an unconstrained whole-image winner. Inference,
negative-query BCE, balanced local BCE, original negative replay and all
source/schedule/model/calibration choices stay as in MZ59 DIVERSE. The
[independent results](MZ60_NATIVE_QUERY_RESULTS_20260911.md) fail the retaining
gate: only 2 of 7 clauses pass. Both MZ59 arms and the retained OLD_NEG union
remain preserved; no output of the old experiments is overwritten.

## What the source and objective audits established

MZ59's 100 shallow-awning BODY_NEAR training examples were not exhaustively
trained in the difficult profile. The exact schedule presents 79 unique
examples 140 times: IDEAL46 presentations/40 unique, MERGE54/42 and DROP40/34.
Sixty-six of the 100 never appear under DROP. All 600 batches nevertheless
have 32 supervised query bits, because native labels do not disappear when
ToF returns disappear. The low training-set DROP acceptance is not a measured
architecture ceiling or proof that more effective exposure cannot help.

The independent information audit finds that all 160 shallow-awning held-site
native count tensors exactly equal tensors from actually fitted examples,
including all 40 held BODY_NEAR positives. Those targets retain 96 to 216
positive raster cells and always have native cells inside the sensor field.
The five coupled size/yaw/position settings are repeated across sites; this
partition mainly changes scene context, rather than testing unseen geometric
configurations. This explains why a future source should vary geometry and
split by configuration, while the present experiment can test how existing
labels are used without paying for another identical collection.

DROP packet duplicates contain nine conflicting-truth groups with 97 awning
frames. A deterministic packet-only predictor must make at least 6/6/21/42
BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR errors on those groups. This lower bound
does not apply to RGB+packet input: one inspected all-missing pair is visually
distinct. Unique PNG hashes do not by themselves prove preservation of that
distinction by the frozen encoder. Near/far bins must remain independent:
240/640 awning frames have additional actual range positives beyond the
declared anchor, with native truth retained. The complete audit is in
`artifacts.local/work/mz60-objective-design-20260911/info-audit-v1/REPORT.md`.

At the saved MZ59 DIVERSE endpoint, 58/100 of those training examples have
native BODY_NEAR winners but only 16 are accepted. The 84 misses contain 51
native winners below cutoff and 33 known nonnative winners; no UNKNOWN winner
occurs in that subset. The unresolved errors therefore include both ranking
and score/threshold behavior.

With the original batch denominators, each positive query has coefficient
0.25/32=0.0078125. For the awning DROP presentations, median individual
positive-cell and negative-cell coefficients are 0.00036973 and 0.0000131745.
The concentrated query term has about 21.13 times one positive-cell weight
and 593 times one negative-cell weight. This compares coefficients, not
actual shared-parameter gradients; total balanced-positive mass is not tiny.

Seventeen of the 40 DROP presentations have known nonnative winners at the
saved endpoint. The original local-plus-query derivative at each of those
winners would still increase its score. This is an endpoint counterfactual
using saved scores and original batch denominators, not an observation of
the actual training trajectory. The CPU synthetic gradient check demonstrates
the same conflict directly, and the new loss removes the positive query
gradient from nonnative/UNKNOWN winners while preserving negative-query and
no-witness behavior. These checks do not prove a useful learned model.

The full objective audit, including all presentation identities, coefficients,
gradient examples and its preserved JSON-serialization preflight failure, is
under `artifacts.local/work/mz60-objective-design-20260911/`.

## Prior art and why this comparison is still useful

This is not a claim of inventing native-positive supervision. The project's
[MZ20](MZ20_RANK_OBJECTIVE_RESULTS_20260910.md) already rewarded true BODY
witnesses together with cross-frame ranking and changed negative coverage.
It restored some pole recall with residual false alerts on its ROI model.
MZ60 isolates just positive-query responsibility on the current full-raster,
echo-independent model and richer source; it adds no ranking term.

The WELDON paper describes standard multiple-instance max scoring and an
extension that aggregates selected positive and negative evidence. It supports
the general distinction between image classification and region evidence.
Our present change instead uses available native instance labels during
training and retains the current prediction pool; it does not implement
WELDON or transfer that paper's performance claims.
[WELDON, CVPR2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Durand_WELDON_Weakly_Supervised_CVPR_2016_paper.pdf)

One paper was inspected through its primary-paper search excerpts; duplicate
DOI/CVF links are the same publication. The direct motivation and falsifier
here come from the current code, exact source schedule and gradient checks.

All data remain consumed controlled Development. Native-positive winning
cells establish label agreement, not causal feature use or true sensor
detectability. No hardware, natural-scene, clearance or App-safety conclusion
follows from this experiment.

## Runtime identity observed before scoring

The rebuilt 5542502528-byte FP32 training cache has SHA256
`21c6b56efe15131757c2f248680e4eeaac0e8ade2f231af915ecb11023b835f0`, exactly the
same as the preserved MZ59 feature-cache receipt. The old cache remains
deleted; this compares newly computed bytes against its sealed hash. All
6014 cached RGB feature rows therefore match the prior training material,
in addition to exact schedules, label arrays and initial model parameters.
This strengthens the matched objective comparison without rerunning the
old fitted baseline. No inference result is inferred from this identity.

## Completed result and execution cost

The single 600-step fit from the exact MZ56 GLOBAL initialization completed
on CUDA in 192.818753 seconds, including feature preparation 71.620009,
head fitting 19.138 seconds and evaluation 88.458526. The independent CPU
saved-output score took 3.064275 seconds, exit 0. Execution and audits pass;
the scientific gate fails. All 1546 prior arrays, 7 schedule arrays and 4830
prior metric rows remain exact. The scorer checked 229056 known scalar
decisions and 61440 native-winner lookups; UNKNOWN is preserved. No old
baseline cohort was reinferred and no threshold search was performed.

Final OR DROP gives held640 TP453/FP83, legacy noncal TP2722/FP48 and MZ48
nonfit TP518/FP25. New-family held480 native BODY_NEAR additions remain 1,
not greater than 1. Only legacy TP and MZ48 nonfit TP meet the seven frozen
clauses. Retain this positive-pool-only recipe as NEGATIVE_CONTROL.

Against MZ59 DIVERSE final OR, MZ55 gains 4 TP and loses 9, with 15 added FP
and 1 removed. All 15 added FP are HEAD_NEAR in unsupported contexts:
rod6, grille7 and sign2; 11 winning cells are known nonnative and 4 UNKNOWN.
All nine lost TP are HEAD_FAR (rod8/sign1). HEAD_NEAR cutoff falls from
3.303526 to 2.318817 while HEAD_FAR rises from 3.326738 to 4.367750 under
the unchanged calibration rule. Eleven added MZ55 FP have lower raw scores
than DIVERSE, so this is not uniform growth in false-alarm logits.

Shallow-awning DROP TP changes from 16 to17 of100 training positives and
1 to2 of40 held positives, but native-winning TP stays 7 and1 respectively.
Both new awning hits retain a known nonnative winner at the bottom raster
row. Across all160 positives, native winners decrease 99 to91; remaining
misses split into82 native winners below cutoff and58 known nonnative
winners below cutoff. Native-winning final OR counts do not independently
credit the new branch for retained-baseline positives. The isolated loss
change has not supplied the needed localization or error-cost retention.
The fixed schedule still exposes only34/100 training positives under DROP;
this is not a fully trained performance ceiling or proof of absent RGB
information. Detailed immutable evidence is under
`artifacts.local/work/mz60-native-query-20260911/score-interpretation-v1/`.

The temporary feature cache was deleted only after scoring, releasing
5542506496 allocated bytes (logical size5542502528); durable source,
checkpoints and receipts remain. Cleanup verified no owning process and an
absent cache in `cache-cleanup-v1/receipt.json`; its SHA256 is recorded in
the delivery record. Scratch directory remains empty. No active MZ60
process or feature handle is retained.

MZ61 is only a prepared 4096-frame geometry-source design at this point;
it has neither been registered nor captured. It is not a result or an
automatic extension of this stopped experiment.
Root's static review passed all 8 inputs/12 outputs, 1024 geometry identities,
4096 unique cases, 2048 support pairs and zero split collisions; the receipt
is `artifacts.local/work/mz61-geometry-source-20260911/root-review.json`.
