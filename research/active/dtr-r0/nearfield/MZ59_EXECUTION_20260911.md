# MZ59: coverage result, execution cost and storage reduction

The [matched training comparison](MZ59_TRAINING_DIVERSITY_RESULTS_20260911.md)
fails its declared no-added-FP gate. DIVERSE gains one native-winning BODY_NEAR
event over the retained OLD_NEG union on the 480 MZ55 new-family heldout frames;
CONTROL gains zero. But final union FP increases from 79 to 81 on all 640
heldout frames and from 45 to 51 on legacy noncalibration. Both arms start from
the same MZ56 GLOBAL checkpoint, use the same model, loss and 600-step budget,
and calibrate only on the original 1256 rows. MZ55 supplies no cutoff rows.

Across all 2560 MZ55 frames the DIVERSE union gives 1852 TP / 322 FP, compared
with CONTROL 1836 / 319 and OLD_NEG 1832 / 317. These totals include training
rows. On heldout640 the corresponding results are 454 / 81, 451 / 79 and
451 / 79. On the older MZ48 nonfit partition CONTROL gives 514 / 23 and
DIVERSE 505 / 21. More varied training changes the tradeoff, but this run
does not establish a preferable operating point or justify replacing the
retained comparator. No threshold, checkpoint or extra training was selected
after seeing these outcomes. All named heldout sites remain consumed
Development, not fresh confirmation.

## Where coverage helped and where it failed

Saved-output inspection gives shallow-awning DROP BODY_NEAR TP of 7/160 for
OLD_NEG, 11/160 for CONTROL and 18/160 for DIVERSE. On the 100 training
positives these are 7/11/16; on the 20 calibration-named positives 0/0/1;
on the 40 heldout positives 0/0/1. Thus the large remaining miss count is
present in training data too, although this budget cannot establish the
ceiling of the architecture or training method.

CONTROL's four added awning events all have locally known nonnative winning
cells. DIVERSE's eleven additions over OLD_NEG have three native winners and
eight known nonnative winners. All three native winners are inside both
crop and sensor field. The single heldout native gain is the unsupported
near BOTH awning frame2296; its paired supported frame2297 remains negative.
That pair prevents an inference that surviving support context is uniformly
beneficial or that the new detector has learned context-invariant geometry.

The two extra heldout false alerts are BODY_NEAR on actual far obstacles:
an awning BOTH frame426 and a rod BODY_ONLY frame638. The six extra legacy
false alerts are all HEAD_FAR, five on relation10000 and one on distance5000.
The unchanged calibration rule yields a HEAD_FAR cutoff of 3.326738 for
DIVERSE versus 5.842582 for CONTROL; those six DIVERSE raw scores lie between
3.333 and 3.518. Training coverage changes both learned scores and the
calibration maximum. The comparison measures their combined effect under
the fixed rule; it cannot attribute the failure solely to representation
or solely to calibration. No additional cutoff search was performed.

On MZ48 nonfit, DIVERSE versus CONTROL gains five and loses fourteen true
events, a net loss of nine; its FP falls by two. On all MZ55 it gains seventeen
and loses one true event, but that aggregate includes fitting rows. These
paired counts distinguish a coverage tradeoff from a general improvement.
The next mechanism question is how to preserve near/far separation and old
negative behavior when learning new shape evidence under missing returns.
It requires a separately declared contrast rather than extending this run.
Details are preserved in this task's `score-interpretation-v1/`.

## Diagnosis available before fitting

The saved-output MZ58 shallow-awning diagnosis found 160/160 BODY_NEAR
positives with target witnesses inside both full RGB and the sensor field.
This particular failure was not caused by cropping those targets away.
IDEAL and MERGE old-union TP were 158/160 and 160/160; DROP was 7/160.
DROP removed the target return in 152 frames and all measured packets in 49.
The original GLOBAL model's 153 misses included 48 native winners below its
cutoff and 105 locally known but nonnative winners below cutoff. Its highest
correct native score was 1.092, below the frozen 3.488 cutoff.

These observations separate target visibility, surviving ToF evidence, local
ranking and the final alert threshold. They motivated equal-budget coverage
training without deleting native-positive labels when their ToF return was
missing. A native winning cell is localization evidence; it does not prove
causal use of that cell or sensor detectability. Retain UNKNOWN local regions
and do not turn a missing return into a negative obstacle label.

Evidence is under `artifacts.local/work/mz58-diverse-transfer-20260911/awning-diagnostic-v1/`.
This analysis consumes existing evidence and does not alter the sealed MZ58
result or introduce a new hardware model.

## Actual execution

The primary GPU performed two 600-step fits and a shared seven-cohort,
three-profile evaluation in 252.975 seconds. Feature construction took
88.742 seconds; CONTROL fitting took 19.176 seconds and DIVERSE 16.902;
evaluation took 114.631 seconds. These are measured stages of this run, not
an end-to-end capture benchmark or device deployment latency.

Both heads shared one task-owned FP32 training cache for 6014 unique frames
and reused 2481 cached frames during evaluation. Another 7063 evaluation
frames were encoded in bounded batches, for 13077 RGB loads in total.
No dense native depth was read by the training/inference source. The model
forward receives RGB features, observed ranges/validity and static calibration;
native labels remain supervision/evaluator data. Sealed old predictions were
reused: there were no baseline cohort reruns. A 16-frame TRAIN-only initial
arithmetic comparison preceded fitting; initial parameters matched exactly.

The independent CPU scorer takes 3.829 seconds, reconstructs two cutoff
vectors and checks 458112
known scalar decisions, 122880 native winner lookups and 3630 prior metric
rows. All 1124 old MZ56 arrays and 105 MZ58 arrays are preserved. MZ36 retains
all 400 attempts and 80 UNKNOWN bits. MZ55 full-frame UNKNOWN remains 7417741.

The primary FP surface was explicitly fixed to the final OR before fitting
in `primary-definition.json`; the originally registered brief is retained.
A pre-score mechanical correction maps the old `attempted_frames` metric
denominator to the new `frames` name and rejects any differing count or
extra key. The original scorer and admission remain preserved under
`scorer-schema-correction-v1/`; no scientific output was overwritten.

The completed scorer does not need the temporary 5542502528-byte feature
file. The task's cache cleanup receipt records its verified release. Source
indexes, schedules, feature indexes, checkpoints, predictions and diagnostic
receipts remain durable. Primary training never overlaps primary UE capture.

## Actual storage reduction on both hosts

NTFS transparent compression of the 2560 independent MZ55 original native
depth files reduced their actual allocated storage from 2369781760 to
1506394112 bytes: **863387648 bytes saved, or 36.4332%**. Primary saved
273178624 bytes across 936 files; the worker saved 590209024 across 1624.
This is measured file allocation, not a claim based on ZIP logical size or
the whole volume's free-space change. Logical raw file size stays 2359623680.

Every file's SHA256, decoded array bytes, file identity and modification time
remain unchanged. Eighty same-source hardlink aliases retain all 2640 names;
physical storage is counted once. Original source bindings and package hashes
are unchanged. No original depth file was removed. Worker return traffic was
4706770 bytes of metadata/logs plus small outer receipts, with zero raw data
transferred. Both compactor processes exited and handles were released.

Compression used only exact file paths after ownership and writer checks.
The initial single-hardlink assumption failed before any mutation; that
preflight is retained alongside the corrected explicit alias audit. Combined
compact work was 21.613 seconds; per-host application plus full checks sums
to 41.508 seconds. These sums are not a measured dual-host wall clock.

The constrained helper and complete per-file receipts are retained under
`artifacts.local/work/mz59-training-diversity-20260911/`. This result authorizes
no bulk action on other datasets. Existing compact training packages are a
separate storage surface; their RGB bytes dominate, so native compression
alone does not establish an optimal RGB resolution or codec.

## Hardware learning and the boundary of the simulation

ST's DS14161 revision 12 gives a 45-degree horizontal and vertical detection
volume under specified full-FoV white-target conditions. Table 20, for 8x8
continuous 15 Hz and a full-FoV 17% gray target at 5 klux, lists typical
maximum ranges of 1150 mm for inner zones and 950 mm for corners; the stated
minimum values are 900 and 700 mm. Nominal 4 m capability cannot be assumed
for outdoor, dark or partially occupied zones. [ST datasheet](https://www.st.com/content/st_com/en/technical-documents/DS14161.html)

ST's UM3109 section 4.10 describes up to four returned targets per zone,
configured at compile time, with one by default and a stated 600 mm minimum
separation for distinguishing two targets. This is distinct from the same
document's 600 mm minimum distance for crosstalk calibration.
[ST user manual, distributor-hosted copy](https://www.pololu.com/file/0J2030/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf)

Our inference from these specifications is to preserve partial returns and
missing evidence as central cases, especially for thin/partial obstacles.
The current deterministic MERGE/DROP profiles are controlled stress tests,
not calibrated probabilities of real VL53L8CX outcomes. Their 0.6 m rule does
not establish reflectance, ambient-light or mixed-zone sensor fidelity.
Neither this learning nor MZ59 changes the frozen simulation profiles.

The useful innovation question remains whether RGB plus limited, unreliable
metric observations can localize a head/body hazard while controlling false
alerts. The current coverage result narrows that question but does not answer
it successfully; natural scenes, real hardware and product safety remain
unmeasured here.
