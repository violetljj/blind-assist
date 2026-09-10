# MZ47: twelve near examples do not beat matched replay

2026-09-11 EXPLORE. [Frozen comparison](MZ47_LOCAL_ENRICHMENT_20260911.md).
Native-local enrichment does not improve the predeclared non-fit transfer
cohort over equal-budget replay. Under DROP_CLOSE, ENRICH has14 true positives,
zero false positives and18 false negatives; REPLAY has18/0/14. Both improve
some decisions over the frozen comparator, but that alone does not establish
a contribution from the new examples. Retain this scoped negative control,
all checkpoints and the fixed original comparators. No default replacement.

## Matched experiment

Both arms start from the exact MZ20 BODY_RANK checkpoint and saved crop grid.
Architecture, feature backbone, normalization, availability bank, branch
selector and restoration parameters are fixed. Each arm uses300 Adam0.001
steps, the original balanced-local plus0.25 BODY-ranking objective, the same
16 original TRAIN examples per step, and four extra examples. ENRICH cycles
the12 pipe/ladder/pouch near frames. REPLAY uses original TRAIN examples with
the same four event-label patterns, selected with seed147. There is no new
fit-cohort outcome selection, schedule extension or checkpoint search.

Training and one original1000-DEV calibration per arm use DROP_CLOSE. Deleted
echoes lose their slot supervision; surviving labels keep their native
meaning. IDEAL and MERGE_CLOSE are evaluation-only, and merged midpoint slots
never inherit native IDEAL contributor labels. Positive events lacking an
eligible witness stay in the task denominator rather than becoming CLEAR.

## Held-out-from-fitting effects

Each entry is TP/FP/FN over32 non-fit positive event bits and96 negative bits
in32 frames. These are consumed Development data, not fresh confirmation.

| Observation | Fixed MZ37 | REPLAY | ENRICH |
|---|---:|---:|---:|
| IDEAL comparator | 19/1/13 | 22/1/10 | 18/1/14 |
| Close echoes merged | 22/1/10 | 24/1/8 | 19/1/13 |
| Close echoes missing | 13/0/19 | 18/0/14 | 14/0/18 |

Under DROP_CLOSE, birch form transfer changes REPLAY4TP to ENRICH2TP across
eight frames; far pipe/ladder/pouch changes4TP to2TP across12 frames. The12
rod/context controls stay10TP in both arms. All three groups have zero false
positives in this condition. On the12 fit frames, both new arms have4TP/0FP/8FN,
versus the original3TP/0FP/9FN. All44 unique rich/context frames are reported;
the four original supported rod states are not duplicated.

Legacy costs also prevent automatic replacement. Relative to fixed MZ37,
ENRICH gains11TP but adds10FP on the2000 relation-DEV frames under DROP_CLOSE;
the1000 distance-DEV frames gain16TP with no net added FP. MZ36's missing-return
counts do not improve. Its IDEAL condition loses oneTP and adds oneFP. The
full result includes every condition and paired gains/losses, not just totals.
All400 MZ36 attempts remain visible:380 admitted frames and80 UNKNOWN bits
from20 excluded frames; UNKNOWN is not scored as a correct negative.

The [applied reading](LOCAL_EVIDENCE_LEARNING_20260911.md) motivates a spatial
intrusion head whose candidates can survive absent ToF echoes, supervised by
native angular-cell events during training. This is an untested proposal.
The larger MZ48 source is excluded from this comparison; its separate
collection does not retrospectively validate this fit.

CPU diagnosis finds that equal event-label exposure did not equalize native
witness exposure: the1200 extra positive-event presentations contain991
surviving witnesses in REPLAY and500 in ENRICH. Twelve repeated frames also
differ from REPLAY's992 unique extra frames. This comparison tests the specified
data replacement, not shape diversity alone. Seven of the12 fit-positive
events have no remaining local candidate; only five have native witnesses,
and the bank removes the witness for one of those. Four is the number supported
by that retained native evidence, not an upper bound on unrestricted guessing.

The four additional non-fit DROP_CLOSE misses are far events. Their raw scores
drop enough to fail even under the existing REPLAY cutoffs; the loss cannot
be assigned entirely to raised calibration thresholds. Separately, all four
ENRICH calibration maxima come from samples with no banked candidate. The
independent [MZ49 comparison](MZ49_BANK_DOMAIN_CALIBRATION_RESULTS_20260911.md)
tests that domain mismatch without extending or relabeling this experiment.
The3.74s CPU-only diagnostic, its exact event IDs and supervision counts are
under `diagnostic-v1/{REPORT.md,summary.json,result.json,receipt.json}`.

## Verification and cost

Native contributor reconstruction matches the44 rich packets within1e-6m,
with exact validity and source IDs. Frozen rich comparators preserve all
decision signs under all three conditions; maximum raw-score discrepancy is
8.59e-6 from batching/floating arithmetic. The380-frame MZ36 IDEAL replay also
preserves all checked decisions. This is numerical agreement, not a claim of
bitwise equality. Initial checkpoint states are exact in both training arms.

Scalar scoring independently recounts1,008,672 known decision bits, recomputes
both calibration cutoffs and checks TRAIN membership and the12 fitted new IDs.
No additional fit or threshold search was used. Training retains MZ20's CUDA
grid-sample backward, whose atomic reductions are not bitwise reproducible;
these are two single fits, not a multi-seed significance result.

Preparation, two fits and fixed inference took47.26s on CUDA; fitting took
5.25s and3.57s. Separate CPU scoring took0.42s. These are internal execution
timings, not device latency. The existing2,320,539,776-byte high-detail mmap
was reused. New features for424 images stayed in memory; no permanent dense
cache was created. Predictions occupy3,933,575bytes, each checkpoint134,837
bytes, and44-frame local labels10,234bytes. Original datasets remain intact.

Evidence: `artifacts.local/work/mz47-local-enrichment-20260911/run-v1/`
contains input bindings, schedules, initial-source identity, both fits,
calibration cutoffs, numerical parity and predictions. `score-v1/` contains
the complete metrics, paired changes, scalar audit and receipt. All inputs
and executed source files are SHA-bound. The runtime process has ended and
the primary GPU is available for the separately authorized collection.
