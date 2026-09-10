# MZ10 frozen-output error diagnosis

This is a post-hoc audit attached to the existing rejected MZ10 terminal, not a
new trained candidate or reopened admission run. Inputs are its frozen run-v2
predictions, the admitted MZ8 observation cache, and original MZ5 heads. No EVAL,
new capture, model fit, deployed threshold change, or temporal work occurred.
All 1000 old DEV frames and 200 sequence frames are consumed Development;
the sequences also participated in MZ9 training. Their adjacent frames are not
independent obstacle configurations.

Question: can existing scores and candidate extent distinguish new false
activations from useful recoveries, and can SOURCE score calibration alone meet
the complete composition's false-positive budget? Geometry and branch replay
use CUDA; the small descriptive calculations use CPU. Truth only groups and
scores diagnostic records; it does not enter feature extraction or inference.

## Whole-composition budget

Holding the MZ10 availability routing and original fallback fixed, enumerate all
distinct SOURCE score cutoffs. The table reports the best TP count under each
query's original MZ5 FP budget. These are truth-selected descriptive frontier
points on the same DEV set, not deployable calibrated results or a bound on other
models, non-monotone rules, or joint features.

| Query | Original FP budget | Fallback FP | Remaining SOURCE FP | Original TP | Best frontier TP |
|---|---:|---:|---:|---:|---:|
| BODY_NEAR | 6 | 6 | 0 | 187 | 196 |
| BODY_FAR | 11 | 11 | 0 | 180 | 90 |
| HEAD_NEAR | 10 | 9 | 1 | 183 | 183 |
| HEAD_FAR | 7 | 7 | 0 | 165 | 132 |

Thus a SOURCE threshold-only repair cannot retain old far detection at the
same per-query FP budgets under this fixed routing. BODY_NEAR has a narrow
development opportunity; it does not establish a general fusion solution.
The original MZ10 13 lost far TPs remain an additional reason not to let candidate
availability automatically override positive baseline evidence.

## Observable separation and its limits

The following AUC compares only MZ10 gained TPs against newly introduced FPs,
with larger feature values interpreted as stronger positive evidence. It is a
small, outcome-selected diagnostic subgroup, not a classifier evaluation.

| Query | Gained TP / new FP | SOURCE margin AUC | MZ5 margin AUC | Candidate count AUC |
|---|---:|---:|---:|---:|
| BODY_NEAR | 10 / 5 | .900 | 1.000 | .840 |
| BODY_FAR | 14 / 9 | .563 | .992 | .976 |
| HEAD_NEAR | 13 / 9 | .641 | .923 | .897 |
| HEAD_FAR | 13 / 7 | .846 | .857 | .791 |

BODY_FAR newly introduced FPs have candidate counts 7–90 (median 7); recovered
old DEV TPs have 80–451 (median 210.5). Their SOURCE margins overlap strongly:
FP .149–3.143, recovered TP 0–3.734. Candidate extent contains information that
the final score does not capture well on these particular samples. Counts are
angular hypotheses, not independent measured points or actual occupied extent.
They may encode layout and distance correlations; increasing them is not itself
proof of a real obstacle.

MZ5 confidence looks highly discriminative on the old changed-bit subgroup,
but fails an important cross-cohort check. Thin-pole positive opportunities have
MZ5 BODY_FAR margins -6.323 to -4.656 and HEAD_FAR -6.776 to -4.197. These are
strong negative scores, overlapping the old false-activation range. Requiring
the baseline to be near its threshold would suppress intended thin-pole recovery.
The thin pole does have larger candidate counts: BODY_FAR 115–246 and HEAD_FAR
119–238. This is one trained configuration and cannot establish transfer.

Full records retain original FPs, new FPs, recovered/lost TPs, both MZ5 branch
scores, source/baseline margins, candidate counts, distinct zones and returns.
Unavailable SOURCE values are excluded from source-margin distributions rather
than treating the negative sentinel as confidence. There are no newly introduced
sequence FPs relative to MZ5, so sequence subgroup AUC is undefined, not perfect.

## Decision

Keep MZ5 and the MZ9 component dispositions unchanged. Do not launch a SOURCE-only
threshold rescue or infer that a learned arbitrator is already justified. Existing
inputs show partial separation, so this audit also does not prove representation
insufficiency. A subsequent bounded Development candidate could test whether
candidate extent plus local evidence supports selective corrections while scoring
the whole output's FP budget; it must retain the low-confidence thin-pole cases
and compare against equal-exposure MZ5 adaptation. No such candidate is selected
or implemented here. Fresh configuration confirmation and temporal work remain
unperformed.

## Reproduction and verification

Run `mz10_error_diagnostic.py --root E:/linnan/linnan --output <new-artifact-dir>`.
Evidence: `artifacts.local/work/mz10-availability-20260910/error-diagnostic-v2/`
contains summary/frontier JSON, 5600 scalar records, feature arrays, hashed input
receipt and verification. All 5600 geometry availability and branch prediction
signs match frozen MZ10. Independent scalar replay verifies every enumerated
frontier point and all 48 group counts. The first attempt stopped on a float64
ToF versus float32 head mismatch; v1 retains a repair record, and v2 explicitly
casts model inputs while preserving geometry. No partial results were admitted.
All processes exited; only durable diagnostic evidence remains.
