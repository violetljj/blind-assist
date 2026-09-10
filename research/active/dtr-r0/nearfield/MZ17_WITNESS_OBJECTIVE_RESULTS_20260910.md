# MZ17: loss conflict exists, concentrated in BODY; candidate fit not launched

2026-09-10 · EXPLORE · consumed TRAIN diagnostic · no model update.

The proposed optimization conflict is present:55 known incorrect spatial winners
receive a positive-query reward that outweighs their local negative penalty.
It is more concentrated in BODY than HEAD. The frozen aggregate fit-admission
rule is not met, so the prepared witness-objective fit was not launched. This
is not an evaluation or failure of the new loss, and not a new obstacle result.
MZ5 baseline and the bounded MZ9 component remain unchanged.

## What was actually inspected

The final MZ16 HIGH_DETAIL checkpoint was evaluated on the first64 saved MZ15
training batches:1024 frame draws,873 unique frames,4096 frame/query output rows.
The batches include the already disclosed clean diagnostic sequence used in
training. No DEV inference, collection, EVAL, fitting or optimizer step occurred.
Existing cached labels are diagnostic authority, not inference inputs.

Among779 full-query positive outputs admitted by the original supervision mask,
55 have a known incorrect maximum,67 have an unknown maximum, and657 have a
known correct maximum. There are no ties in these779 outputs. Draws repeat:
deduplication by frame/query leaves686 supervised positives and41 known incorrect
winners. These are neither independent obstacle events nor a random population
sample; the main counts preserve the original batch weighting of the diagnostic.

| Query | Supervised positive draws | Known incorrect winners | Fraction | Unknown winners | Net upward wrong winners |
|---|---:|---:|---:|---:|---:|
| BODY_NEAR | 101 | 21 | 20.79% | 8 | 21/21 |
| BODY_FAR | 153 | 22 | 14.38% | 15 | 22/22 |
| HEAD_NEAR | 241 | 5 | 2.07% | 13 | 5/5 |
| HEAD_FAR | 284 | 7 | 2.46% | 31 | 7/7 |
| All | 779 | 55 | 7.06% | 67 | 55/55 |

The difference by query matters. Pooling into7.06% must not hide the BODY rates,
nor justify saying this mechanism is unimportant. Conversely, the67 unknown
winners cannot be added to known errors to inflate the admission numerator.

## Gradient meaning and limits

Inference takes the maximum over geometrically eligible candidate logits.
The original positive query BCE rewards that maximum regardless of whether it
belongs to an actual query contributor. The local BCE separately penalizes
known incorrect candidates, averaged over the local negatives.

For all55 known incorrect positive winners, the weighted query derivative is
negative (would raise the score under independent logit descent), the local
derivative is positive (would lower it), and their sum is negative. The concern
is therefore observed in the actual final checkpoint and batches, not only a
symbolic possibility. All individual winner derivatives are retained in rows.json.

This differentiates losses with respect to candidate logits. It does not measure
a shared-network parameter update, Adam state, the intervening training trajectory,
or the cause of the16 MZ16 placement false positives. The diagnostic uses positive
TRAIN queries, not those placement negatives. A relationship to generalization
remains a hypothesis. Final checkpoint weights/buffers and checkpoint bytes stayed
identical; there was no optimizer or backward accumulation into model parameters.

## Frozen decision and prepared implementation

The [protocol](MZ17_WITNESS_OBJECTIVE_PROTOCOL_20260910.md), written before the
diagnostic result, required at least10% known incorrect winners among supervised
positive outputs and at least50% net upward pressure among those errors before
launching the one fit. The result is7.06% and100%, so the conjunction is false.
The10% threshold is an agent-chosen experiment-budget heuristic, not a statistical
significance boundary or proof that the loss intervention would fail. The
aggregate choice may underweight the observed BODY-specific problem. We preserve
the declared stop and do not retrospectively change its denominator or threshold.

The implemented training-only witness objective rewards the highest known valid
eligible true contributor and penalizes the highest known valid eligible false
candidate. Unknown positions have no new loss, and multiple true contributors
can coexist. It replaces only the original weighted query-loss term; original
local BCE and inference stay unchanged. Three focused tests pass:

- On an adversarial known-wrong maximum, the combined new loss suppresses it
  while rewarding a true witness; the original combined loss raises the error.
- Unknown/ineligible candidates receive no new gradient; all-unknown batches
  produce finite zero loss and zero gradient.
- Multiple true witnesses are not mislabelled as competing false candidates.

`mz17_witness_loss.py`, `mz17_train.py` and `mz17_audit.py` are prepared but the
trainer and candidate-result auditor were NOT run. Trainer admission is enforced
before creating a run directory. Their syntax was checked; passing the loss
unit tests does not establish real model performance. No candidate checkpoint,
new cutoff or new Development score exists. The existing MZ16 HIGH_DETAIL
checkpoint is an unchanged comparator, not a promoted baseline.

## Verification and inheritance

CUDA RTX5060 Laptop,1.79s probe loop,6.31s tool-recorded total; shell process wall
time includes interpreter startup. The probe also checks exact tied-max gradient
sharing and UNKNOWN local-gradient exclusion on an analytical case.

The independent scalar audit verifies4096 rows, batch identity, file/code hashes,
gradient sums/signs, per-query counts, unique-frame/query counts and the false
admission decision. Model parameters/buffers and disk checkpoint are verified
unchanged. A first test command used the wrong working directory and could not
import the test module; rerunning from nearfield passes3/3. No model execution
or training occurred in that failed import.

Retain the diagnostic as COMPONENT_OR_CHALLENGER in COMPONENT mode: gradient
inspection is reusable, and the observed BODY concentration is a scoped finding.
The untrained witness objective has no measured algorithm status. Do not inherit
the result as a failed loss, a proof of absent feature information, or an excuse
to recategorize unknown sources as negative. A future training proposal needs a
new explicit scope acknowledging the BODY-specific counts and unknown coverage;
no successor is started in this run.

Artifacts: `artifacts.local/work/mz17-witness-objective-20260910/probe-v1/`
contains admission.json, result.json,4096-row rows.json, batches.npy, start.json,
receipt.json and independent audit.json. Delivery receipts and resource-release
checks are in the sibling delivery directory. There is no run-v1 directory.
Source code, tests, the unused guarded trainer and protocol are retained for audit.
