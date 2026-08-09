# Assistive Geometry R2 F1-P TRAIN-only factor learnability protocol lock

## Decision

F1-P is a protocol lock, not a training authorization.

> F0 PASS purchases eligibility to design F1, not eligibility to train F1.

This lock freezes the factor interface, supervision semantics, independent losses, TRAIN-internal roles, checkpoint-selection rule, factor-first success criteria, Kill Gates and F2 admission boundary. It creates no model, trainer, label materializer, checkpoint or optimizer step.

## Frozen factor question

F1 asks whether three factor families are independently learnable:

1. metric depth shape, global metric scale, validity and residual uncertainty;
2. support probability, support-plane normal, camera height, validity and residual uncertainty;
3. obstacle evidence, continuous boundary probability/localization, validity and localization uncertainty.

The learned graph may emit only these declared fields. Clearance, occupancy, free/blocked, risk, task confidence, final state, TTC and future clearance are forbidden as prediction or supervision shortcuts. `GeometryR2Reducer` remains outside the learned graph; no reducer or task metric may select an F1 checkpoint or rescue a failed factor.

## UNKNOWN and uncertainty

Every supervision value carries an explicit validity mask and provenance. Unsupported factor values are `MASKED_UNKNOWN`, never negative. A validity-false target may train only the corresponding validity head; it cannot train the factor value or its uncertainty.

Direct sigma pseudo-labels are not required and must not be invented. Depth, support and boundary uncertainty may be learned only through a frozen heteroscedastic proper score against valid factor residuals on parent-disjoint roles, with a homoscedastic baseline. Zero or constant sigma pseudo-truth and final task state as uncertainty truth are forbidden.

## Current data front door

The governed AG-DCA R0 atlas is bound as capability evidence, not as an F1 roster:

| source capability | frames | parents | current interpretation |
|---|---:|---:|---|
| metric depth | 4,767 | 16 | source exists; canonical R2 label contract not frozen |
| support | 320 | 11 | below the frozen 12-parent joint-role minimum |
| crisp obstacle | 1,557 | 11 | not continuous boundary truth |
| depth uncertainty direct truth | 0 | 0 | residual proper-score contract still required |
| support uncertainty direct truth | 0 | 0 | residual proper-score contract still required |
| continuous boundary truth | 0 | 0 | blocking |
| complete R2 factor-schema truth | 0 | 0 | blocking |

Therefore current F1 execution admission is `FAIL_NOT_AUTHORIZED`.

## Loss and selection contract

Thirteen named component losses remain separately reported. An optimizer objective may eventually use their equal mean only after each component is normalized by a FIT-only nonlearned-baseline scale frozen before initialization. The aggregate loss is never a checkpoint metric.

A future TRAIN roster must assign at least 8 / 2 / 2 parent-disjoint identities to `FIT`, `CHECKPOINT_SELECTION` and `TRAIN_CANARY`, with at least 12 jointly factor-complete parents. Only `CHECKPOINT_SELECTION` may select a checkpoint. Candidates must be non-worse than every frozen factor baseline and improve at least one factor; selection minimizes maximum factor-normalized regret, retains the Pareto table, and breaks ties by earliest step then SHA-256. The candidate schedule is intentionally unresolved, so execution remains blocked.

## Success and Kill Gates

Every factor family must independently improve over its frozen nonlearned baseline on parent-macro TRAIN_CANARY evidence, with a 10,000-parent-bootstrap 95% lower bound above zero and favorable sign on at least 75% of eligible parents. Uncertainty must improve proper score over a homoscedastic baseline and order residual magnitude monotonically across frozen uncertainty quantiles.

Any schema shortcut, incomplete supervision, failed factor family, uncalibrated uncertainty, invalid checkpoint selection or downstream rescue attempt stops F1 and denies F2. Final reducer metrics cannot purchase success.

## Unique successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK`

Its execution authority is `false`. It may only freeze a parent-complete TRAIN supervision source, continuous-boundary label transform, residual-based uncertainty scoring contract, provenance receipts and role roster. It may not define a model, create a trainer, materialize labels or run an optimizer without a separate authorization.
