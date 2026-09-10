# MZ13 training coverage: weak sign correction and near-recall regression

One expanded-coverage fit does not solve the dominant sign attribution problem.
Added sign FP on the unchanged3000-frame replay falls17->16, while all15 erroneous
BODY sign additions remain. Old DEV loses six near TP bits previously recovered
by MZ11. The predeclared retention criterion fails. Stop this fit, retain MZ5 and
the MZ9 component, and record the completed coverage intervention as a scoped
negative control. No new acquisition, cutoff rescue or successor fit occurred.

## Controlled change

[Protocol](MZ13_TRAINING_COVERAGE_PROTOCOL_20260910.md) retains the MZ11 architecture,
four input summaries,20 parameters, zero initialization, seed111,600 steps and
optimizer. Only the training pool/mix expands: existing original relation
TRAIN5000 and distance train2500 join old TRAIN2500 and consumed MZ6 clean200.
All families enter by original partition, not by the17 observed error identities.
The MZ12 relation DEV2000 and distance DEV1000 remain excluded from gradients.
The7500 added training frames share neither site identifiers nor RGB identities
with those3000 replay frames. The upstream encoder has already trained on relation
TRAIN; this is not an unseen-data claim.

Frozen MZ5, SOURCE and ContextEvidence weights/thresholds remain unchanged.
Old DEV1000 alone selects gate cutoffs with the original zero-added-FP procedure.
The3000 replay, old DEV and sequence have all been consumed previously; they are
Development evidence even though replay labels do not train gate weights.
Balancing nonempty cohort/query/class groups changes the original cohorts'
relative weights when new cohorts enter; this is not a fixed-presentation causal
isolation of sample coverage alone.

## Effective training exposure

Only MZ5-negative, supported SOURCE-positive bits train this gate. The effective
eligible bit counts are much smaller than the10200 source-frame pool:

| Training cohort | Frames | Positive BN/BF/HN/HF | Negative BN/BF/HN/HF |
|---|---:|---|---|
| old TRAIN | 2500 | 0/12/2/13 | 7/21/20/2 |
| MZ6 clean | 200 | 7/29/0/38 | 0/0/0/0 |
| relation TRAIN | 5000 | 3/25/6/18 | 8/30/32/8 |
| distance train | 2500 | 0/0/11/46 | 25/29/0/0 |

Thus the7500 newly used frames contribute109 positive and132 negative eligible
bits. These are correlated output opportunities, not241 independent obstacles.
No missing class is manufactured; empty class groups are excluded from the loss.

## Unchanged evaluation, MZ11 versus MZ13

| Cohort | Exact MZ11 -> MZ13 | TP MZ11 -> MZ13 (BN/BF/HN/HF) | FP MZ11 -> MZ13 |
|---|---:|---|---|
| old DEV1000 | 928 ->926 | 195/190/190/172 ->194/190/185/173 | 6/11/10/7 unchanged |
| relation DEV2000 | 1905 ->1907 | 392/375/395/387 ->391/375/397/388 | 6/7/27/16 ->6/7/27/15 |
| distance DEV1000 | 929 ->931 | 0/0/498/473 ->0/0/498/476 | 6/3/25/9 unchanged |
| MZ6 clean200 | 174 ->175 | 9/35/9/36 ->9/35/9/37 | 5/3/19/0 unchanged |
| MZ6 stress200 | 172 ->174 | 9/34/9/35 ->9/34/9/37 | 5/3/19/0 unchanged |

All original MZ5 positive bits remain unchanged by construction. Relative to
MZ11, old DEV gains one HEAD_FAR TP but loses one BODY_NEAR and five HEAD_NEAR
TPs. Its added-near count versus MZ5 falls15->9, failing the retention check.
The3000 replay gains four far TP bits, so far recovery versus MZ5 rises57->61,
but loses one relation BODY_NEAR TP. No compensating cutoff change was made.

Relation complete-group correctness improves334->336/400 and complete sites
55->56/100; distance complete pairs433->435/500 and sites57->58/100. These modest
gains do not compensate for the failed old-near retention criterion.

Sign-added FP falls17->16 by removing one relation HEAD_FAR false activation.
The10 BODY_NEAR and5 BODY_FAR sign additions remain, including all9 distance
BODY errors on HEAD_ONLY endpoints. Total added FP on3000 falls18->17, retaining
the unrelated cabinet HEAD_NEAR error. This is weak correction of the primary
failure despite the expanded negative coverage.

Thin recovery improves46->47/49 clean and44->46/49 stress. Correct correspondence
still matters: wrong RGB gives1/49 clean and0/49 stress. These are training
regression on one thin-pole layout, and remain below the prior48/49 admission
requirement. Existing premature near warnings are preserved, not repaired.

## Decision and limits

Seven of eight comparison checks pass; old DEV near retention fails. Do not
promote this gate or reinterpret the failure as a need for more total frames.
The experiment weakens the specific hypothesis that this fixed four-summary
linear readout becomes a reliable sign arbiter merely through wider existing
training coverage and the prescribed mix. It does not prove that all existing
representations, nonlinear readouts or supervision schemes are inadequate.

Next work, if separately authorized, should distinguish informative local return
attribution from these coarse summaries rather than automatically adding more
frames or continuing this fit. No further implementation or experiment is
started here. MZ5 and MZ9 inheritance roles remain unchanged.

## Verification and durable evidence

Artifacts: `artifacts.local/work/mz13-training-coverage-20260910/features-v1/`
and `run-v1/`. All7500 selected RGB/native payload hashes were verified before
frozen CUDA extraction (193.06s). Gate fitting plus scoring took8.793s on CUDA;
neither number is online frame latency. The initial shell log-redirection attempt
failed before Python because its parent was absent; the parent was created,
then the extractor ran once successfully. No partial model run was admitted.

Scalar audit verifies59200 output bits across all training/regression/control
cohorts, exact DEV nextafter cutoffs, original-positive preservation, transitions,
family/group/site metrics and thin outcomes. Training/replay site and RGB overlap
are both zero. Independent code review found no gradient access to MZ12 labels
or deviation from the declared fit. No new source, UE, worker or port was started;
processes exited and temporary delivery resources were released. Invalid sensor
support remains missing evidence, not CLEAR; no hardware or safety claim.
