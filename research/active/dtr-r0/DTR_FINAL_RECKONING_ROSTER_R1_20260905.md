# DTR Final Reckoning Roster R1

Status: `DESIGN_AND_ARMS_FROZEN / SOURCE_MATERIALIZATION_NOT_YET_AUTHORIZED`

Date: 2026-09-05

## Decision

Stop attempting to reconstruct a fair raw-input comparison on the deleted
eleven-cohort payload.  Build one new retained raw roster whose purpose is to
adjudicate the paper identity, not to develop X97 or rescue X94.

The adjudication question is now:

> What collision-state information does X94 contribute independently of
> temporal event smoothing?

The C35 pilot motivates this decomposition but does not answer it.  X94 and the
raw Kalman plus 0.60 s emitter tied on Event F1 and false segments, while X94
had better frame F1, lead, and CLEAR and the simple emitter fragmented less.

## Frozen roster

R1 contains ten mechanism-discriminating strata under three source groups:

- one `FIT_ONLY` group for the tiny learned predictor and X95;
- two untouched `FINAL_ADJUDICATION` groups;
- 10 episodes per group, 30 total.

The strata are clean constant motion, one-frame dropout, 2/3/6-frame dropout,
lateral crossing, receding near-miss, same-class association instability,
curved wearer route, static-object ego rotation, partial-visibility surface
fragmentation, and full disappearance/reappearance with a known-negative gap.

These are controlled mechanism tests.  Controlled dropout is not evidence of
natural detector-dropout prevalence.

## Frozen arms

The eleven arms are radial TTC, current-only finite-difference CV, Kalman CV,
Kalman CV plus 0.60 s hysteresis, causal-history CTRV, one fixed tiny logistic
reference, X24, X73, X94, X94 plus the same simple 0.60 s emitter, and X95.

`X94 + simple emitter` is a first-class arm.  It directly tests whether X94's
framewise evidence quality and a separate temporal organizer compose better
than either an end-to-end X94 alert or a simple Kalman pipeline.

All existing mechanisms and new classic modules are byte-locked by the
machine-readable protocol.  CTRV yaw rate is estimated only from causal target
track history; source-native dynamics remain evaluator-only.  The tiny learned
arm has eight fixed features, fixed optimization, and a 0.50 threshold.

## Truth-opening order

1. Materialize and source-gate all three raw groups.
2. Seal one shared raw/intervened detector ledger.
3. Seal every nonlearned prediction on fit and final groups.
4. Open only `FIT_ONLY` truth; fit the tiny arm and X95 once.
5. Seal learned predictions on both final groups.
6. Open final truth once; run one shared maximum one-to-one event matcher.
7. Accept the terminal result without X97.

Final metrics are Event Precision/Recall/F1, false segments and rate, median and
p10 lead, fragmentation count/rate, CLEAR delay/censoring, and secondary frame
Precision/Recall/F1.  Every metric is also reported by stratum.  Uncertainty is
a fixed 10,000-replicate paired episode-cluster bootstrap.

## Retention

The canonical source root is
`artifacts.local/evidence/dtr-final-reckoning-roster-r1`.  Raw RGB, depth,
instance segmentation, witness frames, joined model/evaluator roots, raw and
intervened detector ledgers, evaluator-only object identities/dynamics, route
receipts, manifests, predictions, scores, and bootstrap receipts all carry the
`sealed_final` authority token.

They may not be deleted or compacted without explicit new user authority and a
hash-verified recovery plan.  Hardlink deduplication is allowed only when every
logical path and hash remains present.  C35 scaling estimates the full capture
at about 22.24 GiB; the protocol reserves 48 GiB and requires 80 GiB free.
Current F: free space was 357.92 GiB at design time.

## Current gate

The roster and all eleven arm specifications/implementations are frozen.  The
only remaining pre-capture blocker is the source-cell materializer plus
source-only geometry and visibility gates.  Formal materialization is still
unauthorized: attaching the ten labels to recycled C44 scenes would not satisfy
the protocol.

Protocol:
`research/active/dtr-r0/carla/dtr_final_reckoning_roster_protocol.json`

The validator must pass immediately before any probe or formal source run.  A
formal cell that misses its frozen source semantics becomes `NOT_EVALUABLE`;
it cannot be repaired after durable pixels exist.
