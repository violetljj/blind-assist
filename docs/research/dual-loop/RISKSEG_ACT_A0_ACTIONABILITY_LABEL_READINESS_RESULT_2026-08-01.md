# RISKSEG-ACT A0 actionability label readiness result

## Decision

`RISKSEG-ACT A0` closes as:

```text
STOP_ACTIONABILITY_REFERENCE_CONSTRUCT_UNSTABLE
```

The fixed-anchor screen failed. Do not start the 1,920-frame A0-P1 annotation,
do not freeze or collect the proposed matched-pair cohort under this ontology,
and do not train an actionability-aware segmentation model. The default App and
YOLO baseline remain unchanged.

This is a valid negative Development result, not a statement that
actionability is impossible to perceive. It says the present RGB review
protocol, fixed image-space corridor grid, intrusion definition, boundary
relation and phase labels did not form a stable reference construct on the
consumed 30-event cohort.

## Execution integrity

The pre-output contract and implementation were committed and pushed before
the review bundle was generated. The bundle then froze:

- 30 consumed parent events from 30 source sessions;
- four content-blind anchors per event, 120 observations per pass;
- `CURRENT_ONLY`, `CAUSAL_HISTORY` and `HINDSIGHT_REFERENCE`;
- two fresh isolated reviewers per condition, six distinct identities;
- 720 completed review items;
- no source identity, bucket, old interval, oracle, truth, model output or
  other review in a reviewer packet;
- no third-Agent adjudication or disagreement repair.

All six submissions passed schema, coverage, isolation-receipt and SHA binding
checks. The independent validator recomputed the condition and cross-condition
metrics and returned `VALID`.

## Gate result

| Metric | CURRENT_ONLY | CAUSAL_HISTORY | HINDSIGHT_REFERENCE |
|---|---:|---:|---:|
| Condition gate | FAIL | FAIL | FAIL |
| Alertable exact | 0.850 | 0.850 | 0.658 |
| Passed exact | 0.942 | 0.925 | 0.867 |
| Knownness exact | 0.958 | 1.000 | 0.933 |
| Boundary-relation exact | 0.173 | 0.241 | 0.241 |
| Intrusion spatial F1 | 0.632 | 0.649 | 0.553 |
| Derived exact | 0.850 | 0.850 | 0.658 |
| Derived macro Jaccard | 0.544 | 0.817 | 0.338 |
| Actionable-union agreement | 0.643 | 0.667 | 0.446 |
| Parent-event sequence match | 0.533 | 0.600 | 0.400 |
| Union abstain burden | 0.050 | 0.008 | 0.067 |
| UNKNOWN→NON_ACTIONABLE violations | 0 | 0 | 0 |

Current and causal views reached the marginal alertable/passed/knownness
agreements, but the variables that would supply the missing supervision did
not pass:

- boundary relation remained extremely unstable;
- spatial intrusion F1 stayed far below the frozen `0.80` gate;
- actionable-union agreement failed;
- parent-event four-anchor sequences failed;
- CURRENT macro Jaccard failed.

CAUSAL improved macro Jaccard by about `+0.273`, but that does not rescue the
route construct: four major gates still failed, and the worst hidden-stratum
drop was `0.0625`, above the allowed `0.05`.

## Why hindsight is decisive

Hindsight was reference-only and had the strictest gate. It failed more
strongly, not less strongly:

- alertable exact `0.658`;
- boundary relation exact `0.241`;
- intrusion F1 `0.553`;
- derived exact `0.658`;
- parent-event sequence match `0.400`.

CAUSAL-to-HINDSIGHT consensus changed on `43.3%` of anchors; exact consensus
agreement was only `68/120 = 0.567`. More visual context therefore did not
produce a stable shared reference. This rules out the interpretation that the
only problem was insufficient current-frame information.

## False-signal checks

Low abstain rates do not imply readiness. Reviewers often made confident but
different spatial or lifecycle judgments. Likewise, overall derived exact
agreement of `0.85` in CURRENT/CAUSAL cannot override failed classwise, spatial
and parent-event gates.

The protocol did succeed at one safety-relevant bookkeeping property:
`UNKNOWN→NON_ACTIONABLE` violations were zero in all conditions. That proves
the abstention firewall worked, not that the target was learnable.

True alert onset remains independently `NOT_EVALUABLE`: 14 of 16 positive
events are left-truncated at the old alertable interval. No result here can
establish warning lead time or training readiness.

## Stop rule

Per the frozen A0 progression:

```text
A0-P0 fixed-anchor screen: FAIL
A0-P1 full-frame annotation: NOT AUTHORIZED
new matched-pair cohort contract: NOT AUTHORIZED
model training: NOT AUTHORIZED
default App: UNCHANGED
```

Do not repair this result by changing the prompt, adding a third reviewer,
turning UNKNOWN into a negative, retuning the grid, adding a special
parallel-curb rule, returning to four-class P1, or training a larger model.

## Evidence

- Frozen contract:
  `RISKSEG_ACT_A0_ACTIONABILITY_LABEL_READINESS_CONTRACT_2026-08-01.json`
  (`a8c66a...6326`)
- Bundle receipt: `7fac856c...c9b2`
- Scoring key: `88d63e42...7ed`
- Score: `0658c2c6...b3e3`
- Independent validation: `VALID`, `a014be25...9dd9`

The evidence root is ignored and local:
`artifacts.local/evidence/riskseg-act-a0/actionability-readiness-v1`.
