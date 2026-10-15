# Android research-preview admission contract

Status: **contract implementation; no experiment or thresholds frozen**.
Schema: [admission v1](../tools/contracts/schemas/android_research_preview_admission_v1.schema.json).
Scope: the schema/receipt slice of [#21](https://github.com/violetljj/blind-assist/issues/21).

## Authority boundary

This receipt format describes a separately labelled Android research preview for
DA2 or A2-392. Committing it admits no candidate, reopens no stopped route, and
changes no default App, evaluator, Android runtime, or protected-outcome access.
All fixtures are synthetic. Their numeric thresholds are examples, not approved
research gates. Actual protocol and thresholds require their own prospective
freeze and measured evidence. Issue #21's evaluator and device evidence remain
outstanding after this schema contribution.

The validator checks receipt consistency, not the truth of measurements or the
existence/authority of a Git commit. A consumer must verify `freeze_commit`, the
protocol artifact, roster and frozen-contract digest against the independently
retained pre-outcome artifacts. Replacing a contract and recomputing its hashes
is not a new authorization. A terminal run must be persisted immutably by the
future evaluator; this stateless validator cannot enforce history across files.

## Frozen and observed evidence

The contract records model, export, preprocessing, postprocessing, input-manifest
and state-reset SHA-256 identities, exact input shape, supported backend and
reference-runtime identity. Observed identities must match; requested and
observed backends must both equal the supported backend. Parity uses the same
hash-bound input manifest identified by the receipt. Missing observations are
UNKNOWN; mismatches and actual fallback are FAIL.

`freeze_commit` identifies the independently retained freeze record.
`outcome_opened_at` must be at or after `frozen_at`; the summary boolean is
derived from these timezone-aware timestamps. A missing timestamp is UNKNOWN.
A syntactically valid receipt does not authenticate these timestamps.

Hashes use SHA-256 of UTF-8 JSON with sorted keys, compact separators, ASCII
escapes and no NaN/Infinity (the Python `canonical_sha256` function is normative).
The threshold digest covers `{version, values}`; the roster digest covers
`{version, roles, members}`; `evidence.frozen_contract_sha256` covers the entire
contract. Numeric encodings are significant (`1` and `1.0` differ).
Protocol SHA-256 refers to an external artifact and is not verified by this
receipt-only tool. Roster members bind parent/session IDs to training,
development or evaluation roles. Parents cannot cross roles, and a session ID
cannot belong to multiple parents. Multiple sessions of the same parent in one
role are allowed. Quality and performance records must come from the frozen
evaluation roster; missing expected sessions prevent PASS.

## Quality aggregation

Each metric has its own nonnegative integer opportunity denominator, sum/count,
and derived value. `null` or zero required denominators and null sums produce
UNKNOWN rather than negative evidence. Rate counts must be integer and no larger
than their own denominator. Clearance-error sums are continuous metres.

| Metric | Numerator / sum | Opportunity population | Unit / direction |
| --- | --- | --- | --- |
| false_clear | Incorrect clear predictions | Evaluator-labelled obstructed queries | Proportion, at most threshold |
| false_block | Incorrect blocked predictions | Evaluator-labelled clear queries | Proportion, at most threshold |
| known_coverage | Queries with a known prediction | All eligible queries | Proportion, at least threshold |
| clearance_error | Sum of absolute clearance errors | Queries with evaluable metric clearance | Metres, at most threshold |
| transition_consistency | Consistent eligible transitions | Evaluable adjacent transitions | Proportion, at least threshold |

Every session value is sum/denominator. Pooled values use summed numerators and
summed metric-specific denominators, not an unweighted average of sessions.
All rate values and thresholds are in [0,1]. Pooled quality determines the quality
gate; per-session values remain visible. Missing any required session component
makes its pooled metric null. Non-finite raw values are invalid JSON evidence:
producers must retain the diagnostic externally and emit null in the receipt,
as illustrated by `nonfinite-aggregate.json`, never substitute zero.

## Android performance aggregation

Every evaluation parent/session has one uniquely identified device-run record
with explicit arrays of cold-start milliseconds, warm-start milliseconds,
per-inference latency milliseconds, memory MB and continuous thermal-window
seconds. Array lengths are the sample denominators. Empty arrays are missing
evidence. Session and pooled summaries are recomputed: maximum startup and
memory, nearest-rank latency p50/p95 (`ceil(q*n)`), and minimum thermal window.
Pooled percentiles use concatenated latency samples. Every session and the pooled
summary must meet the frozen Android limits; a large session cannot hide another
session's latency failure. Missing expected runs or samples prevent PASS.

These samples are receipt evidence, not an implementation of a benchmark. The
future evaluator must define warmup, device identity, sampling and thermal
measurement procedures in the hash-bound protocol before outcomes are opened.

## Deterministic decisions and stop conditions

Quality, Android feasibility and product authority are separate. Product authority
is always DENIED, scopes empty, with the exact reason
`RESEARCH_PREVIEW_HAS_NO_PRODUCT_AUTHORITY`.

| Condition | Terminal | Reason |
| --- | --- | --- |
| Missing denominators, observations, runs or incomplete evidence | UNKNOWN | Field-specific MISSING code or INCOMPLETE_EVIDENCE |
| Parent/session role overlap | FAIL | PARENT_SESSION_OVERLAP |
| Pre-freeze outcome access | FAIL | PRE_FREEZE_OUTCOME_ACCESS |
| Frozen/observed identity mismatch | FAIL | Field-specific MISMATCH |
| Backend mismatch/fallback/inconsistent fallback state | FAIL | BACKEND_MISMATCH, BACKEND_FALLBACK or FALLBACK_STATE_INCONSISTENT |
| Parity outside bound | FAIL | REFERENCE_PARITY_FAILURE |
| Quality or Android threshold violation | FAIL | Field-specific THRESHOLD |
| All gates pass | PASS | ALL_RESEARCH_PREVIEW_GATES_PASSED |

FAIL takes precedence over UNKNOWN, then PASS; reasons for the selected terminal
are unique and sorted. Invalid schema, content hashes or inconsistent aggregates
are rejected rather than accepted as valid terminal receipts. Both FAIL and
UNKNOWN authorize NONE. PASS has only ANDROID_RESEARCH_PREVIEW_ONLY scope.
All receipts are terminal with rescue_allowed=false; evidence must not be added
to rescue a previously terminal run.

## Synthetic fixtures and validation

[Ten fixtures](../tools/contracts/fixtures/android-admission-v1/) cover PASS,
missing/zero denominators, non-finite-origin missing evidence, backend mismatch,
fallback, identity mismatch, parity failure, incomplete evidence and simultaneous
FAIL/UNKNOWN precedence. They are examples, not model or device measurements.

```sh
python -m unittest tools.contracts.test_validate_android_research_preview_admission
python tools/contracts/validate_android_research_preview_admission.py tools/contracts/fixtures/android-admission-v1/pass.json
```

Schema validation alone checks the shape and domains. The executable validator
also enforces the cross-field identities, hashes, aggregation and decisions.
