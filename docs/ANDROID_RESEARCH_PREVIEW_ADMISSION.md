# Android research-preview admission contract

Status: `draft / review required`
Schema: [`android_research_preview_admission_v1.schema.json`](../schemas/android_research_preview_admission_v1.schema.json)
Scope: issue [#21](https://github.com/violetljj/blind-assist/issues/21)

## Authority boundary

This contract governs whether a frozen DA2 or A2-392 candidate may enter the separately labelled Android **research-preview** flavor. It does not reopen the closed DA2 research route, authorize access to protected outcomes, change the default App, or grant production or safety authority. No candidate is admitted by committing this contract.

The protocol, roster roles, thresholds, candidate/export identity, input
manifest, and supported backend must be frozen by both version and SHA-256
before any new candidate outcome is opened. A terminal run cannot be rescued
by adding later metrics or post-outcome tuning.

## Decision model

The machine-readable receipt separates quality, android_feasibility, and
product_authority. The last remains DENIED under this contract. The top-level
decision is PASS | FAIL | UNKNOWN: UNKNOWN is never a negative result and
never authorizes admission; PASS authorizes only
ANDROID_RESEARCH_PREVIEW_ONLY.

When conditions coexist, precedence is deterministic: **FAIL > UNKNOWN >
PASS**. Within the selected terminal, reason codes are unique and sorted
lexicographically. A contract violation therefore cannot be hidden by
simultaneously incomplete evidence.

## Normative gates and stop conditions

| Condition | Decision | Reason code | Effect |
| --- | --- | --- | --- |
| Required evidence is incomplete | `UNKNOWN` | `INCOMPLETE_EVIDENCE` | Terminal; no admission |
| Parent/session separation is violated | `FAIL` | `PARENT_SESSION_OVERLAP` | Terminal contract violation |
| Outcome opened before contract freeze | `FAIL` | `PRE_FREEZE_OUTCOME_ACCESS` | Terminal protocol violation |
| Candidate, export, preprocessing, postprocessing, or input hash differs | `FAIL` | field-specific identity mismatch | Terminal identity violation with both values retained |
| Observed backend differs or fallback occurs | `FAIL` | `BACKEND_MISMATCH` or `BACKEND_FALLBACK` | Terminal runtime violation |
| Reference-runtime parity exceeds the frozen bound | `FAIL` | `REFERENCE_PARITY_FAILURE` | Terminal parity violation |
| A frozen quality or Android threshold is violated | `FAIL` | Metric-specific frozen code | Terminal threshold violation |
| All quality and Android gates pass | `PASS` | `ALL_RESEARCH_PREVIEW_GATES_PASSED` | Research-preview flavor only |

## Required evidence

The receipt retains both the frozen and observed identities, the requested and
observed backend, any fallback backend, the reference-runtime SHA-256, the
measured parity error, and its frozen bound. Reports contain actual
per-parent/session records with denominators plus a pooled aggregate whose
denominator must equal the session sum.

Quality evidence covers false_clear, false_block, known coverage, clearance
error, and transition consistency. Android evidence covers cold/warm startup,
p50/p95 latency, peak memory, thermal window, requested/observed backend,
fallback identity, and output parity against the hash-bound reference runtime.

A claimed PASS is valid only when every required measurement is finite and
non-null, evidence is complete and denominator-consistent, both subordinate
gates are PASS, identities and backend match, no fallback occurred, parity is
within the frozen bound, and the only authorized scope is
ANDROID_RESEARCH_PREVIEW_ONLY.

## Fixtures

Synthetic fixtures in [`schemas/fixtures/android-admission-v1/`](../schemas/fixtures/android-admission-v1/) demonstrate a scoped pass, insufficient-evidence `UNKNOWN` results, and terminal failures. They are receipt examples, not model or device measurements.

## Implementation boundary

This revision defines only the contract. It adds no evaluator, opens no candidate outcome, runs no Android benchmark, changes no runtime behavior, and has no default-App impact. Evaluator implementation requires a separate review after this contract is accepted and frozen.
