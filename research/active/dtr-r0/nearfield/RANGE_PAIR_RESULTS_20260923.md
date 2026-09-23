# Matched range diagnostic: ordering survives, admission not established

2026-09-23 EXPLORE. Decision: `PUBLIC_RANGE_ADMISSION_NOT_ESTABLISHED`.
None of three frozen public readouts passes the joint crossing and OUTSIDE
criteria. Retain these exact readouts as a negative control for range admission;
retain the measured ordering signal without claiming sensor impossibility.
A, LOCAL, UNKNOWN and App behavior are unchanged.

## Scope and result

Two consumed controlled cohorts contribute 192 recorded pairs, deduplicated to
144 geometry/history pairs across 16 geometry groups. Each cohort has 48
intended crossing pairs and 24 OUTSIDE pairs after deduplication. Near endpoints
are 2.65--2.90 m, far endpoints 3.08--3.26 m. Saved observations and 16 fixed
noise realizations are evaluated; the latter are not new scenes or independent
confirmation. Original simulation identities reproduce all 576 required saved
ToF frames. No capture, fitting or cutoff selection occurs.

Correct crossing requires BOTH endpoints available and `.3 <= near <= 3 < far`.
The frozen gate requires at least 80% correct crossings and at most 10% OUTSIDE
endpoint false alerts in BOTH cohorts on saved and pooled noise observations.

| Readout | Stability saved crossing / outside alerts | Rescue saved crossing / outside alerts | Stability noise crossing / outside alerts | Rescue noise crossing / outside alerts |
| --- | --- | --- | --- | --- |
| Global nearest control | 42/48; 23/48 | 37/48; 25/48 | 656/768; 353/768 | 625/768; 397/768 |
| Corridor-compatible nearest | 39/48; 12/48 | 42/48; 18/48 | 588/768; 201/768 | 676/768; 319/768 |
| Median of three compatible returns | 18/48; 4/48 | 28/48; 7/48 | 329/768; 61/768 | 449/768; 117/768 |

The nearest compatible readout ranks near/far correctly in 762/768 (99.22%)
and 767/768 (99.87%) noise pairs, yet correct crossings are only 76.56% and
88.02%, with OUTSIDE false alerts 26.17% and 41.54%. Median-of-three reduces
false alerts to 7.94% and 15.23% but crossing accuracy falls to 42.84% and
58.46%. All estimates happen to be available in this package; missing-return
handling remains explicit and is tested, never replaced by a distant return.

## Attribution and interpretation

Public full-zone compatibility is not ownership of the measured return. In
saved intended pairs, nearest-compatible selected returns have ANY target
contributor at both endpoints in only 21/48 stability and 24/48 rescue pairs.
Conditional correct crossings are 20/21 and 24/24; without that two-endpoint
witness they are 19/27 and 18/24. The witness comes from evaluator-only native
geometry and cannot be used as a public gate. For median-of-three, any target
contributor among selected zones does not prove the median itself belongs to
the target, the same target is tracked, or the target lies inside the corridor.

Camera motion also changes background distance. Strong ordering alone therefore
does not establish useful target range. These results reject the three fixed
readouts as reliable standalone admission in this scope; they do not rule out
all information in RGB/ToF, all learned decoders or all temporal mechanisms.
They do not measure the end-to-end effect of applying a range gate to A/LOCAL.
No gate, loss, cutoff or learner is rescued on this consumed package.

Causal nearest-compatible history has the expected entry direction in 445/512
and 397/512 pooled-noise endpoint comparisons, and exit direction in 797/1024
and 785/1024. Missing comparisons are respectively 67, 103, 55 and 82, included
in those denominators; no zero deltas occur. History can describe motion but
does not resolve absolute range or return ownership. Whole-image RGB pair L1
medians are 0.04875/0.06298 versus same-pose dwell medians 0.00607/0.00515;
these descriptive differences include perspective/background and are not a
tested RGB range decoder.

## Verification and reproduction

Eight focused tests pass for public footprint, missing values, attribution,
fixed geometry, deduplication and exact 3 m semantics. Main governed execution
succeeds in 28.974 s using CPU `TASK_NOT_GPU_SUITABLE`. Public readouts are saved
and sealed before evaluator labels and target attribution are joined.

Independent audit passes 103,397 assertions, reconstructing 384 saved public
endpoint readouts and their causal deltas, pair/group denominators, summaries
and seals. Native target lineage is not independently regenerated: the audit
checks exported support-count invariants. Its v2 attempt stopped before
scientific execution because the plan lacked catalog registration; unchanged
code/plan succeeds under v3 after registration. Both receipts are retained.

Protocol: `RANGE_PAIR_PROTOCOL_20260923.md`. Implementation:
`range_pair_diagnostic.py`, `range_pair_observations.py`; independent audit:
`range_pair_audit.py`. Use `tools/ba.ps1 run research-ue -RunSpec <saved-spec>`;
replay needs a new identity/output, never overwrite a terminal receipt.

Durable evidence under `artifacts.local/evidence/`:

- `ba-range-pair-20260923`: frozen plan, hashes, RunSpec and console log.
- `ba-range-pair-20260923-run`: public readouts/seals, all pairs and strata,
  result, backend receipt and output seal.
- `ba-range-pair-audit-20260923` and `-run`: independent audit plan and result.

These artifacts remain in the canonical F:-backed tree. No new simulator,
paid allocation or continuously running experiment is required.
