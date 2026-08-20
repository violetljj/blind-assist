# L10M B6-0 reachability hypothesis triage

This module freezes a two-phase, zero-experimental-model-call historical
analysis. B5-A/B5-C is hypothesis-generating only. Inventory seals trajectory
eligibility before the analysis command may expose any association or outcome
metric. The only positive authority is permission to run a separate B6-A RPS
development experiment; fresh cohorts and admission remain forbidden.

The strict-path quantity exactly preserves B5-C's finite graph ordering:
legal one-field neighbors, strictly increasing score edges, and termination at
any global optimum. For rank analysis only, unreachable is encoded as zero and
a reachable state as `1/(1 + shortest steps)`.

## Terminal result

The sealed analysis completed with:

`REACHABILITY_HYPOTHESIS_NOT_SUPPORTED`

The minimum-evidence and cross-policy gates passed, but the incremental-
information and cross-cohort gates failed. Pooled LOCO relative MAE improvement
was `-0.3385251442242789`, and the median per-cohort absolute MAE improvement
was `-0.8021951095660658`. Confirmatory cohort partial associations were:

- B1 V2 fresh successor: `-0.7875615306482168`
- B3-A balanced exploration: `-1.0`
- B4-A harder cohort: `-0.18967726633981435`

B1 and B3-A therefore met the frozen strong-reversal condition. The two
evaluable policies were positive when aggregated across their eligible
confirmatory cohorts (`structured_balanced=0.37677301213696784`,
`structured_control=0.32122097900690816`), but pooled or policy-level evidence
cannot rescue a failed cohort gate.

B5-A remained `HYPOTHESIS_GENERATING_ONLY` throughout and did not contribute to
any support gate. Experimental model calls and fresh tasks consumed were both
zero. B6-A, RPS development, fresh-cohort use, and operator admission are not
authorized; the L10M exploration-policy route is closed.
