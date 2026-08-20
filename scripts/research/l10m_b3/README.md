# L10M-B3-I0: Canonical Intervention Lineage Autopsy

B3-I0 performs a read-only, zero-model-call autopsy of all 48 completed B1 V2
proposals. It parses Raw and Structured outputs into the shared `PolicySpec`,
canonicalizes the actual mechanism values, and reconstructs the exact
strictly-improving single-incumbent lineage used by the frozen B1 runner.

It does not run search, call the model, re-evaluate candidates, change the
algorithm, or use seed 89 as evidence that a future fix generalizes.

Run the bound autopsy once with:

```text
python -m scripts.research.l10m_b3.lineage_autopsy \
  --repo-root . \
  --b1-run-dir artifacts.local/evidence/l10m_b1/runs/b1-20260820T115002-98733875 \
  --b2-dir artifacts.local/evidence/l10m_b2/seed89_candidate_transplant \
  --output artifacts.local/evidence/l10m_b3/b3_i0_lineage_autopsy/result.json
```

The result binds the B1 ledger, closure, runner, policy-space implementation,
and B2 target receipt by SHA-256. Its claim ceiling is diagnostic localization
inside the consumed B1 V2 traces. Any later mechanism admission must use fresh
seeds or fresh instances.

## Terminal

`B3_I0_EVALUABLE_COMPLETE /
B3_I0_PROPOSAL_EXPLORATION_FAILURE_OBSERVED_SEED89`

All 48 B1 V2 completions canonicalized successfully. The seed-89 Raw arm
proposed the exact B2 target at generations 2, 4, and 8; generation 2 was the
first proposal and was retained as a strict improvement. The Structured arm
never proposed that target. Across its full eight-generation budget, it
proposed `fallback_min_quality: 0.35 -> 0.50` five times (score tied the initial
candidate) and `action_selection_turn_threshold: 0.20 -> 0.30` three times
(score decreased). Its incumbent therefore remained the initial candidate.

The first target-reachability breakpoint is generation 2: Raw proposed and
retained `action_selection_turn_threshold: 0.20 -> 0.10`, while Structured
proposed the fallback tie. This is not an evaluator/ranking loss of the good
candidate. The frozen runner has one incumbent rather than a parent population,
has no dedup gate, evaluated repeated proposals, and completed all eight
Structured generations. The remaining qualification is finite-budget scope:
the target was absent in this consumed eight-generation trace, not proven to
have zero probability under a general Structured proposal distribution.

The result selects the B3-A proposal/exploration experiment class if later
authorized, but authorizes no algorithm fix or new search. Seed 89 remains
diagnostic-only; any admission test must use fresh seeds or fresh instances.

Bound local result receipt SHA-256:
`c839e053f55691d74a6341c2c68ab147e9302cda060559a968e8ed9a7010720f`.
