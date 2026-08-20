# L10M-B4: Search Pressure / Benchmark Escalation

## B4-I0 zero-call saturation audit

B4-I0 asks whether the current finite synthetic L10M search benchmark still
has enough pressure to discriminate final search value. It makes no model
calls and does not alter the evaluator, traces, generation budget, or sealed
B1/B3-A conclusions.

The primary baseline population is every evaluable run of the unmodified
Structured proposal mechanism: the three B1 Structured trajectories and the
three fresh B3-A Structured Control trajectories. All formal B1 and B3-A arms
are summarized separately as context. Transport failures, qualification
canaries, the B2 transplant, and the B3-I0 diagnostic autopsy are excluded
because they are not independent baseline search trajectories.

Run the create-once audit:

```text
python -m scripts.research.l10m_b4.saturation_audit \
  --b1-run artifacts.local/evidence/l10m_b1/runs/b1-20260820T115002-98733875 \
  --b3a-run artifacts.local/evidence/l10m_b3a/runs/b3a-20260820T124003-69a8df8a \
  --output artifacts.local/evidence/l10m_b4/b4_i0_saturation_audit/result.json
```

The rubric is a descriptive governance decision made after the headline B3-A
outcomes were known, not a blind hypothesis test. A saturation classification
is limited to the current instance distribution and the frozen eight-generation
budget. It must not be used to claim general search equivalence.

If saturation is confirmed, the current benchmark becomes
`MECHANISM_DEBUG_BENCHMARK / NOT_SUITABLE_FOR_SEARCH_VALUE_DISCRIMINATION`.
The next scientific comparison requires a harder fresh cohort frozen before
any model call. The generation budget remains eight; any later
time-to-discovery or token-to-discovery study needs its own preregistration.

## B4-I0 terminal

`B4_I0_SATURATION_CONFIRMED`

The unmodified Structured baseline reached the observed ceiling in 5/6 formal
trajectories. All three fresh B3-A Controls reached it by generation 4, with
zero final-score variance. Across every formal B1 and B3-A search arm, 11/12
trajectories reached the same ceiling. Generations 5--8 produced no strict
improvement and zero realized best-score gain in either the baseline population
or the complete formal-search population.

The sole pooled-baseline miss is the already diagnosed B1 seed-89 proposal
collapse. It leaves `0.04137931034482756` theoretical headroom after generation
4 but realizes none of it by generation 8. This is a localized search-path
failure, not evidence that the current instance distribution maintains broad
late-budget pressure.

The current benchmark is therefore classified as
`MECHANISM_DEBUG_BENCHMARK / NOT_SUITABLE_FOR_SEARCH_VALUE_DISCRIMINATION`
under the eight-generation budget. This does not invalidate the B3-A efficiency
signal and does not authorize a post-hoc two-generation comparison.

Bound evidence:

- result: `artifacts.local/evidence/l10m_b4/b4_i0_saturation_audit/result.json`
- result SHA-256: `daa42c0c15d3e1122c1daa507a4868d991bd54d9049a8c19ebec413527c3a429`
