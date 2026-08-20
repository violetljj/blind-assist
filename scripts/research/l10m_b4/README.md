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

## B4 harder fresh cohort qualification

The B4 harder cohort contains three independent evaluator instances named
`amber`, `cobalt`, and `jade`. They retain the frozen B1 state, safety, score,
finite policy space, and strict-selection mechanics. Only the hidden synthetic
episode distribution changes. Preferred turn, fallback, quality-interaction,
and recovery directions differ across instances, so the cohort is not built
around one known target candidate or around Balanced Exploration's prior path.

Before any B4-A model call, the finite landscape is exhaustively enumerated.
Every instance must have at least three improving and three non-improving legal
initial moves, forbid a one-step ceiling hit, contain a quality/fallback
interaction, and expose a strictly improving path of at least two steps to a
global optimum. This is benchmark qualification, not a search-arm result.

After the implementation commit, create the frozen certificate once:

```text
python -m scripts.research.l10m_b4.certify_hard_benchmark \
  --repo-root . \
  --benchmark scripts/research/l10m_b4/hard_benchmark_v1.json \
  --output artifacts.local/evidence/l10m_b4/hard_benchmark_v1/certificate.json
```

No B4-A runner is authorized until this certificate is terminal and its exact
path and SHA-256 are bound into a separate paired comparison protocol. That
future protocol must keep eight generations and use fresh outcome-blind
prompt/session identities.

### Harder-cohort terminal

`B4_HARD_BENCHMARK_QUALIFIED`

All three instances contain 17 episodes and exhaustively cover the same 162
finite PolicySpecs. Each has four improving and four non-improving legal moves
from the initial candidate, six local maxima, no one-step global-optimum hit,
and a shortest strictly improving path of five moves. `amber` and `cobalt`
have one global optimum each; `jade` has two tied global optima. The latter is
intentional threshold symmetry, not an arm outcome.

Bound evidence:

- source commit: `462b47d89517c9afac453d2e67b5ebff79bdecae`
- certificate: `artifacts.local/evidence/l10m_b4/hard_benchmark_v1/certificate.json`
- certificate SHA-256: `7f2cf3a1fb4db8534e5af3839c264dc377be48db63538d7c85c023aabf3c2696`
- model calls used for construction and qualification: `0`

## B4-A paired comparison

B4-A compares the unchanged B3-A Structured Control and Balanced Exploration
mechanisms on all three qualified harder instances. Each instance receives
three fresh paired prompt/session identities, for nine pairs total. Both arms
retain eight generations, so the frozen budget is 144 model calls.

The primary estimand is paired final normalized progress within each instance:
`(final - initial) / (qualified global optimum - initial)`. Balanced search
value is established only if the median paired delta is positive, Balanced
wins at least six of nine pairs with zero losses, global-optimum reach is not
lower, unsafe count does not increase, and operator integrity passes. Earlier
discovery alone is supporting evidence and cannot satisfy admission.

After committing the implementation, freeze the create-once protocol:

```text
python -m scripts.research.l10m_b4.run_b4a freeze \
  --repo-root . \
  --output artifacts.local/evidence/l10m_b4/b4a/protocol.json
```

Only then may the single formal cohort run:

```text
python -m scripts.research.l10m_b4.run_b4a run \
  --repo-root . \
  --output-root artifacts.local/evidence/l10m_b4/b4a/runs \
  --protocol artifacts.local/evidence/l10m_b4/b4a/protocol.json \
  --transport-qualification F:/ba-data/blindassist-artifacts-20260805/evidence/l10m_b1/transport_qualification/b1-i0-proxy-20260820T025833-4e438512/result.json
```

Any provider, transport, isolation, evaluator, or ledger-integrity failure
seals the entire cohort `B4A_NOT_EVALUABLE_RUNTIME / NO_SCIENTIFIC_VERDICT`.
There is no retry, replacement, or resume.

Read progress without mutating the run:

```text
python -m scripts.research.l10m_b4.summarize_b4a --run-dir <run-dir>
```
