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

### B4-A V2 terminal

`B4A_EVALUABLE_COMPLETE / B4A_BALANCED_SEARCH_VALUE_ESTABLISHED`

The V2 cohort completed all 144 model calls with no provider, isolation,
semantic-validity, unsafe, or operator-integrity failure. Balanced Exploration
won final normalized progress in all nine paired identities, with zero losses
and zero ties. The median paired normalized-progress delta was
`+0.3076923076923076`; mean normalized progress was `0.324786324786325`
for Control and `0.584249084249084` for Balanced.

Balanced covered eight canonical moves in every trajectory and admitted a mean
of `7.89` unique candidates, versus Control's mean `4.11` unique candidates.
This mechanical coverage difference is supporting evidence only; admission
comes from the preregistered 9/9 paired final-progress wins.

Neither arm reached a qualified global optimum in any trajectory (`0/9` vs
`0/9`). The result therefore establishes relative final search value for the
frozen Balanced operator on this qualified finite harder cohort. It does not
establish complete search, general algorithm or model superiority, device or
user value, safety effect, or production readiness.

## B5-A fresh generalization replication

B5-A freezes the B4-A V2 Balanced and Control mechanisms without an algorithm
change and moves to three new outcome-blind identities in the same finite
benchmark family: `obsidian`, `coral`, and `silver`. Their motif weights and
names are frozen in `fresh_benchmark_v1.json` before any B5-A model call.
Exhaustive qualification may inspect the finite evaluator landscape, but it may
not use a B5-A arm outcome. Each admitted landscape must retain the B4 pressure
criteria and require at least five strict improvement steps from the initial
candidate to a global optimum.

This stage asks only whether B4-A's relative final normalized-progress result
replicates on a fresh harder cohort. Global-optimum reach is supporting and must
not be lower for Balanced, but positive global-optimum reach is not required for
replication. Any progress-conditioned proposal or search-state-memory mechanism
belongs to a later B5-B version and is forbidden in B5-A.

The frozen replication retains three prompt identities per instance, eight
generations per arm, and 144 total model calls. Admission requires a positive
median paired normalized-progress delta, at least six wins with zero losses,
non-lower global-optimum reach, non-increased unsafe and semantic-invalid
counts, matched model-call cost, and intact operator semantics. A pass admits
the mechanism only as `ADMITTED_L10M_SEARCH_OPERATOR` within this qualified
finite benchmark family.

After the implementation commit is fixed, freeze and execute exactly once:

```text
python -m scripts.research.l10m_b4.run_b5a freeze \
  --repo-root . \
  --output artifacts.local/evidence/l10m_b5/b5a/protocol.json

python -m scripts.research.l10m_b4.run_b5a run \
  --repo-root . \
  --output-root artifacts.local/evidence/l10m_b5/b5a/runs \
  --protocol artifacts.local/evidence/l10m_b5/b5a/protocol.json \
  --transport-qualification F:/ba-data/blindassist-artifacts-20260805/evidence/l10m_b1/transport_qualification/b1-i0-proxy-20260820T025833-4e438512/result.json
```

The progress reader is read-only:

```text
python -m scripts.research.l10m_b4.summarize_b5a --run-dir <run-dir>
```

Bound evidence:

- source commit: `d32d88c565bd339651ab8acd618ab74261677639`
- protocol SHA-256: `d289af12c4c9726958320f0ba8b807e375f0963793a3dba7a6c6b0f0af2b5e67`
- run: `b4av2-20260820T133016-815ed378`
- event ledger SHA-256: `6f1d4b7b40a7e9c763c7d072e75b42379c970199b3c8434b800ae2db972a3103`
- execution manifest SHA-256: `5696a8fd86f74872a6a1384e0d72e647a31bbefad45c99bf1fc266a9d592dbba`
- result SHA-256: `50102673579283c1ab4552c3827eb98d297e0e5b19c22dfdf28042b2280a1370`

Any provider, transport, isolation, evaluator, or ledger-integrity failure
seals the entire cohort `B4A_NOT_EVALUABLE_RUNTIME / NO_SCIENTIFIC_VERDICT`.
There is no retry, replacement, or resume.

Read progress without mutating the run:

```text
python -m scripts.research.l10m_b4.summarize_b4a --run-dir <run-dir>
```

### B4-A V1 fail-closed terminal

`B4A_NOT_EVALUABLE_RUNTIME / NO_SCIENTIFIC_VERDICT`

V1 failed on its first dispatch before a container or model call began. The
runner passed a relative Windows worker directory to Docker `--mount`; Docker
returned exit 125 and rejected the path. The entire V1 cohort is sealed with
no retry or resume, and all nine V1 paired identities are excluded from any
successor.

Bound V1 evidence:

- run: `b4a-20260820T132702-0a00c0ec`
- closeout SHA-256: `183d409e58fdc7c32cd58f19186775d68cd5c10332685f431a2fb7f17f643c46`
- event ledger SHA-256: `14f8fb13f537b93078f99d9c15db55a272590c14d7d4d1e4e2b21528a02b77e7`
- scientific model calls completed: `0`

### B4-A V2 execution-mechanism successor

V2 changes only worker-path resolution: repo, output, protocol, receipt, and
derived worker paths are resolved absolutely before Docker preflight or any
dispatch. Both arms receive the identical change. Benchmark, evaluator,
search mechanisms, eight-generation budget, estimands, and verdict rules are
unchanged. V2 uses nine newly derived paired identities.

Freeze and run V2 under a separate evidence root:

```text
python -m scripts.research.l10m_b4.run_b4a freeze \
  --repo-root . \
  --output artifacts.local/evidence/l10m_b4/b4a_v2/protocol.json

python -m scripts.research.l10m_b4.run_b4a run \
  --repo-root . \
  --output-root artifacts.local/evidence/l10m_b4/b4a_v2/runs \
  --protocol artifacts.local/evidence/l10m_b4/b4a_v2/protocol.json \
  --transport-qualification F:/ba-data/blindassist-artifacts-20260805/evidence/l10m_b1/transport_qualification/b1-i0-proxy-20260820T025833-4e438512/result.json
```
