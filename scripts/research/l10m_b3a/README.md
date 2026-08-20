# L10M-B3-A: Balanced Exploration Causal Test

B3-A is the single causal follow-up authorized by B3-I0. It compares the
frozen B1 Structured proposal mechanism with the same mechanism plus an
outcome-blind canonical move coverage operator. It does not test retention and
does not put the known seed-89 target value or target hash into the treatment.

The paired fresh seeds are `1768`, `7368`, and `1872`. They are derived from the
protocol ID by SHA-256 before execution and exclude all consumed B1 V1/V2 seeds.
Both arms receive eight model calls per seed through the same frozen Docker
Codex provider. Evaluator, hidden cohort, model, interface, incumbent, strict
selection, budget, and provider settings are shared.

As in B1, these are outcome-blind paired prompt/session identities. The provider
does not expose a deterministic sampling-seed control, so they must not be
described as bitwise-reproducible model RNG seeds.

The treatment records canonical one-step moves as parameter, incumbent
from-value, adjacent direction or categorical destination, and to-value. It
admits an untried direction proposed by the model; when the model repeats or
omits covered directions while a legal untried direction remains, an
outcome-blind seed/generation hash projects the proposal to one untried legal
move. The rank has no score, evaluator output, target value, or target hash.

After the implementation is committed, freeze the create-once protocol:

```text
python -m scripts.research.l10m_b3a.run_experiment freeze \
  --repo-root . \
  --output artifacts.local/evidence/l10m_b3a/protocol.json
```

Then run exactly one paired cohort:

```text
python -m scripts.research.l10m_b3a.run_experiment run \
  --repo-root . \
  --output-root artifacts.local/evidence/l10m_b3a/runs \
  --protocol artifacts.local/evidence/l10m_b3a/protocol.json \
  --transport-qualification F:/ba-data/blindassist-artifacts-20260805/evidence/l10m_b1/transport_qualification/b1-i0-proxy-20260820T025833-4e438512/result.json
```

Read progress without mutating the run:

```text
python -m scripts.research.l10m_b3a.summarize_progress --run-dir <run-dir>
```

After a complete 48/48 execution, create the result once:

```text
python -m scripts.research.l10m_b3a.analyze_result \
  --repo-root . \
  --run-dir <run-dir> \
  --protocol artifacts.local/evidence/l10m_b3a/protocol.json \
  --output <run-dir>/result.json
```

The run is admitted only if Balanced Exploration has a strictly higher fresh
discovery rate, wins at least one paired best-final score and loses none, does
not increase unsafe candidates, and passes operator-integrity checks. Increased
direction diversity without paired reach/final-score value is explicitly not
admission. Any provider or execution-integrity failure seals the whole cohort
`NOT_EVALUABLE` with no retry, replacement, or resume.

The maximum claim is causal search-value evidence for this operator inside the
frozen finite synthetic Structured interface. It is not general model,
end-to-end, device, user, safety-effect, or production evidence.

## Terminal

`B3A_EVALUABLE_COMPLETE / B3A_BALANCED_EXPLORATION_NOT_ADMITTED`

The one authorized fresh cohort completed 48/48 model calls with no provider,
semantic-validity, unsafe, or operator-integrity failure. Control and Balanced
Exploration both reached the improvement threshold in all three paired seeds,
and every final best score was `0.993103448275862`. Paired final-score outcomes
were therefore zero Balanced wins, zero losses, and three ties. The
preregistered admission rule did not pass.

The intervention worked mechanically but did not add the required search
value. Balanced Exploration covered eight unique canonical moves in every seed,
versus three in every Control seed, and reached the exact diagnostic target at
generations `2/2/1`; Control reached it at generations `4/4/3`. This is a
consistent fresh efficiency signal, but earlier discovery alone was not an
admission criterion. Higher diversity without higher reach or final score is
explicitly classified as `diversity_without_search_value`.

This closes the specific balanced-coverage repair without promotion into the
searcher. It does not erase the B3-I0 seed-89 proposal-collapse diagnosis; it
shows that the fresh Controls did not reproduce a reach/final-score deficit
that this operator could repair. No further seed-89 use or post-hoc rescue is
authorized.

Bound evidence:

- run: `b3a-20260820T124003-69a8df8a`
- protocol SHA-256: `a6ab7a39855862a86bca15f0e00a5b4936c3d19b4fdff2423a72317903fd3a27`
- event ledger SHA-256: `050a536080ea7856ed38fd6e711874d9652ee4f2e9855694b3c63bec194e98e5`
- execution manifest SHA-256: `7bf8f69479e8646683945df9376bae62eb28c7c97707563563f7669c64161a05`
- result SHA-256: `bfa265c677e2ff733456ec4c873ba9573ee6b425ffd24001d670a3b785fbeb1b`
