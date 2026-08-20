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
