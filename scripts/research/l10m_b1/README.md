# L10M-B1: Matched Structured Searchability

B1 asks one narrow question: with B0 state representation and semantics frozen,
does exposing the same mutable policy space as named components make a searcher
more likely to find a behaviorally better candidate?

The control arm edits a small source-level policy surface. The treatment arm
edits component-grouped JSON (`progress_contract`, `stuck_response`,
`recovery_transition`, `action_selection`, and `fallback`). The progress
contract is visible but read-only. Both representations compile to the same
finite 162-candidate `PolicySpec` space, start from the same candidate, and are
checked exhaustively for interface equivalence.

The evaluator, hidden synthetic cohort, episode truth, unsafe definition, hard
safety shield, B0 progress/stuck/recovery/terminal semantics, model, prompt
information, seeds, generation/evaluation budget, feedback, and selection rule
are matched or frozen. Search workers must run outside the repository and must
not receive the evaluator, cohort, truth, other-arm state, or hashes.

Freeze and inspect the preregistration with:

```text
python -m scripts.research.l10m_b1.protocol --output artifacts.local/evidence/l10m_b1/protocol.json
```

The current status is `B1_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED`: this command
does not call a model or expose hidden outcomes. It authorizes neither B0-E nor
large-scale Structured Search. A later execution manifest must bind one verified
Codex CLI executable/version/hash and one model before any formal run artifact,
then reuse them identically across paired arms.

The historical provider attempts remain infrastructure-only and do not consume
a B1 seed or support a scientific verdict. Before any first evaluable B1 run,
qualify the transport independently with `L10M-B1-I0-TRANSPORT-QUALIFICATION-V1`:

```text
python -m scripts.research.l10m_b1.transport_qualification --route direct
python -m scripts.research.l10m_b1.transport_qualification --route proxy
```

Each route uses the same Docker image, auth mount, Codex provider invocation,
Responses streaming configuration, isolation guard, and response scale as B1,
but a fixed research-free JSON canary. The frozen gate is 10/10 non-empty,
strictly parseable terminal responses, zero nonzero provider exits, zero
reconnect exhaustion, zero decode failures, and an isolation `PASS`, with no
application retry. A direct-path pass diagnoses the proxy boundary but does not
authorize the current proxy-bound B1. Only a 10/10 proxy result can emit
`B1_TRANSPORT_QUALIFIED` with `b1_execution_authorized=true`; every failure
leaves B1 unauthorized and has `NO_SCIENTIFIC_VERDICT`. The formal Docker
runner additionally requires the exact proxy `result.json` through
`--transport-qualification` and binds its path and SHA-256 into the execution
manifest.

The current scientific protocol is the fresh V2 successor. V1 was sealed
`B1_NOT_EVALUABLE_TRANSPORT_RUNTIME` after partial observations and is never
resumed or compared. V2 uses fresh paired seeds `53/71/89`; no V1 candidate,
feedback, score, or seed is reused. Any provider nonzero exit, empty terminal
caused by provider failure, timeout, or interrupted dispatch immediately seals
the entire V2 cohort `NOT_EVALUABLE`, with no replacement request or resume.
After a complete 48/48 run, `analyze_result.py` applies only the preregistered
verdict rules and writes a create-once result receipt; it refuses incomplete or
provider-failed runs.

Primary analysis is the paired difference in best improvement from the shared
initial candidate. Secondary analysis records discovery hit-rate, best-of-budget
behavioral vector, first-hit generation/evaluation, unsafe and semantic-invalid
candidate rates, changed components, and paired-seed stability. A small isolated
final-score difference is explicitly inconclusive.
