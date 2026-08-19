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

Primary analysis is the paired difference in best improvement from the shared
initial candidate. Secondary analysis records discovery hit-rate, best-of-budget
behavioral vector, first-hit generation/evaluation, unsafe and semantic-invalid
candidate rates, changed components, and paired-seed stability. A small isolated
final-score difference is explicitly inconclusive.
