# Documentation governance

Keep the current operating surface small enough for a developer or agent to load
without scanning history.

## Authority order

1. `AGENTS.md` defines stable repository policy.
2. `docs/PROJECT_STATE.md` routes current work.
3. `docs/CURRENT_DECISION.md` owns the active research decisions.
4. `research/active/<route>/CURRENT.md` owns the compact route decision.
5. `research/active/<route>/README.md` owns detailed results and reproduction.
6. Formal freeze/access gates apply only to protected final claims or external
   actions. Evidence identity and applicable inheritance rules also govern research
   reuse; they do not turn ordinary Development into a final evaluation.

Dynamic results, route names, and successor decisions do not belong in
`AGENTS.md`. Closed work is indexed in `experiments/index.jsonl` and preserved
by the archive tag documented in [history-index.md](history-index.md).

## Process value

Keep a research procedure when its concrete decision benefit or named material
risk justifies the effort. When it delays useful work without changing a decision,
simplify, replace, or retire the local procedure in the owning policy. State the
reason in the existing change/result note; do not add a routine meta-review,
new checklist, or approval gate. Preserve evidence identity and protected
blind/final boundaries.

## Budgets

- one active directory per explicitly parallel product line under
  `research/active/`;
- `AGENTS.md` at most 120 lines and 8 KiB;
- `docs/PROJECT_STATE.md` and `docs/CURRENT_DECISION.md` each at most 200 lines
  and 20 KiB;
- each route `CURRENT.md` at most 150 lines and 16 KiB;
- `DEVELOPMENT_LOG.md` at most 200 lines and 100 KiB;
- no generated dataset ledger, checkpoint, model, APK, or raw result in tracked
  source;
- no machine-specific absolute path in hot documentation or active route files.

Run `scripts/check_project_structure.ps1` after layout changes and
`scripts/check_docs_index.ps1` after editing hot navigation. For wording-only
changes, inspect the scoped diff and document budgets; do not trigger a full
knowledge refresh, archive audit, or unrelated route checks.

## Work in progress

Uncommitted files and untracked candidate directories are not authority. Update
the compact current only in the scoped delivery that accepts, closes, or parks
the result. Stage route, documentation, and tooling batches by exact path so
parallel work is neither absorbed nor silently reclassified.
