# Documentation governance

Keep the current operating surface small enough for a developer or agent to load
without scanning history.

## Authority order

1. `AGENTS.md` defines stable repository policy.
2. `docs/PROJECT_STATE.md` routes current work.
3. `docs/CURRENT_DECISION.md` owns the single active research decision.
4. `research/active/<route>/README.md` owns runnable route details.
5. Formal rules apply only to protected final claims or external actions.

Dynamic results, route names, and successor decisions do not belong in
`AGENTS.md`. Closed work is indexed in `experiments/index.jsonl` and preserved
by the archive tag documented in [history-index.md](history-index.md).

## Budgets

- one active directory under `research/active/`;
- `AGENTS.md` at most 150 lines and 10 KiB;
- `DEVELOPMENT_LOG.md` at most 200 lines and 100 KiB;
- no generated dataset ledger, checkpoint, model, APK, or raw result in tracked
  source;
- no machine-specific absolute path in hot documentation or active route files.

Run `scripts/check_project_structure.ps1` after layout changes and
`scripts/check_docs_index.ps1` after editing hot navigation.
