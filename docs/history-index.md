# Historical research index

The current branch contains only operational code and concise indexes. The full
pre-cleanup research surface is preserved at the remote annotated tag:

`archive/pre-agent-surface-2026-08-26`

To inspect it without disturbing the current checkout:

```powershell
git fetch origin tag archive/pre-agent-surface-2026-08-26
git worktree add ..\blindassist-history archive/pre-agent-surface-2026-08-26
```

Remove that inspection worktree when finished:

```powershell
git worktree remove ..\blindassist-history
```

## Searchable ledgers

- `experiments/index.jsonl`: one JSON object per current or closed experiment
- `data/dataset-ledger-summary.csv`: compact dataset aggregate
- `data/dataset-ledger-manifest.json`: hashes and row counts for the externalized
  full ledger

The historical tree is audit/reproduction evidence, not current execution
authority. Do not copy a closed runner back into the active tree without a new
decision that identifies the new information source and claim boundary.
