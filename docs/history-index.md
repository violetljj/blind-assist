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

- `experiments/index.jsonl`: one JSON object per indexed current route or
  canonical closed milestone; owning current documents remain the live authority
- `data/dataset-ledger-summary.csv`: compact dataset aggregate
- `data/dataset-ledger-manifest.json`: hashes and row counts for the externalized
  full ledger

The historical tree is audit/reproduction evidence, not current execution
authority. Do not copy a closed runner back into the active tree without a new
decision that identifies the new information source and claim boundary.

## Post-compaction closed routes

Routes completed after the archive tag remain recoverable from their terminal
commits; canonical milestones are indexed in `experiments/index.jsonl`:

- GRAIL R1C-L: `15fddda3a8c58b0287feb04cd20d72ac59934eee`
- unseen-location router: `ebc003eb427187bf6f5d26fce17dca67cc30abd4`
- GRAIL G0/G1 active geometry and multiview appearance: G1 terminal
  `4db9a11964ff9af9b5b500d59a60d8bb6fc0213b`

These commits are evidence and reproduction anchors, not current execution
authority. In particular, the consumed GRAIL cohorts and prior dynamic-risk
signals must not be reopened through tuning or relabeling.
