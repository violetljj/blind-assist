# Development log

This file records only current milestones. Full earlier history is preserved at
`archive/pre-agent-surface-2026-08-26` and searchable through
`experiments/index.jsonl`.

## 2026-08-26 — agent surface reset

- Reduced the root agent map to stable policy and routed dynamic state through
  `docs/PROJECT_STATE.md` and `docs/CURRENT_DECISION.md`.
- Preserved the complete pre-cleanup tree with an annotated remote tag.
- Kept one active research route: `research/active/grail-r1cl/`.
- Added isolated workstation profiles and a Codex desktop environment.
- Externalized the generated full dataset ledger; retained a compact summary,
  hashes, row counts, and a reproducible generator.
- Removed closed runners, contracts, schemas, snapshots, and reports from the
  current branch. Their experimental terminals remain historically true.
- Verified the configured R1CL runtime with a two-sample DINOv2 CUDA
  forward/loss/backward smoke. This is mechanics evidence only.
