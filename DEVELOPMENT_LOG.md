# Development log

This file records only current milestones. Full earlier history is preserved at
`archive/pre-agent-surface-2026-08-26` and searchable through
`experiments/index.jsonl`.

## 2026-08-27 — dynamic travel risk mainline

- Preserved GRAIL R1C-L/G0/G1 and the unseen-location router at their terminal
  commits and removed their closed runners from the tracked active surface.
- Promoted `research/active/dtr-r0/` as the sole tracked algorithm route.
- Kept the first DTR-R0 question narrow: shared lifecycle, three credible
  baselines, and one route-tube future-occupancy change.
- Implemented the shared 0.50-second clear grace before real-input capture and
  froze route intersection versus radial TTC as the primary comparison.
- Added a truth-blind RGB, causal-track, flat-ground, and pose observation
  materializer for the 24-event input canary.
- The dependency-free synthetic run is mechanics evidence only. A 24-event
  real-input materialization canary and then an exactly 120-event staged RGB
  Development cohort are still required before any scientific gate or claim.

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
