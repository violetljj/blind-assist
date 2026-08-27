# Development log

This file records only current milestones. Full earlier history is preserved at
`archive/pre-agent-surface-2026-08-26` and searchable through
`experiments/index.jsonl`.

## 2026-08-28 — v10.10.0 default app visual promotion

- Promoted the refined Compose home and settings experience into the default
  `com.linnan.blindassist` application rather than a candidate package.
- Bumped the install identity from `versionCode=37 / versionName=10.9.0` to
  `versionCode=38 / versionName=10.10.0` so the new APK upgrades the previous
  default app in place when the signing identity matches.
- Kept camera, detector, risk, feedback, permissions, packaged model, and
  experimental build isolation unchanged; this is a UI and version promotion.
- Built the default debug APK and verified package `com.linnan.blindassist`,
  `versionCode=38`, `versionName=10.10.0`, debug signing metadata, and 16KB
  alignment; SHA256 is
  `0D12E61078246C10946EE8557BCCF8EB8A2DEE18A7EB68E9BAE78CBA0CE58309`.
- Archived the verified default APK as
  `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v10.10.0-debug-20260828-003935.apk`.

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
