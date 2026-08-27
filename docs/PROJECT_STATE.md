# Project state

Updated: 2026-08-27

BlindAssist is a runnable Android showcase research prototype. The default
research policy is effect-first: demonstrate a genuine, visible effect in a
controlled setup, report the setup and metric honestly, and add only checks
that protect interpretation.

## Current operating surface

- Current ten-meter route: [L10-R0 Goal-Lock Copilot](../research/active/l10-r0/README.md)
- Current obstacle/risk route: [Dynamic Travel Risk R0](../research/active/dtr-r0/README.md)
- Current question and stop condition: [CURRENT_DECISION.md](CURRENT_DECISION.md)
- Workstation entrypoint: `tools/ba.ps1`
- Android entrypoint: `scripts/run_android_gradle.ps1`
- Closed experiment lookup: [history-index.md](history-index.md)

## Current evidence

GRAIL owner orientation is a completed negative-result chain rather than the
daily mainline. Its latest terminal remains
`STOP_G1_ACTIVE_MULTIVIEW_APPEARANCE / DEVELOPMENT_GATE_NOT_MET /
NO_FINAL_TEST`: fixed three-view appearance increased the PRESERVE tendency,
collapsed Doorway FLIP to `0/24` in both seeds, and left owner-group macro
balanced accuracy near chance. No view selector, G0 fusion, or further
pose/model sweep is authorized from that consumed result.

Dynamic Travel Risk R0 is now the next algorithm mainline. It asks whether
short causal tracks plus predicted target-occupancy intersection with the
wearer route can suppress irrelevant alerts without losing crossing or
oncoming events. The four-arm mechanics smoke is implemented, and the primary
comparison is frozen as route intersection versus radial TTC. A 24-event real
RGB input-materialization canary precedes the exactly 120-event controlled
Development cohort. Its truth-blind RGB/pose adapter is implemented, but no
eligible 24-event input exists yet. Neither real stage has run, so there is no
DTR-R0 scientific result.

L10-R0 is active in parallel and does not depend on GRAIL owner orientation.
Its first controlled closed-loop result is positive: 87.5% task completion,
81.2% reacquisition, 93.1% direction accuracy, 1.7% wrong-lock frames, and zero
false completions in 50 target-absent episodes. The result proves only the
goal-belief controller mechanics under seeded synthetic observations; real RGB
perception and a live-device loop remain untested.

## Demonstration track

Semantic Anchor to Marker Pose remains the live-device showcase closure. Its
device evidence and DTR-R0 research evidence must be reported separately.

## Boundaries

- Historical routes remain closed unless a new versioned experiment changes
  the task representation or introduces a genuinely new information source.
- Curated Development evidence may support the showcase, but its scope must be
  stated.
- Live-camera behavior requires a ready device and a device run; a build or JVM
  test is not a substitute.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence or proof of safety.
- Protected final claims follow [formal governance](formal/RESEARCH_GOVERNANCE.md).
