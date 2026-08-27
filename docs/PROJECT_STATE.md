# Project state

Updated: 2026-08-27

BlindAssist is a runnable Android showcase research prototype. The default
research policy is effect-first: demonstrate a genuine, visible effect in a
controlled setup, report the setup and metric honestly, and add only checks that
protect interpretation.

## Current operating surface

- Current decision route: [GRAIL R1C-G0](../research/active/grail-r1cg/README.md)
- Current question and stop condition: [CURRENT_DECISION.md](CURRENT_DECISION.md)
- Workstation entrypoint: `tools/ba.ps1`
- Android entrypoint: `scripts/run_android_gradle.ps1`
- Closed experiment lookup: [history-index.md](history-index.md)

## Current evidence

R1C-G0 completed a fresh 24-house synthetic ProcTHOR Development probe. A fixed
source-native relative-camera-yaw transport rule scored 75.26% on 2,094
discriminative pairs versus 75.45% for always PRESERVE and only 15.95% on 514
FLIP-only pairs. The pose-only mechanism stops; active multiview appearance was
not evaluated.

## Boundaries

- Historical routes remain closed unless a new versioned experiment introduces
  a genuinely new information source.
- Curated Development evidence may support the showcase, but its scope must be
  stated.
- Live-camera behavior requires a ready device and a device run; a build or JVM
  test is not a substitute.
- Protected final claims follow [formal governance](formal/RESEARCH_GOVERNANCE.md).
