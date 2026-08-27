# Project state

Updated: 2026-08-27

BlindAssist is a runnable Android showcase research prototype. The default
research policy is effect-first: demonstrate a genuine, visible effect in a
controlled setup, report the setup and metric honestly, and add only checks that
protect interpretation.

## Current operating surface

- Current decision route: [GRAIL R1C-G1](../research/active/grail-r1cg/README.md)
- Current question and stop condition: [CURRENT_DECISION.md](CURRENT_DECISION.md)
- Workstation entrypoint: `tools/ba.ps1`
- Android entrypoint: `scripts/run_android_gradle.ps1`
- Closed experiment lookup: [history-index.md](history-index.md)

## Current evidence

R1C-G0 closed after relative-camera-yaw transport failed its gate. R1C-G1 then
tested a fresh matched comparison between one reference image and a fixed
anchor/left/right scan. It also closed: three-view balanced-accuracy uplift was
-0.05pp and -0.49pp in the two frozen seeds, FLIP accuracy declined, and mean
Doorway uplift was negative. The state is
`STOP_G1_ACTIVE_MULTIVIEW_APPEARANCE / DEVELOPMENT_GATE_NOT_MET / NO_FINAL_TEST`.
No view-selector or further pose/model sweep is authorized from this result.

## Boundaries

- Historical routes remain closed unless a new versioned experiment introduces
  a genuinely new information source.
- Curated Development evidence may support the showcase, but its scope must be
  stated.
- Live-camera behavior requires a ready device and a device run; a build or JVM
  test is not a substitute.
- Protected final claims follow [formal governance](formal/RESEARCH_GOVERNANCE.md).
