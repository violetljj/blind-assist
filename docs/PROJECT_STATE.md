# Project state

Updated: 2026-08-26

BlindAssist is a runnable Android showcase research prototype. The default
research policy is effect-first: demonstrate a genuine, visible effect in a
controlled setup, report the setup and metric honestly, and add only checks that
protect interpretation.

## Current operating surface

- Exactly one active research route: [GRAIL R1CL](../research/active/grail-r1cl/README.md)
- Current question and stop condition: [CURRENT_DECISION.md](CURRENT_DECISION.md)
- Workstation entrypoint: `tools/ba.ps1`
- Android entrypoint: `scripts/run_android_gradle.ps1`
- Closed experiment lookup: [history-index.md](history-index.md)

## Current evidence

The R1CL environment doctor and a two-sample DINOv2 CUDA forward/loss/backward
smoke have passed on the configured local workstation. This proves the local
mechanics only; it is not a training result, natural-distribution result, live
camera result, or product-safety claim.

## Boundaries

- Historical routes remain closed unless a new versioned experiment introduces
  a genuinely new information source.
- Curated Development evidence may support the showcase, but its scope must be
  stated.
- Live-camera behavior requires a ready device and a device run; a build or JVM
  test is not a substitute.
- Protected final claims follow [formal governance](formal/RESEARCH_GOVERNANCE.md).
