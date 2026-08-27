# Project state

Updated: 2026-08-27

BlindAssist is a runnable Android showcase research prototype. The default
research policy is effect-first: demonstrate a genuine, visible effect in a
controlled setup, report the setup and metric honestly, and add only checks
that protect interpretation.

## Current operating surface

- Current ten-meter route: [L10-R0 Goal-Lock Copilot](../research/active/l10-r0/README.md)
- Current obstacle/risk route: [Dynamic Travel Risk R2](../research/active/dtr-r0/README.md)
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

Dynamic Travel Risk R2 now combines robust finite-horizon occupancy consensus
with one fixed imminent route-intersection guard and stable
`ONSET / HOLD / ESCALATE / CLEAR` events. On the 19-session, route-authoritative
THÖR-MAGNI ceiling it recalls `10/10` events with 42 false-alert segments,
strictly improving R0's `9/10` and 55. On 27 JRDB test sequences it recalls
`164/175` with 256 false alerts versus R0's `161/175` and 260. CODa adds a hard
pose-authoritative development/holdout check: R2 recalls `119/122` pedestrian
events with 285 false alerts versus R0's `122/122` and 286, retaining the small
recall cost instead of tuning it away.

The static CODa ceiling adds causal curved-route and bounded vertical
occupancy for walls/barriers, fixed structures, and temporary obstacles. It
recalls all `12/12` observed path-contact events while reducing three-metre
proximity false alerts from 104 to 10 (`90.4%`) and clearing `4/4` eligible
events. The selected data contains no positive vegetation/head-clearance event;
GOOSE lacks the synchronized route/ground-clearance truth needed to turn tree
crown semantics into honest human head-collision truth, so that partition is
`NOT_EVALUABLE`. These remain privileged public-data algorithm ceilings, not
RGB/LiDAR detector, Android runtime, natural-distribution, user-benefit, or
safety evidence.

L10 is active in parallel and does not depend on GRAIL owner orientation. Its
first controlled closed-loop result remains positive: 87.5% task completion,
81.2% reacquisition, 93.1% direction accuracy, 1.7% wrong-lock frames, and zero
false completions in 50 target-absent episodes. A new real-RGB L10-SC0 source
keeps OCR as identity authority and uses DINOv2-S crop embeddings only for
short/long continuity. On the two-video Development replay it improved sticky
text from 80.76% to 84.43% target-frame accuracy, cut wrong selections from 102
to 5, and raised gap reacquisition from 90% to 100%, at the cost of 48 to 104
misses. On one previously unseen video12 clip it transferred to 97.24% accuracy,
zero wrong target-present selections, and 96.97% reacquisition, but did not beat
that clip's unusually strong sticky OCR baseline on accuracy/reacquisition.
This is a real-image continuity source effect plus an honest relative holdout
failure; it is not open-world identity, active-view, navigation, product, or
safety evidence.

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
