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

Dynamic Travel Risk R0 crossed its exact public-real privileged ceiling on the
19-session THÖR-MAGNI Pupil subset: C route intersection raised geometric
critical-event recall from 8/10 to 9/10 and reduced target-level false alert
segments from 96 to 55 (42.7%) relative to radial TTC. Its fixed 143-frame JRDB
real-RGB bridge is also directionally positive: both arms recalled 3/3 events,
while C reduced false alert segments from 7 to 4 (42.9%) after YOLO11n and a
causal tracker. C was weaker on lead time and CLEAR, and overall known prediction
coverage was 45.97%.

The current metric center has now been replaced by a truth-blind causal raw
sensor chain: YOLO11n-seg person masks gate motion-compensated upper/lower JRDB
Velodyne points. It covered 90.41% of detector-track occurrences with 0.106 m
median / 0.284 m p90 position error against evaluator-only native centers. Both
arms still recalled 3/3 events and cleared 2/2 eligible events; C reduced false
alert segments from 18 to 13 (27.8%). This is a genuine but weaker effect and
does not reproduce the frozen 40% strong-effect line. No more recording,
detector/matcher sweep, Android, product, natural-distribution, user-benefit, or
safety claim follows.

A fixed phone-transferable RGB source now uses person-box height, one `1.70 m`
upright-person prior, and camera focal length. On the same curated JRDB window,
C retained `3/3` event recall and `2/2` CLEAR while reducing false alert
segments from 17 to 9 (`47.1%`); geometry error was `0.386 m` median / `1.016 m`
p90 against evaluator-only centers. This source is implemented in the isolated
Android `dtrKnownHeight` build with frame-bound Camera2 calibration, causal
multi-person tracking, and explicit `ONSET / HOLD / CLEAR / UNKNOWN` feedback
semantics. The build compiles and its one focused mechanics check passes, but
there is not yet a live-device result or default-App/product/safety claim.

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
