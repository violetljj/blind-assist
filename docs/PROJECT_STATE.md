# Project state

Updated: 2026-08-28

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

L10 is active in parallel and does not depend on GRAIL owner orientation. SC1W
separates fresh semantic identity, DINO/motion continuity, and a RapidOCR CTC
word carrier for current-camera steering. SC2 adds opportunity-correct active
search: an incomplete OCR miss is `UNKNOWN + SEARCH`, with explicit
`SWEEP/SCAN/PAN/APPROACH/SIDESTEP/HOLD`, rather than false semantic absence or
terminal STOP. SC4 adds source-isolated, belief-latched OCR routing: CRAFT may
rescue cold-start acquisition, but after RapidOCR has acquired once it cannot
override LOST or navigate.

On video1+video10 Development, the broader SC3 word-scope source reached 88.84%
identity recall, 96.18% target support, 91.19% correct-direction coverage,
99.18% navigation precision, five wrong frames, and 30/30 end-to-end success.
On consumed video14, the final SC4 routing diagnostic rescued the two
primary-never-acquired goals: end-to-end rose 18/24 -> 24/24, identity recall
84.51% -> 95.77%, target support 88.73% -> 100%, with 100% navigation precision
and zero wrong identities. Fresh video16 independently showed the primary path
is strong on locally unique text: 96.15% recall, 100% support, 21/21 end-to-end,
100% navigation precision, and zero wrong identities. A broader per-frame SC3
fallback failed its video16 gate after one target-gap false identity; SC4 closes
that override route.

Fresh video17 then passed SC4's non-regression/safe-neutral gate but exposed the
next base-controller limit: repeated `Dairy/Milk` package instances reduced the
unchanged exact-track score to 48.23% identity recall, 61.11% end-to-end, and
53.26% navigation precision. SC5 now represents up to four source-latched
physical hypotheses with independent local/context DINO, OCR-context, motion,
and evidence-age state; appearance can propagate but cannot create identity or
navigate. On the consumed video17 diagnostic, a legal `SET_VALUED` reading kept
100% navigation precision and raised natural-frame navigation coverage from
73.45% to 81.42%. The exact-instance `53.26%` score is not a legal baseline for
that result: every video17 goal maps to multiple physical IDs, while the public
input supplies no reference, initial box, context anchor, or functional role.
Exact-instance and functional disambiguation are therefore `NOT_EVALUABLE`, not
a failed matcher.

The first 2x context fingerprint also did not beat the existing local crop
signal (`0.9297` versus `0.9408` pairwise AUC), so it is retained as state but
not promoted as the next information source. Frozen UNIQUE behavior remained
identical on all 651 video16 and 546 video14 parity decisions. The next SC5
effect gate requires a public exact-instance binding or evaluator-authoritative
functional/aperture association; it is not another threshold sweep. Actions
remain unexecuted, and metric arrival, live product benefit, user benefit, and
safety remain unproven.

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
