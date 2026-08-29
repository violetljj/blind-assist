# DTR-CARLA-C0 causal benchmark canary

Date: 2026-08-30
Terminal: `DTR_CARLA_C0_PARTIAL_OBSERVATION_LADDER_REPORTED`

## Outcome

The first runnable CARLA-to-DTR benchmark is complete. One frozen 12-episode
cohort contains six causal twin families, 161 samples per episode at 20 Hz,
and four deterministic single-modality replays. Instance, RGB, depth, and raw
CARLA optical flow each completed `1,932/1,932` frames. Cross-modality state
equivalence, pair isolation, expected contact truth, visibility intervention,
physical occlusion, and occluder route clearance all passed.

Predictions were produced without reading realized future truth and sealed as
SHA-256 `7DBBBD9AE8830670ED4E06DCCA001E427861B80BE59BBEF1AB4BC15D25D28BED`
before scoring opened truth.

| Frozen R2 input arm | Recall | False segments | Event precision | Event F1 | Median lead | Known-frame coverage | Pair relation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O0 RGB | 7/7 | 3 | 70.00% | 82.35% | 3.00 s | 41.61% | 5/6 |
| O1 RGB + depth | 7/7 | 6 | 53.85% | 70.00% | 3.00 s | 41.61% | 5/6 |
| O2T RGB + depth + CARLA flow | 7/7 | 8 | 46.67% | 63.64% | 3.00 s | 54.30% | 4/6 |
| O3 privileged current state | 7/7 | 7 | 50.00% | 66.67% | 3.00 s | 97.52% | 5/6 |

O2T is a synthetic teacher arm, not deployable estimated flow. O3 receives
current CARLA target identity and metric position, never realized future
contact.

## Decision

The observation increment effect was not demonstrated by this canary. O0 RGB
has the fewest false segments and the best F1. Depth preserved recall but added
three false segments. Teacher flow increased known coverage by 12.69 percentage
points over O1 but added two more false segments and broke the static/dynamic
background pair.

The route-turn pair failed in all four arms, including privileged current
state. This is the strongest actionable result: the shared downstream model
lacks the wearer's planned future route, so adding more current visual state
cannot resolve that twin. Do not tune detector, depth, flow, route, or lifecycle
thresholds on this consumed 12-episode cohort. The next fresh CARLA increment
should add an explicit planned-route representation and test it first on a new
route twin cohort. Naive flow carry is not promoted.

## Evidence

- Raw evidence:
  `E:\linnan\CARLA\experiments\dtr-carla-c0\evidence\c0-canary-20260830-002109`
- Project result:
  `artifacts.local/evidence/dtr-carla-c0/c0-canary-20260830-002109/result.json`
- Result SHA-256:
  `191CDA07ADDAB5441A0BE60C7CED102F4581D24494EF6066281A996E684BF69E`
- Truth SHA-256:
  `16450A6ACC6006C1D350A911464CFB57F199BA45872AF69BF16699832C9A68B6`
- Protocol SHA-256:
  `2D7A249892D8A72CE830E4162F04B7A08ECB3D297406AC346E7755C74AAC490E5`
- Capture script SHA-256 (identical in all four modality receipts):
  `193A887DCC877AE69F08708D818E89640E67A346B33889B38F937EA3E8E86D58`
- YOLO11n model SHA-256:
  `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`
- Raw archive: 7,904 files, 0.407 GiB.

Two earlier task-owned runs remain only as excluded development diagnostics:
`c0-canary-20260830-000747` used the wrong head-turn polarity, and
`c0-canary-20260830-001218` stopped on an unsupported flow-visualization API.
Neither has a scored result and neither contributes to the table above.

This is a small single-map synthetic Development canary. It is not
source-disjoint real confirmation, synthetic-to-real generalization, Android
runtime evidence, user-benefit evidence, or a reliability or safety claim.
