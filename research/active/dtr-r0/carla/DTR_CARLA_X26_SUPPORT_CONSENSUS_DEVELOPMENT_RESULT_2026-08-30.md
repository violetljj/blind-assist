# DTR CARLA X26 support-consensus development result — 2026-08-30

## Decision

`DTR_CARLA_X26_SUPPORT_CONSENSUS_DEVELOPMENT_GATE_NOT_MET`

X26 is a large fresh-cohort improvement over X24, but it is not promotable. On C4 it raised frame F1 from `0.5308` to `0.8571`, raised recall from `47.46%` to `96.61%`, and gained more than two seconds of warning on both new edge-contact targets. It nevertheless produced four false-alert segments across the two fresh SAFE episodes, while the frozen gate required zero.

Do not rescue this result by changing the route tube, confirmation duration, HOLD duration, association distance, target placement, or C4 scene. C4 is consumed Development evidence.

## Structural change

X25 replaced a single median 3D point with a metric 2D quantile-OBB footprint. Its C3 replay improved aggregate F1 from `0.4929` to `0.6797`, but retained one bicycle SAFE false-alert segment.

X26 retained X25's footprint, detector, candidate set, issued-plan authority, route geometry, confirmation, HOLD, and all X24 gates. It changed only motion authority: per axis, it transported the footprint by the smaller-magnitude displacement of the two opposing support boundaries. The intent was to distinguish rigid translation, which moves both supports, from partial visibility, which often erodes only one support.

## Frozen C4 source

- Cohort: `DTR_CARLA_C4_SUPPORT_CONSENSUS_FRESH_DEVELOPMENT_V1`
- Successful source run: `E:\linnan\CARLA\experiments\dtr-carla-c4-support-consensus\evidence\c4-support-consensus-20260830-132608`
- Protocol SHA-256: `96BBD3C8B99CECFC3812709FA497AB205DDC9129EEBF3C5BABA8A8A24ED36F68`
- Source result SHA-256: `842C59D56334851936630F9C411F5BCDED35198D1BEEE5DA119D8B2000D0A095`
- Model manifest SHA-256: `38ED951C50B7C655826F2CD416424466D5FB39B8934A8B2837B0BBD76C2411FB`
- Capture: four sensors, 336 frames per sensor, 1,344 payloads, 1280×720, six episodes, three layouts, and at least 60 unique actual blueprints
- Fresh changes relative to C3: seed `73129`, changed weather, a delivery HGV contact/SAFE pair, and a side-on Harley contact/SAFE pair

The earlier `c4-support-consensus-20260830-132418` attempt ended before any episode capture because its CARLA server exited after opening the streaming port. That failed infrastructure attempt is preserved and was not scored. The successful run started only after no CARLA process, listener, or material GPU allocation remained.

## Frozen replay

- Model run: `E:\linnan\linnan\artifacts.local\evidence\dtr-carla-x26-support\c4-support-consensus-20260830-132608`
- Truth-blind RGB index: 336 frames
- YOLO segmentation candidates: 3,390
- Candidate manifest SHA-256: `225D47C14C73D9170085C12568A40513D731A2E15D5343EEE32E3FB2D48E9798`
- X24 freeze SHA-256: `A7BE0693EC318AE62E35182993B12BC80A8F333AA9B79A68589684B733959DE5`
- X24 predictions SHA-256: `B60FB84357DF5972F8F795FE2ACCAB1516F674B6538E27F2617FCE5009E60AB3`
- X26 freeze SHA-256: `8FBE410A81A67249C39488BA392AFE000267E329E60FFA7B9CCABF9EC7ED2A5C`
- X26 predictions SHA-256: `E8BD29BFC71EF71B63D9B52F90A78F46AA95AF0A3B230FFF8993CDC08CF92D63`
- Result SHA-256: `88FE4F72EA4F9270613DBE7C4E23BB1340620663F0DB327F0545846E8FCAC4AA`

| Metric | X24 | X26 | Delta |
|---|---:|---:|---:|
| Frame precision | 0.6022 | 0.7703 | +0.1681 |
| Frame recall | 0.4746 | 0.9661 | +0.4915 |
| Frame F1 | 0.5308 | 0.8571 | +0.3263 |
| True positives | 56 | 114 | +58 |
| False negatives | 62 | 4 | -58 |
| False positives | 37 | 34 | -3 |

### CONTACT behavior

| Episode | Target | X24 lead | X26 lead | Gain | X24 future-positive recall | X26 future-positive recall |
|---|---|---:|---:|---:|---:|---:|
| `ep_01` | original occlusion target | 3.0 s | 3.0 s | 0.0 s | 0.9375 | 0.9063 |
| `ep_03` | delivery HGV edge | 0.9 s | 3.1 s | +2.2 s | 0.3269 | 0.9808 |
| `ep_05` | side-on Harley edge | 0.6 s | 3.0 s | +2.4 s | 0.2647 | 1.0000 |

Both arms covered all `5/5` physical-occlusion frames in the original episode.

### SAFE behavior

| Episode | X24 false segments / frames | X26 false segments / frames |
|---|---:|---:|
| `ep_02` original SAFE | 0 / 0 | 0 / 0 |
| `ep_04` fresh HGV SAFE | 2 / 11 | 1 / 4 |
| `ep_06` fresh Harley SAFE | 3 / 15 | 3 / 16 |

The following frozen checks passed: both fresh edge contacts detected, each gained at least one second of warning, aggregate F1 exceeded X24, and original occlusion coverage was retained. The zero-fresh-SAFE-false-segment check failed.

## Post-score failure attribution

The evaluator was opened only after both prediction files were sealed.

1. In `ep_04`, the responsible HGV is fixed in the protocol, but its measured track acquired about `-0.31` to `-0.60 m/s` lateral velocity at `4.1–4.4 s`. Support consensus reduced the error but did not stop both OBB supports from moving together when the visible set and OBB orientation changed.
2. In `ep_06`, one person track at `2.1–2.7 s` matched the wearer's issued motion: about `1.94 m/s` forward versus the wearer's `2.0 m/s`, at the wearer's current route position. This is an ego-carried/self component treated as an external obstacle.
3. A protocol-fixed plaza pedestrian later acquired about `-1.23 m/s` lateral velocity after a visible-support transition. The erroneous state then remained authoritative through HOLD, creating the longest false segment.
4. One final false frame came from a short held person track, showing that a single rigid-motion fit still grants too much authority to weak or identity-unstable support.

These are representation failures, not evidence that the route tube is too wide. X26 should remain a recorded negative Development gate.

## Next structural route

The next successor should replace the single fitted-motion hypothesis with an occupancy-support authority decomposition in the fixed route frame:

- `EGO_CARRIED`: support colocated with the wearer and moving with issued ego motion is excluded from external-obstacle risk;
- `STATIC_SCENE`: temporally overlapping metric support is transported with zero world velocity;
- `RIGID_DYNAMIC`: non-overlapping support must demonstrate coherent translation before nonzero velocity and HOLD receive authority;
- otherwise `UNAUTHORIZED_MOTION`, which may preserve current occupancy but may not sweep a speculative velocity into the route.

That successor requires a fresh cohort containing both static and genuinely moving contact/SAFE pairs. It must not be selected or tuned on C4.

## Claim boundary

This is fresh scripted-CARLA Development evidence. It is not source-disjoint confirmation, real-world evidence, product validation, or a deployment claim.
