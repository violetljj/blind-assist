# DTR-CARLA-C1 complex-scene and asset result — 2026-08-30

## Decision

`DTR_CARLA_C1_COMPLEX_SCENE_ASSET_CANARY_COMPLETE`

The engineering question is closed positively: Codex can programmatically build, replay, capture, verify, and visualize a multi-actor CARLA wearable scene with an immutable planned route, independently scripted wearer execution, physical occlusion, a work-zone asset cluster, and evaluator-only contact truth.

This is a synthetic Development scene/asset result. It is not an algorithmic C1 gain, a real-wearer result, or a safety claim.

## Frozen run

- Run: `c1-complex-20260830-020400`
- Evidence root: `E:\linnan\CARLA\experiments\dtr-carla-c1-complex\evidence\c1-complex-20260830-020400`
- CARLA: `0.9.16`, `Carla/Maps/Town10HD_Opt`
- Protocol SHA-256: `C6489EB50689B9EBCBAC4D8D70B22F01A9E32B0A1277C728E53CE1FAA0B8E4D7`
- Capture SHA-256: `DC4ACCEF185FEE0DFB3D5155866C522F4F160C9C41680A8F4038556B2E1932AA`
- Helper SHA-256: `CC37C5FB972277E4B3C24D623F4D58B5DBA83E4589770A84675B37269FF01FB4`
- Join SHA-256: `E68C79FB7533362CFDC286DC93FA738276E9C7A4860DA808895068242E56AF30`
- Sealed evidence manifest SHA-256: `D0986EFE20820C815493D740641470B862B045BE44377EFE96D57387E72BFBEA`
- Sealed files: `4,937`; total run size: `503.14 MiB`

## Materialized scene

The scene contains 15 task-owned actors:

- one wearable pedestrian;
- one crossing pedestrian, one cyclist, one parallel pedestrian, and one child distractor;
- one parked Mercedes Sprinter as a 3-D physical occluder;
- two street barriers, six construction cones, and one construction warning sign.

Every preferred CARLA blueprint existed and spawned. No fallback blueprint was used. Four fresh-server shards were captured in the host-safe order `instance → wearable RGB → depth → witness RGB`. Each shard contains `8 × 141 = 1,128` raw PNG payloads with byte length and SHA-256 inventory, for `4,512` raw sensor frames total.

Dynamic scene actors are recreated at every episode boundary. The wearer, static work zone, physical occluder, and one long-lived camera remain stable within each shard. CARLA actor IDs and world frame IDs are diagnostic only; cross-shard identity uses the frozen asset key and sample index.

## Gate result

| Gate | Result |
|---|---:|
| Four fresh-server shards complete | PASS |
| Raw payload inventories rehashed | PASS |
| Actual actor/plan/contact replay identical across shards | PASS |
| Expected CONTACT/SAFE relation | PASS, 8/8 |
| Physical van occlusion while target is inside camera FOV | PASS, every episode |
| Model root contains no evaluator truth keys | PASS |
| RGB/depth/instance/witness keyframes complete | PASS |
| Model root and full evidence root sealed | PASS |
| Visual contact sheet, layout, and timeline materialized | PASS |
| Task-owned CARLA processes and ports 2000–2002 released | PASS |

All five client/server stderr logs and the join stderr log are empty.

## Episode truth table

| Episode | Plan/execution condition | Expected | Observed | First contact | Van-occluded frames |
|---|---|---:|---:|---:|---:|
| `ep_01` | valid straight / follows | CONTACT | CONTACT | 5.50 s | 66 |
| `ep_02` | valid straight / follows / target stops | SAFE | SAFE | — | 66 |
| `ep_03` | valid turn-away / follows | SAFE | SAFE | — | 50 |
| `ep_04` | valid turn-away / ignored, actual straight | CONTACT | CONTACT | 5.50 s | 66 |
| `ep_05` | expired turn-away / actual straight | CONTACT | CONTACT | 5.50 s | 66 |
| `ep_06` | no plan / actual straight | CONTACT | CONTACT | 5.50 s | 66 |
| `ep_07` | multi-target / primary stops | SAFE | SAFE | — | 66 |
| `ep_08` | multi-target / primary crosses | CONTACT | CONTACT | 5.50 s | 66 |

The issued plan receipt is generated and hashed independently of the realized wearer trajectory. `ep_03` and `ep_04` have byte-identical planned-route receipts and identical execution prefixes through 2.5 s, then differ only in realized execution. `ep_05` is causally `EXPIRED`; `ep_06` is `NO_PLAN`.

## Evidence layout

- `model/`: wearable RGB/depth keyframes, current actor state, current/past wearer state, and immutable issued-plan receipts only.
- `evaluator/`: instance visibility, witness RGB, realized execution, semantic roles, OBB contact union, twin labels, reports, and visual artifacts.
- `shards/`: all raw fresh-server sensor payloads, frame sidecars, asset manifests, dynamic spawn histories, and SHA-256 inventories.
- `sealed_evidence_manifest.json`: content identity for every file beneath `shards/`, `model/`, and `evaluator/`.

Primary visual artifacts:

- `evaluator/contact_sheet.png`
- `evaluator/scene_layout.svg`
- `evaluator/timeline.svg`

## Reproduction

From the BlindAssist checkout:

```powershell
pwsh -NoProfile -File .\tools\run_dtr_carla_c1_complex.ps1 -RunId <new-unique-run-id>
```

The runner refuses an existing run directory or shared CARLA process, uses one GPU camera per fresh server, stops after a failed shard, and verifies that task-owned processes and ports are gone before starting the next shard.

## Claim boundary

- Motion is deterministic scripted kinematics, not native pedestrian or bicycle dynamics.
- Plans are synthetic immutable receipts, not inferred human intent.
- Current actor state and depth are CARLA Development inputs; this result does not establish real sensing or tracking.
- Instance segmentation, witness RGB, realized execution, contact, semantic roles, and twin labels are evaluator-only.
- Passing this canary proves that complex CARLA scenarios and assets can be created and controlled with auditable evidence. It does not prove that a BlindAssist risk algorithm benefits from them until a separate frozen predictor/evaluator experiment is run.
