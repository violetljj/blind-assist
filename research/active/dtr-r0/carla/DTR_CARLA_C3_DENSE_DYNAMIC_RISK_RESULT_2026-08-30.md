# DTR-CARLA-C3 dense dynamic-risk source result — 2026-08-30

## Decision

`DTR_CARLA_C3_DENSE_DYNAMIC_RISK_SOURCE_COMPLETE`

C3 is complete. The frozen Town10 canary contains `40` unique actual CARLA blueprint types with zero fallbacks, including `16` registered dynamic risk targets. Every one of the 16 targets is actually spawned, moves in both episodes, is visible to the model camera for at least 10 frames in both episodes, and enters the declared 3.0 m wearer-risk corridor in both episodes.

This is a synthetic Development source result. It proves that a dense, controlled, truth-blind CARLA input/evaluator package can be built. It does not prove that an avoidance algorithm succeeds and is not a real-world safety claim.

## Frozen run

- Run ID: `c3-dynamic-risk-20260830-165705`
- Evidence root: `E:\linnan\CARLA\experiments\dtr-carla-c3-dynamic-risk\evidence\c3-dynamic-risk-20260830-165705`
- Experiment: `DTR_CARLA_C3_DENSE_DYNAMIC_RISK_SOURCE_V1`
- Scene: `town10_dense_risk_canary`
- CARLA map: `Carla/Maps/Town10HD_Opt`
- Frozen protocol canonical SHA-256: `9037F405A74A6DB46662AF28053D8F9A748EAF1F78F10EEFE7B00AB61CD34FA6`
- Frozen protocol file SHA-256: `99C80B46FE5DC41DAF46C486203A521CFCC9D1985D3F0D6DAE498A9D299F48C8`
- Final model manifest SHA-256: `6E93281ABF64756AD4B83771BCE393E9794F6B066E27C67AEFFC0980B1C1CFA2`
- Final sealed evidence manifest SHA-256: `4681346104C1F870A3CBAD7865F0369B9AC7236CC847DC352AEE01A5D6767574`
- Sealed evidence files: `1,376`
- Evidence-root size: `1,478,723,188` bytes, about `1.38 GiB`
- Dynamic-risk audit SHA-256: `AF22C054ABDBD75D0533E0CEA1EA6B195B28DA8E83DE619494EF639930EFF452`

All 12 final source checks passed. The model schema and model-truth failure lists are both empty.

## Asset and scene registration

The requested assets are not merely mentioned in scene code. They are frozen in evaluator-only registries and bound to the compiled protocol by a compiler receipt:

| Frozen object | File SHA-256 | Canonical content SHA-256 |
|---|---|---|
| asset registry | `EF929DAA5BACCDBA36249947DF17EE04D3D2838C4369E6E1F6A3244FABD0636F` | `62B696AD0930CF4D6BF2CE00B525CD2E2B437008425957D6E7AFC4295C607A05` |
| scene registry | `B41627789652FEE2285A61F0AE9496E7F4A0FD18D4F39D1CF055EF666B7D2523` | `640EBB9EA9AF800C0E8A4460D3CCB7F3CEC4160D7AD024D0A2EBA6A0471AD54F` |
| compiler receipt | `EA8181701C08F39423C59F4A31D045A1C9B54289B5F5B891E04BEAD374135EB8` | binds compiled protocol `9037F405A74A6DB46662AF28053D8F9A748EAF1F78F10EEFE7B00AB61CD34FA6` |

The captured scene has `39` non-wearer assets plus the wearer and `40` unique actual blueprints with zero blueprint fallbacks. It combines a child, police and multiple adult motion patterns, bicycle, motorcycle, sedan, moving ambulance occluder, city bus, delivery HGV, emergency sedan, additional vehicles, road furniture, construction props, signals, vegetation, and the Town10 urban background.

The registries and compiler receipt exist only below `evaluator/c3_registry/`; no registry hash, role, target identity, actor truth, outcome, or twin identity is exposed under `model/`.

## Sixteen dynamic risk targets

The table is computed from captured per-frame CARLA actor transforms, model-camera visibility, wearer-relative polygon clearance, and contact responsibility. `visible`, `risk`, and `motion` are shown as `CONTACT/SAFE` episode frame counts.

| Track | Registered asset / role family | Exact blueprint | Visible | Inside 3.0 m risk corridor | Observed transform motion | Minimum clearance | CONTACT responsibility |
|---|---|---|---:|---:|---:|---:|---:|
| `d_01` | `c3_dynamic_01_child_crossing` / child | `walker.pedestrian.0009` | 67/47 | 44/44 | 80/80 | 0.000 m | 9 frames |
| `d_02` | `c3_dynamic_02_police_crossing` / police | `walker.pedestrian.0030` | 44/44 | 22/22 | 80/80 | 1.970 m | 0 |
| `d_03` | `c3_dynamic_03_adult_parallel` / adult | `walker.pedestrian.0024` | 44/44 | 45/45 | 80/80 | 0.662 m | 0 |
| `d_04` | `c3_dynamic_04_adult_crossing` / adult | `walker.pedestrian.0025` | 69/69 | 20/20 | 80/80 | 1.604 m | 0 |
| `d_05` | `c3_dynamic_05_adult_parallel` / adult | `walker.pedestrian.0026` | 68/68 | 81/81 | 80/80 | 0.265 m | 0 |
| `d_06` | `c3_dynamic_06_adult_crossing` / adult | `walker.pedestrian.0027` | 64/64 | 24/24 | 80/80 | 1.207 m | 0 |
| `d_07` | `c3_dynamic_07_adult_approach` / adult | `walker.pedestrian.0028` | 53/53 | 5/5 | 80/80 | 2.761 m | 0 |
| `d_08` | `c3_dynamic_08_adult_receding` / adult | `walker.pedestrian.0029` | 71/71 | 8/8 | 80/80 | 2.728 m | 0 |
| `d_09` | `c3_dynamic_09_adult_crossing` / adult | `walker.pedestrian.0031` | 69/69 | 6/6 | 80/80 | 2.590 m | 0 |
| `d_10` | `c3_dynamic_10_bicycle_parallel` / bicycle | `vehicle.bh.crossbike` | 81/81 | 81/81 | 38/38 | 0.681 m | 0 |
| `d_11` | `c3_dynamic_11_motorcycle_crossing` / motorcycle | `vehicle.harley-davidson.low_rider` | 15/15 | 20/20 | 28/28 | 0.797 m | 0 |
| `d_12` | `c3_dynamic_12_sedan_approach` / sedan | `vehicle.audi.a2` | 71/71 | 19/19 | 38/38 | 0.905 m | 0 |
| `d_13` | `c3_dynamic_13_ambulance_van` / van | `vehicle.ford.ambulance` | 21/21 | 21/21 | 26/26 | 0.876 m | 0 |
| `d_14` | `c3_dynamic_14_bus` / bus | `vehicle.mitsubishi.fusorosa` | 74/74 | 5/5 | 6/6 | 0.986 m | 0 |
| `d_15` | `c3_dynamic_15_hgv` / HGV | `vehicle.carlamotors.european_hgv` | 73/73 | 7/7 | 10/10 | 0.560 m | 0 |
| `d_16` | `c3_dynamic_16_emergency_sedan` / emergency | `vehicle.dodge.charger_police_2020` | 66/66 | 22/22 | 80/80 | 2.085 m | 0 |

The weakest visibility case is still `15` frames per episode, above the required `10`. The weakest risk-corridor case is `5` frames per episode. The slowest large actor still has `6` captured transform-motion frames per episode.

All scripted actors have CARLA engine collisions disabled so that a collision response cannot perturb one fresh-server sensor shard differently from another. That does not disable evaluation: the evaluator computes risk and physical contact from the actual per-frame CARLA bounding-box polygons. The CONTACT episode has only `target_primary` as responsible; the SAFE episode has no responsible actor.

## CONTACT/SAFE counterfactual and real occlusion

Both episodes have 81 frames at 20 Hz and share the same scene, wearer, target, occluder, plan prefix, and selected occlusion indices.

| Episode | Pre-track | Complete physical occlusion | Reappearance | Expected / observed |
|---|---:|---:|---:|---|
| `c3_town10_e01` | samples `1–20`, 20 frames (`1.0 s`) | samples `21–30`, 10 frames (`0.50 s`) | 47 frames | CONTACT / CONTACT at `3.6 s` |
| `c3_town10_e02` | samples `1–20`, 20 frames (`1.0 s`) | samples `21–30`, 10 frames (`0.50 s`) | 27 frames | SAFE / SAFE |

The target therefore has a genuine tracking opportunity before complete physical occlusion, remains fully hidden for a valid `0.30–0.60 s` interval, and reappears. The evaluator can adjudicate physical occlusion, but the model receives no actor, target, occluder, outcome, or twin truth.

## Formal sensors and calibration

Four fresh-server shards were captured: instance segmentation, wearable RGB, metric depth, and witness RGB.

| Sensor | Resolution | Payloads | Capture calibration SHA-256 |
|---|---:|---:|---|
| wearable RGB | 1280×720 | 162 | `ECD4944C71C564D575F7AA6C7303AE6904406D244FDA55D87F6B6BA8107C78BA` |
| metric depth | 1280×720 | 162 | `4D37E1E7A239BAE4D32D4D871B1C1EA1A7627B483747971753EB0EE035D82CDE` |
| instance segmentation | 1280×720 | 162 | `FEB213CB3D90894CAB8DBC75B991486D6FA92901A671C53EA47078786BA28BA4` |
| witness RGB | 1280×720 | 162 | `E83B93C124AABFEC6CA581577E7FEBB03A82DEFC4FC21A14D8806BF010BF31C0` |

There are `648` raw sensor payloads. The shared formal camera contract is:

- field of view: `90°`;
- sample/sensor period: `0.05 s`;
- intrinsic matrix: `K=[[640,0,640],[0,640,360],[0,0,1]]`;
- wearable rigid extrinsic: `x=0.08 m, y=0, z=0.65 m, pitch=-5°, yaw=0°, roll=0°`;
- depth codec: `CARLA_RGB24_NORMALIZED_DEPTH`, maximum depth `1000 m`;
- depth formula: `meters=1000*(R+256*G+65536*B)/(16777215)`;
- truth-blind camera calibration SHA-256: `56FCEF1152CB6BD30A3DBD8F18B16F9FD622EA66F3E97E93069A0EB1CFF0C0A5`.

No gate relies on a fixed pixel count. Trackability uses normalized image area; full hiding requires zero visible fraction plus evaluator-only field-of-view, line-of-sight, and physical-occluder geometry.

## Truth-blind model-only contract

The model package contains 162 dense RGB-D observations, immutable plan receipts, navigation identity, layout anchors, camera/wearer transforms, timestamps, source CARLA frames, and explicit deterministic replay alignment. `current_actors` is disabled.

Exact observation keys:

- top level: `camera, episode_id, frame_alignment, metric_depth, navigation, sample_index, schema_version, time_s, timestamp_s, wearable_rgb, wearer_pose_current, world_frame`;
- `camera`: `fov_degrees, height, K, rigid_extrinsic, width, world_transform`;
- `wearable_rgb`: `bytes, height, path, sha256, source_world_frame, width`;
- `metric_depth`: `bytes, codec, height, path, sha256, source_world_frame, width`;
- `navigation`: `issued_plan, navigation_session_id`;
- `navigation.issued_plan`: `authority, path, receipt_sha256`;
- `frame_alignment`: `authority, depth_minus_wearable_source_world_frame_offset, receipt_path, receipt_sha256, reference_modality`;
- `wearer_pose_current`: `pitch, roll, x, y, yaw, z`.

Exact plan keys:

- top level: `episode_id, issued_plan, layout_anchor, navigation_session_id, schema_version`;
- `layout_anchor`: `world_center_xy_m, world_forward_xy, world_right_xy`.

Exact root-manifest keys are `camera_calibration, episodes, experiment_id, model_contract, rgbd_alignment_receipt, schema_version`. Exact episode-manifest keys are `depth_payloads, episode_id, frames, issued_plan, navigation_session_id, observations_sha256, rgb_payloads, rgbd_alignment, schema_version`.

Both episodes have an observed depth-minus-wearable source CARLA-frame offset of `-776`. The deterministic alignment receipt SHA-256 is `B2501DFDB5E20B1715D770B5D0E45005D685152C86E188DBDE0277A94602D147`; it verifies equal episode/sample/time identity, camera world transform, and current wearer pose before mapping depth into the wearable `world_frame` namespace. The offset is a receipt for this frozen capture, not a general CARLA constant.

Recursive exact-schema and semantic truth scans found no enabled actor list, actor/role/target/occluder/twin/scenario/contact/outcome/visibility/evaluator path, or C3 registry hash under `model/`.

## Replay and old-root relationship

The final four C3 fresh-server shards have exact actual replay identity: `replay_failures=[]`. Actor and wearer trajectories match sample-for-sample across instance, wearable, depth, and witness captures. Pixel bytes are modality-specific and are not expected to equal one another.

C3 is intentionally **not** pixel- or trajectory-identical to the old C2 root. It adds and repositions actors and changes their trajectories to create the denser risk scene. The old C1/C2 canaries and every failed C3 attempt remain preserved and were not overwritten. C2 compatibility was rechecked against the live final tree:

- C2 compatibility status: `DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE`;
- compatibility model-manifest SHA-256: `87A1CD5E7560DBED6FE2BB6671E14059E8FB923E6F2AB0BAF0D3B4C3AC6FB4EE`;
- compatibility result SHA-256: `A3BE7C97805372DADE6FCEBB946D979101CD03B2C75BCD58B5245B97CE428FD5`.

## Evidence layout

- `model/`: truth-blind dense RGB-D, calibration, transforms, world frames, timestamps, navigation, plan receipts, layout anchors, and manifests.
- `evaluator/`: instance and witness evidence, actor/replay truth, physical occlusion, contact/outcome adjudication, the C3 risk audit, and visual contact sheet.
- `evaluator/c3_registry/`: frozen asset/scene registries, compiler receipt, and C2 compatibility result.
- `shards/`: raw fresh-server sensor payloads, actor sidecars, inventories, and capture receipts.
- `sealed_model_manifest.json`: immutable identity of the deployable model-only package.
- `sealed_evidence_manifest.json`: immutable identity of the complete evidence package.

Primary visual artifact: `evaluator/contact_sheet.png`.

## Reproduction

From the BlindAssist checkout, with the shared CARLA/GPU idle:

```powershell
pwsh -NoProfile -File .\tools\run_dtr_carla_c3_dynamic_risk.ps1 -RunId <new-unique-run-id> -RpcPort 2020
```

The runner refuses an existing evidence root or conflicting CARLA port group, compiles and freezes the registries, captures one fresh server per sensor, joins the truth-blind/evaluator trees, audits all 16 dynamic targets, seals both manifests, and verifies task-owned CARLA resource release.

## Claim boundary

- Motion is deterministic scripted kinematics, not native pedestrian or vehicle policy.
- Plans are synthetic immutable receipts, not inferred wearer intent.
- CARLA RGB-D is Development input, not proof of real sensing fidelity.
- Passing C3 proves dense controlled source construction, registered dynamic-risk participation, deterministic replay, and a deployable truth-blind input contract. Algorithm benefit requires a separate frozen predictor/evaluator replay.
