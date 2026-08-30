# DTR-CARLA-C4 multi-map world-pack result — 2026-08-30

## Decision

`DTR_CARLA_C4_MULTIMAP_SOURCE_COMPLETE`

C4 is complete. The formal source now spans six CARLA maps and eight distinct scene classes, with four 1280×720 sensor modalities, 40 registered asset types, 68 dynamic-risk placements, and 43 static-support placements. All 16 episodes and all 136 per-dynamic-per-episode risk checks passed.

This is a sealed scripted CARLA Development source pack. It proves that the requested varied worlds, dense dynamic-risk participation, deployable truth-blind RGB-D/navigation inputs, and evaluator-only physical-occlusion authority can be built. It does not prove an obstacle-avoidance algorithm, real-device generalization, or product safety.

## Frozen run

- Experiment: `DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1`
- Final evidence outer root: `E:\linnan\CARLA\experiments\dtr-carla-c4-multimap\evidence\c4-multimap-world-pack-v3-20260830-195834`
- Final package: `E:\linnan\CARLA\experiments\dtr-carla-c4-multimap\evidence\c4-multimap-world-pack-v3-20260830-195834\final-package`
- Result status: `DTR_CARLA_C4_MULTIMAP_SOURCE_COMPLETE`
- Result file SHA-256: `67BB11646250D60DB1E4F5C44BE706DBDB44739A61F4C6FA3D6C3EEB062BF1BA`
- Final runtime index SHA-256: `DA89DE21F404D5547D6435CF7B3FB7A8AE8DEE2AB9D82B063D823167B450340B`
- Asset registry SHA-256: `BB40AE5367E6ABAD58490C4CA6646D2E802F135327A421A19BFF6406AD971FF9`
- Scene registry SHA-256: `65CAAF85C5568E46A0FCDB586B440CF4E42277F5A59671462369231FC47C5B1C`
- Outer model-root manifest SHA-256: `C8CE20AD72ED8F6B17FC9C74409C473707E515482D50CCFF77EEBBB0F122733C`
- Sealed model manifest SHA-256: `B36AAE6B0070E3A98459719833FB0BE26102981F2FD59DA177937F450A8738FE`
- Sealed evidence manifest SHA-256: `7FDA98E991EF0A4081B3EA484C847B0E5FA94FCAEA029C1364FB718A31C9EB21`
- Sealed files: `2,665` model-only and `5,333` full-evidence files
- Layout coverage audit SHA-256: `5DE7FFCDDAC8A285747E5E74D6F61805D736633FB0069CA5851CD2A6E4121F1D`
- Pack physical-occlusion audit SHA-256: `2A1CCBEBE0401789FCD2416F7F2DCBF5D2714083EC91AD8287717AB14F6C078B`

All final result checks are true. `copy_failures=[]` and `model_truth_failures=[]`.

## Scene and map inventory

| Scene class | Layout | CARLA map | Episodes |
|---|---|---|---|
| narrow alley | `c4_layout_01` | `Town01` | `c4_l01_e01`, `c4_l01_e02` |
| mall exit | `c4_layout_02` | `Town03_Opt` | `c4_l02_e01`, `c4_l02_e02` |
| parking lot | `c4_layout_03` | `Town05` | `c4_l03_e01`, `c4_l03_e02` |
| bus stop | `c4_layout_04` | `Town04` | `c4_l04_e01`, `c4_l04_e02` |
| construction zone | `c4_layout_05` | `Town10HD_Opt` | `c4_l05_e01`, `c4_l05_e02` |
| rainy night | `c4_layout_06` | `Town02` | `c4_l06_e01`, `c4_l06_e02` |
| backlight | `c4_layout_07` | `Town03_Opt` | `c4_l07_e01`, `c4_l07_e02` |
| crowded pedestrians | `c4_layout_08` | `Town01` | `c4_l08_e01`, `c4_l08_e02` |

The pack therefore contains six maps, eight layout families, and 16 CONTACT/SAFE counterfactual episodes. Each episode has 81 samples at 20 Hz. That yields 1,296 frames per modality and 5,184 formal sensor payloads across wearable RGB, metric depth, instance segmentation, and witness RGB.

The frozen per-map protocol hashes are:

| Group | Protocol SHA-256 |
|---|---|
| `town01` | `2A73B959647BCF7CDC2F5E34F13149F9CDC9983041A0979AE91012A66DC165B3` |
| `town02` | `E6D8865966FE866E723E76F99FAF5AD79CEA503B8BF693D74835B78B48066366` |
| `town03_opt` | `C5ABC0A5F884ECDDF08933F11ABC74B688842F08F1D979FF90B0A2BE4C0DB249` |
| `town04` | `F6A60ACA80335469431AAA3F9A4D0BF96335982FA5E3CF26D34D5D696B5CB2BF` |
| `town05` | `B7F07F01755CA573664FB4A6EEEE5E3472C80015171DD3A45B243F10A3A73F25` |
| `town10hd_opt` | `A6FF94C144D46147B29DAAE3CA003DDF10AD2BC32160380994B1D46EDB9C0E7E` |

## Asset registration and risk participation

The frozen asset registry contains 40 exact CARLA blueprint types:

- one wearer blueprint;
- 16 dynamic risk types: child, police and adult pedestrians, bicycle, motorcycle, sedan, ambulance, bus, HGV, and emergency sedan families;
- 23 static support types: crowd actors, parked micromobility and vehicles, construction barriers/cones/warnings/container, street furniture, bins, mailboxes, chain barriers, plants, garbage, and accident signage.

Across the eight layout definitions there are 119 registered placements: eight wearers, 68 dynamic-risk targets, and 43 static-support actors. The CONTACT/SAFE episode twin for each layout reuses the same declared scene assets, so the evaluator performs 68 × 2 = 136 dynamic target checks.

Every one of the 136 checks has captured state, nonzero motion, sufficient model-camera visibility, and wearer-risk-corridor participation. There are no failed episodes. No visibility gate uses a fixed pixel count; the audit uses normalized image visibility and evaluator-only physical geometry.

The user-facing machine catalog at `E:\linnan\CARLA\asset-catalog.json` registers this source as `c4-multimap-world-pack-v1`, including evidence paths, hashes, counts, calibration, reuse boundaries, and claim ceiling.

## Track → physical occlusion → reappearance authority

C4 contains two independently qualifying CONTACT/SAFE pairs, exceeding the required one pair:

| Layout / map | Episodes | Pre-track | Complete physical occlusion | Post-reappearance |
|---|---|---:|---:|---:|
| `c4_layout_01` / Town01 | `c4_l01_e01` CONTACT, `c4_l01_e02` SAFE | 20 / 20 frames | samples 21–30, 10 frames, 0.50 s | 48 / 32 frames |
| `c4_layout_02` / Town03_Opt | `c4_l02_e01` SAFE, `c4_l02_e02` CONTACT | 21 / 21 frames | samples 22–30, 9 frames, 0.45 s | 33 / 50 frames |

Each pair has at least ten clear pre-track frames, identical nonempty complete-occlusion indices across its twins, a real 0.30–0.60 s physical occlusion, at least ten post-reappearance frames, and observed outcomes exactly CONTACT/SAFE. Actor identity, visibility, physical occlusion, trajectories, contact, and outcome remain evaluator-only.

## Formal sensors and calibration

Every map group has four fresh-server shards in the formal order `instance → wearable → depth → witness`. All four modalities are native 1280×720; no low-resolution source is upscaled.

- resolution: `width=1280`, `height=720`;
- horizontal field of view: `90°`;
- sensor/sample period: `0.05 s`;
- intrinsic matrix: `K=[[640.0000000000001,0,640],[0,640.0000000000001,360],[0,0,1]]`;
- wearable rigid extrinsic: `x=0.08 m, y=0, z=0.65 m, pitch=-5°, yaw=0°, roll=0°`;
- depth codec: `CARLA_RGB24_NORMALIZED_DEPTH`, maximum depth `1000 m`;
- depth formula: `meters=1000*(R+256*G+65536*B)/(16777215)`;
- shared camera-calibration file SHA-256 in all six groups: `56FCEF1152CB6BD30A3DBD8F18B16F9FD622EA66F3E97E93069A0EB1CFF0C0A5`.

Visual inspection of a native wearable frame from each of the eight classes found no black, corrupt, or low-resolution output. Large vehicles occupying most of the image at the showcase time are the intended near-field occlusion event, not a resolution downgrade. CARLA texture filtering and application display scaling can still make a 1280×720 preview look softer than its stored pixel dimensions.

## Truth-blind model-only schema

The outer `model/manifest.json` exact keys are `experiment_id, groups, schema_version`. Each `groups[]` item has `child_model_root_manifest_sha256, child_sealed_model_manifest_sha256, group_id, model_file_count, model_root`.

Each group root manifest exact keys are `camera_calibration, episodes, experiment_id, model_contract, rgbd_alignment_receipt, schema_version`. Each episode-manifest exact keys are `depth_payloads, episode_id, frames, issued_plan, navigation_session_id, observations_sha256, rgb_payloads, rgbd_alignment, schema_version`.

Each `observations.jsonl` record has these exact keys:

- top level: `camera, episode_id, frame_alignment, metric_depth, navigation, sample_index, schema_version, time_s, timestamp_s, wearable_rgb, wearer_pose_current, world_frame`;
- `camera`: `K, fov_degrees, height, rigid_extrinsic, width, world_transform`;
- `camera.rigid_extrinsic`: `pitch_degrees, roll_degrees, x_m, y_m, yaw_degrees, z_m`;
- `camera.world_transform`: `pitch, roll, x, y, yaw, z`;
- `wearable_rgb`: `bytes, height, path, sha256, source_world_frame, width`;
- `metric_depth`: `bytes, codec, height, path, sha256, source_world_frame, width`;
- `metric_depth.codec`: `formula, maximum_depth_m, name`;
- `navigation`: `issued_plan, navigation_session_id`;
- `navigation.issued_plan`: `authority, path, receipt_sha256`;
- `frame_alignment`: `authority, depth_minus_wearable_source_world_frame_offset, receipt_path, receipt_sha256, reference_modality`;
- `wearer_pose_current`: `pitch, roll, x, y, yaw, z`.

Every frame carries the wearable-aligned `world_frame` plus both RGB and depth source CARLA frames. Deterministic replay/alignment receipts verify episode/sample/time identity and equal camera/wearer poses before mapping depth into the wearable frame namespace. The observed depth-minus-wearable source-frame offsets are frozen per group—Town01 `872`, Town02 `190`, Town03_Opt `-1205`, Town04 `156`, Town05 `-244`, and Town10HD_Opt `-361`—and are evidence receipts, not general CARLA constants.

Each immutable `plans/<episode>.json` has exact top-level keys `episode_id, issued_plan, layout_anchor, navigation_session_id, schema_version`. `issued_plan` has `authority, receipt, receipt_sha256, time_parameterized_waypoints_world, world_coordinate_frame`; `layout_anchor` has `world_center_xy_m, world_forward_xy, world_right_xy`. All 16 C4 episodes explicitly carry `authority=VALID`; the bridge does not infer a route from actor truth.

`current_actors_enabled=false`. Recursive exact-schema and semantic scans found no actor, target, occluder, role, scenario, CONTACT/SAFE outcome, twin identity, collision truth, visibility truth, or evaluator path under `model/`.

## Reuse, recapture, and old-root boundary

The final source did not overwrite any prior evidence:

- first complete but visibility-gate-failed root: `E:\linnan\CARLA\experiments\dtr-carla-c4-multimap\evidence\c4-multimap-world-pack-20260830-183257`;
- partial V2 root: `E:\linnan\CARLA\experiments\dtr-carla-c4-multimap\evidence\c4-multimap-world-pack-v2-20260830-195025`;
- successful V3 root: `E:\linnan\CARLA\experiments\dtr-carla-c4-multimap\evidence\c4-multimap-world-pack-v3-20260830-195834`.

Town01, Town02, Town03_Opt, and Town05 had unchanged protocol hashes and were copied from V2 through a byte-verified reuse receipt. Town04 and Town10HD_Opt intentionally changed the affected dynamic target trajectory and were fully recaptured in all four sensor modalities. The fixes raised the previously failed target visibility to Town04 CONTACT/SAFE `35/41` frames and Town10HD_Opt CONTACT/SAFE `38/43` frames without weakening the denominator or threshold.

Therefore four map groups are pixel- and trajectory-identical verified copies of their earlier completed children. The two recaptured groups intentionally differ. It would be false to claim that the entire V3 pack is pixel- or trajectory-identical to either older root.

## Evidence layout

- `final-package/model/`: truth-blind dense RGB-D, hashes, calibration, camera/wearer pose, timestamps, source frames, navigation sessions, immutable plans, anchors, and alignment receipts.
- `final-package/evaluator/`: copied child evaluator evidence, layout coverage, dynamic target participation, physical occlusion, contact/outcome truth, and frozen registries.
- `child-evidence/<group>/`: six independently sealed C2-compatible map-group captures.
- `frozen-inputs/`: immutable compiled inputs and per-map protocols.
- `reused_children_receipt.json`: source/destination identity and exact reuse verification for four unchanged groups.
- `final-package/sealed_model_manifest.json`: immutable identity of the deployable model-only tree.
- `final-package/sealed_evidence_manifest.json`: immutable identity of the full final package.

## Reproduction

With the shared CARLA/GPU idle, run from the BlindAssist checkout:

```powershell
pwsh -NoProfile -File .\tools\run_dtr_carla_c4_multimap.ps1 -RunId <new-unique-run-id>
```

The runner compiles the registries, creates six map-group protocols, captures one fresh server per sensor, validates all dynamic targets, joins truth-blind and evaluator trees, audits the pack-level occlusion contract, seals both manifests, and releases only task-owned CARLA resources. `-ReuseChildEvidenceRoot` may copy only a completed child whose exact protocol hash matches; every copy is revalidated before final joining.

## Claim boundary

- Motion is deterministic scripted kinematics, not native pedestrian or vehicle policy.
- Layout routes are synthetic immutable receipts, not inferred wearer intent.
- CARLA RGB-D is Development input, not proof of real sensing fidelity.
- C4 establishes varied controlled source construction and evaluator-ready dynamic-risk participation. Algorithm benefit still requires a separately frozen truth-blind predictor/evaluator replay, and real-world confirmation must remain source-disjoint.
