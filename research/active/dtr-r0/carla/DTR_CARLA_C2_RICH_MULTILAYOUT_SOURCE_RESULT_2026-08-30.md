# DTR-CARLA-C2 rich multi-layout source result — 2026-08-30

## Decision

`DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE`

The requested source-engineering upgrade is complete. C2 materially expands the CARLA scene bank, captures every formal camera at 1280×720, provides a dense truth-blind RGB-D/navigation contract for a deployable algorithm bridge, and includes a CONTACT/SAFE counterfactual pair with a trackable target before a real physical occlusion.

This is a synthetic Development source and scene/asset result. It proves that the controlled input and evaluator evidence can be built; it does not prove that a tracker or avoidance algorithm succeeds, nor is it a real-world safety claim.

## Frozen run

- Final run: `c2-rich-sync-20260830-030054`
- Final evidence root: `E:\linnan\CARLA\experiments\dtr-carla-c2-rich-scene\evidence\c2-rich-sync-20260830-030054`
- Pixel/trajectory source root: `E:\linnan\CARLA\experiments\dtr-carla-c2-rich-scene\evidence\c2-rich-20260830-023610`
- Experiment: `DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2`
- CARLA: `0.9.16`, `Carla/Maps/Town10HD_Opt`
- Protocol SHA-256: `7BE6EDD47EE861E4F830EECB31A07244DBE014151115044CA079D15190AF81EF`
- Capture SHA-256: `FB57372F3750B63D9317EA55C7F113AE472BCDAA9861C773B6B61C7C1F2EDD9B`
- Capture helper SHA-256: `110AC8F4A441015B77BEB97A5BA6D8A3665711857A32134EAAD4CA5E679D4DE9`
- Final join helper SHA-256: `63AB5D24D8F20E5652CE65D064F6338DB0718363933B81ADB620A321B352771A`
- Final join SHA-256: `17C23B69EBA43AD36BF3DE19E3634A0B4FC6B8282E2BDD616CB8245478CA925C`
- RGB-D deterministic replay/alignment receipt: `A6F0F5D8672A98DDF3B923BE19F7838546F87B8850A3253FEE6C168BC390EED5`
- Sealed evidence manifest SHA-256: `B3DD5961C4002FC2D8F4128B92AF83C21DA0B0329220A5B28247C48B89430C6E`
- Sealed model manifest SHA-256: `F5E61479E5586018A866BFDA1B58A2E52D77ED20D759FF2E9C145D231C8092D5`
- Sealed files: `1,913` full-evidence files and `464` model-only files; total evidence-root size: about `1.86 GiB`

The older C1 canary and its 320×180 evidence remain unchanged. The first sealed C2 root also remains unchanged; the final root is a lossless rejoin that adds only the explicit synchronization contract.

## Scene and asset expansion

C2 contains three distinct layouts and four episodes:

| Layout | Purpose | Non-wearer assets | Episode(s) |
|---|---|---:|---|
| `layout_01` | occluded work-zone boulevard | 28 | `ep_01`, `ep_02` |
| `layout_02` | curbside transit and delivery corridor | 32 | `ep_03` |
| `layout_03` | plaza, cafe, and micromobility crossing | 30 | `ep_04` |

The frozen capture used `74` unique actual CARLA blueprint types with zero blueprint fallbacks and no zero-area spawned bounding boxes. The layouts combine pedestrians and crowds, cyclists and powered two-wheelers, cars, a moving Sprinter occluder, buses, HGV/delivery and emergency vehicles, construction assets, transit furniture, kiosks, vending, benches, cafe/food-cart assets, a fountain, and landscaping.

## Sensor result and calibration

Four fresh-server shards were captured in host-safe order: `instance → wearable RGB → metric depth → witness RGB`.

| Sensor | Resolution | Frames | Capture calibration SHA-256 |
|---|---:|---:|---|
| wearable RGB | 1280×720 | 224 | `ECD4944C71C564D575F7AA6C7303AE6904406D244FDA55D87F6B6BA8107C78BA` |
| metric depth | 1280×720 | 224 | `4D37E1E7A239BAE4D32D4D871B1C1EA1A7627B483747971753EB0EE035D82CDE` |
| instance segmentation | 1280×720 | 224 | `FEB213CB3D90894CAB8DBC75B991486D6FA92901A671C53EA47078786BA28BA4` |
| witness RGB | 1280×720 | 224 | `E83B93C124AABFEC6CA581577E7FEBB03A82DEFC4FC21A14D8806BF010BF31C0` |

There are `896` raw formal sensor payloads. The shared model-camera contract is:

- field of view: `90°`;
- sample period: `0.05 s`;
- intrinsic matrix: `K=[[640,0,640],[0,640,360],[0,0,1]]`;
- rigid wearable-to-camera extrinsic: `(x=0.08 m, y=0, z=0.65 m, pitch=-5°, yaw=0°, roll=0°)`;
- depth codec: `CARLA_RGB24_NORMALIZED_DEPTH`;
- decode formula: `meters=1000*(R+256*G+65536*B)/16777215`;
- truth-blind camera calibration file SHA-256: `56FCEF1152CB6BD30A3DBD8F18B16F9FD622EA66F3E97E93069A0EB1CFF0C0A5`.

No gate uses a fixed pixel count. Target trackability uses normalized image area; complete hiding is an exact zero-visible-fraction condition combined with evaluator-only field-of-view, line-of-sight, and physical occluder geometry.

## Track → physical occlusion → reappearance pair

`ep_01` and `ep_02` share the same scene, wearer, target, occluder, and plan prefix. Their target behavior diverges only after the common evidence interval.

| Episode | Pre-track | Complete physical occlusion | Reappearance | Expected/observed outcome |
|---|---:|---:|---:|---|
| `ep_01` | 21 frames (`1–21`) | 8 frames (`22–29`), `0.40 s` | 51 frames | CONTACT / CONTACT at `3.85 s` |
| `ep_02` | 21 frames (`1–21`) | 8 frames (`22–29`), `0.40 s` | 36 frames | SAFE / SAFE |

Both episodes therefore exceed the required `>=10` consecutive visible frames before a real `0.30–0.60 s` complete occlusion. The occlusion indices are identical across the pair. Instance visibility, actor states, physical-occlusion adjudication, contacts, semantic roles, twin labels, and outcomes exist only under the evaluator root.

## Truth-blind model contract

The sealed `model/` root contains, for every one of the `224` observations:

- dense RGB and depth relative paths, byte lengths, SHA-256 hashes, width, and height;
- timestamp, sample index, and authoritative wearable-aligned CARLA `world_frame`;
- each modality's own raw `source_world_frame` plus a deterministic replay/alignment receipt;
- camera intrinsics, FOV, depth codec, current camera world transform, current wearer pose, and rigid extrinsic;
- `navigation_session_id` and `issued_plan { authority, path, receipt_sha256 }`;
- per-episode immutable `plans/<episode>.json` with layout world center/forward/right and layout-relative waypoints converted to CARLA world coordinates;
- explicit `NO_PLAN` authority for the no-plan episodes;
- a root `manifest.json` containing only experiment identity, calibration identity, and each episode-manifest path/hash.

`current_actors` is disabled. A recursive key scan found no actor, role, target, occluder, twin, scenario, contact, outcome, visibility, or evaluator truth in the model root. RGB/depth camera transforms and deterministic replay states match across fresh-server shards.

The observation schema is `dtr-c2-model-observation-v2`. Its exact top-level keys are `camera`, `episode_id`, `frame_alignment`, `metric_depth`, `navigation`, `sample_index`, `schema_version`, `time_s`, `timestamp_s`, `wearable_rgb`, `wearer_pose_current`, and `world_frame`. `wearable_rgb` adds `source_world_frame`; `metric_depth` adds the same field alongside its codec. `frame_alignment` contains `authority`, `reference_modality`, `receipt_path`, `receipt_sha256`, and `depth_minus_wearable_source_world_frame_offset`.

All four episodes have a verified depth-minus-RGB source-frame offset of `70` CARLA frames. The receipt verifies equal episode/sample/time identity, camera world transform, and current wearer pose before mapping depth into the wearable `world_frame` namespace.

## Lossless synchronization rejoin

The final root was joined from the unchanged sealed sensor shards of `c2-rich-20260830-023610`; no CARLA replay or image regeneration was performed. Independent old-root/new-root comparison found:

- `1,793 / 1,793` PNG files byte-identical, including every raw/model/evaluator sensor pixel and contact sheet;
- `992 / 992` raw shard files byte-identical;
- all `16` raw frame sidecars byte-identical;
- all `4` immutable plan files byte-identical;
- all preserved fields across `224` model observations exactly equal after removing only the v2 synchronization additions and schema-version change;
- all `224` evaluator records semantically equal; their only byte-level difference is the expected absolute destination-root string in two `relative_path` fields.

Thus all pixels, physical trajectories, camera/wearer trajectories, timing, navigation, plans, evaluator truth, and outcomes are unchanged from the first sealed C2 root.

## Gate result

All frozen result checks passed, including:

- all four 1280×720 fresh-server shards and raw-payload inventories;
- exact cross-sensor replay identity;
- `74` unique actual blueprint types and zero fallbacks;
- CONTACT/SAFE outcome pair;
- track-before-complete-physical-occlusion contract;
- dense model RGB-D, calibration, world-frame, plan, navigation, anchor, and root-manifest contract;
- deterministic RGB/depth replay alignment with both source CARLA frames;
- zero model-root actor/evaluator truth keys;
- sealed model and full-evidence manifests;
- high-resolution contact sheet and evaluator summary;
- release of all task-owned CARLA processes and ports `2000–2002`.

## Evidence layout

- `model/`: truth-blind dense RGB-D observations, camera/wearer pose, calibration, navigation sessions, immutable plan receipts, layout anchors, and root/episode manifests.
- `evaluator/`: instance visibility, witness views, actor/replay truth, physical-occlusion and outcome reports, and the contact sheet.
- `shards/`: raw fresh-server sensor payloads, sidecars, inventories, and capture receipts.
- `sealed_model_manifest.json`: content identity for the deployable model-only package.
- `sealed_evidence_manifest.json`: content identity for all model, evaluator, and raw-shard evidence.

Primary visual artifact: `evaluator/contact_sheet.png`.

## Reproduction

From the BlindAssist checkout:

```powershell
pwsh -NoProfile -File .\tools\run_dtr_carla_c2_rich_scene.ps1 -RunId <new-unique-run-id>
```

The runner refuses an existing evidence root or a shared CARLA process, starts one fresh server per sensor, captures one long-lived 1280×720 camera in each server, stops after a failed shard, performs the model/evaluator join, seals both roots, and verifies resource release.

## Claim boundary

- Motion is deterministic scripted kinematics, not native pedestrian or vehicle policy.
- Plans are synthetic immutable receipts, not inferred human intent.
- CARLA RGB-D is Development input, not proof of real sensing fidelity.
- Passing C2 proves rich controlled scene construction, dense deployable sensor packaging, and evaluator-ready counterfactual evidence. Algorithm benefit requires a separate frozen predictor/evaluator replay.
