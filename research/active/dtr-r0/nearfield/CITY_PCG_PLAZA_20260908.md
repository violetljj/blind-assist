# City Sample PCG: first native plaza integration

2026-09-08. Engineering capability check, no model experiment or promotion.
Status: FIRST_PLAZA_CAPTURE_PASS; broader visual acceptance remains pending.

## Delivered capability

The downloaded City Sample runs on local UE5.8.2 (BuildId 55116800). A new
`/Game/BAResearchSlice/PlazaV2` map reuses the unchanged official graph
`/CitySamplePCG/Examples/Plaza/Plaza_Square_Tree_lines_and_benches_on_each_side`.
Seed 42 generates 545 instances: 289 pavers, 36 trees, 36 tree bases, 20 benches,
4 trash cans, 66 curbs and 94 border pieces. This is roughly a 51 m square
plaza, not a complete city or a 100–300 m research world.

The new map lives in the downloaded project under the F:-backed
`artifacts.local/unreal/CitySample/Content/BAResearchSlice/` namespace. Source
maps and the project descriptor were not resaved. Map SHA256:
`45936938056d0759511f053bce52e2fc49ccb9aa32b84596b8ee4620529383a3`.
The frozen Willow map is not used by this adapter.

`tools/build_city_pcg_slice.py` creates a new map and rejects existing targets.
`tools/run_city_pcg_capture.py` loads it without saving, inserts temporary
controlled cubes, and exports native RGB/depth plus appearance images.
`tools/verify_city_pcg_capture.py` checks analytic cube geometry, local floor,
and exports evaluator-only BODY/HEAD masks for all visible native surfaces.
Model inputs contain RGB only; depth, controls and support masks stay separate.

## Actual checks and retained evidence

All paths below are relative to
`artifacts.local/nearfield/city-pcg-20260908/`.

| Evidence | Observed result |
| --- | --- |
| `plaza-build-2/build-receipt.json` | 545 instances; warm-cache PCG generation 5.563 s |
| `plaza-control-3/receipt.json` | Three clear/BODY/HEAD frames PASS; source unchanged |
| `plaza-control-3/completion.json` | Native probe RGB pixel identity and depth byte identity passed |
| `plaza-control-3/verification.json` | CUDA RTX5060 Laptop; geometry and support export PASS |
| `plaza-control-3/process-release.json` | All observed task descendants exited; released=true |
| `verification-final/verification.json` | Final verifier rechecked the same cached capture: PASS |
| `verification-mask-failure/verification.json` | Injected mask-write exception correctly yields FAIL |

The camera is at (0, 0, 1.9) m, pitch -5 degrees, HFOV100, 1.70 m above the
declared 0.20 m paving surface. RGB/depth is 640x360; appearance is 1280x720.
All three appearance images were inspected. Exposure EV13.2 and antialiasing
fix the observed over/underexposure and improve foliage readability; distant
city enclosure and natural-looking hazard skins remain absent.

The BODY cube has 4,714 eligible interior pixels and the HEAD cube 3,911;
100% lie within 3 cm of analytic axial depth. Median errors are 0.00000215 m
and 0.00000346 m. Fifty-one HEAD interior pixels with unknown clear-baseline
depth are excluded explicitly. The 4,800-pixel floor patch has median error
0.00769 m, all within 5 cm. Support targets are respectively [0,0], [1,0],
[0,1], with 4,250 BODY pixels and 2,334 HEAD pixels. Unknown native depth stays
UNKNOWN, not free space. These are static controlled engineering checks.

Final capture-script time is 29.0 s, map load 1.578 s; host process lifecycle
is 76.812 s including startup and shutdown. Generation used a warm cache;
these numbers do not describe cold-start or bulk capture throughput.

Earlier attempts remain retained: `overview-1` stopped after 270.625 s with
2,282 mesh compilation jobs still pending and zero frames; this motivated
isolating the official plaza graph. `plaza-build-1` exposed duplicate ground
and an absolute-path receipt defect, fixed in V2. `plaza-control-1` and `-2`
passed geometry but had exposure defects, corrected in `-3`. Cold initial
inventory incurred about twelve minutes of engine shader preparation.

## Secondary-machine preparation

The AssetRegistry hard/soft dependency closure contains 205 files and
5,895,031,234 bytes. With the PCG plugin descriptor, 206 files and
5,895,032,165 bytes were transferred and individually SHA256-verified in
402.6 s. Destination:
`G:/DevWorkspace/BlindAssist/artifacts/work/city-pcg-20260908/CitySampleSliceV2`.
See `worker-transfer-assets-v2/` for the manifest and transfer receipt.
The worker's UE5.8.2 engine has all 156 referenced engine package/module
entries and seven declared plugin dependencies (`engine-resources-audit-v2.json`).

Source/native-plugin snapshots and the successful capture evidence are also
staged under the worker's task artifact tree. This is verified prepositioning,
not a completed worker UE run. Runtime string-loaded dependencies are not
proven complete by AssetRegistry closure alone. No worker training or bulk
capture was started, and the worker checkout was not changed.

## Reproduction and remaining scope

Use the repository GPU Python runtime for the host launchers/verifier. The
successful `plaza-control-3/launch.json` records the exact engine, project,
native plugin, source hashes and command. `plaza-final-spec.json` records
camera, cubes, map hash, exposure and 120 settling ticks. Capture with
`tools/run_city_pcg_capture.py --project <uproject> --spec <spec> --plugin <uplugin>
--output <new-artifact-directory>`, then verify with
`tools/verify_city_pcg_capture.py --capture <same-directory>`.
New maps use `tools/build_city_pcg_slice.py --project <uproject> --map-name
BAResearchSlice/<new-name> --output <new-artifact-directory>`.
Never overwrite previous evidence; adapt absolute paths on the worker.

PCG placement rules now work with the controlled capture layer. Collision
floor probes returned no hits, so walkable collision is not certified by this
result. Full street surroundings, 100–300 m world families, natural hazard
skins, dynamics and Electric Dreams remain next implementation work. Thirty
image visual acceptance, real-device transfer and algorithm gains are not
claimed. Asset-specific training/redistribution terms remain to be checked;
no license conclusion is inferred from NoAI alone.
