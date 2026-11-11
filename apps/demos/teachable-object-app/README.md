# Teachable object app (Graduation thesis line)

Standalone demo for the Graduation thesis method (teachable personal-object recognition: discriminative local memory +
on-demand zoom). It shares this repository's QAIRT/SNPE build setup; the obstacle-avoidance app is not touched.

## Personal teaching (0.2.0-personal-wcl, versionCode 2)

The launcher now opens `WclTeachActivity`: enter a name, record the object for ten seconds (or finish early),
then choose **保存教学并开始识别**. Capture samples about three centre-cropped 224-pixel frames per second;
at least five frames are required. Names are unique, new objects are appended, and incomplete captures do not enter
the teaching library. The screen uses named standard controls and accessibility live status; manual TalkBack use
has not yet been evaluated. **继续识别已保存的物品** reuses the saved prototypes after reopening the app.

Personal images and the atomically saved prototype library live in `files/wcl/libraries/personal/`, separate from
the old `m0` validation packages and `m2` local-memory teaching. Prototypes are tied to the SHA256 of the actual
`files/wcl/wcl_cached.dlc` and the preprocessing version; changing either requires rebuilding the library.
New teaching invalidates the previous library until saving completes. The model must already be provisioned;
this change does not bundle or download the 2.57 GB model. The light local-memory demo remains available through
**打开轻量识别演示**. Personal teaching currently uses WCL only; PF remains the separate validation/research path.

The live chain keeps the existing confidence-weighted EMA and announces after three consecutive analysed frames
agree at the default 3 fps. This is a frame-count rule, not a measured one-second hold. **结束识别，返回教学**
releases the live camera and restores the teaching preview. There is no personal-usage rejection guarantee or
new real-object accuracy claim. The targeted `WclPersonalTeachingTest` checks persistence/model mismatch,
camera lifecycle, and a controlled two-object teaching/reopen path using existing Development frames.
Names containing Chinese characters request Chinese speech; other names retain English. Missing language data
or speech-service failure is shown on the recognition screen instead of silently claiming an announcement.

```powershell
.\scripts\run_android_gradle.ps1 :teachable-object-app:assembleDebug :teachable-object-app:assembleDebugAndroidTest
adb shell am instrument -w -r -e class com.linnan.blindassist.teachable.WclPersonalTeachingTest com.linnan.blindassist.teachable.test/androidx.test.runner.AndroidJUnitRunner
```

The model bytes are hashed on saving and on reopening before recognition, so startup includes that verification.
The demo version changes because the launcher and user teaching flow changed; the obstacle app version is unchanged.

## Earlier research and replay paths

M0 (offline chain, `M0DeviceTest`): teach -> build memory on the phone -> identify, on ORBIT validation-user packages
exported by `E:\Graduation\research\active\assistive-thesis-screen\benchmark_screen\phone_m0_20261022\export_m0.py`.

- Backbone DINOv2-S/14 224 on the HTP (FP16); matching + pooling head + localiser on the CPU (FP32), memory as input.
- Memory build (`teachable_jni.cpp`): FP32 port of the thesis `part_memory_from` with portable sampling.

Run:

```powershell
.\scripts\run_android_gradle.ps1 :teachable-object-app:assembleDebug :teachable-object-app:assembleDebugAndroidTest
adb install -r -t <build>\outputs\apk\debug\teachable-object-app-debug.apk
adb install -r -t <build>\outputs\apk\androidTest\debug\teachable-object-app-debug-androidTest.apk
# stage backbone_fp32.dlc, head_dyn.dlc, head_ref/ and the user packages in /data/local/tmp/m0, then:
adb shell "run-as com.linnan.blindassist.teachable sh -c 'mkdir -p files; cp -r /data/local/tmp/m0 files/m0'"
adb shell am instrument -w -e class com.linnan.blindassist.teachable.M0DeviceTest com.linnan.blindassist.teachable.test/androidx.test.runner.AndroidJUnitRunner
# results: run-as ... cat files/m0/results/<user>.json
```

Native code cannot open adb-pushed files under `Android/data` on the SM-S9280, hence the app-internal copy.
M1 (live, `LiveActivity` / `M1LiveTest -e durationSec 600 -e user P189`): CameraX 30 fps -> centre crop 224 -> identify -> EMA30 ->
speak once stable for 15 frames; logs to filesDir/m1/live_<user>.json.

Results: Graduation `phone_m0_20261022/RESULTS.md` (M0) and `RESULTS_M1.md` (M1).

WCL chain (optional, separate from the main method): `WclPipeline` + `WclLiveActivity` run the ORBIT-adapted SigLIP2-So400m
whole-frame prototypes on the HTP at 3 frames per second. Copy the offline-prepared `wcl_cached.dlc` (2.57 GB, Graduation
`orbit_wcl_phone_20261102`) to `files/wcl/`, then `WclDeviceTest#offlineChain` or `WclDeviceTest#liveRun -e durationSec 600 -e rate 3`.
Scores are smoothed by a confidence-weighted EMA (`ConfidenceEma`, half-life 0.5 s, power 64, Graduation `orbit_wcl_stream_20261106`); `-e confPower 0 -e halfLife 1` restores the old plain EMA.

Spatial readout (optional, on the same chain): `PfPipeline` runs the WCL body and the prototype-guided read head as two HTP graphs,
with up to 20 objects per graph and a mask. Copy `pf_body_cached.dlc` (2.49 GB) and `pf_head_m20_cached.dlc` (Graduation
`orbit_wcl_protofull_phone_20261104`) to `files/wcl/`, then `WclDeviceTest#offlineSpatial` or `#liveRun -e spatial true`.

r10 relook (optional): put `loc_r10_htp.dlc` (HTP; or `loc_r10.dlc` for the CPU) and an empty `r10_htp` marker in files/m0;
`TeachablePipeline(..., locR10, r10Htp)` + `identify(useR10 = true)` centre the half-frame relook crop on the 224-px dilated
exemplar-conditioned localiser's box (only on frames that look again; ~2.4 ms on the HTP). An `r10_only` marker skips the
HTP memory-build arms in `M0DeviceTest`. Export and results: Graduation `phone_r10_20261024/` (`export_r10.py deploy`, RESULTS.md).

Full system B+WCL (offline and replay tests only; no live activity, the app default is unchanged): `BwclDeviceTest` runs the
DINOv2-B local-memory branch (`TeachablePipeline(..., dim = 768)`: B backbone on the HTP, B pooling head + localiser on the CPU,
memory built on the phone by `Native.build(..., dim = 768)` on the CPU) next to the WCL branch and fuses z(B) + z(WCL) per frame.
Copy `backbone_b_cached.dlc`, `bhead_fp32.dlc` and `<user>_pc_parts_b.raw` (Graduation `bwcl_phone_chain_20261111`) to `files/bwcl/`,
then `BwclDeviceTest#offlineChain` (results in `files/bwcl/results/`) or `BwclDeviceTest#replay -e durationSec 600 -e rate 3`.
The HTP memory-build graphs (`simmax`, `simtop`) stay 384-wide; the 768-wide build runs on the CPU.
