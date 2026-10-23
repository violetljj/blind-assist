# Teachable object app (Graduation thesis line)

Standalone demo for the Graduation thesis method (teachable personal-object recognition: discriminative local memory +
on-demand zoom). It shares this repository's QAIRT/SNPE build setup; the obstacle-avoidance app is not touched.

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

r10 relook (optional): put `loc_r10_htp.dlc` (HTP; or `loc_r10.dlc` for the CPU) and an empty `r10_htp` marker in files/m0;
`TeachablePipeline(..., locR10, r10Htp)` + `identify(useR10 = true)` centre the half-frame relook crop on the 224-px dilated
exemplar-conditioned localiser's box (only on frames that look again; ~2.4 ms on the HTP). An `r10_only` marker skips the
HTP memory-build arms in `M0DeviceTest`. Export and results: Graduation `phone_r10_20261024/` (`export_r10.py deploy`, RESULTS.md).
