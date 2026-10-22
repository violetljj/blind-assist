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
Results of the first run: Graduation `phone_m0_20261022/RESULTS.md`.
