# HFTF Stage C D45 phone metric-depth source canary readiness

日期：2026-08-03

执行状态：`D45_SOURCE_CANARY_READY_FOR_DEVICE_EXECUTION`

科学终态：`D45_NOT_EVALUATED_NO_READY_DEVICE`

## 已落地

- 新增独立 JVM 模块 `:hftf-metric-depth-canary-core`，不在 App dependency graph：
  - registered metric-depth frame + camera intrinsics contract；
  - fixed person-box center-60% sampler；
  - coverage/confidence/IQR/staleness fail-closed receipts；
  - camera-relative x/y/z metric measurement；
  - exact same-target/source/registration 7-point OLS `+1.0 s` forecast；
- 10 个 focused JVM tests 全部通过：
  - accepted person measurement；
  - insufficient coverage rejection；
  - stale receipt rejection；
  - constant-motion OLS recovery；
  - mixed-target rejection；
  - raw+confidence capability precedence；
  - automatic-only confidence-contract rejection；
  - padded row/pixel stride 下 unsigned raw depth/confidence 解码；
  - truncated plane 在任何 partial raster 逸出前 fail closed；
  - decoded raw frame 保持 `SOURCE_REGISTRATION_UNVERIFIED`；
- ARCore 1.33.0 仅加入现有 `:hftf-device-canary` test APK；
- locked API semantic receipt 确认只有 raw depth 暴露对应 confidence image，因此
  R0.1 在任何 device outcome 前把 raw+confidence 设为唯一 measurement-ready
  ARCore source；automatic-only 不伪造 confidence；
- capability test 只执行：
  - `ArCoreApk.checkAvailability`；
  - 未 resume 的 `Session` depth-mode/camera-config query；
  - canonical JSON receipt；
- capability test 不打开 camera、不请求安装 ARCore、不调用 risk/feedback。
- 在既有 isolated `:ustrf-shadow-benchmark` 中新增 source decoder adapter：
  - 只接受与当前 ARCore frame timestamp 完全一致的 raw depth + confidence；
  - 正确处理 `Image.Plane` row stride、pixel stride、buffer position 和 unsigned
    16-bit millimetres；
  - 只产出 `D45UnregisteredRawMetricDepthFrame`，不能进入 person sampler；
- 新增小型 aggregate device canary：
  - 只记录 acquisition failure counts、timestamp advancement、decoded dimensions、
    有效像素 coverage 和 acquisition+decode P50/P95；
  - 不持久化 raster，不产生人物/事件结果；
  - 单 receipt 上限 `256 KiB`，使用 `AtomicFile`；
  - 静止或无纹理条件下零观测终态为 `NOT_EVALUABLE_*`，不是算法负结果。

## 既有物理机 source-class prior，不是 D45 outcome

ignored local evidence 中，同一台 `SM-S9280 / Android 16` 已显示 acquisition
强烈依赖运行上下文：

- 2026-07-20 moving capability r1：900 次 update，586 tracking，585 raw-depth
  acquisition；
- moving capability r2：900 次 update，813 次连续 tracking observation、813
  raw-depth acquisition；但只有 874 个 strictly advancing timestamp，因此不能把
  813 全读成不同 camera image；
- 2026-07-22 autonomous frame-bound r1/r2：各 150 次 update，均为 0 tracking、
  0 advancing timestamp、0 raw depth、0 confidence。

这些 receipt 证明“API supported”远弱于“本次测量可取得”，也说明静止 canary
的零 acquisition 不能烧毁 D45。它们没有保存 depth pixels、没有验证 registration、
没有人物真值，因而不能替代本轮 source decoder device run，更不能构成 D45
正结果。

## non-interference receipt

冻结 commit `9f47a7d` 到当前工作树的 `app/core/feature/gradle` diff 为 0。
Gradle `:app:debugRuntimeClasspath` 中没有：

- `com.google.ar:core`
- `hftf-metric-depth-canary-core`

default debug App merged manifests 中也没有 `com.google.ar.core` 或
`InstallActivity`。ARCore metadata/native libraries 只出现在 test APK。

## build receipt

命令：

```text
.\gradlew.bat :hftf-metric-depth-canary-core:test
  :hftf-device-canary:assembleDebug
  --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m

.\gradlew.bat :app:assembleDebug
  --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

两次均为 `BUILD SUCCESSFUL`。

- unchanged production target APK：
  - path：`app/build/outputs/apk/debug/app-debug.apk`
  - bytes：`259451666`
  - SHA-256：
    `afa7a774b9f47074b2bf2e59755e712e92421484140789513578b32b68f0f149`
- D45-capable test APK：
  - path：
    `hftf-device-canary/build/outputs/apk/debug/hftf-device-canary-debug.apk`
  - bytes：`1385159`
  - SHA-256：
    `1b0142c94abd19a5b0702f67c3c7a38115251f51bd04a25411d6867a570a64ca`
- source-decoder benchmark APK：
  - path：
    `ustrf-shadow-benchmark/build/outputs/apk/debug/ustrf-shadow-benchmark-debug.apk`
  - bytes：`33636481`
  - SHA-256：
    `4b316a5895da000023f24ba19e118d5c1aa97024f8702c0f2e6e9904aa3b3087`
- source-decoder instrumentation APK：
  - path：
    `ustrf-shadow-benchmark/build/outputs/apk/androidTest/debug/ustrf-shadow-benchmark-debug-androidTest.apk`
  - bytes：`425437`
  - SHA-256：
    `d4b90e06c1d0430885dcb9498f305a747555653c078e4d3733dcbf1b67d5f83c`

新增 source-decoder build 命令：

```text
.\gradlew.bat :hftf-metric-depth-canary-core:test
  :ustrf-shadow-benchmark:assembleDebug
  :ustrf-shadow-benchmark:assembleDebugAndroidTest
  --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

结果为 `BUILD SUCCESSFUL`；JVM tests 为 `10/10`。

## 剩余 device actions

当前 `adb devices -l` 无设备，因此没有伪造 capability 结果。连接设备后只运行
D45 class，避免同时触发 D35：

```text
.\gradlew.bat :hftf-device-canary:connectedDebugAndroidTest
  -Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.hftf.HftfD45ArCoreDepthCapabilityCanaryTest
  --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

receipt path：

```text
/sdcard/Android/data/com.linnan.blindassist/files/hftf-d45/
arcore-depth-capability-r0.json
```

若 capability receipt 为 `READY_RAW_DEPTH_REGISTRATION_REQUIRED`，先执行
raw source decoder canary：

```text
.\gradlew.bat :ustrf-shadow-benchmark:connectedDebugAndroidTest
  -Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.ustrfbenchmark.D45ArCoreRawSourceDecoderCanaryTest
  -Pandroid.testInstrumentationRunnerArguments.hftfD45RawSourceFrameAttempts=300
  --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

receipt path：

```text
/sdcard/Android/data/com.linnan.blindassist.ustrfbenchmark/files/
hftf-d45/raw-source-decoder-r0/<run-id>/summary.json
```

只有 `RAW_SOURCE_DECODER_OBSERVED` 才允许进入 registration calibration；它也不
直接授权 person measurement。`AUTOMATIC_ONLY_*_CONFIDENCE_UNAVAILABLE` 与
所有 `NOT_EVALUABLE_*` 都不是 depth 精度负结果；无设备也不是 source 负结果，
不关闭 D45。
