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
- 7 个 focused JVM tests 全部通过：
  - accepted person measurement；
  - insufficient coverage rejection；
  - stale receipt rejection；
  - constant-motion OLS recovery；
  - mixed-target rejection；
  - raw+confidence capability precedence；
  - automatic-only confidence-contract rejection；
- ARCore 1.33.0 仅加入现有 `:hftf-device-canary` test APK；
- locked API semantic receipt 确认只有 raw depth 暴露对应 confidence image，因此
  R0.1 在任何 device outcome 前把 raw+confidence 设为唯一 measurement-ready
  ARCore source；automatic-only 不伪造 confidence；
- capability test 只执行：
  - `ArCoreApk.checkAvailability`；
  - 未 resume 的 `Session` depth-mode/camera-config query；
  - canonical JSON receipt；
- capability test 不打开 camera、不请求安装 ARCore、不调用 risk/feedback。

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

## 唯一剩余的 capability device action

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

只有 receipt 为 `READY_RAW_DEPTH_REGISTRATION_REQUIRED`，才进入已冻结的物理
measurement canary。`AUTOMATIC_ONLY_*_CONFIDENCE_UNAVAILABLE` 是当前合同的
`NOT_EVALUABLE`，不是 depth 精度负结果；无设备也不是 source 负结果，不关闭
D45。
