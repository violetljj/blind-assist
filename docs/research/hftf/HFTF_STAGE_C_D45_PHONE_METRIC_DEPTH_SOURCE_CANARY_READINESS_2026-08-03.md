# HFTF Stage C D45 phone metric-depth source canary readiness

日期：2026-08-03

执行状态：
`D45_SOURCE_REGISTRATION_PERSON_MEASUREMENT_AND_AGGREGATION_READY_FOR_DEVICE_EXECUTION`

科学终态：`D45_NOT_EVALUATED_NO_READY_DEVICE`

## 已落地

- 新增独立 JVM 模块 `:hftf-metric-depth-canary-core`，不在 App dependency graph：
  - registered metric-depth frame + camera intrinsics contract；
  - fixed person-box center-60% sampler；
  - coverage/confidence/IQR/staleness fail-closed receipts；
  - camera-relative x/y/z metric measurement；
  - exact same-target/source/registration 7-point OLS `+1.0 s` forecast；
- 24 个 focused JVM tests 全部通过：
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
  - CPU-image crop/scale affine recovery；
  - detector 90° rotation 与 inverse round-trip；
  - non-affine coordinate receipt fail closed；
  - registration identity 对微小浮点噪声稳定；
  - cross-frame registration 不能解锁 raw raster；
  - native sparse depth pixels 不经 upsampling 重复计入 coverage；
  - detector region 落在 depth crop 外时报告 `NO_REGISTERED_PIXELS`，不混入
    depth-quality failure；
  - padded/interleaved YUV_420_888 到 RGBA 解码；
  - truncated YUV plane 在 partial image 逸出前 fail closed；
  - 1/2/3/5 m error median/P90 与 relative-error summary；
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

## R0.3 pre-device coordinate semantic repair

ARCore 官方坐标合同明确：

- raw depth 是 GPU aspect ratio、native orientation；
- CPU camera image 与 depth image 可能宽高比不同，depth 是 camera image 的 crop；
- camera pixel 必须用
  `Frame.transformCoordinates2d(IMAGE_PIXELS, ..., TEXTURE_NORMALIZED, ...)`
  转为 depth coordinate，不能只按宽高缩放。

参考：

- `https://developers.google.com/ar/develop/java/depth/developer-guide#converting_coordinates_between_camera_images_and_depth_images`
- `https://developers.google.com/ar/develop/java/depth/raw-depth`

因此 R0.3 在没有 device outcome 前修复原实现中的简单 scale 假设：

- `detector display -> native CPU image` 使用 CameraX detector rotation；
- `native CPU image -> raw depth` 使用同一 ARCore frame 的 9-point coordinate
  receipt 拟合并检查 affine residual；
- sampler 逐个 inverse-map native raw-depth pixel center 到 person inner box，
  不 upsample sparse depth，也不把同一个 depth pixel 重复计入 coverage；
- registration observation 与 exact source frame id/timestamp 绑定，不能跨帧解锁；
- transform id 对微小浮点抖动 canonicalize，避免纯数值噪声切断 7-point history；
- depth byte order 使用 Android `nativeOrder()`，与官方 uint16 读取语义一致。

设备 receipt 中
`AFFINE_REGISTRATION_OBSERVED_DEVICE_ONLY` 只表示 ARCore 内部 coordinate mapping
可被一致地恢复；`external_alignment_verified=false` 与
`person_registration_verified=false` 保持不变。它不能替代 1/2/3/5 m 人工量测。

## R0.4 person measurement runner readiness

在没有 device/person outcome 前，首个物理 canary 已从“需要再开发”推进到
“设备接入即可执行”：

- `:ustrf-shadow-benchmark` 只读复用
  `app/src/main/assets/yolo11n_fp16_320.tflite` 与 `coco_labels.txt`，不依赖或启动
  default App runtime；
- exact production detector asset：
  - bytes：`5359428`
  - SHA-256：
    `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`
- 同一 ARCore frame 中依次取得 fresh raw depth/confidence、affine registration
  与 CPU camera image；
- stride-safe `YUV_420_888 -> RGBA` 后调用既有 CPU YOLO；
- controlled scene 固定要求 exactly one `person` detection，target key 固定为
  `manual-single-person`，不在多人场景做 post-outcome target selection；
- measurement `producedAtNs` 现在是 detector+sampling 完成时间，P95 不再漏算
  YUV conversion 与 detector latency；
- 每个距离输出 error、coverage、latency、7-point history availability 和失败类型；
- 不保存 camera image、depth raster 或 person box；只保留至多 1,800 个 depth/
  latency 标量供四距离精确合并，receipt 仍限制 `256 KiB`；
- 缺参数时该 instrumentation class `SKIP`，不会被通用 connected-test 意外烧毁；
  source/detector/registration 不可用仍为 `NOT_EVALUABLE_*`。

R0.4 没有增加支持门；它只实现已冻结的 1/2/3/5 m contract。

## R0.5 recoverable four-distance aggregation

在任何 device/person outcome 前，新增 host-only
`aggregate_stage_c_d45_phone_metric_depth_canary.py`，把协议中已经冻结的支持门
实现为一个小型、可测试的 reader：

- 只接受显式传入的最多 4 个 `<=256 KiB` receipt，不扫描 cohort 目录；
- strict UTF-8 JSON，拒绝 duplicate key、NaN/Infinity、超长数组和 size ceiling
  违规；
- 重新用 bounded depth/latency scalars 计算 per-distance error，不能盲信 receipt
  aggregate；
- 四个 receipt 必须绑定同一 device、target APK、instrumentation APK、camera、
  rotation、detector asset/backend；APK 内容哈希由 device runner 自身记录；
- overall error/latency 使用 pooled accepted observations；
  accepted-person coverage 使用
  `sum(accepted) / sum(exact-single-person frames)`；history availability 使用
  `sum(available forecasts) / sum(eligible windows)`；
- percentile 固定为 linear rank `(n-1)`，不在 outcome 后选择算法；
- `risk_feedback_invocation_count=0` 与 frozen baseline App SHA-256
  `afa7a774b9f47074b2bf2e59755e712e92421484140789513578b32b68f0f149`
  均纳入既有 non-interference gate；
- 10/10 host unit tests 覆盖支持、门失败、source 不可评估、缺距离、跨构建、
  summary 不一致、baseline 不一致、duplicate key、oversized receipt 和
  non-overwriting final write。

最关键的恢复语义：

- 缺少任一距离：
  `INCOMPLETE_DISTANCE_SET` + `scientific_terminal=null`；
- malformed/mismatched receipt：
  `CONTROL_PLANE_INPUT_REJECTED` + `scientific_terminal=null`；
- baseline artifact 缺失或不一致：
  `CONTROL_PLANE_BASELINE_*` + `scientific_terminal=null`；
- 这些状态只打印诊断，不创建最终 output，因此修复后可原路径重跑；
- 只有四距离完整且输入合法时，才允许写
  `SUPPORTED_DEVELOPMENT_ONLY`、`NOT_SUPPORTED` 或 source
  `NOT_EVALUABLE` 终态。

R0.5 不新增 gate，也不改变测量 outcome；它把控制面失败从科学负结果中显式拆开。

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
- source/registration/person benchmark APK：
  - path：
    `ustrf-shadow-benchmark/build/outputs/apk/debug/ustrf-shadow-benchmark-debug.apk`
  - bytes：`38966599`
  - SHA-256：
    `e9129cd71704f93ff9f8834821f8126b916a286a94db3ed999e4bb369753e195`
- source-decoder instrumentation APK：
  - path：
    `ustrf-shadow-benchmark/build/outputs/apk/androidTest/debug/ustrf-shadow-benchmark-debug-androidTest.apk`
  - bytes：`440221`
  - SHA-256：
    `27a33b1097cb46f09e14692bc7a24957dac9b86837dd784f5f606936c87c7a66`

新增 source-decoder build 命令：

```text
.\gradlew.bat :hftf-metric-depth-canary-core:test
  :ustrf-shadow-benchmark:assembleDebug
  :ustrf-shadow-benchmark:assembleDebugAndroidTest
  --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

结果为 `BUILD SUCCESSFUL`；JVM tests 为 `24/24`。

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
hftf-d45/raw-source-registration-r0/<run-id>/summary.json
```

`RAW_SOURCE_DECODER_OBSERVED` 只证明 source decode；
`AFFINE_REGISTRATION_OBSERVED_DEVICE_ONLY` 只证明同帧坐标映射内部一致。二者
均成立后才进入 1/2/3/5 m person measurement，仍不授权 event 或 App。
`AUTOMATIC_ONLY_*_CONFIDENCE_UNAVAILABLE` 与所有 `NOT_EVALUABLE_*` 都不是
depth 精度负结果；无设备也不是 source 负结果，不关闭 D45。

source+registration receipt 成立后，每个距离单独执行：

```text
.\gradlew.bat :ustrf-shadow-benchmark:connectedDebugAndroidTest
  -Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.ustrfbenchmark.D45ArCorePersonMeasurementCanaryTest
  -Pandroid.testInstrumentationRunnerArguments.hftfD45ReferenceDistanceMeters=<1|2|3|5>
  -Pandroid.testInstrumentationRunnerArguments.hftfD45PersonFrameAttempts=900
  --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

operator contract：人物 torso plane 到 camera optical center 保持声明距离，并轻微
平移手机以维持 ARCore depth；画面中只保留一个 person。receipt path：

```text
/sdcard/Android/data/com.linnan.blindassist.ustrfbenchmark/files/
hftf-d45/person-measurement-r0/<distance>m/<run-id>/summary.json
```

四个距离未全部执行前不产生 D45 支持/不支持总终态。

四个 receipt 拉回 host 后执行：

```text
python scripts/research/hftf/aggregate_stage_c_d45_phone_metric_depth_canary.py
  --receipt <1m-summary.json>
  --receipt <2m-summary.json>
  --receipt <3m-summary.json>
  --receipt <5m-summary.json>
```

默认同时核验：

```text
app/build/outputs/apk/debug/app-debug.apk
```

只有合法完整输入才会原子、非覆盖写入：

```text
artifacts.local/evidence/hftf/
stage-c-d45-phone-metric-depth-source-canary-r0/report.json
```
