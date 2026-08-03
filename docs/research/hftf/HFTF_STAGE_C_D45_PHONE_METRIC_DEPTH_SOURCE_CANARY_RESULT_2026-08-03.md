# HFTF Stage C D45 phone metric-depth source canary result

日期：2026-08-03

终态：`D45_PHONE_METRIC_DEPTH_SOURCE_NOT_EVALUABLE`

## 结论

当前 `SM-S9280 / Android 16 / ARCore 1.33.0` context 没有提供足以进入人物测距的
fresh raw-depth stream。能力声明为 supported，但在显式受控平移和纹理场景中，900
次 session update 得到 844 个 tracking frame、864 个 distinct camera timestamp，
exact-timestamp raw-depth observation 为 0。

因此停止在 source admission，不执行 1/2/3/5 m person measurement。这个终态不否定
depth decoder、registration、person sampler、7-point history solver 或 D44 的算法
假设，也不授权 App、risk、feedback 或主线变更。

更重要的是，预期最终摄像头是普通外接 RGB camera，不能假设具备 ARCore 或硬件深度。
所以即使 D45 在手机本机摄像头上通过，它也只能成为开发期 teacher/diagnostic bridge，
不能成为 HFTF 在线核心。运行时边界由
[external-camera decision](HFTF_EXTERNAL_CAMERA_RUNTIME_BOUNDARY_R0_2026-08-03.md)
接管。

## 设备与执行绑定

- serial：`192.168.5.4:41487`（无线 ADB，仅为执行地址）；
- model/device：`SM-S9280 / e3q`；
- Android：`16 / SDK 36`；
- build fingerprint：
  `samsung/e3qzcx/e3q:16/BP4A.251205.006/S9280ZCS6DZF2:user/release-keys`；
- device project test：
  `D45ArCoreRawSourceDecoderCanaryTest`，debug benchmark + debug androidTest；
- R4 motion protocol：`OPERATOR_CONTROLLED_TRANSLATION_TEXTURED_SCENE`；
- R4 frame attempts：`900`；instrumentation：`OK (1 test)`。

执行前 focused verification：

```text
:hftf-metric-depth-canary-core:test
:ustrf-shadow-benchmark:assembleDebug
:ustrf-shadow-benchmark:assembleDebugAndroidTest
BUILD SUCCESSFUL; 89 tasks: 4 executed, 85 up-to-date
```

## Source evidence

R4 capability fields：

- ARCore availability：`SUPPORTED_INSTALLED`；
- automatic depth：`true`；raw-depth-only：`true`；
- camera configs：`3`；hardware-depth configs：`0`；
- camera id：`0`；sensor orientation：`90`；detector rotation：`90`。

R4 acquisition fields：

- attempted updates：`900`；tracking frames：`844`；
- distinct camera timestamps：`864`；
- fresh raw-depth observations：`0`；
- failure counts：`DEPTH_TIMESTAMP_MISMATCH=844`；
- registration observations：`0`，因为没有 fresh source frame；
- receipt SHA-256：
  `27b58a8bb0491f84edce5674df7d99d4ba9a1d6aea1ed12356dbb7f95a5e4b66`。

ignored host evidence path：

```text
artifacts.local/evidence/hftf/stage-c-d45-phone-metric-depth-source-canary-r0/
raw-source-controlled-r4-summary.json
```

R2 曾在 300 次 update 中观察 18 个 fresh raw frame，但 valid pixels 为 0。R3 加入
depth/confidence 独立诊断后只观察 2 个 fresh raw frame，28,800 个 decoded pixels 的
depth 与 confidence 全为 0；receipt SHA-256：
`69453f0fe2e51057e5c211fd0eb20bf68bece09e23790999ced89d8cd92b42c6`。
R4 没有用 R2/R3 的偶发旧结果替代受控 source admission。

## 控制面修正

原 `:hftf-device-canary` 是 `com.android.test`、target 为默认 App；默认 App manifest
有意不声明 ARCore。把 test APK 自身的 optional metadata 当成 target App capability
上下文会得到无效的 `UNKNOWN_ERROR`，因此该重复 capability path 被撤销，而不是继续
增加 lifecycle 对齐和额外 manifest 分支。

capability、camera config、freshness、decoder 与 registration 现在由合法的 isolated
ARCore optional benchmark context 在同一 receipt 中给出。默认 App 仍不依赖 ARCore，
D35 device canary 仍留在 `:hftf-device-canary`。

## 恢复边界

当前停止的是这个 device/build/source context。恢复需要新的 source context，例如：

- 不同物理设备，尤其实际报告 hardware-depth camera config 的设备；
- 明确变更后的 ARCore/runtime 版本；
- 新的、预先冻结且仍能保持 frame timestamp truth 的 metric-depth source。

禁止放宽 exact timestamp 条件、接受 reprojected stale depth、改变 quality threshold，
或在本次已见 outcome 后用人物距离实验反向挑 source。主线保持不变。
