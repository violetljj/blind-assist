# HFTF Stage C D35：Android device shadow parity/runtime canary

日期：2026-08-03

证据角色：Development / isolated Android device implementation canary

研究主线：不变

默认 App：不变

## 问题

D34 已在 host JVM 上证明 production Kotlin
`CausalTrackTristateGeometryProducer` 与 D33 Python source rule 零漂移。D35
只把相同 source-only corpus 搬到真实 Android runtime，回答：

1. isolated `dualLoopShadow` variant 的 build flags 是否正确；
2. 5,366 条真实 detector-track occurrences 在设备上是否仍逐条 parity；
3. production producer 的 device P50/P95/P99 是否足够低；
4. shadow mode 是否保持 baseline risk/event/feedback frame-exact 不变。

## 冻结输入

- D34 `parity_input.tsv` 原始 SHA-256：
  `d1f24dc7c61890e912d2a4a1cbca23e4b729dfceb1ef76b435cd573c97e6021e`；
- 5,366 rows、165 tracks；
- 以 gzip payload（`.gzbin`，避免 Android asset pipeline 自动解包）打包到独立
  `hftf-device-canary` test APK；
- 不含 annotation association、native identity、3D range 或 future truth。

## 冻结执行

- target variant：`dualLoopShadow`；
- target application id：`com.linnan.blindassist.dualloop.shadow`；
- 必须 `DUAL_LOOP_SHADOW=true`；
- 必须 `DUAL_LOOP_ACTIVE=false`；
- production/default App 不安装、不修改；
- parity corpus 第一遍 warm-up、第二遍逐 call 计时；
- track gap 显式 reset；
- non-interference 使用 production `AssistDecisionKernel`，同一七帧 growing-person
  series 分别运行 OFF 与 `SHADOW_ABSTAIN_ONLY`；
- shadow 必须 admitted，但 risk、stable risk、event、feedback decision、session
  summary 与 gateway call count 必须和 OFF 完全相同。
- device report 使用 Android `AtomicFile` interruption-safe 写入；报告写入失败属于
  repairable engineering failure，不得解释为科学负结果。

## gate

1. device instrumentation target flags 正确；
2. corpus rows = 5,366；
3. decision mismatch = 0；
4. slope presence mismatch = 0；
5. max absolute slope error <= `1e-5/s`；
6. producer device P95 <= `0.10 ms`；
7. shadow admitted 且 event/feedback mutation flags 均 false；
8. baseline/shadow risk-event-feedback-session-gateway 全部 frame-exact；
9. instrumentation test task PASS。

通过：

`D35_ANDROID_DEVICE_SHADOW_PARITY_RUNTIME_NONINTERFERENCE_SUPPORTED`

未通过：

`D35_ANDROID_DEVICE_SHADOW_PARITY_RUNTIME_NONINTERFERENCE_NOT_SUPPORTED`

无设备、ADB、安装、授权、Gradle、路径、asset 或 instrumentation transport 失败均为
engineering/availability failure，允许修复重跑，不构成科学负结果。

## 主张边界

通过只建立 isolated test APK 中 production Kotlin state 与 decision seam 的设备级
parity、runtime 和 non-interference。它仍不建立 live CameraX detector-track
coverage、真实走路 event utility、提醒增量、默认 App、产品效果或 human safety。

下一步才是在 `dualLoopShadow` build 上做 bounded live-camera shadow census；仍不
驱动提醒。
