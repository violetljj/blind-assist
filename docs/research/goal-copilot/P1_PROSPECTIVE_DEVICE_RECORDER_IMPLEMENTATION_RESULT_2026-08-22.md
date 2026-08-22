# P1 prospective device recorder implementation result

状态：`IMPLEMENTED_AND_BUILT / REAL_DEVICE_COHORT_NOT_CAPTURED / PROVIDER_CALLS=0 / PA3_INFERENCE_NOT_AUTHORIZED`

## 结论

prospective first-person capture 不再缺 device-owned producer。新增的独立 Android 模块 `:goal-capture-app`
使用 CameraX rear-camera video capture 消费 host 端预先冻结的完整 `capture_plan.json`，并只在所有 episode 的
异步 `Finalize` 成功、媒体 metadata/timeline/SHA-256 校验完成后生成 `physical_capture_receipt.json`。
冻结 plan 可由系统文件选择器导入；App 在写入自己的 session 前先验证合同与摘要，并原子替换 inbox 副本，因此不依赖
ADB 才能启动真实采集。

它没有修改默认 BlindAssist App，也没有读取 private truth、运行 proposal provider 或授予 PA3。真实手机尚未执行该
计划，因此本结果只关闭 recorder implementation seam，不建立真实 cohort。

## Fail-closed 边界

- plan 必须含至少 5 个 episode，并通过 body hash、C0-binding hash、全局 instruction、固定文件名与固定抽帧 offset 校验；
- rear-camera 视频不录音，按冻结 roster 顺序录制，已有目标文件一律拒绝覆盖；
- `VideoRecordEvent.Finalize` 是唯一完成边界；cancel、lifecycle interruption、recorder error 或 partial roster 只写
  `capture_hold.json`，不得写正式 receipt；
- 每段视频必须为 `3–45 s`，分辨率有效，设备 wall timeline 与媒体 duration 误差不超过 `1 s`；
- 正式 receipt 绑定 goal receipt、capture plan、每段媒体 SHA-256，且不含 `pa3_inference_authorized`；
- host 端 `materialize_capture.py` 仍会独立复核 chronology、hash、单视频流、metadata 与 outcome-blind fixed-offset extraction。

## 验证

- `:goal-capture-app:testDebugUnitTest`：通过，5 项 pure contract tests；
- `:goal-capture-app:assembleDebug`：通过，生成独立 debug APK；
- Kotlin 完整 sample receipt body SHA-256 与 Python `content_sha256` 一致：
  `0e5c857e19b053f5bbed7ad103cdc80b4b40eae359cc1c25b76f616402388c71`；
- C0 / prospective capture / proposal availability 聚焦回归：`5 + 13 + 38` 项通过；
- 全 Goal Copilot 广域 discovery 另有一个既有 `p1_w1_stage_a` 裸导入路径错误，不属于本次变更面，未用它替代聚焦结果。

本机默认 JDK 26 超出当前 Gradle/Kotlin DSL 解析能力；验证固定使用已有 `E:\codex-tools\jdk-17`，没有修改项目
科学合同。

## 下一门

下一动作只是在真实第一视角设备上执行完整 frozen roster，导出 ZIP，并由 host materializer 生成 outcome-blind frames。
随后才创建 private truth。只有冻结门同时达到 `>=5 visible episodes / >=8 visible frames`，才可授权一次 frozen PA3
goal-semantic proposal availability run；否则继续保持 `NOT_EVALUABLE_INPUT_CONTRACT`，不得运行模型或补写 truth。
