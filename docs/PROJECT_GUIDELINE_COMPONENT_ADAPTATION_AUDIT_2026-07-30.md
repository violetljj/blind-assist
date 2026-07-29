# Project Guideline → BlindAssist 组件适配审计（2026-07-30）

状态：`COMPLETE / REFERENCE_ONLY / NO_IMPLEMENTATION_AUTHORITY`

适用范围：硕士论文与 BlindAssist 当前手机原型的轻量借鉴审计。

后续路线说明：用户于 2026-07-30 暂停 RCLE，并将
[神经—几何双环阶段−1准入](research/dual-loop/README.md)设为当前论文系统路线。
本审计的八项组件决定与 `NO_IMPLEMENTATION_AUTHORITY` 继续有效。

## 1. 结论先行

Project Guideline 值得借用，但应当借它已经验证过的**工程问题拆分方式**，不应迁移它的
整套 C++ / Bazel / MediaPipe / ARCore / Unreal 技术栈，也不应再建一套地图、状态机或
反馈系统。

本轮八项结论固定为：

| # | 候选能力 | 决定 |
| ---: | --- | --- |
| 1 | STOP / 失效语义 | `ADAPT` |
| 2 | 日志字段 | `ADAPT` |
| 3 | 回放与失效重算 | `ADAPT` |
| 4 | Occupancy Map | `REFERENCE` |
| 5 | 位姿与时间对齐 | `HOLD` |
| 6 | 深度尺度对齐 | `HOLD` |
| 7 | 音频反馈 | `REFERENCE` |
| 8 | Unreal 与预训练模型 | `DROP` |

三个 `ADAPT` 只形成两个后续候选改动：

1. **统一失效原因码**：把运行时、感知、时效性、后端降级与反馈不可用映射到一套
   稳定、可记录的原因码；保留现有状态机，不新增第二状态机。
2. **最小可重算证据账本**：把最小日志字段与确定性回放合并设计，使一次手机会话的
   关键决定可以离线重算；不默认保存原始图像，不引入 protobuf 或 Project Guideline
   的日志实现。

这两个候选在本审计中**只获得设计建议，不获得代码实施权限**。

## 2. 证据基线与适用边界

### 2.1 外部基线

外部参考固定为 Google Research 官方仓库
[`google-research/project-guideline`](https://github.com/google-research/project-guideline/tree/b5fa173de36ab591d875492a899358cdc5843291)，
提交 `b5fa173de36ab591d875492a899358cdc5843291`（2026-02-20 UTC）。本审计不跟随
`main` 后续变化自动更新。

Project Guideline 自身明确限定为研究演示：面向 Pixel 6/7/8、腰部固定手机、特定紫色
地面引导线和有视力陪同者，并声明不能覆盖所有场景或障碍
（[README：要求与安全提示](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/README.md#L213-L244)）。
因此它能证明“这些组件在其限定系统中有工程价值”，不能直接证明它们适合 BlindAssist
的手持/手机摄像、开放环境、YOLO 风险提醒或真实助行安全。

### 2.2 BlindAssist 当前边界

- 正式手机链路已有 CameraX、YOLO、统一决策内核、语音/震动反馈和离线图片回放。
- Snapdragon 8 Gen 3 / SM8650 的 QNN HTP 路由已是
  `PROMOTED_WITH_CPU_FALLBACK`；同机延迟和持续性能证据成立，但能耗优势尚未测得
  （[NPU_DEFAULT_CANDIDATE.md](NPU_DEFAULT_CANDIDATE.md)）。
- USTRF 中已有结构化风险场、STOP/SCAN/SLOW_DOWN、失效原因和确定性故障回放，
  但仍是生产隔离/影子能力；既有 route-conditioned program 已关闭，不得借
  Project Guideline 之名重启
  （[USTRF 收口 R1](research/ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)）。
- 本审计形成时 RCLE 仍是论文算法研究主线；后续用户已将其暂停。既有结论上限仍不
  产生 Android、产品或安全权限
  （[RCLE 当前真源](research/rcle/README.md)）。

## 3. 八项逐项审计

### 3.1 STOP / 失效语义 — `ADAPT`

**Project Guideline 的做法。** `ControlSignalLite` 把 `stop` 与类型化
`StopReason` 一起传递，现有原因包括应用主动停止、跟踪状态变化、连续无关键点、
横向量异常和无有效运动
（[`control_signal.proto`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/proto/control_signal.proto#L25-L67)）。
跟踪丢失和无关键点超时会清空环境并发出 STOP
（[`guidance_system.cc`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/environment/guidance_system.cc#L257-L376)）。

**BlindAssist 已有对应。**

- [`AssistRuntimeStateMachine.kt`](../feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistRuntimeStateMachine.kt)
  已处理启动、运行、暂停、权限拒绝和相机错误。
- [`FeedbackModels.kt`](../core/assist/src/main/java/com/linnan/blindassist/feedback/FeedbackModels.kt)
  已有类型化反馈原因。
- [`UstrfContracts.kt`](../core/ustrf/src/main/java/com/linnan/blindassist/ustrf/UstrfContracts.kt)
  已有 `STOP_AND_REASSESS` 及感知过期、位姿丢失、坐标帧不匹配、热降级等原因。

**真实缺口。** 正式运行时错误仍主要是自由文本 `Error(message)` /
`CameraSourceFailed(message)`；USTRF 原因、反馈原因、相机错误和 NPU 回退原因彼此
分散，无法稳定聚合“为什么本帧没有给出提醒”。

**允许的借鉴。** 后续仅增加一层稳定原因码与现有类型的映射；STOP/暂停/错误仍由
现有状态机执行。原因码至少区分：

- 用户主动停止；
- 权限、相机、模型或后端不可用；
- 输入过期、时钟域未知、坐标帧不匹配；
- 感知证据不足或质量降级；
- 热状态降级；
- 反馈设备不可用或冷却抑制。

**停止线。** 不复制 Project Guideline 的 ControlSystem，不把 USTRF 影子 STOP 接成
用户导航命令，不声称类型化原因码提升了安全性。

### 3.2 日志字段 — `ADAPT`

**Project Guideline 的做法。** 其 protobuf 事件同时携带墙钟时间、帧时间戳和
跟踪、位姿、检测、世界更新、控制信号、点云、占用图、深度等事件类型
（[`guideline_log.proto`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/proto/guideline_log.proto#L27-L147)）；
文件 logger 以流式 protobuf 写盘
（[`file_guideline_logger.cc`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/logging/file_guideline_logger.cc#L37-L78)）。

**BlindAssist 已有对应。**

- [`VisionFrame.kt`](../core/assist/src/main/java/com/linnan/blindassist/vision/VisionFrame.kt)
  已记录帧 ID、采集/接收时刻、来源、坐标帧和时钟域。
- [`SessionTrace.kt`](../core/assist/src/main/java/com/linnan/blindassist/session/SessionTrace.kt)
  已保留最近 30 帧的风险、延迟和反馈摘要。
- [`AssistRuntimePerformanceLogger.kt`](../feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistRuntimePerformanceLogger.kt)
  已按秒输出延迟、丢帧、风险、反馈和模型状态。
- [`FramePipelineStats.kt`](../feature/assist/src/main/java/com/linnan/blindassist/runtime/FramePipelineStats.kt)
  已区分 busy、inactive 和 detector unavailable 丢帧。

**真实缺口。** 当前信息主要是内存摘要和 Logcat 文本，不是版本化、会话级、可持久化
的事件合同；后端路由/回退、时效判定、失效原因与反馈结果也没有统一关联到同一帧。

**允许的借鉴。** 只设计最小账本字段，不照搬“大而全”的 proto：

`schemaVersion, sessionId, frameId, capturedAtNs, receivedAtNs, decisionAtNs,
clockDomain, coordinateFrame, inputSource, detectorBackend, backendRouteReason,
configDigest, freshness/validUntil, dropOrFailureReason, riskAction/reasons,
speechAccepted, vibrationAccepted, preprocess/inference/postprocess/totalMs`。

温度、电量和功耗若进入论文实验，应作为有采样周期与设备信息的 session 观测，不伪装成
每帧精确功耗；原始图像默认不写入账本。

**停止线。** 不引入 protobuf、数据库、遥测服务或新的运行依赖；先用项目现有能力能
表达的稳定 schema。隐私、容量、轮转和导出策略未冻结前，不允许默认持久化。

### 3.3 回放与失效重算 — `ADAPT`

**Project Guideline 的做法。** 官方只明确承诺将事件文件加载后做详细调试和分析
（[README：Logging](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/README.md#L367-L370)）；
本审计未发现官方声明“生产日志可完整重算所有控制结果”。因此这里借的是
“事件可追溯”原则，不扩大为不存在的全量 replay 能力。

**BlindAssist 已有对应。**

- [`ReplayFrameSource.kt`](../core/device/src/main/java/com/linnan/blindassist/camera/ReplayFrameSource.kt)
  能让固定图片走同一帧回调。
- [`AssistDecisionKernel.kt`](../core/assist/src/main/java/com/linnan/blindassist/session/AssistDecisionKernel.kt)
  让生产与 benchmark 复用同一决策顺序。
- [`UstrfControlledFaultReplay.kt`](../core/ustrf/src/main/java/com/linnan/blindassist/ustrf/UstrfControlledFaultReplay.kt)
  已能显式注入位姿、采集、几何、运动、帧不匹配和过期故障。
- [`UstrfSessionTraceDigest.kt`](../core/ustrf/src/main/java/com/linnan/blindassist/ustrf/UstrfSessionTraceDigest.kt)
  已能对结构化结果和失效原因生成确定性摘要。

**真实缺口。** 正式手机会话没有持久化全部“重算决策所必需且允许保存”的输入、配置
版本和原因码；现有图片回放证明管线可复用，现有 USTRF 回放证明故障可确定重现，但二者
尚未形成正式手机链路的统一可重算证据。

**允许的借鉴。** 与 3.2 合并成一个候选改动：先定义账本，再用已存在的
`AssistDecisionKernel` 做确定性重算测试。若缺少原始检测输出、配置摘要或版本标识，
重算结果必须标记 `NOT_RECOMPUTABLE`，不得补值或推测。

**停止线。** 不新建第二套业务管线，不把合成故障回放当真实用户或安全证据。

### 3.4 Occupancy Map — `REFERENCE`

**Project Guideline 的做法。** 它把深度点云按人相对坐标更新为 frame-based
occupancy map，并在当前轨迹清空区检查障碍
（[`occupancy_map.h`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/environment/occupancy_map.h#L31-L57)；
[`guidance_system.cc`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/environment/guidance_system.cc#L148-L244)）。

**BlindAssist 已有对应。** [`UstrfRiskField.kt`](../core/ustrf/src/main/java/com/linnan/blindassist/ustrf/UstrfRiskField.kt)
已经是用户局部栅格，包含 occupancy、traversability、drop/head risk、TTC、不确定性、
年龄、来源、衰减和可选的已验证位姿 warp。

**真实缺口。** 缺的是可准入的手机端 metric geometry / pose 输入，不是“缺一张地图”。
两者的安装姿态、场景假设、感知输入和输出权限不同，不能把 Project Guideline 的
OccupancyMap 结果当成 BlindAssist 的迁移验证。

**决定与停止线。** 只参考其测试分层和“未知/失效时不沿用旧环境”的思想；不复制实现，
不建立第二个 occupancy map，不重启已关闭的 route-conditioned USTRF program。

### 3.5 位姿与时间对齐 — `HOLD`

**Project Guideline 的做法。** ARCore 在同一帧时间轴提供相机 pose 和 3D tracking
features；GuidanceSystem 对 camera pose 要求对应时间戳，对 features 选择精确或最近
样本
（[`arcore_motion_tracker.cc`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/android/arcore/arcore_motion_tracker.cc#L127-L175)；
[`guidance_system.cc`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/environment/guidance_system.cc#L51-L98)）。

**BlindAssist 已有对应。** CameraX 已记录 capture/receive 时间、时钟域和坐标帧；
[`UstrfVioPoseReceipt.kt`](../core/ustrf/src/main/java/com/linnan/blindassist/ustrf/UstrfVioPoseReceipt.kt)
已预留 fail-closed VIO 准入合同，要求帧匹配、稳定世界系、时效、置信度、倾角和独立验证
的相机到机身外参。

**真实缺口。** 当前正式手机链路没有已准入的 ARCore/VIO adapter、稳定 world-frame
政策或独立外参收据；现有合同只是边界，不是设备能力证明。

**进入条件。** 只有当一个已冻结的手机相机实验明确需要跨帧 metric warp，且能提供
同源时间戳、稳定世界系、跟踪状态、外参、丢跟踪重置规则与离线重放时，才可另立小型
adapter 评估。

**停止线。** 本审计不添加 ARCore / MediaPipe 依赖，不把 Project Guideline 的腰挂
相机外参迁移到手持手机，不解锁产品或用户反馈权限。

### 3.6 深度尺度对齐 — `HOLD`

**Project Guideline 的做法。** 它用 ARCore 3D tracking features 与相对 ML depth 做
RANSAC，估计 `scale + shift` 后再生成世界点云
（[`depth_align_ransac.h`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/depth/depth_align_ransac.h#L25-L40)；
[`guidance_system.cc`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/environment/guidance_system.cc#L381-L430)）。

**BlindAssist 已有对应。** 已有相对深度 TFLite 候选和 benchmark 脚手架，但它不是
metric depth。既有 MiDaS 真机 A/B 虽把关键漏报 `9 → 7`，却把误提醒率
`0.037 → 0.185`，总 P50/P95 `54/56 ms → 276/292 ms`，结论为
`do_not_promote_depth_fusion`
（[DETECTOR_BENCHMARK.md](DETECTOR_BENCHMARK.md#单目深度融合候选路线)）。

**真实缺口。** 缺少与手机相机同帧、已准入的稀疏 metric 3D 特征；没有它，复制
RANSAC 只会产生形式上的 scale，而不是可信尺度。

**进入条件。** 只有 3.5 的 pose/feature 输入先通过准入，且论文确实需要 metric depth，
才允许一个隔离、小样本、预冻结阈值的 `relative depth vs aligned depth` 实验。

**停止线。** 不纳入论文必做项，不接正式提醒，不以 Project Guideline 结果替代本机
准确率、误提醒、延迟和热/功耗测试。

### 3.7 音频反馈 — `REFERENCE`

**Project Guideline 的做法。** AAudio + 空间化 sound pack 根据 ControlSignal 提供
低延迟连续引导，并把初始化、ready、guiding、stopping 分成明确声音状态
（[README：Audio System](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/README.md#L353-L360)；
[`audio_system.h`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/audio/audio_system.h#L45-L135)）。

**BlindAssist 已有对应。** 当前
[`FeedbackController.kt`](../core/device/src/main/java/com/linnan/blindassist/feedback/FeedbackController.kt)
已有语音、震动、冷却、疲劳控制和“实际是否接受反馈”的回执；任务是离散风险提醒，
不是沿固定线持续转向。

**真实缺口。** 当前没有空间化连续音频，但这不是已证明的论文主线缺口。更直接的缺口是
系统失效时是否给出可区分、可审计且不过度打扰的反馈；它已被 3.1 的统一原因码覆盖。

**决定与停止线。** 仅参考“正常提醒与系统失效提示必须可区分”的设计原则。不得复制
sound pack，不引入 AAudio/Resonance Audio，不重构现有语音/震动系统；若未来研究声音，
必须先有人因协议和真实反馈证据。

### 3.8 Unreal 与预训练模型 — `DROP`

**Project Guideline 的做法。** Unreal 4.27 simulator 通过共享 C++ 库/插件注入渲染图像
和 ground-truth pose，但官方说明障碍检测尚未完整支持
（[`unreal/README.md`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/unreal/README.md#L1-L26)）。
其模型面向紫色引导线分割和单目深度
（[`vision/models/README.md`](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/project_guideline/vision/models/README.md#L1-L17)）。

**BlindAssist 已有对应。** 当前论文问题、数据协议、YOLO/QNN 手机链路和 RCLE 几何
研究均不依赖紫色引导线或 Unreal；加入它们会形成新的数据域、构建系统和验证任务。

**决定与停止线。** 本论文周期 `DROP`：不下载/集成模型，不建立 Unreal 工程，不接
C++/JNI/Bazel/MediaPipe 构建链，不把合成仿真当真实场景证据。

## 4. 两个后续候选改动的最小验收条件

以下只是未来另行授权时的验收边界。

### 候选 A：统一失效原因码

必须同时满足：

1. 只新增原因码及现有状态/原因的映射，不新增状态机；
2. 至少覆盖相机失败、模型/后端不可用、时钟/帧/时效问题、证据不足、热降级、
   反馈不可用和用户主动停止；
3. 每个原因有稳定机器值与用户显示文案，二者分离；
4. CPU fallback 必须保留真实原因，不能记成 NPU 成功；
5. 默认行为、提醒阈值和 USTRF/RCLE 权限不改变。

### 候选 B：最小可重算证据账本

必须同时满足：

1. 先冻结 schema、隐私、容量、轮转、导出和版本策略；
2. 单个记录能关联 source frame、配置/模型/后端、输入时效、风险决定、原因和反馈回执；
3. 固定输入与固定配置重复重算得到一致决定或显式 `NOT_RECOMPUTABLE`；
4. 丢失必要字段时 fail closed，不补值、不用摘要冒充原始输入；
5. 不引入 Project Guideline 的 protobuf/C++ logger，不默认保存图像；
6. 账本故障不能阻塞相机或反馈主链。

## 5. 许可证、成本与采用规则

- Project Guideline 根仓库以
  [Apache-2.0](https://github.com/google-research/project-guideline/blob/b5fa173de36ab591d875492a899358cdc5843291/LICENSE)
  发布；本轮只阅读和引用，没有复制代码。
- 若未来复制或改写代码，必须记录精确提交、原文件、修改说明并保留许可证/NOTICE 要求；
  这需要独立的代码级许可证审计。
- 仓库内的模型二进制、声音素材、第三方库、ARCore/MediaPipe SDK、Unreal Engine 与
  Marketplace 资产不能因为根代码许可证就自动视为可再分发；每项采用前单独核验。
- 成本按当前范围排序：原因码映射 `S`；最小账本与重算测试 `M`；ARCore/VIO 与深度尺度
  对齐 `L`；Unreal/全栈迁移 `XL`。当前只推荐前两项。

## 6. 论文主线与明确不做

本审计支持的论文工程叙事仍是：

> 当前 BlindAssist 主链 + 一个经证据准入的轻量几何信号 + YOLO + 既有风险/时间机制
> + 异步新鲜度、质量与失效管理
> + 真机事件、延迟、热与功耗实验。

其中“轻量几何信号”按
[双环阶段−1准入合同](research/dual-loop/BLINDASSIST_DUAL_LOOP_PHASE_MINUS1_ADMISSION_CONTRACT_R0_2026-07-30.md)
顺序验证；当前默认候选为已有 Sparse LK，RCLE 已暂停且不是阶段−1依赖。本文件不把
任何几何信号写成已经验证。简单三态或其他算力调度仅是阶段−1之后、由实测热/功耗
证据决定的未来候选，当前没有实施权限。

明确不做：

- 全量迁移 Project Guideline；
- 第二套状态机或第二张占用图；
- Bazel / MediaPipe / C++ / JNI 合并；
- ARCore、深度尺度或 Unreal 自动解锁；
- 多安装姿态、开放世界导航或安全产品化；
- 将 SM8550/其他 SoC 自动纳入现有 SM8650 路由；
- learned scheduling；
- 用外部项目的演示结果替代 BlindAssist 的设备实验。

## 7. 完成检查

- [x] 八项候选全部给出唯一决定；
- [x] 每项均指出 BlindAssist 现有对应和真实缺口；
- [x] 立即推荐的后续改动不超过两个；
- [x] ARCore / 深度 / Unreal 未获得实施权限；
- [x] 未新增 package、依赖、模型、数据或运行链；
- [x] 未改算法、提醒阈值、NPU 路由或研究终态；
- [x] 外部证据绑定官方固定提交；
- [x] 结论没有超出 Project Guideline 与 BlindAssist 各自证据边界。
