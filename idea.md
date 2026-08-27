# BlindAssist 待决方向

状态：current（只记录尚未决定的方向，不授予研究或产品权限）

本页是待决想法的短入口，不复制动态实验数字、执行计划或历史流水。旧记录完整保留在 [想法池历史归档](docs/history/idea/IDEA_ARCHIVE_THROUGH_2026-07-28.md)；当前产品事实以 [README.md](README.md) 为准，当前双环、SANPO 与暂停的 RCLE 状态以各自 current 文档为准。

## 维护规则

- 只有用户明确要求保留，或团队明确延后一个非平凡方向时，才新增条目。
- 每条只写问题、尚未验证的假设、进入实施前的门槛和权威入口。
- 一旦采用、拒绝或完成，就把决定写入对应 current/decision/snapshot 文档并从本页移除；不要在这里追加实验流水。
- 本页任何内容都不开放训练、formal claim、Android 接入、默认模型替换或产品安全表述。

## 当前待决

### 反事实风险事件与生命周期建模

- 问题：单帧/像素指标不能直接回答“同一真实风险事件是否及时提醒且只提醒一次”。
- 假设：具备来源隔离、matched negative、完整生命周期标注的事件级数据，可能比继续调单帧 head 更能改善误提醒与重复提醒。
- 实施门槛：先按 [SANPO 当前状态](docs/SANPO_CURRENT_STATUS.md) 和 [反事实采集协议](docs/SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md) 关闭许可、来源隔离、复核与数据覆盖门；未闭合前只允许协议/接口工作。

### TwinScene 与 AC4D 后续方向（含 TARO 迁移指针）

- 状态：`TwinScene 与 AC4D 待决 / GaugeFix+PARA 已迁出为 TARO 独立并行支线`。
- 共同问题：继续换 depth backbone 或直接学习 final clearance，无法区分尺度、support、boundary、
  可观测性和动态预测究竟哪一层提供信息增益；新方向必须先用 oracle/反事实把机制拆开。
- `GaugeFix` 想法：在有效 K/crop/rotation/resize receipt 和独立米制锚之后，只估计会改变
  body-swept clearance 的低维残余 gauge posterior，并以 TSVD/observable mask 保留不可观测方向；
  RGB-only、纯旋转或无 metric anchor 时必须保持 UNKNOWN。
- `PARA` 想法：先复用合法 ring-buffer 历史，再在身体保持静止时选择 yaw/pitch 或手机左右微基线，
  直接优化 task-query uncertainty/risk reduction；不得退化成通用 NBV、时序平滑或要求用户迈步。
- 采用决定：GaugeFix 与 PARA 已合并为 **TARO — Task-directed Active Risk Observability**，
  不再作为两个待决子路线；当前入口为 [TARO current](docs/research/taro/README.md)，详细命题、
  数学定义、数据合同、实验阶梯、强基线与 kill gates 见
  [TARO R0 路线指南](docs/research/taro/TARO_R0_RESEARCH_ROUTE_GUIDE_2026-08-10.md)。
- `TwinScene` 待决想法：构造 `真实基线 → 同 pose 数字基线 → 单变量 3D 干预` 三元组，
  以 exact depth/support/boundary treatment effect 做 factor finite-difference supervision；只有通过
  collider/render parity、effect-mask 外 artifact probe、跨 renderer/asset/site 和少量真实物理 pair
  审计后，才可另立离线数据/蒸馏路线。它不能把 synthetic effect 写成真实因果。
- `AC4D` 待决想法：先预测与候选 wearer path 无关的 stochastic future metric world belief，再由
  deterministic 4D body tube 查询任意 path/profile 的 first-contact survival；必须先在加速/转向、
  遮挡重现、多目标和 1.5–3 秒困难分层上以 oracle 超过 D44 + Kalman/IMM，普通 1 秒 ADE 改善不足以立项。
- 依赖关系：TwinScene 未来只能作为 TARO 的可选离线监督；TARO 未来只能在独立 oracle 通过后向
  AC4D 提供 current metric posterior；三者不得互相背书或一次性组成无法归因的大系统。
- 当前权限：TARO 只获得路线设计与唯一 P0 protocol-lock successor，execution=false；TwinScene、
  AC4D 没有 active route、数据、实现、训练或 outcome 权限；默认 App、产品与 safety 均不变。

### 真实眼镜终端

- 问题：当前正式能力仍以手机摄像头为主，模拟中心不等于真实硬件连接。
- 假设：先建立 `GlassesFrameSource`、设备状态与反馈适配边界，再以 USB 有线样机验证，能避免过早把 BLE、视频、音频和 UI 耦合。
- 实施门槛：以 [眼镜硬件路线](docs/GLASSES_HARDWARE_ROUTE.md) 为 current 入口；具备真实硬件、协议、断连降级、功耗/热与隐私验证后才进入实现。

### 答辩/课堂轻量展示包

- 问题：临场演示需要稳定材料，但不应重新污染日常主流程或把历史验证当成当前事实。
- 假设：文档、外部 APK 归档、debug-only 回放和少量授权截图已足以构成轻量展示包。
- 实施门槛：只有在明确的提交或答辩需求出现时，按 [演示指南](docs/DEMO_GUIDE.md) 生成一次性材料并重新验证版本、构建和设备状态。

### DA3 × Metric3D GeoMetric-Lite 双教师轻量几何

- 状态：`待评估 / 候选算法支线`
- 主张：DA3 的空间结构/跨帧几何与 Metric3D 的米制尺度可能互补，可蒸馏为移动端稠密几何学生。
- 保留理由：若教师互补性成立，可能改善 clearance、地面/障碍几何和时序稳定性；但必须先证明互补性，再购买双教师训练复杂度。
- 进入门槛：先做固定数据上的教师互补性、单帧/时序对照、disagreement 与 error 关系及 kill gate；通过后才能建立新版本研究路线。
- 禁止动作：未通过 R0 证据门前不得直接开始完整双教师训练、替代 DepthART/DA2、接入 Android 或改变默认 App。
- 原始方案：[BlindAssist_DA3_Metric3D_GeoMetric-Lite_R0.1.md](D:/edge/BlindAssist_DA3_Metric3D_GeoMetric-Lite_R0.1.md)

### DepthART R1 QAIRT/QNN/HTP 部署准入

- 状态：`待推进 / 已存在主线候选的部署工作包`
- 主张：DepthART-S 可沿 numerical parity、QAIRT graph compatibility、SelectiveScan lowering、HTP graph/runtime/performance 分层验证，争取成为可部署候选；R0 FAIL 必须保持不变。
- 保留理由：这是当前 DepthART 主线的部署与准入执行思路，能够把“算法收益”和“部署可行性”分开管理。
- 进入门槛：按 R1 顺序完成原始 ONNX parity、等价图改写 parity、Camera Embedder 处理、QAIRT 转换、SelectiveScan 实际判定、HTP smoke 和性能；任何阶段未闭合都不能写成 HTP PASS/FAIL 或生产替换。
- 禁止动作：不得修改 R0 FAIL、把转换阻塞提前写成 HTP FAIL、把单机性能当作准确率/安全证据、或直接接入默认 App。
- 原始纲领：[BlindAssist_DepthART_R1_QAIRT_QNN_HTP_部署与准入执行纲领_2026-08-07.md](D:/edge/BlindAssist_DepthART_R1_QAIRT_QNN_HTP_部署与准入执行纲领_2026-08-07.md)

### 异步慢快感知、流式记忆与潜在未来风险

- 状态：`待评估 / 系统与算法交叉想法`
- 主张：通过 slow/fast 异步感知、跨帧 streaming memory、轻量 latent future predictor、uncertainty gate 和 dynamic compute，可能在移动端降低平均计算成本并提升动态风险感知时效性。
- 保留理由：它不是单一模型替换，而是可拆成独立的时序、数据、延迟和性能研究，适合在 DepthART 周边并行验证。
- 去重边界：本条只保留 slow/fast scheduling、streaming memory 与 dynamic-compute 系统想法；
  task-query active observation 已迁入 TARO，4D first-contact/world-belief 算法已由上面的 AC4D 待决项承载。
- 进入门槛：先做单变量、可停止的最小实验：异步刷新收益、记忆/时序模型收益、未来风险预测收益、动态计算节省与最坏延迟；每个方向通过判别实验后再登记为独立路线。
- 禁止动作：不得一次性同时引入十二类技术、把概念综述当作结果、未测量就宣称降低端到端延迟，或绕过 current 入口直接接入默认 App。
- 原始调研：[2024–2026 新型轻量智能学习与实时视觉感知技术路线整理.md](D:/edge/2024–2026%20新型轻量智能学习与实时视觉感知技术路线整理.md)
