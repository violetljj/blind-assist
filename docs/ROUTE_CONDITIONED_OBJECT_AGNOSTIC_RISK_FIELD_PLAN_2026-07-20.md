# 路线条件化无类别风险场方案

## 文档状态

- 方案编号：`route_conditioned_object_agnostic_risk_field_v1`
- 简称：`RC-OARF v1`
- 状态：`active_research_mainline / hard_gates_blocked`
- 文档类型：日期化研究主线与执行合同，不是当前生产真源
- 定位：USTRF-SC 当前优先研究路线；独立于生产 YOLO、SANPO 公共银标与 `secondary-corridor-causal`
- 当前结论：`prepare_u0_and_hard_gates / do_not_train_student / do_not_replace_default_model`
- 默认 App：继续使用 `app/src/main/assets/yolo11n_fp16_320.tflite`
- 训练授权：`false`
- Android shadow 授权：`false`
- 生产模型替换授权：`false`

## 2026-07-21 真实证据切换与实验冻结

- PR #1 已按历史集成基线、项目结构门、最新 USTRF/CI 三层拆分；不 force-push、不改写旧分支历史。
- 从本节点停止新增 detector、teacher、dense arm、公开/合成数据轮次与参数扫描。未完成的实验资产保留但不再扩写；只有真实路线事件采集和同设备米制几何证据允许进入主动队列。
- 真实事件先执行一个 `route_obstacle` matched pair，再扩到 10-episode pilot；必须保留视频、capture clock、frame ledger、非未来 explicit route/projection、GPT/Codex 隔离 review 与裁决哈希。pilot 通过仍不是真实 U0 truth authority。
- SM-S9280 几何 evidence pack 以红灯状态启动：r3 source-aligned depth `1/861`、r5 `0/843`，raw pose 为 `EPHEMERAL_PER_FRAME`，与 `>=100`、`>=0.95` 和 `INTER_FRAME_STABLE` 硬门不相容。冻结 blocker 为 `BLOCKED_ON_SOURCE_ALIGNED_METRIC_DEPTH_AND_INTER_FRAME_STABLE_POSE`；确定新的同帧 metric-depth/stable-pose 来源前不重复同一 ARCore 900-update 审计。
- `validate_ustrf_sc_device_metric_geometry.py` 现解析五类 typed artifact，要求分项 metrics 与设备/mount/calibration identity 精确一致，并递归校验至少一项 raw/gate source evidence；`blocked/in_progress` 包也必须校验已有收据，不再允许只凭五个空 JSON 的 SHA 假绿。

本方案现作为 USTRF-SC 的优先研究主线，目标允许最终不依赖 YOLO 安全决策。主线激活不等于模型通过：它仍不授权训练、App 接入、模型导出、blind 访问或默认模型替换，也不改变生产 YOLO、SANPO 或 Corridor-Causal Student 的既有结论。

## 2026-07-20 主线激活与当前阻塞

本轮把“方案方向”推进为三个互相独立、可机器拒绝的工作包：

1. `core:ustrf` 新增连续 `UstrfRouteFieldReceipt` 与 `UstrfRouteConditionedRiskInteractor`：同一类别无关风险场只在与外部用户路线相交时产生 route-intrusion evidence；路线缺失、过期、低置信、由风险模型反推或来自未来帧时统一返回 `route_unknown_or_invalid`，绝不回退固定中心走廊。该 seam 只产出 evidence，不打开提醒生命周期。
2. 新增 `configs/ustrf_sc_route_conditioned_event_collection_v1.json`：真实事件门改为路线绑定的双审 `should_alert / critical / lifecycle`，要求逐帧重算的 hashed capture clock、非未来 route trace、两份互不可见的人类 review 与独立 hashed adjudication。为补齐 `unknown_low_obstacle`，完整矩阵冻结为 6 session × 5 scene × 每格 2 个 matched pair，即 120 episode / 60 pair；独立 contract/schema 的 10-episode pilot 只审计采集链，不授权 truth、U0、训练或效果结论。
3. 新增 `validate_ustrf_sc_device_metric_geometry.py`：设备几何门要求同设备/同 mount 的独立标定、`INTER_FRAME_STABLE` pose、严格 source-aligned metric depth、body-local ground truth、完整路线事件真值和同机 P95/热/新鲜度收据。通过也只授权 isolated geometry shadow；`production_authority=false`。

独立校准 admission 现被原子传入 `UstrfMetricGeometryReceiptPromoter`，并把 calibration ID 与 source artifact SHA 继续绑定到 projection admission；单独构造 `independentlyVerified=true` 不再足以打开几何门。

现有 r812–r825 只作为前驱证据：r816 合成 route interaction 与 r817 合成生命周期通过，但 r819 的 16 个 provisional 真实事件仅 balanced accuracy `.7083` 且 context recall `.6667`，r823 lifecycle `.5417`，r825 direct patch×route `.5000`。它们使用 provisional 标签或 future-route oracle，不是 U0 的人类事件分母，也不能改称 RC-OARF 已实现。

因此当前精确阶段为 `E0/U0 gate preparation`：真实 eligible episode 为 0，设备 metric geometry admission 为 false，学生训练仍禁止。

## 2026-07-21 E0 合同加固与路线特异性负控

- 真实事件 validator 现把 manifest 的 `contract_id`、`benchmark_only`、生产替换权限、route `parent_source_id` 与 episode `source_receipt_id` 原子绑定；`training_eligible` 必须同时服从 config authority。因而完整的人类真值矩阵可以授权 U0 评价，但在当前 `full_matrix_training=false` 下仍不能被误报为可训练。
- `UstrfRouteConditionedRiskInteractor` 新增 risk-field 决策时新鲜度门。来自未来或超过 500ms 的风险场即使路线 TTL 尚未到期也会 fail closed，并用独立 `risk_field_unknown_or_invalid` 与路线失效区分；有效 evidence 的 TTL 不得晚于风险场新鲜度上限。
- 冻结 `configs/ustrf_sc_rc_oarf_route_specificity_control_v1.json`，只复用 r816 的 216 个现有预测和同一图像的 LEFT/STRAIGHT/RIGHT 三元组，不重跑 DINO、不重拟合、不选阈值。正确路线 BA 为 `.91555`；两个互不相同、无固定点的循环错路线负控为 `.72492/.79515`，相对增益 `.19064/.12040`，三个父来源均同方向下降，观察到强路线特异性信号。
- 旧 r816 report 没有逐预测 `example_id`，故原 r3 收据仍保留为 `BLOCKED_ON_PREDICTION_IDENTITY_BINDING`，不覆写、不事后补 ID。随后用原 `.venv-export312`、原 checkpoint/输入/层/距离教师/seed/ridge/阈值重跑 r816c；216 个预测 ID 与 route rows 逐项一致，global/route/exact 预测、指标、fold 与系数 SHA 与旧 r816 精确一致。因而路线特异性 gate 正式转为 `PASS_IDENTITY_BOUND_SYNTHETIC_ROUTE_SPECIFICITY`：正确路线 BA `.91555`，两个错路线 BA `.72492/.79515`。
- 同一收据原子绑定 r818 五 seed 稳定门，并从冻结阈值重算六项 check：mean BA `.87737 < .90`，worst-seed no-alert recall `.79710 < .80`，其余四项通过，所以组合决策仍是 `BLOCKED_ON_R818_STABILITY`。不放宽阈值回救，不解除真实事件真值 0、非未来 route provider、设备米制几何、学生训练或生产门。当前收据：`artifacts.local/evidence/ustrf-sc/rc-oarf-route-specificity-control-identity-bound-v1-20260721-r4/report.json`。

## 一、结论先行

当前 BlindAssist 的生产链以单帧目标检测为感知入口：

```text
CameraX
  -> YOLO11n boxes/classes/confidence
  -> RiskAnalyzer 框位置、面积、中心偏置与类别规则
  -> temporal stability / event lifecycle
  -> speech / vibration
```

这条链路适合回答“画面中识别到了什么”，却不能可靠回答：

1. 未在类别表中的低矮、散落或不规则区域是否阻断行进路线；
2. 同一种物体位于路线内、路线边缘或路线外时是否需要提醒；
3. 风险正在接近、持续、被绕开还是已经清除；
4. 路线意图未知、图像退化或模型不确定时，系统应提醒、保持环境注意还是拒绝给出方向；
5. 如何避免把“检测框更准”误当成“助盲事件判断更安全”。

本方案把核心学习目标改为：

> **给定显式、非未来的用户路线场，直接预测局部连续风险场及其因果生命周期；物体类别只用于解释，不再拥有安全决策权。**

目标不是先识别某个 COCO 类别再判断风险，而是先判断图像中的局部区域是否与用户路线发生危险关系。YOLO 可以完全退出安全主链，也可以作为低频语义 Adapter 保留，但它不能单独开启或关闭风险事件。

## 二、与现有三条路径的区别

| 路线 | 核心表示 | 路线输入 | YOLO 角色 | 当前状态 |
| --- | --- | --- | --- | --- |
| 正式 App | 类别 + 矩形框 + 几何规则 | 固定中心区域近似 | 主感知和主要风险证据 | 生产基线 |
| 公共银标 / 显式路线几何 | detector bbox 与外部 route field 的确定性交叠 | 必需，非未来 | 主障碍证据 | benchmark-only；端侧几何已闭环，真实 provider 仍缺 |
| Corridor-Causal Student | YOLO/空间特征 + 运动补偿 + 走廊网格 + causal TCN | 走廊/heading | 输入证据之一 | 独立实验；真实事件数据与训练门未闭合 |
| 本方案 RC-OARF | 密集无类别风险场 + 显式 route field + 因果生命周期 | 必需，非未来；未知时 fail closed | 可选语义解释，不是安全权威 | `active_research_mainline / hard_gates_blocked` |

隔离约束：

- 不读取、复制或重写 `secondary-corridor-causal` 的数据、权重、统计量、实验编号和晋级结论；
- 不把公共银标主线的 provisional 标签、未来轨迹 proxy 或单次高分当作本方案通过证据；
- 允许引用既有失败的“方法层教训”，但本方案必须生成独立配置、产物、报告和 go/no-go 结论；
- 未来本地产物只能写入：

  ```text
  artifacts.local/experiments/route-conditioned-object-agnostic-risk-field/<run-id>/
  ```

- 不写入 `app/src/main/assets`，不修改默认 App 行为；
- blind holdout 禁止用于训练、选特征、选阈值、错误挖掘和停止条件调整。

## 三、为什么不是“继续换检测器”

D-FINE、RT-DETR 等实时端到端检测器可能提高公开检测基准中的定位或速度，但它们仍以“类别 + 框”为主要输出。即使框更准，也不会自动获得以下能力：

- 判断某个物体是否位于用户选择的路线内；
- 检测类别表之外的未知阻挡；
- 区分平行边界、侧向安全物和真正侵入；
- 形成 approach、alertable、post-event、clear 的稳定事件；
- 在路线或图像证据不足时显式 abstain。

因此，检测器替换只保留为低成本横向 Adapter A/B，不作为本方案核心。若未来 D-FINE-N、RT-DETR 或其他模型不能在同设备、同事件集上同时改善关键漏报、误提醒和时延，就停止该分支，不继续以 COCO AP 为理由扩张。

## 四、候选系统结构

```text
RGB history-only clip: 6–8 frames @ 8–10 FPS
frame timestamp / capture quality
explicit route field from external navigation or user choice
optional gyro + sparse-LK motion evidence
             |
             v
MobileNetV4-Conv-small class shared encoder
or current MobileNetV3 control encoder
             |
             +--> local obstacle/objectness field
             +--> walkable / nonwalkable / boundary / unknown field
             +--> relative depth-gradient / surface-change auxiliary field
             +--> route-relative interaction field
             |
             v
history-only causal temporal head
             |
             +--> context_attention
             +--> intervention_needed
             +--> route_clear
             +--> lifecycle phase
             +--> risk_score / uncertainty / abstain_reason
             |
             v
shared Assist Decision Module
             |
             +--> feedback request
             +--> suppression / one-reminder lifecycle
             +--> no directional instruction when route is unknown
```

### 4.1 第一版输入合同

第一版只允许以下输入：

- `rgb_history`：严格 history-only 的 6–8 帧 RGB；当前帧之后的图像不得进入特征、标签生成或运行时决策；
- `frame_timestamp_ms`：单调时间戳，用于拒绝乱序、过期和不连续输入；
- `capture_quality`：至少区分 `valid / degraded / missing`；
- `route_field`：来自外部导航或明确用户选择的相机坐标连续路线场；
- `route_provider_receipt`：绑定 provider、投影、生成时间、有效期和置信度；
- `sparse_motion`：可选稀疏 LK、gyro 或已独立验证的姿态对齐运动特征；
- `semantic_hint`：可选低频类别提示，仅用于文案或诊断。

禁止输入：

- 从未来视频帧反推出的路线；
- 当前待评标签、episode ID 特例或人工答案字段；
- blind 标签、blind 错误样本或以 blind 选出的阈值；
- 未绑定来源、时间、坐标系和哈希的 route payload；
- 把模型不确定性、语义 unknown、路线 unknown 和采集缺失压成同一个字段。

### 4.2 显式路线合同

本方案不假设视觉模型能够从一段普通视频中唯一推断用户下一步要左转、直行或右转。路线必须由以下任一来源提供：

1. 用户明确选择；
2. 本地导航规划；
3. 已独立验证的非未来 route provider；
4. 仅在离线 teacher upper-bound 中使用、且不进入 runtime 声明的人工路线标注。

当路线未知、过期、置信度不足或投影无效时：

- 允许输出 `context_attention`；
- 禁止输出带方向的 `intervention_needed`；
- 必须给出 `abstain_reason=route_unknown_or_invalid`；
- 不得自动回退到固定中心走廊并称其为真实用户路线。

### 4.3 密集输出合同

候选模型至少输出：

| 输出 | 含义 | 是否可独立触发提醒 |
| --- | --- | --- |
| `local_obstacle_field` | 局部不透明或实体阻挡概率 | 否 |
| `walkability_field` | 可通行与不可通行结构 | 否 |
| `boundary_field` | 路沿、台阶、边缘和结构边界 | 否 |
| `unknown_field` | 模型未覆盖或证据不足区域 | 否 |
| `route_relative_risk_field` | 与显式路线相交后的连续风险 | 进入事件头后才允许 |
| `risk_score` | 当前因果风险强度 | 需通过生命周期与安全门 |
| `lifecycle_logits` | context / approach / alertable / post-event / clear | 需通过事件门 |
| `abstain_reason` | route、motion、capture 或 model unknown | 不触发方向提醒 |

安全提醒必须来自 `route_relative_risk_field + lifecycle`，不得由某个辅助 mask、深度值、类别或单帧峰值直接触发。

### 4.4 无 YOLO与混合模式

本方案允许两个实现模式，但必须分别报告：

#### A. `object_agnostic_core`

- 安全主链不运行 YOLO；
- 风险由密集风险场、显式路线和 causal lifecycle 决定；
- 语音只说方向和风险等级，不保证物体类别名称；
- 用于证明“不依赖类别检测也能完成事件风险判断”。

#### B. `object_agnostic_core_with_semantic_adapter`

- 与 A 使用完全相同的安全判断；
- YOLO或另一轻量分类/检测 Adapter 低频运行，仅为已开启事件补充“人、车、路障”等解释；
- 语义 Adapter 不得改变 open/clear、风险等级和 abstain；
- 必须分别报告语义 Adapter 的增量延迟、功耗和错误文案率。

若两种模式的安全结果不同，说明语义泄漏进入了安全主链，实验必须 fail closed。

## 五、端侧学生模型候选

### 5.1 编码器

首轮只允许两个冻结候选：

1. `MobileNetV3 control`：复用现有经验，作为可归因基线；
2. `MobileNetV4-Conv-small candidate`：参考其面向移动 CPU、GPU、DSP 和加速器的效率设计。

首轮不允许同时比较 MobileNetV4、RepViT、FastViT、PIDNet-S、Transformer 和多个输入分辨率。只有上界与 MobileNetV3 学生证明任务可学后，才允许一次更换一个 encoder。

为降低 LiteRT 风险，第一版优先使用：

- Conv / depthwise Conv / pointwise Conv；
- pooling、concat、resize；
- 基础激活；
- Conv1D 或可等价展开的 causal temporal blocks；
- 明确、静态的输入输出 shape。

### 5.2 密集解码头

参考 PIDNet 的 detail/context/boundary 分工，但不直接移植完整网络：

- detail branch：保留台阶、路沿和小障碍边界；
- context branch：估计可通行结构与大范围路线环境；
- boundary branch：防止细边界被上下文淹没；
- interaction head：只在显式 route field 上计算 route-relative risk。

### 5.3 因果事件头

- 只读 history，不读未来帧；
- 第一版采用 4–8 帧 causal TCN；
- 输出 `context_attention / intervention_needed / route_clear` 与细粒度 lifecycle；
- 风险可以在 approach 阶段上升，但必须允许在 post-event 阶段下降；
- 不使用跨过 clear 阶段的全程单调约束；
- 同一事件短时遮挡或语义变化后保持身份，但必须有明确超时、证据丢失与清除状态。

## 六、离线教师与论文角色

教师只产生辅助表示、上界或 provisional target，不产生人类事件真值。

| 来源 | 允许用途 | 禁止用途 |
| --- | --- | --- |
| [MobileNetV4](https://arxiv.org/abs/2404.10518) | 移动端共享编码器设计参考 | 用论文设备结果代替 SM-S9280 实测 |
| [PIDNet](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) | detail/context/boundary 解码设计参考 | 直接把驾驶数据结果当作步行安全证据 |
| [Depth Anything V2](https://arxiv.org/abs/2406.09414) | 相对深度、表面梯度、teacher upper-bound | 把相对深度称为米制距离或安全真值 |
| [Video Depth Anything](https://arxiv.org/abs/2501.12375) | 离线时序一致深度与梯度 teacher | Android 主链或未来帧泄漏 |
| [SAM 2](https://arxiv.org/abs/2408.00714) | 离线 mask、传播和遮挡上界 | 无提示直接触发提醒 |
| [EdgeTAM](https://arxiv.org/abs/2501.07256) | 端侧视频 mask benchmark、teacher 或身份上界 | 未过同机事件门即进入生产 |
| [D-FINE](https://arxiv.org/abs/2410.13842) / [RT-DETR](https://github.com/lyuwenyu/RT-DETR) | 检测 Adapter 横向 A/B | 把更高检测 AP 当作事件晋级 |
| [RiskProp](https://openaccess.thecvf.com/content/CVPR2026/html/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.html) | approach 风险传播与平滑曲线参考 | 对 post-event 强制单调上升 |
| [ViNT](https://arxiv.org/abs/2306.14846) / [NoMaD](https://arxiv.org/abs/2310.07896) | 离线路线意图研究、长期 teacher | 机器人策略直接控制人类助盲提醒 |

所有教师产物必须绑定：

- 模型名称、版本、权重 SHA256 和许可证；
- 输入视频/图像来源、许可和 SHA256；
- 推理尺寸、采样频率和预处理；
- 输出角色：`auxiliary_only / provisional / not_model_consensus`；
- 禁止训练、校准、blind 或生产的默认授权字段。

## 七、真实监督层级

监督优先级从高到低固定为：

1. GPT/Codex 独立复核并形成共识或第三模型仲裁的物理事件锚点与 actionability；
2. 同一用户选择或导航计划绑定的显式 route field；
3. matched positive/negative 的 route-relative 风险关系；
4. 人工或受控来源的局部边界、可通行和 unknown 标注；
5. 深度、SAM/EdgeTAM、DINO 等离线教师辅助目标；
6. 合成反事实，只允许 train-only augmentation。

合成与银标不得进入：

- 真实事件分母；
- 最终混淆矩阵；
- blind；
- 校准；
- 生产替换授权。

每个合成后代必须继承真实父来源；父来源被留出时，其全部合成后代同时退出训练折，防止 source 泄漏。

## 八、必须先完成的架构前置项

本方案在训练前必须解决两处现有 seam 问题。

### 8.1 共用 Assist Decision Module

当前生产与 benchmark 分别理解风险分析、时序稳定、事件 update、反馈抑制、实际反馈和 `recordFeedback` 的调用协议。候选算法若穿过不同实现，离线或真机 A/B 将无法代表 App 行为。

前置目标：

- 风险融合、时序稳定、事件生命周期、重复抑制和反馈请求集中到一个深 Module；
- 生产和 benchmark 调用同一个 interface；
- TTS、震动和报告输出保持外部 Adapter；
- 重构前后在冻结回归输入上逐帧、逐事件结果完全一致；
- 本步骤不改变规则阈值、用户体验或默认模型。

### 8.2 模型无关 Perception Evidence Seam

现有 `DetectorFrameResult(List<Detection>)` 强制上游提供类别、标签和矩形框；当前分割实现已经把 dense mask 压缩成 `Detection`。这种 interface 不适合承载 route field、密集风险、motion、uncertainty 和直接事件输出。

前置目标：

- 保留 `TfliteYoloDetector` 作为 box detector Adapter；
- 新建更高层、模型无关的导航感知证据 seam；
- 每种 Adapter 在进入决策内核前负责来源、时间、新鲜度、坐标和质量归一化；
- 不在方案阶段先设计无限宽的万能数据结构；具体 interface 必须由至少 YOLO Adapter 和 RC-OARF Adapter 两个真实实现共同证明。

### 8.3 Benchmark Candidate Adapter

候选执行、耗时、证据归一化和输出报告必须通过统一 Adapter 进入同一决策内核。不得继续用多个布尔字段在大型 benchmark 中手工拼接新的算法路径。

## 九、分阶段可证伪路线

### P0：生产与 benchmark 语义一致性

目标：只重构 seam，不改算法。

通过条件：

- 同一冻结输入下，重构前后 detector、risk、stable risk、event、feedback decision 完全一致；
- benchmark 与生产决策内核使用同一实现；
- 当前 YOLO 默认 App 构建和回归通过。

失败处理：修复一致性，不进入 U0。

2026-07-21 状态：`P0_HOST_VERIFIED`。

- 新增 Android-free 的 `AssistDecisionKernel` 编排器；生产 `AssistSessionCoordinator` 与 `DetectorAbDeviceBenchmarkTest` 现在共享 `Analyzer -> Temporal -> Stabilizer -> Event -> confirmation -> FeedbackGateway receipt -> trace` 的顺序。它是带状态且通过 effect port 调用反馈的共享编排器，不是函数式纯 kernel。
- 独立黄金序列冻结无深度中心 `stairs` 的四帧结果：`LOW/DISTANCE_TOO_FAR -> MEDIUM-to-NONE/UNSTABLE -> MEDIUM/TRIGGERED -> EVENT_ALREADY_ALERTED`；另有 unavailable receipt 后不消费事件、下一帧可重试的回归。
- device benchmark 报告升级为 `blindassist_detector_ab_device_benchmark_v2`，绑定 shared-kernel contract、`STANDARD` profile、manifest scenario、100ms 合成时钟与 `planner_accept_all_v1`。旧 `model_risk` 字段继续表示 temporal raw risk，新 `stable_model_risk` 承载生产稳定层结果；所有 alert/event 聚合使用 stable/receipt 语义，跨 v1/v2 不可直接比较。
- SANPO device-event extractor 现 fail-closed 拒绝旧 schema、旧决策实现或未知 feedback adapter。`planner_accept_all_v1` 只表示确定性计划接受，不证明 TTS/震动在物理设备上送达。
- host 验证覆盖 `:core:assist:test`、benchmark Kotlin 编译、默认 App/feature/core:device 聚焦回归和 debug APK 构建；P0 不改变风险阈值、默认 YOLO 或用户可见 App 行为。真机 benchmark 尚未在本轮重跑，因此 P0 只关闭代码/host parity，不替代 U0、真实事件或设备门。
- 2026-07-21 后续已在同一 SM-S9280/API 36 对 90 帧 SANPO v2 连续基准重跑 `SanpoTraversabilityOracle/current`。原始 v2 报告 SHA256 为 `6b2d39b...b96b4a25`，绑定 `blindassist_shared_decision_kernel_v1`、`STANDARD`、100ms 合成序列时钟与 `planner_accept_all_v1`；candidate total P95 `57.674ms`、event recall `1.0`、critical miss `0`、delivered repeat `0`、post-event clearance `1.0`、false alerts/min `0`，49 次重复尝试被抑制。仍有 2 次 event ID regeneration，且 planner acceptance 不是物理反馈投递；该数据含历史 benchmark-only 来源，不能替代 U0 人类 truth 或生产门。

### U0：冻结教师上界

目标：在不训练端侧学生前，证明“无类别密集风险场 + 显式路线”存在足够上界。

输入：

- 非 blind、来源分组的真实连续 episode；
- 人工或外部非未来显式 route field；
- 离线 teacher 产生的 mask、相对深度和时序一致性辅助场；
- 人类复核事件真值作为唯一评价答案。

比较：

1. 当前 YOLO + 现有风险规则；
2. detector bbox + 显式路线确定性交叠；
3. teacher dense field + 显式路线；
4. teacher dense field + route + causal lifecycle。

U0 只用于判断问题是否值得学习，不授权训练或生产。

2026-07-21 gate 状态：`U0_TWO_ANDROID_ADAPTERS_AND_DENSE_KERNEL_SEAM_DEVICE_VERIFIED_WAITING_GPT_CODEX_EVENT_CONSENSUS_AND_FOUR_REAL_ADAPTERS`。

- `configs/ustrf_sc_u0_teacher_upper_bound_v1.json` 已冻结四个正式比较臂与 uniform/shuffled route 两个负控；所有臂必须绑定同一 ordered frame ledger、shared decision kernel、实现/模型/阈值 SHA，并逐 episode 绑定视频、原始 route、frame IDs 与预测 trace。
- `scripts/evaluate_ustrf_sc_u0_teacher_upper_bound.py` 会在评价前重算完整 route-conditioned truth gate，以 canonical JSON 和 LF-normalized text 的跨平台合同钉死官方 full-matrix config SHA 与 route/frame/review 四个 validator SHA，严格要求 120 episode / 60 matched pair / 120 route-bound episode、唯一 episode/event ID、LOSO holdout 与每 fold critical 分母。matched pair 共享 `route_plan_id + provider policy + route choice`，但各自 current-camera 投影必须绑定自己的 video/frame ledger；禁止复制逐像素 route trace 冒充反事实一致性，并拒绝 blind、future input、漏臂、漏 episode、哈希漂移和 synthetic 授权。
- 每个 causal-arm LOSO fold 沿用 event recall、critical miss、false alerts/min、repeat、post-event clearance 硬门，并补冻 `delivered alerts/event <= 1.10`、`P95 clearance latency <= 500ms`、`event regeneration <= 0.10`。aggregate 另要求 dense-route 相对 detector/uniform/shuffled 的 matched-pair BA 增益至少 `.10`，unknown-low-obstacle recall 相对最佳 bbox 臂至少增益 `.10` 且来自至少 2 个 session；causal lifecycle 不得以 recall、critical miss、误提醒或清除退化换取平滑。
- 10-episode pilot 已有确定性空槽生成器、独立 manifest schema 与 fail-closed 审计器。它重算 frame index、capture ns、video PTS、clock summary、route source/generated/consuming-frame 因果链、投影收据以及双审/裁决哈希；即使 10 集全部合格，报告中的 truth/U0/S0/training/Android/production 权限仍固定为 false。当前正式与 pilot manifest 都为空；空 pilot CLI 实跑 exit 2 且不写报告。因此 S0、学生训练、Android 和生产权限全部保持 false。
- 新增 U0 prediction-evidence admission：六臂不再能提交占位 SHA 与手写 `alert_timestamps_ms`。每臂必须提供本地 hash-bound implementation/artifact/threshold、execution receipt 和逐 episode/frame trace；validator 逐项对齐 truth frame ledger、adapter、shared-kernel 顺序及 feedback receipt，并从 `delivered=true` 重算提醒。摘要漂移、单字节篡改、漏帧、adapter/kernel 漂移或 execution failure 均 fail closed。合成 trace 即使公式全过，在正式 contract 下仍固定 `u0_passed=false`。
- v2 unified runner 进一步关闭“手写整份 trace”与 LOSO/route-control 漂移：runner 实际启动 preregistered subprocess wrapper，adapter 只接收去除 review、adjudication、`should_alert` 和事件标签的 inference manifest；每个 session 的 exact train inventory、fold artifact/training receipt、500ms exact-grid cadence、kernel 原生 event/feedback 映射及 request/output/exit code 都进入不可变证据链。baseline/bbox 只允许 kernel-native optional event ID，禁止 writer 伪造 ID；dense 臂必须有 kernel-native identity。
- uniform route 是明确的 full-frame equal-weight field；shuffled route 是同一 held-out session 内按 episode ID 排序的 cyclic shift-one，无 seed、无标签、无 refit。adapter 实际 route input SHA 与 truth route SHA 分开记录，避免控制臂声称使用原路线。正式 backend 钉死 Android/Kotlin `AssistDecisionKernel`；dependency-free synthetic fixture 使用单独 backend，只证明协议。
- `baseline_yolo_geometry` 真实 adapter 已完成并在 SM-S9280/API 36 双次复跑：host 不生成 decision，设备内从编码 sample PTS 到 RGBA8888、shipped YOLO11n TFLite、shared kernel 和最终 JSON 全链执行。receipt 同时绑定 ledger、app/test APK、build fingerprint、模型/标签、host/device 源码及逐帧 PTS/内容/耗时；稳定字段两次一致。这仍是无人类真值的 public-video pipeline smoke，只关闭 baseline 执行链。
- `detector_bbox_explicit_route` 真实 adapter 也已完成：固定用因果最新 route sample、相机底部中心至 1/2/3 秒 waypoint 的 0.08 frame-width 走廊、bbox 底部 25% footprint 做二值 gate，未改写 detection 或 shared-kernel 参数；unknown route 空 gate。SM-S9280/API 36 负控在相同 encoded sample/RGBA/APK 上只改路线，中心 route 排除 person/raw `NONE`，左侧 route 保留相同 bbox/raw `MEDIUM`；左侧 route 复跑的 gate/decision 稳定字段一致。route receipt 由 host 与 admission 重算，最终内核哈希证据位于 `artifacts.local/evidence/ustrf-u0-bbox-route-device-smoke-20260721-r3/`；该 public-video smoke 没有人类事件真值，不提供精度或晋级权限。
- object-agnostic dense→shared-kernel seam 已冻结并在真机验证：`AssistRiskEvidenceFrame` 禁止 bbox、检测式 distance、预置 trend/event/feedback、矛盾 NONE 语义、越界/不一致分数和乱序时间；`UstrfU0DenseRiskEvidenceAdapter` 只把 current-frame route intrusion 与 local peak 按固定阈值归一化，再交给同一 temporal/stabilizer/event/feedback 链。U0 admission 同时要求四个 dense/control 臂提供 teacher 模型名称/版本/许可证/权重与实现 SHA、LOSO fold、route 和逐帧 field/unknown/归一化算术 receipt。SM-S9280/API 36 的 3 个 seam instrumentation tests 与 11 files / 54 tests 统一合同套件通过。当前仍没有 teacher field generator、fold artifact 或第三条真实 adapter，不能把 seam 计作第三臂。

停止条件：

- dense teacher 不能提高未知/低矮障碍覆盖；
- 事件召回提高但误提醒或清除显著恶化；
- 增益只出现在同一来源或合成样本；
- 显式路线不可获得，只能依赖未来视频 proxy；
- teacher 输出必须经过人工事后解释才能通过。

### S0：确定性表示 probe

只在 U0 通过后运行：

1. MobileNetV3 frozen features + route interaction；
2. MobileNetV4-Conv frozen features + route interaction；
3. uniform route field 负控；
4. shuffled route field 负控；
5. no-route ablation。

要求：

- 完整 parent-source 留一；
- 同一来源的 real、silver 和 synthetic descendants 保持同折；
- 闭式 ridge 或确定性线性/双线性 probe；
- 报告最差 source、最差 scene、正负 recall 与 matched-pair 一致性；
- route field 相比 uniform/shuffled 必须提供可重复增益。

若 S0 不优于当前 detector geometry 路线，停止本方案，不转向 optimizer、SAM/ASAM 或大规模 backbone 搜索。

### S1：微型密集学生

- 固定一个输入尺寸；
- 固定 MobileNetV3 或 MobileNetV4-Conv 中 S0 更强者；
- 先训练 dense auxiliary heads，不训练事件头；
- 一个阶段只增加一个 loss；
- 训练只读 train/dev，不读 blind；
- 输出必须可视化并检查是否真正定位路线相交区域，而不是依赖全局背景。

### S2：因果事件头

- 冻结或低学习率保持 dense encoder；
- 只加入一个 4–8 帧 causal TCN；
- 固定五个预注册 seed；
- 报告 mean、std、worst seed、worst source、worst scene；
- 使用事件级 recall、critical miss、false alerts/min、clearance 和 repeated alerts；
- frame recall、mIoU 和 boundary IoU 仅作诊断。

### S3：full INT8 保真

- 模型文件绑定 SHA256；
- float 与 TFLite 输出在同一 dev 输入上比较；
- 所有输入输出均为 INT8，禁止运行时 float tensor；
- dense field、risk、lifecycle 和 abstain 分别检查保真；
- fold-local normalization 只能来自训练折；
- 不允许用未训练 synthetic fixture 的延迟替代真实 checkpoint 结果。

### S4：同机 shadow

目标设备：`SM-S9280`，同一脚本、同一连续 evalset。

- 当前 YOLO 提醒仍是用户可见结果；
- RC-OARF 只记录 shadow 候选和耗时；
- 使用 `LatestOnlySidecar` 或等价单槽新鲜度语义，禁止积压旧帧；
- 报告端到端 P50/P95、掉帧、新鲜度、功耗/发热观察和事件指标；
- 不读取生产用户隐私素材，除非有明确同意、数据合同和本地保存策略。

### S5：人工发布决策

即使所有自动门通过，报告仍固定：

```json
{
  "production_model_replacement_authorized": false
}
```

替换默认模型需要单独的人工决策、Android集中回归、APK资产检查和发布流程。

## 十、预注册事件与设备硬门

最终阈值沿用项目现有候选事件门，不因本方案修改：

| 指标 | 硬门 |
| --- | ---: |
| `event_recall` | `>= 0.90` |
| `critical_miss_rate` | `<= 0.05` |
| `false_alerts_per_minute` | `<= 0.50` |
| `delivered_repeated_alert_rate` | `<= 0.10` |
| `post_event_clearance_rate` | `>= 0.90` |
| 同机总 P95 | `<= 70 ms` |

同时必须满足：

- 相对当前 YOLO 基线，关键漏报不得增加；
- 不能通过大量 abstain 人为降低误提醒；
- route unknown、capture degraded、semantic unknown 和 model uncertainty 的错误安全声明率必须单独报告；
- parallel boundary、step/curb、center obstacle、lateral cut-in、unknown low obstacle 分项报告；
- 无 YOLO 与带语义 Adapter 两种模式的安全事件结果必须一致。

这些是未来设备晋级门，不是当前已达到结果。

## 十一、明确停止条件

出现任一情况即停止并保存负向报告：

1. U0 教师上界不能稳定胜过 detector geometry；
2. route field 相对 uniform/shuffled 负控没有稳定增益；
3. 收益依赖未来路线、episode ID、同源背景或 synthetic-only 测试；
4. dense field 看似更平滑，但事件漏报、误提醒或清除不改善；
5. 未知障碍召回提高，同时关键人物/车辆/台阶召回下降；
6. 通过扩大 unknown/abstain 压低误报；
7. route provider 在真实设备上无法保证非未来、相机投影、时效和置信度；
8. full INT8 丢失边界、risk 或 lifecycle 语义；
9. 同机 P95、发热或帧新鲜度超过预算；
10. 需要修改默认 App 行为才能证明候选收益；
11. 需要在失败后搜索大量 backbone、阈值、sigma、ridge、optimizer 或 SAM/ASAM 才能恢复结果；
12. 无法获得至少两个独立来源、同机制的真实正负事件。

## 十二、当前证据强弱

### 支持本方向的证据

- 显式 route field + detector bbox 的离线 oracle proxy 曾明显强于不带路线的全局分类，说明“路线是缺失变量”；
- bbox 距离场在合成路线条件化任务上曾显著强于全局 readout，说明局部连续表示值得研究；
- 当前端侧显式路线几何已通过 Python/Kotlin 数学一致性和 SM-S9280 benchmark-only 验证，证明路线条件化不是纯概念；
- YOLO、分割、深度和时序实验反复显示，仅优化类别、框、全局特征或 head 不能闭合事件语义。

### 反对或限制本方向的证据

- frozen DINO patch × route field 在真实事件上接近随机，不能当作可用风险表示；
- 合成距离场迁移到真实 provisional 事件后只部分成立，context 误激活严重；
- 使用真实 marker bbox 学“哪里有物体”仍不能判断它是否阻断所选路线；
- 当前真实 route provider 与 LEFT/RIGHT intervention 覆盖尚未闭合；
- 公共银标和单次大模型输出不能替代 GPT/Codex 隔离事件共识；
- 现有真实事件数量不足以支持复杂端到端模型或大规模结构搜索。

因此，本方案当前可信度为：

- 问题重定义：`medium-high`；
- 教师上界可行性：`unknown`；
- 端侧学生可训练性：`unknown`；
- 真机产品收益：`unproven`；
- 当前生产替换：`not_authorized`。

## 十三、首个执行包

如果未来启动，本方案的首个执行包只包含：

1. 冻结本方案配置、目录、来源和禁用字段；
2. 建立生产与 benchmark 共用决策内核的行为等价测试计划；
3. 定义最小的模型无关感知证据 seam，但不提前设计万能 interface；
4. 选择一小组非 blind、经 GPT/Codex 隔离复核、显式路线可用的真实连续 episode；
5. 运行 U0 教师上界与 uniform/shuffled route 负控；
6. 写出单一 `go / no_go` 报告。

首个执行包明确不做：

- 不训练 MobileNetV4；
- 不导出 TFLite；
- 不连接 App；
- 不读取 blind；
- 不更换 YOLO；
- 不新增 SAM/ASAM；
- 不下载大量无审阅视频；
- 不进行 backbone 或超参数大扫。

只有 U0 证明问题存在可迁移上界，才创建 S0 学生实验。

## 十四、决策摘要

本方案不是第三个“更复杂的 YOLO 增强器”，而是一个允许最终重定义安全核心的独立候选：

> **检测器负责解释“可能是什么”；路线条件化无类别风险场负责判断“是否正在阻断用户选择的路线”；因果事件头负责决定“何时提醒、何时清除、何时拒判”。**

它的首要价值是消除“必须先识别成已知类别才能成为风险”的前提；首要风险是缺少真实 route provider、route-relative 监督和跨来源事件。当前最正确的下一步不是立即训练，而是冻结 U0 教师上界，快速判断这条路线是否值得继续。
