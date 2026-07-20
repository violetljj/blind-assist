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

本方案现作为 USTRF-SC 的优先研究主线，目标允许最终不依赖 YOLO 安全决策。主线激活不等于模型通过：它仍不授权训练、App 接入、模型导出、blind 访问或默认模型替换，也不改变生产 YOLO、SANPO 或 Corridor-Causal Student 的既有结论。

## 2026-07-20 主线激活与当前阻塞

本轮把“方案方向”推进为三个互相独立、可机器拒绝的工作包：

1. `core:ustrf` 新增连续 `UstrfRouteFieldReceipt` 与 `UstrfRouteConditionedRiskInteractor`：同一类别无关风险场只在与外部用户路线相交时产生 route-intrusion evidence；路线缺失、过期、低置信、由风险模型反推或来自未来帧时统一返回 `route_unknown_or_invalid`，绝不回退固定中心走廊。该 seam 只产出 evidence，不打开提醒生命周期。
2. 新增 `configs/ustrf_sc_route_conditioned_event_collection_v1.json`：真实事件门改为路线绑定的双审 `should_alert / critical / lifecycle`，要求 hashed capture clock、非未来 route trace、两名独立人工复核与 hashed adjudication。为补齐 `unknown_low_obstacle`，完整矩阵冻结为 6 session × 5 scene × 每格 2 个 matched pair，即 120 episode / 60 pair；10-episode pilot 只审计采集链，不授权训练或效果结论。
3. 新增 `validate_ustrf_sc_device_metric_geometry.py`：设备几何门要求同设备/同 mount 的独立标定、`INTER_FRAME_STABLE` pose、严格 source-aligned metric depth、body-local ground truth、完整路线事件真值和同机 P95/热/新鲜度收据。通过也只授权 isolated geometry shadow；`production_authority=false`。

独立校准 admission 现被原子传入 `UstrfMetricGeometryReceiptPromoter`，并把 calibration ID 与 source artifact SHA 继续绑定到 projection admission；单独构造 `independentlyVerified=true` 不再足以打开几何门。

现有 r812–r825 只作为前驱证据：r816 合成 route interaction 与 r817 合成生命周期通过，但 r819 的 16 个 provisional 真实事件仅 balanced accuracy `.7083` 且 context recall `.6667`，r823 lifecycle `.5417`，r825 direct patch×route `.5000`。它们使用 provisional 标签或 future-route oracle，不是 U0 的人类事件分母，也不能改称 RC-OARF 已实现。

因此当前精确阶段为 `E0/U0 gate preparation`：真实 eligible episode 为 0，设备 metric geometry admission 为 false，学生训练仍禁止。

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
| 本方案 RC-OARF | 密集无类别风险场 + 显式 route field + 因果生命周期 | 必需，非未来；未知时 fail closed | 可选语义解释，不是安全权威 | `proposal_only / not_started` |

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
- 输出角色：`auxiliary_only / provisional / not_human_truth`；
- 禁止训练、校准、blind 或生产的默认授权字段。

## 七、真实监督层级

监督优先级从高到低固定为：

1. 双人独立复核的物理事件锚点与 actionability；
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
- 公共银标和大模型复核不能替代双人事件真值；
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
4. 选择一小组非 blind、人工复核、显式路线可用的真实连续 episode；
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
