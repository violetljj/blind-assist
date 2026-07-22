# USTRF 前沿论文研究指导（2026-07-22）

状态：`research guidance / dated snapshot / production authority unchanged`
范围：将深度、跟踪、自由空间、开放词汇、VLM 提醒、多模态反馈和完整系统论文转化为 BlindAssist 当前 USTRF 研究的可证伪实验指导。
本地论文包：[`artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/`](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/README.md)

## 一、结论与当前定位

这批论文整体支持 BlindAssist 从“识别画面中有什么”转向“给定可信路线，持续判断哪里可通行、危险如何演化、何时以最低认知负担提醒”。它们不会改变当前证据边界：正式 App 仍以既有 YOLO 风险链为基线；USTRF 的 TTC、风险场、走廊、route-risk seam 和 dense teacher 均处于研究、shadow 或 offline 层。

当前最重要的研究缺口不是再堆一个模型，而是：

1. 准入真正包含身体绑定前向路线、连续 RGB-D/pose、障碍接近和 `passed/cleared` 生命周期的数据源；
2. 在来源分组、固定协议和最坏分层下比较 tracker、深度 teacher 与 dense risk 表示；
3. 把 route-conditioned event recall、critical miss、false alerts/min、clearance、repeat alert、unknown/abstention 和 evidence age 作为主指标；
4. 保持模型输出、自动评审、设备事实、真实用户结果与生产授权互不冒充。

研究顺序冻结为：

```text
数据源与可评分事件闭环
  -> 跟踪/ego-motion/TTC 消融
  -> 单帧深度与时序深度 teacher 消融
  -> 显式路线下的 dense traversability-risk field
  -> 提醒时机、语言和触觉评价
  -> 独立设备、用户与生产晋级门
```

## 二、证据—论点映射

| ID | 论文与本地文件 | 原文可支持的事实 | 可用于 BlindAssist 的论点 | 建议引用位置 | 证据强度与风险 |
| --- | --- | --- | --- | --- | --- |
| P01 | [Depth Anything V2](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/01_Depth_Anything_V2_NeurIPS_2024.pdf) ([source](https://arxiv.org/abs/2406.09414)) | 提供泛化较强的单帧单目相对深度；Small 权重可作为轻量控制模型 | 深度 teacher 可替代框面积成为更密集的相对空间表征，但不能自动成为米制距离 | Method-Depth-Control | 强方法证据；无助盲事件和手机安全结论 |
| P02 | [Video Depth Anything](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/02_Video_Depth_Anything_CVPR_2025.pdf) ([source](https://arxiv.org/abs/2501.12375)) | 在 DAv2 上加入时空头和长视频窗口策略，提升视频深度时间一致性 | 时序深度适合作为离线 teacher 或非因果上界，检验深度闪烁是否造成风险事件翻转 | Method-Temporal-Depth | CVPR 证据较强；A100 速度不是手机速度，窗口含未来帧，非直接因果 runtime |
| P03 | [PathFinder](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/03_PathFinder_Wayfinding_Assistant_arXiv_2025.pdf) ([source](https://arxiv.org/abs/2504.20976)) | 从单目深度中递归搜索局部自由路径，并做小规模 BLV 可用性研究 | 深度自由空间可以作为简单 baseline，但“最长自由路径”不等于用户路线或安全路线 | Baseline-Free-Space | 预印本；动态目标、低光、楼梯、坡面与全局路线受限 |
| P04 | [ByteTrack](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/04_ByteTrack_ECCV_2022.pdf) ([source](https://arxiv.org/abs/2110.06864)) | 低置信检测二次关联可恢复部分被遮挡真目标并减少轨迹碎片 | 对已冻结 detections 做二阶段关联，可能改善障碍生命周期连续性 | Method-Tracking-A | ECCV 方法证据强；MOT 指标不等于助盲事件指标，未解决强 ego-motion |
| P05 | [OC-SORT](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/05_OC_SORT_CVPR_2023.pdf) ([source](https://arxiv.org/abs/2203.14360)) | observation-centric re-update 可修正遮挡期 Kalman 累积误差 | 可构建无需重型 ReID 的 observation-centric tracker 对照臂 | Method-Tracking-B | CVPR 方法证据强；CPU 关联速度不含检测，非穿戴相机 TTC 证明 |
| P06 | [Wearable Steering Assistance](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/06_Wearable_Steering_Assistance_ICRA_2024.pdf) ([source](https://arxiv.org/abs/2408.00332)) | 共享主干联合跑道边界感知、障碍检测与路径规划 | 支持把 boundary/detail、obstacle/context 与 planning 分开建模 | RelatedWork-Multitask | ICRA 系统证据；结构化跑道、约千图数据和 Orin NX 不能外推开放街道/手机 |
| P07 | [YOLOE](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/07_YOLOE_ICCV_2025.pdf) ([source](https://arxiv.org/abs/2503.07465)) | 统一文本、视觉和无提示开放词汇检测/分割 | 适合长尾候选发现、数据构建和低频语义解释，不拥有安全事件 authority | Adapter-Open-Vocabulary | ICCV 方法证据；LVIS/COCO AP 不证明透明、细小或低反射危险可靠 |
| P08 | [WalkVLM](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/08_WalkVLM_arXiv_2024_v4_2025.pdf) ([source](https://arxiv.org/abs/2412.20903)) | 以约 12k 视频—标注对研究层次化规划和时间自适应提醒，减少冗余输出 | 风险事件闭合后，可研究提醒触发、去重和行动导向短句 | Method-Reminder-Timing | ICCV 方法证据；VLM 不能成为实时几何或避障 authority |
| P09 | [VLM Navigation BLV Evaluation](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/09_VLM_Navigation_BLV_arXiv_2026.pdf) ([source](https://arxiv.org/abs/2603.15624)) | 现有 VLM 在拥挤计数、相对空间关系和适应性上存在明显差异与失败 | VLM 应隔离在解释、问答和高层语义层，不能独立开启方向性避障动作 | SafetyBoundary-VLM | 预印本；模型/API 快速变化，非端到端碰撞实验 |
| P10 | [VL-GUIDE BLV Evaluation](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/10_VL_GUIDE_BLV_Evaluation_arXiv_2025_v2_2026.pdf) ([source](https://arxiv.org/abs/2510.00766)) | 建立面向 BLV 导航描述的多维偏好数据和自动 evaluator | 提醒评价应是多维、BLV-aware 的，不能只依赖 BLEU/CLIP 或一个任意加权总分 | Evaluation-Reminder | 当前有效后继预印本；自动 judge 不能冒充真实 BLV 用户结果 |
| P11 | [Human-centred Multimodal Wearable](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/11_Human_Centred_Multimodal_Wearable_NMI_2025.pdf) ([source](https://doi.org/10.1038/s42256-025-01018-6)) | 视觉、音频、触觉、用户适应和训练需要协同设计 | 产品反馈不能只优化模型准确率；音频、触觉与训练应有独立角色 | Discussion-Human-Centred | 同行评审系统证据强；完整硬件体系与手机原型差异大 |
| P12 | [GuideTouch](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/12_GuideTouch_HRI_2026.pdf) ([source](https://arxiv.org/abs/2601.13813)) | 两个 ToF、四点触觉和少量方向模式获得较高静态识别率 | 触觉编码应少而明确，先验证方向识别再验证动态避障 | Method-Haptics | 初步研究；93.75% 是方向提示识别率，不是避障成功率 |
| P13 | [Sight Guide](../../../artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/13_Sight_Guide_Cybathlon_arXiv_2025.pdf) ([source](https://arxiv.org/abs/2506.02676)) | RGB、双类深度、VIO、语义与反馈的互补能覆盖不同传感器失败模式 | 长期系统应采用可审计的传感器互补，而不是假设单目/ToF/立体任一万能 | Discussion-System | 受控比赛系统；95.7% 不是开放世界成功率，细杆、黑色物体和低光仍困难 |

## 三、论文到实验的执行规则

### 3.1 跟踪、ego-motion 与 TTC

**假设 H1**：在冻结检测器、输入帧和路线的条件下，轻量二阶段关联或 observation-centric 修正能减少轨迹碎片，并提高 route-conditioned 动态事件召回，而不增加误提醒和错误身份延续。

**实验臂**：

```text
T0: 当前 label/direction/IoU/中心距离时序匹配
T1: IoU + 中心距离 + 类别门控 + alpha-beta filter
T2: ByteTrack-style 高/低置信二阶段关联
T3: OC-SORT-style observation-centric re-update
```

**冻结项**：同一 detector 输出、阈值、帧率、路线 receipt、事件生命周期和 decision kernel；不得在跟踪结果出来后调 NMS、confidence 或 route polygon。

**主指标**：route-conditioned event recall、critical miss、false alerts/min、clearance、repeat alert、identity switch、track fragmentation、evidence age/TTL；按 source/session/scene 和 physical-TTC 分层报告最坏值。

**停止条件**：

- 只改善 IDF1/HOTA 或画面观感，没有改善事件主指标；
- identity switch、离场后错误续轨或 false alerts/min 超过冻结基线；
- 相机旋转、变焦或 pose 无效时仍输出高置信 TTC；
- 为回救结果需要联动调 detector、路线或事件阈值。

框尺度 TTC 只作为单独 proxy：

```text
TTC_h ~= 1 / d(ln h)/dt
TTC_A ~= 2 / d(ln A)/dt
```

它要求目标物理尺度近似固定、检测框稳定并持续接近；必须输出符号、平滑窗、旋转/遮挡降权和不确定度，不能冒充当前米制 closest-approach TTC。

### 3.2 单帧深度与时序深度

**假设 H2**：相较无深度和单帧 DAv2，时序一致深度能减少 route-risk field 翻转和重复提醒，并在最坏来源上改善事件召回或 clearance。

**实验臂**：

```text
D0: 无深度，仅现有 bbox/route 几何
D1: Depth Anything V2 Small 单帧 relative-depth teacher
D2: Video Depth Anything Small 非因果离线上界
D3: 仅当 D2 有事件收益后，提出 history-only 因果学生或在线候选
```

**冻结项**：同一视频、route receipt、depth decode policy、空间网格、calibration split 和 event kernel。D2 使用未来帧的事实必须写入 receipt，禁止与 D0/D1 一起声明实时因果能力。

**主指标**：空间 depth error/排序、temporal alignment/flicker、route-risk flip/min、unknown rate、事件召回、critical miss、false alerts/min、clearance、最坏 source/session/scene。

**停止条件**：

- 只改善深度平滑或公开 depth metric，没有改善 route-event 指标；
- relative depth 被解释为跨视频固定 FAR/MID/NEAR 或米制距离；
- D2 的未来帧泄漏进入 runtime 或因果效果结论；
- 模型缺失、过期或尺度不稳定时仍产生方向性“可走”结论。

### 3.3 自由空间、显式路线与 dense risk field

**假设 H3**：在可信显式路线下，dense walkability/boundary/unknown/risk 表示比固定中心走廊和 DFS 最长自由路径更能区分 matched positive/negative，并减少路线外障碍误提醒。

**实验臂**：

```text
R0: 固定中心图像走廊，仅作历史诊断基线
R1: PathFinder-style 深度自由空间候选
R2: bbox/footprint x explicit route 确定性交互
R3: object-agnostic dense field x explicit route x causal lifecycle
```

**硬边界**：R0/R1 不能被称为用户路线；路线未知、未来、过期或低置信时必须 `route_unknown_or_invalid`，禁止回退中心走廊并输出“左/右安全”。

掩码交互至少区分：

```text
object_intrusion_ratio = |M_i intersect C| / |M_i|
corridor_occupancy_ratio = |M_i intersect C| / |C|
```

两个比例都不能替代米制 clearance、人体包络、TTC、边界不确定性和生命周期。

### 3.4 开放词汇和 VLM

**假设 H4**：开放词汇模型可以提高长尾候选发现或语义解释覆盖，但不应改变相同风险证据下的安全事件开闭。

- YOLOE 只进入 discovery、annotation candidate 或低频 semantic Adapter；
- semantic Adapter 开关前后，安全事件、风险等级、clearance 和 abstain 必须完全一致；
- VLM 只在事件后解释、按需问答或生成离线提醒候选；
- VLM 对物体计数、空间关系或方向意见不能直接成为 `CONTINUE/ADJUST/STOP` authority。

若开放词汇/VLM 改变安全决策，实验因语义泄漏 fail closed；若只提升类别命名但增加错误文案，保留 object-agnostic 提醒。

### 3.5 提醒与触觉

**假设 H5**：在安全事件 truth 已冻结后，时间自适应触发、短行动语言和少量方向触觉可降低重复与认知负担，而不降低关键提醒到达率。

评价维度分开报告，不先压成单个总分：

- actionability / nonactionability；
- sufficiency；
- conciseness；
- afraidness；
- repetition / fatigue；
- event-to-feedback latency；
- critical delivery、误导方向和环境声音遮挡。

WalkVLM、VL-GUIDE 或其他自动 evaluator 只产生模型评价；没有真实 BLV 用户证据时，不得声称用户偏好、事故率或独立行走安全改善。

## 四、研究优先级与进入条件

| 优先级 | 当前动作 | 进入条件 | 退出/冻结条件 |
| --- | --- | --- | --- |
| P0-A | 获取并准入可下载的连续前向研究来源 | 通过普通公开渠道能够下载或自动拉取；先保存 URL、抓取时间和文件哈希即可进入隔离研究，许可、隐私、同步、pose、route 和事件 anchor 不作为下载或初步使用的前置门槛 | 下载失败、文件损坏或内容与研究问题无关才拒绝；缺失字段由自动同步、模型复核、派生标注或降级任务补齐，并在结论中披露，不因收据不全拖慢研究 |
| P0-B | T0-T3 tracker/TTC 固定协议消融 | 至少有 source-grouped 动态轨迹和 matched negative | 只改善 MOT、不改善事件，或误提醒/identity switch 恶化 |
| P1-A | D0-D2 深度 teacher 消融 | 同一视频与 route receipt，可重算 field 与事件 | 无 route-event 增益则 VDA 只保留 teacher/上界角色 |
| P1-B | R0-R3 route/dense risk 消融 | 显式非未来路线和 boundary/unknown 评价可用 | route unknown 仍给方向，或只刷像素/mIoU 不改善事件 |
| P2 | WalkVLM/VL-GUIDE/触觉提醒研究 | 安全事件 truth、反馈 receipt 和延迟测量已冻结 | 自动 judge 冒充用户结果，或反馈降低关键提醒到达率 |
| P3 | 多传感器与完整可穿戴系统 | 独立硬件路线、标定、时钟、外参与同源数据 | 只因单篇论文/单一传感器成绩选择硬件 |

## 五、统一报告模板

每个后续实验至少记录：

```text
hypothesis:
candidate_vs_control:
source_and_license:
train_dev_blind_or_loso_split:
frozen_inputs_and_thresholds:
route_provider_and_causality:
event_truth_or_model_reference_role:
primary_event_metrics:
worst_seed_session_scene:
unknown_abstention_and_failure_modes:
compute_device_latency_and_power_if_measured:
stop_condition:
research_verdict:
production_authority: false
```

成功条件必须同时覆盖主事件指标、最坏分层、重复性和失败关闭。论文中的公开 AP、MOTA、depth error、语言分数、提示识别率或比赛成功率只可作为外部背景，不能单独晋级 BlindAssist 候选。

## 六、引用与版本注意事项

- Video Depth Anything 的正确编号是 `arXiv:2501.12375`；`arXiv:2406.09414` 是 Depth Anything V2。
- WalkVLM 首次提交于 2024-12，后续修订并发表于 ICCV 2025；引用时应明确使用的版本。
- `arXiv:2502.14883` 已撤回并由 `arXiv:2510.00766` 取代；禁止继续把旧 Eye4B 稿件当当前有效基准。
- PathFinder 的 arXiv 页面题名与 PDF 内题名存在版本元数据差异；引用时使用 arXiv ID 和实际读取版本绑定。
- Nature Machine Intelligence 论文的本地副本来自共同作者机构公开 PDF，仅作研究阅读，不对外重新分发。
- 本文是日期化研究指导，不覆盖当前实施状态、AI review governance、promotion gate 或生产授权真源。
