# TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 设计合同

状态：`PROPOSAL_ONLY / EXECUTION_NOT_AUTHORIZED`

协议 ID：`TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT`

版本：`R0`

阶段：`DISCOVERY_DESIGN_ONLY`

## 1. 合同目的与权限

本文件只把一个独立 successor 假设收敛为可审查的设计合同。它不实现算法、不读取
候选输出、不执行数据回放、不接入 Android，也不改变默认、debug 或 release 行为。

本合同不继承以下任何权限或结果：

- scene-scale active R1 或其失败分解；
- D0 ego-motion error attribution R0/R1/R2/R3；
- RCLE 的历史协议、formal identity 或 one-shot authority；
- 既有 shadow、active contradiction-only 构建或 annotation-track Confirmation。

当前唯一允许的动作是：审查、修订并冻结本设计合同。B 阶段 Development 实现、C1
新 session canary、C2 独立复现和任何运行时工作均需后续明确授权。

上位依据：

- [BlindAssist 渐进式研究治理](../../RESEARCH_GOVERNANCE.md)
- [渐进式研究协议模板](../../RESEARCH_PROTOCOL_TEMPLATE.md)
- [双环研究主线](README.md)
- [R1 失败分解](DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)

## 2. 唯一研究问题

> 在 production-selected target、target track 和相同相邻帧输入不变时，目标邻域背景
> 的局部视觉运动能否预测相机诱导的目标框变化，使目标框高度相对于该假设的残余，
> 相比 raw target log-scale，在事件级和独立 session 级产生可重复的方向增量？

这里的 residual 是一个相对图像运动观测，不是真实目标速度、真实接近率、米制 TTC、
碰撞概率、安全证据或产品行为依据。

## 3. Claim ceiling 与明确非目标

### 3.1 允许的最高主张

按阶段只能使用以下受限标签：

```text
B:  DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY
C1: FIRST_NEW_SESSION_SIGNAL_CANARY
C2: CROSS_SESSION_SIGNAL_REPLICATION
```

即使 C2 通过，也只能说明冻结的连续信号在两个独立 session 上出现方向一致的事件级
增量；不得写成主动提醒改善、误提醒下降、实时可用、助行有效、安全或生产授权。

### 3.2 明确不做

本合同不包含：

- Homography、Essential Matrix、模型自动选择或相机内参估计；
- affine、projective、shear 或多模型拟合；
- bearing、中心残余、TTC、深度、分割、SLAM、VIO 或完整光流；
- 神经光流、NPU 调度、风险驱动计算升级或异构运行时；
- hold、latch、事件状态机、阈值策略、反馈策略或 active APK；
- Android、CameraX、QNN、GPU、CPU 热量/时延/内存验证；
- 任何默认生产、真人、独立行走、产品或安全结论。

如果相似变换 canary 不支持，终点为：

```text
SIMILARITY_CANARY_NOT_SUPPORTED
```

不得在本合同内事后升级 affine、Homography、Essential Matrix、bearing、神经光流、
深度或事件状态进行救援。

## 4. 数据角色与访问边界

### 4.1 B：BURNED DEVELOPMENT DISCOVERY

候选来源可以包括已看过 R1/既有几何输出的 CrowdBot、Matoaka 和 Shiraz。它们只允许
用于实现调试、候选选择、失败分类和回归诊断：

```text
role: DEVELOPMENT_DIAGNOSTIC
outcome_access: OUTPUT_INSPECTED / TUNED_ON
claim_ceiling: DEVELOPMENT_ONLY
```

B 不产生 unseen、Confirmation、跨来源泛化、Android 或事件策略效果主张。

B 打开任何候选输出前，必须冻结本合同所列：模型、背景环候选集、特征与 LK 参数、
质量门、pairwise residual、唯一事件汇总器、对照臂、指标、选择规则和停止规则。

### 4.2 C1：首个新 session canary

C1 必须使用一个与 B 不重叠的完整新 session。session、sequence、frame identity、
truth identity 和 ancestry 在任何候选输出打开前以 metadata-only 清单冻结。

C1 的候选输出不得用于重新选择模型、背景环、质量门、汇总器、deadband 或指标。
C1 只回答首个新 session 是否出现信号级增量；C1 失败立即关闭候选，不进入 C2。

### 4.3 C2：第二个独立 session 复现

只有 C1 按本合同通过，才可另行授权 C2。C2 必须与 B、C1 使用独立 session，且在
C1 结束后不得修改任何算法、参数、质量门、汇总器、阈值、评价单位或停止规则。

若 C2 方向不复现，终点为 `NO_CROSS_SESSION_REPLICATION`，候选关闭。

## 5. 因果输入与目标条件

### 5.1 生产选择目标

候选只读取 semantic main loop 已选中的一个 target：

```text
target_id
track_epoch
previous_frame
current_frame
previous_bbox
current_bbox
captured_at_ns_previous
captured_at_ns_current
```

不得读取未来帧、truth、pose、Vicon、oracle ROI、旧 decision 输出或后验事件标签。

### 5.2 目标邻域背景环

对上一帧目标框 `B_(t-1)` 定义带 guard band 的背景环：

```text
Omega_bg = dilate(B_(t-1), r_outer) - dilate(B_(t-1), r_inner)
```

其中 `r_inner < r_outer`，均按 `d = max(width(B), height(B))` 的倍数确定并在 B 前
冻结。环带必须：

- 排除目标框及其 guard band 内部；
- 排除上一帧和当前帧所有已知动态检测框的膨胀区域；
- 排除图像边界外区域；
- 排除低纹理、无效像素、严重模糊与遮挡区域；
- 记录有效环面积、候选点数量和空间分布；
- 点不足或空间退化时整体弃权，不缩小成更小环带补救。

YOLO/检测框在此只作为主环目标和动态区域掩码来源，不构成独立第二感知源。

### 5.3 预注册有限背景环候选集

`dilate` 使用确定性的像素向外取整。B 前只允许以下四个环配置，不得增加未预注册
配置：

| `ring_config_id` | `r_inner / d` | `r_outer / d` |
|---|---:|---:|
| `R1` | `0.10` | `0.50` |
| `R2` | `0.10` | `0.75` |
| `R3` | `0.20` | `0.75` |
| `R4` | `0.20` | `1.00` |

本候选集只用于 Development B 的选择；C1/C2 只能使用 B 按确定性规则选出的一个配置。

## 6. 唯一视觉模型与质量门

### 6.1 特征与 LK

第一版固定为：

```text
feature detector: Shi–Tomasi / goodFeaturesToTrack
max corners: 80
quality level: 0.01
min distance: 5 px
block size: 5 px
LK window: 15 x 15 px
LK pyramid max level: 2
forward-backward error: <= 1.5 px
```

禁止改用神经光流、稠密光流或第二种特征跟踪器。

### 6.2 RANSAC 2D similarity canary

对背景环内上一帧/当前帧有效点只拟合一个二维 similarity transform：

```text
p_t ~= s R p_(t-1) + b
```

它只表达统一尺度、旋转和二维平移。它不是对真实世界局部运动的完整模型，而是
自由度较低、可解释的首个可证伪 canary。

以下数值在 B 打开输出前冻结，不得因单个来源或结果修改：

| 门 | 固定要求 |
|---|---|
| 输入时间 | 两帧 `captured_at_ns` 严格单调，`0 < Δt <= 100 ms` |
| track identity | `target_id` 与 `track_epoch` 相同；reset 即弃权 |
| box validity | 宽高有限且为正；目标框未被图像边界截断 |
| surviving points | 至少 `8` 个通过 LK 与前后向检查的点 |
| inlier ratio | RANSAC 内点率至少 `0.50` |
| reprojection | 内点中位重投影误差不超过 `2.0 px` |
| spatial support | 至少占据四分区中的 `2` 个；任一分区不得超过有效点 `75%` |
| degeneracy | 点集空间协方差非退化，设计矩阵条件数不超过 `100` |
| transform | `s`、`R`、`b` 有限，`s > 0`，不得出现数值溢出或不可逆结果 |
| support consistency | 环带有效面积、点数、内点数和分区计数全部可追溯 |

任一门失败，输出 `ABSTAIN`，并记录唯一、可枚举的 `abstention_reason`。不得通过
减少环带、降低门限、跨 reset 桥接或借用未来帧恢复。

## 7. Camera-induced target-box hypothesis 与 residual

### 7.1 相机诱导目标框假设

将上一帧目标框四角通过背景 similarity warp 外推到当前帧：

```text
B_hat_t_cam = T_bg(B_(t-1))
```

该框只是“若目标区域服从周边背景估计出的相机诱导变换，当前帧应出现的框”这一
假设，不是目标真实运动预测。

### 7.2 唯一连续信号

对当前实际检测框和相机诱导框高度定义：

```text
r_t = (log(h(B_t)) - log(h(B_hat_t_cam))) / Δt
```

约定：

- `r_t > 0`：目标框比相机诱导假设放大得更快；
- `r_t < 0`：目标框比相机诱导假设放大得更慢或相对缩小；
- `r_t = null`：任一输入、模型或质量门失败。

该符号不得写成真实接近率、目标速度或 TTC。

raw 对照只使用同一 target、同一 track epoch、同一相邻帧和同一时间差：

```text
r_t_raw = (log(h(B_t)) - log(h(B_(t-1)))) / Δt
```

候选与 raw 必须共享输入身份和固定事件分母；候选的额外弃权单独报告，不得删除。

## 8. 唯一事件汇总器

为避免隐性窗口搜索，R0 不在 B 中搜索汇总窗口、deadband 或状态机。唯一汇总规则
固定为：

```text
event_score = median(all finite r_t within the fixed parent event)
```

事件最低要求：

- 至少 `3` 个有限 pairwise residual；
- 有效 pair 数 / truth-eligible pair 数至少 `0.50`；
- 所有缺失、reset、质量失败和弃权仍留在事件分母；
- 不跨越时间间隔、track epoch、目标身份或 truth-event 边界；
- 不能用 3/5/7 帧轮换、中位数/OLS/最大值试探或 deadband 搜索救援。

R0 的方向诊断只使用预注册的对称 deadband `0.02 / s`：

```text
approach      : event_score > +0.02 / s
quasi-static  : -0.02 / s <= event_score <= +0.02 / s
receding      : event_score < -0.02 / s
```

该 deadband 是图像信号诊断门，不是物理速度阈值或安全阈值。

## 9. 事件、session 与对照评价

### 9.1 统计层级

```text
source -> session -> target -> parent event -> frame pair
```

primary unit 是 parent event；session 是独立复现单位；frame/pair 是纵向观测，不是
独立样本。禁止随机拆帧、把同一长视频切 clip 或用 pair 数膨胀样本量。

### 9.2 固定事件分母

每个 truth-eligible parent event 都保留在 raw 与 residual 两臂分母中。完全弃权、
不可评价或质量不足的事件不得从分母删除，必须单独计数为 `abstained_event`。

至少报告：

- parent-event 总数、可评价数、弃权数；
- correct direction count/fraction；
- wrong-signed event count/fraction；
- event coverage；
- residual 与 raw 的 paired event difference；
- 每个 session、target、truth state 和场景类别的分层结果；
- 每个 abstention reason 的数量和分母。

不报告帧池化 AUROC/F1、伪独立置信区间或安全概率。

## 10. B 阶段确定性选择规则

B 只允许比较 `R1`–`R4` 四个背景环配置；similarity、LK、质量门、residual 公式、
event median、deadband、raw 对照和评价分母均固定不变。

候选选择在打开 B 输出前冻结为以下字典序：

1. 先排除未达到事件最低可评价要求的配置；
2. 最大化各独立 B session 中 `residual_correct - raw_correct` 的最小值；
3. 若并列，最大化所有 B session 的 paired event gain 总和；
4. 若仍并列，最大化中位 event coverage；
5. 若仍并列，选择较小的环面积（`r_outer` 更小）；
6. 仍无法唯一选择则终止 B，不人工裁决。

只有同时满足以下 Development 条件，才允许产生 C1 候选：

- 至少两个独立 B session 的 paired event gain 为正；
- 任一 B session 的 paired event gain 不为负；
- residual 的 wrong-signed 事件不高于 raw；
- residual 的事件 coverage 相比 raw 不损失超过 `5%`；
- 增量不能由单一 parent event 独占。

若无配置满足，终点为 `NO_DEVELOPMENT_INCREMENT / CLOSE_CANDIDATE`。不得扩大候选集、
换模型、改汇总器或读取新的来源进行救援。

## 11. C1 与 C2 晋级门

### 11.1 C1

C1 只使用 B 已选出的一个冻结配置。C1 source 必须在 metadata-only 阶段具备可解析的
truth/event identity、至少一个正向与一个非正向事件，并满足固定 event/session 分母。

C1 通过条件：

- residual 相对 raw 的 paired event gain 为正；
- 没有 wrong-signed 增加；
- event coverage 损失不超过 `5%`；
- 结果不是由单一 event 或单一 target 独占；
- identity、时间、truth join、输出 schema 和 abstention 账本均有效。

C1 通过只产生：

```text
FIRST_NEW_SESSION_SIGNAL_CANARY
```

不自动授权 C2。

### 11.2 C2

C2 必须使用与 B、C1 独立的 session，并完整复用 C1 冻结配置。C2 通过需要：

- residual 相对 raw 的 paired event gain 方向与 C1 一致；
- C2 没有 wrong-signed 增加；
- C2 event coverage 损失不超过 `5%`；
- C1 与 C2 两个 session 均出现非零正向增量；
- 增量不由单个 event、target 或 source 层级伪造。

C2 通过最多产生：

```text
CROSS_SESSION_SIGNAL_REPLICATION
```

仍不产生 active policy、Android、产品或安全权限。

## 12. 输出与审计字段

未来任何实现或离线 producer 必须逐 pair 记录至少：

```text
protocol_id
implementation_id
parameter_set_id
source_id
session_id
sequence_id
previous_source_frame_id
current_source_frame_id
previous_captured_at_ns
current_captured_at_ns
target_id
track_epoch
previous_bbox
current_bbox
ring_config_id
model_id
predicted_bbox_cam
raw_rate_per_s
residual_rate_per_s
feature_count
surviving_track_count
inlier_count
inlier_ratio
median_forward_backward_error_px
median_reprojection_error_px
spatial_support
condition_number
quality
abstention_reason
```

`residual_rate_per_s` 在弃权时必须为 `null`；非弃权时必须有限。truth、event label、
R1/D0 输出和任何后验策略结果只能由独立 evaluator 读取，不能进入 producer。

## 13. 终点与失败学习

本合同采用最小失败范围：

| 条件 | 终点 | 关闭范围 |
|---|---|---|
| 设计、身份、schema 或因果规则不一致 | `INVALID_DESIGN` | 本合同版本 |
| B 无候选满足确定性选择门 | `NO_DEVELOPMENT_INCREMENT` | 本候选问题 |
| similarity 质量支持不足或系统性不稳定 | `SIMILARITY_CANARY_NOT_SUPPORTED` | similarity 候选 |
| C1 无新 session 信号增量 | `C1_NO_SIGNAL_INCREMENT` | C1/C2 路线 |
| C2 不复现 C1 方向 | `NO_CROSS_SESSION_REPLICATION` | 本 successor |
| 输入、truth、identity 或输出账本不可评价 | `NOT_EVALUABLE` | 对应 evidence version |

任何失败结果都必须记录 observation、supported inference、alternative explanations、
被挑战约束、可复用回归夹具和下一条新假设；不得改写 R1、D0 或本合同之前的历史终点。

## 14. 修改与授权规则

```text
outcome_access_started: false
amendment_mode: NEW_VERSION_ONLY_AFTER_B_OUTPUT_ACCESS
```

在 B 输出打开前，设计评审可修订本 R0；一旦 B 输出访问开始，任何模型、环配置、
质量门、汇总器、deadband、指标、分母或停止规则变更都必须新建版本，不得原地修改。

以下动作永远不会由本合同自动授权：

```text
B implementation
B output access
C1 execution
C2 execution
Android / shadow / active APK
production behavior change
Confirmation / Deployment
```

## 15. 当前状态摘要

```text
protocol_status: DESIGN_FROZEN_CANDIDATE / PROPOSAL_ONLY
scientific_status: NOT_RUN
claim_eligibility: CLAIM_NOT_SIGNABLE
execution_authority: NOT_AUTHORIZED
runtime_status: NOT_RUN
data_status: NOT_RUN
next_authorizable_action: INDEPENDENT_DESIGN_REVIEW
```

本文件落地不创建实现、runner、validator、receipt、候选输出、Android 接线或实验
产物。任何后续动作必须引用本合同版本并获得独立明确授权。
