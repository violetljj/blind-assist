# TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 设计合同

状态：`DESIGN_REVIEW_PASS / PROPOSAL_ONLY / EXECUTION_NOT_AUTHORIZED`

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

C2 在任何候选输出打开前，也必须以 metadata-only manifest 冻结 session、sequence、
frame、truth/event identity、source/session ancestry 和不重叠证明；其 truth state
构成与至少 `2` 个 parent event、`2` 个 `target_id` 的最低分母门沿用 C1。manifest
不足或 ancestry 无法复核时，C2 只能终止为 `NOT_EVALUABLE`。

若 C2 方向不复现，终点为 `NO_CROSS_SESSION_REPLICATION`，候选关闭。

## 5. 因果输入与目标条件

### 5.1 生产选择目标

候选只读取 semantic main loop 已选中的一个 target：

```text
target_id
track_epoch
previous_source_frame_id
current_source_frame_id
previous_frame_index
current_frame_index
previous_frame
current_frame
previous_bbox
current_bbox
captured_at_ns_previous
captured_at_ns_current
previous_dynamic_bboxes
current_dynamic_bboxes
previous_frame_shape
current_frame_shape
```

`previous_dynamic_bboxes` 与 `current_dynamic_bboxes` 只能来自同一 semantic detector
在对应帧已经产生的全部检测框，只携带 bbox 与预注册的 dynamic-mask 类别位并只用于
背景掩码；不得读取分数或任何后验 decision 字段，也不得把它们当作第二感知源。`previous_frame_index` 与
`current_frame_index` 必须是同一 sequence 的严格相邻帧（`current = previous + 1`）；
无法证明相邻关系时整体弃权，不以 `Δt` 窗口替代。不得读取未来帧、truth、pose、Vicon、
oracle ROI、旧 decision 输出或后验事件标签。

坐标和图像输入在本合同内固定为：原点在左上、`x` 向右、`y` 向下；bbox 使用原生解码
像素坐标的半开区间 `[x0, y0, x1, y1)`，`width=x1-x0`、`height=y1-y0`；LK 使用
对应帧的 8-bit 单通道 luma。前后帧宽高必须完全相同且为正；禁止隐式 resize、crop、
padding 或旋转补偿，shape 不同即 `IMAGE_SHAPE_MISMATCH`。所有 frame、detection
manifest 和 shape 信息必须由输入 receipt 绑定。

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
- 无效像素和动态框以固定掩码排除；低纹理、模糊或遮挡不引入未定义的 oracle，统一由
  下方固定的特征、LK、空间支持和重投影门表现为弃权；
- 记录有效环面积、候选点数量和空间分布；
- 点不足或空间退化时整体弃权，不缩小成更小环带补救。

对每个 `ring_config_id`，动态检测框的膨胀半径固定为该配置的 `r_inner`（按同一
`ceil` 像素规则、分别在上一/当前帧坐标中裁切）；不得使用检测分数、类别置信度或
后验策略再调整掩码。

YOLO/检测框在此只作为主环目标和动态区域掩码来源，不构成独立第二感知源。

### 5.3 预注册有限背景环候选集

`dilate` 对四个边界均使用 `ceil` 向外取整并在图像边界裁切；B 前只允许以下四个环配置，不得增加未预注册
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
LK termination: COUNT=20, EPS=0.03
forward-backward error: <= 1.5 px
```

禁止改用神经光流、稠密光流或第二种特征跟踪器。

`goodFeaturesToTrack` 只在上一帧有效环掩码内运行；输入 luma、像素中心坐标、边界
模式和 LK 终止条件必须由 `model_id=SHI_TOMASI_LK_V1` 固定，不能按来源切换。实现
不得把当前帧检测结果用于补点或重新选择特征。

### 6.2 RANSAC 2D similarity canary

对背景环内上一帧/当前帧有效点只拟合一个二维 similarity transform：

```text
p_t ~= s R p_(t-1) + b
```

它只表达统一尺度、旋转和二维平移。它不是对真实世界局部运动的完整模型，而是
自由度较低、可解释的首个可证伪 canary。

为保证跨实现可复算，`model_id=SIMILARITY_RANSAC_PROCRUSTES_V1` 的确定性细节为：

- 坐标使用像素中心；最小样本为两个不重合点；RANSAC 重投影门为 `2.0 px`，最多
  `1000` 次采样，置信度 `0.99`，不做额外 refine，伪随机种子为 `0`；
- 每个候选模型先按内点数降序、内点中位重投影误差升序、样本索引字典序选择；最终
  在选定内点上用固定的闭式 similarity 拟合，要求 `det(R)=+1`，不接受 reflection；
- `condition_number` 定义为居中上一帧点集二维协方差的最大/最小特征值比；最小
  特征值非正或非有限即视为退化。不得以库默认随机状态、隐式 refine 或其他 tie-break
  替代这些规则。

以下数值在 B 打开输出前冻结，不得因单个来源或结果修改：

| 门 | 固定要求 |
|---|---|
| 输入时间 | 两帧 `captured_at_ns` 严格单调，`0 < Δt <= 100 ms`，且 frame index 相邻 |
| track identity | `target_id` 与 `track_epoch` 相同；reset 即弃权 |
| image shape | 前后帧原生宽高完全相同；shape 不同即弃权 |
| box validity | 宽高有限且为正；目标框未被图像边界截断，且四角在图像内 |
| surviving points | 至少 `8` 个通过 LK 与前后向检查的点 |
| inlier ratio | RANSAC 内点率至少 `0.50` |
| reprojection | 内点中位重投影误差不超过 `2.0 px` |
| spatial support | 至少占据四分区中的 `2` 个；任一分区不得超过有效点 `75%` |
| degeneracy | 上一帧点集二维协方差非退化，`condition_number <= 100` |
| transform | `s`、`R`、`b` 有限，`s > 0`，不得出现数值溢出或不可逆结果 |
| predicted quad | warp 后四角及其轴对齐 bbox 有限且完全在图像内；越界不裁剪，直接弃权 |
| support consistency | 环带有效面积、点数、内点数和分区计数全部可追溯 |

数字门的治理说明在 B 输出访问前冻结：`8` 个点是 similarity 最小可拟合规模并兼顾
四分区支持；`1.5 px` 前后向误差和 `2.0 px` 重投影误差控制像素跟踪噪声与混合运动；
`0.50` 内点率、`2/4` 分区和 `<=75%` 单区占比控制背景点退化；`condition_number<=100`
拒绝近共线点集；`3` 个 pair 与 `0.50` pair coverage 防止单 pair 事件冒充信号；
`0.02/s` 仅定义图像信号的中性带；`5%` coverage loss 防止候选靠弃权制造增量。其
calibration source 只允许固定 synthetic known-transform/zero-motion fixtures 与
治理规则，B 结果不得用于校准；邻近阈值只可作独立敏感性诊断，不得改写本 R0。任何
阈值修改必须建立新版本并重新进行设计复核。

任一门失败，输出 `ABSTAIN`，并记录唯一、可枚举的 `abstention_reason`。多门同时失败时
按以下固定优先级只记录第一个原因：

```text
INPUT_TIMESTAMP_INVALID > FRAME_ADJACENCY_INVALID > IMAGE_SHAPE_MISMATCH
> TRACK_ID_MISMATCH > BOX_INVALID > BOX_BOUNDARY_TRUNCATED
> DYNAMIC_MASK_INVALID > RING_EMPTY_OR_LOW_AREA > FEATURE_COUNT_LOW
> LK_TRACK_COUNT_LOW > SPATIAL_SUPPORT_LOW > GEOMETRY_DEGENERATE
> RANSAC_INLIER_RATIO_LOW > REPROJECTION_ERROR_HIGH > TRANSFORM_INVALID
> PREDICTED_BOX_INVALID > NUMERIC_NONFINITE
```

不得通过减少环带、降低门限、跨 reset 桥接、裁剪越界框或借用未来帧恢复。

## 7. Camera-induced target-box hypothesis 与 residual

### 7.1 相机诱导目标框假设

将上一帧目标框四角通过背景 similarity warp 外推到当前帧：

```text
B_hat_t_cam = T_bg(B_(t-1))
```

该框只是“若目标区域服从周边背景估计出的相机诱导变换，当前帧应出现的框”这一
假设，不是目标真实运动预测。

四角按半开 bbox 的四个几何角点 `(x0,y0)、(x1,y0)、(x1,y1)、(x0,y1)` 变换；
`predicted_bbox_cam` 是四个变换点的轴对齐半开包围框，不做取整或图像裁剪。任一
变换点或包围框越界、宽高非正或非有限时，该 pair 必须弃权。

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
raw 只有在同一组时间、identity、bbox 和 shape 门通过时才是有限值；否则 `r_t_raw=null`。

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

`truth-eligible pair` 是已通过 source/session/sequence/target/track/event identity
join 且具有 source-native truth state 的相邻 pair；它不依赖 residual 或 raw 是否有限。
`truth state` 只允许 `approach`、`quasi-static`、`receding`，由独立 evaluator 提供。
raw 与 residual 的 event score 都是各自有限 pair 的中位数；有限 pair 不足 3 或有效
pair 比例不足 `0.50` 的 event 记为 `NOT_EVALUABLE_EVENT`，仍留在固定分母。

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

`correct_direction` 定义为 event score 的三态分类与 truth state 相同；`ABSTAIN` 或
`NOT_EVALUABLE_EVENT` 记为不正确但不计入 `wrong-signed`。定义
`coverage = evaluable_event / truth_eligible_event`，
`paired_event_gain_count = residual_correct_count - raw_correct_count`，
`paired_event_gain = paired_event_gain_count / truth_eligible_event_count`；coverage
损失是 `raw_coverage - residual_coverage` 的绝对比例，必须 `<= 0.05`。两臂始终使用
同一个 truth-eligible event 分母。

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
2. 最大化各独立 B session 中 `paired_event_gain_count` 的最小值；
3. 若并列，最大化所有 B session 的 paired event gain 总和；
4. 若仍并列，最大化中位 event coverage；
5. 若仍并列，选择较小的实际环面积（按当前 bbox 与图像边界裁切后的有效像素数）；
6. 仍无法唯一选择则终止 B，不人工裁决。

只有同时满足以下 Development 条件，才允许产生 C1 候选：

- 至少两个独立 B session 的 paired event gain 为正；
- 任一 B session 的 paired event gain 不为负；
- residual 的 wrong-signed 事件不高于 raw；
- residual 的事件 coverage 相比 raw 不损失超过 `5%`；
- 至少有 `2` 个 parent event 对 paired gain 作出正向贡献，且任何单一 parent event
  的正向贡献不超过全部正向贡献的 `50%`。

若无配置满足，终点为 `NO_DEVELOPMENT_INCREMENT / CLOSE_CANDIDATE`。不得扩大候选集、
换模型、改汇总器或读取新的来源进行救援。

## 11. C1 与 C2 晋级门

### 11.1 C1

C1 只使用 B 已选出的一个冻结配置。C1 source 必须在 metadata-only 阶段具备可解析的
truth/event identity、至少一个 `approach` 与一个 `quasi-static`/`receding` 事件、至少
`2` 个 parent event、至少 `2` 个 `target_id`，并满足固定 event/session 分母；否则为
`NOT_EVALUABLE`，不以小样本晋级。

C1 通过条件：

- residual 相对 raw 的 paired event gain 为正；
- 没有 wrong-signed 增加；
- event coverage 损失不超过 `5%`；
- 至少 `2` 个 parent event 与 `2` 个 `target_id` 对正向 paired gain 有贡献，且单一
  event 或 target 的正向贡献均不超过 `50%`；
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
- 两个 session 均满足至少 `2` 个 parent event、`2` 个 `target_id` 与单项贡献 `<=50%`；
- 增量不由单个 event、target 或 source 层级伪造，且 source/session ancestry 可复核。

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
previous_frame_shape
current_frame_shape
previous_captured_at_ns
current_captured_at_ns
target_id
track_epoch
previous_bbox
current_bbox
ring_config_id
model_id
input_manifest_sha256
detection_manifest_sha256
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
R1/D0 输出和任何后验策略结果只能由独立 evaluator 读取，不能进入 producer。字段枚举
固定为：`quality ∈ {PASS, ABSTAIN}`；`spatial_support` 为
`{occupied_quadrants: integer 0..4, max_quadrant_fraction: finite [0,1]}`；
`abstention_reason` 为上方优先级列表中的一个值，非弃权时为 `null`。`parameter_set_id`
必须绑定本合同、model、ring 和实现版本的 SHA-256；producer 还须记录 frame shape、
有效环面积、动态掩码框数与候选/内点计数。library/version、随机种子和输入/检测
manifest hash 由 implementation lock 绑定，不能由 evaluator 事后补写。

## 13. 终点与失败学习

本合同采用最小失败范围：

| 条件 | 终点 | 关闭范围 |
|---|---|---|
| 设计、身份、schema 或因果规则不一致 | `INVALID_DESIGN` | 本合同版本 |
| B 无候选满足确定性选择门 | `NO_DEVELOPMENT_INCREMENT` | 本候选问题 |
| 四个 ring 配置在每个 B session 均无法达到事件最低可评价要求 | `SIMILARITY_CANARY_NOT_SUPPORTED` | similarity 候选 |
| C1 无新 session 信号增量 | `C1_NO_SIGNAL_INCREMENT` | C1/C2 路线 |
| C2 不复现 C1 方向 | `NO_CROSS_SESSION_REPLICATION` | 本 successor |
| 输入、truth、identity 或输出账本不可评价 | `NOT_EVALUABLE` | 对应 evidence version |

任何失败结果都必须记录 observation、supported inference、alternative explanations、
被挑战约束、可复用回归夹具和下一条新假设；不得改写 R1、D0 或本合同之前的历史终点。

若至少一个配置达到事件最低可评价要求、但没有配置满足 B 的 Development 晋级门，
唯一终点为 `NO_DEVELOPMENT_INCREMENT`；不得在两者之间事后择一。

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
protocol_status: DESIGN_REVIEW_PASS / PROPOSAL_ONLY
scientific_status: NOT_RUN
claim_eligibility: CLAIM_NOT_SIGNABLE
execution_authority: NOT_AUTHORIZED
runtime_status: NOT_RUN
data_status: NOT_RUN
next_authorizable_action: EXPLICIT_B_DEVELOPMENT_AUTHORIZATION
```

本文件落地不创建实现、runner、validator、receipt、候选输出、Android 接线或实验
产物。任何后续动作必须引用本合同版本并获得独立明确授权。
