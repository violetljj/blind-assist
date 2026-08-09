# BlindAssist Assistive Geometry R2 因子化几何假设协议

状态：`PRE_OUTCOME_HYPOTHESIS_FROZEN / OUTCOME_BLIND_DESIGN_ONLY / F0_NOT_STARTED / ALL_EXECUTION_NOT_AUTHORIZED`

日期：`2026-08-09`

机器协议：[JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.json)

研究模式：`WILD_LAB / CANARY_LITE / F0`

## 1. 本协议只冻结一个新假设

R2 要检验的不是“给 B1-A0 换 seed、loss 或训练时长”，而是一个系统角色不同的构造：

```text
RGB + K / transform / gravity validity
  -> replaceable lightweight encoder or initialization
  -> continuous factor evidence
       metric-ish depth + scale uncertainty
       support / ground surface + support uncertainty
       obstacle boundary / evidence + localization uncertainty
  -> versioned deterministic body-swept interval geometry
  -> clearance interval + left/center/right occupancy evidence + UNKNOWN
```

`DepthART` 是优先但可替换的轻量初始化或 encoder 候选，不是路线终点，也不提供任务
authority。学习图只产生连续因子及其 validity/uncertainty；只有版本化
`GeometryR2Reducer` 可以产生 clearance、occupancy 和最终
`CLEAR_OBSERVED / OCCUPIED_OBSERVED / UNKNOWN`。

本协议没有运行实现、训练、数据物化、教师、时序、导出、设备或 outcome。当前可签署的
唯一事实是：R2 假设、接口、最小执行顺序和 authority ceiling 已在 outcome 前写明。

## 2. B1-A0 与历史路线边界保持不变

`B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES` 是永久保留的 Development 负终态。
R2 不重评、不放宽、不修补它，也不激活旧 A1–A4。B1 已消费的 Development Selection
不得进入 R2 调参、选模、阈值标定或晋级；B1 Calibration 与 Confirmation 继续封存，
不会转给 R2。

B1 失败 anatomy 只允许提供设计问题，例如“尺度、support、boundary 或聚合中哪一层造成
保守偏差”。它不能提供 R2 候选排序、阈值、停止决策或新鲜证据。即使 anatomy 后续发现
某种集中错误，也只能用于提出另一个 pre-outcome 版本，不能回写本协议。

R2 也不得重建旧 Clearance-Student S1/S1.1：禁止随机/mobile screen 上的 direct multitask
target、禁止消费旧 120-frame screen、禁止在几何输出覆盖不完整时计算 task score。历史
USTRF 的可复用方法论只是一条边界：关闭一个失败构造不关闭算法研究，但新版本必须具有
独立信号假设与独立证据；它不继承旧 threshold、selection outcome 或 execution authority。

R2 的实质变化维度是：

| 维度 | B1/旧 direct-task 构造 | R2 |
| --- | --- | --- |
| `INPUT_SIGNAL` | depth 或从 RGB 直接到终端任务量 | depth、support、boundary 三类连续因子及 uncertainty |
| `SYSTEM_ROLE` | 学习图可承担最终 task target | 学习图只提供证据；确定性 reducer 独占最终状态 |
| `COMPENSATION` | 离散任务 loss 或阈值吸收误差 | 显式传播 scale/support/boundary interval，歧义进入 UNKNOWN |
| `EVALUATION_TARGET` | 以最终任务量为首个可见成败 | 先验因子完整性和连续可学性，再看 task Pareto |
| `DATA` | 已消费 Selection | 只有 F0/F1 通过后才分配全新的 R2 Development Selection |

## 3. 表示合同

### 3.1 输入与可替换 encoder

输入必须绑定 display-upright 全有效 FOV、动态 `K_tensor`、crop/rotation/resize receipt、
gravity 或其 validity、capture identity 与 timestamp。缺 K、变换、实际 shape 或非有限输入时，
该帧只能产生 `UNKNOWN_INPUT_GEOMETRY`，不能使用 nominal K 回退。

encoder adapter 必须声明 feature strides/channels、normalization、orientation 和
checkpoint/initialization identity。换 encoder 不得改变 factor schema 或 reducer；任何 encoder
都禁止输出 final-task logits。

### 3.2 连续因子

| 因子 | 最小输出 | 语义和单位 | 必须分开的 uncertainty |
| --- | --- | --- | --- |
| metric-ish depth | `depth_shape_positive[H,W]`、`log_metric_scale_m`、`depth_log_sigma[H,W]`、valid probability | valid shape 的几何均值归一为 1；`exp(log_metric_scale_m) × shape` 得到米制 depth | frame scale 与 per-pixel depth residual |
| support surface | support probability、camera-frame plane normal、camera height、support residual sigma、valid | support/ground 证据、单位法向、米制相机高度 | normal、height、surface residual 与 gravity validity |
| obstacle boundary/evidence | obstacle evidence probability、boundary probability、boundary localization sigma、valid | 连续 non-support/depth-discontinuity 证据；boundary target 使用连续 distance-to-boundary，不是最终 blocked 标签 | evidence validity 与像素定位 interval |

每个字段都必须有显式 validity。`NaN/Inf`、缺 field 或 invalid factor 不能参与 observed
clear/occupied。禁止的 learned outputs 包括：direct clearance、direct occupancy、task
confidence、free/blocked 和 learned final UNKNOWN。

R2 在任何任务门之前要求连续因子监督：admitted evaluation frame 必须同时具有 depth、
support 和 boundary/evidence 的连续 target，或在 admission 前带明确的 non-evaluable reason；
不得在计算 task score 时临时删掉缺因子样本。

## 4. 确定性 `GeometryR2Reducer`

reducer 的输入是 factor tensors、全部 uncertainty/validity、相机几何和版本化 research body
profile。处理顺序固定为：

1. 用绑定的 `K` 把有效 metric depth 重投影到相机三维；
2. 在 gravity、plane、height 与 residual uncertainty 下构造 support-surface interval；
3. 从 non-support 与 depth discontinuity 构造 obstacle evidence，并按 boundary localization
   uncertainty 建立空间 interval；
4. 与 left/center/right 的 body-swept volume 在冻结 horizons 上求交；
5. 生成每个 band/horizon 的 clearance interval、occupancy evidence 和 coverage；
6. 只有所有 admissible geometry 都 clear 时输出 `CLEAR_OBSERVED`；只有所有 admissible
   geometry 都 occupied 时输出 `OCCUPIED_OBSERVED`；其余输出 `UNKNOWN` 和 reason code。

这是一种三值 interval 决策。scale 不确定、support plane 模糊或 boundary 跨阈值时进入
`UNKNOWN`，不是自动写成 blocked。这样既 fail-closed，又不允许“把不知道全部伪装成堵塞”来
购买较低 false-clear。

reducer 输出必须携带 factor identity、encoder/checkpoint、transform receipt、body profile 和
reducer version。相同输入字节与相同版本必须产生相同序列化 `GeometryState`。

## 5. 必须先由 F0 证明的几何不变量

| ID | 不变量 | 失败范围 |
| --- | --- | --- |
| `R2-I01` | reducer 是 clearance、occupancy 与最终三态的唯一 producer | implementation version |
| `R2-I02` | 缺 K/transform/scale/support/boundary 或非有限 factor 只能 UNKNOWN | item / implementation |
| `R2-I03` | UNKNOWN 从不作为 clear、occupied 或 negative supervision | implementation version |
| `R2-I04` | 增大 horizon 不得提高最小 clearance，也不得把近场 occupied 变成远场 clear | implementation version |
| `R2-I05` | 增大 body width、margin、obstacle evidence 或 uncertainty 不得把 occupied/UNKNOWN 变 clear | implementation version |
| `R2-I06` | crop/rotation/resize 与同步变换的 K 保持 metric geometry equivariance | implementation version |
| `R2-I07` | clear 与 occupied interval 不重叠；跨阈值或区间重叠只能 UNKNOWN | implementation version |
| `R2-I08` | 相同 factors、receipt、profile 和 reducer version 得到确定性重放 | implementation version |
| `R2-I09` | left/center/right swept-volume ownership 无缝且不重叠 | implementation version |
| `R2-I10` | 学习图和 loss 中不存在 final clearance/occupancy/free/blocked/UNKNOWN shortcut | implementation version |

上述是 mechanics，不是 utility evidence。F0 即使全部通过，也不能说明真实图像可学或可通行。

## 6. 防止 all-blocked collapse 的任务门

R2 不使用一个加权总分在 false-clear 与 false-block 之间交换。所有任务门都是 conjunctive，
并且在连续因子与输出完整性前门通过后才允许计算。

### 6.1 完整性前门

| 指标 | 门 |
| --- | ---: |
| admitted frame 的 factor-supervision contract coverage | `== 1.0` |
| serialized factor-output field coverage | `== 1.0` |
| finite geometry-output contract coverage | `== 1.0` |
| undefined 或 zero denominator | `FAIL_NOT_DROP` |

这里的 `1.0` 表示 schema 与 admission 完整，不要求所有像素都可观测。像素不可观测必须由
validity 明示，frame 若缺少任何任务必需 factor truth，则在 outcome 前就不得进入 task
evaluation denominator。

### 6.2 coverage / false-block / false-clear Pareto

F1 必须在 TRAIN-only sensitivity 上、F2 roster 内容和 outcome 打开前冻结：

```text
C_min   = class-conditional known-coverage lower bound
FB_max  = false-block | truth-clear upper bound
FC_max  = false-clear | truth-occupied upper bound
delta_C / delta_FB / delta_FC = causal-anchor non-inferiority margins
```

这些数值只允许来自 F0 synthetic corruption envelope、显式 research body-profile task-cost
假设和 F1 TRAIN-only sensitivity。不得复制 B1 threshold，不得根据 B1 Selection、R2 F2 或
Clearance-Student outcome 调整。即使数值偶然相同，也必须有独立 provenance、单位、敏感性和
revision policy。

候选必须同时满足：

1. truth-clear 和 truth-occupied 两类的 known coverage 都 `>= C_min`；
2. `false_block | truth-clear <= FB_max`；
3. `false_clear | truth-occupied <= FC_max`，并同时报告 all-known 口径；
4. 相对同 encoder/depth、同 reducer、但 support/boundary 为冻结非学习估计的 causal anchor，
   coverage、false-block、false-clear 三轴都在预冻结 non-inferiority margin 内；
5. 在 false-block、false-clear、clearance interval error、support error 或 boundary error 中至少
   一个预声明轴严格改善；
6. coverage 下降不能购买 false-clear 或 false-block 改善。

完整 surface 与邻近 operating points 全部报告，不能看到 F2 后选择点。degenerate control 的
预期是确定的：all-blocked 必须因 false-block 失败，all-clear 必须因 false-clear 失败，
all-UNKNOWN 必须因 coverage 失败。

## 7. Minimal-First 执行梯

### F0：synthetic/unit factor geometry canary

只允许 analytic synthetic scenes：平/斜 support、空 corridor、孤立与边缘 obstacle、scale
扰动、support ambiguity、boundary uncertainty、缺 K 和所有支持的 orientation。实现 factor
schema、reducer 与单元 fixture，不使用 learned model 或真实数据。

通过条件：十项不变量、factor schema coverage 和 geometry-output coverage 全部通过；analytic
clearance interval/tri-state 在执行前冻结的数值 tolerance 内；degenerate controls 全被门禁拒绝。
失败只关闭该 reducer implementation version。通过只允许冻结 F1 协议。

### F1：TRAIN-only factor learnability canary

F1 必须先把 `R2_TRAIN_FIT` 与 `R2_TRAIN_CANARY` 按 parent/session 分开。它可在来源和许可审计后
使用有明确 TRAIN role 的连续 factor evidence，但禁止任何 B1 Development、Calibration 或
Confirmation。至多训练一个最小 factorized candidate 和一个 causal depth-only factor anchor；
不允许 direct task head，也不允许靠增加 seed/epoch 搜索。

先检验三类连续 factor 各自相对固定常数/非学习 control 的 parent-macro error 改善，其 lower
confidence bound 必须大于 0；再检验完整性前门；最后才运行 TRAIN-only task Pareto。任何前门
失败都在分配 fresh Development 之前停止 factor learnability hypothesis。F1 positive 只说明
TRAIN-only learnability，不说明泛化或可用性。

### F2：全新 Development Selection

只有 F0、F1 依次通过，才允许建立一个全新的、至少 8 个 independent parent 的 metadata-first
roster。它必须与 R2 TRAIN 及 B1 所有角色在 parent/session/capture 上 disjoint；replacement 只能
基于 license、identity、内容完整性和输入几何，不得基于 truth 或 target-model outcome。

打开任何 F2 outcome 前必须冻结一个 candidate、一个 causal anchor、factor/reducer identity、
body profile、horizons、全部数字 threshold/margin、三个新 seed、聚合、缺失处理和 zero-denominator
规则。不得选 best seed；至少 2/3 seed 同方向且 median seed 通过。

F2 fail 只关闭 R2 factorized candidate version，不关闭“助盲几何”研究问题，也不给重复训练、改门
或复用 roster 的许可。F2 pass 也只允许另写 Calibration/Confirmation proposal，不自动分配数据。

## 8. 数据与 outcome 防火墙

| Role | 当前状态 | 允许用途 | 禁止 |
| --- | --- | --- | --- |
| `R2_F0_SYNTHETIC_FACTOR_FIXTURES` | uncreated | mechanics/unit/regression | utility 或真实任务 claim |
| `R2_F1_TRAIN_LEARNABILITY` | unallocated | fit + parent/session-disjoint TRAIN canary | selection、Confirmation、安全 |
| `R2_F2_DEVELOPMENT_SELECTION` | unallocated | 一次 Development candidate/anchor evaluation | B1 identity、重复调参、Confirmation |
| `R2_CALIBRATION` | unallocated | 无 | 在 F2 pass 与新 pre-outcome 协议前分配/打开 |
| `R2_CONFIRMATION` | unallocated | 无 | 用户未明确激活时创建或读取 |

B1 Calibration/Confirmation 并不是 R2 reserve，它们永远不转入 R2。F2 第一次 task outcome 后，
该 roster 对受其影响的候选永久是 consumed Development。

## 9. Teacher-oracle 只作 upper bound

未来 teacher 必须输出同一 factor schema，并走同一个 deterministic reducer。teacher/oracle 从不
是真值，最终 task state 也不得成为 pseudo-label。teacher factor 必须先生成并冻结，再 join truth。

任何 heterogeneous distillation 前，新的 `R2_TEACHER_DIAGNOSTIC` cohort 上必须同时通过：

1. 每个 teacher 的 factor 和 geometry-output contract coverage 都为 `1.0`；
2. truth-bound oracle 相对最佳单 teacher 的 Pareto gain，其 parent-bootstrap lower confidence
   bound `> 0`；
3. 每个 teacher 至少在两个 independent parent 上 exclusive-correct；
4. disagreement 区域的 truth error enrichment lower confidence bound `> 0`；
5. oracle factor benefit 仍通过同一 coverage/false-block/false-clear conjunctive gate；
6. teacher generation 不读取 target-roster truth、final task state 或 post-truth selection。

任一失败都关闭该 teacher set 的蒸馏主张；全部通过也只允许另写 factor-distillation 协议。当前
teacher identity、cohort、execution、oracle evaluation 和 distillation 全部未授权。

## 10. 未来时序与移动接口

时序模块未来只能读取 factor tensors/validity、deterministic geometry intervals、timestamp 和
ego-motion validity；只能输出 causal factor residual、future uncertainty、TTC evidence 或
compute-gate diagnostic。它不能输出最终三态，不能把 UNKNOWN 改成 clear，不能跨 session 保留
state。`GeometryR2Reducer` 仍是最终 producer。

移动 graph 未来只导出 raw factor tensors 与 uncertainty；camera transform、body profile、
deterministic reducer、UNKNOWN 和 final tri-state 留在 host。必须先验证 factor parity 和 task
Pareto，再谈 latency。当前不授权 ONNX、QNN、HTP、device、默认 App 或产品变更。

## 11. 停止规则与 authority

```text
F0 fail -> stop reducer implementation version
F0 pass -> only freeze F1 TRAIN-only protocol
F1 fail -> stop factor learnability hypothesis before fresh Development allocation
F1 pass -> only freeze F2 roster and pre-outcome evaluation protocol
F2 fail -> close R2 candidate version; no extra seed/epoch/head/threshold/data reuse
F2 pass -> only propose separately frozen Calibration/Confirmation; no automatic allocation
```

当前所有 execution authority 均为 `false`：synthetic execution、training、materialization、
Development access、teacher、temporal、export、ONNX/QNN/HTP/device、Calibration、Confirmation、
deployment、默认 App、product 和 safety。

## 12. 唯一 successor

```text
BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PROTOCOL_AND_FIXTURES
```

该 successor 当前也没有执行 authority。它只允许另写 F0 协议、fixture schema 和 deterministic
reducer tests；不得训练、读取真实 outcome、创建 checkpoint 或提前分配 F2 Development。
