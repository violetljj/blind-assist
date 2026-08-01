# RISKSEG-R0 任务、数据与顺序执行合同

状态：`ACTIVE / TASK_AND_DATA_CONTRACT_FROZEN /
FULL_SEQUENTIAL_EXECUTION_AUTHORIZED /
EVENT_EVAL_COHORT_NOT_YET_MATERIALIZED /
NO_TRAINING_BEFORE_EVENT_DATA_GATE / DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

机器可读数据角色：
[RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json](RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json)

## 1. 当前决定与权限

`RISKSEG-R0` 是当前唯一算法主线。用户已经授权按下列顺序完整推进，不需要在每个阶段
重新申请执行权限：

```text
冻结任务与数据合同
  -> 物化 >=30 个新 parent events 的 session-disjoint event-eval
  -> PIDNet-S 512x288 / INT8 技术预检
  -> 单架构三 seed 训练
  -> YOLO-only / learned segmentation-only / truth-mask oracle 事件评价
  -> SM-S9280 总链路与持续运行门
  -> 通过全部门才执行默认 App 替换与发布核验
```

`NO_TRAINING_BEFORE_EVENT_DATA_GATE` 是顺序停止门，不是权限缺失。若 event-eval 无法达到
本合同的身份、数量和场景覆盖要求，则终点为 `HOLD_EVENT_EVAL_DATA`；若技术预检失败，
则关闭 PIDNet-S，不消耗正式训练预算。任何下游阶段都不得用“已获完整授权”绕过前置门。

这条主线取代继续修补当前 YOLO 决策规则，也不继承旧 DDRNet、SegFormer-B0、静态
conditional gate、FP-weighted sampler 或 learned component validator 的候选身份。
旧结果和负终态保持不可变。

## 2. 模型任务

问题固定为：

> 对每个像素预测与当前行进可通行性直接相关的四类语义，并回答区域能否通行、阻塞是否
> 侵入行进区域以及是否存在边界/落差；不以识别物体实例类别为优化目标。

新输出 ID 和名称固定如下：

| ID | 类名 | 语义 |
|---:|---|---|
| 0 | `walkable` | 当前视角下具有可通行地面证据的区域 |
| 1 | `blocking_obstacle` | 人员、车辆、杂物等阻塞区域，不细分物体类别 |
| 2 | `boundary_level_change` | 台阶、路沿、落差及其边界证据 |
| 3 | `unknown_nonwalkable` | 不确定、不可评价或不可通行区域；不得自动解释为安全 |

四类互斥，输出固定为每像素四类 logits/quantized scores 和 argmax class ID。训练、转换、
TFLite/QNN、host/device evaluator 与 truth-mask oracle 必须共享同一 class order；出现
通道交换、未知 ID、尺寸不符或非有限值时 fail closed。

旧 canonical mask 的 ID 顺序是：

```text
0 walkable
1 boundary_step_curb
2 obstacle
3 unknown_nonwalkable
```

因此旧 520 帧进入 RISKSEG-R0 前必须物化新的、带逐帧 source hash 与 output hash 的
重编码视图：

```text
old 0 -> new 0
old 1 -> new 2
old 2 -> new 1
old 3 -> new 3
```

禁止把旧 mask 原样解释为新顺序；`boundary_step_curb` 与 `obstacle` 只分别作为
`boundary_level_change` 与 `blocking_obstacle` 的历史别名。

## 3. 冻结的 incumbent 与决策链

以下组件在 RISKSEG-R0 训练、回归与三臂事件评价期间固定，不再加 gate、阈值、类别规则、
FP sampler、组件分类器、latch 或事件特例：

| 组件 | 冻结身份 |
|---|---|
| YOLO baseline | `yolo11n_fp16_320.tflite`, SHA-256 `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2` |
| mask adapter | `TraversabilitySegmentation.kt`, SHA-256 `729d4c7ed0651dbdbd0f8eede6024fd1cfe9c7afd7dc03be06d7e4cb45977f9a` |
| risk analyzer | `RiskAnalyzer.kt`, SHA-256 `9fa3115620626835b1b108a3c3ac2de4aaab0b4a2f5cc1de9e1332347fbe3329` |
| temporal tracker | `TemporalRiskTracker.kt`, SHA-256 `62428714ea2d0f43e025a538d638c415c2247381efcfa6c91299741f007badba` |
| event tracker | `RiskEventTracker.kt`, SHA-256 `6bacd02539437a7960d317184770332a0d7c2155dca2375cdc55109c30e82dd0` |
| feedback planner | `FeedbackPlanner.kt`, SHA-256 `80bc2ad84af09415c534c7e0ddfc5b3cc11e5110e483a8696c1df54656493a0e` |
| decision kernel | `AssistDecisionKernel.kt`, SHA-256 `19d7d820b32fddacd89377233e79f86f3bf257bfd85f1adb9a69f2d54a3a3f07`; contract `blindassist_shared_decision_kernel_v1` |

冻结基线所在 Git commit 为 `e263188d9d2fff753ed1a92894961ce57c6ffc67`。后续实现可以新增
隔离的模型、dataset、benchmark 和 adapter 接口，但不得修改上表行为来帮助 candidate
过门。若并行工程使这些文件发生变化，RISKSEG-R0 必须仍从冻结 commit/哈希执行三臂，
或先以独立版本重新冻结共同链；不得静默漂移。

## 4. 数据角色与独立性

### 4.1 已消费 520 帧

520 帧、10 个 source sessions 全部降级为可重复的 train/dev 数据。固定拆分为：

- train：6 sessions / 320 frames；
- dev：4 sessions / 200 frames；
- source session 在 train 与 dev 间零重叠；
- checkpoint、epoch、augmentation、loss 权重、量化 calibration 与任何 operating point
  只能由 train/dev 决定。

具体 session、帧数和上游 manifest hash 见机器可读 ledger。帧、crop、相邻 pair 和
同 session 的多个窗口都不是独立样本。

这是只对 RISKSEG-R0 向前生效的 Development role overlay，不修改旧 R1 amendment 的
历史事实，也不把其中 4 个已消费 session 恢复成 fresh/unseen/formal。重编码 view
materializer 必须报告每个 session 的四类像素覆盖；`blocking_obstacle` 与
`boundary_level_change` 必须各自在至少 2 个 train sessions 和 2 个 dev sessions
出现非零像素，否则为 `DATA_SPLIT_NOT_READY`，不得只靠改名宣称数据充分。

### 4.2 固定 90 帧回归集

`blindassist-sanpo-v2-event-labeled-20260711` 固定为 90 frames / 3 sessions /
3 parent events 的 consumed regression：

- 1 个 `parallel_curb` 负事件；
- 1 个 `center_obstacle` 正事件；
- 1 个 `front_stairs` 正事件。

manifest SHA-256 为
`3d7168ac975aed57ac6b437ecfa0e668c13dc5d509c7b6584353383eed19d217`。
它只能发现兼容性和行为回归，禁止用于训练、checkpoint/seed 选择、阈值、adapter、
taxonomy、事件规则或晋级判断。即使其中两个 source sessions 与 520 的 train 角色存在
session ancestry 重叠，也不得把该集合包装成 session-disjoint event-eval。当前已知
`i2jg...` 中有 22 张 RGB 与 520 slice 逐 hash 相同，因此本角色的准确名称是
`CONTAMINATED_NON_GATING_REGRESSION_SMOKE`：它不支持泛化、unseen 或独立回归主张。

### 4.3 新 event-eval

训练前必须先物化一个与 520 train/dev 以及固定 90-frame regression
**按 source session 零重叠**的事件评价集。
最低充分性固定为：

- 至少 30 个唯一 parent events、至少 8 个 source sessions；
- `blocking_obstacle` 正事件至少 8 个；
- `boundary_level_change` 正事件至少 8 个；
- `parallel_curb` 负事件至少 7 个；
- `normal_walkable` 负事件至少 7 个；
- 每个 bucket 至少来自 2 个 source sessions；
- 任一 session 不得贡献总 event 数的 25% 以上，也不得贡献单一 bucket 的 50% 以上。
- 必须排除 520 的 10 个 train/dev sessions 和 90-frame regression 的 3 个 sessions，
  包括 `-5OCP... / GxMb... / i2jg...`。

一个 event 可以包含多个帧，但只能计作一个独立单位。event manifest 必须在任何
candidate output、YOLO 对照输出或 truth-mask oracle 事件结果打开前冻结：

```text
source_dataset
source_session_id
source_ancestry_id
sequence_id
parent_event_id
scene_bucket
start/end frame and timestamp
expected alert interval
critical interval
passed/clearance interval
positive_or_negative
truth mask provenance
event label provenance
RGB/mask hashes
license/access note
```

event label 可以是明确标注的 Development silver truth，但必须独立于候选输出生成并披露
来源；模型输出、YOLO 命中或当前规则行为不得参与挑 event、定边界或改标签。缺少 source
session identity、mask truth、事件边界、四桶配额或不重叠证明时，终点为
`HOLD_EVENT_EVAL_DATA`，不开始技术预检或训练。

## 5. 唯一候选与技术预检

唯一架构为 `PIDNet-S`。官方实现/预训练来源为
[XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet)；正式使用时必须把仓库 commit、
license、预训练权重来源和 SHA-256 写入 implementation lock。Qualcomm AI Hub 当前页面
列出 8.06M 参数、W8A8 8.02 MB、Snapdragon 8 Gen 2/3 与 S23/S24 等设备，但同页也显示
顶层 “not supported on any All Models chipset” 提示；这些网页信息只构成部署起点，
不替代本机验证：
[Qualcomm AI Hub PIDNet](https://aihub.qualcomm.com/models/pidnet)。

技术预检使用官方 PIDNet-S 主体、四类 head 和确定性未训练/预训练初始化，不读取
event-eval outcome。部署合同固定为：

- RGB input：`512x288`（width x height）；
- output：`288x512x4`，class order 与第 2 节完全一致；
- full integer W8A8；
- 分别产出可加载的 TFLite 与 QNN artifact；
- 预处理、量化参数、tensor layout、输出反量化/argmax 必须写入 receipt。

预检全部通过才允许训练：

1. TFLite 与 QNN 转换/编译都成功，零 unsupported-op/partition failure；
2. synthetic canary 与至少一组非评价 RGB canary 输出尺寸、dtype、量化参数正确，
   全部反量化值有限，argmax 仅为 `0..3`；
3. SM-S9280 上
   `preprocess + inference + dequantize/argmax + frozen mask adapter +
   frozen decision/event chain` 的 steady-state total P95 `<=100 ms`，failure count 为 0；
4. 连续 10 分钟运行中，最后 2 分钟 total P95 不得超过最初稳定 2 分钟的 `1.20x`，
   且不得出现 severe/critical thermal status、解释器关闭、delegate fallback 或持续
   非有限输出。

任一项失败，终点为 `PIDNET_S_TECHNICAL_PREFLIGHT_FAILED`；关闭该候选，不先训练、
不换 DDRNet/SegFormer，也不临时改输入尺寸、精度或规则救援。

## 6. 训练合同

技术预检通过后使用官方 PIDNet 训练结构与 ImageNet 预训练权重，只训练 PIDNet-S：

- 输出 head 固定为四类；input 固定 `512x288`；
- 固定三个 seed：`20260801 / 20260802 / 20260803`；
- seed `20260801` 为预声明 decision seed，另两个只做稳定性复核，不允许按 event-eval
  结果挑最好 seed；
- checkpoint 和停止点只由冻结 dev 指标决定；
- 使用官方主分割/边界训练结构；具体 optimizer、schedule、augmentation、loss 与
  checkpoint rule 必须在首个训练 step 前写入 implementation lock；
- 禁止新增手工 gate、FP sampler、hard-negative outcome mining、component classifier、
  YOLO pseudo-label、event-eval feedback 或候选间架构搜索；
- pixel mIoU、每类 IoU、boundary F1、finite-output、量化漂移和 worst-session 指标全部
  保留，但像素指标不能替代事件晋级门。

训练故障若发生在 event-eval outcome access 前，可以在不改变科学变量的独立 evidence
version 中修复 runner/serialization/dependency 并重跑；一旦 event-eval outcome 被访问，
不得在同一 R0 上据此改 loss、sampling、checkpoint 或阈值。

## 7. 三臂事件评价

三臂必须使用完全相同的 event membership、RGB、因果时钟、reset、frozen adapter、
risk/event/feedback chain 和 scoring truth：

```text
A_CURRENT_YOLO_ONLY
B_LEARNED_SEGMENTATION_ONLY
C_TRUTH_MASK_ORACLE_REFERENCE
```

- A 只输入冻结 YOLO detections；
- B 只输入 PIDNet-S 四类 mask，不混入 YOLO；
- C 只输入同一 event 的四类 truth mask，不混入 YOLO；
- C 是信息上限参考，不是可部署模型；
- 90-frame regression 单独报告，不并入晋级分母。

每个 seed 都必须报告 parent-event 级结果。decision seed 晋级至少同时满足：

1. 正事件召回相对 YOLO 提高至少 `0.15` absolute 且多命中至少 3 个 parent events；
2. critical miss 至少减少 2 个，且 obstacle/boundary 两个正类 bucket 均不得增加；
3. false-alert event count 不高于 YOLO；
4. 含 passed interval 的事件中，cleared event count 和 clearance rate 均不低于 YOLO；
5. 共同命中事件若至少 4 个，candidate 相对 YOLO 的首次有效提醒延迟中位数
   `<=+1 frame`，且延迟 `>2 frames` 的比例 `<=25%`；若只有 1--3 个共同命中事件，
   不得有事件晚 `>2 frames`；零共同命中记为 `NOT_APPLICABLE`，不伪造 timing gain；
6. 第 5 节的 SM-S9280 性能与持续运行门在最终 INT8 checkpoint 上复测通过。

稳定性要求：decision seed 必须通过全部六门；至少 2/3 seeds 通过全部事件质量门，
且三个 seed 均不得增加 false-alert event 或降低 cleared event count。不得选择
event-eval 上最好的 seed 覆盖不稳定性。

结果必须同时报告 C 相对 A/B 的 headroom、每个 bucket、每个 session、每个 parent event、
miss、false alert、clearance、共同命中时序、逐阶段 P50/P95、失败和热状态。统计置信区间
可以作为补充，但不得用帧数冒充 event 独立样本。

## 8. 终态与默认 App

按顺序只允许以下终态：

| 条件 | 终态 | 后续 |
|---|---|---|
| 新 event-eval 不足或身份不完整 | `HOLD_EVENT_EVAL_DATA` | 停止，不预检、不训练 |
| 技术预检任一硬门失败 | `PIDNET_S_TECHNICAL_PREFLIGHT_FAILED` | 关闭 PIDNet-S，不换候选 |
| 训练有效但任一晋级/稳定性/性能门失败 | `RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO` | YOLO 保持默认 |
| 全部事件、稳定性、性能门通过 | `RISKSEG_R0_DEVELOPMENT_PROMOTION_PASS` | 进入同一已授权链的默认 App 替换与发布核验 |
| 默认替换后的回归/build/release gate 失败 | `RISKSEG_R0_INTEGRATION_FAILED_ROLL_BACK_TO_YOLO` | 恢复冻结 YOLO 默认 |
| 默认替换及发布核验通过 | `RISKSEG_R0_DEFAULT_APP_PROMOTED` | 仅为研究原型默认，不构成独立助行或安全证明 |

任何关键漏报、false-alert event、clearance、共同命中时序或 SM-S9280 性能出现本合同
定义的明显 trade-off，都保持 YOLO baseline，不把单一像素 IoU、论文速度、模型大小或
truth-mask oracle 成功解释成默认替换证据。
