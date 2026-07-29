# BlindAssist 神经—几何双环阶段−1准入合同 R0

状态：`DESIGN_FROZEN / ADMISSION_ONLY / PROPOSAL_ONLY / EXECUTION_NOT_AUTHORIZED`

协议 ID：`BLINDASSIST_DUAL_LOOP_PHASE_MINUS1_ADMISSION_R0`

上位规则：

- [BlindAssist 渐进式研究治理](../../RESEARCH_GOVERNANCE.md)
- [渐进式研究协议模板](../../RESEARCH_PROTOCOL_TEMPLATE.md)
- [GPT / Codex 端到端自主工作流治理](../../AI_REVIEW_GOVERNANCE.md)
- [双环研究主线](README.md)

## 1. 决策与研究问题

本合同把神经—几何双环纳入 BlindAssist 当前论文系统路线，但只授予**顺序准入设计**，
不授予实验、代码接线或产品行为变更。

研究问题限定为：

> 在相同 CameraX 输入、YOLO、风险规则和提醒语义下，已有轻量 Sparse LK 候选输出，
> 是否能在结果真实可用的时刻，为接近危险事件提供 YOLO-only 尚未及时提供的有效信息；
> 若存在，该信息能否在指定手机上以有界延迟、时效、队列和热负担实现？

双环架构、NPU/GPU/CPU 分工、时间衰减和 latest-only 本身均不作为算法新颖性结论。
阶段−1只筛选以下一种或多种可证伪的事件级增量：

1. `EARLY_RESPONSE`：首次对外有效提醒更早；
2. `RISK_DISCRIMINATION`：在不增加关键漏报的前提下减少不应提醒事件；
3. `RISK_CONTINUITY`：在有效危险区间减少因语义间歇缺失造成的风险空洞。

若三者均不成立，双环论文主张停止，不以工程完成度回救。

## 2. 候选系统与明确不做

阶段−1候选固定为：

```text
同一 CameraX 视频
  ├─ 语义证据：现有 YOLO11n → 现有检测/风险接口
  └─ 几何证据：已有 Sparse LK 候选输出 → 待验证的区域级几何/接近候选证据
             ↓
     真实时间、区域、质量、TTL、失效原因
             ↓
       离线参考融合与同一提醒语义
```

边界如下：

- 默认几何候选只有一个：已有 Sparse LK；不新发明光流、跟踪器或深度方法。
- RCLE 已暂停，不是阶段−1依赖，不重跑、不修复、不消费任何 RCLE formal identity。
- 区域匹配首版只允许 `LEFT / CENTER / RIGHT`；如 F-1A 标签确实需要，可在结果访问前
  增加固定的上/中/下分区，不进入复杂跟踪或三维关联。
- F-1B可以建立一个隔离的离线参考融合器；它不得接正式 App、语音、震动或生产路由。
- F-1C只允许 benchmark/shadow A/B；不得改变正式提醒阈值、默认模型或用户反馈。

阶段−1禁止：

```text
正式双环重构
自适应或 learned scheduling
新风险场或第二状态机
深度、分割、ARCore、VIO、occupancy map
新模型训练或 YOLO 重训练
新数据集市场漫游
RCLE retry / replacement / threshold repair
USTRF route-conditioned program 重启
Android 产品、真人或安全主张
```

## 3. 已有事实、候选与未知

| 项目 | 当前分类 | 合同中的使用方式 |
| --- | --- | --- |
| CameraX `640×480`、Preview 请求 `24 FPS`、ImageAnalysis `KEEP_ONLY_LATEST` | `CONFIGURED / REQUESTED_IN_CODE` | 作为当前输入政策；真实 analysis cadence、jitter、drop 必须实测 |
| SM-S9280 / SM8650 QNN HTP 路由 | `OBSERVED / PROMOTED_WITH_CPU_FALLBACK` | 只支持该设备当前生产路由，不外推 SM8550 |
| QNN 100图完整检测 `12/15 ms`、十分钟 `16/21 ms` P50/P95 | `OBSERVED` | 提供绑定设备的延迟观测；不能代替真实 `availableAt` 轨迹 |
| QNN 能耗优势 | `NOT_EVALUATED` | 不得宣称；无外部功率测量时只报告调用量、温度和设备指标 |
| latest-only owned-luma sidecar | `IMPLEMENTED / TESTED_COMPONENT` | 可复用队列、所有权和过期拒绝，不视为双环已接通 |
| 旧 Sparse LK 回放并发路径 | `TIMING_ONLY / CPU_ERA` | 说明工程形态可能有界，不支持当前 QNN 或事件效果 |
| 旧 live CameraX + YOLO + sidecar | `MIXED_NEGATIVE / UNMATCHED_CONTROL` | 保留旧 `P95 > 70 ms` 事实；不能归因给 sidecar |
| 双环事件级收益 | `NOT_EVALUATED` | F-1B 生死门 |
| SM8550 / Snapdragon 8 Gen 2 | `NOT_ADMITTED` | 只可另立设备迁移验证，不是 R0 准入前提 |

本表不把旧性能夹具改写成当前系统结论。关键本地依据见：

- [NPU 正式设备能力路由](../../NPU_DEFAULT_CANDIDATE.md)
- [`CameraXFrameSource.kt`](../../../core/device/src/main/java/com/linnan/blindassist/camera/CameraXFrameSource.kt)
- [`LatestOnlySidecar.kt`](../../../core/vision/src/main/java/com/linnan/blindassist/vision/LatestOnlySidecar.kt)
- [`RgbaLumaSidecar.kt`](../../../core/vision/src/main/java/com/linnan/blindassist/vision/RgbaLumaSidecar.kt)
- [Corridor-Causal 进度快照](../../CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md)
- [Project Guideline 组件适配审计](../../PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md)

## 4. 状态向量与顺序权限

不得用一个总 `PASS` 覆盖不同问题。每轮至少独立记录：

```text
DATA_STATUS
TIMING_STATUS
SCIENCE_STATUS
RUNTIME_STATUS
DATA_PROTOCOL_STATUS
TIMING_PROTOCOL_STATUS
SCIENCE_PROTOCOL_STATUS
RUNTIME_PROTOCOL_STATUS
EXECUTION_AUTHORITY
CLAIM_CEILING
```

取值为：

```text
DATA_STATUS:
  NOT_RUN | READY | HOLD_DATA | INVALID

TIMING_STATUS:
  NOT_RUN | READY | HOLD_TIMING | INVALID

SCIENCE_STATUS:
  NOT_RUN | EARLY_RESPONSE | RISK_DISCRIMINATION |
  RISK_CONTINUITY | MULTIPLE_INCREMENT |
  NO_INCREMENT | NOT_EVALUABLE

RUNTIME_STATUS:
  NOT_RUN | FEASIBLE | ONE_BOUNDED_REPAIR_CANDIDATE |
  STOP_RUNTIME | NOT_EVALUABLE

各阶段 PROTOCOL_STATUS:
  DESIGN_FROZEN | VALID | INVALID
```

四个阶段各自记录协议状态，不得用后阶段的有效性覆盖前阶段。状态转换固定为：

- `DESIGN_FROZEN`：在读取本阶段结果前冻结输入身份、角色、实现与配置哈希、指标、停止
  条件和输出路径；
- `VALID`：执行后由与该次执行分离的复核人或确定性检查器，对照冻结设计完成
  hash-bound 复核；
- `INVALID`：存在身份漂移、角色污染、冻结后改规则、输出不完整或无法完成复核。

F-1A 与 F-1B0 的轻量结果可以在同一结果文件内保留独立 review section；F-1B 与 F-1C
必须保留阶段 spec SHA、结果 SHA 和执行后独立复核凭据。`VALID` 只产生申请下一阶段的
资格，不自动授权或开始执行。

证据状态彼此独立，执行权限严格单向：

```text
PROPOSAL_ONLY
→ F-1A_AUTHORIZED
→ F-1B0_AUTHORIZED（仅在缺少可用时间凭据时）
→ F-1B_AUTHORIZED
→ F-1C_AUTHORIZED
→ DUAL_LOOP_ADMITTED
```

前级结果只能产生“后级可被另行授权”的资格，不自动开始后级。后级成功不得补偿前级
`HOLD / NO_INCREMENT / NOT_EVALUABLE / INVALID`。

## 5. F-1A：数据能否评价

### 5.1 目的

只检查已有数据与低成本补充能力能否回答最小事件问题。不得为了寻找“完美数据集”
重新进入长期公开数据搜索，也不要求三维轨迹、精确碰撞时间或完整路线真值。

### 5.2 最小身份与标签

每个连续片段至少绑定：

```text
source_id
content_hash_or_stable_identity
session_id
sequence_id
clock_or_frame_order_basis
event_id
event_type
positive_or_negative
onset_interval
alertable_start_interval
end_or_clear_interval
LEFT_CENTER_RIGHT region
truth_provenance
outcome_access_state
```

正例至少覆盖相机/用户接近静态障碍或动态目标接近中的一种；未覆盖的机制不能进入主张。
负例至少从下列类别中形成可区分窗口：

- 转头或近原地旋转；
- 正常步行抖动；
- 横向经过或远离；
- 静止场景；
- 低纹理、模糊或遮挡。

标签生成、隔离复核和必要仲裁遵循项目现有
[AI review governance](../../AI_REVIEW_GOVERNANCE.md)，只使用自动化标签、隔离模型
复核与既有仲裁规则；无可追溯真值时直接 `HOLD_DATA`。

“YOLO没有检测到”不得被用来定义未知障碍。只有独立事件标签已经把类别外对象判定为
真实接近危险时，才能单列类别外分析。

### 5.3 最小可评价规模

阶段−1是路线筛选，不是论文最终确认。最小 `READY` 条件为：

- 至少 `3` 个相互独立的 capture session；
- 至少 `6` 个正事件，分布在至少 `2` 个 session；
- 至少 `12` 个明确 `should_alert=false` 的负窗口；
- 负窗口覆盖至少 `4` 类上述负场景，每类至少 `2` 个；
- 所有纳入项具有连续 RGB、可靠顺序、事件区间、区域和真值来源；
- 至少 `1` 个 development session 仅用于选择几何质量门、TTL、区域规则或融合规则；
- 至少 `2` 个含正事件的 decision session 完全不参与上述规则选择，正向方向必须分别
  在这两个 session 中成立。

这些是路线筛选下限，只允许形成“值得/不值得继续”的判断。后续论文确认必须另有
session-disjoint 的冻结证据，不能把帧、pair 或 cell 数量冒充独立样本量。

### 5.4 终点

- `READY`：全部最小条件满足，生成数据清单与角色表；
- `HOLD_DATA`：所需数据或标签不足，只输出缺口和一个有界补充方案；
- `INVALID`：身份、时间顺序、角色污染或真值来源无法追溯。

F-1A不得运行 Sparse LK、RCLE 或双环结果，不得因为候选输出“看起来好”选择片段。

## 6. F-1B0：语义与几何结果真实可用时间凭据

### 6.1 触发条件

F-1A为 `READY` 后，先分别检查现有生产 QNN 证据和隔离 Sparse LK 证据是否已经包含
本阶段需要的真实时间字段。字段完整则直接复用；任一来源缺失时，才允许另行授权一次
baseline-only 补测。

F-1B0不是效果 A/B，不改变风险或提醒。若现有几何时间凭据缺失，可以只运行隔离的
Sparse LK timing-only 测量；不得输出事件级增量、查看候选效果或选择质量门、TTL、区域
和融合规则。

### 6.2 时间语义

每条语义结果至少记录：

```text
capturedAt
receivedAt
queuedAt
startedAt
completedAt
publishedAt
availableAt
consumedAt
clockDomain
dropReason
detectorBackend
backendRouteReason
```

每条几何结果至少记录：

```text
previousObservationAt
currentObservationAt
geometryQueuedAt
geometryStartedAt
geometryCompletedAt
geometryPublishedAt
geometryAvailableAt
geometryConsumedAt
clockDomain
dropReason
abstainReason
```

`availableAt` 定义为下游决策层第一次能合法读取完整结果的单调时钟时刻。它不得早于
结果发布，不得用 `capturedAt + 平均/P50延迟` 回填，也不得把最终结果用于它完成前的
历史决策。

现有 `LatestOnlySidecar` 只直接提供 `capturedAtNanos / completedAtNanos`。F-1B0 必须
补齐发布与消费接缝；不得把 `completedAtNanos` 改名后直接冒充真实 `availableAt`。
离线 Sparse LK 也不得被视为零延迟：历史时刻 `t` 只能消费
`geometryAvailableAt <= t` 的几何结果。

若 camera hardware clock 无法证明映射到 Android monotonic clock，则基于 capture time
的 age 为 `NOT_EVALUABLE`；可以退回能被证明的 `receivedAt` 语义，但必须降低主张。

### 6.3 24 Hz 的操作定义

`B24` 表示：

> 在相同真实帧释放轨迹上，每秒最多处理 24 个最新合格帧；没有新帧不复制，过载时按
> 冻结的 latest-only 规则丢弃，禁止离线追赶。

必须分别报告：

```text
CameraX received
CameraX analyzable
YOLO submitted
YOLO completed
YOLO consumed
Coverage_YOLO = completed / analyzable
Sparse LK submitted
Sparse LK completed
Sparse LK consumed
Coverage_Geometry = completed / eligible
cadence / jitter / age / drop（按语义与几何分别报告）
```

请求 24 FPS 不等于实际产生 24 Hz 语义或几何结果。

### 6.4 终点

- `READY`：语义与几何两路的设备/实现身份、时钟、队列和完整可用时间轨迹均可重算；
- `HOLD_TIMING`：缺少真实可用时刻或时钟映射，无法进行因果离线模拟；
- `INVALID`：使用未来结果、伪造零延迟或把 CPU fallback 记为 QNN 成功。

## 7. F-1B：几何是否产生事件级增量

### 7.1 固定比较

主比较只允许：

```text
A: YOLO-only + 同一既有提醒语义
B: YOLO + Sparse LK 离线参考融合 + 同一既有提醒语义
```

可附加 `geometry-only` 作为诊断，但不得替代 A/B 主比较。两个分支必须共享：

- 同一 source frame 与真实 release/available 轨迹；
- 同一 YOLO 模型、阈值、后处理和结果；
- 同一事件分母、区域定义、状态确认、冷却和提醒语义；
- 同一过期、丢帧和 abstain 记账方式。

YOLO每帧最多计算一次，频率分支复用冻结输出。任意时刻 `t` 只能读取对应来源
`availableAt <= t` 或 `geometryAvailableAt <= t` 的证据。

### 7.2 频率

- `B24`：主要严格基线，先判断当前接近逐帧语义预算下是否仍有风险增量；
- `B12 / B8`：次要预算交换分析，不能先行成为主结果；
- `B6`：压力或降级边界，不作为主要成功依据。

12/8 Hz 即使结果更好，也只能形成计算预算候选。只有 F-1C 证明降低语义调用确实改善
温升、延迟漂移、调用量或有可信功率证据，才能写成系统收益。

### 7.3 离线参考融合

F-1B开始前必须用 development session 冻结一份薄实现说明：

- Sparse LK 实现身份、输入尺寸和参数；
- 几何质量门与 abstain；
- 区域映射；
- 几何 TTL、语义 TTL；
- 最小连续支持规则；
- 对外提醒如何复用现有状态/冷却语义；
- 唯一主要科学终点和最小有意义差异。

看过 decision session 输出后，不得原地修改这些规则。离线参考融合只为判断是否值得
进入手机 A/B，不获得生产实现权限。

### 7.4 固定分母与反伪增量护栏

主要独立单位是 session 和 hazard event；使用 session-grouped 比较或在最小三 session
上做 leave-one-session-out。禁止随机拆帧。

所有过期、排队丢弃、后端失败、低质量和 abstain 必须保留在固定事件分母中。除主结果
外可以报告条件结果，但不得只在“成功返回且可评价”的子集上宣布改善。

所有正向终点必须同时满足：

1. critical miss count 不增加；
2. event recall 不降低；
3. evaluable coverage 不因更多 abstain 获得虚假改善；
4. 改善出现在至少 `2` 个独立 session，且不由单一事件贡献全部方向；
5. false alerts/min 不越过结果访问前冻结的非劣界；
6. 比较的是实际可交付提醒，而不是内部 `WATCH`、几何曲线或风险分数。

`EARLY_RESPONSE` 还要求在符合提醒条件的配对正事件中：

- 多数事件的首次有效提醒提前量 `> 0`；
- 配对中位提前量至少达到该数据实际观测的一个 camera frame interval；
- recall、critical miss、false alert 和 coverage 护栏全部通过。

若 F-1B 只能使用隔离/离线测得的几何时间而尚未在 F-1C 复现真实组合路径，则
`EARLY_RESPONSE` 只能记为 `EARLY_INFORMATION_OPPORTUNITY / DEVELOPMENT_SCREEN`；
不得在 F-1C 前写成已实现的端到端首次提醒提前。

`RISK_DISCRIMINATION` 与 `RISK_CONTINUITY` 的最小有意义差异，必须在 F-1A 分母已知、
F-1B decision 输出尚未访问时冻结。若数据规模只支持方向筛选，结果必须标为
`DEVELOPMENT_SCREEN`，不得写成论文确认性效果。

### 7.5 数据污染与确认边界

- `CONTENT_INSPECTED` 可以用于预先冻结的评价，但须披露筛选依据；
- 看过 A/B、几何或语义输出后标为 `OUTPUT_INSPECTED`；
- 用于选择参数、TTL、区域或融合规则后标为 `TUNED_ON`；
- `OUTPUT_INSPECTED / TUNED_ON` 只能支持开发和路线准入，不能再用于同一命题的确认。

若双环获准进入论文主线，必须另行冻结 session-disjoint 的确认集和统计合同。

### 7.6 终点

- `EARLY_RESPONSE`
- `RISK_DISCRIMINATION`
- `RISK_CONTINUITY`
- `MULTIPLE_INCREMENT`
- `NO_INCREMENT`
- `NOT_EVALUABLE`

`NO_INCREMENT` 无论运行时多快都停止双环论文主张。`NOT_EVALUABLE` 只允许回到明确的
数据/协议缺口，不允许把不利结果包装成“还需调参”。

## 8. F-1C：指定手机实时承载

### 8.1 进入条件

只有 F-1B 至少一个正向科学终点成立，且 `DATA_PROTOCOL_STATUS=VALID`、
`TIMING_PROTOCOL_STATUS=VALID`、`SCIENCE_PROTOCOL_STATUS=VALID`，才允许准备 F-1C；
开始执行仍需用户另行授权。F-1C 完成后还必须独立产生
`RUNTIME_PROTOCOL_STATUS=VALID`，才能签署运行时终点。

首个正式设备固定为已有生产路由证据的 Samsung SM-S9280 / SM8650。执行时必须绑定：

```text
device_model / soc / soc_id
android_version
camera_config
qnn_version
model_hash / quantization
production_route / fallback_policy
clock_mapping
queue_policy
power_connection
initial_thermal_state
ambient_condition
test_duration
```

SM8550 / Snapdragon 8 Gen 2 必须另立迁移验证，不能继承 SM8650 结论。

### 8.2 匹配 A/B

```text
A: CameraX + production QNN YOLO
B: CameraX + production QNN YOLO
   + owned-luma latest-only Sparse LK sidecar
```

A/B必须匹配 CameraX、模型、量化、QNN路由、风险分析、输入场景、预热、初始温度、
电源方式与运行时长。先做短 canary；canary 未触发停止门后，再做每个分支至少
`20 min` 的交替顺序持续运行。

### 8.3 必报指标

- received / analyzable / YOLO submitted / completed / consumed；
- `Coverage_YOLO`、CameraX cadence、busy/drop；
- sidecar submitted / replaced / completed / fresh / expired / failed；
- preprocess / inference / postprocess / decision / end-to-end 的 P50/P95/P99；
- YOLO 与 geometry 的 input age、result age、available age；
- 实际 backend、route reason、CPU fallback count；
- 内存、温度、thermal status 与前后段延迟漂移；
- NPU调用次数与 worker duty；无可信功率测量时不报告绝对能耗收益。

### 8.4 阻断门

`FEASIBLE` 至少要求：

1. 无 crash、worker leak 或无界队列；
2. 没有未解释的 CPU fallback，两个分支均保留真实 route reason；
3. B 的端到端 P95 `<= 70 ms`，且相对 matched A 的 P95 增量 `<= 15 ms`；
4. F-1B所需几何 TTL 下，新鲜结果覆盖率通过预冻结门；
5. CameraX/YOLO coverage 不因只保留成功子集而虚高；
6. 持续运行无 thermal throttle，内存和后段 P95 漂移通过执行前冻结的门；
7. 当前 F-1B 科学增量在真实时序回放中没有因排队、过期或丢帧失效。

P99、温度和漂移即使未设阻断阈值也必须披露，不能在看到结果后选择性升级或降级为门。

### 8.5 终点

- `FEASIBLE`：只证明绑定设备与合同条件下可承载；
- `ONE_BOUNDED_REPAIR_CANDIDATE`：仅一个明确工程缺陷可能修复；
- `STOP_RUNTIME`：实时路线停止，离线科学结果可作为部署限制保留；
- `NOT_EVALUABLE`：运行、设备或测量合同无法合法作答。

`FEASIBLE` 不授权正式 App 接线，不产生用户、安全、产品或跨设备结论。

## 9. 一次有限修复

`repair_budget = 1` 是整个 R0 的全局上限，不是每个门各一次。修复资格也不是自动权限；
F-1C结果只能提出候选，仍须用户明确授权。

允许的修复类型仅限：

- 几何输入分辨率降低一次；
- 去除明确重复的 RGBA/YUV/luma 复制或重复预处理；
- 修复线程优先级、单槽队列、资源释放；
- 修复明确的 CPU fallback；
- 修复时间戳、`availableAt` 或过期判定实现错误。

禁止借修复改变：

- 数据 cohort、事件分母、baseline、主要指标或科学门；
- Sparse LK 算法族、阈值搜索空间或多轮参数扫描；
- YOLO训练、深度、分割、ARCore、新跟踪器或架构重写；
- 已访问确认数据后的科学失败。

若修复发生，必须创建新实现版本，保留 R0 原结果：

```text
一次修复
→ 完整重新 A/B
→ FEASIBLE 或 STOP_RUNTIME
```

不得再次返回 `ONE_BOUNDED_REPAIR_CANDIDATE`。若对应数据已作为确认集打开，修复后不得
在同一确认身份上重跑。

## 10. 最终路线派生

| Data | Timing | Science | Runtime | 路线 |
| --- | --- | --- | --- | --- |
| `READY` | `READY` | `EARLY_RESPONSE` | `FEASIBLE` | 双环准入；论文主线为首次有效提醒提前 |
| `READY` | `READY` | `RISK_DISCRIMINATION` 或 `RISK_CONTINUITY` | `FEASIBLE` | 双环准入；论文主线为具体风险质量增量 |
| `READY` | `READY` | 任一正向增量 | `ONE_BOUNDED_REPAIR_CANDIDATE` | 只可申请一次有限修复 |
| `HOLD_DATA` | 任意 | `NOT_RUN` | `NOT_RUN` | 停止开发，先决定是否有界补数据 |
| `READY` | `HOLD_TIMING` | `NOT_RUN` | `NOT_RUN` | 停止机会模拟，补齐真实时间语义 |
| `READY` | `READY` | `NO_INCREMENT` | 任意 | 停止双环论文主张 |
| `READY` | `READY` | 正向增量 | `STOP_RUNTIME` | 保留离线开发结果，停止手机实时主线 |
| 任意 | 任意 | 任意 | 任意且任一阶段 `PROTOCOL_STATUS=INVALID` | 当前 evidence version 不可签署 |

正向路线也只获得后续“正式融合合同准备资格”，不自动修改代码。

## 11. 阶段−1输出

每道门只生成一份短结果，避免重复建设：

1. `DUAL_LOOP_DATA_READINESS_R0`
2. `DUAL_LOOP_TIMING_RECEIPT_R0`（仅需要补测时）
3. `DUAL_LOOP_INCREMENT_SCREEN_R0`
4. `DUAL_LOOP_RUNTIME_PREFLIGHT_R0`

结果必须写出本轮新事实、被削弱假设、未评价项、停止范围和下一项可申请权限。
不提前创建 runner、validator、机器状态机或大日志框架；只有进入确认阶段后，才按上位
治理增加严格合同、独立 validator 和 receipt。

## 12. 当前授权收口

本次用户授权仅包括：

- 编写本合同；
- 将本合同接入项目 current 主线；
- 暂停 RCLE 的默认执行队列并保留其全部历史事实与权限状态。

当前明确不包括：

```text
F-1A / F-1B0 / F-1B / F-1C 执行
数据采集、补标或候选输出检查
Android、benchmark 或生产代码修改
RCLE恢复、重跑、修复或formal消费
双环效果、实时性、能效或安全结论
```

因此本合同落地后的唯一当前状态仍是：

```text
DUAL_LOOP_STATUS = PROPOSAL_ONLY
EXECUTION_AUTHORITY = NONE
NEXT_AUTHORIZABLE_ACTION = F-1A_DATA_AUDIT_ONLY
```
