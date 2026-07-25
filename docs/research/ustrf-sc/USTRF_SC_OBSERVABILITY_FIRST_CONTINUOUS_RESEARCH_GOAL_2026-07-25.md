# USTRF-SC 可观测性优先架构与持续研究总目标（2026-07-25）

状态：`TERMINAL_RESEARCH_OUTCOME / EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY / VALID`

持续授权：`OBSERVABILITY_AUDIT_AND_FALSIFIABLE_OFFLINE_RESEARCH_ONLY`

当前研究边界：JRDB sensor-support/bias 已完成 3 个 metadata-frozen 新 sequence × 120 帧的跨序列复现；合计 8,118/9,771 个 annotation-derived 3D object-frame 与 7,822/9,679 个 motion pair 得到冻结 LiDAR 支持，pooled centroid residual median/P95 为 `0.168/0.446m`。但 worst object/pair support 为 `73.67%/70.15%`、worst P95 为 `0.669m`；3D-only residual 方向复现，远距仅 1/3 sequence 可评，终态 `CROSS_SEQUENCE_PROFILE_AVAILABLE_WITH_PARTIAL_REPLICATION / VALID`。权限上限仍为 diagnostic，正式 G1–G7 仍缺 independent person trajectory truth、intended-route 与独立 event lifecycle truth

算法成熟度：`CORE_ALGORITHM_EFFECT_NOT_PROVEN`

用户与生产权限：`CLOSED`

## 一、这份总目标如何使用

本文件是 USTRF-SC 后续研究的长期总目标和当前研究指导，不是一次性实验合同，也不是把所有后继阶段一次性打开。

“持续允许”表示：在不改变本文权限边界的前提下，Codex 可以持续选择当前证据所允许的**下一个最小、独立、可失败垂直切片**，完成仓库内审计、实现、离线运行、验证和日期化记录，而不需要反复要求用户重新表达“继续研究 USTRF-SC”这一总体意图。

“持续允许”不表示：

- 自动执行整条 roadmap；
- 在前一门失败后绕过它进入后一门；
- 把 discovery 成功直接变成 producer、opener、Android 或产品权限；
- 以总目标代替每个阶段自己的冻结配置、结果、机器收据和终态；
- 授权人体、盲人独立行走、真实提醒投递、公开发布、生产替换、commit 或 push；
- 授权破坏性文件操作、凭据操作、系统级安装或超出 `E:\linnan` 的扩张动作。

每个后继阶段仍必须：

1. 重新绑定父证据；
2. 写明唯一研究问题；
3. 先审计输入可用性上界；
4. 冻结数据角色、变量、指标、停止门和唯一合法终态；
5. 生成可复算结果；
6. 根据终态决定下一阶段是否存在。

本文件负责长期方向和授权边界；日期化子 goal、结果和收据负责单次执行事实。若两者冲突，以更严格的 fail-closed 边界为准。

## 二、总判断

USTRF-SC 不需要推倒重来，但需要一次架构降维和研发顺序纠偏。

应当长期保留：

- 路线条件化，而不是全图目标出现即危险；
- 对象无关风险，而不是封闭类别表决定安全；
- 稠密或空间化风险表示，而不是仅用单框标签；
- 事件级 onset、alertable、clearance 和 repeat，而不是逐帧 AP；
- evidence freshness、reset、unknown、abstention 和 fail-closed；
- 人体包络、路线走廊、安全监督器和反馈层分离；
- producer 与 truth audit 隔离；
- true route、uniform route、shuffled route 和 bbox baseline 的负控；
- discovery、validation、sealed holdout、device、human、production 的权限分层。

必须优先重建：

- canonical source frame 的 width、height、rotation 和坐标权威；
- capture timestamp、frame membership、gap 和 reset 的统一时序权威；
- camera/body 外参、pose/ego-motion 及其质量与失效原因；
- depth 的 relative/metric、registration、scale 和 freshness 边界；
- explicit route provider、projection 和 route validity；
- target/object truncation、occlusion、visibility 和 unknown；
- route-conditioned event ontology、matched negative 和独立 session；
- 所有 signal arm 共享的 immutable observation packet。

必须关闭：

- 已证伪 current-input policy family 内的 TTL、qualification、renewal、association 和 opening timing 搜索；
- 已拒绝 radial/lateral/scale `2-of-3` 定义的窗口、组合和零阈值回救；
- 在同一 11 个 discovery event 上把表面成功称为验证；
- 用硬编码 `640×480`、假定 `rotation=0`、显示层坐标或 bbox 触边 heuristic 冒充 canonical authority；
- 在 signal availability/separability 前写 Android、Kotlin runtime、opener 或新通用框架；
- 继续把 person-track token 分支当成 dense object-agnostic USTRF 核心；
- 用 detector、MOT、depth、VLM 或机器人 benchmark 指标冒充助盲事件或用户安全证据。

一句话总纲：

```text
不是继续完善一个尚未被信息支持的算法实现，
而是先建设能证明或证伪它的感知与评价系统。
```

## 三、当前事实基线

本总目标从以下已经冻结的结果出发，不改写其终态：

| 父证据 | 当前事实 | 本文件如何使用 |
| --- | --- | --- |
| [current-input policy feasibility bound R0](USTRF_CURRENT_INPUT_POLICY_FEASIBILITY_BOUND_R0_RESULT_2026-07-24.md) | `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID`；乐观最大 coverage `8/11=24/33`，经验风险 `<=2` 时仅 `2/11=6/33` | 关闭旧 timing/TTL/renewal family |
| [causal route-relative intrusion signal R0](USTRF_CAUSAL_ROUTE_INTRUSION_SIGNAL_R0_RESULT_2026-07-24.md) | `SIGNAL_REJECT / VALID`；`7/11=21/33`，负机会 `43/4.956min` | 关闭冻结的 `2-of-3` 定义，不做参数回救 |
| [scale-growth separability R0](USTRF_ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0_RESULT_2026-07-25.md) | `FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED / VALID`；未计算 slope | 证明 observation authority 必须先于 signal 实现 |
| [Evidence Maturity V2](USTRF_ROUTE_TARGET_EVIDENCE_MATURITY_STANDARD_V2.md) | 当前最高仍以各指标独立 eligibility 为准；L3/L4 未开放 | 保持样本、来源、LOSO、shadow 与生产权限边界 |
| [正式研究设计](../GROUP_MEETING_PROGRESS.md) | 正式 U0 为 6 session、120 episode、60 matched pair、6-fold LOSO 和六臂对照 | 作为核心算法验证目标，不由 proxy 分支替代 |

当前 discovery 分母仍只有：

- `11` 个 oracle-supported unique event；
- `33` 个 mechanically mapped supported candidate cell；
- `4.95626851575min` 负暴露；
- `41` 条 candidate-independent sequence；
- `62,229` 个 canonical replay frame。

`33/33` 是 11 个事件到 C1–C3 的机械映射，不是 33 个独立运行时事件。任何统计、摘要和权限判断必须以 event/session/source cluster 为主要独立单位。

当前 scale R0 的精确 gap 为：

- `62,229 / 62,229` 帧没有逐帧绑定 canonical `source_size`；
- `62,229 / 62,229` 帧没有 source-to-canonical rotation/orientation receipt；
- `263,680 / 263,680` 个 observed-track record 没有 authoritative severe-truncation 状态；
- signal score、truth decode、event decode、oracle decode、negative decode 和 candidate decode 均为 `0`；
- 没有 inventory、frontier 或 frozen candidate。

因此当前没有“尺度无效”结论，只有“当前输入合同不足以合法执行尺度实验”结论。

## 四、架构分层决定

### 4.1 问题定义：保留

USTRF-SC 的核心问题继续定义为：

> 在跨相机、跨场景、连续真实路线事件中，路线条件化、对象无关的时空风险表示，能否比类别依赖 bbox baseline 更早、更稳定地识别真正进入用户路线的危险事件，同时不增加 critical miss、false alert、repeat、clearance delay、计算负载和错误方向输出，并在证据不足时 fail closed？

这个问题具有明确对照、可失败条件和产品相关事件指标，优于“换一个 detector 能否提高 AP”。

### 4.2 风险表示：保留，但尚未被证明

稠密 field、occupancy、geometry、motion、unknown、route interaction 和 body envelope 的组合在架构上合理，但正式六臂 U0 尚未运行。

当前只允许说：

- dense/object-agnostic representation 有研究动机；
- route interaction 有机制性 proxy 信号；
- safety kernel 和 evidence seam 具有工程闭环；
- 核心算法是否优于 bbox/uniform/shuffled 仍未证明。

不允许说：

- USTRF-SC 已实现；
- dense field 已优于 detector；
- 当前实验 App 是完整 USTRF；
- synthetic、proxy 或 public-source 指标证明助盲效果；
- risk field 已具有用户侧安全能力。

### 4.3 安全与证据架构：保留并压缩复用

以下机制长期保留：

- hash-bound immutable input；
- candidate-blind producer；
- output-first、truth-later audit；
- fail-closed terminal；
- unknown/stale hard veto；
- reset-scoped state；
- complete threshold frontier；
- worst-source/session/scene；
- discovery 成功只冻结 candidate；
- validation 与 sealed holdout 隔离；
- 权限字段默认 false。

但这些机制必须服务于“快速减少不确定性”，不能成为每个微实验都重新复制的专用协议栈。

允许在两个或更多边界出现稳定重复后，另立小型研究基础设施重构；在此之前禁止为了预想复用创建通用框架。

### 4.4 当前 person-track token 分支：降级为 adapter diagnostic

下列组合已成为局部死胡同：

```text
per-track active route relation
+ elapsed-time qualification
+ finite/persistent token
+ one-shot opening
```

它的问题不是参数不够好，而是当前输入无法区分：

- 长期位于 route-side 的普通背景人物；
- 相机前进导致的全局扩张；
- 目标自身接近；
- 横向穿过；
- 同向远离；
- episode 已被背景 activity 提前打开；
- 真正 alertable target 后到；
- route unknown、stale、gap 和 reset。

这个分支可以继续作为：

- dynamic-person evidence adapter 的回归诊断；
- opener/lifecycle 顺序问题的机制样本；
- attribution-before-opening 与 background namespace isolation 的状态机参考。

它不得继续作为：

- USTRF 核心风险表示；
- dense object-agnostic field 的替代证明；
- 通过增加 TTL、renewal、association 或特殊规则进行能力搜索的主线。

### 4.5 object-agnostic 的正式解释

`object-agnostic` 表示最终风险资格不依赖封闭类别表，不表示系统必须丢弃实例、运动或语义信息。

允许的 evidence adapter 包括：

- detector/objectness；
- tracker/instance continuity；
- relative/metric depth；
- optical flow/expansion；
- ego-motion/VIO/IMU；
- ground、drop、head-height geometry；
- semantic/open-vocabulary/VLM 辅助说明。

这些 adapter 只能贡献带质量、时钟、坐标和 provenance 的 evidence；任何单一 adapter 都不得直接取得用户提醒或生产安全权威。

### 4.6 route-conditioned 的正式解释

route 必须是显式、非未来、frame-bound、可失效的外部输入。

允许：

- route provider 输出 body/camera 绑定的走廊或轨迹；
- route projection 输出 current-frame polygon/field；
- route validity、age、quality 和 reset 进入 supervisor；
- true/uniform/shuffled route 作为研究对照。

禁止：

- 以画面中心走廊冒充真实用户路线；
- 以显著性、最长 free path 或 VLM 建议冒充用户已选择路线；
- 复制另一 episode 的逐像素 route trace；
- route 不可用时继续产生方向性导航结论。

route 不可用时的合法行为是 abstain、STOP/SCAN 研究输出或保持提醒链关闭，具体用户反馈仍需独立人因边界。

## 五、唯一共享的 canonical observation spine

所有后继 signal、teacher 和 U0 arm 必须消费同一版本化 observation spine，而不是逐实验补字段。

```text
source transport authority
        ↓
canonical frame packet
        ↓
geometry / calibration / pose / depth / route quality
        ↓
candidate-independent observation ledger
        ↓
signal or teacher adapter
        ↓
risk evidence frame
        ↓
field / lifecycle / supervisor
        ↓
offline event audit
```

### 5.1 canonical frame packet 最小字段

| 字段族 | 必需内容 | unknown 行为 |
| --- | --- | --- |
| source identity | source、session、sequence、frame、media SHA | 整帧拒绝 |
| time | capture timestamp、PTS、clock domain、gap | gap/reset；禁止插值冒充观测 |
| source geometry | width、height、pixel format、crop、orientation | 依赖归一化几何的 signal 不可执行 |
| canonical transform | source-to-canonical rotation/flip/crop/letterbox | 禁止使用 display 坐标 |
| detection/object | canonical bbox/mask、confidence、truncation、occlusion、observed state | unknown 单列；需要该字段的窗口弃权 |
| track/reset | track identity、association authority、reset scope、unobserved reason | 立即清状态 |
| camera model | intrinsics、distortion、camera/body extrinsics | metric/flow decomposition 不可执行 |
| pose/ego | pose delta、clock binding、quality、inlier/residual、tracking state | 禁止默认 ego-motion=0 |
| depth | relative/metric role、scale、registration、confidence、freshness | 不得把 relative depth 称为米制 |
| route | provider、route ID、projection、validity、age、quality | abstain/fail closed |
| event role | discovery/validation/holdout、truth join key | producer 阶段不可读取 truth |
| provenance | parent receipt、config SHA、ledger SHA、software identity | 终止 |

### 5.2 observation spine 的设计原则

1. source fact 与算法推断分开；
2. unknown 是一等状态，不用默认值填平；
3. 所有坐标都有 source、canonical、camera/body 或 world frame 名称；
4. 所有动态量都有 timestamp 和 clock domain；
5. 所有质量门都输出数值、阈值和 abstention reason；
6. signal adapter 不重新解释上游 geometry；
7. truth、event window 和 oracle 不进入 producer；
8. frame membership 在候选执行前冻结；
9. 同一 observation ledger 可服务 scale、flow、depth 和 U0；
10. 修复 spine 必须版本化，旧 blocked terminal 保持不可变。

## 六、持续研究阶段图

后续只按以下顺序推进。每一阶段必须先终局，再决定下一阶段。

```text
G0  authority / repairability audit
 ├─ authority absent ───────────────→ new data pack
 ├─ availability upper bound low ──→ stop current signal role
 └─ repairable
        ↓
G1  canonical observation repair
        ↓
G2  frozen pure-scale separability rerun
 ├─ blocked ────────────────────────→ repair or stop
 ├─ not sufficient ────────────────→ G3
 └─ discovery candidate ───────────→ independent validation
        ↓
G3  ego-motion signal availability
 ├─ unavailable ───────────────────→ metric depth / VIO / new data pivot
 └─ available
        ↓
G4  ego-motion-aware expansion attribution
        ↓
G5  temporal-depth teacher upper bound
        ↓
G6  independent validation + sealed holdout
        ↓
G7  formal six-arm U0 / LOSO
        ↓
G8  causal producer / lifecycle / offline replay
        ↓
G9  Android shadow
        ↓
G10 controlled device / human-facing review
```

G0–G7 均是研究阶段。G8 之后仍需独立权限，不因本总目标自动开放。

## 七、当前第一可执行边界：G0

> 2026-07-25 进度：G0 已按 A source-only inventory、B aggregate-denominator-only availability、第三进程 validator 完成，终态为 `SOURCE_AUTHORITY_ABSENT / VALID`。现有 41/41 sequence、62,229/62,229 frame 的 geometry/RGB/time/membership 可核验，但 canonical transform 全部 unknown、authoritative severe truncation 全部 absent。按本节优先级不得启动 G1；若继续，须另立 `CANONICAL_OBSERVATION_SOURCE_AUTHORITY_DATA_PACK_R0`，不得用 schema repair、heuristic truncation 或缩分母回救。日期化证据见 [G0 结果](USTRF_CANONICAL_OBSERVATION_AUTHORITY_AND_REPAIRABILITY_AUDIT_R0_RESULT_2026-07-25.md)。

> 2026-07-25 后继进度：JRDB 官方 test labels + sensor PDF 已形成 27 sequence / 27,661 frame / 956,803 object 的 source-authority canary；source-native truncation true/false/missing 为 30,889/925,799/115，独立复算为 `AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING / VALID`。这只证明新公开来源的 label/calibration authority 可行；RGB frame identity、timestamp 与 route-role truth 未物化，父 G0 不变，G1 仍不得启动。证据见 [data-pack R0 结果](USTRF_CANONICAL_OBSERVATION_SOURCE_AUTHORITY_DATA_PACK_R0_RESULT_2026-07-25.md)。

> 2026-07-25 access canary：JRDB 官方 toolkit 与 sample structure 已 hash-bound。static calibration、stitched image 路径合同和 `timestamps/` 目录约定存在，但 sample 只有 16 个空目录、0 payload；toolkit 把 label key 称为 timestamp，不能提供独立 capture clock。公开下载页要求登录，两个浏览器均无 JRDB 登录态，故独立复算终态为 `ACCESS_BLOCKED_LOGIN_REQUIRED / VALID`。当前不得猜 URL 或启动 G1；只有用户自行登录后，才可另立单 sequence、小预算的 RGB/time canary。证据见 [access canary R0 结果](USTRF_JRDB_RGB_TIME_FRAME_TRANSFORM_ACCESS_CANARY_R0_RESULT_2026-07-25.md)。

> 2026-07-25 登录后续：用户自行建立登录态后，官方清单暴露旧版 test images/timestamps/calibration 的精确 URL。R1 以 64 MiB 门对 22.5 GB ZIP64 做 byte-range，只读取 21.9 MB central directory 与一个 compressed JPEG；同一 `cubberly-auditorium-2019-04-22_1/000000.jpg` 的 9-object label、capture timestamp `1555960991.4668088`、3760×480 RGB 与 calibration 闭合。producer/validator 各 22,257,329 bytes，终态 `RGB_TIME_TRANSFORM_CANARY_PRESENT / VALID`。这解除新来源 transport blocker，但不改写父 G0，不开放 G1、route truth 或 signal；下一合法边界只可做短连续窗口的 RGB continuity/ego-motion availability。证据见 [single-frame R1 结果](USTRF_JRDB_SINGLE_FRAME_RGB_TIME_TRANSFORM_CANARY_R1_RESULT_2026-07-25.md)。

> 2026-07-25 短窗后续：冻结 32 帧/31 pair、person+16px mask、sparse LK 与单一 RANSAC full affine。timestamp、657–803 features、649–792 tracks、11–12/12 grid、residual、condition 和 determinant 均通过，但仅 11/31 pair 达到 inlier ratio ≥0.65，低于 28/31 availability 门；独立终态 `EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT / VALID`。不降门、不扩 JRDB sequence、不运行 G3/G4；后续只接受 metric depth、VIO/IMU、真实 route provider 或 route-authoritative 新数据。证据见 [continuity/ego-motion R0 结果](USTRF_JRDB_RGB_CONTINUITY_EGOMOTION_AVAILABILITY_R0_RESULT_2026-07-25.md)。

> 2026-07-25 原生 P1B 恢复：27 条 train bag 中最小 Meyer Green member 已做单 bag payload 审计。动态 `odom -> base_link` TF 3,183 条、`imu/data` 622 条、上下 Velodyne 471/478 条均以原生 header clock 覆盖外部前 120 帧 timestamp，第二进程完整复算为 `NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT / VALID`。这满足“stable pose/IMU 新 authority”恢复条件，但只允许另立 P2 perception/geometry canary；P2 尚未执行，正式 G1–G7 的 intended-route/event lifecycle、Android、human 与 production 权限仍关闭。证据见 [single-rosbag P1B 结果](USTRF_JRDB_SINGLE_ROSBAG_NATIVE_POSE_IMU_TIME_AUTHORITY_CANARY_R0_RESULT_2026-07-25.md)。

> 2026-07-25 P2 终局：Meyer Green 前 120 帧的 120 stitched RGB、240 PCD、2D/3D labels、动态 pose、IMU 与静态 TF 已形成第二进程可精确重建的 immutable packet。clock、PCD、frame chain 与 pose/IMU interpolation 门均通过，但 1,350 个 3D object-frame 中 29 个没有同帧唯一 2D `label_id`；按执行前冻结的全量 join 门，以 `FAIL_CLOSED_LABEL_JOIN / VALID` 关闭，motion pair 保持 0。不得改用 1,321 交集分母、换 sequence 或继续 route/event；证据见 [P2 结果](USTRF_JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R0_RESULT_2026-07-25.md)。

> 2026-07-25 P2 R1 纠错：R0 receipt 和旧合同终态不改写，但“完整 2D join 是 3D-native geometry/motion 的依赖”被确认为过宽。R1 使用 source-native union denominator：1,350/1,350 个 3D object-frame 与 1,336/1,336 个 adjacent motion pair 可计算；29 个 3D-only 只对 cross-modal identity abstain。由于全部 3D label 都是 source-interpolated annotation，direct observation 为 0，最大权威仍为 diagnostic，route/event/alert/Android/human/production 继续关闭。证据见 [弹性标准](USTRF_ELASTIC_EVIDENCE_AND_DEGRADATION_STANDARD_R1.md)与 [P2 R1 结果](USTRF_JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R1_RESULT_2026-07-25.md)。

阶段名：

`CANONICAL_OBSERVATION_AUTHORITY_AND_REPAIRABILITY_AUDIT_R0`

G0 是 scale-growth blocked result 所允许的 canonical input-contract repair 边界的 fail-closed preflight，不是新的 signal 实验，也不构成自动进入 Gate 2。

### 7.1 唯一研究问题

> 当前 41 条 sequence 的冻结上游证据是否真实包含、且能无假设地绑定 pure-scale 与后续 ego-motion 所需的 canonical geometry、truncation、RGB continuity 和 timestamp authority？

G0 不实现 signal，不计算 slope，不读取 truth outcome，不修 schema，不生成 frontier。

### 7.2 审计项目

G0 严格分为两个进程/阶段：

**A. candidate-blind authority inventory**

- 只读取 source transport、media metadata、canonical/frame/track/route membership 和字段 provenance；
- 不读取 event label、event window、oracle、candidate cell、negative-exposure role、signal、candidate 或 truth outcome；
- 逐帧冻结每个字段的 `authoritative / verifiable_transform / inferred / unknown / absent` 状态、原因和 parent receipt；
- 写出 inventory SHA 后结束；任何后续 denominator join 都不得改写 inventory。

**B. denominator-only availability audit**

- 新进程先复验 inventory SHA；
- 之后只联结已经冻结的 event/cell/negative denominator 与 eligibility key；
- 不读取 alert outcome、signal score、candidate output、truth class/outcome 或任何可用于调 signal 的数值；
- 只计算“若 unknown 即弃权，某字段合同最多能保留多少已冻结分母”的 availability upper bound；
- event、cell、negative 和 source/sequence membership 只能用于分母覆盖，不得产生 signal 或 policy 选择。

两个阶段共同逐 source/session/sequence 报告：

- authoritative source width/height 是否存在；
- encoded media orientation、container/display matrix 或 detector bundle rotation 是否存在；
- crop、resize、letterbox、flip 和 source-to-canonical mapping 是否可重建；
- bbox 是否确实位于 source、canonical 或 display frame；
- severe truncation 是否由 annotation/source metadata 提供；
- 若无 severe-truncation authority，能否合法标为 unknown，而不是 heuristic false；
- RGB frame 是否连续、逐帧可读且 SHA 可绑定；
- capture timestamp/PTS 是否单调、同 clock domain 且 gap 可枚举；
- 人物区域剔除后，背景特征的乐观可用范围是否能审计；
- route validity/reset/frame membership 是否能与上述 observation 对齐；
- 在 unknown 即弃权时，11 个 event、33 个 cell 和负暴露的最大可用 coverage。

### 7.3 G0 冻结 availability 门

对于当前 frozen pure-scale standalone role，`AUTHORITY_PRESENT_AND_REPAIRABLE` 必须同时证明修复后的乐观上界可保留：

| 分母 | 最低 availability |
| --- | ---: |
| supported unique event | `11/11` |
| supported candidate cell | `33/33` |
| frozen negative exposure | 完整 `4.95626851575min` |
| source/sequence | 每个冻结 source/sequence 的所需帧均有可判定 authority 或明确合法 invalid reason |
| frame membership | `41/41` sequence、`62,229/62,229` frame 可重建 |

这里的“合法 invalid reason”只允许由 source fact 或预先存在的 authoritative state 产生；它不能通过缩小 event/cell/negative 分母使 coverage 表面达门。

G0 同时报告 ego-motion 后继输入的 RGB/timestamp/background-feature availability，但这些结果只形成 G3 planning information，不改变本次 scale repairability terminal。G3 必须另立自己的 availability floor。

### 7.4 G0 工作量门

- 时间预算：`2–4h`；
- 优先使用现有 inventory、媒体 metadata 和只读 receipt；
- 不新增专用 signal producer、threshold frontier、Android exporter 或通用框架；
- 若需要超过半天或需要大规模重新物化，先以 gap matrix 终止并另立 repair/data goal；
- 输出必须说明每个字段是 source fact、可验证 transform、算法推断还是 absent。

### 7.5 G0 唯一合法终态与判定优先级

按以下顺序判定，命中后不得选择更乐观终态：

1. audit/inventory/denominator 不能完整冻结或复验 → `FAIL_CLOSED_AUDIT_INCOMPLETE`；
2. 至少一个 required field family 在现有 source/media/receipt 中根本没有权威来源，只能靠假定或 heuristic 创造 → `SOURCE_AUTHORITY_ABSENT`；
3. required authority 部分存在，但按 unknown 即弃权的乐观上界不能同时保留 `11/11`、`33/33`、完整负暴露和 41/41 membership → `AVAILABILITY_UPPER_BOUND_INSUFFICIENT`；
4. required authority 均存在、可无假设地修复且完整 availability 门可达 → `AUTHORITY_PRESENT_AND_REPAIRABLE`。

#### `AUTHORITY_PRESENT_AND_REPAIRABLE`

所有关键字段能从现有 source transport 或父 receipt 无假设地绑定；输出精确 repair plan、受影响 ledger、版本变化和预计 coverage。只授权另立 G1 repair，不授权直接计算 signal。

#### `SOURCE_AUTHORITY_ABSENT`

关键字段在源媒体、source metadata 和父 receipt 中均不存在，或只能依赖假定/heuristic。停止修当前数据承担该 signal 的接受职责，转向新采集或其他权威来源。

#### `AVAILABILITY_UPPER_BOUND_INSUFFICIENT`

字段虽部分存在，但 unknown 即弃权后的乐观 event/session/source coverage 已不足以承担 frozen signal 的 discovery 角色。停止实现该 signal，不通过缩小分母回救。

#### `FAIL_CLOSED_AUDIT_INCOMPLETE`

父输入、媒体、receipt 或审计链自身不完整，无法给出以上三种判断。输出精确缺口后终止。

## 八、G1：canonical observation repair

只有 G0 为 `AUTHORITY_PRESENT_AND_REPAIRABLE` 才允许启动。

G1 只做：

- 将 width/height、orientation 和 canonical transform 绑定到每帧；
- 将 authoritative truncation 或明确 unknown 绑定到每个 observed object/track；
- 版本化 observation schema；
- 重建 candidate-independent frame membership；
- 证明 C1–C3 projection 或其他重复 candidate 投影逐帧一致；
- 由独立 validator 从上游 authority 重建字段和 SHA。

G1 不做：

- signal 计算；
- threshold 选择；
- truth join；
- Android；
- source-specific special case；
- 把 absent 字段改成 assumed；
- 修改旧 blocked result。

G1 成功后，才允许以新配置版本重跑冻结 pure-scale R0。

## 九、G2：pure-scale separability

G2 沿用 [下一阶段新信号可分性目标](USTRF_SC_NEXT_STAGE_SIGNAL_SEPARABILITY_GOAL_2026-07-25.md) 的冻结定义：

\[
S_t=\frac{1}{2}\log(w_t^{norm}h_t^{norm})
\]

- past-only `600ms`；
- 至少 `5` 个有效观测；
- 最大相邻 gap `150ms`；
- 真实 timestamp；
- Theil–Sen slope；
- 不插值；
- bbox 触边/严重截断 invalid；
- route unknown、relation gap、track unobserved 或 reset 清窗口；
- 唯一扫描变量为 slope threshold；
- 输出完整 breakpoint frontier；
- discovery 门为 `11/11`、`33/33`、负机会 `<=2` 和冻结 delay；
- 成功只冻结一个 discovery candidate；
- 失败只关闭当前 standalone token role。

G2 不允许在结果后：

- 换成 height；
- 改面积公式；
- 改窗口；
- 改最小观测数；
- 增加 source whitelist；
- 加 qualification/TTL/renewal；
- 把 scale 失败扩大成“视觉扩张永远无用”。

## 十、G3：ego-motion signal availability

只有 pure-scale 终局后才进入。

G3 固定最小实现族：

```text
排除所有 detector-observed person bbox 及固定扩张区域
+ 背景 sparse Lucas–Kanade optical flow
+ RANSAC 2D affine
+ 固定质量门
```

G3 只回答：

- 背景特征数量是否足够；
- 特征空间分布是否足够；
- RANSAC inlier ratio 是否通过；
- reprojection residual 是否通过；
- affine condition 是否通过；
- frame gap/reset/route validity 下窗口是否可用；
- 质量门后的 event、cell、negative、worst-source 可用性上界。

G3 不同时加入：

- homography；
- dense flow；
- IMU；
- VIO；
- depth；
- multi-model voting；
- source-specific fallback。

质量不足时必须 abstain，禁止把 ego-motion 默认为 0。

若 G3 的乐观上界不足，停止 G4，转向 metric depth、VIO/IMU 或新数据。

## 十一、G4：ego-motion-aware expansion attribution

只有 G3 availability 通过才运行。

保留：

\[
L_{absolute},\qquad
L_{ego},\qquad
L_{relative}=L_{absolute}-L_{ego}
\]

解释边界：

- `L_absolute`：目标在图像中的总扩张，包含相机前进和目标自身运动；
- `L_ego`：由可验证背景运动模型解释的全局相机运动分量；
- `L_relative`：相对背景的额外扩张；
- `L_relative` 不是危险、距离、TTC 或碰撞概率；
- camera quality、route quality、bbox quality 和 abstention reason 必须同时输出。

冻结归因臂：

1. `absolute_only`；
2. `residual_only_diagnostic`；
3. `absolute_plus_camera_quality_gate`。

不得在同一 discovery 数据上任意组合多个阈值后声明获得融合 policy。

若 pure scale 与 ego-motion-aware expansion 均失败，立即停止单目 bbox feature search。

## 十二、G5：temporal-depth teacher upper bound

G5 的目的不是部署 Video Depth Anything，而是回答：

> 若提供更稳定的时序深度上界，route-conditioned event 指标是否出现可重复增益？

允许：

- Depth Anything V2 作为 single-frame teacher/control；
- Video Depth Anything 作为 offline temporal teacher；
- relative depth 与 metric depth 分开；
- current/past-only 与含未来帧 teacher 分开；
- teacher 仅输出 field/evidence，不取得 opener 或 feedback 权限。

必须报告：

- event recall；
- critical miss；
- first-alert delay；
- false alerts/min；
- clearance；
- repeat；
- field flicker；
- abstention；
- worst session/source；
- 是否依赖未来帧、共享 scale/shift 或 batch window。

停止条件：

- teacher 相对 bbox/scale/ego arms 无事件级增益；
- 收益只存在于单一来源、最佳 seed 或未来帧；
- relative depth 被错误解释为米制；
- 时序稳定性改善但事件闭环无改善；
- unknown 扩张造成表面 false-alert 下降。

teacher 通过也只授权形成 history-only causal student 的后继候选 goal；不得在本总目标下直接执行 G8、手机实现或 producer 集成。

## 十三、G6：新 validation 数据包

当前 11-event 数据只保留 discovery/falsification 角色。

新数据从采集或来源准入开始冻结为：

```text
discovery
validation
sealed holdout
```

初始目标：

- 至少 `3` 个独立 session；
- 合计约 `15–20min`；
- session/sequence cluster 独立；
- 不把一条长视频切成多个独立样本；
- 路线内 positive 与路线外/未侵入 matched negative；
- source/camera/session/scene 可追踪；
- 从源头绑定 frame geometry、orientation、timestamp、route 和 truncation。

### 13.1 受控场景矩阵

| 相机/用户运动 | 目标状态 | 主要归因 |
| --- | --- | --- |
| 相机静止 | 目标主动接近 | 目标自身 absolute expansion |
| 相机前进 | 静止目标位于路线中 | ego-induced approach 与真实 route obstruction |
| 相机前进 | 目标平行移动 | 长期共现 |
| 相机前进 | 目标同步远离 | net relative motion |
| 相机静止 | 目标横向穿越路线 | scale 弱、route relation 强 |
| 左右转向 | 静止背景人物 | rotation sweep-in 与假扩张 |
| 相机前进 | 远处路线外人物 | global expansion pollution |
| 多目标交叉 | 只有一人侵入路线 | identity 与动态背景 |
| 背景人物先出现 | 真目标后出现 | opening preemption |
| 短暂擦过路线边缘 | 无持续侵入 | transient false opportunity |
| 低矮/悬空障碍 | 类别未知 | object-agnostic geometry |
| 路线临时失效 | 任意目标 | fail-closed 与 reset |

### 13.2 数据来源与权限

- 允许依法、合规、已获授权的正常视力受控采集；
- 允许普通公开渠道可下载数据用于隔离内部研究；
- source URL、retrieval time、SHA 和已知条款必须记录；
- 提醒链必须关闭；
- 不进行盲人独立行走或 human-facing 试验；
- 本持续授权不包含招募、协调或组织新的参与者式物理采集；任何新实地/人体采集必须由用户另行明确授权具体范围，并具有真实可验证的 consent/source authority；
- 不伪造 consent、license、设备测量、route truth 或 event truth；
- 自动模型 review 必须按项目 AI review governance 隔离、复核和仲裁；
- model/proxy truth 必须明确命名，不能称为 human truth。

## 十四、G7：正式六臂 U0

只有 canonical observation、真实 route-conditioned event、独立 session 和所需 adapter 闭合后，才运行正式 U0。

六臂固定为：

1. `baseline_yolo_geometry`；
2. `detector_bbox_explicit_route`；
3. `teacher_dense_explicit_route`；
4. `teacher_dense_explicit_route_causal`；
5. `teacher_dense_uniform_route_control`；
6. `teacher_dense_shuffled_route_control`。

正式规模：

- `6` 个独立 session；
- `5` 类场景；
- `120` episode；
- `60` matched pair；
- `6-fold LOSO`；
- source/session/scene/worst-fold 报告。

[Evidence Maturity V2](USTRF_ROUTE_TARGET_EVIDENCE_MATURITY_STANDARD_V2.md) 的所有更严格 L3 条件继续生效，包括至少 `2` 个 provenance family、每 family 至少 `3` 个独立 session、cluster-aware interval 与 worst-family/source veto；满足上述 U0 数量本身不等于 L3 闭合。

正式 U0 只判断：

- dense 是否优于 bbox；
- explicit route 是否优于 uniform/shuffled；
- causal lifecycle 是否改善 onset/clearance/repeat；
- unknown/low/head-height/dynamic risk 是否有增益；
- fail-closed 是否在故障输入上保持安全方向。

U0 通过也只授权后继 student 或 offline producer 研究，不授权 App feedback、人体或生产。

## 十五、事件级指标和统计纪律

### 15.1 主指标

- non-abstain coverage；
- event recall；
- critical miss rate；
- false alerts per minute；
- first-alert delay；
- repeated-alert rate；
- post-event clearance；
- clearance P95；
- evidence age；
- route unknown/stale alert；
- abstention rate 与 reason；
- source/session/scene/fold worst case；
- device latency/thermal 作为独立工程指标。

### 15.2 禁止替代

下列指标不得单独替代事件结论：

- AP、mAP、small-object recall；
- MOTA、HOTA、IDF1；
- depth AbsRel、delta、temporal consistency；
- flow EPE；
- segmentation mIoU；
- VLM quality；
- GPU/Android latency；
- synthetic collision accuracy；
- 机器人 traversability cost。

### 15.3 样本与不确定性

- 同一 event 的 C1–C3 映射不是独立样本；
- 同一视频切片不是独立 session；
- 报告点估计、分子/分母、置信区间或 working UCB；
- 小样本成功只生成 candidate；
- discovery 数据上的失败可以关闭冻结假设；
- discovery 数据上的成功不能完成接受；
- validation 数据不得在查看结果后继续调参并保留 validation 名义；
- sealed holdout 只运行一次；
- worst-source/session/fold veto 不被 pooled mean 覆盖。

## 十六、研究投入产出规则

近期工作已经证明：协议与代码资产可能增长得比独立证据更快。后续每个阶段必须报告 `evidence delta`。

### 16.1 合法 evidence delta

至少满足一项：

- 新增一个当前 family 不具备的信息源；
- 新增独立 session/source/scene；
- 关闭一个完整假设族；
- 证明一个输入权限或可用性上界；
- 在未参与选择的数据上复验 candidate；
- 证明一项事件级机制归因；
- 关闭一个产品权限缺口。

### 16.2 不算算法进展

- 新增 schema、runner、validator 或 receipt；
- tests 数量增加；
- Android/host parity；
- hash 重算；
- blocked result；
- 负结果；
- benchmark latency；
- public dataset transport；
- paper metric 提高。

这些可以是必要研究工程或科学治理进展，但必须如实分类。

### 16.3 工作量硬规则

1. 一个实验只回答一个问题，只扫描一个主要变量。
2. 先用 inventory 计算 availability 和 optimistic coverage upper bound。
3. 在 preflight 通过前，不写完整 producer/frontier/Android/runtime。
4. 每个新 signal 的 discovery probe 默认在半天至一天内形成终态。
5. 预计超过一天或在 signal outcome 前需要大量专用代码时，拆出 G0/G1。
6. 不为一个微实验复制一套新的千行协议栈。
7. 独立进程复算只能称为确定性/隔离验证，不能称为独立算法实现复现。
8. 新通用框架必须由至少两个已经出现的稳定重复需求支持，并另立重构任务。
9. blocked 必须推动 repair、new data 或 stop；连续 blocker 文档而无输入改善是治理过密。
10. 每个结果文档必须回答“这次减少了什么不确定性”。

## 十七、防止再次进入死胡同

以下任一情况出现时，停止当前分支：

- 在同一输入 family 内继续调 TTL/qualification/renewal；
- availability upper bound 已低于目标；
- signal 成功只存在于 discovery 同一数据；
- 收益只存在于 best seed/source；
- 缺字段只能靠假定或 heuristic；
- route、frame、pose、depth、truth 或 prediction identity 不能绑定；
- residual 被单独解释为危险或 TTC；
- pure scale 与 ego-aware expansion 均失败后仍搜索 bbox 变体；
- temporal depth teacher 无事件增益后仍开发在线 student；
- unknown 扩张制造假性能；
- 结果依赖 future frame 却称为 causal；
- 通过缩小分母、删 source、改 event window 或特殊 track 规则回救；
- 在真实 route/event 未闭合前扩 Android 或 feedback；
- 用 11-event discovery 结果声称稳定跨来源能力；
- 用 model/proxy truth 声称 human outcome；
- 为完成文档而生成没有决策价值的更多文档。

## 十八、值得长期保留的优秀思想

### 18.1 项目内部已经形成的思想

- 从“检测到了什么”转向“什么正在阻断已选路线”；
- fast perception loop 与 slow semantic/feedback loop 分离；
- risk field 与 feedback kernel 分离；
- attribution-before-opening；
- background namespace isolation；
- candidate-blind producer；
- truth-later audit；
- optimistic feasibility bound；
- complete threshold frontier；
- event-level recall/false/repeat/clearance；
- true/uniform/shuffled route 负控；
- fail-closed unknown；
- discovery 失败可淘汰、成功只冻结 candidate；
- human/production authority 与 research evidence 分离。

### 18.2 需要修正的内部倾向

- evidence architecture 走在 perception observability 前面；
- person-track token 分支遮蔽 dense core；
- public source transport 被误当 event suitability；
- 多数投入落在协议、运输、恢复和 receipt，而不是新独立数据；
- 先写完整实验栈、后发现输入不可用；
- 把独立进程复算说得比实际独立性更强；
- 在正式 teacher upper bound 前准备过多后继工程。

## 十九、论文学习地图

论文只用于提出可证伪假设、设计对照和识别限制，不改变 USTRF 的 evidence maturity 或生产权限。

| 主题 | 论文 | 可学习思想 | 不可外推 |
| --- | --- | --- | --- |
| 主动感知 | [Bajcsy, Active Perception](https://doi.org/10.1109/5.5968) | 面向任务主动获取可判定信息；传感器与任务联合设计 | 不直接给出手机助盲架构 |
| looming/TTC | [Lee, Visual Control of Braking](https://doi.org/10.1068/p050437) | 光学扩张可提供 time-to-contact cue | bbox area slope 不是物理 TTC |
| ego-motion | [Bruss & Horn, Passive Navigation](https://doi.org/10.1016/S0734-189X(83)80026-7) | 从全场光流估计相机运动；局部噪声需要多点约束 | 静态环境假设不适用于动态人群 |
| optical expansion | [Yang & Ramanan, Optical Expansion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.html) | expansion 与 motion-in-depth/scene flow 的联系 | 单目存在尺度歧义 |
| scale matching | [Ling et al., Learning Optical Expansion](https://openaccess.thecvf.com/content/CVPR2023/html/Ling_Learning_Optical_Expansion_From_Scale_Matching_CVPR_2023_paper.html) | flow 与 expansion 联合估计优于脆弱两阶段假设 | 论文 benchmark 不证明助盲事件 |
| rigidity/scene flow | [Jiang & Okutomi, EMR-MSF](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_EMR-MSF_Self-Supervised_Recurrent_Monocular_Scene_Flow_Exploiting_Ego-Motion_Rigidity_ICCV_2023_paper.html) | 用静态/rigidity 区域估计 ego-motion | 不应直接复制复杂模型作为 G3 |
| dense motion field | [Occupancy Flow Fields](https://arxiv.org/abs/2203.03875) | occupancy 与 motion 在时空网格中分开建模 | 自动驾驶 agent metric 不是 BLV safety |
| traversability planning | [ViPlanner](https://arxiv.org/abs/2310.00982) | geometry、semantic costmap 与 planning objective 分层 | 机器人 simulation/terrain cost 不能外推人体 |
| field navigation | [WayFAST](https://arxiv.org/abs/2203.12071) | traversability map 与 state estimator/planner 模块分离 | RGB-D 轮式机器人结果不证明手持相机 |
| temporal depth | [Video Depth Anything](https://arxiv.org/abs/2501.12375) | 时序一致 depth teacher、temporal gradient 和长视频稳定 | affine-invariant depth 不是 past-only metric depth |
| abstention | [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) | risk–coverage–reject trade-off | classification reject 不等于 STOP/SCAN 用户策略 |
| ML 系统债务 | [Hidden Technical Debt in ML Systems](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) | boundary erosion、entanglement、data dependency、config debt | 不能用工程简化替代安全证据 |
| 欠定性 | [Underspecification](https://jmlr.org/papers/v23/20-1335.html) | 同等 held-out 表现的模型可能部署行为差异巨大 | 不能只用单一 pooled score 选模型 |
| 真实辅助系统 | [NavCog3](https://publications.ri.cmu.edu/navcog3-evaluation-smartphone-based-blind-indoor-navigation-assistant-semantic-features-large-scale-environment) | localization、routing、guidance、UI 与真实用户环境共同决定效果 | 室内 beacon 导航不证明障碍避让 |

### 19.1 固定学习方式

每次引入论文时使用：

```text
一篇或少量 primary papers
→ 提取一个架构或信号假设
→ 写清适用假设和不可外推范围
→ 建立一个 frozen control / teacher upper bound
→ 只看事件级指标
→ 失败关闭，成功进入新数据 validation
```

默认每个新边界只深读与该问题直接相关的 `1–3` 篇 primary paper；禁止为了显得前沿而堆叠模型。

## 二十、持续允许的动作与禁止动作

### 20.1 本总目标持续允许

- 读取和审计仓库、ignored evidence 与父 receipt；
- 建立最小 observation inventory、gap matrix 和 availability upper bound；
- 在 `artifacts.local/` 生成隔离研究证据；
- 新建日期化子 goal、配置、研究脚本、focused tests、validator 和结果；
- 获取普通公开渠道数据并记录 URL、时间、SHA 和限制；
- 使用已有且可验证授权的数据；不以持续 goal 代替新参与者协调、consent 或实地人体采集授权；
- 使用隔离模型/Agent 做 source、annotation、review 和 adjudication；
- 运行 offline CPU/GPU 研究与 host-only canary；任何 Android/device canary 必须等待对应阶段父门通过并另立独立 goal；
- 修复已证明 repairable 的 canonical observation contract；
- 对冻结 candidate 运行独立 validation 和 sealed holdout；
- 更新 USTRF 研究索引、开发日志和 handoff；
- 在结果允许时自动选择下一最小研究边界。

### 20.2 本总目标不持续允许

- 修改正式 App 风险、反馈或默认模型；
- 启用真实用户提醒；
- 进行人体、盲人或独立行走试验；
- 将 proxy/model truth 改称 human truth；
- 发布、生产替换或安全宣称；
- commit、push、PR 或 release；
- 修改凭据、系统环境或外部权限；
- 破坏性删除、批量迁移或清理其他任务资产；
- 绕过任何 blocked/reject terminal；
- 在同一 discovery 数据上不断调参；
- 同时运行多个主要变量；
- 自动进入 G8–G10。

若后继工作需要上述权限，必须停止并向用户取得新的明确授权。

## 二十一、每个子 goal 的最小模板

每个后继子 goal 至少包含：

1. `状态 / 最大权限 / 当前可执行边界`；
2. 唯一研究问题；
3. 父结果与 SHA；
4. frozen input、forbidden input；
5. data role；
6. availability upper bound；
7. 唯一主要变量；
8. producer/audit 隔离；
9. event-level metrics；
10. cluster/worst-source；
11. 资源与时间预算；
12. 唯一合法终态；
13. stop/pivot 规则；
14. 输出和 validator；
15. 本阶段不能声称的内容；
16. 下一阶段只在何种 terminal 下存在。

## 二十二、持续完成条件

> 2026-07-25 终局：七族 hash-bound 证据由 producer/validator 独立复算为 `EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY / VALID`。当前 source transport 可行，但 canonical transform/truncation、fresh metric geometry、inter-frame stable pose、intended-route truth 与独立 event lifecycle truth 未同时存在；因此 G1–G7 不能完整执行。该结论不是算法 reject，也不是任务在权威输入下不可观测。证据见 [program terminal R0 结果](USTRF_OBSERVABILITY_PROGRAM_REAL_WORLD_AUTHORITY_TERMINAL_R0_RESULT_2026-07-25.md)。

本总目标不是以“某个脚本运行完成”结束，而按里程碑更新状态。

### 22.1 研究架构里程碑

- [x] G0 authority/repairability audit 完成（`SOURCE_AUTHORITY_ABSENT / VALID`；G1 未开放）；
- [ ] G1 canonical observation spine 可复算；
- [ ] G2 pure-scale 获得合法终态；
- [ ] G3 ego-motion availability 获得合法终态；
- [ ] G4 expansion attribution 获得合法终态；
- [ ] G5 temporal-depth teacher upper bound 获得合法终态；
- [ ] G6 independent validation/holdout 数据角色闭合；
- [ ] G7 正式 U0 六臂 LOSO 获得合法终态。

### 22.2 允许的最终研究结局

本总目标可以以以下任一研究结局结束：

#### `USTRF_CORE_HYPOTHESIS_SUPPORTED_FOR_NEXT_EVIDENCE_LEVEL`

在独立 validation/holdout 和正式 U0 中，dense explicit-route causal arm 稳定优于 bbox、uniform 和 shuffled controls，并通过预注册 event/worst-source 门。只授权下一证据等级，不授权人体或生产。

#### `USTRF_CORE_HYPOTHESIS_NOT_SUPPORTED`

在权威输入、足够分母和正式对照下，核心 dense/route/causal 假设未获得稳定增益。保留负结果、可复用 safety kernel 和 adapter，停止核心算法继续扩张。

#### `TASK_NOT_OBSERVABLE_WITH_CURRENT_SENSOR_OR_ROUTE_STACK`

经过 scale、ego-motion、depth teacher 和 route/input 审计后，所需权威输入已经存在且实验可完整执行，但当前传感器/route stack 仍无法提供足够可分信息。停止算法堆叠，转向硬件、VIO/IMU、metric depth、route provider 或任务拆分。

#### `EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY`

核心研究尚不能获得所需 source fact、route truth、event truth、参与者 consent 或用户权限，因而实验不能完整执行。明确阻塞项并保持研究/产品权限关闭；不得把这一结局写成算法不可分。

当前最终结局：`EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY / VALID`（2026-07-25）。

任何结局都必须保留：

- 不扩大负结果；
- 不把研究成功改称产品成功；
- 不把 blocker 改称算法失败；
- 不以工程资产数量代替效果。

## 二十三、当前可直接持续启动的 `/goal`

```text
/goal 按 USTRF_SC_OBSERVABILITY_FIRST_CONTINUOUS_RESEARCH_GOAL_2026-07-25.md 持续推进 USTRF-SC 的 observability-first 研究。

先读取并重新验证：
1. current-input policy feasibility bound R0；
2. causal route-relative intrusion signal R0；
3. route-conditioned scale-growth separability R0；
4. Evidence Maturity V2；
5. 本持续研究总目标。

G0 `CANONICAL_OBSERVATION_AUTHORITY_AND_REPAIRABILITY_AUDIT_R0` 已闭合为 `SOURCE_AUTHORITY_ABSENT / VALID`。JRDB 单帧 authority canary 为 `RGB_TIME_TRANSFORM_CANARY_PRESENT / VALID`，但短窗 global-affine availability 为 `EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT / VALID`。不得启动当前数据的 G1，也不得扩大 JRDB、降低 inlier 门或把新来源当 route truth。下一信息增量只能来自 metric depth、VIO/IMU、真实 route provider 或 route-authoritative 新 data pack。

严格分两阶段：A 阶段 candidate-blind 冻结逐帧字段 authority inventory，event/truth/oracle/cell/negative/signal/candidate decode 均为 0；B 阶段新进程先复验 inventory SHA，再只联结冻结 denominator/eligibility key，计算 unknown 即弃权下的 availability upper bound，不读取 signal、candidate、alert 或 truth outcome。对于当前 scale role，repairable 必须能同时保留 11/11 event、33/33 cell、完整 4.95626851575min 负暴露、41/41 sequence 和 62,229/62,229 frame membership。

已完成的 G0 不实现 signal、不计算 slope、不读取 truth outcome、不修 schema、不生成 frontier、不写 Android/runtime/opener/通用框架；其 gap matrix 与三进程验证必须保留为后继 data-pack 的父终态。

G0 唯一合法终态：
1. AUTHORITY_PRESENT_AND_REPAIRABLE；
2. SOURCE_AUTHORITY_ABSENT；
3. AVAILABILITY_UPPER_BOUND_INSUFFICIENT；
4. FAIL_CLOSED_AUDIT_INCOMPLETE。

终态按 fail-closed 顺序判定：审计不能完整冻结先记 FAIL_CLOSED；字段族根本无权威来源记 SOURCE_AUTHORITY_ABSENT；权威部分存在但完整分母上界不足记 AVAILABILITY_UPPER_BOUND_INSUFFICIENT；只有权威均存在且完整门可达才记 AUTHORITY_PRESENT_AND_REPAIRABLE。

只有 AUTHORITY_PRESENT_AND_REPAIRABLE 才另立 G1 canonical observation repair；repair 通过后才以新版本配置重跑冻结 pure-scale R0。pure-scale 终局后才进入背景 sparse LK + RANSAC affine 的 ego-motion availability；质量不足必须 abstain，不能默认 ego-motion=0。pure scale 与 ego-aware expansion 均失败后，停止单目 bbox feature search，转向 metric depth、VIO/IMU、真实 route provider、新数据或任务拆分。

每个子阶段必须一个问题、一个主要变量、先 availability 后 implementation、失败关闭冻结角色、成功只生成 validation candidate。当前 11-event/4.956min 数据只承担 discovery/falsification；接受必须使用新独立 session 的 validation 与 sealed holdout。所有结论使用事件级 recall、critical miss、false alerts/min、delay、clearance、repeat、abstention 和 worst-source/session/fold。

持续允许仓库内离线研究、版本化子 goal、脚本/测试/validator、artifacts.local evidence、普通公开数据准入和文档更新；不授权新参与者协调/consent/实地人体采集、正式 App、用户反馈、人体/独立行走、生产、commit、push、PR、release 或破坏性操作。任何阶段不得绕过父 terminal；G8 以后的 Android/human/production 边界必须重新取得明确授权。

保持 mobileclip_blt.ts 和所有其他并行改动不动。每个阶段结束后更新日期化结果、USTRF 研究索引、DEVELOPMENT_LOG 和 task handoff，并根据合法 terminal 自动选择下一最小可执行边界；若不存在合法下一边界，停止并报告。
```

## 二十四、维护规则

- 本文件是 `current` 研究指导；变化时直接更新本文并记录原因；
- 单次实验数字和 SHA 写入日期化结果，不反复复制到本文；
- 已关闭终态不得因本文更新而改写；
- 每次架构级变更更新 `docs/research/ustrf-sc/README.md`；
- 每次实际技术决定更新 `DEVELOPMENT_LOG.md`；
- 产品行为未变化时不更新根 `README.md` 或 `CHANGELOG.md`；
- 未来若本文件被新 current goal 取代，保留为 snapshot/archive 并在索引注明 successor。
