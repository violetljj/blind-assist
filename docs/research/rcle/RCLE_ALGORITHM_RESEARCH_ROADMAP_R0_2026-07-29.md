# RCLE 新算法研究推进路线 R0

日期：2026-07-29

路线状态：`ROUTE_ADOPTED / GATED_PROCESS / A_INTERRUPTED_ZERO_COMPLETE_HOLD`

## 结论

RCLE 后续默认研究顺序调整为：

```text
A. 四臂运动分量定位
  -> 按 A 结果确定 B 的诊断重点或先关闭实现问题
  -> B. 平移—深度 oracle 与目标接近正对照
  -> B 后 go / modify / stop
       -> stop：降低地位或关闭 RCLE 主线
       -> freeze：直接进入 C
       -> modify：只做一个针对性升级及 base-vs-upgrade 对照，再进入 C 或关闭
  -> C. 冻结最终保留版本的最小 RCLE 内生特征合同
  -> D. 有合格独立标签数据时才验证融合增量价值
```

这是一条新的算法研究路线，不是既有
`RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2` 两臂正式实验的改名或补跑。
旧结果、失败终态、identity、receipt 和执行权限继续保持不可变；QMS-R1 successor
`480+16` 的实际权限始终以其最新有效 lock、receipt 和 RCLE README 为准，本路线
既不消费也不撤销该权限，并且不把它视为必须自动执行的下一步。当前操作决定为
`HOLD / NOT_CONSUMED`：不与 A、B 并行运行。

这不是必须依次完成 A、B、C、D 的流水线。A 购买 B 的具体问题，B 购买“升级、
直接冻结或关闭”的决定；C 是收口而不是继续开发，D 是最后的条件性价值检验。
任何阶段失败、混合或不可评价，都允许形成论文结论；不得为了得到正结果而跳过
关卡、降低门、增加大系统或提前进入 Android。

## 共同研究约束

1. 科学状态、协议状态和执行权限始终分开报告。
2. scene cluster 或独立生成 identity 是分析单位；frame、pair、cell、周期和实验臂
   都是重复测量，不能作为独立样本。
3. 所有身份、对比、主要指标、停止条件和第二阶段 routing 必须在读取对应 RCLE
   response 前冻结。
4. `signed expansion` 与 `absolute leakage` 分开报告；弃权保留为弃权，不填零。
5. 只允许最小、可复算的研究实现。没有测得收益前，不新增 provider、bus、统一上下文
   或 Android 接口体系。
6. 合成、oracle、development 和正式验证的证据层级不得混写。它们都不能直接支持
   真实场景、助盲效果、产品或安全结论。
7. 既有 R3、strict `>0.01/s`、三 pair、abstention reset 和 PairState 保持不变，
   除非未来另立问题、证据和授权。

## 关卡关系

### A 如何决定下一步

| A 的模式 | 下一动作 |
| --- | --- |
| `TRANSLATION_ONLY` 或 `FULL_6DOF` 明显较高 | B 以平移—深度 oracle subtraction 为主 |
| `ROTATION_ONLY` 仍明显较高 | 先限界审计旋转补偿、坐标、warp、valid mask 与局部拟合；问题未闭合前不升级 |
| 只有 `FULL_6DOF` 明显较高 | B 保留 rotation×translation interaction 对照，不能只做单独平移 |
| 四臂方向混乱、不可复现或 coverage 异常 | 先查 generator、identity、transport 和实现；保持 B `HOLD` |

A 只负责定位敏感来源，不从 A 直接推出新算法有效，也不允许在读取结果后新增运动臂、
改变阈值或把 Stage 2 变成救援样本。

### B 后的正式决策

B 完整结束后必须产生且只产生一个主决策：

| 决策 | 条件含义 | 后续 |
| --- | --- | --- |
| `GO_SINGLE_TARGETED_UPGRADE` | oracle 明显收缩 ego-motion leakage，同时稳定保留目标接近正对照 | 只实现一个 RCLE-v2 候选，做 base-vs-upgrade |
| `FREEZE_AS_CONDITIONAL_RESIDUAL_FEATURE` | oracle 收益一般，但 RCLE 在限定条件下仍稳定可解释 | 不再追求彻底自运动分离，直接进入 C |
| `STOP_OR_DOWNGRADE_RCLE` | oracle 无法改善，或目标接近响应也不稳定 | 降低 RCLE 地位或关闭主线；C、D 不再是必做项 |
| `B_NOT_EVALUABLE` | 几何、visibility、正对照或 paired identity 未闭合 | 只修失败的 B evidence version，不解释升级价值 |

若进入 targeted upgrade，只允许：

```text
原始 RCLE
vs
RCLE + 一个由 B 购买的单项补偿
```

升级必须沿用冻结数据单位和公平基线。有明确、可复现收益才保留为候选版本；无收益即
丢弃候选并回到 freeze 或 stop 决策，不串联第二、第三个补偿模块。

## A. 四臂运动分量定位

### 研究问题

在场景、材质、光照、内参、时间戳和随机流保持配对的情况下，RCLE 的补偿后残余
expansion 主要由旋转、平移还是二者交互产生？

### 冻结设计

每个 scene cluster 运行四个 clean arm：

```text
STATIC           R=I,        T=0
ROTATION_ONLY    R=periodic, T=0
TRANSLATION_ONLY R=I,        T=periodic
FULL_6DOF        R=periodic, T=periodic
```

采用两阶段 `16+16 sequences`：

- Stage 1：4 个新 scene clusters，每 cluster 四臂，共 16 sequences；
- Stage 2：另 4 个新 scene clusters，共 16 sequences；
- 32 个 sequence identities 在 Stage 1 前一次性冻结；
- Stage 2 初始为 `SEALED_NOT_EXECUTABLE`，只能由独立 Stage 1 routing receipt
  打开；
- Stage 1 只承担机制 routing，不做 p 值、置信区间或正式总体推断。

如果现有两臂 QMS-R1 formal、DEV、CAL、PREFLIGHT 或本路线其他 stage 已使用某个
identity、seed、scene geometry 或 token，新路线必须全域不相交。

### 主要指标

每条 sequence 固定报告：

- compensated signed expansion `P50/P90`；
- compensated absolute expansion `P50/P90`；
- evaluable pair 内 strict-positive response ratio；
- 固定 601-pair 分母下的 positive-response density 与 trigger density；
- longest positive streak；
- evaluable fraction。

空间分布、cell fit residual、support、与角速度/平移速度的关系及主频只作解释性
诊断，不参与 Stage 2 routing。

### 预注册对比与 routing

```text
ROTATION_MINUS_STATIC
TRANSLATION_MINUS_ROTATION
FULL_MINUS_MAX_SINGLE
```

Stage 1 只在至少一个冻结对比于 `>=3/4` blocks 呈一致非零方向时打开 Stage 2。
否则终止为 `A_STAGE1_INCONCLUSIVE_HOLD`。Stage 2 必须复现对应方向，才能形成
`A_COMPONENT_DIRECTION_REPLICATED`；不复现则为 `A_COMPONENT_NOT_REPLICATED`。
执行、hash、identity、配对或 firewall 失败单独记为 `INVALID`，不能换 seed 救援。

### A 阶段不做

- 不运行或消费既有 QMS-R1 successor formal；
- 不加入 blur、low-texture 或第二套 tracker；
- 不做目标接近正对照或平移—深度 subtraction；
- 不修改 R3、threshold 或三-pair 状态机；
- 不进入 sequence16、Android、实时或融合模型。

## B. 平移—深度 oracle 与目标接近正对照

### 进入条件

A 的完整结果只决定 B 的重点或是否先 `HOLD` 排错：

- translation/full 主导时，B 直接以 translation-depth subtraction 为主要问题；
- rotation-only 主导时，先闭合旋转补偿、坐标、warp 和局部拟合，再决定 B fixture；
- 只有 full 主导时，B 必须显式保留 rotation×translation interaction；
- A 混乱、不可复现或不可评价时，B 保持 `HOLD`。

A 的 Stage 1 routing 不能直接授权 B；A 的完整终态也只授权另立 B 的几何合同和
fixture review，不自动授权读取 B 的算法结果。

### 研究问题

在 source-known pose、metric depth、内参、可见性和 scene motion 下，解析的相机
平移 flow 能否去除 ego-translation 引起的 expansion，同时保留独立目标接近的
正响应？

### 最小实现

B 只实现一个离线、可复算的 Python 几何脚本：

```text
pixel + Z(t) + K + T(t->t+1)
  -> back-project
  -> camera-translation transform
  -> re-project
  -> u_trans(T,Z)
  -> observed flow - u_trans
  -> unchanged local expansion fit
```

不建立 `DepthProvider`、`PoseProvider`、`EgoFlowEngine`、`FusionContextBus` 或
Android pipeline。

### 几何闭合门

读取 RCLE response 前，至少通过：

1. `T=0` 时预测 flow 严格为零；
2. 恒深度 fronto-parallel plane 与解析解一致；
3. 平移方向反转时径向 flow 符号按预期反转；
4. 原分辨率与缩放内参产生一致 normalized-coordinate 结果；
5. 明确绑定 pose 方向、depth 所属帧和 `dt`；
6. z-buffer/visibility 一致，遮挡、新显露、深度不连续和越界像素为无效，不插补；
7. moving object 不得被静态背景 ego-flow 解释项吞掉；
8. oracle subtraction 后重新执行同一局部 expansion fit，不复用减法前拟合量。

### 最小实验臂

至少包含配对的：

```text
STATIC_SCENE
EGO_TRANSLATION_STATIC_SCENE
OBJECT_APPROACH_STATIC_CAMERA
OBJECT_APPROACH_PLUS_EGO_6DOF
```

主要判据同时要求：

- ego-translation static-scene 的 absolute leakage 明显收缩；
- object-approach positive control 的 signed expansion 与 evaluability 被保留；
- combined arm 中不能通过抹掉全部 flow 获得“成功”。

若几何 fixture、visibility、正对照或 paired identity 任一失败，终止为
`B_ORACLE_NOT_EVALUABLE`。B 只能回答理想条件下的理论可分性，不代表单目深度、
IMU、手机同步或实时部署可行。

## C. 冻结最终保留版本的最小 RCLE 内生特征合同

### 进入条件

只有 B 后决策为 `FREEZE_AS_CONDITIONAL_RESIDUAL_FEATURE`，或一次 targeted
upgrade 已通过 base-vs-upgrade 对照并被保留，才进入 C。若决策为
`STOP_OR_DOWNGRADE_RCLE`，C 与 D 可以直接关闭。

只有 A/B 和可选单项升级已经表明某些量具有稳定、可解释且非重复的信息时，才冻结
字段。字段存在于当前代码、看起来“以后可能有用”或为了 schema 完整，都不是准入
理由。

候选字段仅限：

```text
signed_expansion
absolute_leakage
spatial_distribution
positive_ratio
support
fit_residual
evaluable
confidence
abstention_reason
```

每个入选字段必须冻结定义、单位、时间窗、分母、缺失语义、范围、版本和来源阶段。
允许减少字段，不允许为了统一接口加入未经实验证明的抽象。

以下属于外部上下文，禁止进入 RCLE 内生合同：

```text
depth
camera_translation
YOLO_semantics
route_occupancy
final_risk
```

C 必须同时冻结版本身份。如果单项升级被保留，可明确区分：

```text
RCLE-v1：rotation-compensated residual expansion
RCLE-v2：经 B 验证的单项 ego-motion-explained candidate
```

进入 D 后不得继续修改 RCLE 算法、字段定义、缺失语义或版本选择；否则 D 的融合增量
结论作废并必须另立版本。

C 的交付物最多是一份字段合同、一份 schema fixture 和必要的序列化测试；不建立
provider 层或融合框架。

## D. 融合增量价值实验

### 数据准入门

D 不是必做流水线。只有 RCLE 未在 B 后被关闭，且以下条件同时满足，才能进入实现：

- RCLE 至少在部分条件下稳定，失败边界已明确；
- C 已冻结最终版本与字段合同；
- 接近/非接近标签定义、粒度和时间窗已冻结；
- 标签来源、误差和不确定性可审计；
- session、route、scene 或生成 identity 能形成真正独立的划分；
- baseline 没有直接或间接包含与 RCLE 等价的信息；
- train、tune、test 间不存在共享场景、纹理、轨迹或生成参数泄漏。

如果这些条件不成立，终态为 `D_DATA_NOT_EVALUABLE`，不得以 pair 数量补偿独立数据
不足。

### 最小比较

模型只允许从逻辑回归或小型 GBDT 开始，至少比较：

```text
EXTERNAL_CONTEXT_ONLY
RCLE_ONLY              # 诊断，不承担产品主张
EXTERNAL_PLUS_RCLE
```

主要 endpoint、阈值选择、校准、错误代价和 guardrail 必须依据数据任务在训练前
冻结。测试集只运行一次；所有超参数和阈值选择都在训练/验证 session 内完成。

只有 `EXTERNAL_PLUS_RCLE` 在独立测试单位上相对公平 baseline 产生稳定增益，且没有
以不可接受的关键漏检、假阳性、校准或弃权恶化换取，才能写成
`D_INCREMENTAL_VALUE_SUPPORTED`。否则报告 `NO_MEASURED_INCREMENTAL_VALUE`、
`INCONCLUSIVE` 或 `NOT_EVALUABLE`，不扩模型复杂度救援。

## 阶段权限与默认动作

| 阶段 | 当前权限 | 默认动作 |
| --- | --- | --- |
| 路线文档 | `ADOPTED` | 作为后续算法研究默认顺序 |
| A 合同、identity、静态验证 | `PREPARATION_RECORDED` | 保留现状，不自动继续 |
| A Stage 1 | `OPERATOR_STOPPED_ZERO_COMPLETE / INCOMPLETE / NOT_INTERPRETABLE / HOLD` | 不恢复、不解释 |
| A Stage 2 | `SEALED_NOT_EXECUTABLE` | 等待独立 Stage 1 routing |
| B | `DESIGN_ONLY / A_ROUTED` | 等待完整 A 结果决定诊断重点 |
| B 后决策 | `MANDATORY_BEFORE_UPGRADE_OR_C` | go / freeze / stop / not-evaluable 四选一 |
| 单项升级 | `OPTIONAL / B_EVIDENCE_REQUIRED` | 最多一个候选，只做 base-vs-upgrade |
| C | `CONDITIONAL_CLOSEOUT` | 仅冻结最终保留版本 |
| D | `OPTIONAL / DATA_DEPENDENT` | 合格独立标签存在时才训练 |
| 既有 QMS-R1 successor `480+16` | `HOLD / NOT_CONSUMED / PRESERVE_LATEST_VALID_LOCK` | 不与 A、B 并行运行 |
| sequence16 / Android / 产品 / 安全 | `NOT_AUTHORIZED` | 保持关闭 |

本轮只落实推进过程。并行任务曾启动 A Stage 1，但在用户澄清后已停止：进度为
`0/4 clusters、0/16 sequences`，只有 heartbeat/telemetry，没有 arm、cluster、
run receipt、analysis、independent receipt 或 routing decision。因此它不是 A
结果，也不得解释为完成、失败或 `NOT_EVALUABLE`。

下一次只有在用户明确要求运行 A 时，才重新核验冻结合同、32 identities、formal
firewall、静态测试、独立 validator 和本次 stop record，并另行决定能否沿用 activation
或必须新立执行版本；不得把本次 stale `RUNNING` 目录直接当作可恢复授权。
