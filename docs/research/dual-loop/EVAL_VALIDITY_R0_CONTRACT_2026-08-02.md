# EVAL-VALIDITY R0：先验证评价器，再比较模型

状态：`PRE_OUTPUT_LOCKED / OUTPUT_BLIND_CONTINUOUS_INPUT_MATERIALIZATION_IN_PROGRESS / NO_MODEL_TRAINING / DEFAULT_APP_UNCHANGED`

机器可读合同：[EVAL_VALIDITY_R0_CONTRACT_2026-08-02.json](EVAL_VALIDITY_R0_CONTRACT_2026-08-02.json)。

## 结论与边界

本合同不试图挽救或重算任何旧算法。它要回答的是更基础的问题：在新的、session-disjoint 的自然事件上，当前评价链能否把“场景里有什么”与“现在是否应提醒、是否已经解除”分开，并在输入信息严格变好时给出不更差的事件结果。

已消费的 RISKSEG 30-event cohort 只说明该问题值得审计；它不可重新评分、重标、调阈值或充当本合同的任何分母。这个审计不训练模型，不修改 Android、风险规则、默认 YOLO 或反馈行为。

当前 48-session screening universe 已在 source-mask-only 层冻结并通过连续窗口元数据预检；其中 4 个
越界窗口仅在原 session 内向前移动，保持原始 source-screening reference frame。native RGB/mask
materialization 正在进行，尚未产生最终 payload manifest、P0 review、任何 arm trace 或事件质量结果。

## 两层事实与两个前置门

| 层 | 可使用的事实 | 不可声称的事实 | 前置门 |
|---|---|---|---|
| 场景事实 | native RGB/mask hash、区域、组件、确定性 truth box/mask | “应提醒”或安全真值 | 新 event/source-session 通过完整污染审计 |
| 用户事件事实 | P0 的两名隔离 reviewer 对四个**各自独立 opaque item** 的因果 RGB anchor 的 `reminder_now`、`cleared`、`knownness`；P0 通过后 P1 的两份全事件因果 RGB review 冻结区间 | 语义 mask 本身等于行动真值 | P0/P1 都先于任何模型/oracle trace |

`UNKNOWN` 与 reviewer disagreement 都只能成为 `NOT_EVALUABLE`，绝不自动变成“不提醒”“已通过”或“安全”。P0 不丢弃 anchor 来凑分母：三个原语标签与每个 parent-event 序列都要求 `100%` exact agreement，任一未解决 anchor 即 fail closed。P0 通过也只授权 P1 两份新的全事件 review；P1 仍有任一 unresolved frame/interval 时不读取 trace。trace manifest 必须绑定 P1 全事件事实账本的 SHA-256；不使用第三 reviewer 消除分歧。

## 新 cohort

冻结要求是 48 个 parent events、48 个 native source sessions，四桶各 12 个：障碍正例、边界/落差正例、平行路沿负例和正常通行负例。每个事件必须有 20 个以上连续帧、四个固定因果 review anchors；正例还必须含有 pre-alert、reminder-now 与 passed-clear anchor。

候选 session 必须同时避开 RISKSEG train/dev/fixed regression 与旧 30-event cohort。冻结之后需要跑完整的 exact/RGB/session/ancestry/parent/pHash 污染审计；任何未解决的 pHash 候选均为 `HOLD_EVAL_VALIDITY_DATA`，不是“无重叠”。

## Oracle 单调性阶梯

四臂都使用同一冻结 clock、decision kernel、risk config 和 feedback accounting：

```text
current YOLO → source-fact truth box → source-fact truth mask → event-fact synthetic oracle
```

每一级都必须对前一级“不更差”：正例命中不能减少、关键漏报或 pre-alert 误提醒不能增加、负例误提醒不能增加、通过后清除不能减少，而且共同命中的首次提醒不能更晚；这些比较同时在总计和每个 bucket 上成立。最后的 synthetic oracle 必须全命中、全清除、零 pre-alert/负例提醒，否则评价器/评分链完整性失败，而不是模型失败。

## 输出与终态

审计器总是分开报告。P0 anchor 是构造一致性筛查，event 指标只用 P1 冻结的完整 alertable/passed intervals 和全帧 feedback trace：

- 表征层：coverage、false area、component recall、false components、fragmentation 与 temporal stability。这一层只用于后续研究排序。
- 事件层：命中、critical miss、pre-alert/false-alert event、clearance、首次响应时延及每事件/桶明细。这一层只决定后续候选是否有资格前进。

可能终态为 `STOP_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED`、`STOP_EVALUATOR_INTEGRITY_NOT_ESTABLISHED`、`STOP_ORACLE_MONOTONICITY_NOT_ESTABLISHED` 或 `VALID_EVALUATION_CONSTRUCT_AND_ORACLE_LADDER`。即使最后一个终态成立，也只证明这个评估合同可用于未来比较，不授权模型训练或默认 App 切换。
