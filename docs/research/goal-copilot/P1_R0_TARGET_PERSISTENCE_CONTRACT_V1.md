# P1-R0 Target Persistence representation / evaluator contract V1

状态：`FROZEN_REPRESENTATION_EVALUATOR_CONTRACT / CANARY_LITE / SYNTHETIC_MECHANICS_ONLY / NO_VISUAL_PERSISTENCE_CLAIM`

机器合同：[`P1_R0_TARGET_PERSISTENCE_PROTOCOL_V1.json`](P1_R0_TARGET_PERSISTENCE_PROTOCOL_V1.json)

## 规范性边界

> **Persistence may preserve or reject identity continuity; it may not establish semantic referent validity.**

P1 只能维持、削弱、暂时看不见、丢失或重新确认一个 P0 已建立的 episode-local physical referent。
时间连续性、长期跟踪或多帧一致性都不能反向证明 P0 的语义 grounding 正确。

移动视角为 `AMBIGUOUS / NO_REFERENT` 增加新的 grounding evidence 属于未来独立的 Active/Temporal
Grounding，不属于 P1-R0。

## P0 → P1 hard handoff

P1 对上游只接受：

```text
REFERENT_ESTABLISHED
NO_REFERENT
```

只有 `REFERENT_ESTABLISHED` 可以绑定不可替换的 `referent_id`。P0 的 `AMBIGUOUS / ABSTAIN /
NO_CANDIDATE / NOT_OBSERVED / NO_REFERENT` 在 P1 边界统一为：

```text
state = UNBOUND
referent_id = null
```

即使后续出现高分、稳定或显眼的同类 candidate，P1 仍无权自行建立 referent。机械硬门为
`illegal_bind_rate = 0`。

## 最小 representation

身份是 episode-local physical identity，不是类别或 bbox：

```text
goal_id
referent_id
grounding_provenance
state
current_candidate_id | null
identity_support
identity_contradiction
stability
oscillation
frames_since_confirmed
event
```

四类 score 都只是算法 evidence，明确不是 calibrated probability。

五个状态：

- `UNBOUND`：没有合法 referent；必须 `referent_id=null`。
- `TRACKING`：当前 candidate 被断言为同一个 physical referent。
- `UNCERTAIN`：存在候选，但 identity continuity 不足；不得断言当前位置。
- `TEMP_UNOBSERVABLE`：保留 referent memory，但当前没有 observation；不等于 LOST。
- `LOST`：identity continuity 已断；不得声称知道 referent 当前在哪里。

事件：`TARGET_TEMP_UNOBSERVABLE / LOSS_DETECTED / REACQUIRED / REGROUND_REQUIRED`；`REACQUIRED`
不是永久状态。`TRACKING` 之外的状态不得输出 `current_candidate_id`。

## Evidence 与负证据

接口对具体算法保持中立，可由 appearance、motion、geometry、flow、candidate correspondence、camera
motion、semantic consistency 或 multi-frame embedding 产生 evidence。但 accumulated memory 必须同时接收：

```text
identity_support
identity_contradiction
```

因此新的矛盾、候选 A/B 交换、geometry jump 或 appearance inconsistency 可以使系统从 `TRACKING`
退回 `UNCERTAIN` 或 `LOST`；历史 support 不能永久锁死信心。

## Evaluator isolation

Evaluator 分两层：

1. `P1_CORE`：oracle initialization 直接给出第一帧 physical referent，只测 persistence，不把 P0 质量乘入。
2. `P0_P1_HANDOFF_SAFETY`：输入 `NO_REFERENT`，后续任意候选都不得产生 persistent `referent_id`。

系统只读取独立 `p1_input_schema.json` 的 public frame candidates 及 algorithmic identity evidence。
`candidate_instance_map`、physical instance truth、referent observability、允许状态和事件只存在于 evaluator
episode；baseline 对带 truth 的 envelope 直接拒绝。

## Safety-first lexicographic evaluator

比较顺序固定为：

```text
1. illegal bind frames
2. wrong-instance asserted frames
3. identity switches
4. false reacquisitions
5. maximize correct identity coverage
```

不生成加权总分。一个 coverage 更高但 wrong-instance 或 false-reacquisition 更差的候选不能胜出。

次级指标包括 reacquisition precision/recall、time-to-reacquire、false-loss、loss-detection latency，以及
`wrong-lock persistence max frames / duration`。零分母保持 `null`。

## 最小 mechanics 场景

`scenarios.json` 只含八个 synthetic fixtures：持续可见+相机移动、短遮挡、转头后返回、同类 distractor
横穿、相似 candidate 交替、长期离开、LOST 后错误相似目标，以及 `NO_REFERENT` guard。

它们只覆盖合法值、状态退化、hard guard 和 evaluator attribution；不是图像 benchmark、真实 tracker
性能、ADT 复评或 scientific cohort。

## 最简单 baseline

`baseline.py` 使用固定的 `identity_support - identity_contradiction` 排序、固定阈值、短期
`TEMP_UNOBSERVABLE` 和 eager reacquisition。它没有模型、图像、evaluator truth 或 Sky 调用。

这个 baseline 故意保留普通短期 tracker 的缺点，用于让 evaluator 暴露 distractor switch、候选交替和
false reacquisition；它不是准入候选。

稳定实现入口：[`p1_persistence/`](../../../scripts/research/goal_copilot_bridge/p1_persistence/)。专项检查：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p1_persistence/test_contract.py

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.p1_persistence.evaluator
```

## Claim ceiling 与 successor

当前只建立 schema、deterministic evaluator、synthetic mechanics fixtures 与 simple baseline failure
surface。它不建立真实 RGB persistence、P0×P1 end-to-end、导航、用户、安全、Android 或默认 App 结论。

唯一 successor 是在 mechanics 验证后，单独设计 consumed ADT 的 baseline adapter；不得在本轮启动
Sky、模型搜索、fresh/large cohort 或 Active/Temporal Grounding。
