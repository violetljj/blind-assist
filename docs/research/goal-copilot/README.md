# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / P0_POLICY_DISCOVERY_CLOSED / A1_DEVELOPMENT_INCUMBENT_NOT_ADMITTED / P1_TARGET_PERSISTENCE_R0 / REPRESENTATION_EVALUATOR_CONTRACT_FROZEN / P1_D0_TEMPORAL_DEVELOPMENT_COHORT_READY / NO_SCIENTIFIC_VERDICT / NO_SKY / DEFAULT_APP_UNCHANGED`

完整系统蓝图见 [`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)。本页是
Goal Copilot 动态执行状态真源；详细历史、协议和数字留在链接的独立结果文件，不在 current 重复维护。

## 当前主线：P1 Target Persistence

P1-R0 回答“现在观察到的是否仍是 P0 刚才建立的同一个 physical referent”，不重新判断哪个门满足目标。
规范性边界是：

> **Persistence may preserve or reject identity continuity; it may not establish semantic referent validity.**

`NO_REFERENT` 永远保持 `UNBOUND / referent_id=null`；多帧一致性不能反向证明 P0 correctness。完整合同与
机器合同见 [`P1-R0 contract`](P1_R0_TARGET_PERSISTENCE_CONTRACT_V1.md) /
[`JSON`](P1_R0_TARGET_PERSISTENCE_PROTOCOL_V1.json)。

P1 最小状态固定为：

```text
UNBOUND
TRACKING
UNCERTAIN
TEMP_UNOBSERVABLE
LOST
```

系统同时积累 identity support 与 contradiction；memory 必须能失去信心。Reacquisition 是事件，不是
永久状态；`TRACKING` 之外不得断言当前 candidate。Active/Temporal Grounding 是独立未来模块，不能塞进
P1-R0 越权解决 `AMBIGUOUS / NO_REFERENT`。

本轮只冻结 JSON schemas、deterministic evaluator、八个 synthetic mechanics fixtures 和固定阈值 simple
baseline。它没有真实 RGB tracker、模型、Sky 或 fresh cohort，因此不建立 P1 科学性能。

## P0 policy-discovery 终点

P0 named-building entrance grounding 的冻结合同见 [`Protocol V1`](P0_GROUNDING_PROTOCOL_V1.md) /
[`JSON`](p0_grounding_protocol_v1.json)。它把 `UNIQUE / SET_VALUED / AMBIGUOUS`、Provider availability、
Brain selection、End-to-End outcome 与 P1 source-frame handoff 分开。

P0 Development 先后建立 map+geometry Silver-B、Terra baseline、parent-disjoint failure replication 与
calibration frontdoor。主要证据入口：

- [`Prior art`](P0_PRIOR_ART_ASSIMILATION_2026-08-21.md)：BridgeNav 与 ABot-N1/POIBench 已覆盖入口级
  POI navigation 的关键部分；BlindAssist 不主张任务首创。
- [`Silver-B contract`](P0_SILVER_B_DEVELOPMENT_ADDENDUM_V1.md) /
  [`result`](P0_SILVER_B_BRAIN_DEVELOPMENT_RESULT_2026-08-21.md)：只支持 conditioned Development mechanics，
  不授权 detector recall、exact Brain/E2E accuracy 或 Silver-A 等价。
- [`P0-D1 consumed`](P0_D1_AMBIGUITY_CALIBRATION_CONSUMED_CANARY_RESULT_2026-08-21.md) /
  [`parent-disjoint`](P0_D1_PARENT_DISJOINT_CONFIRMATION_RESULT_2026-08-21.md)：place/entrance calibration 降低
  unsupported commitment 时严重 over-refusal，机制跨 venue 复现。
- [`P0-D2`](P0_D2_RESOLVABLE_ENRICHMENT_AND_FRONTDOOR_RESULT_2026-08-21.md) /
  [`P0-D3`](P0_D3_ONE_SHOT_CLOSURE_RESULT_2026-08-21.md)：public source 的 SET_VALUED denominator 不足；
  Logistic/Conformal 未授权，禁止第二数据批。

[`P0-A1`](P0_A1_AMBIGUITY_GATE_DISCOVERY_RESULT_2026-08-21.md) 在统一可观察的 consumed Development surface
上保留 `brain confidence >= 0.85 AND candidate center dispersion <= 0.2423407461`：ambiguous false commit
为 `11/51`、parent macro `19.61%`，同时保持 `20/20` resolvable coverage 与 `17/20` correctness。

[`P0-A2`](P0_A2_COMPACT_AMBIGUITY_POLICY_DISCOVERY_RESULT_2026-08-21.md) 随后确定性枚举 518,570 个 compact
symbolic policies。3,237 个满足硬约束的 unique behaviors 中，最优仍是 A1，parent-macro 增益 `0.00pp`；
relaxed winner 只有 `65%` coverage。终态：

```text
COMPLEXITY_ONLY_BUYS_ABSTENTION
A1_INCUMBENT_RETAINED
NO_POLICY_ADMISSION
NO_SCIENTIFIC_VERDICT
```

P0 commitment-policy discovery 至此关闭：不运行 A3+ threshold/classifier/XGBoost/Sky，不补 SET_VALUED，
不自动购买 fresh confirmation。A1 只保留为 Development incumbent。

## P1 的既有 Development 基础

系统侧只允许 RGB；ADT bbox、object/device trajectory、depth、segmentation 与 visibility GT 只进入隔离的
mining/evaluator。ADT 是 prerecorded trajectory，不能证明 guidance 改变用户动作或 closed-loop navigation。
稳定实现入口见 [`BA-ADT scripts`](../../../scripts/research/ba_adt_real_evidence/README.md)。

既有 consumed Development 只作为 P1 failure vocabulary 与未来 adapter 输入：

[`P1-D0 temporal cohort`](P1_D0_TEMPORAL_COHORT_PROTOCOL_V1.md) 已从 2 条现有 ADT GT/RGB source 以 source
timestamp 和 `object_uid` 自动物化 15 episodes / 15 physical targets / 1,724 frames；六类 temporal mode
均非零，0 model/detector/tracker calls。它只是 consumed indoor-object Development truth，不验证入口
persistence；数据线已止损关闭，EgoTracks fallback 不触发。

- [`ADT0/ADT1 canary`](BA_ADT_REAL_EVIDENCE_ADT0_SELECTION_ADT1_CANARY_RESULT.md)：detector+5-frame flow
  建立 bounded persistence mechanics，但最长 dropout 仍为 162。
- [`Instance redetection R1`](BA_ADT_INSTANCE_REDETECTION_1_RESULT_2026-08-21.md)：13 次确认均正确、0 wrong，
  但长时重捕获仍弱；5 个失败全部为 `NO_CANDIDATE`。
- [`R3 observability`](BA_ADT_REAPPEARANCE_OBSERVABILITY_R3_RESULT_2026-08-21.md)：失败窗同时含不可见/重遮挡
  与 tiny target，不允许把所有失败归因 identity verifier。
- [`R4 scale`](BA_ADT_SMALL_TARGET_SEARCH_SCALE_R4_RESULT_2026-08-21.md)：已测试的 scale arms 未建立 fixed-window
  correct proposal，并产生 wrong-instance 风险。
- [`R5 teacher closure`](BA_ADT_SMALL_TARGET_VISUAL_UPPER_BOUND_R5_RESULT_2026-08-21.md)：DINOv-SwinL 只命中
  `1/3 windows / 1/97 frames`；SAM 3.1 cross-image arm 为 `NOT_EVALUABLE_INTERFACE`，R5 inconclusive 并永久关闭。

这些数字保持 consumed Development claim ceiling；P1-R0 不重跑或改写旧 evaluator、TargetMemory、flow、
confirmation、quarantine、receipt 或终态。

## Legacy GC/Sky 边界

[`GOAL-COPILOT-1`](GOAL_COPILOT_1_SKY_PILOT_RESULT.md) 与
[`GC2-B`](GOAL_COPILOT_2B_RESULT.md) 已封存；[`observability audit`](GOAL_COPILOT_2_OBSERVABILITY_AUDIT_RESULT.md)
已选择停止 synthetic moderate optimization。Sky 不属于当前 pipeline，不能用于 P0 policy rescue、P1 tracker
搜索或绕过真实 observation failure。

## 唯一 successor

`P1_R0_CONSUMED_ADT_BASELINE_ADAPTER_DESIGN`：只设计如何把 P1-D0 RGB/P0 handoff 接入 P1 public-input
schema，并让 episode truth 只进入 evaluator。设计完成前不运行新的 ADT baseline evaluation。

禁止：Sky、模型搜索、fresh/large cohort、P0 policy 续搜、Active/Temporal Grounding、Android/default-App、
产品或安全主张。允许：stdlib mechanics 修复、schema/evaluator 专项测试、只读检查既有 consumed ADT interface。

Claim ceiling：`SCHEMA_EVALUATOR_AND_SYNTHETIC_BASELINE_MECHANICS_ONLY_NO_VISUAL_PERSISTENCE_OR_SCIENTIFIC_VERDICT`。
默认 App：不变。
