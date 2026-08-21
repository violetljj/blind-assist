# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / P0_POLICY_DISCOVERY_CLOSED / P1_A2_DENSE_IDENTITY_SIGNAL_RETAINED / P1_A3_TEMPORAL_POLICY_INSUFFICIENT / NO_POLICY_ADMISSION / P1_A4_ONLINE_CORRESPONDENCE_PROTOCOL_FROZEN / IMPLEMENTATION_SELECTION_NOT_RUN / NO_SCIENTIFIC_VERDICT / NO_SKY / DEFAULT_APP_UNCHANGED`

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

JSON schemas、deterministic evaluator、八个 synthetic mechanics fixtures 与 P1-D0 truth firewall 保持冻结。
[`P1-R0 consumed ADT baseline`](P1_R0_CONSUMED_ADT_BASELINE_RESULT_2026-08-21.md) 已运行一个只读 RGB 的
sparse-flow + fixed-template baseline；它只建立 consumed Development failure structure，不建立科学、产品或 safety 性能。

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

## 当前 P1 结果与唯一 successor

[`P1-A1`](P1_A1_CONSERVATIVE_LOCAL_TRACK_VALIDITY_RESULT_2026-08-21.md) 已一次性完成 3,069 个 compact
RGB-only validity gates。最佳 `>=90%` retention gate 保留 `80/87` correct，但 wrong、episode-macro、max wrong-lock
只下降 `39.64% / 44.73% / 9.41%`；0 wrong gate 只保留 `15/87` correct，false-loss 达 `94.21%`。终态
`VALIDITY_GAIN_ONLY_BY_ABSTENTION / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`，不保留 discovered threshold。

[`P1-A2`](P1_A2_FIXED_REFERENCE_DENSE_IDENTITY_RESULT_2026-08-21.md) 随后以 frozen DINOv2-S initial patch
memory 和 dense correspondence consensus 一次性检查 625 个四特征 AND policy。4 个通过预冻结 admission；
top policy 保留 `80/87` correct，把 wrong `1,221→445`、max wrong-lock `8,498→2,700 ms`，分别改善
`63.55% / 68.23%`，终态 `DENSE_IDENTITY_VALIDITY_SIGNAL_ESTABLISHED / NO_POLICY_ADMISSION`。

边界同样必须保留：14 个 drift episode 没有正 warning lead；false-loss 为 `304/777`，frame-wise gate churn
产生 29 个 evaluator-defined false reacquisition。它只建立 consumed Development representation signal，不能保留
threshold 或接 App。

[`P1-A3`](P1_A3_TEMPORAL_LOSS_DECLARATION_RESULT_2026-08-21.md) 未继承 A2 threshold，只复用 raw dense
evidence，一次性比较 consecutive/sliding/leaky 共 40 个 temporal policies。全部 policy 均保留 correct、消灭
false reacquisition/chatter 并维持 long-loss declaration，但 `wrong<=488` 与 `false-loss<=152` 均为 `0/40`。
代表 policy 为 `correct=81 / wrong=685 / max-lock=2,899 ms / false-loss=205 / false-reacquisition=0`；终态
`TEMPORAL_POLICY_INSUFFICIENT / NO_POLICY_ADMISSION`。这关闭简单 temporal smoothing rescue，不改写 A2 的历史
frame-wise representation signal。

[`P1-A4 protocol`](P1_A4_ONLINE_STRONG_TEMPORAL_CORRESPONDENCE_PROTOCOL_V1.md) 已先于 implementation selection
冻结 strictly causal point-correspondence、25-point initialization、visibility-aware object aggregation、单模型选择顺序
与 capability gates。下一步只做 outcome-blind official interface/checkpoint/license 和本机 smoke；第一候选通过即固定，
不运行模型竞赛，也不读取 private truth。

禁止：继续调 A1/A2/A3 threshold/operator、直接保留 winner、提前实现 stronger tracker 或 global
reacquisition、加入 ReID/Sky、fresh/large cohort、
P0 policy 续搜、Active/Temporal Grounding、Android/default-App、产品或安全主张。

Claim ceiling：`CONSUMED_ADT_INDOOR_OBJECT_DEVELOPMENT_BASELINE_ONLY_NO_SCIENTIFIC_VERDICT`。
默认 App：不变。
