# P0-D2 Commitment Calibration protocol V1

状态：`PROTOCOL_V1_FROZEN / DATA_FRONTDOOR_INSUFFICIENT / FIT_NOT_AUTHORIZED / NO_V3_PROMPT / NO_SKY / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 研究问题

冻结 Terra、Grounding DINO candidate pool、原始 candidate order 与 evaluator 后，一个独立的小型
evidence-authority calibrator 能否显著减少 `AMBIGUOUS` venue parents 上的 unsupported commitment，同时保留
可解析 `UNIQUE / SET_VALUED` parents 上的正确 grounding？

Calibrator 不重新生成或重排 candidate，不修改 prompt，不读取 evaluator truth。原始 Brain 已经 `ABSTAIN` 时保持
`ABSTAIN`；原始 Brain 给出一个或多个选择时，calibrator 只可保留为 `COMMIT / SET`，或降级为
`AMBIGUOUS / ABSTAIN`。

## 输出语义

两层输出不可折叠：

```text
evidence authority = RESOLVABLE
  + singleton referent set      -> COMMIT
  + multiple legal referents    -> SET

evidence authority = REFERENT_AMBIGUOUS -> AMBIGUOUS
evidence authority = INSUFFICIENT       -> ABSTAIN
```

多 candidate conformal set 不能自动解释为 `SET_VALUED`；它也可能只是无法消除竞争入口。空集同样不能自动解释为
`ABSTAIN`。`core.calibrated_action` 对这条边界 fail closed。

## 数据角色与独立单位

- 旧 47-goal cohort、D1 Brussels 24-goal confirmation 与本轮 enrichment 均为 consumed Development；
- venue parent 是 fit、calibration、conformal quantile 与报告的独立单位，frame 数不得代替 parent 数；
- 当前 fit 前门：至少 8 个 `UNIQUE` parents、4 个 `SET_VALUED` parents、12 个 `AMBIGUOUS` parents，且
  resolvable parent 并集至少 12；
- calibration/adjudication 必须另按 venue parent 隔离。任何 candidate lock 后的 adjudication cohort 不得返回调参；
- 若未来宣称 split-conformal `alpha=0.10`，每个实际承诺 coverage 的 calibration stratum 至少需要 9 个独立
  parent scores；相关性多帧必须先做 parent reducer。

## 运行时 feature surface

唯一输入合同为 [`p0_d2_runtime_feature_schema_v1.json`](p0_d2_runtime_feature_schema_v1.json)。允许四类 runtime
evidence：`PLACE_IDENTITY / ENTRANCE_RELATION / CANDIDATE_COMPETITION / BRAIN_RANK_MARGIN`。人工 resolution、
acceptable regions、valid targets、end-to-end outcome、reviewer 字段及由它们派生的标签只能进入隔离训练 sidecar，
不得进入 runtime row。缺失 evidence 保留为 missing/null，不填 0 或 STRONG/WEAK 自评。

## 两个首轮 arms

1. `SELECTIVE_LOGISTIC_V1`：L2-regularized logistic regression；只估计冻结 Terra selection 是否具有足够
   commitment authority。阈值只由 calibration parents 决定。
2. `CONFORMAL_REFERENT_SET_V1`：使用同一冻结 candidate score surface，在 parent-level calibration scores 上形成
   referent set；authority state 与 legal referent membership 分开校准。

只有 logistic 暴露明确、可重复的非线性替代关系缺口后，才允许预注册 shallow GBDT；MLP、VLM replacement、
prompt V3 与 Sky 均不在 V1。

## 评价与停止门

- primary：`AMBIGUOUS` venue-parent macro unsupported-commit rate；
- protection：`UNIQUE` correct-commit retention；
- protection：`SET_VALUED` legal-set coverage；
- guardrail：整体 refusal、resolvable refusal、frame micro 与 parent macro 同时报；
- 不允许以 refusal 上升换取 admission；
- 数据前门或 feature completeness 未通过时，终态必须是 `FIT_NOT_AUTHORIZED`，不得拟合后再解释样本不足。

Claim ceiling：`CONSUMED_DEVELOPMENT_PROTOCOL_AND_DATA_SUFFICIENCY_ONLY_NO_CALIBRATION_OR_MODEL_PERFORMANCE_CLAIM`。
