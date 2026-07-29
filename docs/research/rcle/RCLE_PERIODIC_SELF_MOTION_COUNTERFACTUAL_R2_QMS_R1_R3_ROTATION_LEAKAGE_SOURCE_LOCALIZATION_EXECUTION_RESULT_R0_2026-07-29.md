# RCLE periodic-self-motion R2：R3 rotation-leakage source localization 执行结果 R0

结论先行：一次性 formal runner 已完成全部 `8/8` 个冻结 rotation-only
clusters，但强制独立 validator 在完成 `2/8` 个 cluster 后 fail-closed。
因此本轮没有经过独立验证的 cluster route，也没有 first-visible-layer、multiple
或 not-evaluable 的科学结论。

| 状态轴 | 终态 |
| --- | --- |
| scientific status | `NO_VALIDATED_SCIENTIFIC_RESULT / NOT_EVALUABLE_DUE_TO_EXECUTION` |
| protocol status | `INDEPENDENT_VALIDATION_INVALID` |
| execution status | `RUNNER_COMPLETE_8_OF_8 / VALIDATOR_FAILED_AFTER_2_OF_8` |
| execution authority | `ONE_SHOT_CONSUMED / NO_RERUN / FUTURE_REPAIR_NOT_AUTHORIZED` |

本终态不等于 R3 成功或失败。它只说明：producer 数据已经完整生成，但预注册的
独立验证链没有完成，因而这些 producer 输出仍停留在
`PENDING_INDEPENDENT_VALIDATION`，不能用于算法修改或后继路由。

## Formal runner

- output root：
  `artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/`
  `qms_r1_r3_rotation_leakage_source_localization_r0`；
- `8` 个 cluster、`8` 条 sequence，每条固定 `601` pairs；
- 4 workers，wall time `1,344.616293 s`；
- launch available RAM `8,876,089,344 bytes`；
- coordinator-observed minimum available RAM `6,226,071,552 bytes`，高于在途
  `4,294,967,296 bytes` floor；
- swap-in / swap-out delta 均为 `0`，residual worker 为 `0`；
- R3、strict `>0.01/s`、三 pair、`PairState`、abstention 和输入 identity
  均未修改；
- formal `480+16` 运行数仍为 `0`；
- runner guard 终态为 `COMPLETE`，progress contract 有效。

Runner 的 `success.json` 只签署
`LOCALIZATION_EXECUTION_COMPLETE / INDEPENDENT_VALIDATION_REQUIRED`；八个
cluster receipt 的 route 均为 `PENDING_INDEPENDENT_VALIDATION`。

## 独立验证终止

独立 validator 的唯一一次正式尝试在第三个 cluster
`B_TDO_OA_R0_ADVIO_14_S1` 终止：

```text
PAIR:519:LOCAL_CELL_EXPANSION:COMPENSATED_FINAL:6
```

冻结终态为：

```text
INDEPENDENT_VALIDATION_INVALID / ONE_SHOT_CONSUMED / NO_RERUN
```

失败时 `completed_units=2/8`、`sample_index=112`。以下三个必须存在的结果文件均
没有生成：

- `analysis_result.json`；
- `independent_validation_receipt.json`；
- `execution_decision.json`。

因此不能从前两个已通过的 cluster、producer ledger 或尚未完成的验证过程拼接局部
scientific route，也不能把 `2/8` 当成抽样结果。分析单位始终是 cluster；pair、
frame、track 和 cell 都是纵向重复或审计单元，不是独立样本。
该错误只是 validator 按固定顺序遇到的首个失败；后续 6 个 cluster 未完成，不能
声称这是全量证据中的唯一失败。

## 有界数值事后诊断

在不导入、不调用、不恢复独立 validator，也不创建任何科学输出的前提下，只对报错
cell 做了一次只读重算：

- cluster：`B_TDO_OA_R0_ADVIO_14_S1`；
- pair：`519`；
- path / cell：`COMPENSATED_FINAL / 6`；
- 该 cell 原本已经 `evaluable=false`；
- abstention：`LK_TRACK_SUPPORT_BELOW_12`；
- consensus support：`3`；
- tracked support：`5`；
- pair 同样 `evaluable=false`、common cells `3`，原因为
  `COMMON_GRID_SUPPORT_BELOW_5_OF_9`；
- ledger OLS expansion：`-166.74857193180273/s`；
- 从 sealed float32 consensus tracks 重算：
  `-166.7485719317925/s`；
- absolute difference：`1.0231815394945443e-11/s`；
- validator absolute tolerance：`1e-12/s`；
- difference / tolerance：`10.231815394945443`；
- relative difference：`6.136073776469929e-14`；
- coefficient max absolute difference：`2.2118911147117615e-09`；
- design rank：`3`；
- condition number：`6.150909842569977`。

这说明触发点是一个 audit-only 大幅值 OLS 数的跨序列化纯绝对误差比较，而且该
cell 已因 support `<12` 弃权。它与既有 numeric-representation amendment 所规定的
“审计系数检查有限性、形状和公式自洽，科学 expansion/evaluability 仍独立严格
重建”的边界不一致。

这个诊断支持把失败分类为
`PROTOCOL_NUMERIC_REPRESENTATION_DEFECT`，但不能事后把本轮 validator 改成
`VALID`，也不能产生任何 R3 scientific outcome。当前
`INDEPENDENT_VALIDATION_INVALID` 永久保留。

## Guard wrapper 的独立报告问题

Validator 进程本身以 exit code `2` 写出了明确的
`validation_failure.json` 和 `status=failed` 的 progress。外层 guarded runner
同时把 progress 判为
`last_progress_at predates this runner invocation`，并给出
`PROGRESS_CONTRACT_VIOLATION`。

该包装层时间比较异常是另一个非科学报告缺陷；它不覆盖 validator 的 fail-closed
failure，也不改变 no-rerun 终态。本轮不修复 guarded runner，避免把非阻断维护与
已消费的科学协议混在一起。

## 权限与下一研究边界

- runner retry：`NOT_AUTHORIZED`；
- validator retry：`NOT_AUTHORIZED`；
- replacement / reseed / resume / output delete：`NOT_AUTHORIZED`；
- 当前 validator 或结果的事后 repair：`NOT_AUTHORIZED`；
- R3、阈值、三 pair、abstention 修改：`NOT_AUTHORIZED`；
- single targeted upgrade、feature contract C、fusion experiment D：
  `NOT_AUTHORIZED / CLOSED / CLOSED`；
- formal `480+16`：`NOT_CONSUMED / NOT_RUN`；
- Android、主动告警、产品、危险或安全结论：`NOT_AUTHORIZED`。

若继续推进，只能另立一个明确授权、独立命名和 fail-closed 的薄 successor
protocol，用于未来证据；它必须保留本次 `INVALID`，不能覆盖当前 output root，
不能把现有 producer 输出追认为已验证结果，也不能自动授权算法修改或
`480+16`。

机器关闭记录：
[execution closeout R0](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_EXECUTION_CLOSEOUT_R0_2026-07-29.json)。
