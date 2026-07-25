# USTRF Looming R1 非权威执行隔离结果（2026-07-25）

状态：`NONAUTHORITATIVE_EVALUATION_QUARANTINED / VALID`

权威终态：

```text
R1_CLAIM_SCOPED_SOURCE_PROGRAM_NONAUTHORITATIVE_EVALUATION_QUARANTINED_INPUT_AUTHORITY_BLOCKED
```

## 一、结论

当前不能说 Looming 已失败，也不能说它已通过。

正式预注册的 Bonn `3 × 3 grid / 500ms anchor` static-surface truth ledger
在 signal 执行前已终止于：

```text
BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED
```

冻结要求是至少 `4` 个可用 registered-depth canary，实际只有 `3` 个。该门没有
事后降到 3，也没有换样、开 validation/holdout 或把另一个 unit definition
升级为确认真值。因此 Bonn C2 没有获得本轮 truth-join authority。

随后实际生成的 base/oracle traces 和一次连续评分全部保留，但评分依赖的
full-frame/central-ROI truth ledger 已被独立审计为 `diagnostic-only`。下游评分
不能反向升级其 truth authority；其自报的 stop terminal 不具有停止或接受权限。

## 二、实际执行事实

| 项目 | 实际数量 |
| --- | ---: |
| discovery RGB 解码 | `598` |
| frozen consecutive pairs | `596` |
| base flow traces | `596` |
| oracle rotation traces | `594` |
| full-6DoF diagnostic traces | `594` |
| diagnostic truth-joined pairs | `503` |
| validation/holdout reads | `0` |
| 旧 15 对窗口 selection/tuning/acceptance reads | `0` |
| 报警阈值 | `0` |

Base producer 未读取 pose、depth、truth、cell 或旧 outcome；base trace 哈希冻结后，
oracle namespace 才读取 orientation/full pose 与 source depth。一次后续 truth
join 确实发生，不能再写成 `truth_join_or_scoring_run=false`，但它被显式隔离，
`authoritative_algorithm_result_available=false`。

## 三、仅作风险诊断的数值

使用非权威 central-ROI truth ledger 时：

| 指标 | 结果 |
| --- | ---: |
| oracle 等权 session Spearman | `0.0704` |
| session-block 95% 区间 | `[-0.0343, 0.1751]` |
| uncompensated 等权 session Spearman | `-0.0441` |
| oracle 相对 uncompensated 增量 | `+0.1145` |
| `person_tracking2` oracle Spearman | `0.1751` |
| `balloon` oracle Spearman | `-0.0343` |

这组数值是明显风险信号：相对补偿有改善，但绝对关联弱、最差 session 方向不稳。
它只能用于未来预注册设计和风险判断，不能写成 `worst-source failure`、算法 stop、
Bonn C2 confirmation 或产品结论。

## 四、为什么不能用它判失败

1. 评分 truth 的 unit definition 是 full-frame/central ROI，而正式合同是
   `3 × 3 grid / 500ms anchors`；
2. 该 ledger 在看到 transform outcome 后才物化，没有独立的 pre-outcome
   ledger-specific freeze；
3. 它把 map projection 标为 A 级，而正式 R1 对独立重建和不确定度使用 B 级；
4. 正式四帧 depth canary quorum 已失败，不能被另一条派生 ledger 绕过；
5. source program 在评分发生前已明确关闭本轮 truth join。

## 五、下一边界

本轮 Bonn 不重跑、不降门、不换 discovery 样本、不打开 validation/holdout。

下一项有权改变主结论的工作是：

1. 完成无人体受控装置的 camera、full-pose/linear truth、同步、外参与安全
   hardware/calibration manifest；
2. 获取三个未来独立刚性目标 discovery cluster；
3. 若仍需 Bonn 作为第二个 C2 family，另立版本化、pose-join-aware 的
   truth-authority recovery 合同；新合同必须在读取任何新 depth/RGB outcome 前冻结。

在这之前，R1-B、部署型 rotation estimator、报警阈值、App、route、lifecycle、
人体、安全和生产全部关闭。

## 六、机器证据

权威隔离 receipt：

```text
artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/
bonn_nonauthoritative_continuous_signal_evaluation_review_r0.json
SHA-256 7343ec0d51ff31855cc0d02a1fda4ee315d1c3aa0ed3057b8cc5bf166bc3efaa
```

非权威评分 receipt：

```text
artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/
bonn_r1a_continuous_signal_evaluation_r0.json
SHA-256 054535ae5e83a416424c4b067f5dd8f2c521ca003ae349cd22d4b2ae7c90cc5b
```
