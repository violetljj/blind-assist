# USTRF route-invalid + reset-scoped lifecycle 机制诊断 R1（2026-07-24）

## 结论

本轮严格限定为单变量的 `route-invalid fail-closed + reset-scoped lifecycle` 机制诊断，终态为 `MECHANISM_DIAGNOSTIC_COMPLETE / VALID`，但总体机制门为 **false**。

结果分成两层：

1. route-invalid 与 reset 机制本身通过：三个候选在 unknown/stale route 上的 guarded active 帧都从基线的 `12,621 / 7,165 / 12,759` 降为 `0`；`1,235 / 801 / 1,238` 个 known→invalid 且基线仍 active 的转换均在同一帧 fail closed。每候选 41 个 sequence scope + 15 个 discontinuity reset 均无跨 reset active key，重复 scoped key 为 0。
2. relation-based truth clearance 仍失败：C1/C2/C3 分别只有 `0/12`、`1/12`、`0/12` 在同一 reset-scoped episode key 上于 truth clear 后 `1500ms` 内闭合。route-invalid 与 reset terminalization 均没有被记入 clearance numerator。因此本轮只证明 active latch 可被安全截断，不证明 lifecycle clearance 已修好，更不产生候选胜者或晋级权限。

这仍是冻结 proxy/model truth + replay 的机制证据，不是人体效果、独立行走安全或生产授权。

## 单变量与非目标

guard 只包裹已冻结的 C1–C3 权威输出：

- route invalid 时，同帧关闭全部 guarded active episode，并在出现全新 baseline delivery 前保持 quarantine；
- discontinuity reset 时，先以 `reset_scope_end` 终止旧 scope，再处理新 scope；
- episode key 固定为 `source + sequence + reset segment + local key + activation ordinal`；
- known-route 上的 baseline delivery/closure 输入、route ledger、detector、T0 association、`min_alert_frames`、`min_clear_frames`、阈值、truth 与分母均不变；
- guard 构造期间不可读取 truth；truth 只在 123 条新 lifecycle trace 全部构造后加入 clearance 诊断。

本轮没有重跑 detector 或 C1–C3，没有改写旧 123 条 trace，也没有补造 `candidate_consume_timestamp_ns`。总分、候选比较、排名、selection、L2/L3、Android shadow、H2、人体、独立行走与生产权限全部关闭。

## route-invalid 与 reset 结果

| 候选 | 基线 invalid active 帧 | known→invalid active 转换 | guarded invalid active 帧 | route-invalid terminalization | reset 前 guarded active 帧 | 跨 reset key |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 12,621 | 1,235 | 0 | 3,816 个 active key | 13 | 0 |
| C2 | 7,165 | 801 | 0 | 222 个 active key | 5 | 0 |
| C3 | 12,759 | 1,238 | 0 | 45 个 active key | 10 | 0 |

route-invalid terminalization 数量是被同帧终止的 scoped active key 数，不是 truth-clear 事件数，也不能跨候选比较。C1 可同时保有多个逐人 key，因此该数字明显更大。

每候选均验证 `62,229` 帧、41 个 sequence scope start 与 15 个 discontinuity reset，共生成 `123` 条 guarded lifecycle trace、`186,687` 帧。所有 active key 都留在创建它的 reset segment；同一 local key 在同一 segment 重新激活时使用新的 activation ordinal，不复用旧 episode identity。

## truth-clear 机制诊断

clearance 继续沿用冻结的 12-event 分母、同一 scoped key、source capture timestamp 与 `1500ms` horizon。只有 `baseline_known_route_closure` 可以进入 numerator；`route_invalid` 和 `reset_scope_end` 即使停止了 active，也一律不能冒充人物真实 clear。

| 候选 | 1500ms 内闭合 | 无 guarded eligible delivery | 无同 scope 的 post-truth-clear relation closure | 结论 |
| --- | ---: | ---: | ---: | --- |
| C1 | 0/12 | 9 | 3 | fail |
| C2 | 1/12 | 10 | 1 | fail；唯一成功 delay `66.483968ms` |
| C3 | 0/12 | 11 | 1 | fail |

因此真正剩余的机制缺口仍在 known-route 的 eligible delivery 与 relation-based closure，而不是 route-invalid latch 或 reset key 串接。本轮没有继续拆解 no-delivery、提前闭合、lineage release 或其他下一变量。

## 收据与验证

- 配置：`configs/ustrf_route_target_route_invalid_reset_lifecycle_diagnostic_r1.json`
- config SHA-256：`9b71ab8d8cb95494eb5e54b76559905609bd28c7690f35100c8ba724ee351a29`
- terminal SHA-256：`27d2c9a74d6e5cfc5b848197302aff20db2d996b501872f56e716758e1bf380b`
- validation SHA-256：`67e19ac27d952b42f6ffdde6bba425833feed526d92c1f9527b5723314ff2f7f`
- focused tests：`5 tests OK`
- 独立 validator：`VALID`；重新核对 123 条父 trace / 186,687 帧，并重算 123 条 guarded trace / 186,687 帧
- canonical 本地证据：`artifacts.local/evidence/ustrf-route-invalid-reset-lifecycle-diagnostic-r1/`

首次 preflight 因把父 A2 terminal 名称误写为旧阶段终态而在任何输出前 fail closed；只把身份检查修为权威收据中的 `CANDIDATE_REPLAY_COMPLETE`。随后依次补齐诊断计数、稳定根 Adapter、精确 config/authority/scope/output 合同、truth 输入拒绝、consume-timestamp 实际扫描和“先构造全部 123 条 guarded trace、后加载 truth”的两阶段顺序；较早的六份本地 materialization 完整保留在同级 `attempt-001-superseded` 至 `attempt-006-superseded` 目录，不具有当前 config/terminal 权限。

## 停止边界

本轮在 `route-invalid gate PASS + reset-scope gate PASS + clearance gate FAIL` 处停止。不能因为 unknown/stale active 已归零就进入 selection，也不能用 consume timestamp 工程补齐、阈值调整或 route-invalid/reset terminalization 回救 clearance。下一独立变量需另行冻结后才能启动。
