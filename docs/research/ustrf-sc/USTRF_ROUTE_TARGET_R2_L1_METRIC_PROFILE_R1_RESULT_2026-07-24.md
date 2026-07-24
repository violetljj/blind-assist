# USTRF route-target R2-L1 metric profile R1 结果（2026-07-24）

## 结论

本轮已从冻结的 C1–C3 `123/123` 条权威 trace 构造三个逐指标探索 profile，终态为 `METRIC_PROFILES_COMPLETE / VALID`。评分前重新核对了 `3 × 41` 条 trace、`186,687` 个 candidate-frame、每候选 `15` 个 discontinuity reset，以及全部 trace/authoritative-receipt SHA 和逐帧四元 identity；候选重跑、新权威 trace 和新数据均为 `0`。

三个 profile 都支持同一个研究判断：当前候选已能在 8 个 eligible critical interval 上维持 truth-attributed active relation，观察 critical miss 均为 `0/8`；但 `n=8` 远低于零 miss 风险边界所需的 `59`，只能记为 `estimate_only / bound_sufficient=false`。与此同时，每个 profile 都观察到 unknown/stale route 上的 active alert 硬 veto，clearance 点估计也低于冻结的 `.90` 门，因此本轮不产生候选比较、排名、胜者或 L2 selection。

这仍是 proxy/model-truth + replay 的研究证据，不是人体效果、独立行走安全或生产授权。

## 逐候选 profile

下表只是按冻结候选 ID 分别陈列同口径结果，不构成横向评分或选择。

| 候选 | critical miss | clearance | unknown/stale active alert | repeat | evidence age |
| --- | --- | --- | --- | --- | --- |
| `C1_CAUSAL_ROUTE_RELATION_FSM` | `0/8`；point pass，bound 不足 | `0/12`；point fail | `12,621/62,229`；hard veto | `2/3` events；underpowered，observed failure | `0/62,229` consume timestamp；`not_evaluable` |
| `C2_ROUTE_OCCUPANCY_EPISODE_FSM` | `0/8`；point pass，bound 不足 | `1/12`；point fail；唯一成功 delay `66.484ms` | `7,165/62,229`；hard veto | `1/2` events；underpowered，observed failure | `0/62,229` consume timestamp；`not_evaluable` |
| `C3_DUAL_KEY_CLEARANCE_FSM` | `0/8`；point pass，bound 不足 | `0/12`；point fail | `12,759/62,229`；hard veto | `0/1` event；underpowered，不能判通过 | `0/62,229` consume timestamp；`not_evaluable` |

Wilson 双侧 95% 区间、逐来源 numerator/denominator、来源族贡献和完整 mask/trace 绑定保存在各 candidate profile 中。`critical_miss` 的 `0/8` Wilson 上界约为 `.3244`，不能解释为风险已被充分约束。

## L0 诊断量

`event_recall`、`regeneration` 和 `false_alerts_per_minute` 仍严格保持 L0，不输出 pass/fail：

- event recall 没有冻结 alertable deadline，分母仍为 `0`；
- regeneration 没有完整 post-clear 同人身份区间，分母仍为 `0`；
- false-alert exposure 固定为 `297,376,110,945ns = 4.956268516min`，低于 5 分钟 L1 floor。三个 profile 的 raw diagnostic delivery 分别为 `218`、`57`、`8`，不得作为可比较的 L1 rate 或 selection 输入。

## 评分合同修正

新入口没有修改旧 exploratory runner 或已哈希绑定的 replay 实现。它只读取 A2 inventory 与 A3/A4 completion evidence，并显式修复旧评分路径不适用于本轮的四个问题：

1. critical miss 使用 mask 冻结的精确 critical interval，并检查同帧 truth-attributed active relation，而不是要求 interval 内重新 delivery；
2. 同帧多 delivery 按 delivery track 独立归因；单 episode key 的多 track 作为一个 episode group，不共享跨 delivery 的 event union；
3. closure key 附加 `source + sequence + reset_segment` 作用域，禁止跨 discontinuity 复用数值 key；
4. clearance 使用 source capture timestamp 和冻结 `1500ms` horizon；evidence age 缺少 consume timestamp 时整项 fail closed，不用 pose age 或 scorer wall time 替代。

## 收据与验证

- 配置：`configs/ustrf_route_target_r2_l1_metric_profile_r1.json`
- config SHA-256：`c922a8c88a92794ab17485d00830cc91d31e2f1985e263692c5fb0645add0012`
- terminal SHA-256：`dc2fd06fd0e64ee55b4f8f20475d0fa9f89f54cf8d3a82e5ed5a6b5b6a2d6f6b`
- profile SHA-256：
  - C1：`543f5af429ce5615f97bcce96f226fc96d88b41fa814161718c3918f18097c0f`
  - C2：`42ab5181597311347d2ca6718d2f98c4e090413db1a92c969f15ac1d84205c9a`
  - C3：`15b2125b5d53164500851803bbddc0ba77ea45e0890453191c35e584325455f3`
- focused tests：`7 tests OK`
- 独立重算 validator：`VALID`，复核 `123` trace / `186,687` candidate-frame
- 本地机器证据：`artifacts.local/evidence/ustrf-route-target-r2-l1-metric-profile-r1/`（忽略目录，不是仓库真源）

## 权限与下一独立边界

本轮到此停止：没有总分、比较、排名、胜者、阈值/分母调整，也没有 L2/L3、Android shadow、H2、人体、独立行走或生产权限。

下一独立边界不应进入 selection，也不应继续扩数据。建议预注册一个单变量的 `route-invalid fail-closed + reset-scoped lifecycle` 机制诊断：先证明 unknown/stale route 时 active state 能立即关闭，再证明 truth clear 后 episode key 能在 `1500ms` 内闭合；仍使用冻结数据作机制回归，另设全新候选输出 namespace，不追溯改写本轮 profile。`candidate_consume_timestamp_ns` 的 trace 合同补齐应作为独立工程观测项，不能用来回救上述机制失败。
