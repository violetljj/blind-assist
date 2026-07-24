# USTRF candidate-independent causal token policy/risk gate R1 结果（2026-07-24）

## 结论

本轮先冻结并执行 `CANDIDATE-INDEPENDENT-CAUSAL-TOKEN-POLICY-RISK-GATE-R1`，终态为 **`POLICY_COVERAGE_REJECT / VALID`**。

冻结 policy 把 R0 的 reset 持久 token 改为有限、因果、fail-closed 的 lease：

- 同一 track 必须在 route known 下连续 active relation 至少 `2` 帧且持续 `500ms`；
- token 从资格帧起最长只有效 `500ms`；
- reset、route unknown、track unobserved、relation gap 或 TTL 到期立即失效；
- 同一 track/reset 只允许首枚 token，之后所有再资格化只记账并抑制；
- producer 只读 track/route/reset/causal timestamp；41 条 policy ledger 全部冻结并 hash inventory 后，第二进程才读取 oracle 与负暴露。

该 policy 同时失败于 coverage 与风险，不能连接 isolated opener：

- 33 个既有 supported candidate-event cell 中，仅 `9/33` 在 token 有效期内覆盖；另 3 个无 active relation cell 仍继续 fail closed。冻结 coverage 门要求 `33/33 + 3/3`，因此直接 `REJECT`；
- 完整 62,229 帧共生成 `1,448` 枚 token，仅 3 枚 post-hoc 匹配 supported oracle，`1,445` 枚为 extra；
- 836 个半开负暴露 interval 合计 `4.95626851575min`，出现 `34` 枚 token，点估计 `6.85999959283 token/min`，高于冻结 `0.50/min` 接受线；
- 一侧 95% exact Poisson working UCB 为 `9.13300249444/min`；总暴露低于零事件达到 `0.50/min` 所需的 `5.9915min`，且两个 LILocBench source 各只有 1 条 sequence，cluster bound 支持也不充分；
- `460` 次再资格化全部记录并抑制；`1,448` 枚 token 全部显式失效。unknown-route activation、cross-reset token、duplicate ID 与 unterminated token 均为 0。

因此 `POLICY_COVERAGE_REJECT` 不是“TTL 已足够安全，只差更多负样本”，也不是风险 HOLD。terminal 同时记录 `triggered_rejections=[POLICY_COVERAGE_REJECT, POLICY_RISK_REJECT]`；按预注册 precedence 以 coverage reject 为主终态。即使忽略 coverage，当前点风险率也已超过冻结接受线约 `13.72×`。

## 冻结边界与非目标

接受线沿用既有 USTRF `false_alerts_per_minute <= 0.50`，不是根据 R0 的 `30.87/min` 或本轮输出回调。可信风险合同沿用 Evidence Maturity V2：95% Poisson working bound 只是必要条件，还要求按 source 分层、以 sequence 为 cluster 的确定性 bootstrap、worst-source sentinel 与每 source 至少 3 条 sequence；不允许 `0/0` 通过。

本轮没有修改或重跑 detector、T0、route、C1–C3、truth、event window 或 clearance；没有接 opener，没有比较候选，也没有开放 selection、L2/L3、Android shadow、H2、人体、独立行走或生产权限。

## 两阶段执行

第一进程先复验 R0 config、41 条 truth-blind ledger、inventory 与 validation SHA，只投影允许的 9 个 frame/runtime 字段，丢弃 R0 token 决策。随后对 41 条完整序列 / 62,229 帧应用冻结 policy，持久化每枚 token 的资格、nominal/effective validity、last-valid frame 与失效原因，并形成不可变 inventory：

- `truth_payloads_decoded=0`
- `event_windows_decoded=0`
- `oracle_tokens_decoded=0`

第二进程先逐 ledger 复验 SHA 和全部 token terminalization，之后才 post-hoc 联结 oracle 与完整负暴露 mask。负暴露严格使用半开区间 `start <= timestamp < end`。

## Policy 结果

| 项目 | 结果 |
| --- | ---: |
| candidate-independent sequence | 41 |
| full-sequence frame | 62,229 |
| policy token | 1,448 |
| supported cell validity coverage | 9/33 |
| no-active-relation fail closed | 3/3 |
| matched unique token | 3 |
| full-sequence extra token | 1,445 |
| requalification suppressed | 460 |
| token invalidation | 1,448 |

失效分布为：relation gap `607`、TTL elapsed `514`、track unobserved `306`、route unknown `19`、reset `1`、sequence end `1`。这表明有限 TTL 与 reset-scope one-token suppression 的组合并未形成可用的持续 attribution authority；不能把首枚 token 过期后的 active relation 当成仍有 token。

## 完整负暴露风险

| source | sequence | exposure min | token | point/min | 95% Poisson UCB/min | cluster bootstrap UCB/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crowdbot_0410_mds` | 10 | 1.899799 | 10 | 5.263716 | 10.384720 | 8.376296 |
| `crowdbot_1203_shared_control` | 13 | 1.364599 | 16 | 11.725060 | 20.186799 | 18.905162 |
| `lilocbench_dynamics_0_front` | 1 | 0.070259 | 0 | 0 | 62.369394 | 不可算 |
| `lilocbench_lt_changes_dynamics_0_front` | 1 | 1.621612 | 8 | 4.933362 | 10.491648 | 不可算 |

逐 source Poisson 使用 Bonferroni 修正后的同时置信度 `98.75%`；pooled 95% UCB 保持 `9.133002/min`。CrowdBot 两 source 的 marginal session-cluster bootstrap 已实际执行 `10,000` 次并固定 seed `20260724`；联合 gate 预注册为每次按 source 分层重采样后先取 max-source rate，再形成 95% 分位。两个 LILocBench source 未达到每 source 3 sequence 的冻结 floor，因此联合 cluster bound 必须标为不足，不能由 pooled rate 掩盖。

## 收据与验证

- config SHA-256：`2079eaa684eeacd0b1be6d21e11b5c21bf4140e89e2dbab87666e4c420f7d1af`
- policy inventory SHA-256：`1e365f825277061393194cb8546b6df500b716b48a73537669f37094858f1874`
- risk ledger SHA-256：`1a745abdb9ed967b46340adfce1e35d9f7a7e84572bc7003b42270a53cb05902`
- requalification ledger SHA-256：`bdbeb71ba6d0d2a50a42d124da2242f74c36c3603f2facb8379b7086d70d359e`
- terminal SHA-256：`30a8cb1d768ae656409265f00a1c7562abfb593edf2659f3292ed29aa550a9f2`
- validation SHA-256：`f674999167e6012d667e4e94ad407e5db374c1d4eac5ffa79455f4093b7e32a1`
- focused tests：12 tests OK，覆盖 duration+frame 双门、半开 TTL、relation gap、reset、unknown route、唯一再资格化 attempt、forbidden input、负暴露 overlap/duplicate、Poisson floor、marginal 与 max-source deterministic cluster bootstrap
- validator：`VALID`；从父 R0 truth-blind ledger 精确重建 41 条 policy ledger、62,229 帧、全部 TTL/失效/抑制、oracle validity coverage、半开负暴露、Poisson/cluster bound 与 terminal
- canonical local evidence：`artifacts.local/evidence/ustrf-candidate-independent-causal-token-policy-risk-gate-r1/`

## 权限与下一独立边界

本轮最大权限为 `CANDIDATE_INDEPENDENT_TOKEN_POLICY_AND_RISK_AUDIT_ONLY`。`POLICY_COVERAGE_REJECT` 是本 policy 的终态，不允许用更多负暴露把 coverage 失败回救，也不允许直接放宽 500ms 或改成无限 TTL。

若继续，下一独立边界应只做 **candidate-independent policy failure attribution**：把 24 个 supported-cell miss 互斥分为“未达到 500ms 资格”“资格已形成但 oracle 在 TTL 后”“relation/track/route 先失效”等，并把 34 个负暴露 token 按 source/sequence/失效原因归因。该诊断不产生新 policy、不调阈值、不运行候选，也不连接 opener；只有归因先闭合，才允许另行预注册单变量 policy 候选。
