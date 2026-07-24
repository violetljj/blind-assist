# USTRF known-route eligible-delivery formation 候选盲失败归因 R1（2026-07-24）

## 结论

本轮终态为 `FAILURE_ATTRIBUTION_COMPLETE / VALID`，36 个 `candidate × clearance-eligible event` 单元均获得唯一归因，`unexplained_gap=0`。但这只是把失败机制解释闭合，不是修复：C1/C2/C3 仍只有 `3/12 / 2/12 / 1/12` 形成 eligible delivery，30 个 no-delivery 单元仍然存在。

首因不是普遍 detector 漏见或冻结 `min_alert_frames=2` 不足。30 个失败中：

- 27 个在目标获得可归因 delivery 形成机会之前，候选 one-shot episode 已经打开或被 route-invalid guard 隔离，因而没有产生新的 delivery edge；
- 3 个来自同一 CrowdBot 1203 truth event：窗口 4 帧中只有 1 帧观察到目标，且 `route_known=0/4`，三个候选均从未形成 active relation；
- `no_observation`、`active_streak_lt_min_alert`、`reset_split` 与 `unexplained_gap` 均为 0。

因此，在本冻结的 12-event proxy/model-truth 分母内，下一修复变量若启动，应针对 **eligible target attribution 与 one-shot episode opening 的因果顺序/隔离**，当前失败不能再归因给 detector、reset 串接或 unknown-route fail-closed，也不能通过改阈值、缩分母或把 guard terminalization 计作 delivery 来回救；该结论不能外推到新来源或人体场景。

本结论仍只基于冻结 proxy/model truth 与 replay；不产生候选比较、胜者、selection、L2/L3、Android shadow、H2、人体效果、独立行走安全或生产权限。

## 严格边界与候选盲顺序

本轮没有重跑或修改 detector、T0 association、route、C1–C3、route-invalid/reset guard、truth、clearance、阈值或分母。固定顺序是：

1. 复验 A2 与 guarded terminal，以及全部 123 + 123 条父 trace 的路径、SHA-256、身份和顺序；
2. 在不解码 truth/clearance mask payload 的阶段，对 186,687 帧候选状态形成事实生成 123 个紧凑 trace receipt，并冻结 inventory SHA-256；配置加载时只核对冻结 truth/mask 文件的 SHA-256，不解析其 JSON 内容；
3. 重新读取并复验完整 candidate-blind inventory；
4. 此后解码 12-event clearance eligibility mask，构造本次实际消费的 12 条 event-scope blind trace，并先冻结其内容 inventory SHA-256；
5. 最后才解码冻结 truth，以 IoU `>=0.3` 做 post-output target attribution；
6. 每个候选—事件单元严格分配一个互斥标签；任何剩余 `unexplained_gap` 都令归因门 fail closed。

紧凑 receipt 哈希绑定逐帧 identity、route/reset、observed track ID、active relation track ID、baseline delivery/closure/active 与 guarded lifecycle event；完整 bbox 仍由已哈希绑定的 A2 trace 提供，没有复制成第二份 579MB 权威数据。

## 互斥归因结果

| 归因 | C1 | C2 | C3 | 总计 |
| --- | ---: | ---: | ---: | ---: |
| 已形成 eligible delivery | 3 | 2 | 1 | 6 |
| 无 target observation | 0 | 0 | 0 | 0 |
| 有 observation，但从未形成 active relation | 1 | 1 | 1 | 3 |
| active streak 小于冻结门槛 | 0 | 0 | 0 | 0 |
| episode 在 alertable window 前已打开 | 2 | 4 | 3 | 9 |
| episode 在 window 内、但早于 target attribution 打开 | 5 | 4 | 4 | 13 |
| route-invalid 后 baseline latch 仍在、guard key 已 quarantine | 1 | 1 | 3 | 5 |
| reset 将形成证据切断 | 0 | 0 | 0 | 0 |
| 无法解释的 formation gap | 0 | 0 | 0 | 0 |
| 合计 | 12 | 12 | 12 | 36 |

已形成的 6 个单元与父 clearance receipt 精确一致：C1 为 `event_003 / pedestrian_route_intersection_002 / pedestrian_route_intersection_008`，C2 为 `event_003 / pedestrian_route_intersection_008`，C3 仅 `event_003`。

27 个 latch/pre-open 单元进一步分成三类：

- 9 个 carry-in episode 在 `alertable_start` 前已打开；
- 13 个 opening 位于评分窗口内，但早于该 target 在候选语义下取得可归因 active support；
- 5 个存在 pre-invalid baseline latch 与 guarded key quarantine：C1/C2 的 `pedestrian_route_intersection_010`，以及 C3 的 `pedestrian_route_intersection_002 / 008 / 010`。

12 个评分窗口内均没有 reset frame，因此 `reset_split=0` 是直接观测，不是把 reset 失败归入其他类别。对 C1，冻结 2 帧形成门按同一 track lineage 计算；对 C2/C3，按全局 route-occupancy episode 计算，不能把不同候选的 key 语义混用。

## 机制含义

“窗口内已有连续 active relation”不等于“会产生 eligible delivery”。当前 one-shot FSM 只在 inactive→active 的 opening edge 交付：

- C1 的同一 track 已在 truth attribution 成立前进入 active，后续满足目标归因时不会再次交付；
- C2/C3 的全局 occupancy episode 可能由更早的 route risk 打开，目标稍后进入时也没有新的 opening edge；
- route-invalid guard 能安全关闭 guarded key，但不能改变冻结 baseline FSM 的 internal latch；若 baseline 没有新 delivery，guard 无权凭 truth 或 active relation 自造 episode。

这解释了为什么前一边界能把 unknown/stale guarded active 降为 0，却不能自动提高 eligible delivery 或 relation-based clearance。guard 正确 fail closed 与 delivery formation 缺口可以同时成立。

## 收据与验证

- 配置：`configs/ustrf_route_target_known_route_eligible_delivery_failure_attribution_r1.json`
- config SHA-256：`2cfa3afd2c548be0f43fa18fb36be1ce07493a66cda5b6df0ff51105a2b32c69`
- candidate-blind inventory SHA-256：`4074cba383c130e564386acf17039dfdf9b45c0c8580d32083807dd18a0b6f0f`
- event-scope blind pack inventory SHA-256：`95a8bb61a14d723291f7a04d97f40c17dabc602dbbc5f0c70d47dbe86f1125f5`
- terminal SHA-256：`77880060384f3288b589a23a87da49f95be10554b98ffd2690c4c8b38efc1e33`
- validation SHA-256：`51f2c6ae414b1bad34b1393ea614b86b85e966c1c94fe0b9e5ff3111e552b8d7`
- focused + parent regression tests：`17 tests OK`
- 单独运行 validator：`VALID`；共享同一审计 core，从父 trace 精确重建 123 个 blind trace receipt / 186,687 帧绑定与 12 条 event-scope blind pack，并重算 36 个单元，`unexplained_gap=0`；这不是第二套独立算法实现
- canonical 本地证据：`artifacts.local/evidence/ustrf-known-route-eligible-delivery-failure-attribution-r1/`

首次逐帧物化完整 observation box 会重复生成约 579MB 非必要本地数据；该尝试没有 terminal 权限。最终改为逐帧 formation-fact stream SHA + 父 trace SHA 的紧凑 receipt，保留同一候选盲顺序与精确复算能力。

## 停止边界

本轮在 `candidate-blind attribution PASS / delivery mechanism still unfixed` 处停止。没有修改 FSM，也没有诊断 delivery 之后的 relation-based closure。下一独立边界必须另行冻结；不得直接用本归因选择候选或开放任何更高权限。
