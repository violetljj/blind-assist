# USTRF eligible target attribution → isolated one-shot opening 机制修复诊断 R1（2026-07-24）

## 结论

本轮已按独立边界冻结并完成 `ELIGIBLE-TARGET-ATTRIBUTION-ORDERED-ISOLATED-OPENING-R1`。终态为 `ORACLE_MECHANISM_REPAIR_DIAGNOSTIC_COMPLETE / VALID`，冻结数据上的顺序、隔离与 one-shot 机制门通过：

- 父阶段已经形成 delivery 的 `6/36` 个 candidate-event 单元在新 opener 下仍全部 token-qualified；这不表示逐帧复刻父 delivery 时间；
- 9 个 window 前 pre-open、13 个 window 内但早于 target attribution 的 opening、5 个 pre-invalid baseline latch / guard quarantine 单元全部由隔离 opener 反事实恢复；
- 三个候选均为 `11/12`，合计 `33/36`；
- CrowdBot 1203 同一 event 的三个候选单元仍因没有 eligible active relation 而 fail closed，没有被机制伪造为 delivery；
- 36 个 ledger 均完成 background namespace 强制变异，目标 opening 输出变化为 0；27 个反事实恢复单元都在 qualification 前或当帧实际含 background activity。opening 早于 qualification、one-shot cardinality violation、重复 delivery key 与 accounting gap 也均为 0。

这个结果只证明：在冻结 proxy/model truth 生成的 oracle attribution token 上，把 eligible target attribution 放在 one-shot episode opening 之前，并把 baseline pre-open 状态隔离，足以消除上一轮定位的 27 个 opening-edge 抢占失败。它不是可部署在线修复：仓库当前没有不读取 truth 的 causal candidate-blind token producer，因此不能把 `33/36` 当作候选效果、人体效果或生产能力。

## 冻结顺序与隔离合同

本轮没有修改或重跑 detector、T0 association、route、C1–C3、A2 trace 或 route-invalid/reset guard，也没有改阈值、truth、分母和 clearance。执行顺序固定为：

1. 复验父 attribution config、terminal、validation 与 event-scope blind inventory；
2. 读取已经持久化的 candidate-blind event-scope formation facts，并从父 trace 精确重建每个 event 所在 reset scope 的完整前缀；
3. 仅在此后联结冻结 truth，用 `route_known + target IoU >= 0.3 + active relation` 生成逐帧 eligible-attribution token；ledger 范围从 reset-scope start 到 truth terminal clear；
4. 先持久化并复验全部 36 个 token ledger；
5. one-shot opener 读取 truth-derived event-scoped token、event identity 和由 ledger extent 编码的 truth window，但拒绝 raw truth payload、observed box、baseline key、guard event 和评分窗口字段；
6. ledger 同时携带独立的 background namespace active/opening facts；opener 以未变化的连续 2 帧门在 qualification frame 打开 target episode，background namespace 被强制翻转并增加 opening 后，目标 opening 输出必须完全不变；
7. C1 保持同一 attributed track 连续性；C2/C3 只在同一冻结 event identity 内允许 track handoff；reset 与 unknown/stale route 均打断资格累计；
8. 每个 `candidate + source + sequence + reset segment + event` 最多形成一个 delivery key。

truth/event identity、truth-derived target token 与 truth-window ledger extent 均属于 opener 的 oracle 协议信息，不能作为 Android 或生产 runtime 输入。opener 的严格白名单合同会拒绝 raw truth、truth box、observed tracks、baseline active keys、baseline deliveries 与 guard events，并拒绝 frame gap、非单调 timestamp、无 reset 的 segment 跳变及 unknown-route token。

## 冻结结果

| 机制结果 | 单元数 |
| --- | ---: |
| 父 formed-delivery 单元仍 token-qualified | 6 |
| 反事实恢复 window 前 pre-open | 9 |
| 反事实恢复 window 内 pre-attribution opening | 13 |
| 反事实恢复 pre-invalid latch / quarantine | 5 |
| 仍无 eligible active relation | 3 |
| 合计 | 36 |

| 候选 | 父 eligible delivery | 隔离 opener qualified delivery | 仍 fail closed |
| --- | ---: | ---: | ---: |
| C1 | 3/12 | 11/12 | 1 |
| C2 | 2/12 | 11/12 | 1 |
| C3 | 1/12 | 11/12 | 1 |

不能用 `11/12` 比较候选：三者消费的是同一个 truth-assisted oracle target/event scope，且结果不包含 false alerts、repeat、clearance、evidence age、unknown-route lifecycle 或完整序列 runtime attribution。

## 收据与验证

- 配置：`configs/ustrf_eligible_target_attribution_ordered_isolated_opening_r1.json`
- config SHA-256：`f222d210c7994e8986d69b98df1d0786962790e0a2bc27fb5ce6d60366135145`
- token inventory：36 个 ledger、5,043 个 reset-scope candidate-event frame；SHA-256 `98344298d25dfa4fecb5b3d07699016e4181c3149d6b730b051548c49f14ff64`
- terminal SHA-256：`998bcc46d7fb6dbd39fb192930a90a637762540476b35a474a1e2981d1ca299c`
- validation SHA-256：`bb693bcea53ef9a35e3b0f5496367924e34df615dfaa56a1dc3ec9a67d36f068`
- focused tests：13 tests OK，覆盖 C1 same-track、C2/C3 event-scope handoff、reset/unknown-route 断开、frame/timestamp/segment 合同、持续 token one-shot、background namespace 变异不变性、raw truth/baseline 字段拒绝、opener phase 不解码 raw truth、阈值/authority 漂移、临时副本 token SHA 变异与 terminal 精确复算
- validator：`VALID`；从冻结父 full blind trace 与 truth 精确重建 36 个 reset-scope token ledger，验证 36 次 background namespace 变异不变性，再复算全部 mechanism outcome 与 terminal
- canonical 本地证据：`artifacts.local/evidence/ustrf-eligible-target-attribution-ordered-isolated-opening-r1/`

## 权限与停止边界

本轮最大权限是 `TRUTH_ASSISTED_ORACLE_MECHANISM_DIAGNOSTIC_ONLY`。没有候选比较、winner、ranking、selection、L2/L3、Android shadow、H2、人体效果、独立行走安全或生产权限；没有修复 post-delivery relation clearance。

本轮在“oracle 顺序/隔离机制足以恢复 27 个 opening-edge 单元，但 causal candidate-blind token producer 仍为 0”处停止。后续若继续，必须另行冻结一个不读取 truth/event window 的 causal attribution-token interface 与 producer 验证；不得把本轮 oracle token 或 `33/36` 直接接入 App、选择候选或回救其他指标。
