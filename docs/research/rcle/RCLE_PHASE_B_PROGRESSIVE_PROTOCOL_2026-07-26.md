# RCLE Phase B 渐进式协议

状态：`DISCOVERY_OPEN / CANARY_NOT_STARTED / CONFIRMATION_NOT_FROZEN`

日期：2026-07-26

上位规则：[BlindAssist 渐进式研究治理](../../RESEARCH_GOVERNANCE.md)

机器合同：
[Progressive Discovery R0](RCLE_PHASE_B_PROGRESSIVE_DISCOVERY_R0_CONTRACT_2026-07-26.json)

## 为什么另立版本

B1 R5 把 source discovery、实验角色准入和最终算法 audit 压在一次 one-shot
geometry admission 后面。唯一执行又因 blank-grid serialization mismatch 被判
INVALID。该结果继续保持原样，但它只关闭：

- `RCLE-B1A-R5-CANONICAL-1` evidence instance；
- 已消费的 `RCLE-B1-R5` protocol version；
- 明确依赖该 ledger 的 `RCLE-B1B-R5`。

它不关闭 RCLE 科学问题。作用域由
[B1A closure overlay](RCLE_PHASE_B_BONN_B1A_CLOSURE_SCOPE_2026-07-26.json)
机器记录；旧 terminal 和 artifacts 不改写。

## 新的三层 Phase B

### Discovery

目标是理解真实数据，而不是证明算法：

- 扫描公开来源的 metadata、pose 和受控 depth；
- 保存连续分布，不先把所有数字变成硬门；
- 比较 raw translation speed 与 pose+depth 导出的
  translation-induced radial expansion/parallax；
- 不读取候选 RCLE 算法输出来选择角色；
- 输出只限 candidate 和 source characterization。

Discovery 可重复、可增量、可在结果出现前修改诊断；不要求 one-shot claim。
每次失败必须留下 learning record，并把可复用内容标成 regression fixture、
source characterization、counterexample 或 canary。

### Canary / Development

只有 Discovery 找到有信息增益的候选后才建立：

- 每个机制分支先用少量 sequence；
- 允许看 RGB、曲线和中间量，修坐标、同步、符号与实现；
- 候选按因果区分度和预期信息增益排序，不做无理由穷举；
- 使用过的 sequence/window 从 confirmation 中烧掉；
- 只能形成 mechanism direction 或 implementation readiness 结论。

### Confirmation

只有机制和数据现实都清楚后才冻结：

- 独立、未访问的 confirmation partition；
- 结果前固定阈值、单位、依据、敏感性、缺失处理和统计；
- 独立 validator 与适合证据价值的 publication contract；
- VALID/INVALID 与 scientific PASS/FAIL/NOT_EVALUABLE 分开；
- failure terminal 只传播到有显式依赖的最小 scope。

## 当前 Bonn 资产如何复用

固定六序列十窗口已经发生 geometry outcome access，不再作为完全 outcome-blind 的
Phase B confirmation 角色选择证据。它们仍可合法用于：

- blank-grid producer/validator parity regression；
- source-native geometry 实现 canary；
- Bonn motion/source characterization；
- challenged proxy 的 counterexample/stress case。

INVALID receipt 中的诊断数字仍不能升级为正式数据结论。

## 当前执行边界

当前开放 Discovery，未开放 RCLE RGB algorithm canary、confirmation、Kill Gate B、
Replay、Android、人体、安全或生产晋级。Discovery 发现候选不自动开放下一层；
应先形成 failure/learning summary、数据角色隔离和最小 canary protocol。

真实用户有效性、独立行走安全认证和生产级设备闭环不是当前 RCLE 论文主线的前置
条件；只保留“研究原型、不可作为独立助行工具”的诚实边界。当前研究优先形成可复现
的机制证据、消融、失败分析和论文/Demo 闭环。
