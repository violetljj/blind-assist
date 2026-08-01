# HFTF Stage C multi-source evaluation qualification E0.2

日期：2026-08-01

状态：`FROZEN_BEFORE_FIXED_BATCH_RGB_DEPTH_OR_LABEL_OUTCOME`

## 1. 目的

E0.1 的 heldout 只有 1 个 risk cell/anchor，单 source 评价设计过于脆弱。E0.2 不改
`.4 s` student formulation、模型、训练、阈值或 success margins；它只做一次性、
有限预算的 multi-source dev/heldout qualification。

## 2. 固定 cohort

从 95 条 healthy inventory 排除前十条 consumed trajectories，同时排除它们的十个
recording dates。按总字节、ID 升序取前六个不同日期，排序位置奇数为 dev、偶数为
heldout：

| role | trajectory | date | rows | bytes |
| --- | --- | --- | ---: | ---: |
| dev | `2024_12_15__11_46_53` | 2024-12-15 | 747 | 199,143,693 |
| heldout | `2024_09_28__15_37_24` | 2024-09-28 | 1,101 | 200,441,631 |
| dev | `2024_11_07__20_16_15` | 2024-11-07 | 859 | 201,276,140 |
| heldout | `2024_11_13__11_42_11` | 2024-11-13 | 771 | 203,504,068 |
| dev | `2024_08_16__15_32_22` | 2024-08-16 | 831 | 204,354,359 |
| heldout | `2024_12_28__15_49_23` | 2024-12-28 | 955 | 223,280,846 |

合计 1,232,000,737 bytes。冻结时六条 RGB/depth 与 label outcome 均未读取。机器合同
绑定 18 个 source files。

## 3. 门与停止

每 source 必须通过 exact transport、plane/speed、`.4 s` known `>=.70` 与 UNKNOWN
防火墙。每个角色聚合必须有至少 4 risk cells、4 physical anchors、2 个含 risk 的
source、2 个方向和 300 no-risk cells。

成功只授权生成 exact E0.1 corpus 并执行完全不变的三臂学生合同。任何门失败都终止
该 EgoWalk foot-ground student source route，不再逐条或批量扩大。

E0.2 不支持自然 prevalence、hazard truth、完整 HFTF、主线、App 或安全 claim。

机器可读真源：
[HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.json](HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.json)
