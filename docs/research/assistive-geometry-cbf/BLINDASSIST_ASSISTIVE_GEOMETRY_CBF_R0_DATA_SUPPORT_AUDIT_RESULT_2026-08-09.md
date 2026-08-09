# AG-CBF R0 TRAIN grid data-support audit result

状态：`governed / NOT_EVALUABLE / ROUTE_CLOSED_BEFORE_ORACLE`

## 结论

`BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0` 在第一道 DATA SUPPORT 门 fail-close：

> `AG_CBF_R0_DATA_SUPPORT_NOT_EVALUABLE_ROUTE_CLOSE`

固定 16 parent × 64 source-order-even 帧共 1,024 帧中，仅 44 帧满足预注册的 ground-grid
完整性与覆盖门；portrait/landscape 只有 `36/8`，没有任何 parent 达到 `32/64`。因此本 R0
不能进入 ORACLE CEILING、REPRESENTATION VALUE 或 TRAIN。

这不是对 maximum-bottleneck corridor 数学或其潜在表示价值的反证；它只证明当前 B1 TRAIN
source-geometry target contract 无法在冻结的 `32×31`、forward `0.2–5.0 m`、lateral
`-2.0–2.0 m` 合同下提供足够广的 parent/orientation 支撑。

## 冻结门与实测

| gate | 冻结要求 | 实测 |
|---|---:|---:|
| evaluable frames | ≥ 640 / 1,024 | 44 |
| passing parents | ≥ 12 parents，各 ≥ 32 / 64 | 0 |
| portrait evaluable | ≥ 128 | 36 |
| landscape evaluable | ≥ 128 | 8 |

各 parent evaluable 帧为：`41159448=14`、`42445086=12`、`42898024=8`、
`47204445=10`，其余 12 个均为 `0`。

失败原因允许重叠。1,024 帧中，`974` 帧缺少四个 forward quartile 都达到门槛的 ground
支撑，`816` 帧缺少三个 lateral third 都达到门槛的观测；`288` 帧 ground plane/geometry
contract 无效。median observed/ground cell 只有 `16/14`，且 minimum-forward-quartile
ground cells 的 p90 仍为 `0`。主缺口是可用于 2D 拓扑 corridor 的纵横覆盖，而不只是 target
文件缺失。

## 完整性与权限

- 输入 manifest SHA-256：`A6F809C7...A7C2`；
- 协议 SHA-256：`55DBFE6D...FBB5`；
- 完整 1,024-frame artifact：
  `artifacts.local/evidence/assistive-geometry-cbf/r0-data-support-audit/result.json`，
  `943465 bytes`，SHA-256 `C6D151BC...DA34`；
- 逐 target 复核 bytes/SHA，仅打开八项 TRAIN source-geometry 字段；
- 未打开 RGB、模型、feature、A0 consumed Development、Calibration 或 Confirmation；
- 未计算 CBF oracle outcome，未创建 model/checkpoint，`UNKNOWN` 未被当作 negative。

## 终态与重开规则

本 R0 无 successor。不得根据本结果事后降低 gate、缩短网格或把稀疏 UNKNOWN 填成 free 来
“救活”当前版本。若未来有新的 pre-outcome source-geometry/target contract，或有独立物理依据
支持新的观测模型与网格版本，必须另立路线版本、重新冻结数据支撑门，再从 DATA SUPPORT 开始；
不能继承本 R0 的 oracle、训练或主线 authority，因为这些 authority 从未产生。
