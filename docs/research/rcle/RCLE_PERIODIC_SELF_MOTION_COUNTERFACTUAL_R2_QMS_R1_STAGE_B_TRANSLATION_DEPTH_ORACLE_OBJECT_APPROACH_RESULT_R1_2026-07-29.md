# RCLE periodic-self-motion R2：Stage B translation-depth oracle 结果 R1

结论先行：Stage B 已完整运行并通过独立重建，但科学终态是
`B_ORACLE_NOT_EVALUABLE`，不是 `GO`、`FREEZE` 或 `STOP`。

必须优先通过的 rotation leakage boundary 在 `0/8` clusters 通过；同时
`18` 个 required `arm × cluster` 的 paired coverage 低于冻结的 `0.75`。
因此 translation suppression 和 object-approach retention 只能保留为边界失败后的
描述量，不能购买单项升级，也不能进入 feature contract C 或 fusion experiment D。

## 冻结设计与执行

- `8` 个独立 cluster、每个 `5` 臂，共 `40` 条 sequence；
- 每条 `602` frames / `601` pairs，总计 `24,040` pairs；
- R3、strict `>0.01/s`、三 pair、`PairState`、abstention 和 fixed-601
  denominator 均未修改；
- baseline/oracle 在相同 geometry-valid tracks 与 common cells 上分别重新执行
  unchanged local affine fit；
- signed response 与 `median(abs(cell expansion))` 分开，cluster 是唯一分析单位；
- object-approach 使用 previous/current 同一 `object_id=1001` 材料点的 target-only
  tracks，full-scene pair 仍须先通过原 `>=5/9` gate；
- `STATIC_SCENE` 只作 required-coverage negative-control diagnostic，不新增 routing
  阈值。

response 前先物化全部几何并由独立 validator 检查 `40` 个 arm、`24,080` 个
render-input 和每帧 `25/25` 个固定目标材料点；全部通过后才创建 activation 和
response root。

## 6 GiB 资源门

本次 successor 把未消费工作负载的 launch/refill guard 从 `8 GiB` 改为
`6 GiB = 6,442,450,944 bytes`，已在途 worker 保持原 `4 GiB` emergency floor。
历史 8 GiB receipts 与 source bytes 没有改写。

- launch available：`8,041,189,376 bytes`（约 `7.489 GiB`）；
- minimum available：`6,359,343,104 bytes`（约 `5.923 GiB`）；
- 该最低点发生在最后一波全部已 in-flight 后，合法高于 `4 GiB` floor；
- swap-in / swap-out delta 均为 `0`，未触发 sustained paging；
- `4` workers，residual worker 为 `0`，wall time `4,420.67 s`。

## 独立重建

独立 validator 不导入 runner、R3、tracking、evaluation 或 local-fit 实现，从 sealed
`paired_tracks.npz` 重算：

- `24,040` 个 pair；
- `865,440` 个 cell fit；
- baseline/oracle evaluability、coefficient-derived expansion、fit residual；
- common support、signed pair median、absolute pair median；
- type-7 P50/P90、strict `>0.01/s` 三 pair、fixed-601 density；
- required-arm coverage、cluster estimands、rotation precedence 和最终 routing。

首次 validator 在产生 analysis/receipt/decision 前 fail-closed：runner 的审计系数
来自序列化前 float64，而 sealed R3 track ledger 是 float32，原 validator 错误要求
两者逐元素在 `2e-5` 内相等。薄的 numeric-representation amendment 只把审计系数
改为有限性与公式自洽检查；科学 expansion、residual、coverage、阈值、gate 和
routing 仍从 sealed float32 tracks 独立复算，40 条 sequence 没有重跑、替换或重种子。

## 为什么是 `B_ORACLE_NOT_EVALUABLE`

### Rotation boundary：`0/8 PASS`

translation oracle 在 rotation-only 臂数学上是严格 no-op：

- `u_T max = 0.0 px`，8/8；
- baseline 与 oracle absolute P90 差为 `0.0/s`，8/8。

但 unchanged R3 的 rotation absolute P90 为 `0.0940–0.1806/s`，远高于冻结上界
`0.01/s`。oracle fixed-601 三-pair trigger density 也只有一个 cluster 为零，其余
为 `0.0067–0.0466`；此外两个 rotation cluster coverage 只有 `0.7155` 与
`0.4226`。所以必须通过的 rotation leakage 限界在 8/8 clusters 失败。

### Required coverage：18 个 arm-cluster 失败

低于 `0.75` 的分布为：

| required arm | failed clusters |
| --- | ---: |
| `EGO_ROTATION_STATIC_SCENE` full-scene | 2/8 |
| `EGO_TRANSLATION_STATIC_SCENE` full-scene | 3/8 |
| `OBJECT_APPROACH_STATIC_CAMERA` target-mask | 6/8 |
| `OBJECT_APPROACH_PLUS_EGO_6DOF` target-mask | 7/8 |
| `STATIC_SCENE` full-scene | 0/8 |

这属于 pair/common-cell/target-support 的预冻结可评估性失败，不能通过 zero fill、
插值、换 denominator 或把 scientific nonpass 冒充有效结果来修复。

### 只作描述、不得驱动 routing 的结果

- translation signed suppression 为正：`8/8`；
- cluster median signed suppression：`0.007885/s`；
- absolute leakage suppression `>=0.5`：`0/8`；
- static-camera object approach nominal retention：`8/8`；
- object-approach plus ego nominal retention：`1/8`。

这些量都处在 rotation/coverage failure 之后，不能解释成 oracle 有效或无效，也不能
支持单项升级。尤其 combined arm 的 target coverage 在 `7/8` clusters 不足，其
nominal retention 不能作为稳定性证据。

## 权限与下一步

- single targeted upgrade：`NOT_AUTHORIZED`；
- feature contract C：`CLOSED`；
- fusion experiment D：`CLOSED`；
- Stage B retry / replacement / reseed：`NOT_AUTHORIZED`；
- 正式 `480+16`：`0 run / 0 pair-core calls / NOT_CONSUMED`；
- Android、产品、主动告警、危险或安全结论：`NOT_AUTHORIZED`。

当前正确下一步不是继续运行 Stage B，而是把本次 controlled-generator
`NOT_EVALUABLE` 作为论文/机制审计的负结果收口。若要提出新的研究问题，必须另立
明确授权与新合同；不得沿用本次 identity、阈值或 response 做事后救援。

机器 closeout：
[independent closeout receipt](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_INDEPENDENT_CLOSEOUT_RECEIPT_R1_2026-07-29.json)。
