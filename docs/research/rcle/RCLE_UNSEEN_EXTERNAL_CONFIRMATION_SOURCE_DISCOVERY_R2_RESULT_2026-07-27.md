# RCLE unseen external confirmation Source Discovery R2 result

日期：2026-07-27

## 终态

`EXTERNAL_COHORT_NOT_EVALUABLE / COMPLETION_AUDIT_FAIL`

科学终态来自实际 geometry role 不足，不再来自 R1 的人工 transport/firewall 阻塞。
两个来源的所有锁定 capture 与 60 个完整固定窗均已执行；独立 aggregate replay
`PASS / errors=[]`。

交付一致性复核曾发现早期结果正文与 completion audit 将固定窗总数误录为
`54`；完整 capture receipts 与 source-role 汇总均独立重算为
`28 + 11 + 9 + 12 = 60`，现已纠正。该错误只影响汇总抄录，不改变任何窗的
ledger、角色、门槛或终态。

同一复核还发现早期 Hugging Face downloader 没有在 R2 selective-prefix 路径
完成后退出，重复取得一份完整 `corridor1-1`，并继续写入非稀疏
`corridor1-2` partial。保守可归因 payload 累计为 `47.610525 GiB`，超过冻结
`40 GiB` 预算 `7.610525 GiB`。冗余进程已终止，partial 保留；因此撤销先前
completion audit 的 `PASS / within_budget=true`，但不改写已经独立重算的
geometry ledger 结果。

## 访问规则纠偏

[风险导向访问标准](RCLE_EVIDENCE_ACCESS_AND_TRANSPORT_STANDARD_R1_2026-07-27.md)
和
[R2 access correction](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R2_ACCESS_CORRECTION_2026-07-27.json)
已通过
[独立 review](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R2_ACCESS_CORRECTION_REVIEW_RESULT_2026-07-27.md)：

- compressed transport presence 与 solid transient decode 不再等同于 RGB 使用；
- OpenLORIS solid stream 中技术上不可分离的 color bytes 只经过 decoder 并立即丢弃；
- 允许 range、bounded prefix、完整 archive、resume 和 decoder 的等价切换；
- 可恢复 transport 事件不再提前终止另一来源；
- RGB member 不落盘、不查看、不进入算法、不影响 selection/tuning。

R1 的历史 terminal 与 receipts 保留，但不再是当前执行 authority。

## 获取与完整性

候选、capture、窗口、公式、门槛和 `40 GiB` 预算均沿用已 review 的 R1 candidate
lock，未改变。

实际被科学流程消费的唯一 candidate payload 为 `21,306,637,315` bytes
（`19.843352 GiB`）：

- OpenLORIS `corridor1-1.7z`：
  `13,853,763,765` bytes，LFS SHA-256 `c7ff1a47…8415`；
- OpenLORIS `corridor1-2`：
  只取得已锁定的 nested-7z prefix，`6,075,166,923` bytes；nested slice 仍为
  `offset=512 / length=6,075,166,411`；
- MultiScan `scene_00000_00.zip`：
  `594,277,493` bytes，LFS SHA-256 `4f7278e8…31d0`；
- MultiScan `scene_00000_01.zip`：
  `783,429,134` bytes，LFS SHA-256 `c085b30e…450c`。

但 cumulative acquisition 还必须计入旧 downloader 产生的：

- 第二份完整 `corridor1-1.7z`：`13,853,763,765` bytes；
- 非稀疏 `corridor1-2` partial：`15,961,011,157` bytes。

所以保守累计为 `51,121,412,237` bytes（`47.610525 GiB`），冻结预算
`42,949,672,960` bytes 已超出 `8,171,739,277` bytes。该 operational breach
不授权删除或重写已有 geometry evidence，也不能继续下载。

materialization audit：

| capture | geometry files/members | geometry bytes | status |
| --- | ---: | ---: | --- |
| `corridor1-1` | `8,518` | `2,339,948,559` | PASS |
| `corridor1-2` | `3,481` | `949,239,777` | PASS |
| `scene_00000_00` | `3` | `237,876,755` | PASS |
| `scene_00000_01` | `3` | `329,236,798` | PASS |

OpenLORIS 每个 materialized member 均重新核对目录集合、uncompressed bytes、
container CRC32 与本地 SHA-256。MultiScan 只物化 exact JSON、JSONL 和
depth.zlib，ZIP CRC/bytes 与本地 SHA-256 均闭合。

## Geometry-only 结果

公式 binding 保持：

`0ce6256e12dd4536f284c7047f1e63faf955fa7bcf87f28fcb93c3e5d9de1add`

OpenLORIS 使用发布方 `aligned_depth`（uint16 × `0.001 m`）、color-aligned
intrinsics、`base_link -> d400_color_optical_frame` extrinsic 和 groundtruth pose。
MultiScan 使用官方 raw-DEFLATE float16-meter decoder 语义、逐帧 transform、官方
ARKit-to-CV `diag(1,-1,-1,1)` 轴变换，并把 color intrinsics 等比例缩放到
`256×192` depth resolution。

| source | fixed windows | positive | below-reference | ambiguous | role complete |
| --- | ---: | ---: | ---: | ---: | --- |
| OpenLORIS corridor | `39` | `34` | `0` | `5` | NO |
| MultiScan | `21` | `0` | `0` | `21` | NO |

OpenLORIS 缺 below-reference；MultiScan 各窗虽 geometry coverage 均为 `1.0`，
但没有窗同时满足冻结的 `>=0.80` fixed-denominator band fraction 与连续
`>=5 s` 门。因此两个来源都无法形成各自的 `1 positive + 1 below-reference`。

独立 validator 不调用两个 source runner，只从完整 pair ledger 重算 60 个窗的
pair index、coverage、band counts、fixed-denominator fractions、最长连续段、角色和
source selection：

- decision：`EXTERNAL_COHORT_NOT_EVALUABLE`；
- status：`PASS`；
- errors：`[]`；
- validation receipt SHA-256：
  `655345e06dceb7c058797362aadb1d7867f13f4bcfbafe6199fe5c7f9812dc24`；
- role-complete independent sources：`0/2`。

## 保留边界

- candidate expansion/replacement：`false`；
- threshold/cadence/coverage/denominator/continuity change：`false`；
- post-outcome window addition/sliding：`false`；
- RGB materialization/visualization/algorithm execution：`0`；
- Android/device execution：`false`；
- algorithm change：`false`；
- selected windows：`[]`。
- frozen cumulative download budget exceeded：`true`；
- redundant downloader stopped：`true`；
- partial acquisition file preserved：`true`。

按照冻结 stop rule，外部 cohort 继续保持
`EXTERNAL_COHORT_NOT_EVALUABLE`。不得扩候选、降门、pooled rescue 或启动 Android。
