# Last-10m action-responsive engineering sanity result

状态：`COMPLETE_AND_SEALED / CONTROL_POLICY_BOTTLENECK / NO_P1 / NO_REFERENT_PERSISTENCE / NO_SUCCESSOR`

结论：在冻结的 action-responsive replay 中，未修改的 repeated-grounding 控制逻辑没有进入 `ARRIVAL`。最终 verdict 为
`CONTROL_POLICY_BOTTLENECK`，不是 scientific confirmation。

## 构造与冻结

盘点覆盖了现有 Last-10m 状态机/receipts/tests、冻结 Grounding DINO + Terra P0、Mapillary 本地缓存及其 GPS/heading/
sequence metadata、带 trajectory 的 ARKitScenes RGB 资产，以及 ADT real-RGB/pose 资产。可用 Mapillary inventory 为
14 份 metadata、338 张本地 RGB、12 张既有 `UNIQUE` review frame 和 9 个可映射入口 anchor。按“本地 pose frame
最多、再按 anchor/sequence 字典序”的 outcome-independent 规则，自动选中 `hofbladelin`、OSM
`node/4974210260 (entrance=main)` 与 sequence `dbgdqomGU5W7oPnzKZjxLg`。

原生 Mapillary approach 段能支持连续 forward/rescan，但 6 个可达 start 没有方向正确的同位置 turn frame。因此零次
formal observation 的 `responsive-scenes-v0` 被拒绝；没有拿结果倒推修图。最终 `v1` 冻结 22 张真实 sequence frame、
110 个预生成 viewport states 和 6 个 starts：

- `FORWARD` 与 `RESCAN_HOLD` 只跳真实 GPS/heading/timestamp/sequence 邻接帧；
- `TURN_LEFT/RIGHT` 只改变同一实拍帧的固定五档水平 viewport（每档 12° bookkeeping、10% 水平平移）；
- transition graph、roster、OSM pose-arrival sidecar 和 public manifest 在 provider 调用前 hash 冻结；
- 图中 LEFT/RIGHT/FORWARD/RESCAN edges 分别为 `88 / 88 / 80 / 90`；
- arrival fail closed：距 OSM main-entrance proxy `<=8 m`、绝对 bearing error `<=30°`，且控制逻辑仍必须用新帧二次确认。

viewport 是 outcome-independent simulator，不是真实转头画面；它没有 tracker、物理实例 identity、keyframe、SLAM/VIO
或 world-relative referent state。

## 唯一正式运行

| 指标 | 结果 |
|---|---:|
| episode completion | `0 / 6` |
| false arrival | `0` |
| wrong-target confirmation | `NOT_EVALUABLE`（缺 exact frame-region truth） |
| observations / reliable grounding | `29 / 27` |
| direction commands | `27`（LEFT `17` / RIGHT `3` / FORWARD `7`） |
| rescans | `2` |
| abstentions | `0` |
| exhausted trajectories | `6 / 6` |
| first reliable grounding latency | median `9,679.5 ms`; min `9,129`; max `32,352` |
| failure state | `ADVANCE_AND_REOBSERVE=5`; `RESCAN=1` |

episode `e06` 从距入口 proxy `15.55 m` 推进到 `5.56 m`，证明环境不是固定 playlist：7 次 forward/turn 与真实
sequence transitions 确实改变了 observation 和 pose。但在 10 次 observation 内仍未出现“当前帧居中近距 cue + 新帧
复核”，最终在冻结动作图边界 `EXHAUSTED`。

最大失败原因是当前帧方向选择：虽然 `27/29` observation 为可靠 `GROUNDED`，控制输出仍以 LEFT 为主并在 viewport
间振荡或持续推向同一边界，无法稳定对齐后进入 `ARRIVAL_CONFIRM`。所以本批次不是 grounding availability 主导，
而是 candidate-to-direction/arrival control policy 主导。没有任何完成声明，因此 false arrival 为 0；这不等于
“前方安全”。缺少逐帧 exact entrance region truth，wrong-target confirmation 保持 `NOT_EVALUABLE`，不能写成 0。

## Receipts 与异常

本地 ignored evidence root：
`artifacts.local/evidence/last-10m-regrounding-v0/responsive-run-v1/`。审计为 `29/29 RUN_SUCCESS`、`IN_DOUBT=0`、
provider observation 重跑 `0`、formal outcome 后 graph/model/threshold/roster 改动 `0`；raw result SHA-256 为
`e1348cc89a4012e543677be9e00d512337a6d82bd3bcd8d5d1c3fefb1799fda2`。

sealed raw result 对 6 个 latency 使用了 upper-middle `9,881 ms`。本 closeout 从不可变 episode summaries 按常规定义
报告 `(9,478 + 9,881) / 2 = 9,679.5 ms`；没有修改或重跑任何 provider observation。

## Claim ceiling 与后续判断

结果只回答 deterministic responsive mechanics。它不能外推为真实盲人、真实建筑入口 walk-through、产品导航、
accessibility 或 safety 有效性。值得以后做一次极小真实 responsive walk-through：本次已把 fixed-playlist exhaustion
排除为唯一解释，但 viewport 平移本身也可能制造视觉伪影，只有真实转头/小步才能区分 simulator artifact 与现场控制
行为。这里不自动建立该 successor，不重开 P1，也不建立 referent-persistence 研究线。

机器结果：[`JSON`](BLINDASSIST_LAST_10M_RESPONSIVE_SANITY_RESULT_2026-08-22.json)。
