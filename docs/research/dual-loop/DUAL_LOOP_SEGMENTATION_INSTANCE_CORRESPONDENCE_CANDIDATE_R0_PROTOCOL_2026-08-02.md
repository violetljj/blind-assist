# Instance correspondence candidate annotation R0

状态：`THESIS_DEVELOPMENT / CANDIDATE_ANNOTATION_ONLY / NOT_CONNECTED_TO_ALERTS`

## 目的

Failure Atlas 已能定位 segmentation hazard component，但只有“YOLO box 与 hazard
像素相交”的 frame-level 关系，不能判断该 hazard 是否被某一个 YOLO instance 有效
覆盖。本合同新增一层候选 annotation：

```text
mask connected component
    <-> YOLO detection
    <-> component temporal track / detector temporal track
    <-> semantic class
    <-> depth cluster (when supplied)
```

它只提供候选 correspondence，不产生 instance truth。历史 R0/R1/R2 结果、消费身份、
residual truth 和默认 App 权限保持不变。

## 输入与配对

- segmentation `frames.jsonl` 与 `components.jsonl` 来自同一 rehearsal/canonical
  view；component mask 从 `packed_masks.candidate_<class>` 按 `component_index` 重建；
- YOLO trace 按 `source_id + frame_id + image_sha256` 精确配对，保留原始
  `label/class_id/confidence` 与 source-space bbox，并投影到同一 256x256 analysis grid；
- 组件已有 `temporal_track_id` 时只作为输入证据；缺失时按同一 sequence/class、相邻
  materialized observation 和预设 IoU 派生 deterministic track；
- detection 已有 `track_id`/`temporal_track_id` 时优先使用；否则按同类框的相邻 IoU
  派生 track；
- depth cluster 与 optical flow 是 optional sidecar。没有 sidecar 时值为 `null` 并
  写入 `missing_evidence`，不伪造 identity flow、metric depth 或 cluster identity。

## Pair evidence

每个 component/detection pair 固定写出：

- component mask 与 projected detection box 的 IoU、交集、component coverage、box
  coverage、bbox IoU；
- 两者 foot point 的 analysis-grid 距离；foot point 定义为 component 底部像素的
  稳健 x 中位数/最大 y 与 detection bottom-center；
- `class_compatibility`：`COMPATIBLE / INCOMPATIBLE / UNKNOWN`；标签映射仅来自冻结
  config，不从 truth 或结果回推；
- `temporal_continuity`：同一 component/detection track 的上一相邻 observation 是否
  保持该 correspondence；
- `optical_flow`：在有 2x3 `previous -> current` flow 时的 propagated component/box
  overlap；
- `depth_consistency` 与双方 `depth_cluster_id`；只有两侧有有效 depth summary 才能
  判定一致/不一致；
- evidence present/missing、score、state_reason 与最终三态。

## 三态与 one-to-one 规则

- `MATCH`：类别兼容，至少两个非空证据参与评分，核心几何证据达到预声明门槛，且
  没有强类别/几何/深度反证；组件内最佳候选与次佳候选的分差不足时不进入 MATCH；
- `NO_MATCH`：类别明确不兼容，或已有足够观测的候选同时显示强几何分离/强深度冲突；
  这不是“检测器漏检”的别名；
- `ABSTAIN`：可选证据缺失导致不确定、类别未知、候选 tie、检测被其他 component
  唯一占用，或几何证据不足。没有任何 detection 的 component 也只能是 ABSTAIN。

每帧 component/detection 采用 deterministic one-to-one assignment。冲突时按冻结
score、component id、detection id 排序；不满足 margin 的冲突全部回落为 `ABSTAIN`，
不得复用一个 detection 支持多个 component。

## 证据边界

该输出可以把原来的 `ATTRIBUTION_UNCERTAIN` 拆成候选的 `MATCH/NO_MATCH/ABSTAIN`，
但不能把 candidate `MATCH` 改写为 `A_EFFECTIVELY_COVERED`，不能把 `NO_MATCH` 改写成
pixel-level residual truth，不能驱动 risk、feedback、TTS、振动或默认 App。任何后续
要把候选 annotation 用作 residual labelability 的新证据，必须另冻与当前数据角色/身份
相匹配的评价合同，并报告覆盖、关键漏报、误关联、响应与清除等 event-level 指标。

## 可复现命令

见 [instance correspondence package README](../../scripts/research/dual_loop_segmentation_instance_correspondence/README.md)。
输出目录只允许位于 `artifacts.local/`，已有目录不覆盖；输出内固定保留输入/配置/实现
SHA256 与 `claim_ceiling`。
