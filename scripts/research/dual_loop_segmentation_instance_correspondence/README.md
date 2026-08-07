# dual_loop_segmentation_instance_correspondence

`DUAL_LOOP_SEGMENTATION_INSTANCE_CORRESPONDENCE_CANDIDATE_R0` 为 Failure Atlas
补齐 instance-level 候选关联。它把同一 materialized observation 中的
segmentation mask connected component 与 YOLO detection 组成候选 pair，并附上
component track、detection track、semantic class、depth cluster 和可用的 optical-flow
证据。

状态只表示候选证据强度：

- `MATCH`：类别兼容且几何证据足够，且没有未解决的强冲突；
- `NO_MATCH`：存在明确类别冲突或足够强的几何/深度反证；
- `ABSTAIN`：候选不唯一、缺少输入、类别未知或证据不足。

缺少 depth cluster、external detector track 或 flow sidecar 不会被静默当作负证据；
对应字段为 `null`，并进入 `missing_evidence`。检测器漏检也只能产生 component-level
`ABSTAIN`，不能写成“已确认未覆盖”。每个 component 最多选择一个 detection，冲突
候选会回落为 `ABSTAIN`。

## Stable interface

在仓库根目录运行：

```powershell
python -m scripts.research.dual_loop_segmentation_instance_correspondence.batch `
  --repo-root . `
  --config configs/dual_loop_segmentation_instance_correspondence_v1/default.json `
  --frames artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-dev-rehearsal/frames.jsonl `
  --components artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-dev-rehearsal/components.jsonl `
  --yolo-trace artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/dev/yolo_trace.jsonl `
  --output-root artifacts.local/evidence/dual-loop-segmentation-instance-correspondence-r0/dev-200
```

可重复传入多个 `--frames`、`--components` 和 `--yolo-trace`；输入按
`source_id/frame_id/image_sha256` 精确配对。可选 sidecar：

```powershell
  --depth-clusters <depth-clusters.jsonl> `
  --optical-flow <motion-trace.jsonl>
```

`depth-clusters.jsonl` 的每行至少包含 `source_id`、`frame_id`、`cluster_id` 和
`median_depth`，并提供 `bbox_xyxy`（analysis-grid 坐标）或 `mask_packed` + `shape`。
`optical-flow` 使用现有的 causal `matrix_previous_to_current` 2x3 affine 格式。

## Outputs

输出目录必须位于 `artifacts.local/` 且不能覆盖已有目录：

- `pair_evidence.jsonl`：每个 component/detection pair 的全部指标、缺失证据和三态；
- `component_annotations.jsonl`：每个 component 的最终 one-to-one 候选结果；
- `detection_annotations.jsonl`：每个 detection 的反向候选摘要；
- `summary.json`：数量、状态、evidence availability 和 conflict 统计；
- `provenance.json`：输入、配置和实现 hash。

这是 `DEVELOPMENT_CANDIDATE_ANNOTATION_ONLY` 结果：它不修改 residual truth，不接
Android/risk/feedback/提醒，不把 `MATCH` 直接变成 `A_EFFECTIVELY_COVERED`，也不改变
默认 YOLO App 或既有 Failure Atlas 的历史终态。

状态：`development`

## 稳定 Interface

公开入口、输入不变量和失败模式以本目录脚本帮助和专项协议为准；跨域调用不得依赖私有 Implementation。

## 输出

只写入 artifacts.local/ 下的明确证据目录；不写仓库根目录或正式 App 资产。

## 安全边界

本模块不产生默认 App、生产、安全或 unseen confirmation authority；结果按当前协议声明的 Development/diagnostic 角色使用。

## 停止条件

最小判别实验完成、输入权威缺失、预算耗尽或重复失败时停止当前 evidence version，并保持最小 failure scope。
