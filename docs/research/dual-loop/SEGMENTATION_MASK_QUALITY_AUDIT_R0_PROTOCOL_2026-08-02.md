# Segmentation mask quality audit R0

状态：`IMPLEMENTED / READ_ONLY / PROPOSAL_ONLY / NOT_YET_RUN`

## 目标边界

本路线只检查既有 segmentation mask 的质量，不重新大规模画 mask，也不
把模型输出写回原标签。每一帧绑定 RGB、原 mask、frame identity 和 hash；模型
mask 只能作为隐藏于初次视觉复核之外的 `proposal` sidecar。若后续接受 proposal，
必须另建 derived view，并保留原文件和原 hash。

这是一条数据质量门禁，不是安全、产品或默认 App 授权。`PASS` 只表示该帧通过
声明的完整性门、已完成视觉复核且没有未解决的 QA reason code；它不证明模型效果。

## 固定语义与重编码边界

当前 audit config 绑定：

```text
0 walkable
1 blocking_obstacle
2 boundary_level_change
3 unknown_nonwalkable
```

旧 canonical mask 的 obstacle/boundary ID 顺序相反时，必须先建立带 source hash
和 derived hash 的显式重编码视图（`old 1 -> new 2`、`old 2 -> new 1`）。本 audit
不会根据 proposal、像素分布或文件名猜测并覆盖这个映射。未声明 class order、label
space 或 source-to-expected mapping 的帧不能成为干净 `PASS`。

`unknown_nonwalkable` 不是安全或可通行；任何把它映射到 walkable 的合同都
`INVALID`。视觉审阅中的 UNKNOWN/abstention 也不能被当作 walkable。

## 三态输出

| 状态 | 规则 |
| --- | --- |
| `PASS` | RGB/mask 同帧、可解码、hash/尺寸/ID/映射/resize policy 均通过；已提供该帧视觉 `PASS`；没有剩余 reason code。 |
| `REVIEW` | 原始证据可保留，但需要 RGB 回看或相邻帧判断，例如边界漂移、远处填充、地面不连续、细杆/树枝漏标、UNKNOWN/walkable 冲突、proposal 分歧或标签闪烁。 |
| `INVALID` | 无法安全解释或视觉确认错误：缺文件、错帧、hash/尺寸不符、未知 ID、ID 顺序/映射未绑定或错位、非 nearest resize、空 mask 等。 |

没有 `visual_review.jsonl` 时，默认每帧至少得到
`MANUAL_REVIEW_REQUIRED`，因此技术 PASS 不会伪装成质量 PASS。

## 检查项与 reason codes

### `INVALID`

- `RGB_MISSING_OR_UNREADABLE`, `MASK_MISSING_OR_UNREADABLE`：文件缺失或无法解码；
- `RGB_MASK_FRAME_KEY_MISMATCH`, `RGB_MASK_DIMENSION_MISMATCH`, `MASK_SHAPE_MISMATCH`：
  RGB/mask 错帧或几何合同不符；
- `RGB_HASH_MISMATCH`, `MASK_HASH_MISMATCH`：manifest 与实际 bytes 不一致；
- `RGB_HASH_UNVERIFIED`, `MASK_HASH_UNVERIFIED`：没有把原始 RGB/mask bytes 绑定到 manifest hash；
- `CLASS_ID_ORDER_MISMATCH`, `CLASS_ID_ORDER_UNVERIFIED`,
  `CLASS_ID_MAPPING_MISMATCH`, `CLASS_ID_MAPPING_UNVERIFIED`,
  `LABEL_SPACE_UNVERIFIED`：类别 ID 错位或没有可核验的 ID 合同；
- `MASK_CLASS_ID_OUT_OF_RANGE`, `MASK_NOT_2D_INTEGER`, `MASK_EMPTY`：未知 ID、形状/类型
  错误或确实没有像素；
- `MASK_NON_NEAREST_RESIZE`, `MASK_RESIZE_INTERPOLATION_CONTAMINATION`,
  `MASK_RESIZE_HISTORY_UNVERIFIED`, `MASK_ALPHA_CONTENT_REQUIRES_EXPLICIT_POLICY`：
  resize/interpolation/alpha 可能污染 categorical label；
- `RGB_MASK_GEOMETRY_UNVERIFIED`：固定尺寸 mask 与 RGB 不同且没有可核验的几何变换；
- `SOURCE_MAPPING_MISSING`, `SOURCE_MAPPING_UNSAFE_UNKNOWN_TO_WALKABLE`：source mask
  重编码缺失或把 UNKNOWN 映射为 walkable；
- `MANUAL_REVIEW_INVALID`, `MANUAL_REVIEW_MISSING_REASON`, `MANUAL_REVIEW_ORPHAN_ID`：复核回执本身不合约。

### `REVIEW`

- `MASK_CONSTANT_SINGLE_CLASS`, `MASK_ALL_WALKABLE`, `MASK_ALL_UNKNOWN`：均匀/塌缩信号，
  需要结合 RGB，不能自动判错；
- `BOUNDARY_DRIFT_SUSPECTED`：边界区域与 proposal 明显不一致；
- `OBSTACLE_BOUNDARY_SWAP_SUSPECTED`：proposal 在交换 obstacle/boundary 后更一致，
  只触发复核，不自动换 ID；
- `FAR_REGION_OVERFILL_SUSPECTED`、`WALKABLE_DISCONTINUITY_SUSPECTED`：远处错误填充
  或地面区域断裂的视觉复核候选；
- `THIN_OBJECT_OR_BRANCH_MISSED_SUSPECTED`：细杆、树枝、小障碍 proposal 与原 mask
  的高风险差异候选；
- `UNKNOWN_AS_WALKABLE_SUSPECTED`, `UNKNOWN_WALKABLE_SEMANTIC_CONFLICT`：UNKNOWN 与
  walkable 语义冲突，必须看 RGB/原标签；
- `ADJACENT_LABEL_FLICKER_SUSPECTED`, `TEMPORAL_ADJACENCY_UNVERIFIED`：相邻帧变化大但
  RGB 变化小，或没有真正相邻帧可核验；
- `RGB_MASK_FRAME_KEY_UNVERIFIED`：配对 key 缺失，需按 RGB 与原 mask 人工核对；
- `PROPOSAL_INVALID`, `PROPOSAL_DISAGREEMENT_REQUIRES_REVIEW`：proposal 无法使用或与
  原标签分歧；不改变原标签；
- `MANUAL_REVIEW_REQUIRED`, `MANUAL_REVIEW_MARKED`：视觉复核尚未完成或明确要求回看。

## 最小运行与证据

```powershell
& E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.dual_loop_segmentation_mask_quality_audit.audit `
  --manifest artifacts.local/evidence/<bundle>/mask_qa_manifest.jsonl `
  --base-root . `
  --config configs/dual_loop_segmentation_mask_quality_audit/default.json `
  --output-root artifacts.local/evidence/<bundle>/mask-quality-audit-r0
```

输出为 append-only sidecar：`summary.json`、`frame_results.jsonl` 和
`review_queue.json`。output root 非空时拒绝覆盖。建议在写入审阅回执前保存原始
manifest 和原 mask 的 hash；工具的 `original_label_immutable=true` 与
`proposal_replacement_applied=false` 也会写入报告。

## 停止条件

任何 `INVALID` 保持 fail-closed，不通过修图、换 interpolation、改类别名或把 proposal
写回原标签来消除。`REVIEW` 必须逐帧回看 RGB 与原 mask；只有清除 reason code 后才能
进入训练/真值评价。这个 R0 不训练模型、不改变 Android/YOLO/风险链、不修改默认 App。
