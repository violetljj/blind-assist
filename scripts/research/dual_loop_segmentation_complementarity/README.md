# YOLO + semantic segmentation image-space complementarity

状态：`development` / `DEVELOPMENT_ONLY` / `NO_EFFECT_AUTHORITY`

## 研究问题与版本

`DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R1` 只回答：在同一 RGB frame 和同一
YOLO box union 上，固定 semantic-segmentation reference 是否产生可计算的、未被
YOLO 覆盖的 image-space class regions，以及这些区域的时间稳定性和主机成本。

本轮使用用户明确授权的固定 Development 诊断，不把 burned Shiraz source 当作
held-out、Confirmation 或泛化证据。它不读取中央阻塞 Agent 标签、risk、feedback
或 event 字段，不产生可通行性、风险、提醒、安全、Android 或生产结论。

## 稳定 Interface

从仓库根目录运行：

```powershell
& E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts\research\dual_loop_segmentation_complementarity\complementarity.py `
  --manifest artifacts.local\evidence\dual-loop-r1-unseen-natural-event-r0\rank2-shiraz\input-10hz-r1\manifest.jsonl `
  --trace artifacts.local\evidence\dual-loop-r1-unseen-natural-event-r0\rank2-shiraz\device-r1\baseline-output\trace.jsonl `
  --model artifacts.local\evidence\segmentation-candidate\sanpo-v3-pretrained-weighted-best-int8-20260713.tflite `
  --output artifacts.local\evidence\dual-loop-segmentation-complementarity-r1\report.json `
  --frames-output artifacts.local\evidence\dual-loop-segmentation-complementarity-r1\frames.jsonl `
  --threads 2
```

Inputs must have exact frame identity and image SHA matches. The runner uses every YOLO
rectangle without confidence/NMS/risk filtering, projects boxes to the model output grid
with clipped normalized coordinates, and uses raw segmentation argmax class masks. Missing,
duplicate, reordered, or mismatched identities fail closed; no interpolation or nearest-frame
repair is allowed.

## 输出

All outputs stay under `artifacts.local/`:

- `report.json`: frozen contract, input/model hashes, pairing, class-wise uncovered fractions,
  geometric union increment, temporal IoU/component summaries, runtime and stop checks;
- `frames.jsonl`: one paired-frame descriptive row, with no risk/feedback/event fields;
- `progress.json`: bounded progress receipt for long host execution.
- `validation.json`: independent recomputation receipt for frame count, ordering, class partition,
  union arithmetic, forbidden fields, and input hashes.

The four argmax classes remain separate: `walkable`, `boundary_step_curb`, `obstacle`, and
`unknown_nonwalkable`. Because the union of all four argmax masks covers the analysis grid by
construction, the primary complementary interpretation is class-specific
`uncovered_fraction`; `union_increment` is reported transparently as the geometric union
quantity and is not called obstacle or risk discovery.

## 安全边界

- Observation unit is `source_id + frame_id + image_sha256`; frames are repeated observations,
  not independent samples.
- The current input is one previously consumed Development session (`burned`), so all results
  are mechanism diagnostics only.
- No Android assets, production model, feedback path, default behavior, or safety authority is
  modified.
- A non-zero uncovered region is not a traversability, obstacle, event, or risk truth.
- This runner does not select among models or tune thresholds; the declared reference is fixed.

## 停止条件

Stop the current evidence version with `NOT_EVALUABLE`/failure if model interface or finite
values fail, image identity pairing fails, any image hash or dimensions mismatch, or the
segmentation output collapses to one class across the evaluated input. These conditions close
only this candidate/evidence version, not the segmentation research question.

## 假设与规则质疑

The estimand is image-space only and uses session-first summaries. No p-value, event gate,
effect gate, or frame-independent uncertainty claim is emitted. If later uncertainty is
needed, source/session clustering must be retained.

## 失败资产复用

Reports and frame rows may be reused as Development diagnostics, regression fixtures, or
candidate failure records. They must not be relabeled as held-out confirmation, risk truth,
or production evidence.
