# Failure Case Atlas 批量图册

`batch_album.py` 把已经物化的 Failure Atlas 帧证据渲染成可浏览的 PNG 图册、分类 contact sheet、`cases.jsonl` 和 `index.html`。它是 host-side、Development-only 的可视化 sidecar，不改变模型决策，也不驱动提醒。

每张图固定包含：原图、YOLO 框、segmentation mask、truth mask、depth、risk heatmap、previous/current/future、错误图、错误类型/置信度/事件阶段、来源与证据链。缺失的上游层会写入 `NOT_AVAILABLE` 或 `NOT_EVALUABLE` 面板；不会从相邻数据集猜测风险或事件结论。

项目索引入口：[`scripts/README.md`](../../README.md) 与 [`dual-loop README`](../../../docs/research/dual-loop/README.md)。

## 维护类别规则

类别定义和分类谓词位于同目录的 [`category_rules.json`](category_rules.json)。新增类别时，在 `categories` 数组中增加唯一 `slug`、显示 `label` 和已有 rule type 的 `rules`，不需要修改 `CATEGORY_SPECS` 或分类器代码；运行时可通过 `--category-config path/to/custom_rules.json` 使用另一份版本化配置。当前支持的 rule type 包括 `mask_relation`、`component_tag`、`bucket_or_event_token`、`event_bool_any`、`event_number_positive`、`event_token`、`event_field_token` 和 `explicit_appearance_token`。

每次输出都会复制实际使用的 `category_rules.json`，并在 `provenance.json` 中记录配置路径和 SHA-256，保证后续能够按当时规则复现历史图册。若需要一种全新的证据谓词，才需要先扩展 rule evaluator，并为其补充测试。

## 当前 320 帧扩展

在 `E:\linnan\linnan` 执行：

```powershell
$py = 'C:\Users\26442\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m scripts.research.failure_case_atlas.batch_album `
  --repo-root . `
  --frames artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-dev-rehearsal/frames.jsonl `
  --frames artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-consumed-old-blind-rehearsal/frames.jsonl `
  --atlas-components artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-v4/atlas_components.jsonl `
  --view-root artifacts.local/evidence/dual-loop-segmentation-r2-p0/canonical-view `
  --yolo-trace artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/dev/yolo_trace.jsonl `
  --yolo-trace artifacts.local/evidence/dual-loop-segmentation-candidate-utility-r0/formal/yolo_trace.jsonl `
  --atlas-result artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-v4/result.json `
  --depth-index artifacts.local/evidence/dg-srf-image-space-structural-complementarity-f0/producer-v1/depth_index.jsonl `
  --depth-maps artifacts.local/evidence/dg-srf-image-space-structural-complementarity-f0/producer-v1/depth_maps.npy `
  --output-root artifacts.local/evidence/failure-case-atlas-batch-r0/segmentation-expansion-320
```

输出根目录必须位于 `artifacts.local/`，并且默认拒绝覆盖已有目录。可重复执行时换一个新的 `--output-root`，或先由操作者确认并移除旧产物。

## 类别与证据边界

自动索引包含：漏检、误检、晚提醒、提醒无法清除、上部视场误激活、小碎片、稳定高置信错误、temporal flicker、boundary dilation、YOLO attribution ambiguity、parallel curb、shadow/texture、行人横穿、台阶和落差，以及其他/未归类。

其中漏检/误检来自当前帧的 residual truth、candidate mask 和 Atlas false-activation 记录；机制类来自已有 `mechanism_tags`；场景类来自 manifest 的 `scene_bucket` 或显式事件账本。`晚提醒`、`提醒无法清除`、事件阶段只有在通过 `--event-ledger` 提供事件级字段时才会归类和显示。`shadow/texture` 只有显式 appearance label 才会归类，当前 Atlas 的 `NOT_EVALUABLE_NO_APPEARANCE_LABEL` 不算证据。

当前扩展有 DG-SRF depth producer，可作为 `AVAILABLE_DIAGNOSTIC_ONLY` 诊断层显示。当前 320 帧没有同 cohort 的冻结 risk-map；因此 risk heatmap 显示 `NOT_AVAILABLE`。这两个字段都不构成安全、生产或提醒授权。

## 输出结构

```text
index.html                 # 按类别浏览，附状态和证据边界
result.json                # 运行摘要、计数、状态
provenance.json            # 输入路径、哈希和 renderer/schema 身份
cases.jsonl                # 每帧一行，可供后处理或筛选
category_index.json        # 类别计数、状态、contact sheet 路径
category_rules.json        # 本次运行实际使用的类别规则快照
figures/<case>.png         # 每个失败帧的 12-panel 图
thumbnails/<case>.jpg
categories/<slug>/contact_sheet.jpg
```

状态：`development`

## 稳定 Interface

公开入口、输入不变量和失败模式以本目录脚本帮助和专项协议为准；跨域调用不得依赖私有 Implementation。

## 安全边界

本模块不产生默认 App、生产、安全或 unseen confirmation authority；结果按当前协议声明的 Development/diagnostic 角色使用。

## 停止条件

最小判别实验完成、输入权威缺失、预算耗尽或重复失败时停止当前 evidence version，并保持最小 failure scope。
