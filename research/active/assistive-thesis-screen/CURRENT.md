# Assistive thesis screen

所属：**毕设线**，不是避障线（见[两条工作线](../../../docs/WORK_LINES.md)）。

2026-10-15: EXPLORE screen and product-field follow-up complete. Initial four
small-data experiments and one source contract are in the [screen](README.md).
The follow-up ran PP-OCRv5 server and Qwen3-VL-2B on the existing96 product images:
[field results](product_fields/RESULTS_20261015.md), [scope](product_fields/PLAN.md).

[Scope and budget](PLAN.md) · [Runs](RUNS.md). Keep existing object finding in its
[own route](../assistive-object-finding/CURRENT.md). New candidate priority:
task-conditioned product evidence/reading, then route-map rejection. Crop-based
detail refinement and response compression did not show consistent benefit.
Object-state negative evidence remains a hypothesis with an inaccessible data
source and an author-code indexing defect; no model result.

On32 evaluation images/76 annotated fields, image-only reading covers39 required
field phrases; adding OCR covers38. Field evidence improves image-only error
AUROC0.617->0.676 against all global/confidence controls, but the simple field/
length control already reaches0.660 and errors at61/76 acceptance stay27.
Two-pass agreement helps the OCR-conditioned arm (AUROC0.715->0.787;27->25 errors
at61/76), but hurts the image-only arm. This is not a stable general improvement.

Six-image model-assisted review found field-role/granularity confusion and
physically upside-down packages. In14/37 image-only proxy errors, required text
appears in another output field. A separately recorded posthoc all-image180-degree
pass covers34/76 fields, below the original39; no blanket rotation improvement.
Next decision: separate literal text errors, field assignment and semantic scoring
before promoting a field-evidence method. No thesis commitment; actual recapture
still needs paired subsequent views. All reused data remain Development, with no
deployment or user-benefit claim. Payload stays in artifacts.local; compute released.

2026-10-15 later: a [quantifiable-benchmark screen](benchmark_screen/RESULTS.md) (public benchmark + released
baselines + standard metrics) ranks ORBIT teachable object recognition first: frozen full-frame prototypes reach
70–78% versus the official meta-trained 85.67% on validation, while ground-truth object crops reach 88–94%, so
personalised localization in clutter is the measured headroom; naive 14-crop grids recover part (best 83.5%).
VizWiz-Classification is a secondary robustness benchmark (best modern zero-shot 55.1%). Next: ORBIT test users.
