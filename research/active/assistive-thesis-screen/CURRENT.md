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

ORBIT thesis direction (details: [results](benchmark_screen/RESULTS.md), [story](benchmark_screen/THESIS_STORY.md),
[registered rules](benchmark_screen/PLAN.md)). Test = 17 official test users, official 50-task protocol, each round
registered before one run. Annotation-free personal part memory on frozen DINOv2-S/224: 72.54/76.53/80.44%
(single frame / EMA half-life 30 frames / from first frame); whole frame 66.29%; Swiss DINO 61.63% and PerSAM 20.64% under
the same annotation-free condition; best official meta-trained model 75.38% (single frame).
2026-10-16: learned matching on the 44 train users (frame labels only) gave +4.9 in train CV but +1.1 on test (first
test use). Second registered test use: a 25 k-parameter localizer trained on train-split boxes says where the filmed
object is, and only unsure frames get a zoomed second look (crop from the GPU frame, same memory) - single frame
76.84 +- 0.53% vs the official best 75.38 +- 0.51% (1 s window 79.55, from first frame 81.82), 1.42 forwards per
frame on average (8.6 GMACs; uniform 448 is 31.7). Dev oracle: where to look, not zoom size, is the bottleneck.
Storyline, figures and claim limits: benchmark_screen/THESIS_STORY.md. Next candidates: phone latency, accuracy-compute
and per-object-count figures, context-frame selection. Laptop safety rules: docs/HOST_RESEARCH_COMPUTE.md.
