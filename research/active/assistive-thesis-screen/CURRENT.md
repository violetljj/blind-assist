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
2026-10-16: a low-rank matching adapter learned on the 44 train users with frame labels only (49 k parameters, no boxes)
gives +4.9 in user-disjoint train CV but +1.1 on test by the official video-weighted metric (73.67/77.79/81.06, three
seeds; ensemble 74.07/78.35/81.54; pooling-only 74.60 single frame); user-mean gain +3.5, negative for users with
>=11 objects. Single frame still below the official best; with the allowed history it is above. Box supervision,
colour-aware matching and cohort score normalisation gave no stable gain.
Next decision: object-count augmentation for large object sets (a second registered test use), phone latency, or
context-frame selection. Laptop safety rules: docs/HOST_RESEARCH_COMPUTE.md.

Independent worker [validation](benchmark_screen/independent_20261016/RESULTS.md): paired user-cluster intervals support
the +6.25 pp local-memory and +2.06 pp pooling gains; query-competition scoring loses140/11,064.
The negative-only [training-user holdout](benchmark_screen/negative_train_holdout_20261016/RESULTS.md) completes44 users:
primary11 users/3,192 queries hit2,571→2,550, delta−0.658pp [−2.840,+0.987]; all44 delta−0.172pp [−0.933,+0.567].
Warm construction205.80→144.75s (−29.66%); extraction-plus-build decomposition saves9.73%.
Keep combined screening: the0.5pp accuracy-loss bound is unestablished. Four center-consistency failures are retained;
cached three-repeat diagnostics change0/9,216 predictions.

Cheap-consistency [successor](benchmark_screen/cheap_consistency_20261016/RESULTS.md):33 development users select
K4096; the same11-user holdout loses59/3,192 queries (80.545→78.697%, Δ−1.848pp [−3.614,−0.411]).
Warm construction46.925→34.397s saves26.70%; the three builds give identical predictions.
Retain the original combined selector: shrinking the own-neighbor pool did not preserve accuracy.
All features reused, no new test inference;8.29 minutes GPU job wall. Caches remain on the worker; owned jobs released.
