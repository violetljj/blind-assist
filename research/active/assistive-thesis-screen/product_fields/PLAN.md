# Product field evidence: stronger local baselines

2026-10-15, EXPLORE. User requested continuing with the existing product images,
running stronger OCR and visual-model baselines, and checking whether field-level
evidence predicts reading errors better. No topic commitment or app integration.

## Inputs and ownership

Use exactly the previous 96 images ordered by
`artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa/image_receipts.json`.
Keep first64 development training / last32 development evaluation. Earlier image
quality inspection does not make this fresh confirmation. Do not read protected
datasets or add images. Empty/optional-only product labels are NOT_EVALUABLE,
not unanswerability labels. Evaluator truth is separate from inference manifests.

Code belongs in this directory. Payload, caches and runtime receipts belong in
`artifacts.local/work/product-fields-dev-20261015/` on the canonical F: junction.
Do not change the other chat's object-finding experiment or unrelated dirty files.

## Bounded execution

| Work | New download ceiling | Compute ceiling |
| --- | --- | --- |
| OCR | 300 MB | 1,200 CPU command-wall seconds; GPU only by coordination |
| Local VLM | 6,000 MB including runtime dependencies | 1,800 GPU process-wall seconds; 300 CPU diagnosis seconds |
| Evaluation contracts | 20 MB | 300 CPU command-wall seconds |
| Observed evidence features | 0 MB | 300 CPU command-wall seconds |
| Integration/error prediction | 20 MB | 600 CPU command-wall seconds |

Total new payload <=6.34 GB. No API credentials/calls, paid compute, hardware,
full-model training or new photography. GPU belongs to the VLM branch; OCR uses
available CPU/ONNX while that GPU is reserved. Metadata/sparse fitting is
TASK_NOT_GPU_SUITABLE. Download time is recorded separately. Inspect one canary
before the bounded batch; resumable per-image outputs preserve failures.

Models: official PP-OCRv5 server detection/recognition; Qwen3-VL-2B-Instruct BF16
at fixed revision with available Transformers. A second fixed image+OCR VLM pass
is allowed only if the representative timing fits the same GPU budget; it is
chosen before evaluating predictions, not after seeing errors. No precision or
resolution retuning against the32 evaluation outcomes.

## Comparison fixed before outcomes

Three fields: product / brand / variety. Keep empty outputs. Run actual OCR
and VLM; OCR's truth-matching text coverage is a diagnostic, not an OCR-only
field assignment system. Main reading labels are automatic required-span
coverage proxies, with stricter token precision/exactness and relaxed matching
sensitivity. They do not reproduce the source paper's human semantic scoring
or certify absence of extra hallucinations. Inspect disagreements without
silently editing evaluation labels or tuning methods on them.

Error predictors use only observations: field identity, output empty/length,
16 existing raw-pixel quality features, whole-image OCR confidence, VLM output
log-probabilities and at most10 field OCR-support/spatial-quality features.
No truth string may be used to choose evidence boxes. Preserve original image
coordinates and source identity; lexical matches do not prove object ownership.

Fixed model groups: field/length baseline; +global image/OCR; +VLM confidence;
field text support alone; all baselines +field evidence; optional second-pass
agreement as separately costed extension. Use regularized logistic models with
identical group-held cross-validation on the64 training images to select among
C={0.1,1,10}; imputation/scaling only within each training fold. Eval remains
untouched until models and predictions are saved. If a class is absent, report
not evaluable instead of inventing AUROC.

Report error-positive AUROC/AP, accuracy-risk coverage curves and error counts
at matched60%/80% output coverage; descriptive curve points are not validated
operating thresholds. Also apply an80%-coverage threshold chosen on train-only
out-of-fold scores and report actual eval coverage. Break down brand/product/
variety and empty outputs; image-cluster uncertainty is descriptive for32 images.
Do not select the best evaluation feature set or field as a confirmed winner.

## Decision and delivery

Pre-outcome comparison clarification: while the second VLM pass was still
running and before either model's error-predictor fit/eval, add one32-dimensional
control: all28 baseline features plus the4 lexical-support features. This tests
whether the6 additional local geometry/clarity features add anything beyond
text agreement, rather than attributing lexical support gains to localization.
No extra downloads or inference; same fitting budget and training-only choices.

Continue only while evidence can resolve baseline error types, incremental
prediction value beyond simple controls, or a concrete implementation defect.
Preserve failed downloads, no-text frames, empty outputs, small denominators and
all compared arms. Stop each branch at the stated limit or answered question.
Deliver scripts, per-image/field evidence, model/input receipts, comparison tables,
representative error inspection, updated route CURRENT/RUNS and scoped Git push.

## Post-outcome orientation diagnostic, separately identified

After sealing and evaluating both planned arms, six original images were visually
inspected. Three (14955,12716,19585) have physically upside-down packaging even
after EXIF correction. This is observed input failure evidence, distinct from
the earlier metadata-coordinate correction. Within the remaining original GPU
and integration budgets, run one additional fixed image-only pass: EXIF transpose,
then lossless180-degree rotation for every one of the same96 images. Keep model,
prompt, precision and resolution fixed. New GPU process-wall ceiling300s, included
in the original1,800s total; no new download. Preserve all earlier outputs.

This is posthoc consumed-Development diagnosis, not an added unseen evaluation
arm or a corrected primary result. No per-eval-image orientation choices, prompt
retuning or further angle search. Seal new risk predictions from the64 train
images before scoring new eval outputs. Report direct rotated-read performance
and pairwise oracle union only as input-sensitivity/potential diagnostics; oracle
selection cannot be presented as an implemented improvement. Stop this diagnostic
after this full fixed pass, its focused validation and interpretation.
