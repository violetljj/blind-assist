# Unseen-Location Evidence Router

Status: `MMS_VPR_COARSE_GPS_SOURCE_REJECTED / ROUTER_NOT_EVALUABLE / TEST_UNOPENED`

## Frozen Development question

Can observation-quality-conditioned evidence routing improve reference-conditioned
retrieval of held-out locations over the strongest static learned fusion baseline?

This is a mechanism experiment, not named-POI grounding. The model receives no
goal string or location label. Given one query observation and a frozen local
candidate set, it ranks candidate locations using only evidence available at
inference time:

- query-to-reference visual match;
- query OCR-to-reference OCR match, when OCR was actually run on both images;
- coarse query-to-candidate geographic relation;
- measured evidence quality and missingness.

It outputs candidate scores, Top-1 location, and confidence. It does not output a
box or mask, maintain video state, find an entrance, or issue navigation advice.

## Leakage boundary

Complete location identities are assigned to exactly one of train, development,
or test. Development and test locations may supply reference images at inference,
but their identities never appear in router training. Within every split,
reference and query images come from disjoint capture groups. Frames derived from
the same source video or social-media event stay in one group.

The following are forbidden model inputs:

- location ID, class index, or map code;
- the dataset's manually verified store/sign list;
- evaluator target identity;
- text copied from annotations instead of OCR run on the actual image;
- any feature selected after observing development/test outcomes.

MMS-VPR's official per-class random split is not used because every location
appears in both training and test. The new split supports a narrower claim:
held-out-location, reference-conditioned retrieval within one Chengdu commercial
district. It does not support unseen-city, named-POI, open-world, or product claims.

## Frozen arms

1. visual evidence only;
2. OCR evidence only;
3. coarse-GPS evidence only;
4. fixed equal fusion over available evidence;
5. static learned fusion without quality inputs;
6. quality-conditioned dynamic Evidence Router.

The router must be compared primarily with arm 5. Any visual backbone, OCR
provider, candidate construction, feature normalization, and train/dev/test
manifest are shared by all arms.

## Primary outcomes and decision

Report Top-1, Top-5, local-candidate Top-1, and local-hard-negative error, plus
day/night, OCR present/missing/correct/incorrect, blur, viewpoint, candidate-count,
and reference-view-gap strata when those fields are genuinely observable.

The router advances only if it improves absolute held-out-location Top-1 by at
least 8 percentage points over the strongest static learned fusion in both frozen
seeds, while not increasing local-hard-negative error. A smaller or single-seed
gain closes this scorer without backbone, threshold, weight, quality-feature, or
fusion sweeps on the consumed cohort.

## Development canary terminal

The bounded real-image canary completed feature extraction on 1,060 images, but
rejected MMS-VPR as the source for the frozen coarse-GPS candidate task. Source
video coordinates are not frame-aligned: GPS-to-target distance was about 363 m
at the median. Target coverage for GPS-derived candidates was only 9.4%/37.2%
for train/development at K=8 and 18.0%/63.6% at K=16.

Only 53 train and 45 development queries were therefore usable by the K=8
selection experiment. On that diagnostic subset, the dynamic router gained
0.00 and 2.22 percentage points over static learned fusion in the two frozen
seeds. These numbers neither advance nor reject the router because candidate
coverage destroyed the training/evaluation denominator before the scorer.

Do not repair this by injecting the evaluator location coordinate, forcing the
target into the candidate set, increasing K to the full split after seeing the
result, or treating manual location text as OCR. A successor needs frame-aligned
coarse position, or separate authorization for a no-GPS global-retrieval task.
Exact evidence is in `development_canary_result_v1.json`.

Before training, run the dataset admission audit:

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  research/active/unseen-location-router/build_manifest.py `
  --images-root artifacts.local/datasets/unseen-location-router/mms-vpr/Images `
  --graph-root "artifacts.local/datasets/unseen-location-router/mms-vpr/Graph Structure" `
  --output artifacts.local/datasets/unseen-location-router/mms-vpr/split_manifest.json `
  --audit artifacts.local/evidence/unseen-location-router/data_admission.json
```

Focused contract tests:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  research/active/unseen-location-router -p 'test_*.py'
```

The first bounded real-image canary uses at most two reference and four query
capture groups per train/development location. Test pixels remain unopened:

```powershell
$env:PYTHONPATH = "artifacts.local/runtime/semantic-anchor-v1/site-packages"
E:\codex-tools\bin\blindassist-python.cmd `
  research/active/unseen-location-router/export_features.py `
  --manifest artifacts.local/datasets/unseen-location-router/mms-vpr/split_manifest.json `
  --images-root artifacts.local/datasets/unseen-location-router/mms-vpr/Images `
  --texts-root artifacts.local/datasets/unseen-location-router/mms-vpr/Texts `
  --backbone artifacts.local/models/p1_a2_dinov2_small_ed25f3a `
  --database artifacts.local/datasets/unseen-location-router/mms-vpr/development_features_fast.sqlite `
  --receipt artifacts.local/evidence/unseen-location-router/feature_receipt_fast.json `
  --gallery-per-location 2 --query-per-location 4 --ocr-workers 2 --dino-batch-size 16

E:\codex-tools\bin\blindassist-python.cmd `
  research/active/unseen-location-router/run_development.py `
  --database artifacts.local/datasets/unseen-location-router/mms-vpr/development_features_fast.sqlite `
  --manifest artifacts.local/datasets/unseen-location-router/mms-vpr/split_manifest.json `
  --texts-root artifacts.local/datasets/unseen-location-router/mms-vpr/Texts `
  --output artifacts.local/evidence/unseen-location-router/development_result.json
```
