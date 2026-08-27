# Unseen-Location Evidence Router

Status: `MSLS_SOURCE_ADMITTED / ROUTER_DEVELOPMENT_GATE_NOT_MET / TEST_UNOPENED`

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

MMS-VPR's official per-class random split was not used in the rejected source
canary because every location appears in both training and test. The MSLS
successor instead uses the official city separation: Copenhagen and San
Francisco are unseen during training. This supports only a bounded unseen-city,
reference-conditioned Development result; it does not support universal VPR,
named-POI, open-world, or product claims.

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

## MSLS successor terminal

The successor kept the retrieval question, DINO/OCR providers, learned arms,
seeds, primary static-fusion comparator, and `+8pp` advancement gate fixed.  It
changed only the evidence source to Mapillary Street-Level Sequences (MSLS).
The official city split supplies 22 training cities, Copenhagen and San
Francisco as unseen Development cities, and six unopened test cities.

The metadata-only admission canary passed before pixel extraction. Query GPS was
present for 100% of train and Development frames. With frozen 100 m grid-cell
coarsening, target coverage was 99.96%/100% at K=8 and 99.99%/100% at K=16 for
train/Development. Exact evidence is in `msls_source_canary_v1.json`.

The bounded Development run then used 11,587 real images: 9,731 train and 1,856
Development images from Copenhagen and San Francisco. It read no test images.
Development K=8 coverage was 826/826. The primary result was:

- static learned fusion: 79.66% and 80.27% Top-1 for seeds 1701 and 2701;
- quality-conditioned Router: 79.18% and 76.15%;
- absolute Router gain: -0.48pp and -4.12pp;
- local-hard-negative error increased in both seeds.

The Router therefore missed the `+8pp` gate in both seeds and is closed on this
consumed Development cohort. It must not be rescued with weight, quality-feature,
threshold, backbone, candidate, or seed sweeps. GPS-only already scored 79.66%,
and the Router assigned 93.7% to 99.7% mean target-candidate weight to geography;
the learned dynamic mechanism did not add useful evidence routing. Exact results
are in `msls_development_result_v1.json`.

The run supports a bounded unseen-city Development result, not a universal VPR,
named-POI, grounding, navigation, product, or safety claim. Only five Development
queries were labelled night, and OCR has no independent transcription truth.
The test cities remain unopened.

To reproduce the metadata admission canary, extract official `metadata.zip` under
`artifacts.local/datasets/unseen-location-router/msls/`, then run:

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  research/active/unseen-location-router/build_msls_canary.py `
  --dataset-root artifacts.local/datasets/unseen-location-router/msls `
  --output artifacts.local/evidence/unseen-location-router/msls_source_canary.json
```

The canary reads only train/validation metadata and admits the source only when
query frame GPS is 100%, K=8 target coverage is at least 90%, and K=16 target
coverage is at least 99%.  Test metadata and test pixels remain unopened.  To
prevent exact MSLS GPS from becoming a label shortcut, candidate construction
uses the centre of a frozen 100 m metric grid cell; official database positives
within 10 m define evaluator identity.  If any gate fails, stop before DINO,
OCR, fusion, or Router training.

Focused contract tests:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  research/active/unseen-location-router -p 'test_*.py'
```

The first bounded real-image canary uses at most two reference and four query
capture groups per train/development location. Test pixels remain unopened:

```powershell
$env:PYTHONPATH = "artifacts.local/runtime/ocr-gpu-ort;artifacts.local/runtime/semantic-anchor-v1/site-packages"
E:\codex-tools\bin\blindassist-python.cmd `
  research/active/unseen-location-router/export_features.py `
  --manifest artifacts.local/datasets/unseen-location-router/msls/development_manifest_v3.json `
  --images-root artifacts.local/datasets/unseen-location-router/msls `
  --backbone artifacts.local/models/p1_a2_dinov2_small_ed25f3a `
  --database artifacts.local/datasets/unseen-location-router/msls/development_features_v3.sqlite `
  --receipt artifacts.local/evidence/unseen-location-router/msls_development_feature_receipt_v3.json `
  --gallery-per-location 2 --query-per-location 4 --ocr-workers 1 --ocr-use-cuda --dino-batch-size 16

E:\codex-tools\bin\blindassist-python.cmd `
  research/active/unseen-location-router/run_development.py `
  --database artifacts.local/datasets/unseen-location-router/msls/development_features_v3.sqlite `
  --manifest artifacts.local/datasets/unseen-location-router/msls/development_manifest_v3.json `
  --output artifacts.local/evidence/unseen-location-router/msls_development_result_v1.json
```
