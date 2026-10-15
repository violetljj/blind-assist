# Observation-only field evidence

`build_features.py` reads only the original image receipts/pixels, PP-OCRv5
observations and predicted fields. It never opens reference product strings or
human quality labels. Every image keeps product, brand and variety rows, even
when prediction or OCR is missing. Errors are not equivalent to observed no-text.

Example from the checkout, using the existing Python environment with NumPy and
Pillow (no additional downloads):

```powershell
python research/active/assistive-thesis-screen/product_fields/evidence/build_features.py `
  --receipts artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa/image_receipts.json `
  --images artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa/images `
  --ocr artifacts.local/work/product-fields-dev-20261015/ocr/predictions.jsonl `
  --coordinate-contract artifacts.local/work/product-fields-dev-20261015/ocr/coordinate_contract.json `
  --predictions <model-predictions.jsonl> --model <model-name> `
  --output artifacts.local/work/product-fields-dev-20261015/evidence/<model-name>.jsonl
```

Optional `--second-predictions` adds token F1 agreement as a separately named
group. Empty/empty is null, not positive agreement. Existing output is never
overwritten. A `.receipt.json` records input/output hashes, runtime, missing rows,
coordinate warnings, feature names and definitions shared with this file.
The coordinate-contract sidecar is checked against each image's actual stored
dimensions, EXIF orientation and oriented dimensions before generating features.

Input predictions use `{image, fields: {product: {text}, brand: {text},
variety: {text}}, token_logprobs}`; field text may also be a string. Mean log
probabilities may be supplied per field or per generated sequence. OCR uses
`{image,width,height,status,lines:[{text,confidence,box:[[x,y],...]}]}` with native
pixel coordinates after EXIF orientation and optional source checksum. This
matches RapidOCR's path loader (`ImageOps.exif_transpose`); it is not the JPEG's
stored raster orientation. Each output row records stored/oriented dimensions
and EXIF orientation as well as
`image/model/field/prediction/features/groups/evidence`. Nulls remain null for
train-fold-only imputation downstream. The script does not assign split labels.

## Fixed features

Controls: field one-hot, empty output and token length; the unchanged 16 raw-pixel
features in the old receipts; whole-image OCR mean/median/min confidence, line
count and character count; available field/sequence mean token log probability.
All OCR nonempty lines are included, including low-confidence lines; confidence
then weights lexical support instead of deleting uncertain text.

The main field group has exactly ten numeric features:

1. Fraction of predicted tokens exactly present in OCR after Unicode NFKC,
   case-folding and punctuation splitting. No field aliases or stemming.
2. Mean best token similarity (`SequenceMatcher`, no autojunk).
3. Mean best-match similarity multiplied by that line's OCR confidence.
4. Minimum best token similarity across the predicted field.
5. Median matched line bounding-box height divided by native image height.
6. Union bounding-rectangle area divided by their enclosing rectangle area.
7. Union bounding-rectangle area divided by native image area.
8. Log(1 + mean local four-neighbour Laplacian variance).
9. Mean local luminance standard deviation.
10. Fraction of matched boxes touching the native image edge within one pixel.

Features 5–10 use only lines with a predicted-token match of at least 0.8, fixed
before observing outcomes. Luminance is EXIF-oriented native-pixel PIL `L` divided by 255;
local crops are each OCR line's axis-aligned bounding rectangle without resizing.
Geometry is disabled on OCR/oriented-image dimension mismatch or invalid coordinates.
No matched line gives null geometry, not a zero-quality observation.

Evidence traces retain each token's selected OCR token/index/similarity and each
matched native polygon. Repeated tokens may match the same OCR token: the feature
measures lexical presence, not phrase order or token multiplicity. Scattered
matches and text on a second object can support a string; they do not establish
instance ownership. Spatial compactness is only a descriptive cue. These limits
are explicit so an observed gain cannot be called grounded field reading.

`check_contract.py` checks the relevant integrity contracts on synthetic arrays;
it reads no dataset or labels. Run once after changing evidence behavior.

## Observed coordinate correction

The first end-to-end adapter check compared the stored JPEG raster (960 × 720)
with OCR output (720 × 960) and correctly disabled local geometry. Inspection
then established EXIF orientation 6 and confirmed the OCR path loader explicitly
uses `ImageOps.exif_transpose`. Across the 96 original images, five have EXIF 6
and 91 have EXIF 1. The evidence loader now applies the same exact orientation
operation, preserving original full resolution; the old 16 pixel features stay
as originally computed. This is a coordinate-contract correction, not a new OCR
run or a model improvement.

The first synthetic adapter output with its warning remains at
`artifacts.local/work/product-fields-dev-20261015/evidence/synthetic_adapter_check/features.jsonl`;
the corrected check is `features_exif_corrected.jsonl` beside it. These use a
deliberately copied OCR string to exercise adapters and are not baseline results.
The 96-image raw/EXIF/oriented metadata inventory is `image_coordinate_receipt.json`
in the evidence payload directory.
An independent pass through the actual adapter verified all 96 sidecar records
and all 995 OCR polygons with no coordinate warnings; see `coordinate_validation.json`.

## Completed observation payloads

Both finalized VLM passes have been transformed into `evidence/image_only.jsonl`
and `evidence/ocr_conditioned.jsonl` under the run payload. Each contains 288 rows
(96 images × 3 fields), 38 main control/evidence features and one separately
grouped cross-pass agreement feature. Both retain 12 empty field predictions.
Local geometry remains null for 103 image-only rows and 96 OCR-conditioned rows;
this indicates absent qualifying lexical/box evidence, not an error label.

The two extraction calls took 2.08 and 2.09 seconds of process wall time. Their
receipts retain full input/output hashes. `feature_validation.json` verifies
cardinality, unique keys, unchanged pre-evaluation script hash, feature counts,
empty preservation, input completeness, all 96 coordinate sidecar entries per
arm and symmetric agreement for all 288 field pairs. No evaluator labels or
human quality annotations were accessed in this branch. Predictive performance
is evaluated separately by the parent workflow.
