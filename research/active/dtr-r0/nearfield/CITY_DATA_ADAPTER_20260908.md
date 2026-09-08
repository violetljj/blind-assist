# City data adapter

Independent data adaptation for the accepted City 500-group collection. It
does not modify frozen G13 helpers, models, training receipts or UE scene files.
The concurrent task `扩展 HEAD-only 障碍样本` owns scene/generator work; this
change is confined to new adapter, materializer, tests and this document.

## Interface

`tools/build_city_training_cache.py` validates the split's source hashes,
accepted BODY/HEAD 3 m contract, row identity, complete sample coverage, group
isolation and declared region assignments. It hashes/decode-checks RGB and
hashes native masks, then writes a fresh cache with a final PASS manifest.
Failures produce `failure.json`, not a completed manifest. Original captures
and split files are not edited.

The cache separates:

- `model/train/rgb.npy` and `model/test/rgb.npy`: uint8 NHWC 144x256 RGB;
- `supervision/train.json` and `supervision/train/`: binary near and pooled support;
- `evaluator/test.json` and `evaluator/test/`: test labels, evaluator only;
- `evaluator/*-distance.json`: unchanged static distance evidence, not near labels;
- `manifest.json`: hashes, sample identities, preprocessing and source receipts.

No validation subset is invented. The current split has TRAIN750/TEST750,
with all three variants and neighbouring poses remaining within their region.
The builder is not tied to a three/four-variant count, but future inputs still
need a compatible, validated split and visible-support contract.

`city_data.CityRGBDataset(cache, split)` returns only `rgb` and `sample_index`.
RGB is full-frame Pillow BOX downsampled from 640x360, CHW float32 in [0,1].
Pass only `batch['rgb']` to D; D performs ImageNet normalization internally,
so the adapter must not normalize twice. Pose, depth, asset, group and distance
metadata are never model tensor inputs.

`CitySupervisedDataset(cache, 'train')` additionally returns `near` (float32
BODY/HEAD binary targets) and `support` (int8 2x18x32). It rejects TEST access
before reading any labels, validates cache hashes/types/values and refuses
supervision paths outside `supervision/`. RGB paths are restricted to `model/`.
The test RGB reader does not open supervision/evaluator files. This is a
training API boundary, not an operating-system access-control sandbox.

## Label handling

Near targets are preserved exactly from native verification. They are not
inferred from asset category, center placement, distance state or downsampled
support. The original three-positive-native-pixel threshold is checked before
conversion. Zero means no observed positive in the query, not certified free
space. City support covers all visible scene surfaces, which differs from the
old box/bar-filtered G13 supervision.

Each 20x20 native support cell maps to one 18x32 output cell:

1. Any positive pixel produces +1, including single-pixel traces.
2. Otherwise, any UNKNOWN pixel produces -1.
3. Otherwise the cell is known negative (0).

`pixel_support_bce` computes positive/negative class-balanced BCE over known
pixels only. UNKNOWN pixels have zero gradient; an all-UNKNOWN batch has zero
loss and gradient. Do not feed these masks into the old frozen support loss.

Example integration, without running an optimizer:

```python
from torch.utils.data import DataLoader
import torch.nn.functional as F
from city_data import CitySupervisedDataset, pixel_support_bce

batch = next(iter(DataLoader(CitySupervisedDataset(cache, 'train'), batch_size=8)))
near_logits, support_logits = model(batch['rgb'].cuda())
loss = F.binary_cross_entropy_with_logits(near_logits, batch['near'].cuda())
loss = loss + 0.25 * pixel_support_bce(support_logits, batch['support'].cuda())
```

WARNING/DANGER are retained on the evaluator side; this adapter does not create
a new model head. Approaching remains UNKNOWN/untrained. HEAD-only and HEAD
DANGER coverage gaps from the data audit remain, and no model fit is performed.

## Actual conversion

The worker converts all 1,500 frames with CUDA mask pooling in 15.088 s
(17.794 s complete initial build job). The cache totals 168,482,880
bytes. Both 750-frame partitions retain exactly 135 BODY and 15 HEAD positives;
every near target matches the native report.

A real train batch has RGB `[8,3,144,256]`, near `[8,2]` and support
`[8,2,18,32]`, including -1/0/1. Real TEST batches expose only RGB and sample
identity. A random-logit backward smoke check is finite with zero UNKNOWN
gradient; this is not a model fit or a trained-model result.

The original build snapshot is retained. Additional final source checks for
world contract, region assignment and loader values/boundaries are verified
against that immutable cache; the cache is not relabeled as built by a later
source version. Raw cache stays on the worker; thin manifests/load receipts
remain under `artifacts.local/nearfield/city-pcg-20260908/worker-scale500-v1/`.

The initial 10 focused tests pass with actual CUDA pooling and gradient checks.
After adding path-boundary checks, all seven affected dataset tests pass,
including two new redirected-label/RGB tests using valid hashes. Pool/loss code
was unchanged and its successful tests were not repeated. Source tests are in
`test_city_data.py`; synthetic fixtures are retained under
`artifacts.local/tests/city-data/`. No UE or model optimizer ran for validation.
