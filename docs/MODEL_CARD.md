# BlindAssist default model card

Status: current

Last reviewed: 2026-08-12

This card documents the model packaged in the default public Android App. It is
an identity, intended-use, licensing, and limitation record—not a safety or
accuracy certification.

## Model identity

| Field | Value |
| --- | --- |
| Asset | `app/src/main/assets/yolo11n_fp16_320.tflite` |
| Role | On-device object detector for the default prototype |
| Upstream family | Ultralytics YOLO11n, COCO pretrained |
| Input contract | One `FLOAT32` tensor with shape `[1, 320, 320, 3]` |
| Output contract | One `FLOAT32` YOLO detection tensor validated by `TfliteYoloDetector` |
| Size | `5,359,428` bytes |
| SHA-256 | `00EDB41A528B0A7E709C4AF8CE3E685491492C4539274804E5CFC17A1A867CD2` |
| Runtime | LiteRT/TFLite on the Android device |

The detector is packaged with `app/src/main/assets/coco_labels.txt`
(`621` bytes; SHA-256
`BD17F1EE35D5F3C862A4894605855ABBB9DDA4B0621FDB0AC4C2C8C7BB7E730A`) as
class-name metadata. No independent ownership or license claim is made for the
upstream class names.

The machine-readable identity is
[`configs/public_release_assets.json`](../configs/public_release_assets.json).
CI recomputes the packaged asset size and SHA-256 and fails if this record,
the notice, or the payload drifts independently.

## Intended use

The detector provides object-class, confidence, and bounding-box evidence to
the prototype's deterministic risk and feedback layers. It is intended for
open engineering, accessibility exploration, tests, demonstrations, and
evidence-bounded research.

It is not intended to:

- replace a white cane, guide dog, mobility training, or human judgment;
- certify that a route is safe or clear;
- infer exact physical distance from a monocular bounding box;
- establish performance for an untested device, camera, population, or scene;
- provide biometric identification, surveillance, or face recognition.

## Known limitations

- COCO classes do not cover every obstacle or traversability condition.
- Small, occluded, unusual, low-light, reflective, transparent, or
  out-of-distribution objects may be missed or misclassified.
- A detected object is not the same as an unsafe event, and an absent detection
  is not evidence of safety.
- Runtime output depends on camera geometry, preprocessing, thresholds,
  post-processing, device backend, and temporal policy—not only model weights.
- Public repository checks establish build and artifact integrity. They do not
  establish real-user outcomes or safety effectiveness.

Untested or unsupported conditions must remain `UNKNOWN`; they must not be
silently converted to negative or safe outcomes.

## License and provenance

Ultralytics states that its software and trained models are offered under
AGPL-3.0 or an Enterprise License. BlindAssist does not independently relicense
the model. The repository therefore uses AGPL-3.0-only for its original default
distribution and records third-party scope in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

The repository contains an export helper, `scripts/export_yolo11n_tflite.py`,
but does not currently publish a complete bit-for-bit upstream checkpoint,
toolchain, and export receipt for this exact TFLite payload. The immutable hash
above is the current distribution identity; reproducible re-export remains a
public maintenance gap rather than an implied claim.

## Evaluation and promotion boundary

Tests and benchmarks in this repository have different evidence roles. A unit
test, successful build, TFLite inspection, single-device benchmark, synthetic
fixture, or model-reviewed label cannot by itself authorize a new default
model, deployment, or safety claim. Candidate models remain isolated until the
applicable quality, device, and release gates pass.

For current research authority, start from
[`docs/research/README.md`](research/README.md). For release validation, use
[`docs/RELEASE_AND_VERIFICATION.md`](RELEASE_AND_VERIFICATION.md).
