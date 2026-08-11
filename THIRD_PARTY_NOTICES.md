# Third-party notices and license scope

Last reviewed: 2026-08-11

The root `LICENSE` applies to original source code and documentation contributed to BlindAssist unless a file or directory states otherwise. It does not replace, broaden, or remove the terms that apply to third-party works.

## Ultralytics YOLO11

- Relevant asset: `app/src/main/assets/yolo11n_fp16_320.tflite`
- Role: exported, COCO-pretrained object-detection model used by the default Android prototype
- Upstream: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- Upstream licensing: Ultralytics states that YOLO11 software and models are available under AGPL-3.0 or an Ultralytics Enterprise License.

BlindAssist does not grant an independent license to this model beyond the applicable upstream terms. Converting or exporting the model to TFLite does not, by itself, replace those terms. Users planning a use that is not compatible with AGPL-3.0 must obtain and verify any required upstream commercial license themselves.

## COCO class-name metadata

- Relevant asset: `app/src/main/assets/coco_labels.txt`
- Role: class-name metadata consumed with the COCO-pretrained detector

No separate ownership claim is made over upstream class names. Users should retain applicable attribution and review the terms of the model and dataset sources used in their distribution.

## Libraries and build dependencies

AndroidX, Jetpack Compose, CameraX, TensorFlow Lite/LiteRT, Kotlin, Gradle plugins, test libraries, Python packages, and other dependencies remain under their respective upstream licenses. Dependency declarations in this repository do not relicense those projects under the BlindAssist license.

## Research data, media, models, and hardware references

Datasets, downloaded media, research checkpoints, generated model payloads, device logs, SDKs, and local benchmark evidence belong in ignored local artifact paths and are not licensed for redistribution merely because a document references them. Source-specific provenance, privacy, consent, and license records control their permitted uses. Hardware documentation, trademarks, product names, and vendor SDKs remain the property of their respective owners.

If a third-party work or required notice is missing from this file, please open an issue with the exact path, source, and license evidence. Absence from this notice is not a grant of rights.
