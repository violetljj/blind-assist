#!/usr/bin/env python3
"""Create a machine-redacted, still-quarantined RGB-only public candidate view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


__all__ = [
    "LpdYuNet",
    "RedactionError",
    "WholeObjectPrivacyDetector",
    "candidate_regions",
    "expanded_box",
    "load_json",
    "main",
    "redact_regions",
    "sha256_file",
    "valid_lpd_region",
    "valid_privacy_object_region",
]


class RedactionError(ValueError):
    """The RGB candidate cannot pass the machine-redaction precondition."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RedactionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RedactionError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expanded_box(box: tuple[int, int, int, int], *, width: int, height: int, margin: float = 0.25) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pad_x, pad_y = round(w * margin), round(h * margin)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    return x0, y0, x1, y1


def redact_regions(image: np.ndarray, regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    for region in regions:
        x0, y0, x1, y1 = expanded_box(region, width=width, height=height)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = result[y0:y1, x0:x1]
        kernel_x = max(3, ((crop.shape[1] // 4) * 2) | 1)
        kernel_y = max(3, ((crop.shape[0] // 4) * 2) | 1)
        result[y0:y1, x0:x1] = cv2.GaussianBlur(crop, (kernel_x, kernel_y), 0)
    return result


class LpdYuNet:
    """Small, local adaptation of OpenCV Zoo's Apache-2.0 LPD_YuNet wrapper."""

    output_names = ["loc", "conf", "iou"]
    min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    steps = [8, 16, 32, 64]
    variance = [0.1, 0.2]

    def __init__(self, model_path: Path, input_size: tuple[int, int] = (320, 240), confidence: float = 0.85) -> None:
        self.input_size = np.asarray(input_size, dtype=np.float32)
        self.confidence = confidence
        self.model = cv2.dnn.readNet(str(model_path))
        self._prior_gen()

    def _prior_gen(self) -> None:
        width, height = (int(value) for value in self.input_size)
        feature_map_2 = [int(int((height + 1) / 2) / 2), int(int((width + 1) / 2) / 2)]
        feature_maps = [
            [int(feature_map_2[0] / 2), int(feature_map_2[1] / 2)],
        ]
        for _index in range(3):
            previous = feature_maps[-1]
            feature_maps.append([int(previous[0] / 2), int(previous[1] / 2)])
        priors: list[list[float]] = []
        for level, feature_map in enumerate(feature_maps):
            for y in range(feature_map[0]):
                for x in range(feature_map[1]):
                    for min_size in self.min_sizes[level]:
                        priors.append([
                            (x + 0.5) * self.steps[level] / width,
                            (y + 0.5) * self.steps[level] / height,
                            min_size / width,
                            min_size / height,
                        ])
        self.priors = np.asarray(priors, dtype=np.float32)

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        original_height, original_width = image.shape[:2]
        input_width, input_height = (int(value) for value in self.input_size)
        resized = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)
        self.model.setInput(cv2.dnn.blobFromImage(resized))
        loc, conf, iou = self.model.forward(self.output_names)
        scores = np.sqrt(np.clip(conf[:, 1], 0.0, 1.0) * np.clip(iou[:, 0], 0.0, 1.0))
        indices = np.flatnonzero(scores >= self.confidence)
        if not len(indices):
            return []
        points = np.hstack((
            (self.priors[indices, 0:2] + loc[indices, 4:6] * self.variance[0] * self.priors[indices, 2:4]) * self.input_size,
            (self.priors[indices, 0:2] + loc[indices, 6:8] * self.variance[0] * self.priors[indices, 2:4]) * self.input_size,
            (self.priors[indices, 0:2] + loc[indices, 10:12] * self.variance[0] * self.priors[indices, 2:4]) * self.input_size,
            (self.priors[indices, 0:2] + loc[indices, 12:14] * self.variance[0] * self.priors[indices, 2:4]) * self.input_size,
        )).reshape(-1, 4, 2)
        boxes: list[list[int]] = []
        for corners in points:
            corners[:, 0] *= original_width / input_width
            corners[:, 1] *= original_height / input_height
            x0, y0 = np.floor(corners.min(axis=0)).astype(int)
            x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
            boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
        keep = cv2.dnn.NMSBoxes(boxes, scores[indices].tolist(), self.confidence, 0.3)
        return [tuple(boxes[int(index)]) for index in np.asarray(keep).reshape(-1)] if len(keep) else []


class WholeObjectPrivacyDetector:
    """Use an off-the-shelf COCO detector to conservatively blur people and vehicles."""

    coco_privacy_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

    def __init__(self, model_path: Path, config_dir: Path) -> None:
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RedactionError("Ultralytics is required for whole-person/vehicle privacy redaction") from error
        self.model = YOLO(str(model_path))

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        result = self.model(image, conf=0.15, classes=self.coco_privacy_classes, verbose=False)[0]
        boxes: list[tuple[int, int, int, int]] = []
        for xyxy in result.boxes.xyxy:
            x0, y0, x1, y1 = (int(round(float(value))) for value in xyxy)
            boxes.append((x0, y0, x1 - x0, y1 - y0))
        return boxes


def valid_lpd_region(box: tuple[int, int, int, int], *, width: int, height: int) -> bool:
    x, y, box_width, box_height = box
    return (
        box_width >= 12
        and box_height >= 4
        and box_width * box_height <= width * height * 0.05
        and x < width
        and y < height
        and x + box_width > 0
        and y + box_height > 0
    )


def valid_privacy_object_region(box: tuple[int, int, int, int], *, width: int, height: int) -> bool:
    x, y, box_width, box_height = box
    return (
        box_width >= 12
        and box_height >= 12
        and box_width * box_height <= width * height * 0.75
        and x < width
        and y < height
        and x + box_width > 0
        and y + box_height > 0
    )


def candidate_regions(
    image: np.ndarray,
    face_detector: Any,
    plate_detector: LpdYuNet,
    whole_object_detector: WholeObjectPrivacyDetector,
) -> list[tuple[int, int, int, int]]:
    found: set[tuple[int, int, int, int]] = set()
    height, width = image.shape[:2]
    face_detector.setInputSize((width, height))
    _status, faces = face_detector.detect(image)
    if faces is not None:
        for face in faces:
            x, y, box_width, box_height = (int(round(value)) for value in face[:4])
            if 12 <= box_width <= width // 3 and 12 <= box_height <= height // 3:
                found.add((x, y, box_width, box_height))
    for box in plate_detector.detect(image):
        parsed = tuple(int(value) for value in box)
        if valid_lpd_region(parsed, width=width, height=height):
            found.add(parsed)
    for box in whole_object_detector.detect(image):
        parsed = tuple(int(value) for value in box)
        if valid_privacy_object_region(parsed, width=width, height=height):
            found.add(parsed)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yunet-model", type=Path, required=True)
    parser.add_argument("--lpd-yunet-model", type=Path, required=True)
    parser.add_argument("--yolo-model", type=Path, required=True)
    args = parser.parse_args()
    try:
        extraction = load_json(args.extraction_dir / "rgb_extraction_receipt.json")
        if extraction.get("privacy_processing_required") is not True or extraction.get("source_labels_or_geometry_extracted") is not False:
            raise RedactionError("RGB extraction receipt does not enforce the required source boundary")
        if extraction.get("training_execution_authorized") is not False:
            raise RedactionError("RGB extraction unexpectedly authorizes training")
        source_paths = extraction.get("relative_rgb_paths")
        if not isinstance(source_paths, list) or not source_paths:
            raise RedactionError("RGB extraction receipt lacks source paths")
        if args.output_dir.exists():
            raise RedactionError(f"refusing to overwrite redaction output: {args.output_dir}")
        if not args.yunet_model.is_file():
            raise RedactionError(f"YuNet face model is missing: {args.yunet_model}")
        if not args.lpd_yunet_model.is_file():
            raise RedactionError(f"LPD_YuNet plate model is missing: {args.lpd_yunet_model}")
        if not args.yolo_model.is_file():
            raise RedactionError(f"YOLO privacy model is missing: {args.yolo_model}")
        face_detector = cv2.FaceDetectorYN.create(str(args.yunet_model), "", (320, 320), 0.85, 0.3, 5000)
        plate_detector = LpdYuNet(args.lpd_yunet_model)
        whole_object_detector = WholeObjectPrivacyDetector(args.yolo_model, args.output_dir / ".ultralytics")
        images_dir = args.output_dir / "images"
        images_dir.mkdir(parents=True)
        items: list[dict[str, Any]] = []
        face_or_plate_frames = 0
        region_count = 0
        for index, relative in enumerate(source_paths):
            source = args.extraction_dir / relative
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RedactionError(f"cannot decode extracted RGB image: {source}")
            regions = candidate_regions(image, face_detector, plate_detector, whole_object_detector)
            output = images_dir / f"frame_{index:04d}.png"
            if not cv2.imwrite(str(output), redact_regions(image, regions)):
                raise RedactionError(f"cannot write redacted RGB image: {output}")
            if regions:
                face_or_plate_frames += 1
                region_count += len(regions)
            items.append({"frame_index": index, "file_name": output.name, "sha256": sha256_file(output), "redaction_region_count": len(regions)})
        manifest = {
            "format": "blindassist_public_rgb_machine_redaction_v1",
            "source_id": extraction["source_id"],
            "input_rgb_extraction_receipt": "rgb_extraction_receipt.json",
            "frame_count": len(items),
            "frames_with_machine_redaction_regions": face_or_plate_frames,
            "machine_redaction_region_count": region_count,
            "detectors": [
                "opencv_yunet_face_detection_2023mar",
                "opencv_lpd_yunet_2023mar",
                "ultralytics_yolov8n_whole_person_vehicle_blur",
            ],
            "frames": items,
            "privacy_audit_required": True,
            "human_event_truth_present": False,
            "training_execution_authorized": False,
            "production_model_replacement_authorized": False,
            "important_limit": "Machine redaction is a privacy prefilter, not proof that every face or plate was detected.",
        }
        (args.output_dir / "machine_redaction_receipt.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "frame_count": len(items), "redaction_regions": region_count, "privacy_audit_required": True}, ensure_ascii=False))
    except (RedactionError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
