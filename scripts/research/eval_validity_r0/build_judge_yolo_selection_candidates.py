"""Build a label-blind YOLO selection candidate universe after review sealing.

This is deliberately a narrow selection-only runner.  It reads only the
frozen pilot RGB windows and the review-seal hash, runs the frozen YOLO box
detector, and emits every event-pair/fixed-slot combination with two primary
boxes.  It does not read source masks, primitive labels, physical conditions,
oracle traces or actionability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json
from .prepare_judge_burned_pilot import FREEZE_SCHEMA
from .seal_judge_review_bundle import SEAL_SCHEMA


CANDIDATE_SCHEMA = "blindassist.eval_validity_r0.judge_pair_candidate_ledger.v1"
SELECTION_FIELDS = ["yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity", "selection_time_slot"]


class YoloSelectionError(ValueError):
    """Raised when selection-only YOLO input is unavailable or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise YoloSelectionError(message)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _primary_box(result: Any) -> dict[str, Any] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    height, width = result.orig_shape[:2]
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    candidates: list[dict[str, Any]] = []
    for box, confidence, class_id in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = [float(value) for value in box]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(width * height))
        candidates.append({
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox_norm": [x1 / max(1.0, width), y1 / max(1.0, height), x2 / max(1.0, width), y2 / max(1.0, height)],
            "area": area,
            "cx": ((x1 + x2) / 2.0) / max(1.0, width),
            "cy": ((y1 + y2) / 2.0) / max(1.0, height),
        })
    return sorted(candidates, key=lambda item: (-item["confidence"], -item["area"], item["class_id"], item["bbox_norm"]))[0]


def _selection_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_box, right_box = left["bbox_norm"], right["bbox_norm"]
    centre_distance = math.sqrt((left["cx"] - right["cx"]) ** 2 + (left["cy"] - right["cy"]) ** 2)
    scale_similarity = min(left["area"], right["area"]) / max(left["area"], right["area"]) if max(left["area"], right["area"]) > 0 else 0.0
    return {
        "yolo_box_similarity": _iou(left_box, right_box),
        "distance_scale_similarity": _clip(scale_similarity),
        "position_similarity": _clip(1.0 - centre_distance / math.sqrt(2.0)),
        "visibility_similarity": _clip(1.0 - abs(left["confidence"] - right["confidence"])),
    }


def build_candidates(*, freeze: dict[str, Any], seal: dict[str, Any], source_root: Path, yolo_model: Path, output: Path, batch_size: int, device: str) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite candidate ledger: {output}")
    _require(freeze.get("schema_version") == FREEZE_SCHEMA and freeze.get("protocol_id") == PROTOCOL_ID, "pilot freeze schema/protocol mismatch")
    _require(seal.get("schema_version") == SEAL_SCHEMA and seal.get("protocol_id") == PROTOCOL_ID, "review seal schema/protocol mismatch")
    _require(seal.get("status") == "PRIMITIVE_REVIEWS_SEALED_BEFORE_PAIR_SELECTION", "review bundle is not sealed")
    freeze_hash = sha256_json(freeze)
    _require(seal.get("pilot_freeze_sha256") == freeze_hash, "review seal/freeze binding mismatch")
    for field in ("primitive_labels_opened_to_pair_builder", "derived_labels_opened_to_pair_builder", "reviewed_event_phase_opened_to_pair_builder", "reviewed_motion_relation_opened_to_pair_builder"):
        _require(seal.get(field) is False, f"review seal exposes forbidden field {field}")
    _require(yolo_model.is_file(), f"YOLO model is missing: {yolo_model}")
    event_rows = sorted(freeze.get("items", []), key=lambda item: item["pilot_event_id"])
    _require(event_rows and all(len(item["frame_indices"]) == 60 for item in event_rows), "pilot freeze does not contain 60-frame events")

    import torch
    from ultralytics import YOLO

    resolved_device = "cuda:0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    model = YOLO(str(yolo_model.resolve()))
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for event in event_rows:
        source_event_id = event["source_screening_event_id"]
        for slot in event["frame_indices"]:
            path = source_root / "events" / source_event_id / "rgb" / f"{slot:03d}.png"
            _require(path.is_file(), f"{event['pilot_event_id']}: RGB frame missing at slot {slot}")
            rows.append({"event_id": event["pilot_event_id"], "slot": slot, "path": path})
            paths.append(path)
    predictions: dict[tuple[str, int], dict[str, Any] | None] = {}
    for start in range(0, len(paths), max(1, batch_size)):
        batch_paths = paths[start : start + max(1, batch_size)]
        results = model.predict(source=[str(path) for path in batch_paths], device=resolved_device, imgsz=640, conf=0.15, iou=0.7, verbose=False)
        _require(len(results) == len(batch_paths), "YOLO result count does not match RGB batch")
        for row, result in zip(rows[start : start + len(batch_paths)], results):
            predictions[(row["event_id"], row["slot"])] = _primary_box(result)

    items: list[dict[str, Any]] = []
    for left_index, left_event in enumerate(event_rows):
        for right_event in event_rows[left_index + 1 :]:
            for slot in left_event["frame_indices"]:
                left_box = predictions[(left_event["pilot_event_id"], slot)]
                right_box = predictions[(right_event["pilot_event_id"], slot)]
                if left_box is None or right_box is None:
                    continue
                metrics = _selection_metrics(left_box, right_box)
                items.append({
                    "candidate_id": f"candidate-{left_event['pilot_event_id']}-{right_event['pilot_event_id']}-slot-{slot:03d}",
                    "event_a_id": left_event["pilot_event_id"],
                    "event_b_id": right_event["pilot_event_id"],
                    **metrics,
                    "selection_time_slot": slot,
                    "comparison_frame_index_a": slot,
                    "comparison_frame_index_b": slot,
                })
    result = {
        "schema_version": CANDIDATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "SELECTION_ONLY_YOLO_CANDIDATE_UNIVERSE_FROZEN",
        "pilot_freeze_sha256": freeze_hash,
        "review_bundle_sha256": seal["review_bundle_sha256"],
        "yolo_role": "SELECTION_ONLY",
        "yolo_visible_to_reviewers": False,
        "yolo_used_for_truth": False,
        "primitive_labels_visible_to_pair_builder": False,
        "derived_labels_visible_to_pair_builder": False,
        "reviewed_event_phase_visible_to_pair_builder": False,
        "reviewed_motion_relation_visible_to_pair_builder": False,
        "enumeration_complete": True,
        "enumeration_scope": "all_pilot_event_pairs_and_fixed_sampling_slots_with_two_primary_yolo_boxes",
        "selection_fields": SELECTION_FIELDS,
        "selection_time_slot_source": "fixed_sampling_slot",
        "yolo_inference": {
            "model_path": str(yolo_model.resolve()),
            "model_sha256": sha256_file(yolo_model),
            "runtime": "ultralytics",
            "device": resolved_device,
            "imgsz": 640,
            "confidence": 0.15,
            "iou": 0.7,
            "primary_box_rule": "highest_confidence_then_area_then_class_then_bbox_lexicographic",
            "box_similarity_rule": "normalized_primary_box_IoU",
            "distance_scale_rule": "min_primary_box_area_div_max_primary_box_area",
            "position_rule": "1_minus_normalized_primary_box_center_distance_over_sqrt2",
            "visibility_rule": "1_minus_absolute_primary_box_confidence_difference",
        },
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": result["status"], "candidate_count": len(items), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-freeze", type=Path, required=True)
    parser.add_argument("--review-seal", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--yolo-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    args = parser.parse_args()
    result = build_candidates(
        freeze=read_json(args.pilot_freeze),
        seal=read_json(args.review_seal),
        source_root=args.source_root,
        yolo_model=args.yolo_model,
        output=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"status={result['status']} candidate_count={result['candidate_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
