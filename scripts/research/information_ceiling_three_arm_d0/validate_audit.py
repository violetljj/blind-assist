#!/usr/bin/env python3
"""Independently validate and score the BlindAssist three-arm information audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "INFORMATION_CEILING_THREE_ARM_D0"
COMPARISON_MODE = "InformationCeilingThreeArm"
EXPECTED_MANIFEST_SHA256 = "3d7168ac975aed57ac6b437ecfa0e668c13dc5d509c7b6584353383eed19d217"
EXPECTED_DATASET_SPEC_SHA256 = "6815d8b613eca34d840255e66f02bd751196cb55e1b3b74f8ff0f659babf07bb"
EXPECTED_YOLO_SHA256 = "00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2"
EXPECTED_FRAME_COUNT = 90
EXPECTED_EVENT_COUNT = 3
FRAME_STEP_MS = 100
ARM_IDS = ("A_CURRENT_YOLO", "B_ORACLE_RISK_BOX", "C_ORACLE_RISK_MASK")
RISK_LABELS = {
    2: "curb",
    4: "road barrier",
    9: "hand rail",
    10: "opening door",
    11: "opening gate",
    15: "stairs",
    18: "inaccessible surface",
    20: "generic obstacle",
    24: "pole",
    26: "bike rack",
}
NAVIGATION_HAZARD_IDS = set(RISK_LABELS)
BOX_CLASS_ID_OFFSET = 20_000
MASK_CLASS_ID_OFFSET = 10_000
MASK_SIZE = 256


class ValidationFailure(RuntimeError):
    """Raised only for malformed CLI arguments; audit invalidity is reported as data."""


@dataclass(frozen=True)
class AuditInputs:
    benchmark_json: Path
    manifest: Path
    output_dir: Path
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256
    expected_dataset_spec_sha256: str = EXPECTED_DATASET_SPEC_SHA256
    expected_frame_count: int = EXPECTED_FRAME_COUNT
    expected_event_count: int = EXPECTED_EVENT_COUNT
    verify_source_hashes: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValidationFailure(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValidationFailure(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValidationFailure(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def basename_for_manifest_row(row: dict[str, Any]) -> str:
    return Path(str(row.get("image_path", ""))).name


def normalized_expected_truth(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_risk_direction": row.get("expected_risk_direction"),
        "expected_distance_band": row.get("expected_distance_band"),
        "expected_should_alert": row.get("expected_should_alert"),
        "expected_risk_level": row.get("expected_risk_level"),
        "assist_scenario": row.get("assist_scenario"),
        "primary_object_id": row.get("primary_object_id"),
        "scene_bucket": row.get("scene_bucket"),
        "risk_event_id": row.get("risk_event_id"),
        "sequence_id": row.get("sequence_id"),
        "frame_index": row.get("frame_index"),
        "expected_approach_state": row.get("expected_approach_state"),
        "expected_approach_alert": row.get("expected_approach_alert"),
        "expected_time_to_alert_frames": row.get("expected_time_to_alert_frames"),
        "expected_event_phase": row.get("expected_event_phase"),
    }


def close_number(actual: Any, expected: float, tolerance: float = 0.002) -> bool:
    return isinstance(actual, (int, float)) and abs(float(actual) - expected) <= tolerance


def expected_box_inputs(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for region in row.get("source_regions", []):
        class_id = region.get("sanpo_class_id")
        label = RISK_LABELS.get(class_id)
        if label is None:
            continue
        expected.append(
            {
                "class_id": BOX_CLASS_ID_OFFSET + class_id,
                "label": label,
                "confidence": 1.0,
                "source": "OBJECT_DETECTOR",
                "bbox_xyxy": [float(value) for value in region["bbox_xyxy"]],
                "temporal_promotion_eligible": True,
            }
        )
    return expected


def mask_path_for_row(dataset_root: Path, row: dict[str, Any]) -> Path:
    return dataset_root / "source_masks" / str(row.get("split", "test")) / basename_for_manifest_row(row)


def load_mask_class_ids(mask_path: Path) -> tuple[list[int], int, int]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - deployment environment check
        raise ValidationFailure("Pillow is required to independently validate Arm C") from error
    with Image.open(mask_path) as source:
        resized = source.convert("RGB").resize((MASK_SIZE, MASK_SIZE), resample=Image.Resampling.NEAREST)
        pixels = list(resized.get_flattened_data())
    return [int(pixel[0]) for pixel in pixels], MASK_SIZE, MASK_SIZE


def expected_mask_inputs(row: dict[str, Any], mask_path: Path) -> list[dict[str, Any]]:
    class_ids, width, height = load_mask_class_ids(mask_path)
    corridor = [False] * len(class_ids)
    corridor_top_ratio = 0.42
    corridor_top_half_width_ratio = 0.16
    corridor_bottom_half_width_ratio = 0.42
    top = min(max(int(height * corridor_top_ratio), 0), height - 1)
    for y in range(top, height):
        progress = (y - top) / max(1, height - 1 - top)
        half_width_ratio = corridor_top_half_width_ratio + (
            corridor_bottom_half_width_ratio - corridor_top_half_width_ratio
        ) * progress
        left = min(max(int(width * (0.5 - half_width_ratio)), 0), width - 1)
        right = min(max(int(width * (0.5 + half_width_ratio)), left + 1), width)
        for x in range(left, right):
            corridor[y * width + x] = True

    visited = [False] * len(class_ids)
    detections: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for start, class_id in enumerate(class_ids):
        if visited[start] or class_id not in NAVIGATION_HAZARD_IDS:
            continue
        visited[start] = True
        queue: deque[int] = deque([start])
        pixels = 0
        corridor_pixels = 0
        min_x, min_y, max_x, max_y = width, height, 0, 0
        while queue:
            index = queue.popleft()
            x, y = index % width, index // width
            pixels += 1
            corridor_pixels += int(corridor[index])
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
            for candidate, valid in (
                (index - 1, x > 0),
                (index + 1, x + 1 < width),
                (index - width, y > 0),
                (index + width, y + 1 < height),
            ):
                if valid and not visited[candidate] and class_ids[candidate] == class_id:
                    visited[candidate] = True
                    queue.append(candidate)

        area_ratio = pixels / len(class_ids)
        center_overlap_ratio = corridor_pixels / max(1, pixels)
        bottom_ratio = (max_y + 1) / height
        is_boundary = class_id == 2
        minimum_overlap = 0.35 if is_boundary else 0.10
        minimum_bottom = 0.62 if is_boundary else 0.42
        aspect_ratio = (max_x - min_x + 1) / max(1, max_y - min_y + 1)
        boundary_like = (
            (RISK_LABELS[class_id] == "generic obstacle" or is_boundary)
            and aspect_ratio >= 3.0
            and center_overlap_ratio <= 0.34
            and (min_x <= width * 0.18 or max_x + 1 >= width * (1.0 - 0.18))
        )
        # BlindAssistSanpoTaxonomy does not permit boundary detections, so class 2
        # remains diagnostic-only in the production analyzer.
        passes_gate = (
            center_overlap_ratio >= minimum_overlap
            and bottom_ratio >= minimum_bottom
            and (not is_boundary)
        )
        if not (
            passes_gate
            and pixels >= 12
            and area_ratio >= 0.0008
            and class_id in RISK_LABELS
        ):
            continue
        frame_width = int(row["width"])
        frame_height = int(row["height"])
        scale_x = frame_width / width
        scale_y = frame_height / height
        box = [
            min_x * scale_x,
            min_y * scale_y,
            (max_x + 1) * scale_x,
            (max_y + 1) * scale_y,
        ]
        center_distance = abs(((box[0] + box[2]) / 2.0) / frame_width - 0.5)
        sort_key = (
            center_distance,
            -(box[3] / frame_height),
            ((box[2] - box[0]) * (box[3] - box[1])) / (frame_width * frame_height),
        )
        detections.append(
            (
                sort_key,
                {
                    "class_id": MASK_CLASS_ID_OFFSET + class_id,
                    "label": RISK_LABELS[class_id],
                    "confidence": 1.0,
                    "source": "SEGMENTATION",
                    "bbox_xyxy": box,
                    "temporal_promotion_eligible": not boundary_like,
                },
            )
        )
    detections.sort(key=lambda item: item[0])
    return [item[1] for item in detections[:1]]


def compare_detection_lists(
    actual: Any,
    expected: list[dict[str, Any]],
    context: str,
    errors: list[str],
) -> None:
    if not isinstance(actual, list):
        errors.append(f"{context}: risk_inputs is not a list")
        return
    if len(actual) != len(expected):
        errors.append(f"{context}: expected {len(expected)} risk inputs, found {len(actual)}")
        return
    for index, (actual_detection, expected_detection) in enumerate(zip(actual, expected)):
        prefix = f"{context}: risk_inputs[{index}]"
        if not isinstance(actual_detection, dict):
            errors.append(f"{prefix}: not an object")
            continue
        for key in ("class_id", "label", "source", "temporal_promotion_eligible"):
            if actual_detection.get(key) != expected_detection[key]:
                errors.append(
                    f"{prefix}.{key}: expected {expected_detection[key]!r}, "
                    f"found {actual_detection.get(key)!r}"
                )
        if not close_number(actual_detection.get("confidence"), expected_detection["confidence"]):
            errors.append(f"{prefix}.confidence: expected 1.0")
        actual_box = actual_detection.get("bbox_xyxy")
        expected_box = expected_detection["bbox_xyxy"]
        if not isinstance(actual_box, list) or len(actual_box) != 4:
            errors.append(f"{prefix}.bbox_xyxy: malformed")
        elif not all(close_number(actual_box[i], expected_box[i], 0.006) for i in range(4)):
            errors.append(
                f"{prefix}.bbox_xyxy: expected rounded {expected_box!r}, found {actual_box!r}"
            )


def validate_manifest(
    inputs: AuditInputs,
    manifest_rows: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    manifest_hash = sha256_file(inputs.manifest)
    add_error(
        errors,
        manifest_hash == inputs.expected_manifest_sha256,
        f"manifest SHA-256 mismatch: {manifest_hash}",
    )
    dataset_spec = inputs.manifest.parent / "dataset_spec.json"
    add_error(errors, dataset_spec.is_file(), f"missing dataset_spec.json: {dataset_spec}")
    dataset_spec_hash = sha256_file(dataset_spec) if dataset_spec.is_file() else None
    if dataset_spec_hash is not None:
        add_error(
            errors,
            dataset_spec_hash == inputs.expected_dataset_spec_sha256,
            f"dataset_spec SHA-256 mismatch: {dataset_spec_hash}",
        )
    add_error(
        errors,
        len(manifest_rows) == inputs.expected_frame_count,
        f"expected {inputs.expected_frame_count} manifest rows, found {len(manifest_rows)}",
    )
    ids = [row.get("id") for row in manifest_rows]
    add_error(errors, len(ids) == len(set(ids)), "manifest frame ids are not unique")
    event_ids = {row.get("risk_event_id") for row in manifest_rows}
    add_error(errors, None not in event_ids and "" not in event_ids, "missing risk_event_id")
    add_error(
        errors,
        len(event_ids) == inputs.expected_event_count,
        f"expected {inputs.expected_event_count} events, found {len(event_ids)}",
    )
    required = (
        "id",
        "image_path",
        "sequence_id",
        "frame_index",
        "frame_timestamp_ms",
        "source_regions",
        "expected_should_alert",
        "expected_risk_direction",
        "expected_distance_band",
        "expected_risk_level",
        "expected_event_phase",
        "risk_event_id",
    )
    dataset_root = inputs.manifest.parent
    source_quality_counts: dict[str, int] = defaultdict(int)
    official_split_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(manifest_rows):
        context = f"manifest row {index}"
        for field in required:
            add_error(errors, field in row, f"{context}: missing {field}")
        add_error(
            errors,
            row.get("sequence_id") == row.get("risk_event_id"),
            f"{context}: sequence_id and risk_event_id differ",
        )
        image_path = dataset_root / str(row.get("image_path", ""))
        mask_path = mask_path_for_row(dataset_root, row)
        add_error(errors, image_path.is_file(), f"{context}: missing image {image_path}")
        add_error(errors, mask_path.is_file(), f"{context}: missing mask {mask_path}")
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        if inputs.verify_source_hashes and image_path.is_file():
            add_error(
                errors,
                sha256_file(image_path) == source.get("sha256"),
                f"{context}: image SHA-256 mismatch",
            )
        if inputs.verify_source_hashes and mask_path.is_file():
            add_error(
                errors,
                sha256_file(mask_path) == source.get("mask_sha256"),
                f"{context}: mask SHA-256 mismatch",
            )
        source_quality_counts[str(row.get("source_annotation_quality"))] += 1
        official_split_counts[str(source.get("official_split"))] += 1
    if official_split_counts.get("train", 0) > 0:
        warnings.append(
            "The local evaluation manifest mixes upstream official train/test provenance; "
            "this consumed Development cohort is not an official-test-only result."
        )
    return {
        "manifest_sha256": manifest_hash,
        "dataset_spec_sha256": dataset_spec_hash,
        "frame_count": len(manifest_rows),
        "event_count": len(event_ids),
        "source_annotation_quality_counts": dict(sorted(source_quality_counts.items())),
        "upstream_official_split_counts": dict(sorted(official_split_counts.items())),
    }


def validate_benchmark_contract(
    benchmark: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    manifest_path: Path,
    errors: list[str],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    add_error(
        errors,
        benchmark.get("comparison_mode") == COMPARISON_MODE,
        f"comparison_mode must be {COMPARISON_MODE}",
    )
    add_error(errors, benchmark.get("dataset_kind") == "BlindAssistEvalSet", "wrong dataset_kind")
    add_error(errors, benchmark.get("risk_config") == "current", "risk_config must be current")
    add_error(errors, benchmark.get("alert_profile") == "STANDARD", "alert_profile must be STANDARD")
    add_error(
        errors,
        benchmark.get("synthetic_clock_frame_step_ms") == FRAME_STEP_MS,
        "synthetic clock step must be 100 ms",
    )
    add_error(
        errors,
        benchmark.get("image_count") == len(manifest_rows),
        "benchmark image_count does not match manifest",
    )
    add_error(
        errors,
        isinstance(benchmark.get("decision_kernel_contract_id"), str)
        and bool(benchmark.get("decision_kernel_contract_id")),
        "missing decision_kernel_contract_id",
    )
    models = benchmark.get("models")
    if not isinstance(models, list):
        errors.append("benchmark models is not a list")
        return {}
    by_id = {model.get("id"): model for model in models if isinstance(model, dict)}
    add_error(errors, set(by_id) == set(ARM_IDS), f"expected exactly arms {ARM_IDS}, found {tuple(by_id)}")
    expected_flags = {
        "A_CURRENT_YOLO": (False, True, False, True),
        "B_ORACLE_RISK_BOX": (True, False, False, False),
        "C_ORACLE_RISK_MASK": (False, False, True, False),
    }
    manifest_by_name = {basename_for_manifest_row(row): row for row in manifest_rows}
    dataset_root = manifest_path.parent
    per_arm_frames: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        model = by_id.get(arm_id)
        if model is None:
            continue
        box_oracle, include_yolo, mask_oracle, runtime_comparable = expected_flags[arm_id]
        for key, expected in (
            ("source_region_box_oracle", box_oracle),
            ("include_yolo_risk_inputs", include_yolo),
            ("traversability_oracle", mask_oracle),
            ("runtime_comparable", runtime_comparable),
        ):
            add_error(errors, model.get(key) == expected, f"{arm_id}: {key} must be {expected}")
        add_error(
            errors,
            model.get("model_asset_sha256") == EXPECTED_YOLO_SHA256,
            f"{arm_id}: unexpected YOLO asset hash",
        )
        app = model.get("app_detector")
        if not isinstance(app, dict):
            errors.append(f"{arm_id}: missing app_detector")
            continue
        add_error(
            errors,
            app.get("decision_kernel_contract_id") == benchmark.get("decision_kernel_contract_id"),
            f"{arm_id}: decision kernel contract mismatch",
        )
        add_error(errors, app.get("runs_per_image") == benchmark.get("app_runs_per_image"), f"{arm_id}: runs_per_image mismatch")
        add_error(errors, app.get("failures") == [], f"{arm_id}: app failures present")
        per_image = app.get("per_image")
        if not isinstance(per_image, list):
            errors.append(f"{arm_id}: per_image is not a list")
            continue
        names = [frame.get("image") for frame in per_image if isinstance(frame, dict)]
        add_error(
            errors,
            len(per_image) == len(manifest_rows),
            f"{arm_id}: expected {len(manifest_rows)} per-image rows, found {len(per_image)}",
        )
        add_error(errors, len(names) == len(set(names)), f"{arm_id}: duplicate per-image names")
        add_error(
            errors,
            set(names) == set(manifest_by_name),
            f"{arm_id}: per-image membership differs from manifest",
        )
        frame_map = {frame.get("image"): frame for frame in per_image if isinstance(frame, dict)}
        per_arm_frames[arm_id] = frame_map
        for image_name, manifest_row in manifest_by_name.items():
            frame = frame_map.get(image_name)
            if frame is None:
                continue
            expected_truth = normalized_expected_truth(manifest_row)
            actual_truth = frame.get("expected_blindassist")
            add_error(
                errors,
                actual_truth == expected_truth,
                f"{arm_id}/{image_name}: embedded scoring truth differs from manifest",
            )
            add_error(errors, isinstance(frame.get("actual_alert"), bool), f"{arm_id}/{image_name}: actual_alert missing")
            add_error(
                errors,
                isinstance(frame.get("stable_model_risk"), dict),
                f"{arm_id}/{image_name}: stable_model_risk missing",
            )
            if arm_id == "A_CURRENT_YOLO":
                risk_inputs = frame.get("risk_inputs")
                if not isinstance(risk_inputs, list):
                    errors.append(f"{arm_id}/{image_name}: risk_inputs missing")
                else:
                    for input_index, detection in enumerate(risk_inputs):
                        if (
                            not isinstance(detection, dict)
                            or detection.get("source") != "OBJECT_DETECTOR"
                            or not isinstance(detection.get("class_id"), int)
                            or detection["class_id"] >= MASK_CLASS_ID_OFFSET
                        ):
                            errors.append(
                                f"{arm_id}/{image_name}: risk_inputs[{input_index}] is not a YOLO input"
                            )
            elif arm_id == "B_ORACLE_RISK_BOX":
                compare_detection_lists(
                    frame.get("risk_inputs"),
                    expected_box_inputs(manifest_row),
                    f"{arm_id}/{image_name}",
                    errors,
                )
            else:
                mask_path = mask_path_for_row(dataset_root, manifest_row)
                if mask_path.is_file():
                    compare_detection_lists(
                        frame.get("risk_inputs"),
                        expected_mask_inputs(manifest_row, mask_path),
                        f"{arm_id}/{image_name}",
                        errors,
                    )
    if per_arm_frames:
        memberships = [set(frames) for frames in per_arm_frames.values()]
        if memberships:
            add_error(errors, all(membership == memberships[0] for membership in memberships), "arm memberships differ")
    warnings.append(
        "Arm B uses mask-derived source-region boxes, not independent detector-native instance-box truth."
    )
    warnings.append(
        "Arm C becomes at most one Detection after the current mask adapter, and B/C use different "
        "source-specific temporal/event policies; C-vs-B does not isolate bbox geometry."
    )
    return per_arm_frames


def event_rows(manifest_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["risk_event_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return dict(grouped)


def event_ledger_for_arm(
    arm_id: str,
    grouped_truth: dict[str, list[dict[str, Any]]],
    frame_map: dict[str, Any],
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for event_id, truth_rows in grouped_truth.items():
        joined = [(truth, frame_map[basename_for_manifest_row(truth)]) for truth in truth_rows]
        expected_rows = [(truth, frame) for truth, frame in joined if truth["expected_should_alert"]]
        passed_rows = [(truth, frame) for truth, frame in joined if truth.get("expected_event_phase") == "PASSED"]
        positive = bool(expected_rows)
        critical = any(truth.get("expected_distance_band") == "CRITICAL" for truth, _ in expected_rows)
        delivered_expected = [(truth, frame) for truth, frame in expected_rows if frame["actual_alert"]]
        hit = bool(delivered_expected)
        first_expected_frame = int(expected_rows[0][0]["frame_index"]) if expected_rows else None
        first_delivered_frame = (
            int(delivered_expected[0][0]["frame_index"]) if delivered_expected else None
        )
        first_delay = (
            first_delivered_frame - first_expected_frame
            if first_expected_frame is not None and first_delivered_frame is not None
            else None
        )
        false_alert_frames = [
            int(truth["frame_index"])
            for truth, frame in joined
            if frame["actual_alert"] and not truth["expected_should_alert"]
        ]
        passed_alert_frames = [
            int(truth["frame_index"]) for truth, frame in passed_rows if frame["actual_alert"]
        ]
        direction_matched_hits = 0
        for truth, frame in delivered_expected:
            stable_risk = frame.get("stable_model_risk", {})
            if stable_risk.get("direction") == truth.get("expected_risk_direction"):
                direction_matched_hits += 1
        clearance_delay: int | None = None
        if passed_rows:
            first_passed = int(passed_rows[0][0]["frame_index"])
            if not passed_alert_frames:
                clearance_delay = 0
            else:
                last_alert = max(passed_alert_frames)
                later_clear = [
                    int(truth["frame_index"])
                    for truth, frame in passed_rows
                    if int(truth["frame_index"]) > last_alert and not frame["actual_alert"]
                ]
                if later_clear:
                    clearance_delay = min(later_clear) - first_passed
        ledger.append(
            {
                "protocol_id": PROTOCOL_ID,
                "arm_id": arm_id,
                "risk_event_id": event_id,
                "scene_bucket": truth_rows[0].get("scene_bucket"),
                "frame_count": len(truth_rows),
                "positive_event": positive,
                "critical_event": critical,
                "event_hit": hit,
                "critical_event_miss": critical and not hit,
                "negative_event_false_alert": (not positive) and bool(false_alert_frames),
                "first_expected_alert_frame": first_expected_frame,
                "first_delivered_alert_frame": first_delivered_frame,
                "first_effective_response_delay_frames": first_delay,
                "delivered_alert_count": sum(int(frame["actual_alert"]) for _, frame in joined),
                "false_alert_frame_count": len(false_alert_frames),
                "false_alert_frames": false_alert_frames,
                "has_passed_phase": bool(passed_rows),
                "passed_phase_clear": bool(passed_rows) and not passed_alert_frames,
                "passed_phase_alert_frames": passed_alert_frames,
                "post_event_clearance_delay_frames": clearance_delay,
                "direction_matched_delivered_hit_count": direction_matched_hits,
            }
        )
    return sorted(ledger, key=lambda row: (row["arm_id"], row["risk_event_id"]))


def aggregate_arm(ledger: list[dict[str, Any]], frame_count: int) -> dict[str, Any]:
    positive = [row for row in ledger if row["positive_event"]]
    critical = [row for row in ledger if row["critical_event"]]
    negative = [row for row in ledger if not row["positive_event"]]
    passed = [row for row in ledger if row["has_passed_phase"]]
    hit_count = sum(int(row["event_hit"]) for row in positive)
    critical_misses = sum(int(row["critical_event_miss"]) for row in critical)
    false_event_count = sum(int(row["negative_event_false_alert"]) for row in negative)
    cleared_count = sum(int(row["passed_phase_clear"]) for row in passed)
    false_alert_count = sum(int(row["false_alert_frame_count"]) for row in ledger)
    exposure_ms = frame_count * FRAME_STEP_MS
    delays = {
        row["risk_event_id"]: row["first_effective_response_delay_frames"]
        for row in positive
    }
    delays_by_scene = {
        row["scene_bucket"]: row["first_effective_response_delay_frames"]
        for row in positive
    }
    return {
        "parent_event_count": len(ledger),
        "positive_event_count": len(positive),
        "positive_event_hit_count": hit_count,
        "event_alert_recall": hit_count / len(positive) if positive else None,
        "critical_event_count": len(critical),
        "critical_event_miss_count": critical_misses,
        "negative_event_count": len(negative),
        "false_alert_event_count": false_event_count,
        "false_alert_count": false_alert_count,
        "exposure_ms": exposure_ms,
        "false_alerts_per_minute": false_alert_count * 60_000 / exposure_ms if exposure_ms else None,
        "passed_event_count": len(passed),
        "cleared_passed_event_count": cleared_count,
        "post_event_clearance_rate": cleared_count / len(passed) if passed else None,
        "first_effective_response_delay_frames_by_event": delays,
        "first_effective_response_delay_frames_by_scene": delays_by_scene,
    }


def compare_arms(
    candidate_id: str,
    reference_id: str,
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate = arms[candidate_id]
    reference = arms[reference_id]
    deltas = {
        "positive_event_hit_count": candidate["positive_event_hit_count"]
        - reference["positive_event_hit_count"],
        "critical_event_miss_count": candidate["critical_event_miss_count"]
        - reference["critical_event_miss_count"],
        "false_alert_event_count": candidate["false_alert_event_count"]
        - reference["false_alert_event_count"],
        "cleared_passed_event_count": candidate["cleared_passed_event_count"]
        - reference["cleared_passed_event_count"],
    }
    improvements = {
        "positive_event_hit_count": deltas["positive_event_hit_count"] > 0,
        "critical_event_miss_count": deltas["critical_event_miss_count"] < 0,
        "false_alert_event_count": deltas["false_alert_event_count"] < 0,
        "cleared_passed_event_count": deltas["cleared_passed_event_count"] > 0,
    }
    regressions = {
        "positive_event_hit_count": deltas["positive_event_hit_count"] < 0,
        "critical_event_miss_count": deltas["critical_event_miss_count"] > 0,
        "false_alert_event_count": deltas["false_alert_event_count"] > 0,
        "cleared_passed_event_count": deltas["cleared_passed_event_count"] < 0,
    }
    material = any(improvements.values()) and not any(regressions.values())
    tradeoff = any(improvements.values()) and any(regressions.values())
    counts_equal = all(delta == 0 for delta in deltas.values())
    candidate_delays = candidate["first_effective_response_delay_frames_by_event"]
    reference_delays = reference["first_effective_response_delay_frames_by_event"]
    shared = sorted(
        event_id
        for event_id in candidate_delays
        if candidate_delays[event_id] is not None and reference_delays.get(event_id) is not None
    )
    timing_deltas = {
        event_id: candidate_delays[event_id] - reference_delays[event_id]
        for event_id in shared
    }
    timing_only = (
        counts_equal
        and bool(shared)
        and all(delta <= 0 for delta in timing_deltas.values())
        and any(delta <= -2 for delta in timing_deltas.values())
    )
    return {
        "candidate": candidate_id,
        "reference": reference_id,
        "parent_event_count_deltas": deltas,
        "material_event_gain": material,
        "tradeoff": tradeoff,
        "timing_only_gain": timing_only,
        "shared_hit_timing_delta_frames": timing_deltas,
    }


def derive_terminal(pairwise: dict[str, dict[str, Any]]) -> str:
    b_a = pairwise["B_vs_A"]
    c_b = pairwise["C_vs_B"]
    c_a = pairwise["C_vs_A"]
    if any(pair["tradeoff"] for pair in pairwise.values()):
        return "MIXED_DETECTOR_AND_REPRESENTATION_GAPS"
    if b_a["material_event_gain"] and c_b["material_event_gain"] and c_a["material_event_gain"]:
        return "MIXED_DETECTOR_AND_REPRESENTATION_GAPS"
    if (
        not b_a["material_event_gain"]
        and c_b["material_event_gain"]
        and c_a["material_event_gain"]
    ):
        return "CURRENT_MASK_ADAPTER_AND_SOURCE_POLICY_GAIN_SUPPORTED"
    if b_a["material_event_gain"]:
        return "DETECTOR_MODEL_OR_TAXONOMY_GAP_SUPPORTED"
    if any(pair["timing_only_gain"] for pair in pairwise.values()):
        return "TIMING_ONLY_GAIN_NO_ROUTE_CHANGE"
    return "DECISION_OR_MONOCULAR_OBSERVABILITY_BOTTLENECK_SUPPORTED"


def compare_reported_aggregates(
    benchmark: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    warnings: list[str],
) -> None:
    models = {model["id"]: model for model in benchmark.get("models", [])}
    for arm_id, computed in arms.items():
        reported = models.get(arm_id, {}).get("app_detector", {}).get("blindassist_metrics", {})
        checks = (
            ("criticalEventMissCount", "critical_event_miss_count"),
            ("falseAlertCount", "false_alert_count"),
            ("clearedPassedEventCount", "cleared_passed_event_count"),
        )
        for reported_key, computed_key in checks:
            if reported.get(reported_key) != computed.get(computed_key):
                warnings.append(
                    f"{arm_id}: report {reported_key}={reported.get(reported_key)!r} differs from "
                    f"truth-late recomputation {computed_key}={computed.get(computed_key)!r}; "
                    "the independent ledger is authoritative."
                )
        if isinstance(reported.get("falseAlertsPerMinute"), (int, float)) and not math.isclose(
            float(reported["falseAlertsPerMinute"]),
            float(computed["false_alerts_per_minute"]),
            abs_tol=1e-6,
        ):
            warnings.append(
                f"{arm_id}: report falseAlertsPerMinute differs from independent recomputation."
            )


def render_result_markdown(summary: dict[str, Any]) -> str:
    arms = summary["arms"]
    pairwise = summary["pairwise"]
    lines = [
        "# BlindAssist 信息上限三臂审计 D0 结果",
        "",
        f"结论：`{summary['terminal']}`。",
        "",
        "这是 90 帧、3 个 parent events 的 consumed Development pilot；不支持统计显著性、"
        "默认模型切换、真人助行或安全结论。",
        "",
        "| Arm | 事件命中/正事件 | 关键漏报 | 负事件误提醒 | 误提醒帧/9s | passed 清除 | 首次有效响应延迟 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm_id in ARM_IDS:
        arm = arms[arm_id]
        delays = ", ".join(
            f"{scene}={delay if delay is not None else 'MISS'}"
            for scene, delay in arm["first_effective_response_delay_frames_by_scene"].items()
        )
        lines.append(
            f"| `{arm_id}` | {arm['positive_event_hit_count']}/{arm['positive_event_count']} "
            f"| {arm['critical_event_miss_count']} | {arm['false_alert_event_count']} "
            f"| {arm['false_alert_count']} | "
            f"{arm['cleared_passed_event_count']}/{arm['passed_event_count']} | {delays} |"
        )
    lines.extend(
        [
            "",
            "## 成对判定",
            "",
        ]
    )
    for key in ("B_vs_A", "C_vs_B", "C_vs_A"):
        pair = pairwise[key]
        lines.append(
            f"- `{key}`：material={str(pair['material_event_gain']).lower()}，"
            f"timing_only={str(pair['timing_only_gain']).lower()}，"
            f"tradeoff={str(pair['tradeoff']).lower()}，"
            f"parent-event deltas={json.dumps(pair['parent_event_count_deltas'], ensure_ascii=False, sort_keys=True)}。"
        )
    lines.extend(
        [
            "",
            "## 归因边界",
            "",
            "- Arm B 是由同一 source mask/source regions 派生的风险框，不是独立实例框真值。",
            "- Arm C 经当前 mask adapter 后最多只向风险链送出一个框；B/C 还使用不同 source policy。",
            "- 因此 C 的收益只能归因于当前 mask adapter 与 source-specific policy 的组合，"
            "不能单独证明 bbox 几何上限；C 无收益也不能否定语义分割本身。",
            "- 事件/phase truth 来自既有 AI-review 派生，且只有两个正事件；结论只描述本 cohort。",
            "",
            "## Validator",
            "",
            f"- manifest SHA-256：`{summary['cohort']['manifest_sha256']}`",
            f"- decision kernel：`{summary['decision_kernel_contract_id']}`",
            f"- 状态：`{summary['validation_status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_audit(inputs: AuditInputs) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    benchmark = load_json(inputs.benchmark_json)
    manifest_rows = load_jsonl(inputs.manifest)
    cohort = validate_manifest(inputs, manifest_rows, errors, warnings)
    frame_maps = validate_benchmark_contract(
        benchmark,
        manifest_rows,
        inputs.manifest,
        errors,
        warnings,
    )
    validation = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "warnings": warnings,
        "benchmark_json": str(inputs.benchmark_json.resolve()),
        "manifest": str(inputs.manifest.resolve()),
    }
    if errors:
        summary = {
            "protocol_id": PROTOCOL_ID,
            "validation_status": "FAIL",
            "terminal": "NOT_EVALUABLE",
            "cohort": cohort,
            "errors": errors,
            "warnings": warnings,
        }
        return summary, validation, []

    grouped = event_rows(manifest_rows)
    ledger = [
        event
        for arm_id in ARM_IDS
        for event in event_ledger_for_arm(arm_id, grouped, frame_maps[arm_id])
    ]
    arms = {
        arm_id: aggregate_arm(
            [row for row in ledger if row["arm_id"] == arm_id],
            len(manifest_rows),
        )
        for arm_id in ARM_IDS
    }
    pairwise = {
        "B_vs_A": compare_arms("B_ORACLE_RISK_BOX", "A_CURRENT_YOLO", arms),
        "C_vs_B": compare_arms("C_ORACLE_RISK_MASK", "B_ORACLE_RISK_BOX", arms),
        "C_vs_A": compare_arms("C_ORACLE_RISK_MASK", "A_CURRENT_YOLO", arms),
    }
    compare_reported_aggregates(benchmark, arms, warnings)
    terminal = derive_terminal(pairwise)
    validation["warnings"] = warnings
    summary = {
        "protocol_id": PROTOCOL_ID,
        "validation_status": "PASS",
        "terminal": terminal,
        "claim_ceiling": "CONSUMED_DEVELOPMENT_CURRENT_CHAIN_ONLY",
        "cohort": cohort,
        "decision_kernel_contract_id": benchmark["decision_kernel_contract_id"],
        "risk_config": benchmark["risk_config"],
        "alert_profile": benchmark["alert_profile"],
        "synthetic_clock_frame_step_ms": benchmark["synthetic_clock_frame_step_ms"],
        "arms": arms,
        "pairwise": pairwise,
        "warnings": warnings,
    }
    return summary, validation, ledger


def write_outputs(
    inputs: AuditInputs,
    summary: dict[str, Any],
    validation: dict[str, Any],
    ledger: Iterable[dict[str, Any]],
) -> None:
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(inputs.output_dir / "validation.json", validation)
    write_json(inputs.output_dir / "summary.json", summary)
    ledger_rows = list(ledger)
    (inputs.output_dir / "event_ledger.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )
    if summary.get("validation_status") == "PASS":
        (inputs.output_dir / "result.md").write_text(
            render_result_markdown(summary),
            encoding="utf-8",
        )
    else:
        (inputs.output_dir / "result.md").write_text(
            "# BlindAssist 信息上限三臂审计 D0 结果\n\n"
            "结论：`NOT_EVALUABLE`。\n\n"
            + "\n".join(f"- {error}" for error in summary.get("errors", []))
            + "\n",
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="Skip RGB/mask content hashing (not permitted for the governed canonical run).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = AuditInputs(
        benchmark_json=args.benchmark_json.resolve(),
        manifest=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        verify_source_hashes=not args.skip_source_hashes,
    )
    try:
        summary, validation, ledger = evaluate_audit(inputs)
        write_outputs(inputs, summary, validation, ledger)
    except (OSError, ValidationFailure, ValueError, KeyError, TypeError) as error:
        inputs.output_dir.mkdir(parents=True, exist_ok=True)
        validation = {
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "errors": [f"{type(error).__name__}: {error}"],
            "warnings": [],
        }
        summary = {
            "protocol_id": PROTOCOL_ID,
            "validation_status": "FAIL",
            "terminal": "NOT_EVALUABLE",
            "errors": validation["errors"],
        }
        write_outputs(inputs, summary, validation, [])
        print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["validation_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
