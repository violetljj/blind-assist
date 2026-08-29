"""Run the single frozen PB12 RGB door part-parent topology experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


PROTOCOL_SCHEMA = "l10-named-poi-door-part-topology-protocol-v1"
PROTOCOL_SHA256 = "0f6f3096a305444a832f70543c7974afb10586092d00166eff113c7975e39a7e"
COHORT_SCHEMA = "l10-named-poi-door-part-topology-cohort-v1"
COHORT_SHA256 = "edb68294941c617d4c20f5a502065685f593b6faf221106bc3613dd5df91f71e"
RESULT_SCHEMA = "l10-named-poi-door-part-topology-development-result-v1"
CHECKPOINT_SHA256 = "4f05c7a88fc9350cef88157a60eeab040de549e6bdbfe4ec5f9c628835ee2f43"
CHECKPOINT_BYTES = 5_454_682
EXPECTED_NAMES = {0: "door", 1: "handle", 2: "cabinet door", 3: "refrigerator door"}
EXPECTED_INFERENCE = {
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "batch": 8,
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.7,
    "max_det": 300,
    "device": 0,
    "half": False,
    "rect": False,
    "augment": False,
    "agnostic_nms": False,
}
ROLE_COUNTS = {
    "ARCHITECTURAL_DOOR_WITH_VISIBLE_HANDLE": 4,
    "HANDLED_FURNITURE_DOOR_NEGATIVE": 2,
    "LARGE_DOORLESS_OPENING_OOD": 2,
}
POSITIVE_ROLE = "ARCHITECTURAL_DOOR_WITH_VISIBLE_HANDLE"
CONTROL_ROLES = {
    "HANDLED_FURNITURE_DOOR_NEGATIVE",
    "LARGE_DOORLESS_OPENING_OOD",
}
FILE_KINDS = {"rgb", "polygon", "scene_metadata"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A frozen input, runtime, or output contract was not satisfied."""


def _fail(code: str, detail: str | None = None) -> None:
    raise ContractError(f"{code}:{detail}" if detail else code)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail("DUPLICATE_JSON_KEY", key)
        value[key] = item
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_no_duplicates,
            parse_constant=lambda token: _fail("NONFINITE_JSON_NUMBER", token),
        )
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}_JSON_READ_FAILED", str(exc))
    if not isinstance(value, dict):
        _fail(f"{label}_JSON_ROOT_NOT_OBJECT")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail("FILE_HASH_READ_FAILED", f"{path}:{exc}")
    return digest.hexdigest()


def _existing_file(value: Path, label: str) -> Path:
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}_PATH_INVALID", str(exc))
    if not path.is_file():
        _fail(f"{label}_NOT_REGULAR_FILE", str(path))
    return path


def _existing_directory(value: Path, label: str) -> Path:
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}_PATH_INVALID", str(exc))
    if not path.is_dir():
        _fail(f"{label}_NOT_DIRECTORY", str(path))
    return path


def _new_output(value: Path) -> Path:
    if value.exists() or value.is_symlink():
        _fail("OUTPUT_ALREADY_EXISTS", str(value))
    try:
        parent = value.parent.resolve(strict=True)
    except OSError as exc:
        _fail("OUTPUT_PARENT_INVALID", str(exc))
    if not parent.is_dir():
        _fail("OUTPUT_PARENT_NOT_DIRECTORY", str(parent))
    path = parent / value.name
    if path.exists() or path.is_symlink():
        _fail("OUTPUT_ALREADY_EXISTS", str(path))
    return path


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}_NOT_OBJECT")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{label}_INVALID")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label}_INVALID_SHA256")
    return value


def _relative_posix(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if "\\" in text or "\x00" in text or ":" in text:
        _fail(f"{label}_NOT_CANONICAL_RELATIVE_POSIX", text)
    pure = PurePosixPath(text)
    if pure.is_absolute() or pure.as_posix() != text:
        _fail(f"{label}_NOT_CANONICAL_RELATIVE_POSIX", text)
    if any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{label}_NOT_CANONICAL_RELATIVE_POSIX", text)
    return text


def _manifest_file(root: Path, spec: Any, label: str) -> tuple[Path, dict[str, Any]]:
    obj = _require_object(spec, label)
    if set(obj) != {"path", "bytes", "sha256"}:
        _fail(f"{label}_FIELDS_MISMATCH", ",".join(sorted(obj)))
    relative = _relative_posix(obj["path"], f"{label}_PATH")
    expected_hash = _require_sha256(obj["sha256"], label)
    expected_bytes = obj["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        _fail(f"{label}_BYTES_INVALID")
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"{label}_PATH_OUTSIDE_ROOT", str(exc))
    if not path.is_file():
        _fail(f"{label}_NOT_REGULAR_FILE", relative)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        _fail(f"{label}_SIZE_MISMATCH", f"{actual_bytes}:{expected_bytes}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        _fail(f"{label}_HASH_MISMATCH", f"{actual_hash}:{expected_hash}")
    return path, {"path": relative, "bytes": actual_bytes, "sha256": actual_hash}


def _validate_protocol(protocol: dict[str, Any], protocol_hash: str) -> None:
    if protocol_hash != PROTOCOL_SHA256:
        _fail("PROTOCOL_HASH_MISMATCH", f"{protocol_hash}:{PROTOCOL_SHA256}")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        _fail("PROTOCOL_SCHEMA_MISMATCH")
    if protocol.get("status") != "FROZEN_BEFORE_COHORT_MODEL_OUTPUT":
        _fail("PROTOCOL_NOT_FROZEN")
    if protocol.get("cohort_freeze", {}).get("roles") != ROLE_COUNTS:
        _fail("PROTOCOL_ROLE_COUNTS_MISMATCH")
    source = _require_object(protocol.get("information_source"), "PROTOCOL_INFORMATION_SOURCE")
    if source.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        _fail("PROTOCOL_CHECKPOINT_HASH_MISMATCH")
    inference = _require_object(protocol.get("inference"), "PROTOCOL_INFERENCE")
    protocol_inference = {key: inference.get(key) for key in EXPECTED_INFERENCE if key != "checkpoint_sha256"}
    expected = {key: value for key, value in EXPECTED_INFERENCE.items() if key != "checkpoint_sha256"}
    if protocol_inference != expected:
        _fail("PROTOCOL_INFERENCE_MISMATCH", json.dumps(protocol_inference, sort_keys=True))


def _validate_cohort(cohort: dict[str, Any], cohort_hash: str) -> list[dict[str, Any]]:
    if cohort_hash != COHORT_SHA256:
        _fail("COHORT_HASH_MISMATCH", f"{cohort_hash}:{COHORT_SHA256}")
    if cohort.get("schema") != COHORT_SCHEMA:
        _fail("COHORT_SCHEMA_MISMATCH")
    if cohort.get("status") != "FROZEN_BEFORE_MODEL_OUTPUT":
        _fail("COHORT_NOT_FROZEN")
    protocol = _require_object(cohort.get("protocol"), "COHORT_PROTOCOL")
    if protocol != {"schema": PROTOCOL_SCHEMA, "sha256": PROTOCOL_SHA256}:
        _fail("COHORT_PROTOCOL_MISMATCH")
    if cohort.get("inference") != EXPECTED_INFERENCE:
        _fail("COHORT_INFERENCE_MISMATCH")
    frames = cohort.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        _fail("COHORT_FRAME_COUNT_MISMATCH")
    if [frame.get("index") for frame in frames if isinstance(frame, dict)] != list(range(1, 9)):
        _fail("COHORT_FRAME_INDICES_MISMATCH")
    if not all(isinstance(frame, dict) for frame in frames):
        _fail("COHORT_FRAME_NOT_OBJECT")
    roles = Counter(frame.get("role") for frame in frames)
    if dict(roles) != ROLE_COUNTS:
        _fail("COHORT_ROLE_COUNTS_MISMATCH", json.dumps(roles, sort_keys=True))
    sequences = [_require_string(frame.get("capture_sequence_id"), "CAPTURE_SEQUENCE_ID") for frame in frames]
    if len(set(sequences)) != 8:
        _fail("COHORT_CAPTURE_SEQUENCES_NOT_DISTINCT")
    buckets = Counter(_require_string(frame.get("sensor_source_bucket"), "SENSOR_SOURCE_BUCKET") for frame in frames)
    if len(buckets) < 4 or max(buckets.values()) > 3:
        _fail("COHORT_BUCKET_DIVERSITY_MISMATCH", json.dumps(buckets, sort_keys=True))
    canonical = [_relative_posix(frame.get("canonical_source_path"), "CANONICAL_SOURCE_PATH") for frame in frames]
    if len(set(canonical)) != 8:
        _fail("COHORT_CANONICAL_PATHS_NOT_DISTINCT")
    for frame in frames:
        size = frame.get("source_image_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in size)
        ):
            _fail("SOURCE_IMAGE_SIZE_INVALID", str(frame.get("index")))
        files = _require_object(frame.get("files"), f"FRAME_{frame['index']}_FILES")
        if set(files) != FILE_KINDS:
            _fail("FRAME_FILE_KINDS_MISMATCH", str(frame["index"]))
    return frames


def _normalized_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        result = {index: str(name) for index, name in enumerate(value)}
    elif isinstance(value, dict):
        try:
            result = {int(index): str(name) for index, name in value.items()}
        except (TypeError, ValueError) as exc:
            _fail("MODEL_CLASS_NAMES_INVALID", str(exc))
    else:
        _fail("MODEL_CLASS_NAMES_INVALID")
    return result


def _finite_box(value: Sequence[Any], label: str) -> list[float]:
    if len(value) != 4:
        _fail(f"{label}_BOX_LENGTH_INVALID")
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box):
        _fail(f"{label}_BOX_NONFINITE")
    if box[2] < box[0] or box[3] < box[1]:
        _fail(f"{label}_BOX_ORDER_INVALID")
    return box


def _detections_from_result(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None:
        return []
    if boxes.data.device.type != "cuda":
        _fail("RESULT_BOXES_NOT_ON_CUDA", str(boxes.data.device))
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    if not (len(xyxy) == len(confidences) == len(classes)):
        _fail("DETECTION_ARRAY_LENGTH_MISMATCH")
    detections: list[dict[str, Any]] = []
    for index, (raw_box, raw_confidence, raw_class) in enumerate(zip(xyxy, confidences, classes)):
        class_index = int(raw_class)
        confidence = float(raw_confidence)
        if class_index not in EXPECTED_NAMES or not math.isfinite(confidence):
            _fail("DETECTION_CLASS_OR_CONFIDENCE_INVALID", str(index))
        detections.append(
            {
                "class_index": class_index,
                "class_name": EXPECTED_NAMES[class_index],
                "confidence": confidence,
                "box_xyxy": _finite_box(raw_box, f"DETECTION_{index}"),
            }
        )
    detections.sort(
        key=lambda item: (
            item["class_index"],
            -item["confidence"],
            *item["box_xyxy"],
        )
    )
    for index, detection in enumerate(detections):
        detection["detection_id"] = f"d{index:03d}"
    return detections


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _assign_topology(detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    parents = [item for item in detections if item["class_index"] in {0, 2, 3}]
    handles = [item for item in detections if item["class_index"] == 1]
    assignments: list[dict[str, Any]] = []
    authorized = False
    for handle in handles:
        x1, y1, x2, y2 = handle["box_xyxy"]
        center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        candidates = [
            parent
            for parent in parents
            if parent["box_xyxy"][0] <= center[0] <= parent["box_xyxy"][2]
            and parent["box_xyxy"][1] <= center[1] <= parent["box_xyxy"][3]
        ]
        candidates.sort(
            key=lambda parent: (
                _area(parent["box_xyxy"]),
                -parent["confidence"],
                parent["class_index"],
                *parent["box_xyxy"],
            )
        )
        parent = candidates[0] if candidates else None
        if parent is not None and parent["class_index"] == 0:
            authorized = True
        assignments.append(
            {
                "handle_detection_id": handle["detection_id"],
                "handle_center_xy": center,
                "enclosing_parent_detection_ids_in_assignment_order": [
                    candidate["detection_id"] for candidate in candidates
                ],
                "assigned_parent_detection_id": parent["detection_id"] if parent else None,
                "assigned_parent_class": parent["class_name"] if parent else None,
            }
        )
    return assignments, authorized


def _aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    authorized_by_role = {
        role: sum(row["role"] == role and row["door_handle_topology"] for row in rows)
        for role in ROLE_COUNTS
    }
    positive_authorized = authorized_by_role[POSITIVE_ROLE]
    negative_total = sum(ROLE_COUNTS[role] for role in CONTROL_ROLES)
    negative_authorized = sum(authorized_by_role[role] for role in CONTROL_ROLES)
    true_positive_rate = positive_authorized / ROLE_COUNTS[POSITIVE_ROLE]
    false_positive_rate = negative_authorized / negative_total
    true_negative_rate = 1.0 - false_positive_rate
    balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
    gate_met = (
        len(rows) == 8
        and positive_authorized == 4
        and authorized_by_role["HANDLED_FURNITURE_DOOR_NEGATIVE"] == 0
        and authorized_by_role["LARGE_DOORLESS_OPENING_OOD"] == 0
        and balanced_accuracy == 1.0
    )
    metrics = {
        "frames": len(rows),
        "authorized_frames_by_role": authorized_by_role,
        "door_positive_recall": true_positive_rate,
        "control_false_positive_rate": false_positive_rate,
        "true_negative_rate": true_negative_rate,
        "balanced_accuracy": balanced_accuracy,
    }
    gate = {
        "four_of_four_architectural_doors_authorized": positive_authorized == 4,
        "zero_of_two_handled_furniture_controls_authorized": authorized_by_role[
            "HANDLED_FURNITURE_DOOR_NEGATIVE"
        ]
        == 0,
        "zero_of_two_large_openings_authorized": authorized_by_role[
            "LARGE_DOORLESS_OPENING_OOD"
        ]
        == 0,
        "balanced_accuracy_is_one": balanced_accuracy == 1.0,
        "development_gate_met": gate_met,
        "decision": (
            "L10_PB12_DOOR_PART_TOPOLOGY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_PB12_DOOR_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET"
        ),
    }
    return metrics, gate


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        _fail("OUTPUT_ALREADY_EXISTS", str(path))
    except OSError as exc:
        _fail("OUTPUT_WRITE_FAILED", str(exc))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _run(
    protocol_argument: Path,
    cohort_argument: Path,
    extracted_root_argument: Path,
    checkpoint_argument: Path,
    output_argument: Path,
) -> dict[str, Any]:
    protocol_path = _existing_file(protocol_argument, "PROTOCOL")
    cohort_path = _existing_file(cohort_argument, "COHORT")
    extracted_root = _existing_directory(extracted_root_argument, "EXTRACTED_ROOT")
    checkpoint_path = _existing_file(checkpoint_argument, "CHECKPOINT")
    output_path = _new_output(output_argument)

    protocol_hash = _sha256(protocol_path)
    protocol = _read_json(protocol_path, "PROTOCOL")
    _validate_protocol(protocol, protocol_hash)
    cohort_hash = _sha256(cohort_path)
    cohort = _read_json(cohort_path, "COHORT")
    frames = _validate_cohort(cohort, cohort_hash)
    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
        _fail("CHECKPOINT_SIZE_MISMATCH")
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != CHECKPOINT_SHA256:
        _fail("CHECKPOINT_HASH_MISMATCH", f"{checkpoint_hash}:{CHECKPOINT_SHA256}")

    try:
        import torch
        import ultralytics
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        _fail("INFERENCE_RUNTIME_IMPORT_FAILED", str(exc))
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        _fail("CUDA_UNAVAILABLE")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    image_paths: list[Path] = []
    input_receipts: list[dict[str, Any]] = []
    for frame in frames:
        files: dict[str, dict[str, Any]] = {}
        paths: dict[str, Path] = {}
        for kind in sorted(FILE_KINDS):
            path, receipt = _manifest_file(
                extracted_root, frame["files"][kind], f"FRAME_{frame['index']}_{kind.upper()}"
            )
            paths[kind] = path
            files[kind] = receipt
        try:
            with Image.open(paths["rgb"]) as image:
                size = [int(image.width), int(image.height)]
                image.verify()
        except Exception as exc:
            _fail("RGB_DECODE_FAILED", f"frame={frame['index']}:{exc}")
        if size != frame["source_image_size"]:
            _fail("RGB_SIZE_MISMATCH", f"frame={frame['index']}:{size}:{frame['source_image_size']}")
        image_paths.append(paths["rgb"])
        input_receipts.append(files)

    model = YOLO(str(checkpoint_path), task="detect")
    model_names = _normalized_names(model.names)
    if model_names != EXPECTED_NAMES:
        _fail("MODEL_CLASS_NAMES_MISMATCH", json.dumps(model_names, sort_keys=True))
    results = model.predict(
        source=[str(path) for path in image_paths],
        batch=EXPECTED_INFERENCE["batch"],
        imgsz=EXPECTED_INFERENCE["imgsz"],
        conf=EXPECTED_INFERENCE["conf"],
        iou=EXPECTED_INFERENCE["iou"],
        max_det=EXPECTED_INFERENCE["max_det"],
        device=EXPECTED_INFERENCE["device"],
        half=EXPECTED_INFERENCE["half"],
        rect=EXPECTED_INFERENCE["rect"],
        augment=EXPECTED_INFERENCE["augment"],
        agnostic_nms=EXPECTED_INFERENCE["agnostic_nms"],
        save=False,
        verbose=False,
        stream=False,
    )
    torch.cuda.synchronize(0)
    if len(results) != len(frames):
        _fail("INFERENCE_RESULT_COUNT_MISMATCH", f"{len(results)}:{len(frames)}")

    rows: list[dict[str, Any]] = []
    for frame, expected_path, receipts, result in zip(frames, image_paths, input_receipts, results):
        detections = _detections_from_result(result)
        assignments, authorized = _assign_topology(detections)
        rows.append(
            {
                "index": frame["index"],
                "frame_id": frame["frame_id"],
                "capture_sequence_id": frame["capture_sequence_id"],
                "sensor_source_bucket": frame["sensor_source_bucket"],
                "role": frame["role"],
                "canonical_source_path": frame["canonical_source_path"],
                "model_input_path": str(expected_path),
                "backend_display_path": str(result.path),
                "audit_note": frame["audit_note"],
                "files": receipts,
                "source_image_size": frame["source_image_size"],
                "detections": detections,
                "handle_parent_assignments": assignments,
                "door_handle_topology": authorized,
            }
        )

    metrics, gate = _aggregate(rows)
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "L10-PB12 Door Part-Parent Topology",
        "inputs": {
            "protocol": {"path": str(protocol_path), "schema": PROTOCOL_SCHEMA, "sha256": protocol_hash},
            "cohort": {
                "path": str(cohort_path),
                "schema": COHORT_SCHEMA,
                "sha256": cohort_hash,
                "status": cohort["status"],
            },
            "extracted_root": str(extracted_root),
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_hash,
                "classes_by_index": {str(index): name for index, name in EXPECTED_NAMES.items()},
            },
        },
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_runtime": torch.version.cuda,
            "actual_device_index": 0,
            "actual_device_name": properties.name,
            "actual_device_capability": list(torch.cuda.get_device_capability(0)),
            "device_total_memory_bytes": properties.total_memory,
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "inference": EXPECTED_INFERENCE,
        "execution_history": [
            {
                "attempt": 1,
                "status": "ABORTED_WITHOUT_RESULT_FILE",
                "error": "RESULT_SOURCE_PATH_INVALID: ultralytics batch-list input returned synthetic display path image0.jpg",
                "model_outputs_or_metrics_observed_by_operator": False,
                "result_file_created": False,
                "recovery": "Removed only the non-scoring backend display-path assertion; retained positional result binding to the frozen ordered and hashed eight-image input list. Protocol, cohort, checkpoint, inference parameters, topology, and gate were unchanged.",
            },
            {
                "attempt": 2,
                "status": "THIS_RESULT",
                "reason": "ONE_MECHANICAL_REPLAY_AFTER_UNOBSERVED_NO_OUTPUT_HARNESS_FAILURE",
            },
        ],
        "topology": {
            "parent_classes": ["door", "cabinet door", "refrigerator door"],
            "child_class": "handle",
            "enclosure": "INCLUSIVE_HANDLE_BOX_CENTER_IN_PARENT_BOX",
            "assignment_order": "MIN_PARENT_AREA_THEN_HIGHER_CONFIDENCE_THEN_LOWER_CLASS_INDEX_THEN_LEXICOGRAPHIC_BOX",
            "authorization": "AT_LEAST_ONE_HANDLE_ASSIGNED_TO_CLASS_0_DOOR",
        },
        "frames": rows,
        "metrics": metrics,
        "gate": gate,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json_new(output_path, result)
    return result


def _self_test() -> dict[str, Any]:
    detections = [
        {"detection_id": "d000", "class_index": 0, "class_name": "door", "confidence": 0.8, "box_xyxy": [0.0, 0.0, 10.0, 10.0]},
        {"detection_id": "d001", "class_index": 2, "class_name": "cabinet door", "confidence": 0.9, "box_xyxy": [1.0, 1.0, 9.0, 9.0]},
        {"detection_id": "d002", "class_index": 1, "class_name": "handle", "confidence": 0.7, "box_xyxy": [4.0, 4.0, 6.0, 6.0]},
        {"detection_id": "d003", "class_index": 1, "class_name": "handle", "confidence": 0.6, "box_xyxy": [0.0, 0.0, 0.0, 0.0]},
    ]
    assignments, authorized = _assign_topology(detections)
    if assignments[0]["assigned_parent_detection_id"] != "d001":
        raise AssertionError("smallest enclosing parent did not win")
    if assignments[1]["assigned_parent_detection_id"] != "d000" or not authorized:
        raise AssertionError("inclusive boundary or door authorization failed")
    rows = []
    for index in range(8):
        if index < 4:
            role = POSITIVE_ROLE
            value = True
        elif index < 6:
            role = "HANDLED_FURNITURE_DOOR_NEGATIVE"
            value = False
        else:
            role = "LARGE_DOORLESS_OPENING_OOD"
            value = False
        rows.append({"role": role, "door_handle_topology": value})
    metrics, gate = _aggregate(rows)
    if metrics["balanced_accuracy"] != 1.0 or not gate["development_gate_met"]:
        raise AssertionError("perfect frozen gate self-test failed")
    rows[0]["door_handle_topology"] = False
    _, failed_gate = _aggregate(rows)
    if failed_gate["development_gate_met"]:
        raise AssertionError("imperfect cohort incorrectly passed")
    return {"status": "PASS", "assignments": assignments, "perfect_gate_decision": gate["decision"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen PB12 door part-parent topology evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run deterministic synthetic assignment and gate checks only.")
    run = subparsers.add_parser("run", help="Run the one frozen eight-image CUDA evaluation.")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--extracted-root", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "self-test":
            print(json.dumps(_self_test(), ensure_ascii=False, sort_keys=True))
            return 0
        result = _run(args.protocol, args.cohort, args.extracted_root, args.checkpoint, args.output)
    except ContractError as exc:
        print(f"PB12_CONTRACT_ERROR:{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "decision": result["gate"]["decision"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
