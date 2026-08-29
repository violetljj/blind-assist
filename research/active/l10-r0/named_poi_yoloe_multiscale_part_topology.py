#!/usr/bin/env python3
"""Run the frozen L10-PB14 YOLOE multiscale pixel-part topology gate."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "l10-named-poi-yoloe-multiscale-part-topology-protocol-v1"
COHORT_SCHEMA = "l10-named-poi-yoloe-multiscale-part-topology-cohort-v1"
RESULT_SCHEMA = "l10-named-poi-yoloe-multiscale-part-topology-development-result-v1"
PROTOCOL_SHA256 = "d79d2c09fd47618aad6b8d79d99431613646e7d3e53dea723c991d745b80bf20"
# Root replaces this single conspicuous value after the queue is frozen and hashed.
COHORT_SHA256 = "aded42ce06863c9fa9de6fb164dde739e4ec890f505a92c3019a10669d224c90"

WEIGHTS_RELATIVE = Path("artifacts.local/models/yoloe-26n-seg.pt")
WEIGHTS_BYTES = 11_710_443
WEIGHTS_SHA256 = "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b"
TEXT_ENCODER_RELATIVE = Path("artifacts.local/models/mobileclip2_b.ts")
TEXT_ENCODER_BYTES = 253_794_476
TEXT_ENCODER_SHA256 = "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
PYTHON_EXE_SHA256 = "21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082"

POSITIVE_ROLE = "ARCHITECTURAL_DOOR_WITH_VISIBLE_OPERATION_PART"
FURNITURE_ROLE = "HANDLED_FURNITURE_DOOR_NEGATIVE"
OOD_ROLE = "LARGE_DOORLESS_OPENING_OOD"
ROLE_COUNTS = {POSITIVE_ROLE: 4, FURNITURE_ROLE: 2, OOD_ROLE: 2}
FILE_KINDS = {"rgb", "depth", "intrinsics", "polygon", "scene_metadata"}

PROMPTS = [
    {"class_index": 0, "class_id": "architectural_leaf", "text": "architectural door leaf", "role": "PARENT"},
    {"class_index": 1, "class_id": "architectural_frame", "text": "door frame", "role": "DIAGNOSTIC_ONLY"},
    {"class_index": 2, "class_id": "operation_part", "text": "door handle", "role": "CHILD"},
    {"class_index": 3, "class_id": "hinge", "text": "door hinge", "role": "CHILD"},
    {"class_index": 4, "class_id": "cabinet_door", "text": "cabinet door", "role": "COMPETING_PARENT"},
    {"class_index": 5, "class_id": "closet_door", "text": "closet door", "role": "COMPETING_PARENT"},
    {"class_index": 6, "class_id": "refrigerator_door", "text": "refrigerator door", "role": "COMPETING_PARENT"},
    {"class_index": 7, "class_id": "doorless_opening", "text": "doorless wall opening", "role": "COMPETING_PARENT"},
]
CLASS_IDS = [item["class_id"] for item in PROMPTS]
CLASS_TEXTS = [item["text"] for item in PROMPTS]
CLASS_BY_INDEX = {item["class_index"]: item for item in PROMPTS}
PARENT_IDS = {
    "architectural_leaf",
    "cabinet_door",
    "closet_door",
    "refrigerator_door",
    "doorless_opening",
}
CHILD_IDS = {"operation_part", "hinge"}
PARENT_PRIORITY = {
    "cabinet_door": 0,
    "closet_door": 1,
    "refrigerator_door": 2,
    "doorless_opening": 3,
    "architectural_leaf": 4,
}
INFERENCE = {
    "batch": 1,
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.7,
    "max_det": 300,
    "device": 0,
    "half": False,
    "rect": False,
    "augment": False,
    "agnostic_nms": False,
    "retina_masks": True,
    "save": False,
    "maximum_parent_crops_per_frame": 12,
}
EXPECTED_RUNTIME = {
    "python": "3.11.9",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "ultralytics": "8.4.102",
    "clip": "1.0",
    "numpy": "2.4.4",
    "pillow": "12.2.0",
}


class ContractError(RuntimeError):
    """Raised before a valid formal result can be written."""


def _fail(code: str, detail: str | None = None) -> None:
    raise ContractError(code if detail is None else f"{code}:{detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_file(value: Path, label: str) -> Path:
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}_MISSING", str(exc))
    if not path.is_file():
        _fail(f"{label}_NOT_FILE", str(path))
    return path


def _existing_directory(value: Path, label: str) -> Path:
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}_MISSING", str(exc))
    if not path.is_dir():
        _fail(f"{label}_NOT_DIRECTORY", str(path))
    return path


def _new_output(value: Path) -> Path:
    path = value.resolve(strict=False)
    if path.exists():
        _fail("OUTPUT_ALREADY_EXISTS", str(path))
    if not path.parent.is_dir():
        _fail("OUTPUT_PARENT_MISSING", str(path.parent))
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}_JSON_INVALID", str(exc))
    if not isinstance(value, dict):
        _fail(f"{label}_NOT_OBJECT")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label}_INVALID")
    return value


def _relative_posix(value: Any, label: str) -> Path:
    text = _require_string(value, label)
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        _fail(f"{label}_UNSAFE", text)
    return Path(*posix.parts)


def _exact_mapping(actual: Any, expected: dict[str, Any], label: str) -> None:
    if not isinstance(actual, dict) or actual != expected:
        _fail(f"{label}_MISMATCH", json.dumps(actual, sort_keys=True, default=str))


def _validate_protocol(protocol: dict[str, Any], actual_hash: str) -> None:
    if actual_hash != PROTOCOL_SHA256:
        _fail("PROTOCOL_HASH_MISMATCH", f"{actual_hash}:{PROTOCOL_SHA256}")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "FROZEN_BEFORE_COHORT_MODEL_OUTPUT":
        _fail("PROTOCOL_IDENTITY_MISMATCH")
    if protocol.get("class_prompts_in_order") != PROMPTS:
        _fail("PROTOCOL_PROMPTS_MISMATCH")
    source = protocol.get("information_source")
    if not isinstance(source, dict):
        _fail("PROTOCOL_INFORMATION_SOURCE_MISSING")
    expected_source = {
        "weights_path": WEIGHTS_RELATIVE.as_posix(),
        "weights_bytes": WEIGHTS_BYTES,
        "weights_sha256": WEIGHTS_SHA256,
        "text_encoder_path": TEXT_ENCODER_RELATIVE.as_posix(),
        "text_encoder_bytes": TEXT_ENCODER_BYTES,
        "text_encoder_sha256": TEXT_ENCODER_SHA256,
        "checkpoint_model_type": "YOLOESegModel/segment",
        "checkpoint_ultralytics": "8.4.24",
        "checkpoint_training_commit": "6bec7a00ead88ac515ad0c4b130e19031f1afcc9",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            _fail("PROTOCOL_SOURCE_MISMATCH", f"{key}:{source.get(key)!r}:{expected!r}")
    runtime = protocol.get("runtime")
    if not isinstance(runtime, dict):
        _fail("PROTOCOL_RUNTIME_MISSING")
    for key, expected in {**EXPECTED_RUNTIME, "python_executable_sha256": PYTHON_EXE_SHA256}.items():
        if runtime.get(key) != expected:
            _fail("PROTOCOL_RUNTIME_MISMATCH", f"{key}:{runtime.get(key)!r}:{expected!r}")
    if runtime.get("YOLO_AUTOINSTALL") != "false":
        _fail("PROTOCOL_AUTOINSTALL_MISMATCH")
    inference = protocol.get("inference")
    if not isinstance(inference, dict):
        _fail("PROTOCOL_INFERENCE_MISSING")
    for key, expected in INFERENCE.items():
        if inference.get(key) != expected:
            _fail("PROTOCOL_INFERENCE_MISMATCH", f"{key}:{inference.get(key)!r}:{expected!r}")
    topology = protocol.get("mask_topology")
    if not isinstance(topology, dict):
        _fail("PROTOCOL_TOPOLOGY_MISSING")
    if topology.get("parent_class_ids") != [
        "architectural_leaf", "cabinet_door", "closet_door", "refrigerator_door", "doorless_opening"
    ] or topology.get("child_class_ids") != ["operation_part", "hinge"]:
        _fail("PROTOCOL_TOPOLOGY_CLASSES_MISMATCH")
    if topology.get("learned_or_selected_postprocessing_thresholds_beyond_yoloe") != 0:
        _fail("PROTOCOL_EXTRA_THRESHOLD_MISMATCH")


def _validate_cohort(cohort: dict[str, Any], actual_hash: str) -> list[dict[str, Any]]:
    if len(COHORT_SHA256) != 64 or any(char not in "0123456789abcdef" for char in COHORT_SHA256):
        _fail("COHORT_SHA256_PENDING_ROOT_FREEZE")
    if actual_hash != COHORT_SHA256:
        _fail("COHORT_HASH_MISMATCH", f"{actual_hash}:{COHORT_SHA256}")
    if cohort.get("schema") != COHORT_SCHEMA or cohort.get("status") != "FROZEN_BEFORE_MODEL_OUTPUT":
        _fail("COHORT_IDENTITY_MISMATCH")
    protocol = cohort.get("protocol")
    if protocol != {"schema": PROTOCOL_SCHEMA, "sha256": PROTOCOL_SHA256}:
        _fail("COHORT_PROTOCOL_BINDING_MISMATCH")
    model = cohort.get("model")
    if not isinstance(model, dict):
        _fail("COHORT_MODEL_BINDING_MISSING")
    if model.get("weights_sha256") != WEIGHTS_SHA256 or model.get("text_encoder_sha256") != TEXT_ENCODER_SHA256:
        _fail("COHORT_MODEL_BINDING_MISMATCH")
    if set(cohort) != {
        "schema", "status", "frozen_at", "protocol", "source", "model", "inference", "diversity", "frames"
    }:
        _fail("COHORT_TOP_LEVEL_FIELDS_MISMATCH", repr(sorted(cohort)))
    source = cohort.get("source")
    expected_source = {
        "dataset": "SUN RGB-D",
        "archive_path": "F:/ba-data/blindassist-artifacts-20260805/datasets/sunrgbd-2015/SUNRGBD.zip",
        "archive_bytes": 6885481608,
        "archive_sha256": "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c",
        "extracted_root": "artifacts.local/datasets/sunrgbd-pb14-door-part-audit-pool-v1",
        "official_members": 40,
        "official_member_bytes": 1993686,
        "official_member_verification": "Every extracted member was matched to the official ZIP member by byte count and SHA-256 before freeze.",
        "prior_pb11_sequence_overlap": 0,
        "prior_pb12_sequence_overlap": 0,
        "prior_pb13_sequence_overlap": 0,
        "selection_provenance": "Official annotation prefilter plus human RGB audit; no YOLOE or other PB14 learned output was observed on these eight frames before freeze.",
    }
    _exact_mapping(source, expected_source, "COHORT_SOURCE")
    expected_model = {
        "weights_path": WEIGHTS_RELATIVE.as_posix(),
        "weights_bytes": WEIGHTS_BYTES,
        "weights_sha256": WEIGHTS_SHA256,
        "text_encoder_path": TEXT_ENCODER_RELATIVE.as_posix(),
        "text_encoder_bytes": TEXT_ENCODER_BYTES,
        "text_encoder_sha256": TEXT_ENCODER_SHA256,
    }
    _exact_mapping(model, expected_model, "COHORT_MODEL")
    expected_inference = {
        "class_ids_in_order": CLASS_IDS,
        "class_texts_in_order": CLASS_TEXTS,
        **INFERENCE,
    }
    _exact_mapping(cohort.get("inference"), expected_inference, "COHORT_INFERENCE")
    frames = cohort.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        _fail("COHORT_FRAME_COUNT_MISMATCH")
    if [frame.get("index") for frame in frames if isinstance(frame, dict)] != list(range(1, 9)):
        _fail("COHORT_FRAME_INDEX_MISMATCH")
    roles = Counter(frame.get("role") for frame in frames)
    if roles != Counter(ROLE_COUNTS):
        _fail("COHORT_ROLE_COUNTS_MISMATCH", json.dumps(roles, sort_keys=True))
    sequences = [_require_string(frame.get("capture_sequence_id"), "CAPTURE_SEQUENCE_ID") for frame in frames]
    canonical = [_relative_posix(frame.get("canonical_source_path"), "CANONICAL_SOURCE_PATH") for frame in frames]
    if len(set(sequences)) != 8 or len(set(canonical)) != 8:
        _fail("COHORT_IDENTITY_NOT_DISTINCT")
    buckets = Counter(_require_string(frame.get("sensor_source_bucket"), "SENSOR_SOURCE_BUCKET") for frame in frames)
    if len(buckets) < 4 or max(buckets.values()) > 3:
        _fail("COHORT_BUCKET_DIVERSITY_MISMATCH", json.dumps(buckets, sort_keys=True))
    expected_frame_fields = {
        "index", "frame_id", "capture_sequence_id", "sensor_source_bucket", "role",
        "canonical_source_path", "audit_note", "source_image_size", "official_evidence", "files",
    }
    for frame in frames:
        if set(frame) != expected_frame_fields:
            _fail("FRAME_FIELDS_MISMATCH", f"{frame.get('index')}:{sorted(frame)}")
        size = frame.get("source_image_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size)
        ):
            _fail("SOURCE_IMAGE_SIZE_INVALID", str(frame.get("index")))
        files = frame.get("files")
        if not isinstance(files, dict) or set(files) != FILE_KINDS:
            _fail("FRAME_FILE_KINDS_MISMATCH", str(frame.get("index")))
        for kind, entry in files.items():
            if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
                _fail("FRAME_FILE_ENTRY_FIELDS_MISMATCH", f"{frame.get('index')}:{kind}")
        evidence = frame.get("official_evidence")
        if not isinstance(evidence, dict) or set(evidence) != {"annotation_prefilter", "supporting_object_names"}:
            _fail("OFFICIAL_EVIDENCE_FIELDS_MISMATCH", str(frame.get("index")))
        expected_prefilter = "EXACT_DOOR" if frame.get("role") == POSITIVE_ROLE else "NO_EXACT_DOOR"
        if evidence.get("annotation_prefilter") != expected_prefilter:
            _fail("OFFICIAL_EVIDENCE_ROLE_MISMATCH", str(frame.get("index")))
        names = evidence.get("supporting_object_names")
        if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
            _fail("OFFICIAL_EVIDENCE_NAMES_INVALID", str(frame.get("index")))
    expected_diversity = {
        "distinct_capture_sequences": 8,
        "sensor_source_buckets": dict(buckets),
        "maximum_frames_in_one_bucket": 3,
    }
    _exact_mapping(cohort.get("diversity"), expected_diversity, "COHORT_DIVERSITY")
    return frames


def _validate_artifact(path: Path, expected_path: Path, expected_bytes: int, expected_hash: str, label: str) -> dict[str, Any]:
    resolved = _existing_file(path, label)
    canonical = _existing_file(expected_path, f"EXPECTED_{label}")
    if resolved != canonical:
        _fail(f"{label}_PATH_MISMATCH", f"{resolved}:{canonical}")
    actual_bytes = resolved.stat().st_size
    actual_hash = _sha256(resolved)
    if actual_bytes != expected_bytes or actual_hash != expected_hash:
        _fail(f"{label}_IDENTITY_MISMATCH", f"{actual_bytes}:{actual_hash}")
    return {"path": str(resolved), "bytes": actual_bytes, "sha256": actual_hash}


def _manifest_file(root: Path, entry: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(entry, dict):
        _fail(f"{label}_ENTRY_INVALID")
    relative = _relative_posix(entry.get("path"), f"{label}_PATH")
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"{label}_PATH_INVALID", str(exc))
    if not path.is_file():
        _fail(f"{label}_NOT_FILE", str(path))
    actual_bytes = path.stat().st_size
    actual_hash = _sha256(path)
    if actual_bytes != entry.get("bytes") or actual_hash != entry.get("sha256"):
        _fail(f"{label}_MISMATCH", f"{actual_bytes}:{actual_hash}")
    return path, {"path": str(path), "bytes": actual_bytes, "sha256": actual_hash}


def _validate_runtime(torch: Any, torchvision: Any, ultralytics: Any, np: Any, pil: Any) -> dict[str, Any]:
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "clip": importlib.metadata.version("clip"),
        "numpy": np.__version__,
        "pillow": pil.__version__,
    }
    if actual != EXPECTED_RUNTIME:
        _fail("RUNTIME_VERSION_MISMATCH", json.dumps(actual, sort_keys=True))
    executable = _existing_file(Path(sys.executable), "PYTHON_EXECUTABLE")
    executable_hash = _sha256(executable)
    if executable_hash != PYTHON_EXE_SHA256:
        _fail("PYTHON_EXECUTABLE_HASH_MISMATCH", executable_hash)
    if os.environ.get("YOLO_AUTOINSTALL", "").lower() != "false":
        _fail("YOLO_AUTOINSTALL_MISMATCH", repr(os.environ.get("YOLO_AUTOINSTALL")))
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        _fail("CUDA_UNAVAILABLE")
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        _fail("CUDA_DEVICE_MISMATCH")
    return {**actual, "python_executable": str(executable), "python_executable_sha256": executable_hash}


def _predict(model: Any, source: Any) -> Any:
    results = model.predict(
        source=source,
        batch=INFERENCE["batch"],
        imgsz=INFERENCE["imgsz"],
        conf=INFERENCE["conf"],
        iou=INFERENCE["iou"],
        max_det=INFERENCE["max_det"],
        device=INFERENCE["device"],
        half=INFERENCE["half"],
        rect=INFERENCE["rect"],
        augment=INFERENCE["augment"],
        agnostic_nms=INFERENCE["agnostic_nms"],
        retina_masks=INFERENCE["retina_masks"],
        save=INFERENCE["save"],
        verbose=False,
        stream=False,
    )
    if not isinstance(results, list) or len(results) != 1:
        _fail("PREDICT_RESULT_CARDINALITY_MISMATCH", str(type(results)))
    return results[0]


def _extract_instances(result: Any, expected_hw: tuple[int, int], np: Any, label: str) -> list[dict[str, Any]]:
    expected_names = {index: text for index, text in enumerate(CLASS_TEXTS)}
    if getattr(result, "names", None) != expected_names:
        _fail(f"{label}_CLASS_NAMES_MISMATCH", repr(getattr(result, "names", None)))
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    count = 0 if boxes is None else len(boxes)
    if count == 0:
        if masks is not None and len(masks.data) != 0:
            _fail(f"{label}_EMPTY_BOX_MASK_MISMATCH")
        return []
    if masks is None or not hasattr(masks, "data") or len(masks.data) != count:
        _fail(f"{label}_BOX_MASK_CARDINALITY_MISMATCH", f"{count}:{None if masks is None else len(masks.data)}")
    mask_data = masks.data
    if len(mask_data.shape) != 3 or tuple(int(value) for value in mask_data.shape[-2:]) != expected_hw:
        _fail(f"{label}_MASK_SHAPE_MISMATCH", f"{tuple(mask_data.shape)}:{expected_hw}")
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    xyxy = boxes.xyxy.detach().cpu().tolist()
    instances: list[dict[str, Any]] = []
    for index, (raw_class, confidence, raw_box, raw_mask) in enumerate(
        zip(classes, confidences, xyxy, mask_data, strict=True)
    ):
        class_index = int(raw_class)
        if float(class_index) != float(raw_class) or class_index not in CLASS_BY_INDEX:
            _fail(f"{label}_CLASS_INDEX_INVALID", f"{index}:{raw_class}")
        box = [float(value) for value in raw_box]
        if len(box) != 4 or not all(math.isfinite(value) for value in box) or box[2] < box[0] or box[3] < box[1]:
            _fail(f"{label}_BOX_INVALID", f"{index}:{box}")
        confidence = float(confidence)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            _fail(f"{label}_CONFIDENCE_INVALID", f"{index}:{confidence}")
        raw_array = raw_mask.detach().cpu().numpy()
        if not bool(np.logical_or(raw_array == 0, raw_array == 1).all()):
            _fail(f"{label}_MASK_NOT_NATIVE_BINARY", str(index))
        mask = np.asarray(raw_array, dtype=np.bool_)
        prompt = CLASS_BY_INDEX[class_index]
        instances.append(
            {
                "raw_detection_index": index,
                "class_index": class_index,
                "class_id": prompt["class_id"],
                "class_text": prompt["text"],
                "confidence": confidence,
                "box_xyxy": box,
                "mask": mask,
                "mask_area_pixels": int(mask.sum()),
            }
        )
    return instances


def _mask_receipt(mask: Any, np: Any) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=np.uint8, order="C")
    return {
        "encoding": "C_ORDER_UINT8_BINARY_RECEIPT_ONLY",
        "size_hw": [int(binary.shape[0]), int(binary.shape[1])],
        "true_pixels": int(binary.sum()),
        "binary_uint8_sha256": hashlib.sha256(binary.tobytes(order="C")).hexdigest(),
    }


def _serial_instance(instance: dict[str, Any], np: Any) -> dict[str, Any]:
    return {key: value for key, value in instance.items() if key not in {"mask", "true_xs", "true_ys"}} | {
        "mask": _mask_receipt(instance["mask"], np)
    }


def _rank_parents(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parents = [dict(instance) for instance in instances if instance["class_id"] in PARENT_IDS]
    parents.sort(
        key=lambda item: (
            -item["confidence"],
            PARENT_PRIORITY[item["class_id"]],
            *item["box_xyxy"],
            item["raw_detection_index"],
        )
    )
    retained = parents[: INFERENCE["maximum_parent_crops_per_frame"]]
    for stable_index, parent in enumerate(retained):
        parent["stable_parent_index"] = stable_index
        parent["parent_instance_id"] = f"parent:{stable_index:03d}"
    return retained


def _crop_box(box: Sequence[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, min(width, math.floor(float(box[0]))))
    top = max(0, min(height, math.floor(float(box[1]))))
    right = max(0, min(width, math.ceil(float(box[2]))))
    bottom = max(0, min(height, math.ceil(float(box[3]))))
    if right <= left or bottom <= top:
        _fail("PARENT_CROP_EMPTY", f"{box}:{image_size}")
    return left, top, right, bottom


def _assign_topology(
    parents: list[dict[str, Any]], children: list[dict[str, Any]], image_size: tuple[int, int]
) -> tuple[list[dict[str, Any]], bool]:
    width, height = image_size
    assignments: list[dict[str, Any]] = []
    authorized = False
    for child in children:
        xs = child["true_xs"]
        ys = child["true_ys"]
        if len(xs):
            mean_x = float(sum(xs) / len(xs))
            mean_y = float(sum(ys) / len(ys))
            representative = [
                min(width - 1, max(0, math.floor(mean_x + 0.5))),
                min(height - 1, max(0, math.floor(mean_y + 0.5))),
            ]
            candidates = [parent for parent in parents if bool(parent["mask"][representative[1]][representative[0]])]
        else:
            mean_x = None
            mean_y = None
            representative = None
            candidates = []
        candidates.sort(
            key=lambda parent: (
                parent["mask_area_pixels"],
                PARENT_PRIORITY[parent["class_id"]],
                parent["stable_parent_index"],
            )
        )
        assigned = candidates[0] if candidates else None
        child_authorized = assigned is not None and assigned["class_id"] == "architectural_leaf"
        authorized = authorized or child_authorized
        assignments.append(
            {
                "child_instance_id": child["child_instance_id"],
                "child_class_id": child["class_id"],
                "child_lane": 2,
                "child_mask_area_pixels": child["mask_area_pixels"],
                "child_mean_integer_mask_pixel_xy": [mean_x, mean_y] if mean_x is not None else None,
                "representative_pixel_xy": representative,
                "containing_parent_instance_ids_in_assignment_order": [
                    parent["parent_instance_id"] for parent in candidates
                ],
                "assigned_parent_instance_id": assigned["parent_instance_id"] if assigned else None,
                "assigned_parent_class_id": assigned["class_id"] if assigned else None,
                "authorizes": child_authorized,
            }
        )
    return assignments, authorized


def _aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    authorized_by_role = {
        role: sum(row["role"] == role and row["parent_rescaled_pixel_part_topology"] for row in rows)
        for role in ROLE_COUNTS
    }
    positives = authorized_by_role[POSITIVE_ROLE]
    control_authorized = authorized_by_role[FURNITURE_ROLE] + authorized_by_role[OOD_ROLE]
    recall = positives / ROLE_COUNTS[POSITIVE_ROLE]
    false_positive_rate = control_authorized / (ROLE_COUNTS[FURNITURE_ROLE] + ROLE_COUNTS[OOD_ROLE])
    true_negative_rate = 1.0 - false_positive_rate
    balanced_accuracy = 0.5 * (recall + true_negative_rate)
    gate_met = (
        len(rows) == 8
        and positives == 4
        and authorized_by_role[FURNITURE_ROLE] == 0
        and authorized_by_role[OOD_ROLE] == 0
        and balanced_accuracy == 1.0
    )
    lane_1_counts = Counter()
    lane_2_counts = Counter()
    parent_crops = 0
    for row in rows:
        lane_1_counts.update(row.get("lane_1_full_frame", {}).get("counts_by_class_id", {}))
        for crop in row.get("lane_2_parent_crops", []):
            parent_crops += 1
            lane_2_counts.update(crop.get("counts_by_class_id", {}))
    metrics = {
        "frames": len(rows),
        "lane_1_model_calls": len(rows),
        "lane_2_model_calls": parent_crops,
        "total_model_calls": len(rows) + parent_crops,
        "retained_parent_crops": parent_crops,
        "lane_1_instances_by_class_id": {class_id: lane_1_counts[class_id] for class_id in CLASS_IDS},
        "lane_2_instances_by_class_id": {class_id: lane_2_counts[class_id] for class_id in CLASS_IDS},
        "authorized_frames_by_role": authorized_by_role,
        "architectural_door_positive_recall": recall,
        "control_false_positive_rate": false_positive_rate,
        "true_negative_rate": true_negative_rate,
        "balanced_accuracy": balanced_accuracy,
    }
    gate = {
        "four_of_four_architectural_doors_authorized": positives == 4,
        "zero_of_two_handled_furniture_controls_authorized": authorized_by_role[FURNITURE_ROLE] == 0,
        "zero_of_two_large_openings_authorized": authorized_by_role[OOD_ROLE] == 0,
        "balanced_accuracy_is_one": balanced_accuracy == 1.0,
        "development_gate_met": gate_met,
        "decision": (
            "L10_PB14_YOLOE_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_PB14_YOLOE_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET"
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
    text_encoder_argument: Path,
    output_argument: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = _existing_file(protocol_argument, "PROTOCOL")
    cohort_path = _existing_file(cohort_argument, "COHORT")
    extracted_root = _existing_directory(extracted_root_argument, "EXTRACTED_ROOT")
    output_path = _new_output(output_argument)
    protocol_hash = _sha256(protocol_path)
    protocol = _read_json(protocol_path, "PROTOCOL")
    _validate_protocol(protocol, protocol_hash)
    cohort_hash = _sha256(cohort_path)
    cohort = _read_json(cohort_path, "COHORT")
    frames = _validate_cohort(cohort, cohort_hash)
    expected_extracted_root = _existing_directory(
        ROOT / _relative_posix(cohort["source"]["extracted_root"], "COHORT_EXTRACTED_ROOT"),
        "EXPECTED_EXTRACTED_ROOT",
    )
    if extracted_root != expected_extracted_root:
        _fail("EXTRACTED_ROOT_PATH_MISMATCH", f"{extracted_root}:{expected_extracted_root}")
    weights_receipt = _validate_artifact(
        checkpoint_argument, ROOT / WEIGHTS_RELATIVE, WEIGHTS_BYTES, WEIGHTS_SHA256, "WEIGHTS"
    )
    encoder_receipt = _validate_artifact(
        text_encoder_argument,
        ROOT / TEXT_ENCODER_RELATIVE,
        TEXT_ENCODER_BYTES,
        TEXT_ENCODER_SHA256,
        "TEXT_ENCODER",
    )
    try:
        import numpy as np
        import PIL
        from PIL import Image
        import torch
        import torchvision
        import ultralytics
        from ultralytics import YOLOE
    except ImportError as exc:
        _fail("INFERENCE_RUNTIME_IMPORT_FAILED", str(exc))
    runtime_receipt = _validate_runtime(torch, torchvision, ultralytics, np, PIL)
    prepared: list[dict[str, Any]] = []
    for frame in frames:
        receipts: dict[str, Any] = {}
        paths: dict[str, Path] = {}
        for kind in sorted(FILE_KINDS):
            path, receipt = _manifest_file(
                extracted_root, frame["files"][kind], f"FRAME_{frame['index']}_{kind.upper()}"
            )
            paths[kind] = path
            receipts[kind] = receipt
        try:
            with Image.open(paths["rgb"]) as opened:
                opened.load()
                size = [int(opened.width), int(opened.height)]
                mode = opened.mode
        except Exception as exc:
            _fail("RGB_DECODE_FAILED", f"frame={frame['index']}:{exc}")
        if size != frame["source_image_size"] or mode not in {"RGB", "L", "RGBA"}:
            _fail("RGB_IDENTITY_MISMATCH", f"frame={frame['index']}:{size}:{mode}")
        prepared.append({"frame": frame, "paths": paths, "receipts": receipts})
    model = YOLOE(str(weights_receipt["path"]), verbose=False)
    model.to("cuda")
    if model.task != "segment" or type(model.model).__name__ != "YOLOESegModel":
        _fail("LOADED_MODEL_TYPE_MISMATCH", f"{type(model.model).__name__}:{model.task}")
    if getattr(model.model, "yaml", {}).get("text_model") != "mobileclip2:b":
        _fail("LOADED_TEXT_MODEL_IDENTITY_MISMATCH")
    previous = Path.cwd()
    try:
        os.chdir(Path(encoder_receipt["path"]).parent)
        model.set_classes(CLASS_TEXTS)
    finally:
        os.chdir(previous)
    if model.names != {index: text for index, text in enumerate(CLASS_TEXTS)}:
        _fail("LOADED_PROMPT_ORDER_MISMATCH", repr(model.names))
    if _sha256(Path(encoder_receipt["path"])) != TEXT_ENCODER_SHA256:
        _fail("TEXT_ENCODER_CHANGED_DURING_LOAD")
    first_parameter = next(model.model.parameters())
    if first_parameter.device.type != "cuda" or first_parameter.device.index != 0:
        _fail("LOADED_MODEL_DEVICE_MISMATCH", str(first_parameter.device))
    torch.cuda.reset_peak_memory_stats(0)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for item in prepared:
            frame = item["frame"]
            width, height = frame["source_image_size"]
            with Image.open(item["paths"]["rgb"]) as opened:
                image = opened.convert("RGB")
            lane_1_started = time.perf_counter()
            lane_1_result = _predict(model, str(item["paths"]["rgb"]))
            torch.cuda.synchronize(0)
            lane_1_seconds = time.perf_counter() - lane_1_started
            lane_1_instances = _extract_instances(lane_1_result, (height, width), np, f"FRAME_{frame['index']}_LANE1")
            lane_1_counts = Counter(instance["class_id"] for instance in lane_1_instances)
            parents = _rank_parents(lane_1_instances)
            lane_2_crops: list[dict[str, Any]] = []
            lane_2_children: list[dict[str, Any]] = []
            for parent in parents:
                left, top, right, bottom = _crop_box(parent["box_xyxy"], (width, height))
                crop = image.crop((left, top, right, bottom))
                rgb = np.asarray(crop, dtype=np.uint8)
                if rgb.shape != (bottom - top, right - left, 3):
                    _fail("PARENT_CROP_RGB_SHAPE_MISMATCH", f"{parent['parent_instance_id']}:{rgb.shape}")
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                if not bgr.flags.c_contiguous:
                    _fail("PARENT_CROP_BGR_NOT_CONTIGUOUS", parent["parent_instance_id"])
                crop_started = time.perf_counter()
                crop_result = _predict(model, bgr)
                torch.cuda.synchronize(0)
                crop_seconds = time.perf_counter() - crop_started
                crop_instances = _extract_instances(
                    crop_result,
                    (bottom - top, right - left),
                    np,
                    f"FRAME_{frame['index']}_LANE2_{parent['stable_parent_index']}",
                )
                crop_counts = Counter(instance["class_id"] for instance in crop_instances)
                serial_children: list[dict[str, Any]] = []
                for child in crop_instances:
                    if child["class_id"] not in CHILD_IDS:
                        continue
                    mapped = np.zeros((height, width), dtype=np.bool_)
                    mapped[top:bottom, left:right] = child["mask"]
                    ys, xs = np.nonzero(mapped)
                    child_id = (
                        f"{parent['parent_instance_id']}:child:{child['raw_detection_index']:03d}"
                    )
                    mapped_child = {
                        **{key: value for key, value in child.items() if key != "mask"},
                        "child_instance_id": child_id,
                        "source_parent_crop_instance_id": parent["parent_instance_id"],
                        "crop_box_xyxy_half_open": [left, top, right, bottom],
                        "crop_box_xyxy": child["box_xyxy"],
                        "source_box_xyxy": [
                            child["box_xyxy"][0] + left,
                            child["box_xyxy"][1] + top,
                            child["box_xyxy"][2] + left,
                            child["box_xyxy"][3] + top,
                        ],
                        "mask": mapped,
                        "mask_area_pixels": int(mapped.sum()),
                        "true_xs": xs,
                        "true_ys": ys,
                    }
                    lane_2_children.append(mapped_child)
                    serial_children.append(_serial_instance(mapped_child, np))
                lane_2_crops.append(
                    {
                        "source_parent_instance_id": parent["parent_instance_id"],
                        "source_parent_class_id": parent["class_id"],
                        "crop_box_xyxy_half_open": [left, top, right, bottom],
                        "crop_size_wh": [right - left, bottom - top],
                        "rgb_to_contiguous_bgr": True,
                        "inference_seconds": crop_seconds,
                        "raw_instance_count": len(crop_instances),
                        "counts_by_class_id": {class_id: crop_counts[class_id] for class_id in CLASS_IDS},
                        "retained_child_instances": serial_children,
                    }
                )
            assignments, authorized = _assign_topology(parents, lane_2_children, (width, height))
            retained_ids = {parent["parent_instance_id"] for parent in parents}
            lane_1_serial = []
            for instance in lane_1_instances:
                record = _serial_instance(instance, np)
                matching = [parent for parent in parents if parent["raw_detection_index"] == instance["raw_detection_index"]]
                record["retained_parent_instance_id"] = matching[0]["parent_instance_id"] if matching else None
                record["retained_for_lane_2"] = bool(matching and matching[0]["parent_instance_id"] in retained_ids)
                lane_1_serial.append(record)
            rows.append(
                {
                    "index": frame["index"],
                    "frame_id": frame["frame_id"],
                    "capture_sequence_id": frame["capture_sequence_id"],
                    "sensor_source_bucket": frame["sensor_source_bucket"],
                    "role": frame["role"],
                    "canonical_source_path": frame["canonical_source_path"],
                    "audit_note": frame["audit_note"],
                    "official_evidence": frame["official_evidence"],
                    "source_image_size": frame["source_image_size"],
                    "files": item["receipts"],
                    "lane_1_full_frame": {
                        "inference_seconds": lane_1_seconds,
                        "raw_instance_count": len(lane_1_instances),
                        "counts_by_class_id": {class_id: lane_1_counts[class_id] for class_id in CLASS_IDS},
                        "instances": lane_1_serial,
                        "retained_parent_instance_ids_in_crop_order": [
                            parent["parent_instance_id"] for parent in parents
                        ],
                        "lane_1_children_authorize": False,
                        "architectural_frame_is_diagnostic_only": True,
                    },
                    "lane_2_parent_crops": lane_2_crops,
                    "child_parent_assignments": assignments,
                    "parent_rescaled_pixel_part_topology": authorized,
                }
            )
            del image
    torch.cuda.synchronize(0)
    metrics, gate = _aggregate(rows)
    properties = torch.cuda.get_device_properties(0)
    evaluator_path = Path(__file__).resolve()
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "L10-PB14 YOLOE Parent-Rescaled Pixel Part Topology",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "inputs": {
            "protocol": {"path": str(protocol_path), "schema": PROTOCOL_SCHEMA, "sha256": protocol_hash},
            "cohort": {"path": str(cohort_path), "schema": COHORT_SCHEMA, "sha256": cohort_hash, "status": cohort["status"]},
            "extracted_root": str(extracted_root),
            "weights": weights_receipt,
            "text_encoder": encoder_receipt,
        },
        "evaluator": {"path": str(evaluator_path), "sha256": _sha256(evaluator_path)},
        "runtime": {
            **runtime_receipt,
            "YOLO_AUTOINSTALL": os.environ["YOLO_AUTOINSTALL"],
            "cuda_runtime": torch.version.cuda,
            "actual_device_index": 0,
            "actual_device_name": properties.name,
            "actual_device_capability": list(torch.cuda.get_device_capability(0)),
            "device_total_memory_bytes": properties.total_memory,
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(0),
            "loaded_model_type": type(model.model).__name__,
            "loaded_model_task": model.task,
            "loaded_text_model": model.model.yaml.get("text_model"),
            "loaded_parameter_device": str(first_parameter.device),
            "loaded_parameter_dtype": str(first_parameter.dtype),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "inference": {"class_prompts_in_order": PROMPTS, **INFERENCE},
        "topology": {
            "parent_class_ids": ["architectural_leaf", "cabinet_door", "closet_door", "refrigerator_door", "doorless_opening"],
            "child_class_ids": ["operation_part", "hinge"],
            "representative": "ROUND_HALF_UP_MEAN_INTEGER_TRUE_MAPPED_CHILD_MASK_PIXEL_INDICES",
            "assignment_order": "MIN_PARENT_MASK_AREA_THEN_COMPETITOR_CONSERVATIVE_CLASS_PRIORITY_THEN_STABLE_PARENT_INDEX",
            "authorization": "AT_LEAST_ONE_LANE2_CHILD_ASSIGNED_TO_ARCHITECTURAL_LEAF",
            "lane_1_children_authorize": False,
            "architectural_frame_is_diagnostic_only": True,
        },
        "frames": rows,
        "metrics": metrics,
        "gate": gate,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json_new(output_path, result)
    return result


def _synthetic_mask(width: int, height: int, points: set[tuple[int, int]]) -> list[list[bool]]:
    return [[(x, y) in points for x in range(width)] for y in range(height)]


def _self_test() -> dict[str, Any]:
    size = (10, 10)
    all_points = {(x, y) for y in range(10) for x in range(10)}
    cabinet_points = {(x, y) for y in range(3, 7) for x in range(3, 7)}
    parents = [
        {
            "parent_instance_id": "parent:000",
            "class_id": "architectural_leaf",
            "stable_parent_index": 0,
            "mask_area_pixels": len(all_points),
            "mask": _synthetic_mask(*size, all_points),
        },
        {
            "parent_instance_id": "parent:001",
            "class_id": "cabinet_door",
            "stable_parent_index": 1,
            "mask_area_pixels": len(cabinet_points),
            "mask": _synthetic_mask(*size, cabinet_points),
        },
    ]
    child = {
        "child_instance_id": "parent:000:child:000",
        "class_id": "operation_part",
        "mask_area_pixels": 4,
        "true_xs": [4, 5, 4, 5],
        "true_ys": [4, 4, 5, 5],
    }
    assignments, authorized = _assign_topology(parents, [child], size)
    if assignments[0]["assigned_parent_class_id"] != "cabinet_door" or authorized:
        raise AssertionError("smallest competitor parent did not absorb the child")
    competitor_assignment = assignments[0]["assigned_parent_class_id"]
    assignments, authorized = _assign_topology(parents[:1], [child], size)
    if assignments[0]["assigned_parent_class_id"] != "architectural_leaf" or not authorized:
        raise AssertionError("lane-2 architectural parent authorization failed")
    rows = []
    for index in range(8):
        role = POSITIVE_ROLE if index < 4 else FURNITURE_ROLE if index < 6 else OOD_ROLE
        rows.append(
            {
                "role": role,
                "parent_rescaled_pixel_part_topology": index < 4,
                "lane_1_full_frame": {"counts_by_class_id": {}},
                "lane_2_parent_crops": [],
            }
        )
    metrics, gate = _aggregate(rows)
    if metrics["balanced_accuracy"] != 1.0 or not gate["development_gate_met"]:
        raise AssertionError("perfect frozen gate did not pass")
    rows[0]["parent_rescaled_pixel_part_topology"] = False
    _, failed = _aggregate(rows)
    if failed["development_gate_met"]:
        raise AssertionError("imperfect frozen gate passed")
    return {
        "status": "PASS",
        "model_calls": 0,
        "cohort_images_read": 0,
        "competitor_assignment": competitor_assignment,
        "architectural_assignment": assignments[0]["assigned_parent_class_id"],
        "perfect_gate_decision": gate["decision"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen PB14 YOLOE multiscale pixel topology evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run deterministic synthetic topology and gate checks only.")
    run = subparsers.add_parser("run", help="Run the one frozen eight-image CUDA evaluation.")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--cohort", "--queue", dest="cohort", type=Path, required=True)
    run.add_argument("--extracted-root", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, default=ROOT / WEIGHTS_RELATIVE)
    run.add_argument("--text-encoder", type=Path, default=ROOT / TEXT_ENCODER_RELATIVE)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "self-test":
            print(json.dumps(_self_test(), ensure_ascii=False, sort_keys=True))
            return 0
        result = _run(
            args.protocol,
            args.cohort,
            args.extracted_root,
            args.checkpoint,
            args.text_encoder,
            args.output,
        )
    except ContractError as exc:
        print(f"PB14_CONTRACT_ERROR:{exc}", file=sys.stderr)
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
