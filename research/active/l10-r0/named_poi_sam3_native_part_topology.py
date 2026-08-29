#!/usr/bin/env python3
"""Run the frozen L10-PB16 SAM3.1 native full-frame part-topology gate."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "l10-named-poi-sam3-native-part-topology-protocol-v1"
COHORT_SCHEMA = "l10-named-poi-sam3-native-part-topology-cohort-v1"
RESULT_SCHEMA = "l10-named-poi-sam3-native-part-topology-development-result-v1"
PROTOCOL_SHA256 = "3c3f3984e6acb895a4033990bcab4cc23820b413f9127de894695b34dc51bcc4"
COHORT_SHA256 = "6b383da2f8c9f7d0ad064f53a375bd9d0c102504e74eaaff0512750a76200811"

MODEL_REPOSITORY = "facebook/sam3.1"
MODEL_REPOSITORY_COMMIT = "616acbee0b9ed4177f1f389e3c13594a0a1f6398"
SOURCE_REPOSITORY = "facebookresearch/sam3"
SOURCE_ORIGIN = "https://github.com/facebookresearch/sam3.git"
SOURCE_COMMIT = "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"
CHECKPOINT_RELATIVE = Path("artifacts.local/models/sam3.1-modelscope/sam3.1_multiplex.pt")
SOURCE_RELATIVE = Path("artifacts.local/models/sam3-source-v1")
CHECKPOINT_BYTES = 3_502_755_717
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"

MODEL_FILES = [
    ("config.json", 26_739, "9593725533302a100d8e4ff1e29c0799e5a04c696b76c65fa4ee28dcd036b16b"),
    ("processor_config.json", 1_792, "d76e8fbdc561ba6af3b44937b7e99a433440cb0251ff11237b0b7d0654dd8474"),
]
SOURCE_FILES = [
    ("sam3/model_builder.py", 44_074, "c78af29c3db0f0ef3c9ec66dcfe32d6a9fd70b82f660142e97908a5ce61c469f"),
    (
        "sam3/model/sam3_image_processor.py",
        9_217,
        "e5167f53bc756e121df6c53e3cde2367a0e0f7d2b2de9d2093ef27f4691b805d",
    ),
    (
        "sam3/model/sam3_multiplex_detector.py",
        42_639,
        "dc951b516538ec8bcf9e03f1005b9603c8194f454554af5274618de4364f65ac",
    ),
    (
        "sam3/model/vl_combiner.py",
        15_494,
        "c04e3f911db8c950ceb6a31f77ce49d47cac15d99493bdced9ed57d2a3aa7443",
    ),
    (
        "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        1_356_917,
        "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    ),
    ("LICENSE", 7_414, "67f67e15efa517e0f67f59ff2d2f603a624da5d2f5458b0c5b366ccd66bb3eea"),
]

LAUNCHER_FROZEN = "E:/codex-tools/bin/blindassist-sam3-research-gpu.cmd"
LAUNCHER_SHA256 = "4ca5d26dc0ee957b6574dbe4a518792200fa7f2a1a397806ddeed0a7452e74cb"
PYTHON_EXE_FROZEN = "E:/codex-tools/tools/venvs/blindassist-sam3-gpu/Scripts/python.exe"
PYTHON_EXE_SHA256 = "21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082"
EXPECTED_DISTRIBUTIONS = {
    "numpy": "1.26.4",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "triton-windows": "3.6.0.post26",
    "sam3": "0.1.0",
    "timm": "1.0.28",
    "ftfy": "6.1.1",
    "iopath": "0.1.10",
    "pycocotools": "2.0.11",
    "Pillow": "12.2.0",
}

STRICT_ADAPTER_NAME = "PB16_STRICT_SAM31_IMAGE_ONLY_MULTIPLEX_DETECTOR_ADAPTER"
STRICT_ADAPTER_ARGUMENTS = {
    "recipe_source": "sam3.model_builder.build_sam3_multiplex_video_predictor detector construction block",
    "detector_class": "sam3.model.sam3_multiplex_detector.Sam3MultiplexDetector",
    "backbone_class": "sam3.model.vl_combiner.SAM3VLBackboneTri",
    "tri_neck_factory": "sam3.model_builder._create_multiplex_tri_backbone",
    "tri_neck_scale_factors": [4.0, 2.0, 1.0],
    "bpe_path": (SOURCE_RELATIVE / "sam3/assets/bpe_simple_vocab_16e6.txt.gz").as_posix(),
    "checkpoint_path": CHECKPOINT_RELATIVE.as_posix(),
    "checkpoint_namespace": "detector.",
    "strict_state_dict_loading": True,
    "use_fa3": False,
    "use_rope_real": False,
    "device": "cuda",
    "eval_mode": True,
    "compile": False,
    "tracker_instantiated": False,
    "video_predictor_instantiated": False,
}

PROMPTS = [
    {"global_call_index": 0, "family": "PARENT", "class_id": "architectural_leaf", "text": "architectural door"},
    {"global_call_index": 1, "family": "PARENT", "class_id": "cabinet_door", "text": "cabinet door"},
    {"global_call_index": 2, "family": "PARENT", "class_id": "closet_door", "text": "closet door"},
    {"global_call_index": 3, "family": "PARENT", "class_id": "refrigerator_door", "text": "refrigerator door"},
    {"global_call_index": 4, "family": "PARENT", "class_id": "doorless_opening", "text": "doorway"},
    {"global_call_index": 5, "family": "CHILD", "class_id": "handle", "text": "door handle"},
    {"global_call_index": 6, "family": "CHILD", "class_id": "knob", "text": "door knob"},
    {"global_call_index": 7, "family": "CHILD", "class_id": "push_bar", "text": "door push bar"},
    {"global_call_index": 8, "family": "CHILD", "class_id": "panic_bar", "text": "door panic bar"},
    {"global_call_index": 9, "family": "CHILD", "class_id": "hinge", "text": "door hinge"},
]
PARENT_CLASSES = [item["class_id"] for item in PROMPTS if item["family"] == "PARENT"]
CHILD_CLASSES = [item["class_id"] for item in PROMPTS if item["family"] == "CHILD"]
PARENT_PRIORITY_ORDER = [
    "cabinet_door",
    "closet_door",
    "refrigerator_door",
    "doorless_opening",
    "architectural_leaf",
]
PARENT_PRIORITY = {class_id: index for index, class_id in enumerate(PARENT_PRIORITY_ORDER)}
MAXIMUM_INSTANCES_PER_PROMPT = 12
PROCESSOR_RESOLUTION = 1008
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

POSITIVE_ROLE = "ARCHITECTURAL_DOOR_WITH_VISIBLE_OPERATION_PART"
FURNITURE_ROLE = "HANDLED_FURNITURE_DOOR_NEGATIVE"
OOD_ROLE = "LARGE_DOORLESS_OPENING_OOD"
ROLE_COUNTS = {POSITIVE_ROLE: 4, FURNITURE_ROLE: 2, OOD_ROLE: 2}
FILE_KINDS = {"rgb", "depth", "intrinsics", "polygon", "scene_metadata"}
EXPECTED_BUCKETS = {
    "xtion/sun3ddata": 3,
    "kv2/kinect2data": 3,
    "realsense/lg": 1,
    "realsense/shr": 1,
}


class ContractError(RuntimeError):
    """Raised when a frozen formal-run contract cannot be proven."""


def _fail(code: str, detail: str | None = None) -> None:
    raise ContractError(code if detail is None else f"{code}:{detail}")


def _exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label}_MISMATCH", f"actual={actual!r}:expected={expected!r}")


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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"{label}_JSON_INVALID", str(exc))
    if not isinstance(value, dict):
        _fail(f"{label}_NOT_OBJECT")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label}_INVALID")
    return value


def _hex_digest(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        _fail(f"{label}_INVALID", text)
    return text


def _bound_hash(value: str, label: str) -> str:
    if value.startswith("PENDING_"):
        _fail(f"EVALUATOR_{label}_HASH_BINDING_PENDING")
    return _hex_digest(value, f"EVALUATOR_{label}_SHA256")


def _relative_posix(value: Any, label: str) -> Path:
    text = _require_string(value, label)
    if "\\" in text or "\x00" in text or text.startswith("/") or "//" in text:
        _fail(f"{label}_UNSAFE", text)
    posix = PurePosixPath(text)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        _fail(f"{label}_UNSAFE", text)
    if ":" in posix.parts[0] or posix.as_posix() != text:
        _fail(f"{label}_UNSAFE", text)
    return Path(*posix.parts)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = _existing_file(path, label)
    size = actual.stat().st_size
    _exact(size, expected_bytes, f"{label}_BYTES")
    digest = _sha256(actual)
    _exact(digest, expected_sha256, f"{label}_SHA256")
    return {"path": str(actual), "bytes": size, "sha256": digest}


def _validate_hash_only(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = _existing_file(path, label)
    digest = _sha256(actual)
    _exact(digest, expected_sha256, f"{label}_SHA256")
    return {"path": str(actual), "bytes": actual.stat().st_size, "sha256": digest}


def _validate_protocol(protocol: dict[str, Any], actual_hash: str) -> None:
    _exact(actual_hash, _bound_hash(PROTOCOL_SHA256, "PROTOCOL"), "PROTOCOL_SHA256")
    _exact(protocol.get("schema"), PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    _exact(protocol.get("status"), "FROZEN_BEFORE_FRESH_COHORT_MODEL_OUTPUT", "PROTOCOL_STATUS")
    _exact(
        set(protocol),
        {
            "schema", "created_at", "status", "experiment", "short_name", "question",
            "single_change_from_pb15", "information_source", "execution_readiness_at_freeze",
            "runtime", "model_load", "cohort_freeze", "image_input", "native_text_inference",
            "native_mask_topology", "metrics", "gate", "result_contract", "forbidden_rescue",
            "claim_boundary",
        },
        "PROTOCOL_TOP_LEVEL_FIELDS",
    )
    source = protocol.get("information_source")
    if not isinstance(source, dict):
        _fail("PROTOCOL_INFORMATION_SOURCE_MISSING")
    for key, expected in {
        "model_repository": MODEL_REPOSITORY,
        "model_repository_commit": MODEL_REPOSITORY_COMMIT,
        "model_snapshot_path": CHECKPOINT_RELATIVE.parent.as_posix(),
        "source_repository": SOURCE_REPOSITORY,
        "source_repository_origin": SOURCE_ORIGIN,
        "source_commit": SOURCE_COMMIT,
        "source_path": SOURCE_RELATIVE.as_posix(),
        "source_package_version": "0.1.0",
    }.items():
        _exact(source.get(key), expected, f"PROTOCOL_SOURCE_{key.upper()}")
    _exact(
        source.get("checkpoint"),
        {
            "path": CHECKPOINT_RELATIVE.as_posix(),
            "bytes": CHECKPOINT_BYTES,
            "sha256": CHECKPOINT_SHA256,
            "load_role": "The PB16 strict image-only adapter assembles the exact three-scale Sam3MultiplexDetector component recipe from the official SAM3.1 video builder, loads only detector.* with strict=True, and never instantiates the tracker or video predictor.",
        },
        "PROTOCOL_CHECKPOINT",
    )
    expected_model_files = [
        {"path": path, "bytes": size, "sha256": digest} for path, size, digest in MODEL_FILES
    ]
    expected_source_files = [
        {"path": path, "bytes": size, "sha256": digest} for path, size, digest in SOURCE_FILES
    ]
    _exact(source.get("model_files"), expected_model_files, "PROTOCOL_MODEL_FILES")
    _exact(source.get("source_files"), expected_source_files, "PROTOCOL_SOURCE_FILES")
    readiness = protocol.get("execution_readiness_at_freeze")
    if not isinstance(readiness, dict):
        _fail("PROTOCOL_EXECUTION_READINESS_MISSING")
    _exact(readiness.get("ready"), True, "PROTOCOL_EXECUTION_READY")
    _exact(readiness.get("blockers"), [], "PROTOCOL_EXECUTION_BLOCKERS")
    smoke_receipt = {
        "path": "artifacts.local/models/sam3.1-modelscope/pb16_sam31_strict_image_adapter_smoke_receipt_v1.json",
        "bytes": 3_396,
        "sha256": "ed21163ba2e600b6ac37185980a678d45e1d5d05064d4c4bbb1cef0da4bdefcf",
        "status": "PASS",
    }
    _exact(readiness.get("strict_adapter_smoke_receipt"), smoke_receipt, "PROTOCOL_STRICT_ADAPTER_SMOKE_RECEIPT")
    _exact(
        readiness.get("no_pb16_cohort_model_output_was_observed_before_freeze"),
        True,
        "PROTOCOL_NO_COHORT_OUTPUT_BEFORE_FREEZE",
    )
    _exact(
        readiness.get("no_pb16_cohort_rgb_was_read_to_freeze_this_protocol"),
        True,
        "PROTOCOL_NO_COHORT_RGB_BEFORE_FREEZE",
    )
    actual_smoke = _validate_hash_only(ROOT / _relative_posix(smoke_receipt["path"], "SMOKE_RECEIPT"), smoke_receipt["sha256"], "SMOKE_RECEIPT")
    _exact(actual_smoke["bytes"], smoke_receipt["bytes"], "SMOKE_RECEIPT_BYTES")

    runtime = protocol.get("runtime")
    if not isinstance(runtime, dict):
        _fail("PROTOCOL_RUNTIME_MISSING")
    for key, expected in {
        "launcher": LAUNCHER_FROZEN,
        "launcher_sha256": LAUNCHER_SHA256,
        "python_executable": PYTHON_EXE_FROZEN,
        "python_executable_sha256": PYTHON_EXE_SHA256,
        "python": "3.11.9",
        "numpy": "1.26.4",
        "torch": "2.11.0+cu130",
        "torchvision": "0.26.0+cu130",
        "triton": "3.6.0",
        "triton_distribution": "triton-windows 3.6.0.post26",
        "sam3": "0.1.0",
        "timm": "1.0.28",
        "ftfy": "6.1.1",
        "iopath": "0.1.10",
        "pycocotools": "2.0.11",
        "pillow": "12.2.0",
        "cuda_runtime": "13.0",
        "device": "cuda:0",
        "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "gpu_memory_mib_min": 8_000,
        "model_dtype": "float32 parameters with CUDA BF16 autocast for all model inference",
        "autocast": "torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)",
        "inference_mode": True,
        "compile": False,
        "offline": True,
    }.items():
        _exact(runtime.get(key), expected, f"PROTOCOL_RUNTIME_{key.upper()}")
    _exact(runtime.get("environment"), {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, "PROTOCOL_OFFLINE_ENV")

    model_load = protocol.get("model_load")
    if not isinstance(model_load, dict):
        _fail("PROTOCOL_MODEL_LOAD_MISSING")
    _exact(model_load.get("builder"), STRICT_ADAPTER_NAME, "PROTOCOL_BUILDER")
    _exact(model_load.get("builder_arguments"), STRICT_ADAPTER_ARGUMENTS, "PROTOCOL_BUILDER_ARGUMENTS")
    _exact(model_load.get("processor"), "sam3.model.sam3_image_processor.Sam3Processor", "PROTOCOL_PROCESSOR")
    _exact(
        model_load.get("processor_arguments"),
        {"resolution": PROCESSOR_RESOLUTION, "device": "cuda", "confidence_threshold": CONFIDENCE_THRESHOLD},
        "PROTOCOL_PROCESSOR_ARGUMENTS",
    )

    image_input = protocol.get("image_input")
    if not isinstance(image_input, dict):
        _fail("PROTOCOL_IMAGE_INPUT_MISSING")
    for key, expected in {"batch_size": 1, "depth_use": "NONE", "parent_crop": "FORBIDDEN", "child_crop": "FORBIDDEN"}.items():
        _exact(image_input.get(key), expected, f"PROTOCOL_IMAGE_{key.upper()}")
    inference = protocol.get("native_text_inference")
    if not isinstance(inference, dict):
        _fail("PROTOCOL_NATIVE_INFERENCE_MISSING")
    _exact(inference.get("prompt_calls_in_order"), PROMPTS, "PROTOCOL_PROMPTS")
    _exact(inference.get("maximum_instances_per_prompt"), MAXIMUM_INSTANCES_PER_PROMPT, "PROTOCOL_INSTANCE_CAP")
    _exact(
        inference.get("calls_per_frame"),
        {"set_image": 1, "set_text_prompt": 10, "parent_prompts": 5, "child_prompts": 5},
        "PROTOCOL_CALLS_PER_FRAME",
    )
    topology = protocol.get("native_mask_topology")
    if not isinstance(topology, dict):
        _fail("PROTOCOL_TOPOLOGY_MISSING")
    _exact(topology.get("parent_class_priority_for_equal_area"), PARENT_PRIORITY_ORDER, "PROTOCOL_PARENT_PRIORITY")
    _exact(topology.get("learned_or_selected_postprocessing_thresholds"), 0, "PROTOCOL_EXTRA_THRESHOLDS")
    cohort_freeze = protocol.get("cohort_freeze")
    if not isinstance(cohort_freeze, dict):
        _fail("PROTOCOL_COHORT_FREEZE_MISSING")
    _exact(cohort_freeze.get("frames"), 8, "PROTOCOL_FRAME_COUNT")
    _exact(cohort_freeze.get("distinct_capture_sequences"), 8, "PROTOCOL_SEQUENCE_COUNT")
    _exact(cohort_freeze.get("roles"), ROLE_COUNTS, "PROTOCOL_ROLE_COUNTS")


def _validate_capture_binding(frame: dict[str, Any]) -> None:
    index = frame["index"]
    frame_id = _require_string(frame.get("frame_id"), f"FRAME_{index}_ID")
    sequence = PurePosixPath(_require_string(frame.get("capture_sequence_id"), f"FRAME_{index}_SEQUENCE"))
    canonical = PurePosixPath(_require_string(frame.get("canonical_source_path"), f"FRAME_{index}_CANONICAL"))
    bucket = _require_string(frame.get("sensor_source_bucket"), f"FRAME_{index}_BUCKET")
    _relative_posix(sequence.as_posix(), f"FRAME_{index}_SEQUENCE")
    _relative_posix(canonical.as_posix(), f"FRAME_{index}_CANONICAL")
    prefix = f"SUNRGBD/{bucket}/"
    if not sequence.as_posix().startswith(prefix) or not canonical.as_posix().startswith(prefix):
        _fail("COHORT_BUCKET_PATH_MISMATCH", f"frame={index}")
    if bucket == "kv2/kinect2data":
        if "_rgbf" in sequence.name or canonical.as_posix() != f"{sequence.as_posix()}_{frame_id}" or not frame_id.startswith("rgbf"):
            _fail("COHORT_KV2_SEQUENCE_BINDING_INVALID", f"frame={index}")
    elif bucket == "xtion/sun3ddata":
        if canonical.parent != sequence or canonical.name != frame_id:
            _fail("COHORT_XTION_SEQUENCE_BINDING_INVALID", f"frame={index}")
    elif bucket.startswith("realsense/"):
        if canonical != sequence or canonical.name != frame_id:
            _fail("COHORT_REALSENSE_SEQUENCE_BINDING_INVALID", f"frame={index}")
    else:
        _fail("COHORT_SENSOR_BUCKET_UNSUPPORTED", bucket)


def _validate_cohort(cohort: dict[str, Any], actual_hash: str) -> list[dict[str, Any]]:
    _exact(actual_hash, _bound_hash(COHORT_SHA256, "COHORT"), "COHORT_SHA256")
    _exact(cohort.get("schema"), COHORT_SCHEMA, "COHORT_SCHEMA")
    _exact(cohort.get("status"), "FROZEN_BEFORE_MODEL_OUTPUT", "COHORT_STATUS")
    _exact(cohort.get("protocol"), {"schema": PROTOCOL_SCHEMA, "sha256": PROTOCOL_SHA256}, "COHORT_PROTOCOL")
    _exact(set(cohort), {"schema", "status", "frozen_at", "protocol", "source", "models", "diversity", "frames"}, "COHORT_TOP_LEVEL_FIELDS")
    source = cohort.get("source")
    if not isinstance(source, dict):
        _fail("COHORT_SOURCE_MISSING")
    for key, expected in {
        "dataset": "SUN RGB-D",
        "archive_bytes": 6_885_481_608,
        "archive_sha256": "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c",
        "extracted_root": "artifacts.local/datasets",
        "official_members": 40,
        "official_member_bytes": 1_958_756,
        "prior_pb11_sequence_overlap": 0,
        "prior_pb12_sequence_overlap": 0,
        "prior_pb13_sequence_overlap": 0,
        "prior_pb14_sequence_overlap": 0,
        "prior_pb15_sequence_overlap": 0,
    }.items():
        _exact(source.get(key), expected, f"COHORT_SOURCE_{key.upper()}")
    models = cohort.get("models")
    if not isinstance(models, dict):
        _fail("COHORT_MODELS_MISSING")
    _exact(
        models.get("sam3_native"),
        {
            "repository": MODEL_REPOSITORY,
            "model_repository_commit": MODEL_REPOSITORY_COMMIT,
            "source_repository_commit": SOURCE_COMMIT,
            "checkpoint": {"filename": CHECKPOINT_RELATIVE.name, "bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256},
        },
        "COHORT_MODEL",
    )
    diversity = cohort.get("diversity")
    if not isinstance(diversity, dict):
        _fail("COHORT_DIVERSITY_MISSING")
    _exact(diversity.get("distinct_capture_sequences"), 8, "COHORT_DISTINCT_SEQUENCES")
    _exact(diversity.get("sensor_source_buckets"), EXPECTED_BUCKETS, "COHORT_BUCKET_DISTRIBUTION")
    _exact(diversity.get("maximum_frames_in_one_bucket"), 3, "COHORT_BUCKET_MAXIMUM")

    frames = cohort.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        _fail("COHORT_FRAME_COUNT_MISMATCH")
    if [frame.get("index") for frame in frames if isinstance(frame, dict)] != list(range(1, 9)):
        _fail("COHORT_FRAME_INDEX_MISMATCH")
    if Counter(frame.get("role") for frame in frames) != Counter(ROLE_COUNTS):
        _fail("COHORT_ROLE_COUNTS_MISMATCH")
    _exact(dict(Counter(frame.get("sensor_source_bucket") for frame in frames)), EXPECTED_BUCKETS, "COHORT_FRAME_BUCKETS")
    sequences = [frame.get("capture_sequence_id") for frame in frames]
    if len(set(sequences)) != 8:
        _fail("COHORT_CAPTURE_SEQUENCES_NOT_DISTINCT")

    seen_official: set[str] = set()
    seen_local: set[str] = set()
    total_bytes = 0
    for frame in frames:
        if not isinstance(frame, dict):
            _fail("COHORT_FRAME_NOT_OBJECT")
        _exact(
            set(frame),
            {"index", "frame_id", "capture_sequence_id", "sensor_source_bucket", "role", "canonical_source_path", "audit_note", "source_image_size", "official_evidence", "files"},
            f"FRAME_{frame['index']}_FIELDS",
        )
        _validate_capture_binding(frame)
        if not isinstance(frame.get("audit_note"), str) or not frame["audit_note"]:
            _fail("COHORT_AUDIT_NOTE_INVALID", f"frame={frame['index']}")
        size = frame.get("source_image_size")
        if not isinstance(size, list) or len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
            _fail("COHORT_IMAGE_SIZE_INVALID", f"frame={frame['index']}")
        evidence = frame.get("official_evidence")
        if not isinstance(evidence, dict) or set(evidence) != {"annotation_prefilter", "supporting_object_names"}:
            _fail("COHORT_OFFICIAL_EVIDENCE_INVALID", f"frame={frame['index']}")
        names = evidence.get("supporting_object_names")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
            _fail("COHORT_OFFICIAL_NAMES_INVALID", f"frame={frame['index']}")
        expected_prefilter = "EXACT_DOOR" if frame["role"] == POSITIVE_ROLE else "NO_EXACT_DOOR"
        _exact(evidence.get("annotation_prefilter"), expected_prefilter, f"FRAME_{frame['index']}_PREFILTER")
        if frame["role"] == POSITIVE_ROLE and "door" not in {name.strip().lower() for name in names}:
            _fail("COHORT_POSITIVE_EXACT_DOOR_LABEL_MISSING", f"frame={frame['index']}")
        files = frame.get("files")
        if not isinstance(files, dict) or set(files) != FILE_KINDS:
            _fail("COHORT_FRAME_FILES_INVALID", f"frame={frame['index']}")
        canonical = frame["canonical_source_path"]
        for kind in FILE_KINDS:
            receipt = files[kind]
            if not isinstance(receipt, dict) or set(receipt) != {"path", "local_path", "bytes", "sha256"}:
                _fail("COHORT_FILE_RECEIPT_INVALID", f"frame={frame['index']}:{kind}")
            official = _relative_posix(receipt.get("path"), f"FRAME_{frame['index']}_{kind.upper()}_OFFICIAL_PATH")
            local = _relative_posix(receipt.get("local_path"), f"FRAME_{frame['index']}_{kind.upper()}_LOCAL_PATH")
            if not official.as_posix().startswith(f"{canonical}/"):
                _fail("COHORT_OFFICIAL_PATH_CAPTURE_MISMATCH", f"frame={frame['index']}:{kind}")
            official_parts = official.parts
            if len(local.parts) < len(official_parts) or local.parts[-len(official_parts):] != official_parts:
                _fail("COHORT_LOCAL_OFFICIAL_SUFFIX_MISMATCH", f"frame={frame['index']}:{kind}")
            if official.as_posix() in seen_official or local.as_posix() in seen_local:
                _fail("COHORT_FILE_PATH_DUPLICATE", f"frame={frame['index']}:{kind}")
            seen_official.add(official.as_posix())
            seen_local.add(local.as_posix())
            if type(receipt.get("bytes")) is not int or receipt["bytes"] <= 0:
                _fail("COHORT_FILE_BYTES_INVALID", f"frame={frame['index']}:{kind}")
            _hex_digest(receipt.get("sha256"), f"FRAME_{frame['index']}_{kind.upper()}_SHA256")
            total_bytes += receipt["bytes"]
    _exact(len(seen_official), 40, "COHORT_OFFICIAL_RECEIPT_COUNT")
    _exact(len(seen_local), 40, "COHORT_LOCAL_RECEIPT_COUNT")
    _exact(total_bytes, 1_958_756, "COHORT_RECEIPT_BYTES")
    return frames


def _manifest_file(root: Path, receipt: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    relative = _relative_posix(receipt["local_path"], f"{label}_LOCAL_PATH")
    candidate = _existing_file(root / relative, label)
    if not _is_within(candidate, root):
        _fail(f"{label}_ESCAPES_EXTRACTED_ROOT", str(candidate))
    verified = _validate_file(candidate, receipt["bytes"], receipt["sha256"], label)
    return candidate, {
        "official_path": receipt["path"],
        "local_path": receipt["local_path"],
        **verified,
    }


def _git_output(source_root: Path, arguments: list[str], label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(f"SOURCE_GIT_{label}_FAILED", str(exc))
    if completed.returncode != 0:
        _fail(f"SOURCE_GIT_{label}_FAILED", completed.stderr.strip())
    return completed.stdout.strip()


def _validate_model_and_source(checkpoint_argument: Path, source_argument: Path) -> dict[str, Any]:
    checkpoint = _existing_file(checkpoint_argument, "CHECKPOINT")
    expected_checkpoint = _existing_file(ROOT / CHECKPOINT_RELATIVE, "EXPECTED_CHECKPOINT")
    if checkpoint != expected_checkpoint:
        _fail("CHECKPOINT_PATH_MISMATCH", f"{checkpoint}:{expected_checkpoint}")
    checkpoint_receipt = _validate_file(checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "CHECKPOINT")
    model_files = [
        _validate_file(checkpoint.parent / Path(*PurePosixPath(path).parts), size, digest, f"MODEL_FILE_{index}")
        for index, (path, size, digest) in enumerate(MODEL_FILES)
    ]

    source_root = _existing_directory(source_argument, "SOURCE_ROOT")
    expected_source = _existing_directory(ROOT / SOURCE_RELATIVE, "EXPECTED_SOURCE_ROOT")
    if source_root != expected_source:
        _fail("SOURCE_ROOT_PATH_MISMATCH", f"{source_root}:{expected_source}")
    source_files = [
        _validate_file(source_root / Path(*PurePosixPath(path).parts), size, digest, f"SOURCE_FILE_{index}")
        for index, (path, size, digest) in enumerate(SOURCE_FILES)
    ]
    head = _git_output(source_root, ["rev-parse", "HEAD"], "HEAD")
    _exact(head, SOURCE_COMMIT, "SOURCE_GIT_HEAD")
    origin = _git_output(source_root, ["config", "--get", "remote.origin.url"], "ORIGIN")
    _exact(origin.rstrip("/"), SOURCE_ORIGIN.rstrip("/"), "SOURCE_GIT_ORIGIN")
    status = _git_output(source_root, ["status", "--porcelain=v1", "--untracked-files=all"], "STATUS")
    _exact(status, "", "SOURCE_GIT_STATUS")
    return {
        "checkpoint": checkpoint_receipt,
        "model_files": model_files,
        "source_root": str(source_root),
        "source_repository": SOURCE_REPOSITORY,
        "source_origin": origin,
        "source_commit": head,
        "source_git_clean": True,
        "source_files": source_files,
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        _fail("RUNTIME_DISTRIBUTION_MISSING", name)


def _validate_runtime_and_import(source_root: Path, protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_python = _existing_file(Path(PYTHON_EXE_FROZEN), "FROZEN_PYTHON")
    actual_python = _existing_file(Path(sys.executable), "RUNNING_PYTHON")
    if actual_python != expected_python:
        _fail("PYTHON_EXECUTABLE_PATH_MISMATCH", f"{actual_python}:{expected_python}")
    python_receipt = _validate_hash_only(actual_python, PYTHON_EXE_SHA256, "PYTHON_EXECUTABLE")
    launcher_receipt = _validate_hash_only(Path(LAUNCHER_FROZEN), LAUNCHER_SHA256, "LAUNCHER")
    _exact(platform.python_version(), "3.11.9", "PYTHON_VERSION")
    if not sys.flags.no_user_site:
        _fail("PYTHON_USER_SITE_NOT_DISABLED", "invoke through the frozen launcher")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        _fail("OFFLINE_ENVIRONMENT_NOT_FROZEN")
    if os.environ.get("BLINDASSIST_SAM_LICENSE_ACCEPTED") != "1":
        _fail("SAM_LICENSE_ACCEPTANCE_NOT_ASSERTED", "set BLINDASSIST_SAM_LICENSE_ACCEPTED=1 only after operator acceptance")
    distributions = {name: _distribution_version(name) for name in EXPECTED_DISTRIBUTIONS}
    _exact(distributions, EXPECTED_DISTRIBUTIONS, "RUNTIME_DISTRIBUTIONS")

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        import numpy as np
        import PIL
        import pycocotools
        import torch
        import torchvision
        import triton
        import sam3
        from PIL import Image
        from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.vl_combiner import SAM3VLBackboneTri
        from sam3.model_builder import (
            _create_dot_product_scoring,
            _create_geometry_encoder,
            _create_multiplex_tri_backbone,
            _create_sam3_transformer,
            _create_segmentation_head,
            _create_text_encoder,
        )
    except ImportError as exc:
        _fail("INFERENCE_RUNTIME_IMPORT_FAILED", str(exc))
    _exact(np.__version__, "1.26.4", "NUMPY_MODULE_VERSION")
    _exact(PIL.__version__, "12.2.0", "PILLOW_MODULE_VERSION")
    _exact(torch.__version__, "2.11.0+cu130", "TORCH_MODULE_VERSION")
    _exact(torchvision.__version__, "0.26.0+cu130", "TORCHVISION_MODULE_VERSION")
    _exact(triton.__version__, "3.6.0", "TRITON_MODULE_VERSION")
    sam3_file = _existing_file(Path(sam3.__file__), "IMPORTED_SAM3_PACKAGE")
    if not _is_within(sam3_file, source_root):
        _fail("SAM3_IMPORT_OUTSIDE_FROZEN_SOURCE", str(sam3_file))
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        _fail("CUDA_ZERO_UNAVAILABLE")
    torch.cuda.set_device(0)
    _exact(torch.cuda.current_device(), 0, "CUDA_CURRENT_DEVICE")
    _exact(torch.version.cuda, "13.0", "TORCH_CUDA_RUNTIME")
    runtime = protocol["runtime"]
    device_name = torch.cuda.get_device_name(0)
    _exact(device_name, runtime.get("gpu"), "CUDA_DEVICE_NAME")
    properties = torch.cuda.get_device_properties(0)
    memory_mib = properties.total_memory // (1024 * 1024)
    memory_mib_min = runtime.get("gpu_memory_mib_min")
    if not isinstance(memory_mib_min, int) or isinstance(memory_mib_min, bool):
        _fail("CUDA_DEVICE_MEMORY_MINIMUM_INVALID", repr(memory_mib_min))
    if memory_mib < memory_mib_min:
        _fail("CUDA_DEVICE_MEMORY_BELOW_MINIMUM", f"actual={memory_mib}:minimum={memory_mib_min}")
    receipt = {
        "launcher": launcher_receipt,
        "python_executable": python_receipt,
        "python": platform.python_version(),
        "python_user_site_disabled": True,
        "distributions": distributions,
        "module_versions": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "triton": triton.__version__,
        },
        "offline_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "sam_license_acceptance_asserted": True,
        "sam3_import_path": str(sam3_file),
        "cuda": {
            "device": "cuda:0",
            "name": device_name,
            "total_memory_bytes": properties.total_memory,
            "total_memory_mib_floor": memory_mib,
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime": torch.version.cuda,
        },
    }
    modules = {
        "np": np,
        "PIL": PIL,
        "pycocotools": pycocotools,
        "torch": torch,
        "Image": Image,
        "Detector": Sam3MultiplexDetector,
        "Processor": Sam3Processor,
        "TriBackbone": SAM3VLBackboneTri,
        "create_dot_product_scoring": _create_dot_product_scoring,
        "create_geometry_encoder": _create_geometry_encoder,
        "create_multiplex_tri_backbone": _create_multiplex_tri_backbone,
        "create_transformer": _create_sam3_transformer,
        "create_segmentation_head": _create_segmentation_head,
        "create_text_encoder": _create_text_encoder,
    }
    return receipt, modules


def _stable_native_order(scores: Sequence[float], cap: int = MAXIMUM_INSTANCES_PER_PROMPT) -> list[int]:
    if cap < 0:
        _fail("INSTANCE_CAP_INVALID", str(cap))
    values = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in values):
        _fail("NONFINITE_NATIVE_SCORE")
    return sorted(range(len(values)), key=lambda index: -values[index])[:cap]


def _mask_receipt(mask: Any, np: Any) -> dict[str, Any]:
    binary = np.ascontiguousarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not bool(np.all((binary == 0) | (binary == 1))):
        _fail("MASK_NOT_2D_BINARY")
    return {
        "encoding": "C_ORDER_UINT8_BINARY_RECEIPT_ONLY",
        "shape_hw": [int(binary.shape[0]), int(binary.shape[1])],
        "true_pixels": int(binary.sum(dtype=np.uint64)),
        "binary_uint8_sha256": hashlib.sha256(binary.tobytes(order="C")).hexdigest(),
    }


def _snapshot_prompt_output(
    state: dict[str, Any],
    prompt: dict[str, Any],
    image_size: tuple[int, int],
    elapsed_seconds: float,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    required = {"scores", "boxes", "masks_logits", "masks"}
    if not required.issubset(state):
        _fail("SAM3_OUTPUT_FIELDS_MISSING", f"call={prompt['global_call_index']}")
    scores = state["scores"].detach().cpu()
    boxes = state["boxes"].detach().cpu()
    probabilities = state["masks_logits"].detach().cpu()
    masks = state["masks"].detach().cpu()
    width, height = image_size
    count = int(scores.shape[0]) if scores.ndim == 1 else -1
    expected_masks = (count, 1, height, width)
    if tuple(boxes.shape) != (count, 4) or tuple(probabilities.shape) != expected_masks or tuple(masks.shape) != expected_masks:
        _fail(
            "SAM3_NATIVE_OUTPUT_SHAPE_MISMATCH",
            f"call={prompt['global_call_index']}:scores={tuple(scores.shape)}:boxes={tuple(boxes.shape)}:probs={tuple(probabilities.shape)}:masks={tuple(masks.shape)}",
        )
    if masks.dtype != torch.bool:
        _fail("SAM3_NATIVE_MASK_DTYPE_MISMATCH", str(masks.dtype))
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(probabilities).all()):
        _fail("SAM3_NATIVE_OUTPUT_NONFINITE", f"call={prompt['global_call_index']}")
    if count and (not bool((scores > CONFIDENCE_THRESHOLD).all())):
        _fail("SAM3_NATIVE_SCORE_THRESHOLD_MISMATCH", f"call={prompt['global_call_index']}")
    if probabilities.numel() and (not bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())):
        _fail("SAM3_NATIVE_MASK_PROBABILITY_RANGE_INVALID", f"call={prompt['global_call_index']}")
    if not torch.equal(masks, probabilities > MASK_THRESHOLD):
        _fail("SAM3_NATIVE_MASK_THRESHOLD_IDENTITY_MISMATCH", f"call={prompt['global_call_index']}")

    score_values = [float(value) for value in scores.tolist()]
    order = _stable_native_order(score_values)
    instances: list[dict[str, Any]] = []
    for stable_index, native_index in enumerate(order):
        mask = np.ascontiguousarray(masks[native_index, 0].numpy(), dtype=np.bool_)
        ys, xs = np.nonzero(mask)
        receipt = _mask_receipt(mask, np)
        instance = {
            "global_call_index": prompt["global_call_index"],
            "family": prompt["family"],
            "class_id": prompt["class_id"],
            "text": prompt["text"],
            "native_filtered_output_index": native_index,
            "stable_instance_index": stable_index,
            "score": score_values[native_index],
            "box_xyxy": [float(value) for value in boxes[native_index].tolist()],
            "mask": receipt,
            "_mask": mask,
            "_true_x_sum": int(xs.sum(dtype=np.int64)),
            "_true_y_sum": int(ys.sum(dtype=np.int64)),
        }
        instances.append(instance)
    return {
        "global_call_index": prompt["global_call_index"],
        "family": prompt["family"],
        "class_id": prompt["class_id"],
        "text": prompt["text"],
        "native_filtered_count": count,
        "retained_count": len(instances),
        "discarded_by_per_prompt_cap": max(0, count - len(instances)),
        "invalid_output_count": 0,
        "stable_sort": "DESCENDING_SCORE_NATIVE_ORDER_TIES",
        "inference_and_native_postprocess_seconds": elapsed_seconds,
        "_instances": instances,
    }


def _public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in instance.items() if not key.startswith("_")}


def _public_call(call: dict[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in call.items() if not key.startswith("_")}
    value["retained_instances"] = [_public_instance(instance) for instance in call["_instances"]]
    value["retained_instance_ids"] = [instance["instance_id"] for instance in call["_instances"]]
    return value


def _assign_instance_ids(calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for call in calls:
        target = parents if call["family"] == "PARENT" else children
        target.extend(call["_instances"])
    for index, parent in enumerate(parents):
        parent["instance_id"] = f"parent:{index:03d}"
    for index, child in enumerate(children):
        child["instance_id"] = f"child:{index:03d}"
    return parents, children


def _assign_topology(parents: list[dict[str, Any]], children: list[dict[str, Any]], image_size: tuple[int, int]) -> tuple[list[dict[str, Any]], bool]:
    width, height = image_size
    assignments: list[dict[str, Any]] = []
    frame_authorized = False
    for child in children:
        area = child["mask"]["true_pixels"]
        if area == 0:
            mean = None
            representative = None
            candidates: list[dict[str, Any]] = []
            status = "EMPTY_UNASSIGNED"
        else:
            mean_x = child["_true_x_sum"] / area
            mean_y = child["_true_y_sum"] / area
            mean = [mean_x, mean_y]
            representative = [math.floor(mean_x + 0.5), math.floor(mean_y + 0.5)]
            if not (0 <= representative[0] < width and 0 <= representative[1] < height):
                _fail("CHILD_REPRESENTATIVE_OUT_OF_BOUNDS", child["instance_id"])
            candidates = [
                parent for parent in parents
                if bool(parent["_mask"][representative[1]][representative[0]])
            ]
            candidates.sort(
                key=lambda parent: (
                    parent["mask"]["true_pixels"],
                    PARENT_PRIORITY[parent["class_id"]],
                    parent["stable_instance_index"],
                )
            )
            status = "ASSIGNED" if candidates else "UNASSIGNED"
        assigned = candidates[0] if candidates else None
        authorizes = assigned is not None and assigned["class_id"] == "architectural_leaf"
        frame_authorized = frame_authorized or authorizes
        assignments.append(
            {
                "child_instance_id": child["instance_id"],
                "child_class_id": child["class_id"],
                "child_mask_area_pixels": area,
                "status": status,
                "mean_integer_true_mask_pixel_xy": mean,
                "representative_pixel_xy": representative,
                "containing_parents_in_assignment_order": [
                    {
                        "parent_instance_id": parent["instance_id"],
                        "parent_class_id": parent["class_id"],
                        "mask_area_pixels": parent["mask"]["true_pixels"],
                        "class_priority": PARENT_PRIORITY[parent["class_id"]],
                        "stable_instance_index": parent["stable_instance_index"],
                    }
                    for parent in candidates
                ],
                "assigned_parent_instance_id": assigned["instance_id"] if assigned else None,
                "assigned_parent_class_id": assigned["class_id"] if assigned else None,
                "authorizes": authorizes,
            }
        )
    return assignments, frame_authorized


def _aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) != 8 or Counter(row.get("role") for row in rows) != Counter(ROLE_COUNTS):
        _fail("RESULT_ROLE_DENOMINATOR_MISMATCH")
    key = "sam3_native_full_frame_part_topology"
    tp = sum(row["role"] == POSITIVE_ROLE and row[key] for row in rows)
    fn = ROLE_COUNTS[POSITIVE_ROLE] - tp
    furniture_fp = sum(row["role"] == FURNITURE_ROLE and row[key] for row in rows)
    ood_fp = sum(row["role"] == OOD_ROLE and row[key] for row in rows)
    fp = furniture_fp + ood_fp
    tn = ROLE_COUNTS[FURNITURE_ROLE] + ROLE_COUNTS[OOD_ROLE] - fp
    recall = tp / ROLE_COUNTS[POSITIVE_ROLE]
    true_negative_rate = tn / (ROLE_COUNTS[FURNITURE_ROLE] + ROLE_COUNTS[OOD_ROLE])
    false_positive_rate = fp / (ROLE_COUNTS[FURNITURE_ROLE] + ROLE_COUNTS[OOD_ROLE])
    balanced_accuracy = 0.5 * (recall + true_negative_rate)
    native_counts = Counter()
    retained_counts = Counter()
    capped = 0
    empty_children = 0
    assigned_children = 0
    text_seconds = 0.0
    image_seconds = 0.0
    for row in rows:
        image_seconds += row["native_inference"]["set_image"]["inference_seconds"]
        for call in row["native_inference"]["text_calls"]:
            native_counts[call["class_id"]] += call["native_filtered_count"]
            retained_counts[call["class_id"]] += call["retained_count"]
            capped += call["discarded_by_per_prompt_cap"]
            text_seconds += call["inference_and_native_postprocess_seconds"]
        for assignment in row["native_inference"]["child_parent_assignments"]:
            empty_children += assignment["status"] == "EMPTY_UNASSIGNED"
            assigned_children += assignment["status"] == "ASSIGNED"
    metrics = {
        "frames": len(rows),
        "set_image_calls": len(rows),
        "set_text_prompt_calls": len(rows) * len(PROMPTS),
        "total_model_calls": len(rows) * (1 + len(PROMPTS)),
        "native_instances_by_class_id": {class_id: native_counts[class_id] for class_id in PARENT_CLASSES + CHILD_CLASSES},
        "retained_instances_by_class_id": {class_id: retained_counts[class_id] for class_id in PARENT_CLASSES + CHILD_CLASSES},
        "discarded_by_per_prompt_cap": capped,
        "empty_child_masks_retained_unassigned": empty_children,
        "assigned_children": assigned_children,
        "image_encoding_seconds": image_seconds,
        "text_inference_and_native_postprocess_seconds": text_seconds,
        "true_positive_count": tp,
        "false_negative_count": fn,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "handled_furniture_false_positive_count": furniture_fp,
        "doorless_opening_ood_false_positive_count": ood_fp,
        "positive_recall": recall,
        "negative_true_negative_rate": true_negative_rate,
        "negative_false_positive_rate": false_positive_rate,
        "balanced_accuracy": balanced_accuracy,
    }
    gate_met = tp == 4 and fn == 0 and furniture_fp == 0 and ood_fp == 0 and balanced_accuracy == 1.0
    gate = {
        "four_of_four_architectural_doors_authorized": tp == 4,
        "zero_of_two_handled_furniture_controls_authorized": furniture_fp == 0,
        "zero_of_two_large_doorless_opening_controls_authorized": ood_fp == 0,
        "balanced_accuracy_is_one": balanced_accuracy == 1.0,
        "development_gate_met": gate_met,
        "decision": (
            "L10_PB16_SAM3_NATIVE_PART_TOPOLOGY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_PB16_SAM3_NATIVE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET"
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
    except (OSError, ValueError) as exc:
        _fail("OUTPUT_WRITE_FAILED", str(exc))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _build_strict_sam31_image_detector(
    checkpoint_path: Path,
    source_root: Path,
    modules: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = modules["torch"]
    try:
        tri_neck = modules["create_multiplex_tri_backbone"](
            compile_mode=None,
            use_fa3=False,
            use_rope_real=False,
        )
        backbone = modules["TriBackbone"](
            scalp=0,
            visual=tri_neck,
            text=modules["create_text_encoder"](
                str(source_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
            ),
        )
        model = modules["Detector"](
            num_feature_levels=1,
            backbone=backbone,
            transformer=modules["create_transformer"](use_fa3=False),
            segmentation_head=modules["create_segmentation_head"](use_fa3=False),
            semantic_segmentation_head=None,
            input_geometry_encoder=modules["create_geometry_encoder"](),
            use_early_fusion=True,
            use_dot_prod_scoring=True,
            dot_prod_scoring=modules["create_dot_product_scoring"](),
            supervise_joint_box_scores=True,
            is_multiplex=True,
        )
        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        checkpoint_state = (
            raw["model"]
            if isinstance(raw, dict) and isinstance(raw.get("model"), dict)
            else raw
        )
        if not isinstance(checkpoint_state, dict):
            _fail("SAM31_CHECKPOINT_STATE_NOT_DICTIONARY")
        detector_state = {
            key.removeprefix("detector."): value
            for key, value in checkpoint_state.items()
            if isinstance(key, str) and key.startswith("detector.")
        }
        if not detector_state:
            _fail("SAM31_DETECTOR_NAMESPACE_EMPTY")
        incompatibility = model.load_state_dict(detector_state, strict=True)
        _exact(list(incompatibility.missing_keys), [], "SAM31_STRICT_LOAD_MISSING_KEYS")
        _exact(list(incompatibility.unexpected_keys), [], "SAM31_STRICT_LOAD_UNEXPECTED_KEYS")
        receipt = {
            "adapter": STRICT_ADAPTER_NAME,
            "recipe_source": STRICT_ADAPTER_ARGUMENTS["recipe_source"],
            "checkpoint_total_state_keys": len(checkpoint_state),
            "detector_state_keys": len(detector_state),
            "ignored_non_detector_state_keys": len(checkpoint_state) - len(detector_state),
            "checkpoint_namespace": "detector.",
            "strict_state_dict_loading": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "tracker_instantiated": False,
            "video_predictor_instantiated": False,
        }
        del detector_state, checkpoint_state, raw
        gc.collect()
        model.to(device="cuda").eval()
        return model, receipt
    except ContractError:
        raise
    except Exception as exc:
        _fail("SAM31_STRICT_IMAGE_ADAPTER_LOAD_FAILED", repr(exc))


def _validate_loaded_model(model: Any, processor: Any, torch: Any) -> dict[str, Any]:
    if model.training:
        _fail("SAM3_MODEL_NOT_EVAL")
    if getattr(model, "inst_interactive_predictor", None) is not None:
        _fail("SAM3_INSTANCE_INTERACTIVITY_ENABLED")
    _exact(processor.model is model, True, "SAM3_PROCESSOR_MODEL_BINDING")
    _exact(processor.resolution, PROCESSOR_RESOLUTION, "SAM3_PROCESSOR_RESOLUTION")
    _exact(processor.device, "cuda", "SAM3_PROCESSOR_DEVICE")
    _exact(processor.confidence_threshold, CONFIDENCE_THRESHOLD, "SAM3_PROCESSOR_CONFIDENCE")
    _exact(type(model).__name__, "Sam3MultiplexDetector", "SAM31_DETECTOR_TYPE")
    _exact(getattr(model, "is_multiplex", None), True, "SAM31_DETECTOR_MULTIPLEX_FLAG")
    vision_backbone = getattr(getattr(model, "backbone", None), "vision_backbone", None)
    _exact(type(vision_backbone).__name__, "Sam3TriViTDetNeck", "SAM31_TRI_NECK_TYPE")
    _exact(len(getattr(vision_backbone, "convs", [])), 3, "SAM31_TRI_NECK_SCALE_COUNT")
    parameter_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    _exact(parameter_dtypes, {str(torch.float32)}, "SAM3_PARAMETER_DTYPES")
    _exact(parameter_devices, {"cuda:0"}, "SAM3_PARAMETER_DEVICES")
    return {
        "training": bool(model.training),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "parameter_dtypes": sorted(parameter_dtypes),
        "parameter_devices": sorted(parameter_devices),
        "instance_interactivity_enabled": False,
        "image_only_adapter": STRICT_ADAPTER_NAME,
        "model_type": type(model).__name__,
        "is_multiplex": bool(model.is_multiplex),
        "vision_neck_type": type(vision_backbone).__name__,
        "vision_neck_scale_count": len(vision_backbone.convs),
        "processor": {
            "resolution": processor.resolution,
            "device": processor.device,
            "confidence_threshold": processor.confidence_threshold,
        },
    }


def _run(
    protocol_argument: Path,
    cohort_argument: Path,
    extracted_root_argument: Path,
    checkpoint_argument: Path,
    source_root_argument: Path,
    output_argument: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = _existing_file(protocol_argument, "PROTOCOL")
    cohort_path = _existing_file(cohort_argument, "COHORT")
    extracted_root = _existing_directory(extracted_root_argument, "EXTRACTED_ROOT")
    output_path = _new_output(output_argument)
    evaluator_path = Path(__file__).resolve()
    evaluator_hash = _sha256(evaluator_path)

    protocol_hash = _sha256(protocol_path)
    protocol = _read_json(protocol_path, "PROTOCOL")
    _validate_protocol(protocol, protocol_hash)
    cohort_hash = _sha256(cohort_path)
    cohort = _read_json(cohort_path, "COHORT")
    frames = _validate_cohort(cohort, cohort_hash)
    expected_extracted = _existing_directory(ROOT / _relative_posix(cohort["source"]["extracted_root"], "COHORT_EXTRACTED_ROOT"), "EXPECTED_EXTRACTED_ROOT")
    if extracted_root != expected_extracted:
        _fail("EXTRACTED_ROOT_PATH_MISMATCH", f"{extracted_root}:{expected_extracted}")

    prepared: list[dict[str, Any]] = []
    member_bytes = 0
    for frame in frames:
        paths: dict[str, Path] = {}
        receipts: dict[str, Any] = {}
        for kind in sorted(FILE_KINDS):
            path, receipt = _manifest_file(extracted_root, frame["files"][kind], f"FRAME_{frame['index']}_{kind.upper()}")
            paths[kind] = path
            receipts[kind] = receipt
            member_bytes += receipt["bytes"]
        prepared.append({"frame": frame, "paths": paths, "receipts": receipts})
    _exact(sum(len(item["receipts"]) for item in prepared), 40, "ACTUAL_MEMBER_COUNT")
    _exact(member_bytes, 1_958_756, "ACTUAL_MEMBER_BYTES")

    asset_receipts = _validate_model_and_source(checkpoint_argument, source_root_argument)
    source_root = Path(asset_receipts["source_root"])
    runtime_receipt, modules = _validate_runtime_and_import(source_root, protocol)
    np = modules["np"]
    torch = modules["torch"]
    Image = modules["Image"]

    model_load_started = time.perf_counter()
    try:
        model, strict_load_receipt = _build_strict_sam31_image_detector(
            Path(asset_receipts["checkpoint"]["path"]),
            source_root,
            modules,
        )
        processor = modules["Processor"](
            model,
            resolution=PROCESSOR_RESOLUTION,
            device="cuda",
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
    except Exception as exc:
        _fail("SAM3_MODEL_LOAD_FAILED", repr(exc))
    torch.cuda.synchronize(0)
    model_load_seconds = time.perf_counter() - model_load_started
    loaded_receipt = _validate_loaded_model(model, processor, torch)
    loaded_receipt["checkpoint_load"] = strict_load_receipt

    rows: list[dict[str, Any]] = []
    global_peak_allocated = 0
    global_peak_reserved = 0
    for item in prepared:
        frame = item["frame"]
        width, height = frame["source_image_size"]
        try:
            with Image.open(item["paths"]["rgb"]) as opened:
                if [int(opened.width), int(opened.height)] != [width, height]:
                    _fail("RGB_SIZE_MISMATCH", f"frame={frame['index']}:{opened.size}")
                image = opened.convert("RGB")
                image.load()
        except ContractError:
            raise
        except Exception as exc:
            _fail("RGB_DECODE_FAILED", f"frame={frame['index']}:{exc}")
        if image.mode != "RGB" or image.size != (width, height):
            _fail("RGB_CONVERT_IDENTITY_MISMATCH", f"frame={frame['index']}")

        torch.cuda.reset_peak_memory_stats(0)
        calls: list[dict[str, Any]] = []
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                torch.cuda.synchronize(0)
                encode_started = time.perf_counter()
                state = processor.set_image(image)
                torch.cuda.synchronize(0)
                encode_seconds = time.perf_counter() - encode_started
                if not isinstance(state, dict) or "backbone_out" not in state:
                    _fail("SAM3_SET_IMAGE_STATE_INVALID", f"frame={frame['index']}")
                _exact(state.get("original_width"), width, f"FRAME_{frame['index']}_STATE_WIDTH")
                _exact(state.get("original_height"), height, f"FRAME_{frame['index']}_STATE_HEIGHT")
                backbone_identity = id(state["backbone_out"])
                for prompt in PROMPTS:
                    torch.cuda.synchronize(0)
                    call_started = time.perf_counter()
                    returned = processor.set_text_prompt(prompt["text"], state)
                    torch.cuda.synchronize(0)
                    call_seconds = time.perf_counter() - call_started
                    if returned is not state or id(state.get("backbone_out")) != backbone_identity:
                        _fail("SAM3_IMAGE_FEATURE_STATE_NOT_REUSED", f"frame={frame['index']}:call={prompt['global_call_index']}")
                    calls.append(_snapshot_prompt_output(state, prompt, (width, height), call_seconds, torch, np))
        except ContractError:
            raise
        except Exception as exc:
            _fail("SAM3_FRAME_INFERENCE_FAILED", f"frame={frame['index']}:{exc!r}")
        parents, children = _assign_instance_ids(calls)
        assignments, authorized = _assign_topology(parents, children, (width, height))
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        global_peak_allocated = max(global_peak_allocated, peak_allocated)
        global_peak_reserved = max(global_peak_reserved, peak_reserved)
        rows.append(
            {
                "index": frame["index"],
                "frame_id": frame["frame_id"],
                "capture_sequence_id": frame["capture_sequence_id"],
                "sensor_source_bucket": frame["sensor_source_bucket"],
                "role": frame["role"],
                "expected_binary_label": frame["role"] == POSITIVE_ROLE,
                "canonical_source_path": frame["canonical_source_path"],
                "audit_note": frame["audit_note"],
                "official_evidence": frame["official_evidence"],
                "source_image_size_wh": frame["source_image_size"],
                "files": item["receipts"],
                "native_inference": {
                    "set_image": {
                        "calls": 1,
                        "full_frame_rgb_only": True,
                        "source_image_size_wh": frame["source_image_size"],
                        "processor_input_resolution": PROCESSOR_RESOLUTION,
                        "inference_seconds": encode_seconds,
                    },
                    "text_calls": [_public_call(call) for call in calls],
                    "parent_instance_ids_in_prompt_then_stable_order": [parent["instance_id"] for parent in parents],
                    "child_instance_ids_in_prompt_then_stable_order": [child["instance_id"] for child in children],
                    "child_parent_assignments": assignments,
                    "peak_cuda_allocated_bytes": peak_allocated,
                    "peak_cuda_reserved_bytes": peak_reserved,
                },
                "sam3_native_full_frame_part_topology": authorized,
            }
        )
        del image, state, calls, parents, children

    torch.cuda.synchronize(0)
    metrics, gate = _aggregate(rows)
    if _sha256(evaluator_path) != evaluator_hash:
        _fail("EVALUATOR_CHANGED_DURING_RUN")
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "L10-PB16 SAM3.1 Native Full-Frame Part Topology",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "inputs": {
            "protocol": {"path": str(protocol_path), "schema": PROTOCOL_SCHEMA, "sha256": protocol_hash},
            "cohort": {"path": str(cohort_path), "schema": COHORT_SCHEMA, "sha256": cohort_hash, "status": cohort["status"]},
            "extracted_root": str(extracted_root),
            "verified_local_members": 40,
            "verified_local_member_bytes": member_bytes,
            "model_and_source": asset_receipts,
        },
        "evaluator": {"path": str(evaluator_path), "sha256": evaluator_hash},
        "runtime": {
            **runtime_receipt,
            "model_load_seconds": model_load_seconds,
            "loaded_model": loaded_receipt,
            "maximum_frame_peak_cuda_allocated_bytes": global_peak_allocated,
            "maximum_frame_peak_cuda_reserved_bytes": global_peak_reserved,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "configuration": {
            "builder": STRICT_ADAPTER_NAME,
            "builder_arguments": STRICT_ADAPTER_ARGUMENTS,
            "checkpoint_load": strict_load_receipt,
            "device": "cuda",
            "eval_mode": True,
            "compile": False,
            "enable_segmentation": True,
            "enable_inst_interactivity": False,
            "processor": {"resolution": PROCESSOR_RESOLUTION, "device": "cuda", "confidence_threshold": CONFIDENCE_THRESHOLD},
            "autocast": "CUDA_BFLOAT16",
            "inference_mode": True,
            "full_frame_only": True,
            "set_image_calls_per_frame": 1,
            "prompt_calls_in_order": PROMPTS,
            "maximum_instances_per_prompt": MAXIMUM_INSTANCES_PER_PROMPT,
            "per_prompt_order": "STABLE_DESCENDING_SCORE_NATIVE_OUTPUT_ORDER_FOR_TIES",
            "mask_threshold": "STRICTLY_GREATER_THAN_0.5_ON_RETURNED_POST_SIGMOID_PROBABILITY",
            "boxes_used_for_topology": False,
            "parent_class_priority_for_equal_area": PARENT_PRIORITY_ORDER,
            "authorization": "AT_LEAST_ONE_RETAINED_CHILD_ASSIGNED_TO_ARCHITECTURAL_LEAF",
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


def _synthetic_instance(instance_id: str, class_id: str, stable_index: int, points: set[tuple[int, int]], size: tuple[int, int]) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "class_id": class_id,
        "stable_instance_index": stable_index,
        "mask": {"true_pixels": len(points)},
        "_mask": _synthetic_mask(*size, points),
        "_true_x_sum": sum(x for x, _ in points),
        "_true_y_sum": sum(y for _, y in points),
    }


def _self_test() -> dict[str, Any]:
    forbidden_before = sorted(name for name in sys.modules if name == "torch" or name.startswith("sam3"))
    if forbidden_before:
        raise AssertionError(f"heavy modules imported before self-test: {forbidden_before}")
    if _stable_native_order([0.8, 0.9, 0.9, 0.7], 3) != [1, 2, 0]:
        raise AssertionError("descending score/native tie order mismatch")
    size = (10, 10)
    architectural_points = {(x, y) for y in range(2, 8) for x in range(2, 8)}
    cabinet_points = {(x, y) for y in range(2, 8) for x in range(2, 8)}
    parents = [
        _synthetic_instance("parent:000", "architectural_leaf", 0, architectural_points, size),
        _synthetic_instance("parent:001", "cabinet_door", 0, cabinet_points, size),
    ]
    child_points = {(4, 4), (5, 4), (4, 5), (5, 5)}
    child = _synthetic_instance("child:000", "handle", 0, child_points, size)
    assignments, authorized = _assign_topology(parents, [child], size)
    if assignments[0]["representative_pixel_xy"] != [5, 5]:
        raise AssertionError("round-half-up representative mismatch")
    if assignments[0]["assigned_parent_class_id"] != "cabinet_door" or authorized:
        raise AssertionError("conservative equal-area parent priority mismatch")
    assignments, authorized = _assign_topology(parents[:1], [child], size)
    if assignments[0]["assigned_parent_class_id"] != "architectural_leaf" or not authorized:
        raise AssertionError("architectural authorization mismatch")
    empty = _synthetic_instance("child:001", "knob", 0, set(), size)
    empty_assignment, empty_authorized = _assign_topology(parents, [empty], size)
    if empty_assignment[0]["status"] != "EMPTY_UNASSIGNED" or empty_assignment[0]["representative_pixel_xy"] is not None or empty_authorized:
        raise AssertionError("empty child was not retained as EMPTY_UNASSIGNED")
    rows = []
    for index in range(8):
        role = POSITIVE_ROLE if index < 4 else FURNITURE_ROLE if index < 6 else OOD_ROLE
        rows.append(
            {
                "role": role,
                "sam3_native_full_frame_part_topology": index < 4,
                "native_inference": {
                    "set_image": {"inference_seconds": 0.0},
                    "text_calls": [
                        {
                            "class_id": prompt["class_id"],
                            "native_filtered_count": 0,
                            "retained_count": 0,
                            "discarded_by_per_prompt_cap": 0,
                            "inference_and_native_postprocess_seconds": 0.0,
                        }
                        for prompt in PROMPTS
                    ],
                    "child_parent_assignments": [],
                },
            }
        )
    metrics, gate = _aggregate(rows)
    if metrics["balanced_accuracy"] != 1.0 or not gate["development_gate_met"]:
        raise AssertionError("perfect frozen gate did not pass")
    rows[0]["sam3_native_full_frame_part_topology"] = False
    _, failed = _aggregate(rows)
    if failed["development_gate_met"]:
        raise AssertionError("false negative passed frozen gate")
    rows[0]["sam3_native_full_frame_part_topology"] = True
    rows[4]["sam3_native_full_frame_part_topology"] = True
    _, false_positive = _aggregate(rows)
    if false_positive["development_gate_met"]:
        raise AssertionError("furniture false positive passed frozen gate")
    if _relative_posix("pool/SUNRGBD/frame/image.jpg", "SELF_TEST_PATH").as_posix() != "pool/SUNRGBD/frame/image.jpg":
        raise AssertionError("valid candidate path rejected")
    for unsafe in ("../escape", "/absolute", "C:/drive", "a\\b", "a//b"):
        try:
            _relative_posix(unsafe, "SELF_TEST_UNSAFE_PATH")
        except ContractError:
            continue
        raise AssertionError(f"unsafe candidate path accepted: {unsafe}")
    forbidden_after = sorted(name for name in sys.modules if name == "torch" or name.startswith("sam3"))
    if forbidden_after:
        raise AssertionError(f"heavy modules imported by self-test: {forbidden_after}")
    return {
        "status": "PASS",
        "model_calls": 0,
        "cohort_files_read": 0,
        "cohort_images_read": 0,
        "torch_or_sam3_imports": 0,
        "stable_tie_order": [1, 2, 0],
        "equal_area_competitor_assignment": "cabinet_door",
        "architectural_assignment": "architectural_leaf",
        "empty_child_status": "EMPTY_UNASSIGNED",
        "perfect_gate_decision": gate["decision"],
        "unsafe_candidate_paths_rejected": 5,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen PB16 SAM3.1 native full-frame part-topology evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run standard-library-only sorting, topology, gate, and path checks.")
    run = subparsers.add_parser(
        "run",
        help="Run the single frozen eight-frame CUDA evaluation.",
        epilog="Formal execution also requires HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, and BLINDASSIST_SAM_LICENSE_ACCEPTED=1.",
    )
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--extracted-root", type=Path, default=ROOT / "artifacts.local/datasets")
    run.add_argument("--checkpoint", type=Path, default=ROOT / CHECKPOINT_RELATIVE)
    run.add_argument("--source-root", type=Path, default=ROOT / SOURCE_RELATIVE)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "self-test":
            print(json.dumps(_self_test(), ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        result = _run(args.protocol, args.cohort, args.extracted_root, args.checkpoint, args.source_root, args.output)
    except ContractError as exc:
        print(f"PB16_CONTRACT_ERROR:{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"output": str(Path(args.output).resolve()), "decision": result["gate"]["decision"], "balanced_accuracy": result["metrics"]["balanced_accuracy"]},
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
