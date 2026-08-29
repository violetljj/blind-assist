#!/usr/bin/env python3
"""Run the frozen L10-PB15 Grounded-SAM multiscale part-topology gate."""

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
PROTOCOL_SCHEMA = "l10-named-poi-grounded-sam-multiscale-part-topology-protocol-v1"
COHORT_SCHEMA = "l10-named-poi-grounded-sam-multiscale-part-topology-cohort-v1"
RESULT_SCHEMA = "l10-named-poi-grounded-sam-multiscale-part-topology-development-result-v1"
PROTOCOL_SHA256 = "0fd2641159923cb395ea71b6c5a28a7fc32b44b9d348c1576ddbb8f5cd368cc2"
COHORT_SHA256 = "c88cd328af96ea3c02f5c711c3a7ca3aadcc043f960c36fa019e87d585affe26"

GROUND_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
GROUND_RELATIVE = Path("artifacts.local/models/grounding-dino-tiny-a2bb814")
GROUND_TREE_RELATIVE = GROUND_RELATIVE / ".cache/huggingface/trees" / f"{GROUND_REVISION}.json"
GROUND_TREE_SHA256 = "d541cd1c6f7fa793cff08b87619ed6f6b84f35b650a45af61677e4b89aaae8fc"
SAM_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
SAM_RELATIVE = Path("artifacts.local/models/facebook--sam2.1-hiera-small/snapshots") / SAM_REVISION
SAM_RECEIPT_RELATIVE = Path(
    "artifacts.local/models/facebook--sam2.1-hiera-small/pb15_asset_and_synthetic_smoke_receipt_v1.json"
)
SAM_RECEIPT_SHA256 = "c55b80721e9e88da7d09bc825f2cc9320c7b933b32cb7ab057b63712c93e581b"
PYTHON_EXE_SHA256 = "21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082"
PYTHON_EXE_FROZEN = "E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe"
LAUNCHER_FROZEN = "E:/codex-tools/bin/blindassist-research-gpu.cmd"

GROUND_FILES = [
    (".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    ("added_tokens.json", 82, "909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2"),
    ("config.json", 1644, "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a"),
    ("model.safetensors", 689359096, "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"),
    ("preprocessor_config.json", 457, "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60"),
    ("pytorch_model.bin", 691914602, "f4d69c8403b9a569d90362c5fe8c6ea93dee4c8f08166df5b39e5eca8d227bbe"),
    ("README.md", 2580, "cf46f74c7b6850f1d5cbe406028324d8798148726016d46a69a365b4a2d3e89f"),
    ("special_tokens_map.json", 125, "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
    ("tokenizer_config.json", 1237, "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56"),
    ("tokenizer.json", 711396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    ("vocab.txt", 231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
]
SAM_FILES = [
    (".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    ("README.md", 19974, "6ec2d54879e41ad876d8cded0d641e8bc6ab74d5e095a73afc3f8406372ee6e9"),
    ("config.json", 5698, "97ff9f65b76d107acda4247885f0a5555d0048850ae3c5f97183df289aaecde9"),
    ("model.safetensors", 184305280, "0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60"),
    ("preprocessor_config.json", 683, "6ebf229ee259368ce4a8d4f2fe893a72b053023710853e257253939e601f583d"),
    ("processor_config.json", 95, "f8a68e865cfad115c1c2763f3d93eca7b1c622da06da2a9273eb437fb2389b6d"),
    ("sam2.1_hiera_s.yaml", 3761, "632e5cd0104f5ab6cd4f9d2dfd80a8e7240e481ad7960a13cad2ae3504b88dbd"),
    ("sam2.1_hiera_small.pt", 184416285, "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"),
    ("video_preprocessor_config.json", 705, "9fccfe5f464ec38c2f236d0e6a68e95511c80c22132fc2fa4b9f7b65f24fad95"),
]

PARENT_PROMPTS = [
    {"call_index": 0, "class_id": "architectural_leaf", "text": "door", "role": "PARENT"},
    {"call_index": 1, "class_id": "cabinet_door", "text": "cabinet door", "role": "COMPETING_PARENT"},
    {"call_index": 2, "class_id": "closet_door", "text": "closet door", "role": "COMPETING_PARENT"},
    {"call_index": 3, "class_id": "refrigerator_door", "text": "refrigerator door", "role": "COMPETING_PARENT"},
    {"call_index": 4, "class_id": "doorless_opening", "text": "doorway", "role": "COMPETING_PARENT"},
]
CHILD_PROMPTS = [
    {"call_index": 0, "class_id": "operation_part", "subtype": "handle", "text": "door handle", "role": "CHILD"},
    {"call_index": 1, "class_id": "operation_part", "subtype": "knob", "text": "door knob", "role": "CHILD"},
    {"call_index": 2, "class_id": "operation_part", "subtype": "push_bar", "text": "push bar", "role": "CHILD"},
    {"call_index": 3, "class_id": "operation_part", "subtype": "panic_bar", "text": "panic bar", "role": "CHILD"},
    {"call_index": 4, "class_id": "hinge", "subtype": "hinge", "text": "door hinge", "role": "CHILD"},
]
PARENT_IDS = {item["class_id"] for item in PARENT_PROMPTS}
CHILD_IDS = {item["class_id"] for item in CHILD_PROMPTS}
PARENT_PRIORITY = {
    "cabinet_door": 0,
    "closet_door": 1,
    "refrigerator_door": 2,
    "doorless_opening": 3,
    "architectural_leaf": 4,
}
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
MAX_PARENTS = 12
MAX_CHILDREN = 24

POSITIVE_ROLE = "ARCHITECTURAL_DOOR_WITH_VISIBLE_OPERATION_PART"
FURNITURE_ROLE = "HANDLED_FURNITURE_DOOR_NEGATIVE"
OOD_ROLE = "LARGE_DOORLESS_OPENING_OOD"
ROLE_COUNTS = {POSITIVE_ROLE: 4, FURNITURE_ROLE: 2, OOD_ROLE: 2}
FILE_KINDS = {"rgb", "depth", "intrinsics", "polygon", "scene_metadata"}
EXPECTED_RUNTIME = {
    "python": "3.11.9",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "transformers": "5.14.1",
    "huggingface_hub": "1.24.0",
    "safetensors": "0.8.0",
    "numpy": "2.4.4",
    "pillow": "12.2.0",
    "cuda_runtime": "13.0",
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


def _hex_digest(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        _fail(f"{label}_INVALID", text)
    return text


def _exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label}_MISMATCH", f"actual={actual!r}:expected={expected!r}")


def _official_files(inventory: list[tuple[str, int, str]]) -> list[dict[str, Any]]:
    return [{"path": path, "bytes": size, "sha256": digest} for path, size, digest in inventory]


def _validate_protocol(protocol: dict[str, Any], actual_hash: str) -> None:
    _exact(actual_hash, PROTOCOL_SHA256, "PROTOCOL_HASH")
    _exact(protocol.get("schema"), PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    _exact(protocol.get("status"), "FROZEN_BEFORE_COHORT_MODEL_OUTPUT", "PROTOCOL_STATUS")
    sources = protocol.get("information_sources")
    if not isinstance(sources, dict):
        _fail("PROTOCOL_INFORMATION_SOURCES_MISSING")
    ground = sources.get("grounder")
    sam = sources.get("masker")
    if not isinstance(ground, dict) or not isinstance(sam, dict):
        _fail("PROTOCOL_MODEL_SOURCE_MISSING")
    for key, expected in {
        "repository": "IDEA-Research/grounding-dino-tiny",
        "revision": GROUND_REVISION,
        "snapshot_path": GROUND_RELATIVE.as_posix(),
        "tree_receipt_path": GROUND_TREE_RELATIVE.as_posix(),
        "tree_receipt_sha256": GROUND_TREE_SHA256,
    }.items():
        _exact(ground.get(key), expected, f"PROTOCOL_GROUNDER_{key.upper()}")
    _exact(ground.get("official_files"), _official_files(GROUND_FILES), "PROTOCOL_GROUNDER_FILES")
    for key, expected in {
        "repository": "facebook/sam2.1-hiera-small",
        "revision": SAM_REVISION,
        "snapshot_path": SAM_RELATIVE.as_posix(),
        "asset_receipt_path": SAM_RECEIPT_RELATIVE.as_posix(),
        "asset_receipt_sha256": SAM_RECEIPT_SHA256,
    }.items():
        _exact(sam.get(key), expected, f"PROTOCOL_MASKER_{key.upper()}")
    _exact(sam.get("official_files"), _official_files(SAM_FILES), "PROTOCOL_MASKER_FILES")
    runtime = protocol.get("runtime")
    if not isinstance(runtime, dict):
        _fail("PROTOCOL_RUNTIME_MISSING")
    for key, expected in {**EXPECTED_RUNTIME, "python_executable_sha256": PYTHON_EXE_SHA256}.items():
        _exact(runtime.get(key), expected, f"PROTOCOL_RUNTIME_{key.upper()}")
    _exact(runtime.get("python_executable"), PYTHON_EXE_FROZEN, "PROTOCOL_PYTHON_EXECUTABLE")
    _exact(runtime.get("launcher"), LAUNCHER_FROZEN, "PROTOCOL_LAUNCHER")
    _exact(runtime.get("HF_HUB_OFFLINE"), "1", "PROTOCOL_HF_HUB_OFFLINE")
    _exact(runtime.get("TRANSFORMERS_OFFLINE"), "1", "PROTOCOL_TRANSFORMERS_OFFLINE")
    grounding = protocol.get("grounding")
    if not isinstance(grounding, dict):
        _fail("PROTOCOL_GROUNDING_MISSING")
    _exact(grounding.get("parent_prompt_calls_in_order"), PARENT_PROMPTS, "PROTOCOL_PARENT_PROMPTS")
    _exact(grounding.get("child_prompt_calls_in_order"), CHILD_PROMPTS, "PROTOCOL_CHILD_PROMPTS")
    for key, expected in {
        "batch": 1,
        "model_dtype": "float32",
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "target_size_order": "HEIGHT_WIDTH",
        "maximum_parent_boxes_per_frame": MAX_PARENTS,
        "maximum_child_boxes_per_parent_crop": MAX_CHILDREN,
    }.items():
        _exact(grounding.get(key), expected, f"PROTOCOL_GROUNDING_{key.upper()}")
    _exact(
        grounding.get("processor_text_call"),
        "Pass each frozen text as a single-element candidate-label list [text], never as a scalar string. Transformers 5.14.1 deterministically lowercases and appends the model-required terminal period to candidate-label lists.",
        "PROTOCOL_PROCESSOR_TEXT_CALL",
    )
    defaults = grounding.get("processor_defaults")
    _exact(
        defaults,
        {
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "do_pad": True,
            "shortest_edge": 800,
            "longest_edge": 1333,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "PROTOCOL_GROUNDER_PROCESSOR_DEFAULTS",
    )
    masking = protocol.get("sam_masking")
    if not isinstance(masking, dict):
        _fail("PROTOCOL_SAM_MASKING_MISSING")
    for key, expected in {
        "selection": "Require exactly one bool mask per input box after multimask_output=False; bind by original box order. Ignore iou_scores and do not select, threshold, merge, morph, smooth, or resize beyond the official native-size postprocess.",
        "mask_contract": "Every parent mask must equal source HxW and every child mask must equal exact crop HxW. Record shape, true-pixel area, and SHA-256 of C-order uint8 mask bytes.",
    }.items():
        _exact(masking.get(key), expected, f"PROTOCOL_SAM_{key.upper()}")
    topology = protocol.get("two_scale_topology")
    if not isinstance(topology, dict):
        _fail("PROTOCOL_TOPOLOGY_MISSING")
    _exact(topology.get("learned_or_selected_postprocessing_thresholds_beyond_grounder_and_sam"), 0, "PROTOCOL_EXTRA_THRESHOLDS")


def _validate_cohort(cohort: dict[str, Any], actual_hash: str) -> list[dict[str, Any]]:
    _exact(actual_hash, COHORT_SHA256, "COHORT_HASH")
    _exact(cohort.get("schema"), COHORT_SCHEMA, "COHORT_SCHEMA")
    _exact(cohort.get("status"), "FROZEN_BEFORE_MODEL_OUTPUT", "COHORT_STATUS")
    _exact(cohort.get("protocol"), {"schema": PROTOCOL_SCHEMA, "sha256": PROTOCOL_SHA256}, "COHORT_PROTOCOL")
    _exact(
        set(cohort),
        {"schema", "status", "frozen_at", "protocol", "source", "models", "grounding", "sam_masking", "diversity", "frames"},
        "COHORT_TOP_LEVEL_FIELDS",
    )
    source = cohort.get("source")
    if not isinstance(source, dict):
        _fail("COHORT_SOURCE_MISSING")
    for key, expected in {
        "dataset": "SUN RGB-D",
        "archive_bytes": 6885481608,
        "archive_sha256": "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c",
        "extracted_root": "artifacts.local/datasets/sunrgbd-pb15-grounded-sam-audit-pool-v1",
        "official_members": 40,
        "official_member_bytes": 1936059,
        "prior_pb11_sequence_overlap": 0,
        "prior_pb12_sequence_overlap": 0,
        "prior_pb13_sequence_overlap": 0,
        "prior_pb14_sequence_overlap": 0,
    }.items():
        _exact(source.get(key), expected, f"COHORT_SOURCE_{key.upper()}")
    models = cohort.get("models")
    if not isinstance(models, dict):
        _fail("COHORT_MODELS_MISSING")
    _exact(
        models.get("grounder"),
        {
            "repository": "IDEA-Research/grounding-dino-tiny",
            "revision": GROUND_REVISION,
            "model_safetensors_sha256": GROUND_FILES[3][2],
        },
        "COHORT_GROUNDER",
    )
    _exact(
        models.get("masker"),
        {
            "repository": "facebook/sam2.1-hiera-small",
            "revision": SAM_REVISION,
            "model_safetensors_sha256": SAM_FILES[3][2],
            "asset_receipt_sha256": SAM_RECEIPT_SHA256,
        },
        "COHORT_MASKER",
    )
    grounding = cohort.get("grounding")
    if not isinstance(grounding, dict):
        _fail("COHORT_GROUNDING_MISSING")
    _exact(grounding.get("parent_prompt_calls_in_order"), PARENT_PROMPTS, "COHORT_PARENT_PROMPTS")
    _exact(grounding.get("child_prompt_calls_in_order"), CHILD_PROMPTS, "COHORT_CHILD_PROMPTS")
    for key, expected in {
        "batch": 1,
        "model_dtype": "float32",
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "maximum_parent_boxes_per_frame": MAX_PARENTS,
        "maximum_child_boxes_per_parent_crop": MAX_CHILDREN,
        "nms_or_deduplication": False,
    }.items():
        _exact(grounding.get(key), expected, f"COHORT_GROUNDING_{key.upper()}")
    _exact(
        cohort.get("sam_masking"),
        {
            "model_dtype": "float32",
            "multimask_output": False,
            "mask_threshold": 0.0,
            "binarize": True,
            "max_hole_area": 0.0,
            "max_sprinkle_area": 0.0,
            "apply_non_overlapping_constraints": False,
        },
        "COHORT_SAM_MASKING",
    )
    frames = cohort.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        _fail("COHORT_FRAME_COUNT_MISMATCH")
    if Counter(frame.get("role") for frame in frames) != Counter(ROLE_COUNTS):
        _fail("COHORT_ROLE_COUNTS_MISMATCH")
    if [frame.get("index") for frame in frames] != list(range(1, 9)):
        _fail("COHORT_FRAME_INDEX_MISMATCH")
    sequences = [frame.get("capture_sequence_id") for frame in frames]
    if len(set(sequences)) != 8 or any(not isinstance(value, str) or not value for value in sequences):
        _fail("COHORT_CAPTURE_SEQUENCE_MISMATCH")
    expected_buckets = {"kv2/kinect2data": 3, "xtion/sun3ddata": 3, "realsense/lg": 1, "realsense/sa": 1}
    _exact(dict(Counter(frame.get("sensor_source_bucket") for frame in frames)), expected_buckets, "COHORT_BUCKETS")
    seen_paths: set[str] = set()
    total_bytes = 0
    for frame in frames:
        files = frame.get("files")
        if not isinstance(files, dict) or set(files) != FILE_KINDS:
            _fail("COHORT_FRAME_FILES_MISMATCH", f"frame={frame.get('index')}")
        size = frame.get("source_image_size")
        if not isinstance(size, list) or len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
            _fail("COHORT_IMAGE_SIZE_INVALID", f"frame={frame.get('index')}")
        for kind in FILE_KINDS:
            receipt = files[kind]
            if not isinstance(receipt, dict) or set(receipt) != {"path", "bytes", "sha256"}:
                _fail("COHORT_FILE_RECEIPT_INVALID", f"frame={frame.get('index')}:{kind}")
            relative = _relative_posix(receipt.get("path"), f"FRAME_{frame.get('index')}_{kind.upper()}_PATH")
            canonical = relative.as_posix()
            if canonical in seen_paths:
                _fail("COHORT_FILE_PATH_DUPLICATE", canonical)
            seen_paths.add(canonical)
            if not isinstance(receipt.get("bytes"), int) or receipt["bytes"] <= 0:
                _fail("COHORT_FILE_BYTES_INVALID", canonical)
            _hex_digest(receipt.get("sha256"), f"FRAME_{frame.get('index')}_{kind.upper()}_SHA256")
            total_bytes += receipt["bytes"]
    _exact(len(seen_paths), 40, "COHORT_RECEIPT_COUNT")
    _exact(total_bytes, 1936059, "COHORT_RECEIPT_BYTES")
    return frames


def _validate_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> dict[str, Any]:
    file_path = _existing_file(path, label)
    actual_bytes = file_path.stat().st_size
    if actual_bytes != expected_bytes:
        _fail(f"{label}_BYTES_MISMATCH", f"{actual_bytes}:{expected_bytes}")
    actual_hash = _sha256(file_path)
    if actual_hash != expected_sha256:
        _fail(f"{label}_HASH_MISMATCH", f"{actual_hash}:{expected_sha256}")
    return {"path": str(file_path), "bytes": actual_bytes, "sha256": actual_hash}


def _validate_snapshot(
    argument: Path,
    expected_relative: Path,
    inventory: list[tuple[str, int, str]],
    label: str,
) -> tuple[Path, list[dict[str, Any]]]:
    snapshot = _existing_directory(argument, f"{label}_SNAPSHOT")
    expected = _existing_directory(ROOT / expected_relative, f"EXPECTED_{label}_SNAPSHOT")
    if snapshot != expected:
        _fail(f"{label}_SNAPSHOT_PATH_MISMATCH", f"{snapshot}:{expected}")
    receipts = [
        _validate_file(snapshot / relative, size, digest, f"{label}_{index:02d}")
        for index, (relative, size, digest) in enumerate(inventory)
    ]
    return snapshot, receipts


def _validate_asset_receipts() -> dict[str, Any]:
    tree = _validate_file(
        ROOT / GROUND_TREE_RELATIVE,
        1624,
        GROUND_TREE_SHA256,
        "GROUNDER_TREE_RECEIPT",
    )
    tree_json = _read_json(Path(tree["path"]), "GROUNDER_TREE_RECEIPT")
    expected_names = {path for path, _, _ in GROUND_FILES}
    if tree_json.get("format_version") != 1 or set(tree_json.get("files", {})) != expected_names:
        _fail("GROUNDER_TREE_RECEIPT_CONTENT_MISMATCH")
    sam = _validate_file(
        ROOT / SAM_RECEIPT_RELATIVE,
        5434,
        SAM_RECEIPT_SHA256,
        "SAM_ASSET_RECEIPT",
    )
    sam_json = _read_json(Path(sam["path"]), "SAM_ASSET_RECEIPT")
    if sam_json.get("schema") != "blindassist-pb15-sam2.1-hiera-small-asset-and-synthetic-smoke-receipt-v1":
        _fail("SAM_ASSET_RECEIPT_SCHEMA_MISMATCH")
    repository = sam_json.get("repository", {})
    if repository.get("repo_id") != "facebook/sam2.1-hiera-small" or repository.get("revision") != SAM_REVISION:
        _fail("SAM_ASSET_RECEIPT_REVISION_MISMATCH")
    download = sam_json.get("download", {})
    if download.get("files") != _official_files(SAM_FILES) or download.get("snapshot_file_count") != 9:
        _fail("SAM_ASSET_RECEIPT_FILES_MISMATCH")
    smoke = sam_json.get("synthetic_cuda_smoke", {})
    if smoke.get("status") != "PASS" or smoke.get("source", {}).get("pb11_pb12_pb13_pb14_pb15_cohort_images_read") != 0:
        _fail("SAM_ASSET_RECEIPT_SMOKE_MISMATCH")
    return {"grounder_tree": tree, "masker_asset": sam}


def _manifest_file(root: Path, receipt: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    relative = _relative_posix(receipt.get("path"), f"{label}_PATH")
    path = _existing_file(root / relative, label)
    try:
        path.relative_to(root)
    except ValueError:
        _fail(f"{label}_ESCAPES_ROOT", str(path))
    actual = _validate_file(path, receipt["bytes"], receipt["sha256"], label)
    return path, actual


def _distribution_version(distribution: str, label: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        _fail(f"RUNTIME_{label}_MISSING")


def _validate_runtime(torch: Any, torchvision: Any, transformers: Any, np: Any, pil: Any) -> dict[str, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        _fail("HF_HUB_OFFLINE_NOT_ONE")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        _fail("TRANSFORMERS_OFFLINE_NOT_ONE")
    python_executable = _existing_file(Path(sys.executable), "PYTHON_EXECUTABLE")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": _distribution_version("huggingface-hub", "HUGGINGFACE_HUB"),
        "safetensors": _distribution_version("safetensors", "SAFETENSORS"),
        "numpy": np.__version__,
        "pillow": pil.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    for key, expected in EXPECTED_RUNTIME.items():
        if versions[key] != expected:
            _fail("RUNTIME_VERSION_MISMATCH", f"{key}:{versions[key]}:{expected}")
    executable_hash = _sha256(python_executable)
    if executable_hash != PYTHON_EXE_SHA256:
        _fail("PYTHON_EXECUTABLE_HASH_MISMATCH", f"{executable_hash}:{PYTHON_EXE_SHA256}")
    expected_python = _existing_file(Path(PYTHON_EXE_FROZEN), "FROZEN_PYTHON_EXECUTABLE")
    if python_executable != expected_python:
        _fail("PYTHON_EXECUTABLE_PATH_MISMATCH", f"{python_executable}:{expected_python}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        _fail("CUDA_ZERO_UNAVAILABLE")
    torch.cuda.set_device(0)
    probe = torch.empty((1,), device="cuda:0")
    if probe.device.type != "cuda" or probe.device.index != 0:
        _fail("CUDA_PROBE_DEVICE_MISMATCH", str(probe.device))
    del probe
    properties = torch.cuda.get_device_properties(0)
    return {
        **versions,
        "python_executable": str(python_executable),
        "python_executable_sha256": executable_hash,
        "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
        "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        "actual_device_index": 0,
        "actual_device_name": properties.name,
        "actual_device_capability": list(torch.cuda.get_device_capability(0)),
        "device_total_memory_bytes": properties.total_memory,
    }


def _clamp_box(box: Sequence[Any], image_size: tuple[int, int]) -> list[float] | None:
    if len(box) != 4:
        _fail("BOX_COORDINATE_COUNT_MISMATCH", repr(box))
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        return None
    width, height = image_size
    clamped = [
        min(float(width), max(0.0, values[0])),
        min(float(height), max(0.0, values[1])),
        min(float(width), max(0.0, values[2])),
        min(float(height), max(0.0, values[3])),
    ]
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return clamped


def _ground_single(
    processor: Any,
    model: Any,
    image: Any,
    prompt: dict[str, Any],
    image_size: tuple[int, int],
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    width, height = image_size
    started = time.perf_counter()
    inputs = processor(images=image, text=[prompt["text"]], return_tensors="pt")
    required_inputs = {"input_ids", "token_type_ids", "attention_mask", "pixel_mask", "pixel_values"}
    if set(inputs) != required_inputs:
        _fail("GROUNDER_INPUT_FIELDS_MISMATCH", repr(sorted(inputs)))
    inputs = inputs.to("cuda:0")
    with torch.inference_mode():
        outputs = model(**inputs)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    if tuple(outputs.logits.shape[:2]) != (1, 900) or tuple(outputs.pred_boxes.shape) != (1, 900, 4):
        _fail("GROUNDER_RAW_OUTPUT_SHAPE_MISMATCH", f"{tuple(outputs.logits.shape)}:{tuple(outputs.pred_boxes.shape)}")
    processed = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(height, width)],
    )
    if not isinstance(processed, list) or len(processed) != 1:
        _fail("GROUNDER_POSTPROCESS_BATCH_MISMATCH")
    result = processed[0]
    if set(result) != {"scores", "boxes", "text_labels", "labels"}:
        _fail("GROUNDER_POSTPROCESS_FIELDS_MISMATCH", repr(sorted(result)))
    scores = result["scores"].detach().cpu()
    boxes = result["boxes"].detach().cpu()
    if scores.ndim != 1 or boxes.ndim != 2 or boxes.shape != (scores.shape[0], 4):
        _fail("GROUNDER_POSTPROCESS_SHAPE_MISMATCH", f"{tuple(scores.shape)}:{tuple(boxes.shape)}")
    instances: list[dict[str, Any]] = []
    discarded = 0
    for postprocess_index, (score_tensor, box_tensor) in enumerate(zip(scores, boxes)):
        score = float(score_tensor.item())
        if not math.isfinite(score) or not score > BOX_THRESHOLD:
            _fail("GROUNDER_SCORE_CONTRACT_MISMATCH", f"{score}:{BOX_THRESHOLD}")
        box = _clamp_box(box_tensor.tolist(), image_size)
        if box is None:
            discarded += 1
            continue
        instance = {
            "class_id": prompt["class_id"],
            "prompt_call_index": prompt["call_index"],
            "prompt_text": prompt["text"],
            "postprocess_index": postprocess_index,
            "score": score,
            "box_xyxy": box,
            "class_identity_source": "SINGLE_PROMPT_CALL_IDENTITY_DECODED_LABELS_IGNORED",
        }
        if "subtype" in prompt:
            instance["subtype"] = prompt["subtype"]
        instances.append(instance)
    receipt = {
        "prompt_call_index": prompt["call_index"],
        "class_id": prompt["class_id"],
        "subtype": prompt.get("subtype"),
        "prompt_text": prompt["text"],
        "prompt_serialization": f"{prompt['text']}.",
        "decoded_labels_ignored": True,
        "input_pixel_values_shape": list(inputs.pixel_values.shape),
        "input_ids_shape": list(inputs.input_ids.shape),
        "raw_logits_shape": list(outputs.logits.shape),
        "raw_boxes_shape": list(outputs.pred_boxes.shape),
        "postprocessed_box_count": int(scores.shape[0]),
        "discarded_nonfinite_or_empty_box_count": discarded,
        "valid_box_count": len(instances),
        "inference_and_postprocess_seconds": elapsed,
    }
    return instances, receipt


def _ground_all(
    processor: Any,
    model: Any,
    image: Any,
    prompts: list[dict[str, Any]],
    image_size: tuple[int, int],
    cap: int,
    lane: int,
    torch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    instances: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for prompt in prompts:
        detected, receipt = _ground_single(processor, model, image, prompt, image_size, torch)
        instances.extend(detected)
        calls.append(receipt)
    if lane == 1:
        instances.sort(
            key=lambda item: (
                -item["score"],
                PARENT_PRIORITY[item["class_id"]],
                *item["box_xyxy"],
                item["prompt_call_index"],
                item["postprocess_index"],
            )
        )
    elif lane == 2:
        instances.sort(
            key=lambda item: (
                -item["score"],
                item["prompt_call_index"],
                *item["box_xyxy"],
                item["postprocess_index"],
            )
        )
    else:
        _fail("GROUNDING_LANE_INVALID", str(lane))
    retained = [dict(instance) for instance in instances[:cap]]
    return instances, retained, calls


def _mask_receipt(mask: Any, np: Any) -> dict[str, Any]:
    binary = np.ascontiguousarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not bool(np.all((binary == 0) | (binary == 1))):
        _fail("MASK_NOT_2D_BINARY")
    return {
        "encoding": "C_ORDER_UINT8_BINARY_RECEIPT_ONLY",
        "shape_hw": [int(binary.shape[0]), int(binary.shape[1])],
        "true_pixels": int(binary.sum()),
        "binary_uint8_sha256": hashlib.sha256(binary.tobytes(order="C")).hexdigest(),
    }


def _sam_masks(
    processor: Any,
    model: Any,
    image: Any,
    boxes: list[list[float]],
    image_size: tuple[int, int],
    torch: Any,
    np: Any,
) -> tuple[list[Any], dict[str, Any] | None]:
    if not boxes:
        return [], None
    width, height = image_size
    started = time.perf_counter()
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
    if set(inputs) != {"pixel_values", "original_sizes", "input_boxes"}:
        _fail("SAM_INPUT_FIELDS_MISMATCH", repr(sorted(inputs)))
    original_sizes = inputs["original_sizes"].detach().cpu()
    if tuple(inputs.input_boxes.shape) != (1, len(boxes), 4):
        _fail("SAM_INPUT_BOX_SHAPE_MISMATCH", str(tuple(inputs.input_boxes.shape)))
    with torch.inference_mode():
        outputs = model(**inputs.to("cuda:0"), multimask_output=False)
    torch.cuda.synchronize(0)
    native = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        original_sizes,
        mask_threshold=0.0,
        binarize=True,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        apply_non_overlapping_constraints=False,
    )
    elapsed = time.perf_counter() - started
    if not isinstance(native, list) or len(native) != 1:
        _fail("SAM_POSTPROCESS_BATCH_MISMATCH")
    batch = native[0]
    expected_shape = (len(boxes), 1, height, width)
    if tuple(batch.shape) != expected_shape or batch.dtype != torch.bool:
        _fail("SAM_NATIVE_MASK_CONTRACT_MISMATCH", f"{tuple(batch.shape)}:{batch.dtype}:{expected_shape}")
    masks = [np.ascontiguousarray(batch[index, 0].numpy(), dtype=np.bool_) for index in range(len(boxes))]
    receipt = {
        "input_box_count": len(boxes),
        "input_boxes_shape": list(inputs.input_boxes.shape),
        "raw_pred_masks_shape": list(outputs.pred_masks.shape),
        "raw_iou_scores_shape": list(outputs.iou_scores.shape),
        "native_masks_shape": list(batch.shape),
        "native_masks_dtype": str(batch.dtype),
        "multimask_output": False,
        "iou_scores_ignored": True,
        "postprocess": {
            "mask_threshold": 0.0,
            "binarize": True,
            "max_hole_area": 0.0,
            "max_sprinkle_area": 0.0,
            "apply_non_overlapping_constraints": False,
        },
        "inference_and_postprocess_seconds": elapsed,
    }
    return masks, receipt


def _rank_and_mask_parents(
    retained: list[dict[str, Any]],
    masks: list[Any],
    image_size: tuple[int, int],
    np: Any,
) -> list[dict[str, Any]]:
    if len(retained) != len(masks):
        _fail("PARENT_BOX_MASK_COUNT_MISMATCH")
    width, height = image_size
    parents: list[dict[str, Any]] = []
    for stable_index, (detection, mask) in enumerate(zip(retained, masks)):
        if detection["class_id"] not in PARENT_IDS:
            _fail("PARENT_CLASS_INVALID", detection["class_id"])
        if tuple(mask.shape) != (height, width) or mask.dtype != np.bool_:
            _fail("PARENT_NATIVE_MASK_INVALID", f"{tuple(mask.shape)}:{mask.dtype}")
        parents.append(
            {
                **detection,
                "stable_parent_index": stable_index,
                "parent_instance_id": f"parent:{stable_index:03d}",
                "mask": mask,
                "mask_area_pixels": int(mask.sum()),
            }
        )
    return parents


def _crop_box(box: Sequence[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, min(width, math.floor(float(box[0]))))
    top = max(0, min(height, math.floor(float(box[1]))))
    right = max(0, min(width, math.ceil(float(box[2]))))
    bottom = max(0, min(height, math.ceil(float(box[3]))))
    if right <= left or bottom <= top:
        _fail("PARENT_CROP_EMPTY", f"{box}:{image_size}")
    return left, top, right, bottom


def _public_detection(instance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in instance.items() if key not in {"mask", "true_xs", "true_ys"}}


def _serial_masked(instance: dict[str, Any], np: Any) -> dict[str, Any]:
    serial = _public_detection(instance)
    serial["mask"] = _mask_receipt(instance["mask"], np)
    return serial


def _assign_topology(
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    image_size: tuple[int, int],
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
                "child_subtype": child["subtype"],
                "child_lane": 2,
                "source_parent_crop_instance_id": child["source_parent_crop_instance_id"],
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
        role: sum(row["role"] == role and row["grounded_sam_parent_rescaled_pixel_part_topology"] for row in rows)
        for role in ROLE_COUNTS
    }
    positives = authorized_by_role[POSITIVE_ROLE]
    control_authorized = authorized_by_role[FURNITURE_ROLE] + authorized_by_role[OOD_ROLE]
    recall = positives / ROLE_COUNTS[POSITIVE_ROLE]
    false_positive_rate = control_authorized / (ROLE_COUNTS[FURNITURE_ROLE] + ROLE_COUNTS[OOD_ROLE])
    true_negative_rate = 1.0 - false_positive_rate
    balanced_accuracy = (recall + true_negative_rate) / 2.0
    gate_met = positives == 4 and control_authorized == 0 and balanced_accuracy == 1.0
    parent_counts = Counter()
    child_class_counts = Counter()
    child_subtype_counts = Counter()
    valid_parent_counts = Counter()
    valid_child_class_counts = Counter()
    valid_child_subtype_counts = Counter()
    grounder_calls = 0
    masker_calls = 0
    grounder_seconds = 0.0
    masker_seconds = 0.0
    retained_parents = 0
    retained_children = 0
    for row in rows:
        lane_1 = row.get("lane_1_full_frame", {})
        parent_counts.update(lane_1.get("retained_counts_by_class_id", {}))
        valid_parent_counts.update(instance.get("class_id") for instance in lane_1.get("all_valid_grounded_boxes", []))
        retained_parents += lane_1.get("retained_parent_count", 0)
        for call in lane_1.get("grounder_calls", []):
            grounder_calls += 1
            grounder_seconds += call.get("inference_and_postprocess_seconds", 0.0)
        if lane_1.get("masker_call") is not None:
            masker_calls += 1
            masker_seconds += lane_1["masker_call"].get("inference_and_postprocess_seconds", 0.0)
        for crop in row.get("lane_2_parent_crops", []):
            child_class_counts.update(crop.get("retained_counts_by_class_id", {}))
            child_subtype_counts.update(crop.get("retained_counts_by_subtype", {}))
            valid_child_class_counts.update(
                instance.get("class_id") for instance in crop.get("all_valid_grounded_boxes", [])
            )
            valid_child_subtype_counts.update(
                instance.get("subtype") for instance in crop.get("all_valid_grounded_boxes", [])
            )
            retained_children += crop.get("retained_child_count", 0)
            for call in crop.get("grounder_calls", []):
                grounder_calls += 1
                grounder_seconds += call.get("inference_and_postprocess_seconds", 0.0)
            if crop.get("masker_call") is not None:
                masker_calls += 1
                masker_seconds += crop["masker_call"].get("inference_and_postprocess_seconds", 0.0)
    metrics = {
        "frames": len(rows),
        "grounder_calls": grounder_calls,
        "masker_calls": masker_calls,
        "total_model_calls": grounder_calls + masker_calls,
        "grounder_inference_and_postprocess_seconds": grounder_seconds,
        "masker_inference_and_postprocess_seconds": masker_seconds,
        "retained_parent_boxes": retained_parents,
        "retained_child_boxes": retained_children,
        "lane_1_valid_grounded_boxes_by_class_id": {
            class_id: valid_parent_counts[class_id] for class_id in [item["class_id"] for item in PARENT_PROMPTS]
        },
        "lane_1_retained_instances_by_class_id": {
            class_id: parent_counts[class_id] for class_id in [item["class_id"] for item in PARENT_PROMPTS]
        },
        "lane_2_retained_instances_by_class_id": {
            class_id: child_class_counts[class_id] for class_id in ["operation_part", "hinge"]
        },
        "lane_2_valid_grounded_boxes_by_class_id": {
            class_id: valid_child_class_counts[class_id] for class_id in ["operation_part", "hinge"]
        },
        "lane_2_valid_grounded_boxes_by_subtype": {
            subtype: valid_child_subtype_counts[subtype] for subtype in [item["subtype"] for item in CHILD_PROMPTS]
        },
        "lane_2_retained_instances_by_subtype": {
            subtype: child_subtype_counts[subtype] for subtype in [item["subtype"] for item in CHILD_PROMPTS]
        },
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
            "L10_PB15_GROUNDED_SAM_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_PB15_GROUNDED_SAM_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET"
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


def _validate_loaded_processors_and_models(
    ground_processor: Any,
    ground_model: Any,
    sam_processor: Any,
    sam_model: Any,
    torch: Any,
) -> dict[str, Any]:
    if type(ground_processor).__name__ != "GroundingDinoProcessor":
        _fail("LOADED_GROUNDER_PROCESSOR_TYPE_MISMATCH", type(ground_processor).__name__)
    if type(ground_model).__name__ != "GroundingDinoForObjectDetection":
        _fail("LOADED_GROUNDER_MODEL_TYPE_MISMATCH", type(ground_model).__name__)
    image_processor = ground_processor.image_processor
    for key, expected in {
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "do_pad": True,
    }.items():
        _exact(getattr(image_processor, key, None), expected, f"LOADED_GROUNDER_PROCESSOR_{key.upper()}")
    _exact(image_processor.size, {"shortest_edge": 800, "longest_edge": 1333}, "LOADED_GROUNDER_SIZE")
    _exact(list(image_processor.image_mean), [0.485, 0.456, 0.406], "LOADED_GROUNDER_MEAN")
    _exact(list(image_processor.image_std), [0.229, 0.224, 0.225], "LOADED_GROUNDER_STD")
    _exact(getattr(ground_model.config, "num_queries", None), 900, "LOADED_GROUNDER_NUM_QUERIES")
    _exact(getattr(ground_model.config, "max_text_len", None), 256, "LOADED_GROUNDER_MAX_TEXT_LEN")
    if type(sam_processor).__name__ != "Sam2Processor":
        _fail("LOADED_SAM_PROCESSOR_TYPE_MISMATCH", type(sam_processor).__name__)
    if type(sam_model).__name__ != "Sam2Model":
        _fail("LOADED_SAM_MODEL_TYPE_MISMATCH", type(sam_model).__name__)
    ground_parameter = next(ground_model.parameters())
    sam_parameter = next(sam_model.parameters())
    for label, parameter in (("GROUNDER", ground_parameter), ("SAM", sam_parameter)):
        if parameter.device.type != "cuda" or parameter.device.index != 0:
            _fail(f"LOADED_{label}_DEVICE_MISMATCH", str(parameter.device))
        if parameter.dtype != torch.float32:
            _fail(f"LOADED_{label}_DTYPE_MISMATCH", str(parameter.dtype))
    return {
        "grounder_processor_type": type(ground_processor).__name__,
        "grounder_model_type": type(ground_model).__name__,
        "grounder_parameter_device": str(ground_parameter.device),
        "grounder_parameter_dtype": str(ground_parameter.dtype),
        "masker_processor_type": type(sam_processor).__name__,
        "masker_model_type": type(sam_model).__name__,
        "masker_parameter_device": str(sam_parameter.device),
        "masker_parameter_dtype": str(sam_parameter.dtype),
    }


def _run(
    protocol_argument: Path,
    cohort_argument: Path,
    extracted_root_argument: Path,
    grounder_snapshot_argument: Path,
    masker_snapshot_argument: Path,
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
    expected_root = _existing_directory(
        ROOT / _relative_posix(cohort["source"]["extracted_root"], "COHORT_EXTRACTED_ROOT"),
        "EXPECTED_EXTRACTED_ROOT",
    )
    if extracted_root != expected_root:
        _fail("EXTRACTED_ROOT_PATH_MISMATCH", f"{extracted_root}:{expected_root}")

    grounder_snapshot, grounder_files = _validate_snapshot(
        grounder_snapshot_argument,
        GROUND_RELATIVE,
        GROUND_FILES,
        "GROUNDER",
    )
    masker_snapshot, masker_files = _validate_snapshot(
        masker_snapshot_argument,
        SAM_RELATIVE,
        SAM_FILES,
        "MASKER",
    )
    asset_receipts = _validate_asset_receipts()

    prepared: list[dict[str, Any]] = []
    actual_member_count = 0
    actual_member_bytes = 0
    for frame in frames:
        paths: dict[str, Path] = {}
        receipts: dict[str, Any] = {}
        for kind in sorted(FILE_KINDS):
            path, receipt = _manifest_file(
                extracted_root,
                frame["files"][kind],
                f"FRAME_{frame['index']}_{kind.upper()}",
            )
            paths[kind] = path
            receipts[kind] = receipt
            actual_member_count += 1
            actual_member_bytes += receipt["bytes"]
        prepared.append({"frame": frame, "paths": paths, "receipts": receipts})
    _exact(actual_member_count, 40, "ACTUAL_MEMBER_COUNT")
    _exact(actual_member_bytes, 1936059, "ACTUAL_MEMBER_BYTES")

    try:
        import numpy as np
        import PIL
        from PIL import Image
        import torch
        import torchvision
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    except ImportError as exc:
        _fail("INFERENCE_RUNTIME_IMPORT_FAILED", str(exc))
    runtime_receipt = _validate_runtime(torch, torchvision, transformers, np, PIL)

    for item in prepared:
        frame = item["frame"]
        try:
            with Image.open(item["paths"]["rgb"]) as opened:
                opened.load()
                size = [int(opened.width), int(opened.height)]
                mode = opened.mode
        except Exception as exc:
            _fail("RGB_DECODE_FAILED", f"frame={frame['index']}:{exc}")
        if size != frame["source_image_size"] or mode not in {"RGB", "L", "RGBA"}:
            _fail("RGB_IDENTITY_MISMATCH", f"frame={frame['index']}:{size}:{mode}")

    ground_processor = AutoProcessor.from_pretrained(grounder_snapshot, local_files_only=True)
    ground_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounder_snapshot,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval().to("cuda:0")
    sam_processor = Sam2Processor.from_pretrained(masker_snapshot, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        masker_snapshot,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval().to("cuda:0")
    loaded_receipt = _validate_loaded_processors_and_models(
        ground_processor,
        ground_model,
        sam_processor,
        sam_model,
        torch,
    )
    if _sha256(grounder_snapshot / "model.safetensors") != GROUND_FILES[3][2]:
        _fail("GROUNDER_CHANGED_DURING_LOAD")
    if _sha256(masker_snapshot / "model.safetensors") != SAM_FILES[3][2]:
        _fail("MASKER_CHANGED_DURING_LOAD")

    torch.cuda.reset_peak_memory_stats(0)
    rows: list[dict[str, Any]] = []
    for item in prepared:
        frame = item["frame"]
        width, height = frame["source_image_size"]
        with Image.open(item["paths"]["rgb"]) as opened:
            image = opened.convert("RGB")
        all_parent_boxes, retained_parent_boxes, lane_1_calls = _ground_all(
            ground_processor,
            ground_model,
            image,
            PARENT_PROMPTS,
            (width, height),
            MAX_PARENTS,
            1,
            torch,
        )
        parent_masks, lane_1_masker = _sam_masks(
            sam_processor,
            sam_model,
            image,
            [instance["box_xyxy"] for instance in retained_parent_boxes],
            (width, height),
            torch,
            np,
        )
        parents = _rank_and_mask_parents(retained_parent_boxes, parent_masks, (width, height), np)

        lane_2_crops: list[dict[str, Any]] = []
        lane_2_children: list[dict[str, Any]] = []
        for parent in parents:
            left, top, right, bottom = _crop_box(parent["box_xyxy"], (width, height))
            crop = image.crop((left, top, right, bottom))
            crop_width = right - left
            crop_height = bottom - top
            if crop.size != (crop_width, crop_height) or crop.mode != "RGB":
                _fail("PARENT_CROP_IDENTITY_MISMATCH", parent["parent_instance_id"])
            all_child_boxes, retained_child_boxes, child_calls = _ground_all(
                ground_processor,
                ground_model,
                crop,
                CHILD_PROMPTS,
                (crop_width, crop_height),
                MAX_CHILDREN,
                2,
                torch,
            )
            child_masks, child_masker = _sam_masks(
                sam_processor,
                sam_model,
                crop,
                [instance["box_xyxy"] for instance in retained_child_boxes],
                (crop_width, crop_height),
                torch,
                np,
            )
            if len(retained_child_boxes) != len(child_masks):
                _fail("CHILD_BOX_MASK_COUNT_MISMATCH", parent["parent_instance_id"])
            serial_children: list[dict[str, Any]] = []
            for child_index, (detection, crop_mask) in enumerate(zip(retained_child_boxes, child_masks)):
                if detection["class_id"] not in CHILD_IDS or tuple(crop_mask.shape) != (crop_height, crop_width):
                    _fail("CHILD_NATIVE_MASK_INVALID", parent["parent_instance_id"])
                mapped = np.zeros((height, width), dtype=np.bool_)
                mapped[top:bottom, left:right] = crop_mask
                ys, xs = np.nonzero(mapped)
                child_id = f"{parent['parent_instance_id']}:child:{child_index:03d}"
                source_box = [
                    detection["box_xyxy"][0] + left,
                    detection["box_xyxy"][1] + top,
                    detection["box_xyxy"][2] + left,
                    detection["box_xyxy"][3] + top,
                ]
                child = {
                    **detection,
                    "child_instance_id": child_id,
                    "source_parent_crop_instance_id": parent["parent_instance_id"],
                    "crop_box_xyxy_half_open": [left, top, right, bottom],
                    "crop_native_box_xyxy": detection["box_xyxy"],
                    "source_box_xyxy": source_box,
                    "mask": mapped,
                    "mask_area_pixels": int(mapped.sum()),
                    "true_xs": xs,
                    "true_ys": ys,
                }
                lane_2_children.append(child)
                serial = _public_detection(child)
                serial["native_crop_mask"] = _mask_receipt(crop_mask, np)
                serial["mapped_source_mask"] = _mask_receipt(mapped, np)
                serial_children.append(serial)
            retained_class_counts = Counter(instance["class_id"] for instance in retained_child_boxes)
            retained_subtype_counts = Counter(instance["subtype"] for instance in retained_child_boxes)
            lane_2_crops.append(
                {
                    "source_parent_instance_id": parent["parent_instance_id"],
                    "source_parent_class_id": parent["class_id"],
                    "crop_box_xyxy_half_open": [left, top, right, bottom],
                    "crop_size_wh": [crop_width, crop_height],
                    "grounder_calls": child_calls,
                    "all_valid_grounded_boxes": [_public_detection(instance) for instance in all_child_boxes],
                    "valid_grounded_box_count": len(all_child_boxes),
                    "retained_child_count": len(retained_child_boxes),
                    "retained_counts_by_class_id": {
                        class_id: retained_class_counts[class_id] for class_id in ["operation_part", "hinge"]
                    },
                    "retained_counts_by_subtype": {
                        subtype: retained_subtype_counts[subtype] for subtype in [item["subtype"] for item in CHILD_PROMPTS]
                    },
                    "masker_call": child_masker,
                    "retained_child_instances": serial_children,
                }
            )
        assignments, authorized = _assign_topology(parents, lane_2_children, (width, height))
        parent_class_counts = Counter(parent["class_id"] for parent in parents)
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
                    "grounder_calls": lane_1_calls,
                    "all_valid_grounded_boxes": [_public_detection(instance) for instance in all_parent_boxes],
                    "valid_grounded_box_count": len(all_parent_boxes),
                    "retained_parent_count": len(parents),
                    "retained_counts_by_class_id": {
                        class_id: parent_class_counts[class_id] for class_id in [item["class_id"] for item in PARENT_PROMPTS]
                    },
                    "masker_call": lane_1_masker,
                    "retained_parent_instances": [_serial_masked(parent, np) for parent in parents],
                    "retained_parent_instance_ids_in_crop_order": [parent["parent_instance_id"] for parent in parents],
                    "lane_1_children_authorize": False,
                },
                "lane_2_parent_crops": lane_2_crops,
                "child_parent_assignments": assignments,
                "grounded_sam_parent_rescaled_pixel_part_topology": authorized,
            }
        )
        del image, parents, lane_2_children
    torch.cuda.synchronize(0)
    metrics, gate = _aggregate(rows)
    evaluator_path = Path(__file__).resolve()
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "L10-PB15 Grounded-SAM Multiscale Part Topology",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "inputs": {
            "protocol": {"path": str(protocol_path), "schema": PROTOCOL_SCHEMA, "sha256": protocol_hash},
            "cohort": {
                "path": str(cohort_path),
                "schema": COHORT_SCHEMA,
                "sha256": cohort_hash,
                "status": cohort["status"],
            },
            "extracted_root": str(extracted_root),
            "grounder_snapshot": {"path": str(grounder_snapshot), "files": grounder_files},
            "masker_snapshot": {"path": str(masker_snapshot), "files": masker_files},
            "asset_receipts": asset_receipts,
        },
        "evaluator": {"path": str(evaluator_path), "sha256": _sha256(evaluator_path)},
        "runtime": {
            **runtime_receipt,
            **loaded_receipt,
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(0),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "grounding": {
            "parent_prompt_calls_in_order": PARENT_PROMPTS,
            "child_prompt_calls_in_order": CHILD_PROMPTS,
            "processor_text_argument": "SINGLE_ELEMENT_CANDIDATE_LABEL_LIST",
            "decoded_labels_ignored": True,
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
            "maximum_parent_boxes_per_frame": MAX_PARENTS,
            "maximum_child_boxes_per_parent_crop": MAX_CHILDREN,
            "nms_or_deduplication": False,
        },
        "sam_masking": {
            "multimask_output": False,
            "mask_threshold": 0.0,
            "binarize": True,
            "max_hole_area": 0.0,
            "max_sprinkle_area": 0.0,
            "apply_non_overlapping_constraints": False,
            "iou_scores_ignored": True,
        },
        "topology": {
            "representative": "ROUND_HALF_UP_MEAN_INTEGER_TRUE_MAPPED_CHILD_MASK_PIXEL_INDICES",
            "assignment_order": "MIN_PARENT_MASK_AREA_THEN_COMPETITOR_CONSERVATIVE_CLASS_PRIORITY_THEN_STABLE_PARENT_INDEX",
            "authorization": "AT_LEAST_ONE_LANE2_CHILD_ASSIGNED_TO_ARCHITECTURAL_LEAF",
            "lane_1_children_authorize": False,
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
    architectural_points = {(x, y) for y in range(10) for x in range(10)}
    cabinet_points = {(x, y) for y in range(3, 7) for x in range(3, 7)}
    parents = [
        {
            "parent_instance_id": "parent:000",
            "class_id": "architectural_leaf",
            "stable_parent_index": 0,
            "mask_area_pixels": len(architectural_points),
            "mask": _synthetic_mask(*size, architectural_points),
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
        "subtype": "handle",
        "source_parent_crop_instance_id": "parent:000",
        "mask_area_pixels": 4,
        "true_xs": [4, 5, 4, 5],
        "true_ys": [4, 4, 5, 5],
    }
    assignments, authorized = _assign_topology(parents, [child], size)
    if assignments[0]["representative_pixel_xy"] != [5, 5]:
        raise AssertionError("round-half-up representative mismatch")
    if assignments[0]["assigned_parent_class_id"] != "cabinet_door" or authorized:
        raise AssertionError("smallest competitor parent did not absorb the child")
    competitor_assignment = assignments[0]["assigned_parent_class_id"]
    assignments, authorized = _assign_topology(parents[:1], [child], size)
    if assignments[0]["assigned_parent_class_id"] != "architectural_leaf" or not authorized:
        raise AssertionError("lane-2 architectural parent authorization failed")
    if _crop_box([1.1, 2.9, 8.01, 9.0], size) != (1, 2, 9, 9):
        raise AssertionError("frozen floor/ceil crop mapping failed")
    if _clamp_box([-2.0, 1.0, 12.0, 9.0], size) != [0.0, 1.0, 10.0, 9.0]:
        raise AssertionError("frozen box clamp failed")
    if _clamp_box([5.0, 5.0, 5.0, 8.0], size) is not None:
        raise AssertionError("empty box was retained")
    rows = []
    for index in range(8):
        role = POSITIVE_ROLE if index < 4 else FURNITURE_ROLE if index < 6 else OOD_ROLE
        rows.append(
            {
                "role": role,
                "grounded_sam_parent_rescaled_pixel_part_topology": index < 4,
                "lane_1_full_frame": {
                    "retained_counts_by_class_id": {},
                    "retained_parent_count": 0,
                    "grounder_calls": [],
                    "masker_call": None,
                },
                "lane_2_parent_crops": [],
            }
        )
    metrics, gate = _aggregate(rows)
    if metrics["balanced_accuracy"] != 1.0 or not gate["development_gate_met"]:
        raise AssertionError("perfect frozen gate did not pass")
    rows[0]["grounded_sam_parent_rescaled_pixel_part_topology"] = False
    _, failed = _aggregate(rows)
    if failed["development_gate_met"]:
        raise AssertionError("imperfect frozen gate passed")
    rows[0]["grounded_sam_parent_rescaled_pixel_part_topology"] = True
    rows[4]["grounded_sam_parent_rescaled_pixel_part_topology"] = True
    _, false_positive = _aggregate(rows)
    if false_positive["development_gate_met"]:
        raise AssertionError("control false positive passed the frozen gate")
    return {
        "status": "PASS",
        "model_calls": 0,
        "cohort_files_read": 0,
        "cohort_images_read": 0,
        "candidate_label_argument": ["frozen text"],
        "competitor_assignment": competitor_assignment,
        "architectural_assignment": assignments[0]["assigned_parent_class_id"],
        "perfect_gate_decision": gate["decision"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen PB15 Grounded-SAM multiscale topology evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run zero-model synthetic topology and gate checks only.")
    run = subparsers.add_parser("run", help="Run the one frozen eight-image CUDA evaluation.")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--cohort", "--queue", dest="cohort", type=Path, required=True)
    run.add_argument("--extracted-root", type=Path, required=True)
    run.add_argument("--grounder-snapshot", type=Path, default=ROOT / GROUND_RELATIVE)
    run.add_argument("--masker-snapshot", type=Path, default=ROOT / SAM_RELATIVE)
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
            args.grounder_snapshot,
            args.masker_snapshot,
            args.output,
        )
    except ContractError as exc:
        print(f"PB15_CONTRACT_ERROR:{exc}", file=sys.stderr)
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
