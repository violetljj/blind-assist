#!/usr/bin/env python3
"""Run the frozen L10-PB13 Florence pixel part-parent Development gate."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence


PROTOCOL_SCHEMA = "l10-named-poi-florence-pixel-part-topology-protocol-v1"
COHORT_SCHEMA = "l10-named-poi-florence-pixel-part-topology-cohort-v1"
RESULT_SCHEMA = "l10-named-poi-florence-pixel-part-topology-development-result-v1"
PROTOCOL_SHA256 = "fb6353f2c285e2398ede9bd459af7a482ec89401f3a5a239ae30e972b29b6db2"
COHORT_SHA256 = "6e4de32805a2e8d166994bd0adbd9ad127410c4f85961220f4ad01045a0e0900"
VERIFICATION_SHA256 = "de33b1bcd06cc43a37c6619eefdb47e94793bdd390fd98135bd95f94680e40ea"
MODEL_REVISION = "7c0fd5647b7d9dc0670820f053197167946cc26f"
PYTHON_EXE_SHA256 = "21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082"
TASK_TOKEN = "<REFERRING_EXPRESSION_SEGMENTATION>"

POSITIVE_ROLE = "ARCHITECTURAL_DOOR_WITH_VISIBLE_OPERATION_PART"
FURNITURE_ROLE = "HANDLED_FURNITURE_DOOR_NEGATIVE"
OOD_ROLE = "LARGE_DOORLESS_OPENING_OOD"
ROLE_COUNTS = {POSITIVE_ROLE: 4, FURNITURE_ROLE: 2, OOD_ROLE: 2}
FILE_KINDS = {"rgb", "polygon", "scene_metadata"}

EXPRESSIONS = [
    {"expression_id": "architectural_leaf", "text": "architectural room door leaf", "role": "PARENT"},
    {"expression_id": "architectural_frame", "text": "architectural room door frame", "role": "DIAGNOSTIC_ONLY"},
    {
        "expression_id": "operation_part",
        "text": "architectural room door handle, knob, or push bar",
        "role": "CHILD",
    },
    {"expression_id": "hinge", "text": "architectural room door hinge", "role": "CHILD"},
    {"expression_id": "cabinet_door", "text": "cabinet door", "role": "COMPETING_PARENT"},
    {
        "expression_id": "wardrobe_closet_door",
        "text": "wardrobe or closet door",
        "role": "COMPETING_PARENT",
    },
]
GENERATION = {
    "max_new_tokens": 1024,
    "num_beams": 3,
    "do_sample": False,
    "early_stopping": False,
    "use_cache": True,
}
PARENT_IDS = {"architectural_leaf", "cabinet_door", "wardrobe_closet_door"}
CHILD_IDS = {"operation_part", "hinge"}
PARENT_PRIORITY = {"cabinet_door": 0, "wardrobe_closet_door": 1, "architectural_leaf": 2}

MODEL_FILES = {
    "LICENSE": (1141, "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"),
    "config.json": (2445, "fa081841369aa9c6e42faf5c52368d673b561e2c5f8fa03d1256e7408cb4130e"),
    "configuration_florence2.py": (
        15125,
        "653bafddc9651eaff1583a16db4a2bb27d33ec7d541dfab7201aaa4ecaa1cfbf",
    ),
    "generation_config.json": (51, "30e9865458ecc8ee931eeeb43f44f1d169c5ab95be39e0072142a7a6b8f31990"),
    "model.safetensors": (1540980506, "8b4e610c952eef90a836c56cda0f398a672a3a6ca7b4d96b0e09a86dee42e2c3"),
    "modeling_florence2.py": (127856, "9fd1b4659a6623a075750bf18d8df3dd906ab186f4c4a110411d6c380a9a5617"),
    "preprocessor_config.json": (806, "2f5921bbc53c7cc04251e1027b45b1cec726276be6db23d1bb40641bfbe2cf29"),
    "processing_florence2.py": (46372, "4bd7158536cbf1c7891fc8efd94437d79fd09f07f539c7398fab8a885d7d8bca"),
    "tokenizer.json": (1355863, "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e"),
    "tokenizer_config.json": (34, "79ffcf43af8ebda99d165f61d243180da2e2639952e41e71e11611c18770489c"),
    "vocab.json": (1099884, "394fdc63c71aabe0a9b97117f5d62fb5fcc4d59b2b3ea929a3929e6a53217b3c"),
}
EXPECTED_RUNTIME = {
    "python": "3.11.9",
    "torch": "2.11.0+cu128",
    "transformers": "4.57.1",
    "tokenizers": "0.22.2",
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
    path = value.resolve(strict=True)
    if not path.is_file():
        _fail(f"{label}_NOT_FILE", str(path))
    return path


def _existing_directory(value: Path, label: str) -> Path:
    path = value.resolve(strict=True)
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


def _validate_protocol(protocol: dict[str, Any], actual_hash: str) -> None:
    if actual_hash != PROTOCOL_SHA256:
        _fail("PROTOCOL_HASH_MISMATCH", f"{actual_hash}:{PROTOCOL_SHA256}")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "FROZEN_BEFORE_COHORT_MODEL_OUTPUT":
        _fail("PROTOCOL_IDENTITY_MISMATCH")
    source = protocol.get("information_source")
    runtime = protocol.get("runtime")
    inference = protocol.get("inference")
    if not isinstance(source, dict) or not isinstance(runtime, dict) or not isinstance(inference, dict):
        _fail("PROTOCOL_SECTION_MISSING")
    if source.get("revision") != MODEL_REVISION or source.get("verification_manifest_sha256") != VERIFICATION_SHA256:
        _fail("PROTOCOL_MODEL_MISMATCH")
    actual_expressions = inference.get("expressions_in_order")
    if actual_expressions != EXPRESSIONS or inference.get("task_token") != TASK_TOKEN:
        _fail("PROTOCOL_EXPRESSIONS_MISMATCH")
    if inference.get("generation") != GENERATION or inference.get("batch") != 1:
        _fail("PROTOCOL_GENERATION_MISMATCH")
    if inference.get("decode_skip_special_tokens") is not False or inference.get("decode_clean_up_tokenization_spaces") is not False:
        _fail("PROTOCOL_DECODE_MISMATCH")


def _validate_cohort(cohort: dict[str, Any], actual_hash: str) -> list[dict[str, Any]]:
    if actual_hash != COHORT_SHA256:
        _fail("COHORT_HASH_MISMATCH", f"{actual_hash}:{COHORT_SHA256}")
    if cohort.get("schema") != COHORT_SCHEMA or cohort.get("status") != "FROZEN_BEFORE_MODEL_OUTPUT":
        _fail("COHORT_IDENTITY_MISMATCH")
    protocol = cohort.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("sha256") != PROTOCOL_SHA256:
        _fail("COHORT_PROTOCOL_BINDING_MISMATCH")
    model = cohort.get("model")
    inference = cohort.get("inference")
    if not isinstance(model, dict) or model.get("revision") != MODEL_REVISION:
        _fail("COHORT_MODEL_BINDING_MISMATCH")
    if model.get("verification_manifest_sha256") != VERIFICATION_SHA256:
        _fail("COHORT_VERIFICATION_BINDING_MISMATCH")
    if not isinstance(inference, dict):
        _fail("COHORT_INFERENCE_MISSING")
    expected_ids = [item["expression_id"] for item in EXPRESSIONS]
    if inference.get("expression_ids_in_order") != expected_ids:
        _fail("COHORT_EXPRESSION_ORDER_MISMATCH")
    expected_inference = {
        "task_token": TASK_TOKEN,
        "expression_ids_in_order": expected_ids,
        "batch": 1,
        **GENERATION,
        "decode_skip_special_tokens": False,
        "decode_clean_up_tokenization_spaces": False,
        "model_dtype": "float16",
        "pixel_values_dtype": "float16",
        "attention_implementation": "eager",
    }
    if inference != expected_inference:
        _fail("COHORT_INFERENCE_MISMATCH", json.dumps(inference, sort_keys=True))
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
    for frame in frames:
        size = frame.get("source_image_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in size)
        ):
            _fail("SOURCE_IMAGE_SIZE_INVALID", str(frame.get("index")))
        files = frame.get("files")
        if not isinstance(files, dict) or set(files) != FILE_KINDS:
            _fail("FRAME_FILE_KINDS_MISMATCH", str(frame.get("index")))
    return frames


def _validate_model(model_root: Path) -> dict[str, Any]:
    verification_path = _existing_file(model_root / "verification.json", "MODEL_VERIFICATION")
    if _sha256(verification_path) != VERIFICATION_SHA256:
        _fail("MODEL_VERIFICATION_HASH_MISMATCH")
    verification = _read_json(verification_path, "MODEL_VERIFICATION")
    if (
        verification.get("repository") != "microsoft/Florence-2-large-ft"
        or verification.get("revision") != MODEL_REVISION
        or verification.get("upstream_file_count") != len(MODEL_FILES)
    ):
        _fail("MODEL_VERIFICATION_IDENTITY_MISMATCH")
    receipts: dict[str, Any] = {}
    for name, (expected_bytes, expected_hash) in MODEL_FILES.items():
        path = _existing_file(model_root / name, f"MODEL_{name}")
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            _fail("MODEL_FILE_MISMATCH", f"{name}:{actual_bytes}:{actual_hash}")
        receipts[name] = {"bytes": actual_bytes, "sha256": actual_hash}
    return {
        "verification": {"path": str(verification_path), "sha256": VERIFICATION_SHA256},
        "files": receipts,
    }


def _manifest_file(root: Path, entry: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(entry, dict):
        _fail(f"{label}_ENTRY_INVALID")
    relative = _relative_posix(entry.get("path"), f"{label}_PATH")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError:
        _fail(f"{label}_PATH_ESCAPE", str(path))
    if not path.is_file():
        _fail(f"{label}_NOT_FILE", str(path))
    expected_bytes = entry.get("bytes")
    expected_hash = entry.get("sha256")
    actual_bytes = path.stat().st_size
    actual_hash = _sha256(path)
    if actual_bytes != expected_bytes or actual_hash != expected_hash:
        _fail(f"{label}_MISMATCH", f"{actual_bytes}:{actual_hash}")
    return path, {"path": str(path), "bytes": actual_bytes, "sha256": actual_hash}


def _validate_runtime(torch: Any, transformers: Any, tokenizers: Any, pil_module: Any) -> None:
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "pillow": pil_module.__version__,
    }
    if actual != EXPECTED_RUNTIME:
        _fail("RUNTIME_VERSION_MISMATCH", json.dumps(actual, sort_keys=True))
    executable = _existing_file(Path(sys.executable), "PYTHON_EXECUTABLE")
    if _sha256(executable) != PYTHON_EXE_SHA256:
        _fail("PYTHON_EXECUTABLE_HASH_MISMATCH", str(executable))
    expected_environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_TF": "0"}
    for name, expected in expected_environment.items():
        if os.environ.get(name) != expected:
            _fail("OFFLINE_ENVIRONMENT_MISMATCH", f"{name}={os.environ.get(name)!r}:{expected!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        _fail("CUDA_UNAVAILABLE")


def _rasterize_instances(
    parsed: Any,
    expression: dict[str, str],
    image_size: tuple[int, int],
    image_module: Any,
    image_draw_module: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(parsed, dict) or set(parsed) != {"polygons", "labels"}:
        _fail("POSTPROCESS_STRUCTURE_INVALID", expression["expression_id"])
    polygons = parsed["polygons"]
    labels = parsed["labels"]
    if not isinstance(polygons, list) or not isinstance(labels, list) or len(polygons) != len(labels):
        _fail("POSTPROCESS_ARRAYS_INVALID", expression["expression_id"])
    width, height = image_size
    serializable: list[dict[str, Any]] = []
    runtime_instances: list[dict[str, Any]] = []
    for instance_index, (subpolygons, label) in enumerate(zip(polygons, labels)):
        if not isinstance(label, str) or not isinstance(subpolygons, list) or not subpolygons:
            _fail("POSTPROCESS_INSTANCE_INVALID", f"{expression['expression_id']}:{instance_index}")
        normalized: list[list[float]] = []
        mask = image_module.new("L", image_size, 0)
        draw = image_draw_module.Draw(mask)
        for polygon_index, raw_polygon in enumerate(subpolygons):
            if not isinstance(raw_polygon, list):
                _fail("POLYGON_NOT_LIST", f"{expression['expression_id']}:{instance_index}:{polygon_index}")
            try:
                polygon = [float(value) for value in raw_polygon]
            except (TypeError, ValueError) as exc:
                _fail("POLYGON_COORDINATE_INVALID", str(exc))
            if len(polygon) < 6 or len(polygon) % 2 != 0 or not all(math.isfinite(value) for value in polygon):
                _fail("POLYGON_SHAPE_INVALID", f"{expression['expression_id']}:{instance_index}:{polygon_index}")
            points = list(zip(polygon[0::2], polygon[1::2]))
            if any(x < 0.0 or x >= width or y < 0.0 or y >= height for x, y in points):
                _fail("POLYGON_OUT_OF_IMAGE", f"{expression['expression_id']}:{instance_index}:{polygon_index}")
            draw.polygon(points, fill=1)
            normalized.append(polygon)
        true_indices = [index for index, value in enumerate(mask.get_flattened_data()) if value]
        area = len(true_indices)
        instance_id = f"{expression['expression_id']}:{instance_index:03d}"
        record = {
            "instance_id": instance_id,
            "expression_id": expression["expression_id"],
            "stable_instance_index": instance_index,
            "label": label,
            "subpolygons_xy": normalized,
            "mask_area_pixels": area,
        }
        serializable.append(record)
        runtime_instances.append({**record, "mask": mask, "true_indices": true_indices})
    return serializable, runtime_instances


def _assign_topology(
    instances_by_expression: dict[str, list[dict[str, Any]]], image_size: tuple[int, int]
) -> tuple[list[dict[str, Any]], bool]:
    width, height = image_size
    parents = [
        instance
        for expression_id in ("architectural_leaf", "cabinet_door", "wardrobe_closet_door")
        for instance in instances_by_expression[expression_id]
    ]
    children = [
        instance
        for expression_id in ("operation_part", "hinge")
        for instance in instances_by_expression[expression_id]
    ]
    assignments: list[dict[str, Any]] = []
    authorized = False
    for child in children:
        indices = child["true_indices"]
        if indices:
            mean_x = sum(index % width for index in indices) / len(indices)
            mean_y = sum(index // width for index in indices) / len(indices)
            representative = [
                min(width - 1, max(0, math.floor(mean_x + 0.5))),
                min(height - 1, max(0, math.floor(mean_y + 0.5))),
            ]
            candidates = [parent for parent in parents if parent["mask"].getpixel(tuple(representative))]
        else:
            mean_x = None
            mean_y = None
            representative = None
            candidates = []
        candidates.sort(
            key=lambda parent: (
                parent["mask_area_pixels"],
                PARENT_PRIORITY[parent["expression_id"]],
                parent["stable_instance_index"],
            )
        )
        assigned = candidates[0] if candidates else None
        if assigned is not None and assigned["expression_id"] == "architectural_leaf":
            authorized = True
        assignments.append(
            {
                "child_instance_id": child["instance_id"],
                "child_expression_id": child["expression_id"],
                "child_mask_area_pixels": child["mask_area_pixels"],
                "child_mean_pixel_index_xy": [mean_x, mean_y] if mean_x is not None else None,
                "representative_pixel_xy": representative,
                "containing_parent_instance_ids_in_assignment_order": [
                    parent["instance_id"] for parent in candidates
                ],
                "assigned_parent_instance_id": assigned["instance_id"] if assigned else None,
                "assigned_parent_expression_id": assigned["expression_id"] if assigned else None,
            }
        )
    return assignments, authorized


def _aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    authorized_by_role = {
        role: sum(row["role"] == role and row["pixel_part_topology"] for row in rows)
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
    expression_counts = Counter()
    for row in rows:
        for output in row["expression_outputs"]:
            expression_counts[output["expression_id"]] += len(output["instances"])
    metrics = {
        "frames": len(rows),
        "generation_calls": len(rows) * len(EXPRESSIONS),
        "polygon_instances_by_expression": dict(expression_counts),
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
            "L10_PB13_FLORENCE_PIXEL_PART_TOPOLOGY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_PB13_FLORENCE_PIXEL_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET"
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
    model_root_argument: Path,
    output_argument: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = _existing_file(protocol_argument, "PROTOCOL")
    cohort_path = _existing_file(cohort_argument, "COHORT")
    extracted_root = _existing_directory(extracted_root_argument, "EXTRACTED_ROOT")
    model_root = _existing_directory(model_root_argument, "MODEL_ROOT")
    output_path = _new_output(output_argument)

    protocol_hash = _sha256(protocol_path)
    protocol = _read_json(protocol_path, "PROTOCOL")
    _validate_protocol(protocol, protocol_hash)
    cohort_hash = _sha256(cohort_path)
    cohort = _read_json(cohort_path, "COHORT")
    frames = _validate_cohort(cohort, cohort_hash)
    model_receipt = _validate_model(model_root)

    try:
        import PIL
        from PIL import Image, ImageDraw
        import tokenizers
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        _fail("INFERENCE_RUNTIME_IMPORT_FAILED", str(exc))
    _validate_runtime(torch, transformers, tokenizers, PIL)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    prepared_frames: list[dict[str, Any]] = []
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
            with Image.open(paths["rgb"]) as image:
                image.load()
                size = [int(image.width), int(image.height)]
                mode = image.mode
        except Exception as exc:
            _fail("RGB_DECODE_FAILED", f"frame={frame['index']}:{exc}")
        if size != frame["source_image_size"] or mode not in {"RGB", "L", "RGBA"}:
            _fail("RGB_IDENTITY_MISMATCH", f"frame={frame['index']}:{size}:{mode}")
        prepared_frames.append({"frame": frame, "paths": paths, "receipts": receipts})

    processor = AutoProcessor.from_pretrained(
        str(model_root), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_root),
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda").eval()
    processor_code = _existing_file(Path(inspect.getfile(processor.__class__)), "LOADED_PROCESSOR_CODE")
    model_code = _existing_file(Path(inspect.getfile(model.__class__)), "LOADED_MODEL_CODE")
    if _sha256(processor_code) != MODEL_FILES["processing_florence2.py"][1]:
        _fail("LOADED_PROCESSOR_CODE_HASH_MISMATCH", str(processor_code))
    if _sha256(model_code) != MODEL_FILES["modeling_florence2.py"][1]:
        _fail("LOADED_MODEL_CODE_HASH_MISMATCH", str(model_code))
    if getattr(model.config, "_attn_implementation", None) != "eager":
        _fail("LOADED_ATTENTION_IMPLEMENTATION_MISMATCH")

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prepared in prepared_frames:
            frame = prepared["frame"]
            with Image.open(prepared["paths"]["rgb"]) as opened:
                image = opened.convert("RGB")
            image_size = (image.width, image.height)
            expression_outputs: list[dict[str, Any]] = []
            instances_by_expression: dict[str, list[dict[str, Any]]] = {}
            for expression in EXPRESSIONS:
                inputs = processor(
                    text=TASK_TOKEN + expression["text"], images=image, return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to("cuda")
                pixel_values = inputs["pixel_values"].to(device="cuda", dtype=torch.float16)
                generated = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    **GENERATION,
                )
                decoded = processor.batch_decode(
                    generated,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )[0]
                parsed_wrapper = processor.post_process_generation(
                    decoded, task=TASK_TOKEN, image_size=image_size
                )
                if not isinstance(parsed_wrapper, dict) or set(parsed_wrapper) != {TASK_TOKEN}:
                    _fail("POSTPROCESS_TASK_WRAPPER_INVALID", expression["expression_id"])
                serializable, runtime_instances = _rasterize_instances(
                    parsed_wrapper[TASK_TOKEN], expression, image_size, Image, ImageDraw
                )
                instances_by_expression[expression["expression_id"]] = runtime_instances
                expression_outputs.append(
                    {
                        **expression,
                        "generated_token_count": int(generated.shape[-1]),
                        "decoded_text": decoded,
                        "instances": serializable,
                    }
                )
            assignments, authorized = _assign_topology(instances_by_expression, image_size)
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
                    "files": prepared["receipts"],
                    "expression_outputs": expression_outputs,
                    "child_parent_assignments": assignments,
                    "architectural_frame_instance_count": len(
                        instances_by_expression["architectural_frame"]
                    ),
                    "pixel_part_topology": authorized,
                }
            )
            del image
    torch.cuda.synchronize(0)
    metrics, gate = _aggregate(rows)
    properties = torch.cuda.get_device_properties(0)
    evaluator_path = Path(__file__).resolve()
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "L10-PB13 Florence Pixel Part-Parent Topology",
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
            "model_root": str(model_root),
            "model_revision": MODEL_REVISION,
            "model_receipt": model_receipt,
        },
        "evaluator": {"path": str(evaluator_path), "sha256": _sha256(evaluator_path)},
        "runtime": {
            **EXPECTED_RUNTIME,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": PYTHON_EXE_SHA256,
            "cuda_runtime": torch.version.cuda,
            "actual_device_index": 0,
            "actual_device_name": properties.name,
            "actual_device_capability": list(torch.cuda.get_device_capability(0)),
            "device_total_memory_bytes": properties.total_memory,
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(0),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "model_dtype": str(next(model.parameters()).dtype),
            "loaded_processor_code": {"path": str(processor_code), "sha256": _sha256(processor_code)},
            "loaded_model_code": {"path": str(model_code), "sha256": _sha256(model_code)},
            "offline_environment": {name: os.environ[name] for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "USE_TF")},
            "elapsed_seconds": time.perf_counter() - started,
        },
        "inference": {
            "task_token": TASK_TOKEN,
            "expressions_in_order": EXPRESSIONS,
            "batch": 1,
            "generation": GENERATION,
            "decode_skip_special_tokens": False,
            "decode_clean_up_tokenization_spaces": False,
            "postprocess_original_image_size": True,
            "model_dtype": "float16",
            "pixel_values_dtype": "float16",
        },
        "topology": {
            "parent_expression_ids": ["architectural_leaf", "cabinet_door", "wardrobe_closet_door"],
            "child_expression_ids": ["operation_part", "hinge"],
            "representative": "ROUND_HALF_UP_MEAN_INTEGER_TRUE_MASK_PIXEL_INDICES",
            "assignment_order": "MIN_PARENT_MASK_AREA_THEN_CABINET_THEN_WARDROBE_CLOSET_THEN_ARCHITECTURAL_THEN_STABLE_INSTANCE_INDEX",
            "authorization": "AT_LEAST_ONE_CHILD_ASSIGNED_TO_ARCHITECTURAL_LEAF",
            "frame_is_diagnostic_only": True,
        },
        "execution_history": [
            {
                "attempt": 1,
                "status": "ABORTED_BEFORE_MODEL_LOAD_AND_COHORT_IMAGE_DECODE",
                "error": "OFFLINE_ENVIRONMENT_MISMATCH: USE_TF='0' was incorrectly compared with '1' by the evaluator guard",
                "cohort_model_outputs_observed": False,
                "result_file_created": False,
                "recovery": "Corrected only the pre-output environment assertion to the protocol-frozen USE_TF=0 value; protocol, cohort, model, prompts, generation, topology, and gate are unchanged.",
            },
            {
                "attempt": 2,
                "status": "THIS_RESULT",
                "reason": "ONE_MECHANICAL_REPLAY_AFTER_PRE_OUTPUT_GUARD_FAILURE",
            },
        ],
        "frames": rows,
        "metrics": metrics,
        "gate": gate,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json_new(output_path, result)
    return result


def _synthetic_instance(
    image_module: Any,
    expression_id: str,
    instance_index: int,
    points: Iterable[tuple[int, int]],
    image_size: tuple[int, int] = (10, 10),
) -> dict[str, Any]:
    mask = image_module.new("L", image_size, 0)
    for point in points:
        mask.putpixel(point, 1)
    indices = [index for index, value in enumerate(mask.get_flattened_data()) if value]
    return {
        "instance_id": f"{expression_id}:{instance_index:03d}",
        "expression_id": expression_id,
        "stable_instance_index": instance_index,
        "mask_area_pixels": len(indices),
        "mask": mask,
        "true_indices": indices,
    }


def _self_test() -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        _fail("SELF_TEST_PIL_IMPORT_FAILED", str(exc))
    all_pixels = [(x, y) for y in range(10) for x in range(10)]
    cabinet_pixels = [(x, y) for y in range(3, 7) for x in range(3, 7)]
    child_pixels = [(4, 4), (5, 4), (4, 5), (5, 5)]
    instances = {expression["expression_id"]: [] for expression in EXPRESSIONS}
    instances["architectural_leaf"] = [_synthetic_instance(Image, "architectural_leaf", 0, all_pixels)]
    instances["cabinet_door"] = [_synthetic_instance(Image, "cabinet_door", 0, cabinet_pixels)]
    instances["operation_part"] = [_synthetic_instance(Image, "operation_part", 0, child_pixels)]
    assignments, authorized = _assign_topology(instances, (10, 10))
    if assignments[0]["assigned_parent_expression_id"] != "cabinet_door" or authorized:
        raise AssertionError("smallest competitor parent did not absorb the child")
    competitor_assignment = assignments[0]["assigned_parent_expression_id"]
    instances["cabinet_door"] = []
    assignments, authorized = _assign_topology(instances, (10, 10))
    if assignments[0]["assigned_parent_expression_id"] != "architectural_leaf" or not authorized:
        raise AssertionError("architectural parent authorization failed")
    rows = []
    for index in range(8):
        role = POSITIVE_ROLE if index < 4 else FURNITURE_ROLE if index < 6 else OOD_ROLE
        rows.append(
            {
                "role": role,
                "pixel_part_topology": index < 4,
                "expression_outputs": [
                    {"expression_id": expression["expression_id"], "instances": []}
                    for expression in EXPRESSIONS
                ],
            }
        )
    metrics, gate = _aggregate(rows)
    if metrics["balanced_accuracy"] != 1.0 or not gate["development_gate_met"]:
        raise AssertionError("perfect frozen gate did not pass")
    rows[0]["pixel_part_topology"] = False
    _, failed = _aggregate(rows)
    if failed["development_gate_met"]:
        raise AssertionError("imperfect frozen gate passed")
    return {
        "status": "PASS",
        "competitor_assignment": competitor_assignment,
        "perfect_gate_decision": gate["decision"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen PB13 Florence pixel topology evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run one deterministic synthetic topology and gate check.")
    run = subparsers.add_parser("run", help="Run the one frozen eight-image CUDA evaluation.")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--extracted-root", type=Path, required=True)
    run.add_argument("--model-root", type=Path, required=True)
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
            args.model_root,
            args.output,
        )
    except ContractError as exc:
        print(f"PB13_CONTRACT_ERROR:{exc}", file=sys.stderr)
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
