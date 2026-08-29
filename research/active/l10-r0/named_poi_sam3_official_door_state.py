#!/usr/bin/env python3
"""Frozen PB17 official-Transformers SAM3 direct door-state source ceiling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "research/active/l10-r0/named_poi_sam3_official_door_state_protocol_v1.json"
COHORT_PATH = REPO_ROOT / "research/active/l10-r0/named_poi_sam3_official_door_state_development_cohort_v1.json"
OUTPUT_PATH = REPO_ROOT / "research/active/l10-r0/named_poi_sam3_official_door_state_development_result_v1.json"
MODEL_ROOT = REPO_ROOT / "artifacts.local/models/sam3-modelscope"
DATASET_ROOT = Path(r"F:\ba-data\blindassist-artifacts-20260805\datasets")
LAUNCHER_PATH = Path(r"E:\codex-tools\bin\blindassist-research-gpu.cmd")
EXPECTED_PROTOCOL_SHA256 = "09a4d85f8956379a66be95ee1d0efe19f340625caf1a838e3510a15a02ef7828"
EXPECTED_COHORT_SHA256 = "f188043559d65edca030c80ec30ade743b63325691fe9bee4b42138045f61de4"

PROMPTS: tuple[dict[str, str], ...] = (
    {"class_id": "architectural_leaf", "text": "door"},
    {"class_id": "cabinet_door", "text": "cabinet door"},
    {"class_id": "closet_door", "text": "closet door"},
    {"class_id": "refrigerator_door", "text": "refrigerator door"},
    {"class_id": "doorless_opening", "text": "doorway"},
)
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
MAXIMUM_INSTANCES_PER_PROMPT = 12
POSITIVE_ROLE = "ARCHITECTURAL_DOOR_LEAF"
NEGATIVE_ROLES = {"HANDLED_FURNITURE_CONTROL", "LARGE_DOORLESS_OPENING_CONTROL"}
PASS_DECISION = "L10_PB17_OFFICIAL_SAM3_DOOR_STATE_DEVELOPMENT_GATE_MET"
FAIL_DECISION = "L10_PB17_OFFICIAL_SAM3_DOOR_STATE_DEVELOPMENT_GATE_NOT_MET"
HEAVY_MODULE_PREFIXES = ("torch", "transformers", "PIL", "numpy")


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is not an object: {path}")
    return value


def _require_exact_receipt(path: Path, receipt: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ContractError(f"{label} missing: {path}")
    actual_bytes = path.stat().st_size
    expected_bytes = int(receipt["bytes"])
    if actual_bytes != expected_bytes:
        raise ContractError(f"{label} byte mismatch: {actual_bytes} != {expected_bytes}")
    actual_hash = _sha256(path)
    expected_hash = str(receipt["sha256"]).lower()
    if actual_hash != expected_hash:
        raise ContractError(f"{label} SHA-256 mismatch: {actual_hash} != {expected_hash}")


def _safe_member_path(local_path: str) -> Path:
    if not isinstance(local_path, str) or not local_path or Path(local_path).is_absolute():
        raise ContractError(f"unsafe cohort member path: {local_path!r}")
    normalized = local_path.replace("/", os.sep)
    candidate = (DATASET_ROOT / normalized).resolve()
    root = DATASET_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ContractError(f"cohort member escapes dataset root: {local_path!r}") from error
    return candidate


def _load_and_verify_contract(*, require_output_absent: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ContractError("protocol SHA-256 does not match frozen evaluator constant")
    if _sha256(COHORT_PATH) != EXPECTED_COHORT_SHA256:
        raise ContractError("cohort SHA-256 does not match frozen evaluator constant")
    protocol = _read_json(PROTOCOL_PATH)
    cohort = _read_json(COHORT_PATH)
    if protocol.get("schema") != "l10-named-poi-sam3-official-door-state-protocol-v1":
        raise ContractError("unexpected protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_FRESH_COHORT_MODEL_OUTPUT":
        raise ContractError("protocol is not frozen")
    if cohort.get("schema") != "l10-named-poi-sam3-official-door-state-cohort-v1":
        raise ContractError("unexpected cohort schema")
    if cohort.get("status") != "FROZEN_BEFORE_MODEL_OUTPUT":
        raise ContractError("cohort is not frozen")
    if cohort.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ContractError("cohort does not bind the frozen protocol")
    if protocol.get("inference", {}).get("prompts_in_order") != list(PROMPTS):
        raise ContractError("prompt order differs from evaluator")
    if protocol.get("inference", {}).get("confidence_threshold") != CONFIDENCE_THRESHOLD:
        raise ContractError("confidence threshold differs from evaluator")
    if protocol.get("inference", {}).get("mask_threshold") != MASK_THRESHOLD:
        raise ContractError("mask threshold differs from evaluator")
    if protocol.get("inference", {}).get("maximum_instances_per_prompt") != MAXIMUM_INSTANCES_PER_PROMPT:
        raise ContractError("per-prompt cap differs from evaluator")
    if require_output_absent and OUTPUT_PATH.exists():
        raise ContractError(f"formal output already exists: {OUTPUT_PATH}")

    frames = cohort.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        raise ContractError("cohort must contain exactly eight frames")
    roles = [frame.get("role") for frame in frames]
    if roles.count(POSITIVE_ROLE) != 4:
        raise ContractError("cohort must contain four architectural-door positives")
    if roles.count("HANDLED_FURNITURE_CONTROL") != 2:
        raise ContractError("cohort must contain two handled-furniture controls")
    if roles.count("LARGE_DOORLESS_OPENING_CONTROL") != 2:
        raise ContractError("cohort must contain two doorless-opening controls")
    sequence_ids = [frame.get("capture_sequence_id") for frame in frames]
    if len(set(sequence_ids)) != 8:
        raise ContractError("cohort capture sequences are not unique")

    verified_members = 0
    verified_bytes = 0
    for frame_index, frame in enumerate(frames, start=1):
        if frame.get("index") != frame_index:
            raise ContractError(f"non-canonical frame index at {frame_index}")
        files = frame.get("files")
        if not isinstance(files, dict) or set(files) != {"rgb", "depth", "intrinsics", "scene_metadata", "polygon"}:
            raise ContractError(f"unexpected file receipt set at frame {frame_index}")
        for member_name, member in files.items():
            path = _safe_member_path(member["local_path"])
            _require_exact_receipt(path, member, f"frame {frame_index} {member_name}")
            verified_members += 1
            verified_bytes += int(member["bytes"])

    asset_paths = {
        "model": MODEL_ROOT / "model.safetensors",
        "tokenizer": MODEL_ROOT / "tokenizer.json",
        "config": MODEL_ROOT / "config.json",
        "processor_config": MODEL_ROOT / "processor_config.json",
        "tokenizer_config": MODEL_ROOT / "tokenizer_config.json",
        "vocab": MODEL_ROOT / "vocab.json",
        "merges": MODEL_ROOT / "merges.txt",
        "special_tokens_map": MODEL_ROOT / "special_tokens_map.json",
        "license": MODEL_ROOT / "LICENSE",
    }
    for label, path in asset_paths.items():
        _require_exact_receipt(path, protocol["model"]["files"][label], f"model asset {label}")
    _require_exact_receipt(
        REPO_ROOT / protocol["execution_readiness_at_freeze"]["synthetic_smoke_receipt"]["path"],
        protocol["execution_readiness_at_freeze"]["synthetic_smoke_receipt"],
        "synthetic smoke receipt",
    )
    _require_exact_receipt(LAUNCHER_PATH, protocol["runtime"]["launcher"], "launcher")
    _require_exact_receipt(Path(sys.executable).resolve(), protocol["runtime"]["python_executable"], "python executable")

    return protocol, cohort, {
        "verified_local_members": verified_members,
        "verified_local_member_bytes": verified_bytes,
    }


def _stable_instances(
    scores: Sequence[float],
    boxes: Sequence[Sequence[float]],
    masks: Sequence[Any],
    *,
    class_id: str,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    if not (len(scores) == len(boxes) == len(masks)):
        raise ContractError(f"{class_id} output lengths differ")
    candidates: list[dict[str, Any]] = []
    invalid_count = 0
    empty_count = 0
    for native_index, (score_value, box_values, mask) in enumerate(zip(scores, boxes, masks)):
        score = float(score_value)
        box = [float(value) for value in box_values]
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            invalid_count += 1
            continue
        if len(box) != 4 or any(not math.isfinite(value) for value in box):
            invalid_count += 1
            continue
        if mask.ndim != 2:
            invalid_count += 1
            continue
        unique_values = {int(value) for value in torch_module.unique(mask).detach().cpu().tolist()}
        if not unique_values.issubset({0, 1}):
            invalid_count += 1
            continue
        bool_mask = mask.to(dtype=torch_module.bool)
        true_y, true_x = torch_module.where(bool_mask)
        area = int(true_y.numel())
        if area == 0:
            empty_count += 1
            continue
        mean_x = int(torch_module.div(true_x.sum(), area, rounding_mode="floor").item())
        mean_y = int(torch_module.div(true_y.sum(), area, rounding_mode="floor").item())
        mask_bytes = bool_mask.to(dtype=torch_module.uint8).contiguous().cpu().numpy().tobytes(order="C")
        candidates.append(
            {
                "native_index": native_index,
                "score": score,
                "box_xyxy": box,
                "mask_area_pixels": area,
                "mean_integer_true_mask_pixel_xy": [mean_x, mean_y],
                "mask_c_order_uint8_sha256": hashlib.sha256(mask_bytes).hexdigest(),
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["native_index"]))
    retained = candidates[:MAXIMUM_INSTANCES_PER_PROMPT]
    for stable_index, item in enumerate(retained):
        item["instance_id"] = f"{class_id}:{stable_index:03d}"
    return retained, invalid_count, empty_count


def _score_predictions(expected: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    if len(expected) != len(predicted) or not expected:
        raise ContractError("prediction vectors must be nonempty and equal length")
    tp = sum(1 for truth, guess in zip(expected, predicted) if truth and guess)
    fn = sum(1 for truth, guess in zip(expected, predicted) if truth and not guess)
    fp = sum(1 for truth, guess in zip(expected, predicted) if not truth and guess)
    tn = sum(1 for truth, guess in zip(expected, predicted) if not truth and not guess)
    positives = tp + fn
    negatives = tn + fp
    recall = tp / positives
    tnr = tn / negatives
    balanced_accuracy = (recall + tnr) / 2.0
    return {
        "true_positive_count": tp,
        "false_negative_count": fn,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "positive_recall": recall,
        "negative_true_negative_rate": tnr,
        "negative_false_positive_rate": fp / negatives,
        "balanced_accuracy": balanced_accuracy,
    }


def _self_test() -> dict[str, Any]:
    forbidden_before = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in HEAVY_MODULE_PREFIXES
    )
    if forbidden_before:
        raise AssertionError(f"heavy modules imported before self-test: {forbidden_before}")
    metrics = _score_predictions(
        [True, True, True, True, False, False, False, False],
        [True, True, True, True, False, False, False, False],
    )
    if metrics["balanced_accuracy"] != 1.0:
        raise AssertionError("perfect gate arithmetic failed")
    failure = _score_predictions([True, False], [False, False])
    if failure["balanced_accuracy"] != 0.5:
        raise AssertionError("failure arithmetic failed")
    unsafe = [
        "../escape.jpg",
        "x/../../escape.jpg",
        "/absolute.jpg",
        r"C:\absolute.jpg",
        "",
    ]
    rejected = 0
    for value in unsafe:
        try:
            _safe_member_path(value)
        except ContractError:
            rejected += 1
    if rejected != len(unsafe):
        raise AssertionError("unsafe path rejection failed")
    stable = sorted(
        [{"score": 0.4, "native_index": 2}, {"score": 0.8, "native_index": 1}, {"score": 0.8, "native_index": 0}],
        key=lambda item: (-item["score"], item["native_index"]),
    )
    if [item["native_index"] for item in stable] != [0, 1, 2]:
        raise AssertionError("stable tie order failed")
    forbidden_after = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in HEAVY_MODULE_PREFIXES
    )
    if forbidden_after:
        raise AssertionError(f"heavy modules imported by self-test: {forbidden_after}")
    return {
        "status": "PASS",
        "perfect_balanced_accuracy": metrics["balanced_accuracy"],
        "one_miss_balanced_accuracy": failure["balanced_accuracy"],
        "stable_tie_order": [item["native_index"] for item in stable],
        "unsafe_candidate_paths_rejected": rejected,
        "torch_transformers_pil_numpy_imports": 0,
        "cohort_files_read": 0,
        "cohort_images_decoded": 0,
        "model_calls": 0,
    }


def _runtime_check(protocol: dict[str, Any]) -> dict[str, Any]:
    expected_environment = {
        "BLINDASSIST_SAM_LICENSE_ACCEPTED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    actual_environment = {key: os.environ.get(key) for key in expected_environment}
    if actual_environment != expected_environment:
        raise ContractError(f"offline/license environment mismatch: {actual_environment!r}")

    import torch
    import transformers

    if platform.python_version() != protocol["runtime"]["python"]:
        raise ContractError("Python version differs from protocol")
    actual_versions = {
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "transformers": transformers.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "Pillow": importlib.metadata.version("Pillow"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "safetensors": importlib.metadata.version("safetensors"),
    }
    if actual_versions != protocol["runtime"]["distributions"]:
        raise ContractError(f"runtime distribution mismatch: {actual_versions!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError("exactly one CUDA device is required")
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_name(0) != protocol["runtime"]["gpu"]:
        raise ContractError("GPU name differs from protocol")
    if properties.total_memory // (1024 * 1024) < protocol["runtime"]["gpu_memory_mib_min"]:
        raise ContractError("GPU memory is below frozen minimum")
    if not torch.cuda.is_bf16_supported():
        raise ContractError("CUDA BF16 is unavailable")
    return {
        "python": platform.python_version(),
        "python_executable": _file_receipt(Path(sys.executable).resolve()),
        "distributions": actual_versions,
        "offline_environment": actual_environment,
        "cuda": {
            "device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime": torch.version.cuda,
            "total_memory_bytes": properties.total_memory,
            "total_memory_mib_floor": properties.total_memory // (1024 * 1024),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
    }


def _run() -> dict[str, Any]:
    run_started = time.perf_counter()
    protocol, cohort, preflight = _load_and_verify_contract(require_output_absent=True)
    runtime = _runtime_check(protocol)

    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    load_started = time.perf_counter()
    loaded = Sam3Model.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ContractError("output_loading_info did not return model and receipt")
    model, loading_info = loaded
    normalized_loading = {
        key: list(loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(normalized_loading.values()):
        raise ContractError(f"non-strict model load: {normalized_loading!r}")
    processor = Sam3Processor.from_pretrained(MODEL_ROOT, local_files_only=True)
    model = model.to("cuda").eval()
    model_load_seconds = time.perf_counter() - load_started

    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_devices != ["cuda:0"] or parameter_dtypes != ["torch.float32"]:
        raise ContractError(
            f"unexpected model placement: devices={parameter_devices!r} dtypes={parameter_dtypes!r}"
        )

    frame_results: list[dict[str, Any]] = []
    expected_labels: list[bool] = []
    predictions: list[bool] = []
    class_totals = {prompt["class_id"]: 0 for prompt in PROMPTS}
    total_invalid = 0
    total_empty = 0
    image_encode_calls = 0
    text_conditioned_calls = 0
    image_encoding_seconds = 0.0
    text_inference_seconds = 0.0
    maximum_peak_allocated = 0
    maximum_peak_reserved = 0

    for frame in cohort["frames"]:
        rgb_path = _safe_member_path(frame["files"]["rgb"]["local_path"])
        with Image.open(rgb_path) as opened:
            image = opened.convert("RGB")
        expected_size = tuple(frame["source_image_size_wh"])
        if image.size != expected_size:
            raise ContractError(
                f"frame {frame['index']} image size {image.size!r} differs from {expected_size!r}"
            )
        expected = frame["role"] == POSITIVE_ROLE
        if frame["role"] not in {POSITIVE_ROLE, *NEGATIVE_ROLES}:
            raise ContractError(f"unexpected role: {frame['role']!r}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        image_inputs = processor(images=image, return_tensors="pt")
        target_sizes = image_inputs["original_sizes"].tolist()
        encode_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vision_embeds = model.get_vision_features(
                pixel_values=image_inputs["pixel_values"].to("cuda")
            )
        torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - encode_started
        image_encode_calls += 1
        image_encoding_seconds += encode_seconds

        prompt_results: list[dict[str, Any]] = []
        for call_index, prompt in enumerate(PROMPTS):
            text_inputs = processor(text=prompt["text"], return_tensors="pt").to("cuda")
            inference_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(vision_embeds=vision_embeds, **text_inputs)
            torch.cuda.synchronize()
            processed = processor.post_process_instance_segmentation(
                outputs,
                threshold=CONFIDENCE_THRESHOLD,
                mask_threshold=MASK_THRESHOLD,
                target_sizes=target_sizes,
            )
            inference_seconds = time.perf_counter() - inference_started
            text_conditioned_calls += 1
            text_inference_seconds += inference_seconds
            if not isinstance(processed, list) or len(processed) != 1:
                raise ContractError(f"unexpected postprocess container at frame {frame['index']}")
            native = processed[0]
            if set(native) != {"scores", "boxes", "masks"}:
                raise ContractError(f"unexpected postprocess keys: {sorted(native)!r}")
            if tuple(native["masks"].shape[1:]) != (image.height, image.width):
                raise ContractError(f"mask size differs from source image at frame {frame['index']}")
            retained, invalid_count, empty_count = _stable_instances(
                native["scores"].detach().cpu().tolist(),
                native["boxes"].detach().cpu().tolist(),
                list(native["masks"]),
                class_id=prompt["class_id"],
                torch_module=torch,
            )
            class_totals[prompt["class_id"]] += len(retained)
            total_invalid += invalid_count
            total_empty += empty_count
            prompt_results.append(
                {
                    "global_call_index": call_index,
                    "class_id": prompt["class_id"],
                    "text": prompt["text"],
                    "native_filtered_count": int(native["scores"].shape[0]),
                    "retained_count": len(retained),
                    "discarded_by_per_prompt_cap": max(
                        0,
                        int(native["scores"].shape[0]) - invalid_count - empty_count - len(retained),
                    ),
                    "invalid_output_count": invalid_count,
                    "empty_mask_count": empty_count,
                    "inference_and_native_postprocess_seconds": inference_seconds,
                    "retained_instances": retained,
                }
            )

        architectural_count = next(
            item["retained_count"]
            for item in prompt_results
            if item["class_id"] == "architectural_leaf"
        )
        prediction = architectural_count > 0
        expected_labels.append(expected)
        predictions.append(prediction)
        maximum_peak_allocated = max(maximum_peak_allocated, torch.cuda.max_memory_allocated())
        maximum_peak_reserved = max(maximum_peak_reserved, torch.cuda.max_memory_reserved())
        frame_results.append(
            {
                "index": frame["index"],
                "frame_id": frame["frame_id"],
                "capture_sequence_id": frame["capture_sequence_id"],
                "sensor_source_bucket": frame["sensor_source_bucket"],
                "role": frame["role"],
                "expected_binary_label": expected,
                "official_sam3_direct_door_state": prediction,
                "audit_note": frame["audit_note"],
                "official_evidence": frame["official_evidence"],
                "canonical_source_path": frame["canonical_source_path"],
                "source_image_size_wh": list(image.size),
                "files": {
                    name: {
                        **receipt,
                        "path": str(_safe_member_path(receipt["local_path"])),
                    }
                    for name, receipt in frame["files"].items()
                },
                "native_inference": {
                    "image_encoding": {
                        "calls": 1,
                        "full_frame_rgb_only": True,
                        "source_image_size_wh": list(image.size),
                        "inference_seconds": encode_seconds,
                        "vision_embedding_reused_for_all_prompts": True,
                    },
                    "text_calls": prompt_results,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            }
        )

    metrics = _score_predictions(expected_labels, predictions)
    metrics.update(
        {
            "frames": len(frame_results),
            "image_encoding_calls": image_encode_calls,
            "text_conditioned_model_calls": text_conditioned_calls,
            "total_model_calls": image_encode_calls + text_conditioned_calls,
            "retained_instances_by_class_id": class_totals,
            "invalid_native_outputs": total_invalid,
            "empty_native_masks": total_empty,
            "image_encoding_seconds": image_encoding_seconds,
            "text_inference_and_native_postprocess_seconds": text_inference_seconds,
        }
    )
    gate_met = (
        metrics["true_positive_count"] == 4
        and metrics["false_negative_count"] == 0
        and metrics["false_positive_count"] == 0
        and metrics["true_negative_count"] == 4
        and metrics["balanced_accuracy"] == 1.0
    )
    decision = PASS_DECISION if gate_met else FAIL_DECISION
    result = {
        "schema": "l10-named-poi-sam3-official-door-state-development-result-v1",
        "experiment": "L10-PB17 Official SAM3 Direct Door-State Source Ceiling",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {
            "protocol": {
                "path": str(PROTOCOL_PATH),
                "schema": protocol["schema"],
                "sha256": EXPECTED_PROTOCOL_SHA256,
            },
            "cohort": {
                "path": str(COHORT_PATH),
                "schema": cohort["schema"],
                "sha256": EXPECTED_COHORT_SHA256,
                "status": cohort["status"],
            },
            "model": {
                "repository": protocol["model"]["repository"],
                "repository_commit": protocol["model"]["repository_commit"],
                "files": protocol["model"]["files"],
            },
            "verified_local_members": preflight["verified_local_members"],
            "verified_local_member_bytes": preflight["verified_local_member_bytes"],
        },
        "evaluator": _file_receipt(Path(__file__).resolve()),
        "configuration": {
            "builder": "transformers.Sam3Model.from_pretrained",
            "processor": "transformers.Sam3Processor.from_pretrained",
            "strict_loading_receipt": normalized_loading,
            "prompts_in_order": list(PROMPTS),
            "authorization": "AT_LEAST_ONE_NONEMPTY_NATIVE_MASK_FOR_EXACT_TEXT_DOOR",
            "competitor_prompts_are_diagnostic_only": True,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "mask_threshold": MASK_THRESHOLD,
            "maximum_instances_per_prompt": MAXIMUM_INSTANCES_PER_PROMPT,
            "full_frame_only": True,
            "vision_embedding_reused_for_all_prompts": True,
            "compile": False,
            "inference_mode": True,
            "autocast": "CUDA_BFLOAT16",
        },
        "runtime": {
            **runtime,
            "launcher": _file_receipt(LAUNCHER_PATH),
            "loaded_model": {
                "model_type": type(model).__name__,
                "processor_type": type(processor).__name__,
                "parameter_count": parameter_count,
                "parameter_devices": parameter_devices,
                "parameter_dtypes": parameter_dtypes,
                "checkpoint_loading": normalized_loading,
            },
            "model_load_seconds": model_load_seconds,
            "elapsed_seconds": time.perf_counter() - run_started,
            "maximum_frame_peak_cuda_allocated_bytes": maximum_peak_allocated,
            "maximum_frame_peak_cuda_reserved_bytes": maximum_peak_reserved,
        },
        "frames": frame_results,
        "metrics": metrics,
        "gate": {
            "development_gate_met": gate_met,
            "four_of_four_architectural_doors_detected": metrics["true_positive_count"] == 4,
            "zero_of_two_handled_furniture_controls_detected": all(
                not row["official_sam3_direct_door_state"]
                for row in frame_results
                if row["role"] == "HANDLED_FURNITURE_CONTROL"
            ),
            "zero_of_two_large_doorless_opening_controls_detected": all(
                not row["official_sam3_direct_door_state"]
                for row in frame_results
                if row["role"] == "LARGE_DOORLESS_OPENING_CONTROL"
            ),
            "balanced_accuracy_is_one": metrics["balanced_accuracy"] == 1.0,
            "decision": decision,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(OUTPUT_PATH)
    print(
        json.dumps(
            {
                "decision": decision,
                "metrics": metrics,
                "output": _file_receipt(OUTPUT_PATH),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen PB17 official-SAM3 direct door-state evaluator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run standard-library-only arithmetic and path checks.")
    subparsers.add_parser(
        "preflight",
        help="Verify frozen hashes and local members without importing model libraries or decoding RGB.",
    )
    subparsers.add_parser("run", help="Run the one-shot frozen Development cohort.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "self-test":
        print(json.dumps(_self_test(), ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "preflight":
        protocol, cohort, receipt = _load_and_verify_contract(require_output_absent=True)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol_schema": protocol["schema"],
                    "cohort_schema": cohort["schema"],
                    **receipt,
                    "torch_transformers_pil_numpy_imports": sum(
                        1
                        for name in sys.modules
                        if name.split(".", 1)[0] in HEAVY_MODULE_PREFIXES
                    ),
                    "cohort_images_decoded": 0,
                    "model_calls": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.command == "run":
        _run()
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
