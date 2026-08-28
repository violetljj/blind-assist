from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from research_backend import (  # noqa: E402
    BackendCandidate,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)

LABELS = ("failure", "success")
ROLE_TO_SPLIT = {"training": "train", "calibration": "val", "evaluation": "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)


def _validate_source(protocol: dict[str, Any], data_root: Path) -> None:
    for role, spec in protocol["source"]["splits"].items():
        for key in ("metadata", "media_archive"):
            prefix = "metadata" if key == "metadata" else "media_archive"
            path = data_root / spec[f"{prefix}_file"] if prefix == "metadata" else data_root / spec[prefix]
            if path.stat().st_size != int(spec[f"{prefix}_size_bytes"]):
                raise ValueError(f"{role.upper()}_{prefix.upper()}_SIZE_MISMATCH")
            if _sha256(path) != spec[f"{prefix}_sha256"]:
                raise ValueError(f"{role.upper()}_{prefix.upper()}_HASH_MISMATCH")


def _load_inputs(path: Path, split: str, expected_rows: int) -> list[dict[str, Any]]:
    allowed = {
        "taskvar",
        "task_instruction",
        "episode_id",
        "images",
        "detailed_subtask_name",
        "plan",
    }
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            source = json.loads(line)
            clean = {key: source[key] for key in allowed if key in source}
            required = allowed - {"plan"}
            if not required.issubset(clean):
                raise ValueError(f"MISSING_INPUT_FIELD:{split}:{index}")
            images = [str(value) for value in clean["images"]]
            if len(images) != 6:
                raise ValueError(f"EXPECTED_SIX_IMAGES:{split}:{index}")
            rows.append(
                {
                    "id": f"{split}:{index:04d}:{clean['taskvar']}:{clean['episode_id']}",
                    "split": split,
                    "line_index": index,
                    "task_instruction": str(clean["task_instruction"]),
                    "detailed_subtask_name": str(clean["detailed_subtask_name"]),
                    "images": images,
                    "plan": [str(value) for value in clean.get("plan", [])],
                }
            )
    if len(rows) != expected_rows:
        raise ValueError(f"ROW_COUNT_MISMATCH:{split}:{len(rows)}")
    return rows


def _load_roles(protocol: dict[str, Any], data_root: Path) -> dict[str, list[dict[str, Any]]]:
    roles = {}
    for role, spec in protocol["source"]["splits"].items():
        split = ROLE_TO_SPLIT[role]
        roles[role] = _load_inputs(
            data_root / spec["metadata_file"], split, int(spec["expected_rows"])
        )
        split_root = data_root / split
        missing = [
            image
            for row in roles[role]
            for image in row["images"]
            if not (split_root / image).is_file()
        ]
        if missing:
            raise ValueError(f"MISSING_MEDIA:{role}:{len(missing)}:{missing[:2]}")
    ids = [row["id"] for rows in roles.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("ROLE_ID_OVERLAP")
    return roles


def _model_path(model_root: Path, spec: dict[str, Any]) -> Path:
    path = model_root / spec["local_dir"]
    weights = path / spec["weights_file"]
    if _sha256(weights) != spec["weights_sha256"]:
        raise ValueError(f"MODEL_HASH_MISMATCH:{spec['model_id']}")
    return path


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _grounder_forward(model: Any, inputs: Any) -> Any:
    with torch.inference_mode():
        return model(**inputs)


def _select_grounder_backend(
    model_path: Path,
    processor: Any,
    image: Image.Image,
    prompt: str,
    receipt: Path,
) -> tuple[dict[str, Any], Any]:
    cpu_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, local_files_only=True
    ).eval().to("cpu")
    cpu_inputs = processor(images=[image], text=[prompt], return_tensors="pt", padding=True)
    gpu_model = None
    gpu_inputs = None
    if torch.cuda.is_available():
        gpu_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True
        ).eval().to("cuda")
        gpu_inputs = processor(
            images=[image], text=[prompt], return_tensors="pt", padding=True
        ).to("cuda")
    cpu = BackendCandidate(
        "grounding-dino-task-local-torch-cpu",
        "cpu",
        lambda: _grounder_forward(cpu_model, cpu_inputs),
        lambda output: torch_observation(model=cpu_model, output=output),
    )
    gpu = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "grounding-dino-task-local-torch-cuda",
            "cuda",
            lambda: _grounder_forward(gpu_model, gpu_inputs),
            lambda output: torch_observation(model=gpu_model, output=output),
            torch.cuda.synchronize,
        )
    backend = select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason="ACCELERATOR_UNAVAILABLE" if gpu is None else None,
        record_path=receipt,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )
    selected = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    if selected is None:
        raise ValueError("SELECTED_GROUNDER_MISSING")
    if backend["selected_device_type"] == "cuda":
        del cpu_model, cpu_inputs
    else:
        del gpu_model, gpu_inputs
    gc.collect()
    return backend, selected


def _clip_image(model: CLIPModel, pixels: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        vision = model.vision_model(pixel_values=pixels.to(device))
        return _normalized(model.visual_projection(vision.pooler_output))


def _select_clip_backend(
    model_path: Path,
    representative: torch.Tensor,
    receipt: Path,
) -> tuple[dict[str, Any], CLIPModel]:
    cpu_model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval().to("cpu")
    gpu_model = (
        CLIPModel.from_pretrained(model_path, local_files_only=True).eval().to("cuda")
        if torch.cuda.is_available()
        else None
    )
    cpu = BackendCandidate(
        "clip-local-effect-batch-torch-cpu",
        "cpu",
        lambda: _clip_image(cpu_model, representative, "cpu"),
        lambda output: torch_observation(model=cpu_model, output=output),
    )
    gpu = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "clip-local-effect-batch-torch-cuda",
            "cuda",
            lambda: _clip_image(gpu_model, representative, "cuda"),
            lambda output: torch_observation(model=gpu_model, output=output),
            torch.cuda.synchronize,
        )
    backend = select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason="ACCELERATOR_UNAVAILABLE" if gpu is None else None,
        record_path=receipt,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )
    selected = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    if selected is None:
        raise ValueError("SELECTED_ENCODER_MISSING")
    if backend["selected_device_type"] == "cuda":
        del cpu_model
    else:
        del gpu_model
    gc.collect()
    return backend, selected


def _iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return float(intersection / max(left_area + right_area - intersection, 1e-12))


def _union(boxes: list[np.ndarray]) -> np.ndarray | None:
    if not boxes:
        return None
    values = np.stack(boxes)
    return np.asarray(
        [values[:, 0].min(), values[:, 1].min(), values[:, 2].max(), values[:, 3].max()],
        dtype=np.float32,
    )


def _choose_regions(
    result: dict[str, Any],
    width: int,
    height: int,
    algorithm: dict[str, Any],
) -> dict[str, Any]:
    gripper_tokens = tuple(str(value).casefold() for value in algorithm["gripper_label_tokens"])
    candidates = []
    for box, score, label in zip(
        result["boxes"], result["scores"], result["text_labels"]
    ):
        box = box.detach().cpu()
        normalized = np.asarray(
            [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
            dtype=np.float32,
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        candidates.append((float(score.detach().cpu()), str(label), normalized))
    grippers = [item for item in candidates if any(token in item[1].casefold() for token in gripper_tokens)]
    tasks = [item for item in candidates if item not in grippers]
    grippers.sort(key=lambda item: item[0], reverse=True)
    tasks.sort(key=lambda item: item[0], reverse=True)
    selected_tasks = []
    for candidate in tasks:
        if all(_iou(candidate[2], kept[2]) <= float(algorithm["task_box_nms_iou"]) for kept in selected_tasks):
            selected_tasks.append(candidate)
        if len(selected_tasks) == int(algorithm["task_box_limit"]):
            break
    task_box = _union([item[2] for item in selected_tasks])
    gripper_box = grippers[0][2] if grippers else None
    return {
        "task_box": task_box,
        "gripper_box": gripper_box,
        "task_confidence": selected_tasks[0][0] if selected_tasks else 0.0,
        "gripper_confidence": grippers[0][0] if grippers else 0.0,
        "task_count": len(selected_tasks),
    }


def _ground_regions(
    rows: list[dict[str, Any]],
    data_root: Path,
    model: Any,
    processor: Any,
    device: str,
    spec: dict[str, Any],
    algorithm: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    jobs = [(row, image_index) for row in rows for image_index in range(6)]
    regions: dict[tuple[str, int], dict[str, Any]] = {}
    batch_size = int(spec["batch_size"])
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        images = []
        prompts = []
        for row, image_index in chunk:
            image = _open_rgb(data_root / row["split"] / row["images"][image_index])
            images.append(image)
            prompts.append(
                str(algorithm["grounding_prompt"]).format(
                    detailed_subtask_name=row["detailed_subtask_name"],
                    task_instruction=row["task_instruction"],
                )
            )
        try:
            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=float(spec["box_threshold"]),
                text_threshold=float(spec["text_threshold"]),
                target_sizes=[(image.height, image.width) for image in images],
            )
            for (row, image_index), image, result in zip(chunk, images, results, strict=True):
                regions[(row["id"], image_index)] = _choose_regions(
                    result, image.width, image.height, algorithm
                )
        finally:
            for image in images:
                image.close()
        completed = min(start + batch_size, len(jobs))
        if completed % (batch_size * 25) == 0 or completed == len(jobs):
            print(json.dumps({"grounded_frames": completed, "total_frames": len(jobs)}), flush=True)
    return regions


def _padded_crop(image: Image.Image, box: np.ndarray | None, padding: float) -> Image.Image:
    if box is None:
        return Image.new("RGB", (32, 32), color=(0, 0, 0))
    x1, y1, x2, y2 = [float(value) for value in box]
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    pixels = (
        int(max(0, math.floor((x1 - dx) * image.width))),
        int(max(0, math.floor((y1 - dy) * image.height))),
        int(min(image.width, math.ceil((x2 + dx) * image.width))),
        int(min(image.height, math.ceil((y2 + dy) * image.height))),
    )
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        return Image.new("RGB", (32, 32), color=(0, 0, 0))
    return image.crop(pixels)


def _embed_regions(
    rows: list[dict[str, Any]],
    data_root: Path,
    regions: dict[tuple[str, int], dict[str, Any]],
    model: CLIPModel,
    processor: Any,
    device: str,
    batch_size: int,
    padding: float,
) -> dict[tuple[str, int, str], np.ndarray]:
    jobs = [(row, index, kind) for row in rows for index in range(6) for kind in ("global", "task", "interaction")]
    embedded: dict[tuple[str, int, str], np.ndarray] = {}
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        images = []
        for row, index, kind in chunk:
            source = _open_rgb(data_root / row["split"] / row["images"][index])
            region = regions[(row["id"], index)]
            if kind == "global":
                value = source.copy()
            elif kind == "task":
                value = _padded_crop(source, region["task_box"], padding)
            else:
                pair = _union(
                    [box for box in (region["task_box"], region["gripper_box"]) if box is not None]
                )
                value = _padded_crop(source, pair if region["task_box"] is not None else None, padding)
            source.close()
            images.append(value)
        try:
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            values = _clip_image(model, pixels, device).detach().cpu().numpy()
            for job, value in zip(chunk, values, strict=True):
                row, index, kind = job
                embedded[(row["id"], index, kind)] = value.astype(np.float32, copy=False)
        finally:
            for image in images:
                image.close()
        completed = min(start + batch_size, len(jobs))
        if completed % (batch_size * 20) == 0 or completed == len(jobs):
            print(json.dumps({"embedded_regions": completed, "total_regions": len(jobs)}), flush=True)
    return embedded


def _encode_text(
    rows: list[dict[str, Any]],
    model: CLIPModel,
    processor: Any,
    device: str,
    algorithm: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray]:
    keys = sorted({(row["task_instruction"], row["detailed_subtask_name"]) for row in rows})
    output = {}
    for start in range(0, len(keys), 32):
        chunk = keys[start : start + 32]
        prompts = []
        for task, subtask in chunk:
            prompts.extend(
                template.format(detailed_subtask_name=subtask, task_instruction=task)
                for template in algorithm["state_prompts"]
            )
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            pooled = model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).pooler_output
            values = _normalized(model.text_projection(pooled)).detach().cpu().numpy()
        for index, key in enumerate(chunk):
            output[key] = _unit(values[index * 2 + 1] - values[index * 2]).astype(np.float32)
    return output


def _area(box: np.ndarray | None) -> float:
    if box is None:
        return 0.0
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _distance(left: np.ndarray | None, right: np.ndarray | None, missing: float) -> float:
    if left is None or right is None:
        return missing
    left_center = np.asarray([(left[0] + left[2]) / 2, (left[1] + left[3]) / 2])
    right_center = np.asarray([(right[0] + right[2]) / 2, (right[1] + right[3]) / 2])
    return float(np.linalg.norm(left_center - right_center) / math.sqrt(2.0))


def _row_features(
    row: dict[str, Any],
    regions: dict[tuple[str, int], dict[str, Any]],
    embeddings: dict[tuple[str, int, str], np.ndarray],
    text_axes: dict[tuple[str, str], np.ndarray],
    algorithm: dict[str, Any],
) -> tuple[np.ndarray, float, bool, dict[str, int]]:
    axis = text_axes[(row["task_instruction"], row["detailed_subtask_name"])]
    per_view = []
    task_pairs = 0
    gripper_pairs = 0
    missing = float(algorithm["missing_distance_value"])
    for view in range(3):
        start, end = view, view + 3
        rs, re = regions[(row["id"], start)], regions[(row["id"], end)]
        progress = []
        for kind in ("global", "task", "interaction"):
            before = float(np.dot(embeddings[(row["id"], start, kind)], axis))
            after = float(np.dot(embeddings[(row["id"], end, kind)], axis))
            progress.append(after - before)
        task_pairs += int(rs["task_box"] is not None and re["task_box"] is not None)
        gripper_pairs += int(rs["gripper_box"] is not None and re["gripper_box"] is not None)
        area_start, area_end = _area(rs["task_box"]), _area(re["task_box"])
        distance_start = _distance(rs["task_box"], rs["gripper_box"], missing)
        distance_end = _distance(re["task_box"], re["gripper_box"], missing)
        per_view.append(
            progress
            + [
                rs["task_confidence"],
                re["task_confidence"],
                rs["gripper_confidence"],
                re["gripper_confidence"],
                rs["task_count"] / max(int(algorithm["task_box_limit"]), 1),
                re["task_count"] / max(int(algorithm["task_box_limit"]), 1),
                area_start,
                area_end,
                area_end - area_start,
                distance_start,
                distance_end,
                distance_end - distance_start,
                _iou(rs["task_box"], re["task_box"]),
            ]
        )
    matrix = np.asarray(per_view, dtype=np.float32)
    features = np.concatenate([matrix.reshape(-1), matrix.mean(axis=0), matrix.std(axis=0)])
    eligible = task_pairs >= 2 and gripper_pairs >= 1
    return features, float(matrix[:, 0].mean()), eligible, {
        "task_view_pairs": task_pairs,
        "gripper_view_pairs": gripper_pairs,
    }


def _load_truth(path: Path, rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    by_index = {row["line_index"]: row for row in rows}
    truth = []
    details = {}
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in by_index:
                continue
            source = json.loads(line)
            value = int(source["execution_reward"])
            if value not in (0, 1):
                raise ValueError(f"UNEXPECTED_TRUTH:{index}:{value}")
            truth.append(value)
            details[by_index[index]["id"]] = {
                "truth": LABELS[value],
                "failure_mode": str(source.get("failure_mode", "")),
            }
    if len(truth) != len(rows):
        raise ValueError("TRUTH_ROW_MISMATCH")
    return np.asarray(truth, dtype=np.int64), details


def _class_metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    recalls = {}
    for label, name in enumerate(LABELS):
        mask = truth == label
        recalls[name] = float(np.mean(predictions[mask] == label)) if np.any(mask) else None
    valid = [value for value in recalls.values() if value is not None]
    return {"balanced_accuracy": float(np.mean(valid)), "class_recall": recalls}


def _evaluate(protocol: dict[str, Any], provider: dict[str, Any], truth_path: Path) -> dict[str, Any]:
    rows = provider["predictions"]
    evaluation_rows = provider["evaluation_rows_for_truth_binding"]
    truth, details = _load_truth(truth_path, evaluation_rows)
    baseline = np.asarray([LABELS.index(row["baseline_prediction"]) for row in rows])
    baseline_all = _class_metrics(truth, baseline)
    known_indices = [index for index, row in enumerate(rows) if row["successor_prediction"] != "UNKNOWN"]
    known_truth = truth[known_indices]
    known_successor = np.asarray([LABELS.index(rows[index]["successor_prediction"]) for index in known_indices])
    known_baseline = baseline[known_indices]
    successor = _class_metrics(known_truth, known_successor) if known_indices else {
        "balanced_accuracy": 0.0,
        "class_recall": {name: 0.0 for name in LABELS},
    }
    baseline_same_known = _class_metrics(known_truth, known_baseline) if known_indices else {
        "balanced_accuracy": 0.0,
        "class_recall": {name: 0.0 for name in LABELS},
    }
    coverage = len(known_indices) / len(rows)
    gain = successor["balanced_accuracy"] - baseline_same_known["balanced_accuracy"]
    gate = protocol["gate"]
    checks = {
        "evaluation_rows": len(rows) == int(gate["evaluation_rows"]),
        "known_coverage": coverage >= float(gate["known_coverage_min"]),
        "successor_selective_balanced_accuracy": successor["balanced_accuracy"] >= float(gate["successor_selective_balanced_accuracy_min"]),
        "balanced_accuracy_gain_on_same_known": gain >= float(gate["balanced_accuracy_gain_on_same_known_min"]),
        "successor_known_class_recall": min(successor["class_recall"].values()) >= float(gate["successor_known_class_recall_min"]),
    }
    for index, row in enumerate(rows):
        row.update(details[row["id"]])
    return {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "decision": (
            "SC46_GUARDIAN_LOCAL_EFFECT_CARRIER_TENSOR_PASS"
            if all(checks.values())
            else "SC46_GUARDIAN_LOCAL_EFFECT_CARRIER_TENSOR_GATE_NOT_MET"
        ),
        "protocol_sha256": provider["protocol_sha256"],
        "provider_sha256": provider["provider_sha256"],
        "evaluation_truth_loaded_after_provider_seal": True,
        "denominators": {"evaluation_rows": len(rows), "known_rows": len(known_indices), "unknown_rows": len(rows) - len(known_indices)},
        "baseline_all": baseline_all,
        "baseline_same_known": baseline_same_known,
        "successor_selective": {**successor, "coverage": coverage, "balanced_accuracy_gain_on_same_known": gain},
        "gate_checks": checks,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--grounder-backend-receipt", type=Path, required=True)
    parser.add_argument("--encoder-backend-receipt", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    _validate_source(protocol, args.data_root)
    roles = _load_roles(protocol, args.data_root)
    all_rows = roles["training"] + roles["calibration"] + roles["evaluation"]
    algorithm = protocol["frozen_algorithm"]

    grounder_spec = protocol["models"]["grounder"]
    grounder_path = _model_path(args.model_root, grounder_spec)
    grounder_processor = AutoProcessor.from_pretrained(grounder_path, local_files_only=True)
    representative_row = roles["training"][0]
    representative_image = _open_rgb(args.data_root / "train" / representative_row["images"][0])
    representative_prompt = str(algorithm["grounding_prompt"]).format(
        detailed_subtask_name=representative_row["detailed_subtask_name"],
        task_instruction=representative_row["task_instruction"],
    )
    try:
        grounder_backend, grounder = _select_grounder_backend(
            grounder_path,
            grounder_processor,
            representative_image,
            representative_prompt,
            args.grounder_backend_receipt,
        )
    finally:
        representative_image.close()
    grounder_device = str(grounder_backend["selected_device_type"])
    regions = _ground_regions(
        all_rows,
        args.data_root,
        grounder,
        grounder_processor,
        grounder_device,
        grounder_spec,
        algorithm,
    )
    del grounder, grounder_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    encoder_spec = protocol["models"]["encoder"]
    encoder_path = _model_path(args.model_root, encoder_spec)
    encoder_processor = AutoProcessor.from_pretrained(encoder_path, local_files_only=True)
    representative_paths = [
        args.data_root / "train" / representative_row["images"][index]
        for index in range(6)
    ][: int(encoder_spec["representative_batch_size"])]
    representative_images = [_open_rgb(path) for path in representative_paths]
    try:
        representative_pixels = encoder_processor(images=representative_images, return_tensors="pt")["pixel_values"]
    finally:
        for image in representative_images:
            image.close()
    encoder_backend, encoder = _select_clip_backend(
        encoder_path, representative_pixels, args.encoder_backend_receipt
    )
    encoder_device = str(encoder_backend["selected_device_type"])
    embeddings = _embed_regions(
        all_rows,
        args.data_root,
        regions,
        encoder,
        encoder_processor,
        encoder_device,
        int(encoder_spec["image_batch_size"]),
        float(algorithm["crop_padding_fraction"]),
    )
    text_axes = _encode_text(all_rows, encoder, encoder_processor, encoder_device, algorithm)

    matrices = {}
    for role, rows in roles.items():
        values = [_row_features(row, regions, embeddings, text_axes, algorithm) for row in rows]
        matrices[role] = {
            "x": np.stack([value[0] for value in values]),
            "baseline": np.asarray([value[1] for value in values]),
            "eligible": np.asarray([value[2] for value in values]),
            "localization": [value[3] for value in values],
        }
    if matrices["training"]["x"].shape[1] != int(algorithm["feature_dimension"]):
        raise ValueError("FEATURE_DIMENSION_MISMATCH")

    train_truth, _ = _load_truth(
        args.data_root / protocol["source"]["splits"]["training"]["metadata_file"],
        roles["training"],
    )
    calibration_truth, _ = _load_truth(
        args.data_root / protocol["source"]["splits"]["calibration"]["metadata_file"],
        roles["calibration"],
    )
    scaler = StandardScaler().fit(matrices["training"]["x"])
    learner_spec = algorithm["learner"]
    learner = LogisticRegression(
        solver=learner_spec["solver"],
        penalty=learner_spec["penalty"],
        C=float(learner_spec["C"]),
        class_weight=learner_spec["class_weight"],
        max_iter=int(learner_spec["max_iter"]),
        tol=float(learner_spec["tol"]),
        random_state=int(learner_spec["random_state"]),
    ).fit(scaler.transform(matrices["training"]["x"]), train_truth)
    calibration_probability = learner.predict_proba(scaler.transform(matrices["calibration"]["x"]))
    nonconformity = 1.0 - calibration_probability[np.arange(len(calibration_truth)), calibration_truth]
    alpha = float(algorithm["conformal"]["alpha"])
    quantile_index = min(len(nonconformity) - 1, math.ceil((len(nonconformity) + 1) * (1.0 - alpha)) - 1)
    conformal_q = float(np.sort(nonconformity)[quantile_index])

    evaluation_probability = learner.predict_proba(scaler.transform(matrices["evaluation"]["x"]))
    predictions = []
    for index, row in enumerate(roles["evaluation"]):
        probability = evaluation_probability[index]
        prediction_set = [LABELS[label] for label in (0, 1) if 1.0 - float(probability[label]) <= conformal_q]
        eligible = bool(matrices["evaluation"]["eligible"][index])
        successor = prediction_set[0] if eligible and len(prediction_set) == 1 else "UNKNOWN"
        baseline_score = float(matrices["evaluation"]["baseline"][index])
        predictions.append(
            {
                "id": row["id"],
                "task_instruction": row["task_instruction"],
                "detailed_subtask_name": row["detailed_subtask_name"],
                "baseline_score": round(baseline_score, 8),
                "baseline_prediction": "success" if baseline_score > 0 else "failure",
                "localization_eligible": eligible,
                "localization": matrices["evaluation"]["localization"][index],
                "successor_probability": {LABELS[label]: round(float(probability[label]), 8) for label in (0, 1)},
                "prediction_set": prediction_set,
                "successor_prediction": successor,
                "factor_tensor": [round(float(value), 8) for value in matrices["evaluation"]["x"][index]],
            }
        )

    provider = {
        "schema_version": 1,
        "provider": "L10-SC46-GUARDIAN-LOCAL-EFFECT-CARRIER-TENSOR-PROVIDER",
        "protocol_sha256": _sha256(args.protocol),
        "source_hashes": {
            role: {
                "metadata_sha256": spec["metadata_sha256"],
                "media_archive_sha256": spec["media_archive_sha256"],
            }
            for role, spec in protocol["source"]["splits"].items()
        },
        "execution_backends": {"grounder": grounder_backend, "encoder": encoder_backend},
        "models": {
            "grounder": {**grounder_spec, "actual_device": grounder_device},
            "encoder": {**encoder_spec, "actual_device": encoder_device},
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "role_counts": {role: len(rows) for role, rows in roles.items()},
        "training_class_counts": dict(Counter(LABELS[value] for value in train_truth.tolist())),
        "calibration_class_counts": dict(Counter(LABELS[value] for value in calibration_truth.tolist())),
        "localization_eligibility_counts": {
            role: {"eligible": int(values["eligible"].sum()), "total": len(values["eligible"])}
            for role, values in matrices.items()
        },
        "scaler_mean": [round(float(value), 8) for value in scaler.mean_],
        "scaler_scale": [round(float(value), 8) for value in scaler.scale_],
        "learner_intercept": [round(float(value), 8) for value in learner.intercept_],
        "learner_coefficients": [[round(float(value), 8) for value in values] for values in learner.coef_],
        "learner_iterations": [int(value) for value in learner.n_iter_],
        "conformal": {"alpha": alpha, "calibration_rows": len(calibration_truth), "quantile_index": quantile_index, "q": conformal_q},
        "evaluation_truth_firewall": protocol["truth_firewall"],
        "predictions": predictions,
        "evaluation_rows_for_truth_binding": roles["evaluation"],
    }
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider["provider_sha256"] = _sha256(args.provider_output)
    result = _evaluate(
        protocol,
        provider,
        args.data_root / protocol["source"]["splits"]["evaluation"]["metadata_file"],
    )
    result.pop("evaluation_rows_for_truth_binding", None)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "baseline_all": result["baseline_all"],
                "baseline_same_known": result["baseline_same_known"],
                "successor_selective": result["successor_selective"],
                "gate_checks": result["gate_checks"],
                "localization_eligibility_counts": provider["localization_eligibility_counts"],
                "conformal_q": conformal_q,
                "backends": {
                    name: {key: backend[key] for key in ("selected_backend", "selected_device_type", "selection_reason")}
                    for name, backend in provider["execution_backends"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
