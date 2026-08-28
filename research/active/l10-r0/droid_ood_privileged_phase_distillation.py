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

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, CLIPModel

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
VIEWS = ("left", "right", "wrist")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _inventory_rows(data_root: Path) -> list[tuple[str, int, str]]:
    paths = [
        data_root / "README.md",
        data_root / "norm_stats_recorded.json",
        data_root / "norm_stats_relabel.json",
    ]
    paths += sorted(
        (data_root / "annotations" / "train").glob("*.json"),
        key=lambda path: int(path.stem),
    )
    paths += sorted(
        (data_root / "annotations" / "val").glob("*.json"),
        key=lambda path: int(path.stem),
    )
    paths += sorted(
        (data_root / "videos" / "train").glob("*.mp4"),
        key=lambda path: int(path.stem),
    )
    paths += sorted(
        (data_root / "videos" / "val").glob("*.mp4"),
        key=lambda path: int(path.stem),
    )
    return [
        (path.relative_to(data_root).as_posix(), path.stat().st_size, _sha256(path))
        for path in paths
    ]


def _inventory_hash(rows: list[tuple[str, int, str]]) -> str:
    payload = "".join(f"{path}\t{size}\t{digest}\n" for path, size, digest in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_source(protocol: dict[str, Any], data_root: Path) -> dict[str, Any]:
    expected = protocol["source"]["selected_inventory"]
    rows = _inventory_rows(data_root)
    observed = {
        "files": len(rows),
        "bytes": sum(row[1] for row in rows),
        "aggregate_sha256": _inventory_hash(rows),
    }
    for key in ("files", "bytes", "aggregate_sha256"):
        if observed[key] != expected[key]:
            raise ValueError(f"SOURCE_INVENTORY_{key.upper()}_MISMATCH")
    for prefix, group_expected in expected["groups"].items():
        group = [row for row in rows if row[0].startswith(prefix + "/")]
        group_observed = {
            "files": len(group),
            "bytes": sum(row[1] for row in group),
            "aggregate_sha256": _inventory_hash(group),
        }
        if group_observed != group_expected:
            raise ValueError(f"SOURCE_GROUP_MISMATCH:{prefix}")
    return observed


def _task_id(episode_id_orig: str) -> str:
    return episode_id_orig.rsplit("_", 1)[0]


def _load_inputs(path: Path, split: str, line_index: int) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "texts",
        "episode_id",
        "episode_id_orig",
        "video_length",
        "video_path",
        "num_cameras",
    }
    clean = {key: source[key] for key in allowed if key in source}
    if set(clean) != allowed:
        raise ValueError(f"MISSING_INPUT_FIELD:{split}:{line_index}")
    if int(clean["num_cameras"]) != 3:
        raise ValueError(f"EXPECTED_THREE_CAMERAS:{split}:{line_index}")
    task_text = str(clean["texts"][0])
    episode_id_orig = str(clean["episode_id_orig"])
    return {
        "id": f"{split}:{line_index:04d}:{episode_id_orig}",
        "split": split,
        "line_index": line_index,
        "episode_id": int(clean["episode_id"]),
        "episode_id_orig": episode_id_orig,
        "task_id": _task_id(episode_id_orig),
        "task": task_text,
        "video_length": int(clean["video_length"]),
        "video_path": str(clean["video_path"]),
    }


def _load_roles(protocol: dict[str, Any], data_root: Path) -> dict[str, list[dict[str, Any]]]:
    source_train = [
        _load_inputs(path, "train", int(path.stem))
        for path in sorted(
            (data_root / "annotations" / "train").glob("*.json"),
            key=lambda path: int(path.stem),
        )
    ]
    source_val = [
        _load_inputs(path, "val", int(path.stem))
        for path in sorted(
            (data_root / "annotations" / "val").glob("*.json"),
            key=lambda path: int(path.stem),
        )
    ]
    roles: dict[str, list[dict[str, Any]]] = {
        "training": [],
        "calibration": [],
        "evaluation": source_val,
    }
    expected_tasks = set(protocol["source"]["native_structure"]["tasks"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source_train:
        grouped.setdefault(row["task_id"], []).append(row)
    if set(grouped) != expected_tasks:
        raise ValueError("TRAIN_TASK_SET_MISMATCH")
    for task in sorted(grouped):
        ranked = sorted(
            grouped[task],
            key=lambda row: (_hash_text(row["episode_id_orig"]), row["episode_id_orig"]),
        )
        if len(ranked) != 50:
            raise ValueError(f"TRAIN_TASK_COUNT_MISMATCH:{task}")
        roles["training"].extend(ranked[:40])
        roles["calibration"].extend(ranked[40:])
    role_spec = protocol["role_freeze"]
    for role in roles:
        if len(roles[role]) != int(role_spec[f"expected_{role}_rows"]):
            raise ValueError(f"ROLE_COUNT_MISMATCH:{role}")
    evaluation_counts = Counter(row["task_id"] for row in source_val)
    if set(evaluation_counts) != expected_tasks or set(evaluation_counts.values()) != {20}:
        raise ValueError("EVALUATION_TASK_STRUCTURE_MISMATCH")
    ids = [row["id"] for values in roles.values() for row in values]
    if len(ids) != len(set(ids)):
        raise ValueError("ROLE_ID_OVERLAP")
    for values in roles.values():
        for row in values:
            video = data_root / row["video_path"]
            if not video.is_file():
                raise ValueError(f"MISSING_VIDEO:{row['id']}")
    return roles


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)


def _encode_image_batch(model: CLIPModel, pixels: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        vision = model.vision_model(pixel_values=pixels.to(device))
        return _normalized(model.visual_projection(vision.pooler_output))


def _encode_text_batch(
    model: CLIPModel, inputs: dict[str, torch.Tensor], device: str
) -> torch.Tensor:
    values = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        pooled = model.text_model(
            input_ids=values["input_ids"], attention_mask=values["attention_mask"]
        ).pooler_output
        return _normalized(model.text_projection(pooled))


def _load_model(model_path: Path, device: str) -> CLIPModel:
    return CLIPModel.from_pretrained(model_path, local_files_only=True).eval().to(device)


def _select_backend(
    model_path: Path,
    representative: torch.Tensor,
    receipt_path: Path,
) -> tuple[dict[str, Any], CLIPModel]:
    cpu_model = _load_model(model_path, "cpu")
    gpu_model = _load_model(model_path, "cuda") if torch.cuda.is_available() else None
    cpu = BackendCandidate(
        "clip-privileged-phase-distillation-torch-cpu",
        "cpu",
        lambda: _encode_image_batch(cpu_model, representative, "cpu"),
        lambda output: torch_observation(model=cpu_model, output=output),
    )
    gpu = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "clip-privileged-phase-distillation-torch-cuda",
            "cuda",
            lambda: _encode_image_batch(gpu_model, representative, "cuda"),
            lambda output: torch_observation(model=gpu_model, output=output),
            torch.cuda.synchronize,
        )
    backend = select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason="ACCELERATOR_UNAVAILABLE" if gpu is None else None,
        record_path=receipt_path,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )
    selected = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    if selected is None:
        raise ValueError("SELECTED_MODEL_MISSING")
    if backend["selected_device_type"] == "cuda":
        del cpu_model
    else:
        del gpu_model
    gc.collect()
    return backend, selected


def _sample_indices(frame_count: int, count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("EMPTY_VIDEO")
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def _read_sampled_views(video_path: Path, count: int) -> list[list[Image.Image]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"VIDEO_OPEN_FAILED:{video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(frame_count, count)
    wanted = set(indices)
    sampled: dict[int, np.ndarray] = {}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                sampled[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    if set(sampled) != wanted:
        raise ValueError(f"VIDEO_SAMPLE_MISSING:{video_path}:{len(sampled)}:{len(wanted)}")
    output = []
    for sample_index in indices:
        frame = sampled[sample_index]
        height, width = frame.shape[:2]
        if (width, height) != (960, 192):
            raise ValueError(f"VIDEO_SHAPE_MISMATCH:{video_path}:{width}x{height}")
        view_width = width // 3
        output.append(
            [
                Image.fromarray(frame[:, view * view_width : (view + 1) * view_width].copy())
                for view in range(3)
            ]
        )
    return output


def _prepare_representative(
    processor: Any, video_path: Path, count: int
) -> torch.Tensor:
    sampled = _read_sampled_views(video_path, 4)
    images = [image for timepoint in sampled for image in timepoint][:count]
    try:
        return processor(images=images, return_tensors="pt")["pixel_values"]
    finally:
        for image in images:
            image.close()


def _embed_videos(
    rows: list[dict[str, Any]],
    data_root: Path,
    model: CLIPModel,
    processor: Any,
    device: str,
    timepoints: int,
    batch_size: int,
) -> np.ndarray:
    output = np.zeros((len(rows), timepoints, 3, 512), dtype=np.float32)
    images: list[Image.Image] = []
    bindings: list[tuple[int, int, int]] = []

    def flush() -> None:
        nonlocal images, bindings
        if not images:
            return
        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
        embeddings = _encode_image_batch(model, pixels, device).detach().cpu().numpy()
        for binding, embedding in zip(bindings, embeddings, strict=True):
            output[binding] = embedding.astype(np.float32, copy=False)
        for image in images:
            image.close()
        images, bindings = [], []

    for row_index, row in enumerate(rows):
        sampled = _read_sampled_views(data_root / row["video_path"], timepoints)
        for time_index, views in enumerate(sampled):
            for view_index, image in enumerate(views):
                images.append(image)
                bindings.append((row_index, time_index, view_index))
                if len(images) >= batch_size:
                    flush()
        completed = row_index + 1
        if completed % 25 == 0 or completed == len(rows):
            print(json.dumps({"embedded_videos": completed, "total_videos": len(rows)}), flush=True)
    flush()
    return output


def _text_axes(
    rows: list[dict[str, Any]],
    model: CLIPModel,
    processor: Any,
    device: str,
    prompts: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    tasks = sorted({row["task"] for row in rows})
    axes = {}
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        text = [template.format(task=task) for task in batch for template in prompts]
        inputs = processor(text=text, return_tensors="pt", padding=True)
        values = _encode_text_batch(model, dict(inputs), device).detach().cpu().numpy()
        for index, task in enumerate(batch):
            axis = values[index * 2 + 1] - values[index * 2]
            axes[task] = (axis / max(float(np.linalg.norm(axis)), 1e-12)).astype(np.float32)
    return axes


def _teacher_input(
    row: dict[str, Any], embeddings: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    timepoints = embeddings.shape[0]
    projections = np.einsum("tvd,d->tv", embeddings, axis)
    normalized_time = np.linspace(0.0, 1.0, timepoints, dtype=np.float32)[:, None]
    return np.concatenate(
        [embeddings.reshape(timepoints, -1), projections, normalized_time], axis=1
    ).astype(np.float32)


def _resample(values: np.ndarray, timepoints: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < 1:
        raise ValueError("EMPTY_PRIVILEGED_SIGNAL")
    source_time = np.linspace(0.0, 1.0, len(values))
    target_time = np.linspace(0.0, 1.0, timepoints)
    return np.interp(target_time, source_time, values).astype(np.float32)


def _load_privileged(
    data_root: Path, row: dict[str, Any], timepoints: int
) -> tuple[np.ndarray, int]:
    path = data_root / "annotations" / "train" / f"{row['line_index']}.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    progress = _resample(np.asarray(source["reward_progress"]), timepoints)
    gripper = _resample(
        np.asarray(source["observation.state.gripper_position"], dtype=np.float32)[:, 0],
        timepoints,
    )
    cartesian = np.asarray(source["observation.state.cartesian_position"], dtype=np.float32)
    xyz = np.stack([_resample(cartesian[:, index], timepoints) for index in range(3)], axis=1)
    closedness = 1.0 - np.clip(gripper, 0.0, 1.0)
    vertical = xyz[:, 2] - xyz[0, 2]
    speed = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.linalg.norm(np.diff(xyz, axis=0), axis=1)]
    )
    opening = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.maximum(np.diff(gripper), 0.0)]
    )
    targets = np.stack([progress, closedness, vertical, speed, opening], axis=1)
    return targets.astype(np.float32), int(source["success"])


def _load_evaluation_truth(data_root: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    truth = []
    for row in rows:
        path = data_root / "annotations" / "val" / f"{row['line_index']}.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        value = int(source["success"])
        if value not in (0, 1):
            raise ValueError(f"UNEXPECTED_EVALUATION_TRUTH:{row['id']}:{value}")
        truth.append(value)
    return np.asarray(truth, dtype=np.int64)


def _sequence_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    steps = np.diff(values)
    x = np.linspace(0.0, 1.0, len(values))
    slope = float(np.polyfit(x, values, 1)[0])
    running_max = np.maximum.accumulate(values)
    maximum_drawdown = float(np.max(running_max - values))
    return np.asarray(
        [
            values[0],
            values[-1],
            values[-1] - values[0],
            values.min(),
            values.max(),
            values.mean(),
            values.std(),
            slope,
            float(np.mean(steps > 0.0)),
            maximum_drawdown,
        ],
        dtype=np.float32,
    )


def _baseline_features(embeddings: np.ndarray, axis: np.ndarray) -> np.ndarray:
    projections = np.einsum("tvd,d->tv", embeddings, axis)
    sequences = [projections[:, view] for view in range(3)] + [projections.mean(axis=1)]
    return np.concatenate([_sequence_stats(sequence) for sequence in sequences])


def _phase_features(predicted: np.ndarray) -> np.ndarray:
    progress, closure, lift, _speed, opening = predicted.T
    closure_index = int(np.argmax(closure))
    lift_index = int(np.argmax(lift))
    opening_index = int(np.argmax(opening))
    denominator = max(len(progress) - 1, 1)
    late_start = max(0, len(progress) * 2 // 3)
    progress_stats = _sequence_stats(progress)
    return np.asarray(
        [
            closure_index / denominator,
            lift_index / denominator,
            opening_index / denominator,
            float(closure_index < lift_index < opening_index),
            float(np.max(lift[closure_index:])),
            float(np.max(opening[lift_index:])),
            float(progress[-1]),
            float(progress.max()),
            float(progress_stats[-1]),
            float(progress[late_start:].mean()),
        ],
        dtype=np.float32,
    )


def _successor_features(baseline: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    teacher_stats = np.concatenate(
        [_sequence_stats(predicted[:, index]) for index in range(predicted.shape[1])]
    )
    return np.concatenate([baseline, teacher_stats, _phase_features(predicted)])


def _learner(spec: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(
        solver=spec["solver"],
        penalty=spec["penalty"],
        C=float(spec["C"]),
        class_weight=spec["class_weight"],
        max_iter=int(spec["max_iter"]),
        tol=float(spec["tol"]),
        random_state=int(spec["random_state"]),
    )


def _class_metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    recalls = {}
    for label, name in enumerate(LABELS):
        mask = truth == label
        recalls[name] = float(np.mean(predictions[mask] == label)) if np.any(mask) else None
    valid = [value for value in recalls.values() if value is not None]
    return {"balanced_accuracy": float(np.mean(valid)), "class_recall": recalls}


def _evaluate(
    protocol: dict[str, Any], provider: dict[str, Any], data_root: Path
) -> dict[str, Any]:
    rows = provider["predictions"]
    bindings = provider["evaluation_bindings"]
    truth = _load_evaluation_truth(data_root, bindings)
    baseline = np.asarray([LABELS.index(row["baseline_prediction"]) for row in rows])
    baseline_all = _class_metrics(truth, baseline)
    known_indices = [
        index for index, row in enumerate(rows) if row["successor_prediction"] != "UNKNOWN"
    ]
    known_truth = truth[known_indices]
    known_baseline = baseline[known_indices]
    known_successor = np.asarray(
        [LABELS.index(rows[index]["successor_prediction"]) for index in known_indices]
    )
    baseline_same_known = (
        _class_metrics(known_truth, known_baseline)
        if known_indices
        else {"balanced_accuracy": 0.0, "class_recall": {name: 0.0 for name in LABELS}}
    )
    successor = (
        _class_metrics(known_truth, known_successor)
        if known_indices
        else {"balanced_accuracy": 0.0, "class_recall": {name: 0.0 for name in LABELS}}
    )
    coverage = len(known_indices) / len(rows)
    gain = successor["balanced_accuracy"] - baseline_same_known["balanced_accuracy"]
    gate = protocol["gate"]
    checks = {
        "evaluation_rows": len(rows) == int(gate["evaluation_rows"]),
        "known_coverage": coverage >= float(gate["known_coverage_min"]),
        "successor_selective_balanced_accuracy": successor["balanced_accuracy"]
        >= float(gate["successor_selective_balanced_accuracy_min"]),
        "balanced_accuracy_gain_on_same_known": gain
        >= float(gate["balanced_accuracy_gain_on_same_known_min"]),
        "successor_known_class_recall": min(successor["class_recall"].values())
        >= float(gate["successor_known_class_recall_min"]),
    }
    by_task = {}
    for task in sorted({row["task_id"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["task_id"] == task]
        task_known = [index for index in indices if index in known_indices]
        task_truth = truth[task_known]
        task_predictions = np.asarray(
            [LABELS.index(rows[index]["successor_prediction"]) for index in task_known]
        )
        by_task[task] = {
            "total": len(indices),
            "known": len(task_known),
            "coverage": len(task_known) / len(indices),
            **(
                _class_metrics(task_truth, task_predictions)
                if task_known
                else {"balanced_accuracy": None, "class_recall": {name: None for name in LABELS}}
            ),
        }
    for index, row in enumerate(rows):
        row["truth"] = LABELS[int(truth[index])]
    return {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "decision": (
            "SC47_DROID_OOD_PRIVILEGED_PHASE_DISTILLATION_PASS"
            if all(checks.values())
            else "SC47_DROID_OOD_PRIVILEGED_PHASE_DISTILLATION_GATE_NOT_MET"
        ),
        "protocol_sha256": provider["protocol_sha256"],
        "provider_sha256": provider["provider_sha256"],
        "evaluation_truth_loaded_after_provider_seal": True,
        "denominators": {
            "evaluation_rows": len(rows),
            "known_rows": len(known_indices),
            "unknown_rows": len(rows) - len(known_indices),
        },
        "baseline_all": baseline_all,
        "baseline_same_known": baseline_same_known,
        "successor_selective": {
            **successor,
            "coverage": coverage,
            "balanced_accuracy_gain_on_same_known": gain,
        },
        "gate_checks": checks,
        "by_task": by_task,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--backend-receipt", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_observed = _validate_source(protocol, args.data_root)
    roles = _load_roles(protocol, args.data_root)
    all_rows = roles["training"] + roles["calibration"] + roles["evaluation"]

    model_spec = protocol["model"]
    model_path = args.model_root / model_spec["local_dir"]
    if _sha256(model_path / model_spec["weights_file"]) != model_spec["weights_sha256"]:
        raise ValueError("MODEL_HASH_MISMATCH")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    representative = _prepare_representative(
        processor,
        args.data_root / all_rows[0]["video_path"],
        int(model_spec["representative_batch_size"]),
    )
    backend, model = _select_backend(model_path, representative, args.backend_receipt)
    device = str(backend["selected_device_type"])
    algorithm = protocol["frozen_algorithm"]
    timepoints = int(algorithm["sampled_timepoints"])
    embeddings = _embed_videos(
        all_rows,
        args.data_root,
        model,
        processor,
        device,
        timepoints,
        int(model_spec["image_batch_size"]),
    )
    axes = _text_axes(
        all_rows,
        model,
        processor,
        device,
        algorithm["state_prompts"],
        int(model_spec["text_batch_size"]),
    )
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.feature_cache,
        ids=np.asarray([row["id"] for row in all_rows]),
        embeddings=embeddings.astype(np.float16),
        tasks=np.asarray(sorted(axes)),
        axes=np.stack([axes[task] for task in sorted(axes)]).astype(np.float32),
    )
    feature_cache_sha256 = _sha256(args.feature_cache)

    offsets = {
        "training": (0, len(roles["training"])),
        "calibration": (
            len(roles["training"]),
            len(roles["training"]) + len(roles["calibration"]),
        ),
        "evaluation": (
            len(roles["training"]) + len(roles["calibration"]),
            len(all_rows),
        ),
    }
    teacher_inputs_by_role = {}
    baseline_by_role = {}
    for role, (start, end) in offsets.items():
        role_embeddings = embeddings[start:end]
        teacher_inputs_by_role[role] = np.stack(
            [
                _teacher_input(row, value, axes[row["task"]])
                for row, value in zip(roles[role], role_embeddings, strict=True)
            ]
        )
        baseline_by_role[role] = np.stack(
            [
                _baseline_features(value, axes[row["task"]])
                for row, value in zip(roles[role], role_embeddings, strict=True)
            ]
        )
    if teacher_inputs_by_role["training"].shape[2] != 1540:
        raise ValueError("TEACHER_INPUT_DIMENSION_MISMATCH")
    if baseline_by_role["training"].shape[1] != 40:
        raise ValueError("BASELINE_DIMENSION_MISMATCH")

    privileged = {}
    success_truth = {}
    for role in ("training", "calibration"):
        values = [
            _load_privileged(args.data_root, row, timepoints) for row in roles[role]
        ]
        privileged[role] = np.stack([value[0] for value in values])
        success_truth[role] = np.asarray([value[1] for value in values], dtype=np.int64)

    teacher_x_scaler = StandardScaler().fit(
        teacher_inputs_by_role["training"].reshape(-1, 1540)
    )
    teacher_y_scaler = StandardScaler().fit(privileged["training"].reshape(-1, 5))
    teacher_spec = algorithm["teacher_regressor"]
    teacher = Ridge(
        alpha=float(teacher_spec["alpha"]),
        fit_intercept=bool(teacher_spec["fit_intercept"]),
    ).fit(
        teacher_x_scaler.transform(
            teacher_inputs_by_role["training"].reshape(-1, 1540)
        ),
        teacher_y_scaler.transform(privileged["training"].reshape(-1, 5)),
    )

    predicted_teacher = {}
    for role in roles:
        shape = teacher_inputs_by_role[role].shape
        scaled = teacher.predict(
            teacher_x_scaler.transform(teacher_inputs_by_role[role].reshape(-1, 1540))
        )
        predicted_teacher[role] = teacher_y_scaler.inverse_transform(scaled).reshape(
            shape[0], shape[1], 5
        )
    calibration_teacher_r2 = {
        name: float(
            r2_score(privileged["calibration"][:, :, index].reshape(-1), predicted_teacher["calibration"][:, :, index].reshape(-1))
        )
        for index, name in enumerate(algorithm["privileged_teacher_targets"])
    }

    successor_by_role = {
        role: np.stack(
            [
                _successor_features(baseline, predicted)
                for baseline, predicted in zip(
                    baseline_by_role[role], predicted_teacher[role], strict=True
                )
            ]
        )
        for role in roles
    }
    if successor_by_role["training"].shape[1] != 100:
        raise ValueError("SUCCESSOR_DIMENSION_MISMATCH")

    baseline_scaler = StandardScaler().fit(baseline_by_role["training"])
    successor_scaler = StandardScaler().fit(successor_by_role["training"])
    baseline_learner = _learner(algorithm["baseline_learner"]).fit(
        baseline_scaler.transform(baseline_by_role["training"]),
        success_truth["training"],
    )
    successor_learner = _learner(algorithm["successor_learner"]).fit(
        successor_scaler.transform(successor_by_role["training"]),
        success_truth["training"],
    )

    calibration_probability = successor_learner.predict_proba(
        successor_scaler.transform(successor_by_role["calibration"])
    )
    calibration_truth = success_truth["calibration"]
    nonconformity = 1.0 - calibration_probability[
        np.arange(len(calibration_truth)), calibration_truth
    ]
    alpha = float(algorithm["conformal"]["alpha"])
    quantile_index = min(
        len(nonconformity) - 1,
        math.ceil((len(nonconformity) + 1) * (1.0 - alpha)) - 1,
    )
    conformal_q = float(np.sort(nonconformity)[quantile_index])

    baseline_probability = baseline_learner.predict_proba(
        baseline_scaler.transform(baseline_by_role["evaluation"])
    )
    successor_probability = successor_learner.predict_proba(
        successor_scaler.transform(successor_by_role["evaluation"])
    )
    predictions = []
    for index, row in enumerate(roles["evaluation"]):
        probability = successor_probability[index]
        prediction_set = [
            LABELS[label]
            for label in (0, 1)
            if 1.0 - float(probability[label]) <= conformal_q
        ]
        successor_prediction = prediction_set[0] if len(prediction_set) == 1 else "UNKNOWN"
        predictions.append(
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "task": row["task"],
                "baseline_probability": {
                    LABELS[label]: round(float(baseline_probability[index, label]), 8)
                    for label in (0, 1)
                },
                "baseline_prediction": LABELS[int(np.argmax(baseline_probability[index]))],
                "successor_probability": {
                    LABELS[label]: round(float(probability[label]), 8) for label in (0, 1)
                },
                "prediction_set": prediction_set,
                "successor_prediction": successor_prediction,
                "baseline_features": [
                    round(float(value), 8) for value in baseline_by_role["evaluation"][index]
                ],
                "distilled_teacher_sequences": {
                    name: [round(float(value), 8) for value in predicted_teacher["evaluation"][index, :, channel]]
                    for channel, name in enumerate(algorithm["privileged_teacher_targets"])
                },
                "successor_features": [
                    round(float(value), 8) for value in successor_by_role["evaluation"][index]
                ],
            }
        )

    provider = {
        "schema_version": 1,
        "provider": "L10-SC47-DROID-OOD-PRIVILEGED-PHASE-DISTILLATION-PROVIDER",
        "protocol_sha256": _sha256(args.protocol),
        "source": {
            "dataset_id": protocol["source"]["dataset_id"],
            "revision": protocol["source"]["revision"],
            "observed_inventory": source_observed,
        },
        "feature_cache_sha256": feature_cache_sha256,
        "execution_backend": backend,
        "model": {
            **model_spec,
            "actual_device": device,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "role_counts": {role: len(rows) for role, rows in roles.items()},
        "role_task_counts": {
            role: dict(sorted(Counter(row["task_id"] for row in rows).items()))
            for role, rows in roles.items()
        },
        "training_class_counts": dict(
            Counter(LABELS[value] for value in success_truth["training"].tolist())
        ),
        "calibration_class_counts": dict(
            Counter(LABELS[value] for value in success_truth["calibration"].tolist())
        ),
        "calibration_teacher_r2": calibration_teacher_r2,
        "teacher": {
            "input_scaler_mean": [round(float(value), 8) for value in teacher_x_scaler.mean_],
            "input_scaler_scale": [round(float(value), 8) for value in teacher_x_scaler.scale_],
            "target_scaler_mean": [round(float(value), 8) for value in teacher_y_scaler.mean_],
            "target_scaler_scale": [round(float(value), 8) for value in teacher_y_scaler.scale_],
            "intercept": [round(float(value), 8) for value in teacher.intercept_],
            "coefficients": [
                [round(float(value), 8) for value in channel] for channel in teacher.coef_
            ],
        },
        "baseline_learner": {
            "scaler_mean": [round(float(value), 8) for value in baseline_scaler.mean_],
            "scaler_scale": [round(float(value), 8) for value in baseline_scaler.scale_],
            "intercept": [round(float(value), 8) for value in baseline_learner.intercept_],
            "coefficients": [
                [round(float(value), 8) for value in channel]
                for channel in baseline_learner.coef_
            ],
        },
        "successor_learner": {
            "scaler_mean": [round(float(value), 8) for value in successor_scaler.mean_],
            "scaler_scale": [round(float(value), 8) for value in successor_scaler.scale_],
            "intercept": [round(float(value), 8) for value in successor_learner.intercept_],
            "coefficients": [
                [round(float(value), 8) for value in channel]
                for channel in successor_learner.coef_
            ],
        },
        "conformal": {
            "alpha": alpha,
            "calibration_rows": len(calibration_truth),
            "quantile_index": quantile_index,
            "q": conformal_q,
        },
        "evaluation_truth_firewall": protocol["role_freeze"]["evaluation_truth_firewall"],
        "predictions": predictions,
        "evaluation_bindings": [
            {
                "id": row["id"],
                "line_index": row["line_index"],
                "task_id": row["task_id"],
            }
            for row in roles["evaluation"]
        ],
    }
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider["provider_sha256"] = _sha256(args.provider_output)
    result = _evaluate(protocol, provider, args.data_root)
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
                "by_task": result["by_task"],
                "calibration_teacher_r2": calibration_teacher_r2,
                "conformal_q": conformal_q,
                "backend": {
                    key: backend[key]
                    for key in ("selected_backend", "selected_device_type", "selection_reason")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
