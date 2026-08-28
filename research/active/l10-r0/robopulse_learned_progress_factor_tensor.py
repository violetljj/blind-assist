from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
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
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from research_backend import (  # noqa: E402
    BackendCandidate,
    DeviceObservation,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)

LABELS = ("regression", "progress")
VIEWS = ((2, 5, "front"), (3, 6, "left_wrist"), (4, 7, "right_wrist"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _long_path(path: Path) -> str:
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def _is_file(path: Path) -> bool:
    return Path(_long_path(path)).is_file()


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(_long_path(path)) as image:
        return image.convert("RGB")


def _load_inputs(annotation_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    for source in payload:
        required = {"id", "task", "image", "image_dataset"}
        if not required.issubset(source):
            raise ValueError("MISSING_INPUT_FIELD")
        images = [str(path) for path in source["image"]]
        if len(images) != 8:
            raise ValueError("EXPECTED_EIGHT_IMAGES")
        rows.append(
            {
                "id": str(source["id"]),
                "task": str(source["task"]),
                "image": images,
                "image_dataset": str(source["image_dataset"]),
            }
        )
    return rows


def _load_truth(annotation_path: Path, allowed_ids: set[str]) -> dict[str, int]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    truth = {}
    for row in payload:
        row_id = str(row["id"])
        if row_id not in allowed_ids:
            continue
        value = str(row["conversations"][1]["value"])
        if value == "<score>+1</score>":
            truth[row_id] = 1
        elif value == "<score>-1</score>":
            truth[row_id] = 0
        else:
            raise ValueError(f"UNEXPECTED_TRUTH:{row_id}:{value}")
    if set(truth) != allowed_ids:
        raise ValueError("TRUTH_ID_MISMATCH")
    return truth


def _freeze_roles(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["image_dataset"], []).append(row)
    role_spec = protocol["role_freeze"]
    expected_sources = {
        *role_spec["training_sources"],
        *role_spec["calibration_sources"],
        *role_spec["evaluation_sources"],
    }
    if set(grouped) != expected_sources:
        raise ValueError("SOURCE_ROLE_SET_MISMATCH")
    roles = {"training": [], "calibration": [], "evaluation": []}
    source_to_role = {
        **{source: "training" for source in role_spec["training_sources"]},
        **{source: "calibration" for source in role_spec["calibration_sources"]},
        **{source: "evaluation" for source in role_spec["evaluation_sources"]},
    }
    for source in sorted(grouped):
        ranked = sorted(grouped[source], key=lambda row: (_hash_text(row["id"]), row["id"]))
        if len(ranked) != int(protocol["source"]["rows_per_source"]):
            raise ValueError(f"SOURCE_ROW_COUNT_MISMATCH:{source}")
        roles[source_to_role[source]].extend(ranked[10:])
    for role in roles:
        expected = int(role_spec[f"expected_{role}_rows"])
        if len(roles[role]) != expected:
            raise ValueError(f"ROLE_ROW_COUNT_MISMATCH:{role}")
    ids = [row["id"] for values in roles.values() for row in values]
    if len(ids) != len(set(ids)):
        raise ValueError("ROLE_ID_OVERLAP")
    return roles


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)


def _encode_image_batch(model: CLIPModel, pixel_values: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        vision = model.vision_model(pixel_values=pixel_values.to(device))
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


def _observation(model: CLIPModel, output: torch.Tensor) -> DeviceObservation:
    return torch_observation(model=model, output=output)


def _select_backend(
    cpu_model: CLIPModel,
    gpu_model: CLIPModel | None,
    representative: torch.Tensor,
    receipt_path: Path,
) -> dict[str, Any]:
    cpu = BackendCandidate(
        "clip-global-batch-torch-cpu",
        "cpu",
        lambda: _encode_image_batch(cpu_model, representative, "cpu"),
        lambda output: _observation(cpu_model, output),
    )
    gpu = None
    cpu_reason = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "clip-global-batch-torch-cuda",
            "cuda",
            lambda: _encode_image_batch(gpu_model, representative, "cuda"),
            lambda output: _observation(gpu_model, output),
            torch.cuda.synchronize,
        )
    else:
        cpu_reason = "ACCELERATOR_UNAVAILABLE"
    return select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason=cpu_reason,
        record_path=receipt_path,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )


def _load_model(model_spec: dict[str, Any], cache_dir: Path, device: str) -> CLIPModel:
    return CLIPModel.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=cache_dir,
    ).eval().to(device)


def _prepare_pixels(processor: Any, paths: list[Path]) -> torch.Tensor:
    images = [_open_rgb(path) for path in paths]
    try:
        return processor(images=images, return_tensors="pt")["pixel_values"]
    finally:
        for image in images:
            image.close()


def _embed_media(
    model: CLIPModel,
    processor: Any,
    paths: list[str],
    media_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        pixels = _prepare_pixels(processor, [media_root / path for path in batch])
        embeddings = _encode_image_batch(model, pixels, device).detach().cpu().numpy()
        for relative, embedding in zip(batch, embeddings, strict=True):
            output[relative] = embedding.astype(np.float32, copy=False)
        completed = min(start + batch_size, len(paths))
        if completed % (batch_size * 10) == 0 or completed == len(paths):
            print(json.dumps({"embedded_images": completed, "total_images": len(paths)}), flush=True)
    return output


def _embed_tasks(
    model: CLIPModel,
    processor: Any,
    tasks: list[str],
    algorithm: dict[str, Any],
    device: str,
    task_batch_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(tasks), task_batch_size):
        batch = tasks[start : start + task_batch_size]
        prompts = []
        for task in batch:
            prompts.extend(
                [
                    str(algorithm["task_start_prompt"]).format(task=task),
                    str(algorithm["task_end_prompt"]).format(task=task),
                ]
            )
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        values = _encode_text_batch(model, dict(inputs), device).detach().cpu().numpy()
        for index, task in enumerate(batch):
            output[task] = (
                values[index * 2].astype(np.float32),
                values[index * 2 + 1].astype(np.float32),
            )
    return output


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("ZERO_FACTOR_AXIS")
    return value / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(np.dot(left, right) / denom)


def _factor_tensor(
    row: dict[str, Any],
    image_embeddings: dict[str, np.ndarray],
    task_embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float, dict[str, list[int]]]:
    images = [image_embeddings[path] for path in row["image"]]
    start, end = images[0], images[1]
    text_start, text_end = task_embeddings[row["task"]]
    visual_axis = _unit(end - start)
    text_axis = _unit(text_end - text_start)
    deltas = []
    per_view = []
    for before_index, after_index, _ in VIEWS:
        before, after = images[before_index], images[after_index]
        delta = after - before
        deltas.append(delta)
        per_view.append(
            [
                float(np.dot(delta, visual_axis)),
                float(np.dot(delta, text_axis)),
                float(np.dot(after, end) - np.dot(before, end)),
                float(np.dot(before, start) - np.dot(after, start)),
                float(np.dot(after, text_end) - np.dot(before, text_end)),
            ]
        )
    features: list[float] = []
    groups: dict[str, list[int]] = {}

    def add_group(name: str, values: list[float]) -> None:
        groups[name] = list(range(len(features), len(features) + len(values)))
        features.extend(values)

    add_group("effect_carrier", per_view[0])
    add_group("actuator_contact", per_view[1] + per_view[2])
    before_front, after_front = images[2], images[5]
    add_group(
        "spatial_orientation",
        [
            float(np.linalg.norm(deltas[0])),
            float(np.dot(before_front, end)),
            float(np.dot(after_front, end)),
            float(np.dot(before_front, start)),
            float(np.dot(after_front, start)),
        ],
    )
    visual_scores = [values[0] for values in per_view]
    text_scores = [values[1] for values in per_view]
    add_group(
        "conflict_integrity",
        [
            float(np.std(visual_scores)),
            float(np.std(text_scores)),
            float(min(visual_scores)),
            float(min(text_scores)),
            _cosine(deltas[0], deltas[1]),
            _cosine(deltas[0], deltas[2]),
            _cosine(deltas[1], deltas[2]),
        ],
    )
    before_views = [images[index] for index in (2, 3, 4)]
    after_views = [images[index] for index in (5, 6, 7)]
    distance_reductions = [
        float(np.linalg.norm(before - end) - np.linalg.norm(after - end))
        for before, after in zip(before_views, after_views, strict=True)
    ]
    add_group(
        "handoff_distance",
        [
            distance_reductions[0],
            float(np.mean(distance_reductions)),
            float(np.dot(after_front, end)),
            float(np.mean([np.dot(after, end) for after in after_views])),
            float(np.mean([np.dot(before, end) for before in before_views])),
        ],
    )
    tensor = np.asarray(features, dtype=np.float32)
    if tensor.shape != (32,) or not np.isfinite(tensor).all():
        raise ValueError(f"INVALID_FACTOR_TENSOR:{tensor.shape}")
    baseline_score = float(np.mean(visual_scores))
    return tensor, baseline_score, groups


def _matrix(
    rows: list[dict[str, Any]],
    image_embeddings: dict[str, np.ndarray],
    task_embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    features, baselines = [], []
    expected_groups = None
    for row in rows:
        tensor, baseline, groups = _factor_tensor(row, image_embeddings, task_embeddings)
        features.append(tensor)
        baselines.append(baseline)
        if expected_groups is None:
            expected_groups = groups
        elif groups != expected_groups:
            raise ValueError("FACTOR_GROUP_LAYOUT_CHANGED")
    if expected_groups is None:
        raise ValueError("EMPTY_ROLE")
    return np.stack(features), np.asarray(baselines), expected_groups


def _balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for label in (0, 1):
        members = truth == label
        if not members.any():
            raise ValueError(f"MISSING_CLASS:{label}")
        recalls.append(float(np.mean(prediction[members] == label)))
    return float(np.mean(recalls))


def _evaluate(
    protocol: dict[str, Any],
    provider: dict[str, Any],
    annotation_path: Path,
) -> dict[str, Any]:
    predictions = provider["predictions"]
    truth_map = _load_truth(annotation_path, {row["id"] for row in predictions})
    rows = []
    for prediction in predictions:
        truth = int(truth_map[prediction["id"]])
        successor = prediction["successor_prediction"]
        rows.append(
            {
                "id": prediction["id"],
                "image_dataset": prediction["image_dataset"],
                "truth": LABELS[truth],
                "baseline_prediction": prediction["baseline_prediction"],
                "successor_prediction": successor,
            }
        )
    known = [row for row in rows if row["successor_prediction"] != "UNKNOWN"]
    truth_all = np.asarray([LABELS.index(row["truth"]) for row in rows])
    baseline_all = np.asarray([LABELS.index(row["baseline_prediction"]) for row in rows])
    known_truth = np.asarray([LABELS.index(row["truth"]) for row in known])
    known_successor = np.asarray([LABELS.index(row["successor_prediction"]) for row in known])
    known_baseline = np.asarray([LABELS.index(row["baseline_prediction"]) for row in known])
    coverage = len(known) / len(rows)
    selective_balanced = _balanced_accuracy(known_truth, known_successor) if known else 0.0
    baseline_known_balanced = _balanced_accuracy(known_truth, known_baseline) if known else 0.0
    gain = selective_balanced - baseline_known_balanced
    by_source = {}
    for source in protocol["role_freeze"]["evaluation_sources"]:
        subset = [row for row in rows if row["image_dataset"] == source]
        source_known = [row for row in subset if row["successor_prediction"] != "UNKNOWN"]
        by_source[source] = {
            "rows": len(subset),
            "known": len(source_known),
            "known_coverage": len(source_known) / len(subset),
            "selective_accuracy": (
                sum(row["successor_prediction"] == row["truth"] for row in source_known)
                / len(source_known)
                if source_known
                else 0.0
            ),
        }
    gate = protocol["frozen_gate"]
    if len(rows) < int(gate["minimum_evaluation_rows"]):
        decision = protocol["decision_labels"]["not_evaluable"]
    elif (
        coverage >= float(gate["minimum_known_coverage"])
        and selective_balanced >= float(gate["minimum_selective_balanced_accuracy"])
        and gain >= float(gate["minimum_balanced_accuracy_gain_over_baseline_on_known"])
        and all(
            values["known_coverage"] >= float(gate["minimum_per_source_known_coverage"])
            for values in by_source.values()
        )
    ):
        decision = protocol["decision_labels"]["pass"]
    else:
        decision = protocol["decision_labels"]["fail"]
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": provider["protocol_sha256"],
        "provider_sha256": provider["provider_sha256"],
        "evaluation_truth_loaded_after_provider_seal": True,
        "denominators": {
            "evaluation_rows": len(rows),
            "progress_truth": int(np.sum(truth_all == 1)),
            "regression_truth": int(np.sum(truth_all == 0)),
            "known_rows": len(known),
            "unknown_rows": len(rows) - len(known),
        },
        "baseline_all": {
            "correct": int(np.sum(baseline_all == truth_all)),
            "accuracy": float(np.mean(baseline_all == truth_all)),
            "balanced_accuracy": _balanced_accuracy(truth_all, baseline_all),
        },
        "successor_selective": {
            "correct": sum(row["successor_prediction"] == row["truth"] for row in known),
            "known_coverage": coverage,
            "accuracy": (
                sum(row["successor_prediction"] == row["truth"] for row in known) / len(known)
                if known
                else 0.0
            ),
            "balanced_accuracy": selective_balanced,
            "baseline_balanced_accuracy_on_same_known_rows": baseline_known_balanced,
            "balanced_accuracy_gain_on_same_known_rows": gain,
            "correct_known_per_total": (
                sum(row["successor_prediction"] == row["truth"] for row in known) / len(rows)
            ),
        },
        "by_source": by_source,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--backend-receipt", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = protocol["source"]
    annotation_path = args.data_root / source["annotation_file"]
    archive_path = args.data_root / source["media_archive"]
    if annotation_path.stat().st_size != int(source["annotation_size_bytes"]):
        raise ValueError("ANNOTATION_SIZE_MISMATCH")
    if _sha256(annotation_path) != source["annotation_sha256"]:
        raise ValueError("ANNOTATION_HASH_MISMATCH")
    if archive_path.stat().st_size != int(source["media_archive_size_bytes"]):
        raise ValueError("MEDIA_ARCHIVE_SIZE_MISMATCH")
    if _sha256(archive_path) != source["media_archive_sha256"]:
        raise ValueError("MEDIA_ARCHIVE_HASH_MISMATCH")

    roles = _freeze_roles(_load_inputs(annotation_path), protocol)
    all_rows = roles["training"] + roles["calibration"] + roles["evaluation"]
    media_paths = sorted({path for row in all_rows for path in row["image"]})
    missing = [path for path in media_paths if not _is_file(args.data_root / path)]
    if missing:
        raise ValueError(f"MISSING_MEDIA:{len(missing)}:{missing[:3]}")

    model_spec = protocol["model"]
    processor = AutoProcessor.from_pretrained(
        model_spec["model_id"], revision=model_spec["revision"], cache_dir=args.model_cache
    )
    representative_paths = [args.data_root / path for path in media_paths[: int(model_spec["representative_batch_size"])]]
    representative = _prepare_pixels(processor, representative_paths)
    cpu_model = _load_model(model_spec, args.model_cache, "cpu")
    gpu_model = _load_model(model_spec, args.model_cache, "cuda") if torch.cuda.is_available() else None
    backend = _select_backend(cpu_model, gpu_model, representative, args.backend_receipt)
    device = str(backend["selected_device_type"])
    selected_model = gpu_model if device == "cuda" else cpu_model
    if selected_model is None:
        raise ValueError("SELECTED_MODEL_MISSING")
    if device == "cuda":
        del cpu_model
    else:
        del gpu_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()

    image_embeddings = _embed_media(
        selected_model,
        processor,
        media_paths,
        args.data_root,
        device,
        int(model_spec["image_batch_size"]),
    )
    tasks = sorted({row["task"] for row in all_rows})
    task_embeddings = _embed_tasks(
        selected_model,
        processor,
        tasks,
        protocol["frozen_algorithm"],
        device,
        int(model_spec["text_task_batch_size"]),
    )
    matrices = {
        role: _matrix(rows, image_embeddings, task_embeddings)
        for role, rows in roles.items()
    }
    train_x, train_baseline, factor_groups = matrices["training"]
    calibration_x, calibration_baseline, calibration_groups = matrices["calibration"]
    evaluation_x, evaluation_baseline, evaluation_groups = matrices["evaluation"]
    if not (factor_groups == calibration_groups == evaluation_groups):
        raise ValueError("ROLE_FACTOR_LAYOUT_MISMATCH")
    if train_x.shape[1] != int(protocol["frozen_algorithm"]["feature_dimension"]):
        raise ValueError("FEATURE_DIMENSION_MISMATCH")

    training_ids = {row["id"] for row in roles["training"]}
    calibration_ids = {row["id"] for row in roles["calibration"]}
    train_truth_map = _load_truth(annotation_path, training_ids)
    calibration_truth_map = _load_truth(annotation_path, calibration_ids)
    train_y = np.asarray([train_truth_map[row["id"]] for row in roles["training"]])
    calibration_y = np.asarray(
        [calibration_truth_map[row["id"]] for row in roles["calibration"]]
    )
    del train_truth_map, calibration_truth_map

    learner_spec = protocol["frozen_algorithm"]["learner"]
    scaler = StandardScaler().fit(train_x)
    learner = LogisticRegression(
        solver=learner_spec["solver"],
        penalty=learner_spec["penalty"],
        C=float(learner_spec["C"]),
        class_weight=learner_spec["class_weight"],
        max_iter=int(learner_spec["max_iter"]),
        tol=float(learner_spec["tol"]),
        random_state=int(learner_spec["random_state"]),
    ).fit(scaler.transform(train_x), train_y)
    calibration_probability = learner.predict_proba(scaler.transform(calibration_x))
    nonconformity = 1.0 - calibration_probability[np.arange(len(calibration_y)), calibration_y]
    alpha = float(protocol["frozen_algorithm"]["conformal"]["alpha"])
    quantile_index = min(
        len(nonconformity) - 1, math.ceil((len(nonconformity) + 1) * (1.0 - alpha)) - 1
    )
    conformal_q = float(np.sort(nonconformity)[quantile_index])

    evaluation_probability = learner.predict_proba(scaler.transform(evaluation_x))
    predictions = []
    for index, row in enumerate(roles["evaluation"]):
        probability = evaluation_probability[index]
        prediction_set = [
            LABELS[label]
            for label in (0, 1)
            if 1.0 - float(probability[label]) <= conformal_q
        ]
        successor = prediction_set[0] if len(prediction_set) == 1 else "UNKNOWN"
        predictions.append(
            {
                "id": row["id"],
                "task": row["task"],
                "image_dataset": row["image_dataset"],
                "baseline_score": round(float(evaluation_baseline[index]), 8),
                "baseline_prediction": (
                    "progress" if evaluation_baseline[index] > 0 else "regression"
                ),
                "successor_probability": {
                    "regression": round(float(probability[0]), 8),
                    "progress": round(float(probability[1]), 8),
                },
                "prediction_set": prediction_set,
                "successor_prediction": successor,
                "factor_tensor": [round(float(value), 8) for value in evaluation_x[index]],
            }
        )

    provider = {
        "schema_version": 1,
        "provider": "L10-SC45-ROBOPULSE-LEARNED-PROGRESS-FACTOR-TENSOR-PROVIDER",
        "protocol_sha256": _sha256(args.protocol),
        "annotation_sha256": _sha256(annotation_path),
        "media_archive_sha256": _sha256(archive_path),
        "execution_backend": backend,
        "model": {
            **model_spec,
            "actual_device": device,
            "framework_version": torch.__version__,
            "python": platform.python_version(),
        },
        "role_counts": {role: len(rows) for role, rows in roles.items()},
        "training_class_counts": dict(Counter(LABELS[value] for value in train_y.tolist())),
        "calibration_class_counts": dict(
            Counter(LABELS[value] for value in calibration_y.tolist())
        ),
        "factor_groups": factor_groups,
        "scaler_mean": [round(float(value), 8) for value in scaler.mean_],
        "scaler_scale": [round(float(value), 8) for value in scaler.scale_],
        "learner_intercept": [round(float(value), 8) for value in learner.intercept_],
        "learner_coefficients": [
            [round(float(value), 8) for value in row] for row in learner.coef_
        ],
        "learner_iterations": [int(value) for value in learner.n_iter_],
        "conformal": {
            "alpha": alpha,
            "calibration_rows": len(calibration_y),
            "quantile_index": quantile_index,
            "q": conformal_q,
        },
        "training_baseline_sign_counts": {
            "progress": int(np.sum(train_baseline > 0)),
            "regression": int(np.sum(train_baseline <= 0)),
        },
        "calibration_baseline_sign_counts": {
            "progress": int(np.sum(calibration_baseline > 0)),
            "regression": int(np.sum(calibration_baseline <= 0)),
        },
        "evaluation_truth_firewall": protocol["role_freeze"]["evaluation_truth_firewall"],
        "predictions": predictions,
    }
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider["provider_sha256"] = _sha256(args.provider_output)
    result = _evaluate(protocol, provider, annotation_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "baseline_all": result["baseline_all"],
                "successor_selective": result["successor_selective"],
                "by_source": result["by_source"],
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
