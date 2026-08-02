#!/usr/bin/env python3
"""Screen THOR-MAGNI RGB history against current-only visual features."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small


SCHEMA = "blindassist_hftf_stage_c_d8_thor_magni_rgb_history_screen_v0"
SEED = 17
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
DEFAULT_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d8-thor-magni-local-route-supervision-v0/samples.jsonl"
)
DEFAULT_PRETRAINED = Path(
    "artifacts.local/models/hftf/torch/hub/checkpoints/"
    "mobilenet_v3_small-047dcff4.pth"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def history_design(features: np.ndarray) -> np.ndarray:
    """Use current appearance plus three fixed temporal residual summaries."""
    if features.ndim != 3 or features.shape[1] != 5:
        raise ValueError("Expected features with shape [sample,5,channel]")
    current = features[:, -1]
    return np.concatenate(
        (
            current,
            current - features[:, 0],
            current - features[:, -2],
            np.std(features, axis=1),
        ),
        axis=1,
    )


def binary_metrics(
    target: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | None]:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if target.shape != score.shape:
        raise ValueError("Binary target/score shape mismatch")
    if len(np.unique(target)) < 2:
        return {"auroc": None, "average_precision": None}
    ranks = rankdata(score)
    positive_count = int(np.sum(target))
    negative_count = len(target) - positive_count
    auroc = (
        np.sum(ranks[target == 1])
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_target = target[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(target):
        end = start + 1
        while (
            end < len(target)
            and sorted_score[end] == sorted_score[start]
        ):
            end += 1
        group = sorted_target[start:end]
        true_positive += int(np.sum(group))
        false_positive += len(group) - int(np.sum(group))
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return {
        "auroc": float(auroc),
        "average_precision": float(average_precision),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks


def spearman(target: np.ndarray, prediction: np.ndarray) -> float | None:
    target_rank = rankdata(target)
    prediction_rank = rankdata(prediction)
    if np.std(target_rank) == 0.0 or np.std(prediction_rank) == 0.0:
        return None
    return float(np.corrcoef(target_rank, prediction_rank)[0, 1])


def standardize(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale[scale < 1e-6] = 1.0
    return (
        ((train_x - mean) / scale).astype(np.float32),
        ((test_x - mean) / scale).astype(np.float32),
    )


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    device: torch.device,
    alpha: float = 10.0,
) -> np.ndarray:
    target = np.asarray(train_y, dtype=np.float32)
    one_dimensional = target.ndim == 1
    if one_dimensional:
        target = target[:, None]
    target_mean = np.mean(target, axis=0, keepdims=True)
    centered_target = target - target_mean
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(centered_target).to(device)
    test = torch.from_numpy(test_x).to(device)
    if train_x.shape[1] <= train_x.shape[0]:
        gram = x.T @ x
        gram.diagonal().add_(alpha)
        weight = torch.linalg.solve(gram, x.T @ y)
        prediction = test @ weight
    else:
        gram = x @ x.T
        gram.diagonal().add_(alpha)
        dual = torch.linalg.solve(gram, y)
        prediction = (test @ x.T) @ dual
    output = (
        prediction.cpu().numpy() + target_mean
    ).astype(np.float64)
    return output[:, 0] if one_dimensional else output


def fit_binary(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    values = np.unique(train_y)
    if len(values) == 1:
        return np.full(test_x.shape[0], float(values[0]), dtype=np.float64)
    return ridge_predict(
        train_x,
        train_y,
        test_x,
        device or torch.device("cpu"),
    )


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive_count": int(np.sum(array > 0.0)),
    }


class MobileNetFeatures(torch.nn.Module):
    def __init__(self, pretrained: Path) -> None:
        super().__init__()
        model = mobilenet_v3_small(weights=None)
        state = torch.load(
            pretrained,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        self.features = model.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(frames)).flatten(1)


def frame_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        (224, 128),
        interpolation=cv2.INTER_LINEAR,
    )
    value = resized.astype(np.float32) / 255.0
    value = (value - MEAN) / STD
    return torch.from_numpy(value.transpose(2, 0, 1))


def flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_frames: list[torch.Tensor],
    batch_keys: list[tuple[str, int]],
    output: dict[tuple[str, int], np.ndarray],
) -> None:
    if not batch_frames:
        return
    with torch.inference_mode():
        values = model(
            torch.stack(batch_frames).to(device, non_blocking=True)
        )
    for key, feature in zip(batch_keys, values.cpu().numpy()):
        output[key] = feature.astype(np.float32, copy=False)
    batch_frames.clear()
    batch_keys.clear()


def extract_features(
    records: list[dict[str, Any]],
    pretrained: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    requested: dict[str, set[int]] = defaultdict(set)
    expected_hashes: dict[str, str] = {}
    for record in records:
        video_path = str(Path(record["video_path"]).resolve())
        requested[video_path].update(
            int(value) for value in record["history_scene_frames"]
        )
        expected_hashes[video_path] = str(record["video_sha256"])

    model = MobileNetFeatures(pretrained).to(device).eval()
    by_key: dict[tuple[str, int], np.ndarray] = {}
    video_rows = []
    for video_text in sorted(requested):
        video_path = Path(video_text)
        if sha256(video_path) != expected_hashes[video_text]:
            raise ValueError(f"Video hash mismatch: {video_path}")
        wanted = requested[video_text]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Unable to open video: {video_path}")
        batch_frames: list[torch.Tensor] = []
        batch_keys: list[tuple[str, int]] = []
        frame_number = 0
        captured = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number not in wanted:
                    continue
                batch_frames.append(frame_tensor(frame))
                batch_keys.append((video_text, frame_number))
                captured += 1
                if len(batch_frames) >= batch_size:
                    flush_batch(
                        model,
                        device,
                        batch_frames,
                        batch_keys,
                        by_key,
                    )
            flush_batch(
                model,
                device,
                batch_frames,
                batch_keys,
                by_key,
            )
        finally:
            capture.release()
        if captured != len(wanted):
            missing = sorted(
                index
                for index in wanted
                if (video_text, index) not in by_key
            )
            raise ValueError(
                f"Missing requested video frames for {video_path}: "
                f"{missing[:10]}"
            )
        video_rows.append(
            {
                "video_path": video_text,
                "requested_frame_count": len(wanted),
                "decoded_frame_count": frame_number,
            }
        )

    matrix = np.stack(
        [
            np.stack(
                [
                    by_key[
                        (
                            str(Path(record["video_path"]).resolve()),
                            int(frame),
                        )
                    ]
                    for frame in record["history_scene_frames"]
                ]
            )
            for record in records
        ]
    )
    return matrix, {
        "videos": video_rows,
        "unique_frame_count": len(by_key),
        "feature_shape": list(matrix.shape),
    }


def targets(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proximity = np.asarray(
        [
            int(record["target"]["future_proximity_le_1_25m"])
            for record in records
        ],
        dtype=np.int64,
    )
    corridor = np.asarray(
        [
            int(record["target"]["future_corridor_intrusion"])
            for record in records
        ],
        dtype=np.int64,
    )
    occupancy = np.asarray(
        [record["target"]["occupancy_target"] for record in records],
        dtype=np.int64,
    ).reshape(len(records), -1)
    distance = np.asarray(
        [
            float(
                record["target"][
                    "future_minimum_synchronized_distance_m"
                ]
            )
            for record in records
        ],
        dtype=np.float64,
    )
    return proximity, corridor, occupancy, distance


def evaluate_arm(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    proximity: np.ndarray,
    corridor: np.ndarray,
    occupancy: np.ndarray,
    distance: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    scaled_train, scaled_test = standardize(train_x, test_x)
    train_targets = np.column_stack(
        (
            proximity[train_indices],
            corridor[train_indices],
            occupancy[train_indices],
            distance[train_indices],
        )
    )
    prediction = ridge_predict(
        scaled_train,
        train_targets,
        scaled_test,
        device,
    )
    proximity_score = prediction[:, 0]
    corridor_score = prediction[:, 1]
    occupancy_score = prediction[:, 2:-1]
    predicted_distance = prediction[:, -1]
    rank = spearman(distance[test_indices], predicted_distance)
    return {
        "proximity": binary_metrics(
            proximity[test_indices],
            proximity_score,
        ),
        "corridor": binary_metrics(
            corridor[test_indices],
            corridor_score,
        ),
        "occupancy_micro": binary_metrics(
            occupancy[test_indices].reshape(-1),
            occupancy_score.reshape(-1),
        ),
        "distance": {
            "spearman": rank,
            "mean_absolute_error_m": float(
                np.mean(
                    np.abs(
                        distance[test_indices] - predicted_distance
                    )
                )
            ),
        },
    }


def prior_arm(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    proximity: np.ndarray,
    corridor: np.ndarray,
    occupancy: np.ndarray,
    distance: np.ndarray,
) -> dict[str, Any]:
    count = len(test_indices)
    proximity_probability = np.full(
        count,
        np.mean(proximity[train_indices]),
    )
    corridor_probability = np.full(
        count,
        np.mean(corridor[train_indices]),
    )
    occupancy_probability = np.broadcast_to(
        np.mean(occupancy[train_indices], axis=0),
        (count, occupancy.shape[1]),
    )
    predicted_distance = np.full(
        count,
        np.median(distance[train_indices]),
    )
    return {
        "proximity": binary_metrics(
            proximity[test_indices],
            proximity_probability,
        ),
        "corridor": binary_metrics(
            corridor[test_indices],
            corridor_probability,
        ),
        "occupancy_micro": binary_metrics(
            occupancy[test_indices].reshape(-1),
            occupancy_probability.reshape(-1),
        ),
        "distance": {
            "spearman": None,
            "mean_absolute_error_m": float(
                np.mean(
                    np.abs(
                        distance[test_indices] - predicted_distance
                    )
                )
            ),
        },
    }


def metric_value(result: dict[str, Any], path: str) -> float | None:
    current: Any = result
    for key in path.split("."):
        current = current[key]
    return None if current is None else float(current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite RGB-history screen output")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")

    records = load_jsonl(args.samples)
    if len(records) != 1078:
        raise ValueError("Expected the fixed 1,078-sample THOR corpus")
    if len({record["source_session_id"] for record in records}) != 19:
        raise ValueError("Expected 19 source sessions")
    records.sort(key=lambda row: row["sample_id"])
    folds = np.asarray([int(record["fold"]) for record in records])
    sources = np.asarray(
        [str(record["source_session_id"]) for record in records]
    )
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Expected folds 0..4")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    feature_tensor, extraction = extract_features(
        records,
        args.pretrained,
        device,
        args.batch_size,
    )
    current_x = feature_tensor[:, -1]
    history_x = history_design(feature_tensor)
    proximity, corridor, occupancy, distance = targets(records)

    fold_rows = []
    metric_paths = (
        "proximity.auroc",
        "proximity.average_precision",
        "corridor.auroc",
        "corridor.average_precision",
        "occupancy_micro.auroc",
        "occupancy_micro.average_precision",
        "distance.spearman",
        "distance.mean_absolute_error_m",
    )
    deltas: dict[str, list[float]] = {
        path: [] for path in metric_paths
    }
    for fold in range(5):
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        train_sources = set(sources[train_indices])
        test_sources = set(sources[test_indices])
        if train_sources & test_sources:
            raise ValueError("Source-session leakage across folds")
        current = evaluate_arm(
            current_x[train_indices],
            current_x[test_indices],
            train_indices,
            test_indices,
            proximity,
            corridor,
            occupancy,
            distance,
            device,
        )
        history = evaluate_arm(
            history_x[train_indices],
            history_x[test_indices],
            train_indices,
            test_indices,
            proximity,
            corridor,
            occupancy,
            distance,
            device,
        )
        prior = prior_arm(
            train_indices,
            test_indices,
            proximity,
            corridor,
            occupancy,
            distance,
        )
        fold_delta = {}
        for path in metric_paths:
            history_value = metric_value(history, path)
            current_value = metric_value(current, path)
            value = (
                history_value - current_value
                if history_value is not None
                and current_value is not None
                else None
            )
            if path == "distance.mean_absolute_error_m" and value is not None:
                value = -value
            fold_delta[path] = value
            if value is not None:
                deltas[path].append(value)
        fold_rows.append(
            {
                "fold": fold,
                "train_sample_count": len(train_indices),
                "heldout_sample_count": len(test_indices),
                "train_source_count": len(train_sources),
                "heldout_source_count": len(test_sources),
                "heldout_sources": sorted(test_sources),
                "target_counts": {
                    "proximity_positive": int(
                        np.sum(proximity[test_indices])
                    ),
                    "corridor_positive": int(
                        np.sum(corridor[test_indices])
                    ),
                    "occupancy_positive_cells": int(
                        np.sum(occupancy[test_indices])
                    ),
                },
                "prior": prior,
                "current": current,
                "history": history,
                "history_minus_current": fold_delta,
            }
        )

    aggregate = {
        path: summarize(values)
        for path, values in deltas.items()
    }
    core_paths = (
        "proximity.auroc",
        "proximity.average_precision",
        "corridor.auroc",
        "corridor.average_precision",
        "occupancy_micro.auroc",
        "occupancy_micro.average_precision",
        "distance.spearman",
    )
    supported = all(
        aggregate[path]["median"] is not None
        and float(aggregate[path]["median"]) > 0.0
        and int(aggregate[path]["positive_count"]) >= 3
        for path in core_paths
    )
    status = (
        "D8_RGB_HISTORY_SEPARABILITY_INCREMENT_SUPPORTED"
        if supported
        else "D8_RGB_HISTORY_SEPARABILITY_INCREMENT_NOT_STABLE"
    )

    args.output_root.mkdir(parents=True)
    feature_path = args.output_root / "features.npz"
    np.savez_compressed(
        feature_path,
        sample_ids=np.asarray(
            [record["sample_id"] for record in records]
        ),
        features=feature_tensor,
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development separability screen",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "design": {
            "split": "fixed SHA-256(source_session_id) modulo 5",
            "seed": SEED,
            "backbone": "frozen torchvision MobileNetV3-small ImageNet",
            "current_features": "current-frame 576-D pooled embedding",
            "history_features": (
                "current, current-minus-earliest, current-minus-previous, "
                "five-frame standard deviation; 2,304-D"
            ),
            "binary_readout": (
                "train-fold standardization plus fixed multi-output "
                "L2 ridge(alpha=10) ranking score"
            ),
            "distance_readout": (
                "train-fold standardization plus L2 ridge(alpha=10)"
            ),
            "success_gate": (
                "history-minus-current median > 0 and at least 3/5 "
                "positive folds for both AUROC/AP on proximity, corridor, "
                "48-cell occupancy micro metrics, and distance Spearman"
            ),
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(set(sources)),
            "folds": 5,
            "occupancy_cells_per_sample": occupancy.shape[1],
        },
        "extraction": extraction,
        "folds": fold_rows,
        "aggregate_history_minus_current": aggregate,
        "features": {
            "path": str(feature_path.resolve()),
            "sha256": sha256(feature_path),
        },
        "next_action": (
            "train a compact source-held-out temporal student"
            if supported
            else (
                "do not fine-tune on this representation; inspect target "
                "alignment or acquire more source-diverse local supervision"
            )
        ),
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(report_path)
    Path(str(report_path) + ".sha256").write_text(
        f"{digest}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "device": str(device),
                "samples": len(records),
                "unique_frames": extraction["unique_frame_count"],
                "aggregate_history_minus_current": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
