#!/usr/bin/env python3
"""Run an episode-isolated ridge probe on provisional public-video supervision.

The probe freezes the existing MobileNetV3 OS8+OS32 representation, pools each
multi-frame episode deterministically, and evaluates with leave-one-episode-out
folds.  It consumes only validated v2 provisional-training manifests and
verifies every image SHA256 before feature extraction.  The report is a tiny
data diagnostic, never calibration, blind evaluation, or production evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

import sanpo_deterministic_linear_probe as ridge_probe
import train_export_sanpo_segmentation as shared
from validate_public_video_silver_labels import load_json, validate


SCHEMA = "blindassist_public_silver_frozen_feature_probe_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_frame_map(feature_map: np.ndarray) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError("feature map must be non-empty HxWxC")
    height, width, _ = values.shape
    center = values[:, width // 4: max(width // 4 + 1, (3 * width) // 4)]
    lower_center = values[height // 2:, width // 4: max(width // 4 + 1, (3 * width) // 4)]
    return np.concatenate([
        values.mean(axis=(0, 1)),
        values.max(axis=(0, 1)),
        center.mean(axis=(0, 1)),
        lower_center.mean(axis=(0, 1)),
    ])


def pool_corridor_relative_frame(feature_map: np.ndarray, semantic_logits: np.ndarray) -> np.ndarray:
    """Pool frozen features relative to the predicted lower-center corridor.

    The definition is intentionally fixed: bilinearly-produced semantic logits
    are sampled at feature-cell centers, softmax channel 0 is the existing
    walkable probability, and only the lower half / central half is used.  No
    event labels, thresholds, or source masks influence the spatial weights.
    """
    features = np.asarray(feature_map, dtype=np.float64)
    logits = np.asarray(semantic_logits, dtype=np.float64)
    if features.ndim != 3 or logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("corridor pooling needs HxWxC features and HxWxK semantic logits")
    height, width, channels = features.shape
    y_indices = np.minimum(((np.arange(height) + 0.5) * logits.shape[0] / height).astype(int), logits.shape[0] - 1)
    x_indices = np.minimum(((np.arange(width) + 0.5) * logits.shape[1] / width).astype(int), logits.shape[1] - 1)
    sampled = logits[y_indices[:, None], x_indices[None, :]]
    sampled -= sampled.max(axis=-1, keepdims=True)
    probabilities = np.exp(sampled)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    region = np.zeros((height, width), dtype=bool)
    region[height // 2:, width // 4:max(width // 4 + 1, (3 * width) // 4)] = True
    region_features = features[region]
    walkable_weights = probabilities[..., 0][region]
    nonwalkable_weights = 1.0 - walkable_weights

    def weighted_mean(weights: np.ndarray) -> np.ndarray:
        denominator = float(weights.sum())
        if denominator <= 1e-12:
            return np.zeros(channels, dtype=np.float64)
        return (region_features * weights[:, None]).sum(axis=0) / denominator

    walkable = weighted_mean(walkable_weights)
    nonwalkable = weighted_mean(nonwalkable_weights)
    return np.concatenate([
        walkable,
        nonwalkable,
        nonwalkable - walkable,
        np.asarray([
            float(walkable_weights.mean()),
            float(nonwalkable_weights.mean()),
            float(nonwalkable_weights.max()),
        ]),
    ])


def residual_motion_descriptor(flow: np.ndarray) -> np.ndarray:
    values = np.asarray(flow, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2 or min(values.shape[:2]) < 2:
        raise ValueError("motion descriptor needs an HxWx2 optical-flow field")
    residual = values - np.median(values.reshape(-1, 2), axis=0)
    height, width, _ = residual.shape
    region = np.zeros((height, width), dtype=bool)
    region[height // 2:, width // 4:max(width // 4 + 1, (3 * width) // 4)] = True
    magnitude = np.linalg.norm(residual, axis=-1)
    divergence = np.gradient(residual[..., 0], axis=1) + np.gradient(residual[..., 1], axis=0)
    yy, xx = np.mgrid[:height, :width]
    rx = xx - (width - 1) / 2.0
    ry = yy - (height - 1) / 2.0
    radius = np.maximum(np.sqrt(rx * rx + ry * ry), 1.0)
    radial = (residual[..., 0] * rx + residual[..., 1] * ry) / radius
    raw_magnitude = np.linalg.norm(values, axis=-1)
    return np.asarray([
        float(magnitude[region].mean()),
        float(np.quantile(magnitude[region], 0.90)),
        float(np.maximum(divergence[region], 0.0).mean()),
        float(np.quantile(np.maximum(divergence[region], 0.0), 0.90)),
        float(radial[region].mean()),
        float(np.quantile(radial[region], 0.90)),
        float(raw_magnitude[region].mean()),
    ])


def episode_motion_vector(paths: Sequence[Path], *, size: int = 192) -> np.ndarray:
    if len(paths) < 2:
        raise ValueError("motion episode needs at least two frames")
    import cv2
    grayscale: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
        grayscale.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    descriptors: list[np.ndarray] = []
    for previous, current in zip(grayscale, grayscale[1:]):
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        descriptors.append(residual_motion_descriptor(flow))
    values = np.stack(descriptors)
    return np.concatenate([values.mean(axis=0), values.max(axis=0), values[-1] - values[0]])


def pool_episode(frame_vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_vectors, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("episode needs a non-empty frame-vector matrix")
    return np.concatenate([values.mean(axis=0), values.max(axis=0), values[-1] - values[0]])


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if truth.shape != pred.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("binary metrics need aligned non-empty vectors")
    matrix = np.zeros((2, 2), dtype=np.int64)
    for expected, actual in zip(truth, pred):
        if expected not in (0, 1) or actual not in (0, 1):
            raise ValueError("binary labels must be 0 or 1")
        matrix[expected, actual] += 1
    recalls = [float(matrix[index, index] / matrix[index].sum()) if matrix[index].sum() else None for index in range(2)]
    present = [value for value in recalls if value is not None]
    return {
        "confusion_matrix_rows_truth_columns_prediction": matrix.tolist(),
        "accuracy": float(np.mean(truth == pred)),
        "candidate_no_alert_recall": recalls[0],
        "candidate_alert_recall": recalls[1],
        "balanced_accuracy": float(np.mean(present)),
    }


def counterfactual_delta_alignment(features: np.ndarray, episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(episodes):
        raise ValueError("counterfactual alignment needs aligned episode features")
    grouped: dict[str, list[int]] = {}
    for index, episode in enumerate(episodes):
        pair_id = episode.get("counterfactual_pair_id")
        if isinstance(pair_id, str) and pair_id:
            grouped.setdefault(pair_id, []).append(index)
    deltas: list[np.ndarray] = []
    pair_ids: list[str] = []
    for pair_id, indices in sorted(grouped.items()):
        negatives = [index for index in indices if episodes[index]["label"] == 0]
        positives = [index for index in indices if episodes[index]["label"] == 1]
        if not negatives or not positives:
            continue
        delta = values[positives].mean(axis=0) - values[negatives].mean(axis=0)
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            continue
        deltas.append(delta / norm)
        pair_ids.append(pair_id)
    if len(deltas) < 2:
        return {
            "matched_pair_count": len(deltas),
            "passed": False,
            "minimum_pairs": 2,
            "minimum_mean_pairwise_cosine": 0.20,
            "reason": "at least two non-degenerate matched pairs are required",
        }
    matrix = np.asarray(deltas) @ np.asarray(deltas).T
    off_diagonal = matrix[np.triu_indices(len(deltas), k=1)]
    mean_cosine = float(off_diagonal.mean())
    return {
        "matched_pair_count": len(deltas),
        "pair_ids": pair_ids,
        "pairwise_cosine_matrix": matrix.tolist(),
        "mean_pairwise_cosine": mean_cosine,
        "minimum_mean_pairwise_cosine": 0.20,
        "passed": mean_cosine >= 0.20,
        "interpretation": "Tests whether no-alert to alert feature deltas share a reusable prototype direction; it does not train or calibrate a classifier.",
    }


def fit_episode_ridge(features: np.ndarray, labels: np.ndarray, *, ridge: float, class_balanced: bool) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y_ids = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y_ids) or not len(x) or ridge <= 0:
        raise ValueError("episode ridge needs aligned non-empty features and positive ridge")
    counts = np.bincount(y_ids, minlength=2)
    if np.any(counts == 0):
        raise ValueError("episode ridge requires both classes")
    weights = np.ones(len(y_ids), dtype=np.float64)
    if class_balanced:
        weights = np.asarray([len(y_ids) / (2.0 * counts[label]) for label in y_ids], dtype=np.float64)
    weights /= weights.mean()
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-16))
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (x - mean) / scale
    targets = np.eye(2, dtype=np.float64)[y_ids]
    x_mean = np.average(standardized, axis=0, weights=weights)
    y_mean = np.average(targets, axis=0, weights=weights)
    root_weight = np.sqrt(weights)[:, None]
    centered_x = (standardized - x_mean) * root_weight
    centered_y = (targets - y_mean) * root_weight
    dual = np.linalg.solve(centered_x @ centered_x.T + ridge * np.eye(len(x)), centered_y)
    standardized_kernel = centered_x.T @ dual
    standardized_bias = y_mean - x_mean @ standardized_kernel
    kernel = standardized_kernel / scale[:, None]
    bias = standardized_bias - (mean / scale) @ standardized_kernel
    digest = hashlib.sha256()
    digest.update(np.asarray(kernel, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(bias, dtype="<f8").tobytes(order="C"))
    return {"kernel": kernel, "bias": bias, "coefficient_sha256": digest.hexdigest()}


def leave_one_episode_out(features: np.ndarray, labels: np.ndarray, episode_ids: Sequence[str], *, ridge: float, class_balanced: bool = True) -> dict[str, Any]:
    return leave_one_source_group_out(
        features,
        labels,
        episode_ids,
        episode_ids,
        ridge=ridge,
        class_balanced=class_balanced,
    )


def leave_one_source_group_out(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: Sequence[str],
    source_ids: Sequence[str],
    *,
    ridge: float,
    class_balanced: bool = True,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(episode_ids) or len(y) != len(source_ids):
        raise ValueError("features, labels, episode IDs, and source IDs must be aligned")
    predictions = np.full(len(y), -1, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    ordered_groups = list(dict.fromkeys(source_ids))
    source_array = np.asarray(source_ids, dtype=object)
    for source_id in ordered_groups:
        holdout = source_array == source_id
        train = ~holdout
        if set(y[train].tolist()) != {0, 1}:
            raise ValueError(f"training fold for source {source_id} does not contain both classes")
        fitted = fit_episode_ridge(x[train], y[train], ridge=ridge, class_balanced=class_balanced)
        fold_predictions = ridge_probe.predict_labels(x[holdout], fitted["kernel"], fitted["bias"]).astype(np.int64)
        predictions[holdout] = fold_predictions
        folds.append({
            "held_out_source_id": source_id,
            "held_out_episode_ids": [episode_ids[index] for index in np.flatnonzero(holdout)],
            "expected": y[holdout].tolist(),
            "predicted": fold_predictions.tolist(),
            "coefficient_sha256": fitted["coefficient_sha256"],
        })
    if np.any(predictions < 0):
        raise RuntimeError("source-group evaluation left an episode without a prediction")
    return {"predictions": predictions.tolist(), "folds": folds, "metrics": binary_metrics(y, predictions)}


def load_episode_specs(package_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for silver_path in sorted(package_root.glob("*/silver_labels_v2.json")):
        source_path = silver_path.parent / "source_manifest_v2.json"
        silver = load_json(silver_path)
        source = load_json(source_path)
        result = validate(silver, source_manifest_path=source_path)
        if result.get("training_execution_authorized") is not True:
            raise ValueError(f"v2 package does not authorize provisional training: {silver_path}")
        promotion = source.get("promotion")
        if not isinstance(promotion, dict) or not isinstance(promotion.get("image_root"), str):
            raise ValueError(f"v2 source has no bound image_root: {source_path}")
        image_root = Path(promotion["image_root"]).resolve()
        frames = source.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"v2 source contains no frames: {source_path}")
        by_hash: dict[str, dict[str, Any]] = {}
        for frame in frames:
            if isinstance(frame, dict) and isinstance(frame.get("sha256"), str):
                by_hash.setdefault(frame["sha256"], frame)
        for episode in silver["episodes"]:
            verdict = episode["silver_should_alert"]
            row = {
                "episode_id": episode["episode_id"],
                "source_id": silver["source"]["source_id"],
                "silver_path": str(silver_path.resolve()),
                "silver_sha256": sha256_file(silver_path),
                "source_path": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "image_root": str(image_root),
                "verdict": verdict,
                "confidence": float(episode["confidence"]),
                "counterfactual_pair_id": (
                    episode.get("counterfactual_pair_id")
                    or (episode.get("risk_profile") or {}).get("counterfactual_pair_id")
                ),
                "label": 1 if verdict == "candidate_alert" else 0,
                "frames": [],
            }
            if verdict == "abstain":
                excluded.append({key: value for key, value in row.items() if key != "frames"})
                continue
            for evidence_hash in episode["evidence_frame_sha256"]:
                frame = by_hash.get(evidence_hash)
                if frame is None or not isinstance(frame.get("file_name"), str):
                    raise ValueError(f"episode evidence is absent from source: {episode['episode_id']}: {evidence_hash}")
                image_path = (image_root / frame["file_name"]).resolve()
                if not image_path.is_relative_to(image_root) or not image_path.is_file() or sha256_file(image_path) != evidence_hash:
                    raise ValueError(f"episode image does not match bound SHA256: {image_path}")
                row["frames"].append({"frame_index": frame.get("frame_index"), "sha256": evidence_hash, "path": str(image_path)})
            row["frames"].sort(key=lambda item: (item["frame_index"] if isinstance(item["frame_index"], int) else 0, item["sha256"]))
            included.append(row)
    if not included:
        raise ValueError("no non-abstaining v2 episodes found")
    if len({row["episode_id"] for row in included}) != len(included):
        raise ValueError("episode IDs must be unique")
    return included, excluded


def extract_features(feature_model: Any, episodes: Sequence[dict[str, Any]], *, input_size: int, batch_size: int, pooling: str) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for episode in episodes:
        frame_vectors: list[np.ndarray] = []
        paths = [Path(frame["path"]) for frame in episode["frames"]]
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.float32))
            outputs = feature_model.predict(np.stack(images), batch_size=batch_size, verbose=0)
            if pooling in {"global_center", "global_center_residual_motion"}:
                frame_vectors.extend(pool_frame_map(item) for item in outputs)
            else:
                maps, logits = outputs
                frame_vectors.extend(pool_corridor_relative_frame(item, semantic) for item, semantic in zip(maps, logits))
        episode_vector = pool_episode(np.stack(frame_vectors))
        if pooling == "global_center_residual_motion":
            episode_vector = np.concatenate([episode_vector, episode_motion_vector(paths)])
        vectors.append(episode_vector)
    return np.stack(vectors)


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    weights = args.feature_weights.resolve()
    if not package_root.is_dir() or not weights.is_file():
        raise FileNotFoundError("package root or feature weights are missing")
    episodes, excluded = load_episode_specs(package_root)
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    if set(labels.tolist()) != {0, 1} or min(np.bincount(labels, minlength=2)) < 2:
        raise ValueError("probe requires at least two independent episodes per class")

    os.environ["KERAS_BACKEND"] = args.backend
    import keras
    random.seed(args.seed)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    if args.backend == "torch":
        import torch
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = shared.sanpo_segmentation_model.build_mobilenetv3_lraspp(
        keras, args.input_size, backbone_alpha=1.0, decoder_channels=96,
        detail_output_stride=8, semantic_output_stride=32,
    )
    model.load_weights(weights)
    low = model.get_layer("activation_1").output
    high = model.get_layer("activation_17").output
    scale = int(low.shape[1]) // int(high.shape[1])
    high = keras.layers.UpSampling2D(size=(scale, scale), interpolation="bilinear", name="public_probe_high_up")(high)
    fused_features = keras.layers.Concatenate(name="public_probe_os8_os32")([low, high])
    feature_model = keras.Model(
        model.input,
        [fused_features, model.output] if args.pooling == "segmentation_corridor_relative" else fused_features,
    )
    features = extract_features(feature_model, episodes, input_size=args.input_size, batch_size=args.batch_size, pooling=args.pooling)
    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]
    first = leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True)
    second = leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True)
    deterministic = first == second
    metrics = first["metrics"]
    delta_alignment = counterfactual_delta_alignment(features, episodes)
    gate = bool(
        deterministic
        and metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "class_counts": {"candidate_no_alert": int(np.sum(labels == 0)), "candidate_alert": int(np.sum(labels == 1))},
        "feature_source": {
            "architecture": "MobileNetV3Small+LR-ASPP",
            "layer": "activation_1+activation_17_bilinear_up",
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "feature_dimension": int(features.shape[1]),
            "frame_pool": (
                "global_mean+global_max+center_mean+lower_center_mean"
                if args.pooling in {"global_center", "global_center_residual_motion"}
                else "predicted_walkable_weighted_mean+predicted_nonwalkable_weighted_mean+difference+lower_center_probability_summary"
            ),
            "spatial_supervision": (
                "none"
                if args.pooling == "global_center"
                else (
                    "frozen_model_semantic_logits; walkable channel 0; lower-half central-half; no source mask or event label"
                    if args.pooling == "segmentation_corridor_relative"
                    else "none"
                )
            ),
            "temporal_motion": (
                "none"
                if args.pooling != "global_center_residual_motion"
                else "Farneback flow at 192x192; global median translation removed; fixed lower-center residual magnitude+divergence+radial expansion; episode mean+max+last-minus-first"
            ),
            "episode_pool": "frame_mean+frame_max+last_minus_first",
            "trainable_parameters": 0,
        },
        "evaluation": {
            "split": "leave_one_source_group_out",
            "group_key": "source_id",
            "frame_or_session_leakage": False,
            "ridge": args.ridge,
            "training_fold_class_balance": "inverse_frequency_equal_class_weight",
            **first,
            "repeat_exact": deterministic,
        },
        "counterfactual_delta_alignment": delta_alignment,
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {"balanced_accuracy_gte": args.minimum_balanced_accuracy, "each_class_recall_gte": args.minimum_class_recall},
            "interpretation_if_passed": "Frozen features carry a pilot-level event/no-event signal; investigate head/policy optimization next.",
            "interpretation_if_failed": "No pilot-level linear separation; prioritize representation and more independent episodes before head optimization.",
        },
        "episodes": [{key: value for key, value in row.items() if key != "frames"} | {"frame_count": len(row["frames"])} for row in episodes],
        "excluded_abstentions": excluded,
        "evidence_limit": "Provisional GPT/VLM labels and a very small public episode set; not human accuracy, calibration, blind evaluation, or production promotion.",
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--feature-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("torch", "tensorflow"), default="torch")
    parser.add_argument("--input-size", type=int, choices=(384,), default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument(
        "--pooling",
        choices=("global_center", "segmentation_corridor_relative", "global_center_residual_motion"),
        default="global_center",
    )
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.ridge <= 0:
        parser.error("batch size and ridge must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "episode_count": report["episode_count"],
        "balanced_accuracy": report["evaluation"]["metrics"]["balanced_accuracy"],
        "linear_separable": report["linear_separability_gate"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
