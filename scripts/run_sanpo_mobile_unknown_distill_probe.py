#!/usr/bin/env python3
"""Test whether frozen MobileNet features can reproduce a frozen DINO unknown score.

The teacher is a train-walkable PCA familiarity score.  This is a representation
distillation feasibility probe only: no event/risk truth, model weights,
threshold calibration, blind data, or deployable student are created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_sanpo_corridor_anomaly_probe as anomaly
import sanpo_depth_anything_linear_probe as depth_probe
import sanpo_training_gate as training_gate
import train_export_sanpo_segmentation as shared


SCHEMA = "blindassist_sanpo_mobile_unknown_distill_probe_v1"
DEFAULT_DATASET = anomaly.DEFAULT_DATASET
DEFAULT_SRC = anomaly.DEFAULT_SRC
DEFAULT_CHECKPOINT = anomaly.DEFAULT_CHECKPOINT
DEFAULT_MOBILE_WEIGHTS = "test-artifacts.local/segmentation-candidate/p1-sigmoid-no-pooled-bn-20260713/candidate.weights.h5"
DEFAULT_REPORT = "artifacts.local/evidence/sanpo-mobile-unknown-distill-probe-20260715/report.json"


def evenly_spaced_rows(length: int, *, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    return np.linspace(0, length - 1, min(length, count), dtype=np.int64)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return ranks


def spearman_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape or prediction.ndim != 1 or len(prediction) < 2:
        raise ValueError("prediction and target must be matching one-dimensional vectors")
    left = _rank(np.asarray(prediction, dtype=np.float64))
    right = _rank(np.asarray(target, dtype=np.float64))
    denominator = float(np.linalg.norm(left - left.mean()) * np.linalg.norm(right - right.mean()))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left - left.mean(), right - right.mean()) / denominator)


def fit_ridge_regression(features: np.ndarray, target: np.ndarray, *, ridge: float) -> dict[str, np.ndarray]:
    if features.ndim != 2 or target.shape != (len(features),) or ridge <= 0:
        raise ValueError("invalid ridge regression inputs")
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    normalized = (features.astype(np.float64) - mean) / scale
    target_mean = float(target.mean(dtype=np.float64))
    kernel = np.linalg.solve(normalized.T @ normalized + ridge * np.eye(normalized.shape[1]), normalized.T @ (target - target_mean))
    return {"mean": mean, "scale": scale, "kernel": kernel, "bias": np.asarray(target_mean)}


def predict_ridge(features: np.ndarray, fitted: dict[str, np.ndarray]) -> np.ndarray:
    return ((features.astype(np.float64) - fitted["mean"]) / fitted["scale"]) @ fitted["kernel"] + fitted["bias"]


def _dino_map(model: Any, record: shared.Record, *, input_size: int, layer_index: int) -> tuple[np.ndarray, np.ndarray]:
    return depth_probe._features_for_record(
        model, record, input_size=input_size, layer_index=layer_index,
        mobile_feature_model=None, mobile_input_size=0,
    )


def _mobile_map(feature_model: Any, record: shared.Record, *, input_size: int) -> np.ndarray:
    image, _target = shared.load_example(record, input_size)
    result = np.asarray(feature_model.predict(image[None, ...], verbose=0)[0], dtype=np.float32)
    if result.ndim != 3:
        raise ValueError("MobileNet feature model must produce H×W×C")
    return result


def _distill_samples(
    dino_model: Any, mobile_model: Any, records: Sequence[shared.Record], *,
    fitted_teacher: dict[str, np.ndarray], input_size: int, layer_index: int, mobile_input_size: int,
    pixels_per_record: int,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in sorted(records, key=lambda item: item.sample_id):
        dino_map, _source_target = _dino_map(dino_model, record, input_size=input_size, layer_index=layer_index)
        score = anomaly.reconstruction_error(dino_map.reshape(-1, dino_map.shape[-1]), fitted_teacher).reshape(dino_map.shape[:2])
        mobile_map = _mobile_map(mobile_model, record, input_size=mobile_input_size)
        resized_score = cv2.resize(score.astype(np.float32), (mobile_map.shape[1], mobile_map.shape[0]), interpolation=cv2.INTER_LINEAR)
        positions = evenly_spaced_rows(mobile_map.shape[0] * mobile_map.shape[1], count=pixels_per_record)
        features.append(mobile_map.reshape(-1, mobile_map.shape[-1])[positions])
        targets.append(np.log1p(resized_score.reshape(-1)[positions]).astype(np.float64))
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = shared.resolve(root, args.dataset_root).resolve()
    source_root = shared.resolve(root, args.src_root).resolve()
    checkpoint = shared.resolve(root, args.checkpoint).resolve()
    weights = shared.resolve(root, args.mobile_feature_weights).resolve()
    report_path = shared.resolve(root, args.report).resolve()
    if not source_root.is_dir() or not checkpoint.is_file() or not weights.is_file():
        raise FileNotFoundError("frozen DINO source/checkpoint or MobileNet weights are missing")
    gate_path = shared.resolve(dataset_root, args.training_gate_report).resolve()
    gate = training_gate.consume_training_authorization(dataset_root, gate_path)
    records = sorted(shared.load_records(dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST), key=lambda item: item.sample_id)
    train_records, dev_records = shared.records_by_split(records, "train"), shared.records_by_split(records, "dev")

    import torch
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.use_deterministic_algorithms(True)
    dino_model = depth_probe.depth_anything.load_model(source_root, checkpoint, args.encoder)
    dino_model.eval()
    os.environ["KERAS_BACKEND"] = "torch"
    mobile_args = argparse.Namespace(
        append_mobile_os8_os32=True, mobile_feature_weights=str(weights), mobile_backend="torch",
        mobile_input_size=args.mobile_input_size, mobile_backbone_alpha=1.0, mobile_decoder_channels=96,
        model_seed=args.model_seed,
    )
    mobile_model, mobile_source = depth_probe._mobile_feature_model(mobile_args)
    assert mobile_model is not None and mobile_source is not None

    with torch.no_grad():
        train_source_samples, _counts = anomaly._extract_samples(
            dino_model, train_records, input_size=args.input_size, layer_index=args.layer_index,
            per_class=args.teacher_walkable_pixels_per_record,
        )
        teacher = anomaly.fit_pca_reconstruction(anomaly._stack(train_source_samples[0]), components=args.teacher_components)
        train_x, train_y = _distill_samples(
            dino_model, mobile_model, train_records, fitted_teacher=teacher, input_size=args.input_size,
            layer_index=args.layer_index, mobile_input_size=args.mobile_input_size, pixels_per_record=args.pixels_per_record,
        )
        dev_x, dev_y = _distill_samples(
            dino_model, mobile_model, dev_records, fitted_teacher=teacher, input_size=args.input_size,
            layer_index=args.layer_index, mobile_input_size=args.mobile_input_size, pixels_per_record=args.pixels_per_record,
        )
    fitted = fit_ridge_regression(train_x, train_y, ridge=args.ridge)
    dev_prediction = predict_ridge(dev_x, fitted)
    residual = float(np.mean(np.square(dev_prediction - dev_y)))
    variance = float(np.mean(np.square(dev_y - dev_y.mean())))
    r2 = float(1.0 - residual / variance) if variance > 0 else float("nan")
    correlation = spearman_correlation(dev_prediction, dev_y)
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_only": True, "promotion": "do_not_replace_default_model", "risk_or_event_truth_present": False,
        "purpose": "frozen-feature teacher/student feasibility only; no deployable student or alert semantics",
        "access_contract": {"canonical_rows_read": ["train", "dev"], "blind_holdout_access": "not_accessed_by_probe", "weights_saved": False},
        "teacher": {"type": "frozen DINO train-walkable PCA reconstruction score", "components": args.teacher_components, "walkable_fit_samples": int(sum(len(row) for row in train_source_samples[0])), "source_semantic_role": "auxiliary_geometry_only"},
        "student": {"type": "closed_form ridge over frozen MobileNetV3 raw OS8+OS32 features", "weights_sha256": shared.sha256_file(weights), "feature_dimension": int(train_x.shape[1]), "ridge": args.ridge, "trainable_parameters": 0},
        "sampling": {"train_rows": int(len(train_y)), "dev_rows": int(len(dev_y)), "pixels_per_record": args.pixels_per_record, "record_order": "sample_id_ascending", "pixel_order": "flat_index_even_spacing"},
        "dev_teacher_reproduction": {"mse": residual, "r2": r2, "spearman": correlation, "pre_registered_gate": {"r2_gte": args.minimum_r2, "spearman_gte": args.minimum_spearman, "passed": bool(r2 >= args.minimum_r2 and correlation >= args.minimum_spearman)}},
        "training_gate_report_sha256": gate["report_sha256"], "training_manifest_sha256": shared.sha256_file(dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST),
    }
    shared.write_json(report_path, report)
    Path(str(report_path) + ".sha256").write_text(shared.sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET); parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--src-root", default=DEFAULT_SRC); parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mobile-feature-weights", default=DEFAULT_MOBILE_WEIGHTS); parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vits"); parser.add_argument("--layer-index", type=int, default=11)
    parser.add_argument("--input-size", type=int, default=224); parser.add_argument("--mobile-input-size", type=int, choices=(256, 384, 512), default=384)
    parser.add_argument("--teacher-walkable-pixels-per-record", type=int, default=64); parser.add_argument("--teacher-components", type=int, default=32)
    parser.add_argument("--pixels-per-record", type=int, default=64); parser.add_argument("--ridge", type=float, default=1.0); parser.add_argument("--model-seed", type=int, default=20260715)
    parser.add_argument("--minimum-r2", type=float, default=0.50); parser.add_argument("--minimum-spearman", type=float, default=0.70)
    args = parser.parse_args(argv)
    if args.input_size <= 0 or args.input_size % 14 != 0 or not 0 <= args.layer_index <= 11:
        parser.error("DINO input must be a positive multiple of 14 and layer index must be 0..11")
    if min(args.teacher_walkable_pixels_per_record, args.teacher_components, args.pixels_per_record) <= 0 or args.ridge <= 0:
        parser.error("sample counts, components and ridge must be positive")
    if not -1.0 <= args.minimum_r2 <= 1.0 or not -1.0 <= args.minimum_spearman <= 1.0:
        parser.error("distillation gates must be in [-1, 1]")
    return args


def main() -> None:
    result = run(parse_args())["dev_teacher_reproduction"]
    print(json.dumps({"r2": result["r2"], "spearman": result["spearman"], "gate_passed": result["pre_registered_gate"]["passed"]}))


if __name__ == "__main__":
    main()
