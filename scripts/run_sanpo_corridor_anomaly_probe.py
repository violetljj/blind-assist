#!/usr/bin/env python3
"""Diagnose whether frozen DINO features support a train-only corridor familiarity score.

This is not an alert model.  It fits a PCA reconstruction subspace to only
source-semantic walkable pixels from canonical train and reports source-semantic
outlier ranking on canonical dev.  No event, risk, lifecycle, pseudo, or blind
label is read or created; no feature or application weight is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import sanpo_depth_anything_linear_probe as depth_probe
import sanpo_training_gate as training_gate
import train_export_sanpo_segmentation as shared


SCHEMA = "blindassist_sanpo_corridor_anomaly_probe_v1"
DEFAULT_DATASET = "test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713"
DEFAULT_SRC = "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main"
DEFAULT_CHECKPOINT = "artifacts.local/downloads/depth-lab/checkpoints/depth_anything_v2_vits.pth"
DEFAULT_REPORT = "artifacts.local/evidence/sanpo-corridor-anomaly-probe-20260715/report.json"


def even_indices(mask: np.ndarray, *, count: int) -> np.ndarray:
    """Select deterministic, evenly spaced flat indices from a boolean mask."""
    if mask.ndim != 2 or count <= 0:
        raise ValueError("mask must be H×W and count must be positive")
    available = np.flatnonzero(mask.reshape(-1))
    if not len(available):
        return np.empty((0,), dtype=np.int64)
    positions = np.linspace(0, len(available) - 1, min(count, len(available)), dtype=np.int64)
    return available[positions]


def fit_pca_reconstruction(features: np.ndarray, *, components: int) -> dict[str, np.ndarray]:
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("need at least two feature vectors")
    if not 1 <= components < min(features.shape):
        raise ValueError("components must be between 1 and min(samples, features)-1")
    mean = features.mean(axis=0, dtype=np.float64)
    centered = features.astype(np.float64) - mean
    _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
    return {"mean": mean, "components": right[:components]}


def reconstruction_error(features: np.ndarray, fitted: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    centered = values - fitted["mean"]
    projected = (centered @ fitted["components"].T) @ fitted["components"]
    return np.mean(np.square(centered - projected), axis=1)


def binary_auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Tie-aware AUROC where a higher score denotes an outlier."""
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    if scores.ndim != 1 or positives.shape != scores.shape:
        raise ValueError("scores and positives must be matching 1-D arrays")
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC needs both positives and negatives")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return float((ranks[positives].sum() - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count))


def _sample_vectors(feature_map: np.ndarray, target: np.ndarray, *, class_id: int, count: int) -> np.ndarray:
    indices = even_indices(target == class_id, count=count)
    return feature_map.reshape(-1, feature_map.shape[-1])[indices].astype(np.float64, copy=False)


def _extract_samples(
    model: Any, records: Iterable[shared.Record], *, input_size: int, layer_index: int, per_class: int,
) -> tuple[dict[int, list[np.ndarray]], dict[int, int]]:
    samples: dict[int, list[np.ndarray]] = {index: [] for index in range(len(shared.CLASS_NAMES))}
    counts: dict[int, int] = {index: 0 for index in range(len(shared.CLASS_NAMES))}
    for record in sorted(records, key=lambda item: item.sample_id):
        feature_map, target = depth_probe._features_for_record(
            model, record, input_size=input_size, layer_index=layer_index,
            mobile_feature_model=None, mobile_input_size=0,
        )
        for class_id in range(len(shared.CLASS_NAMES)):
            vectors = _sample_vectors(feature_map, target, class_id=class_id, count=per_class)
            if len(vectors):
                samples[class_id].append(vectors)
                counts[class_id] += len(vectors)
    return samples, counts


def _stack(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("required source-semantic class has no sampled vectors")
    return np.concatenate(rows, axis=0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = shared.resolve(root, args.dataset_root).resolve()
    source_root = shared.resolve(root, args.src_root).resolve()
    checkpoint = shared.resolve(root, args.checkpoint).resolve()
    report_path = shared.resolve(root, args.report).resolve()
    if not source_root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("Depth Anything source root or checkpoint is missing")
    gate_path = shared.resolve(dataset_root, args.training_gate_report).resolve()
    gate = training_gate.consume_training_authorization(dataset_root, gate_path)
    records = sorted(shared.load_records(dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST), key=lambda item: item.sample_id)
    train_records = shared.records_by_split(records, "train")
    dev_records = shared.records_by_split(records, "dev")

    import torch
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(source_root, checkpoint, args.encoder)
    model.eval()
    with torch.no_grad():
        train_samples, train_counts = _extract_samples(
            model, train_records, input_size=args.input_size, layer_index=args.layer_index,
            per_class=args.train_pixels_per_record,
        )
        dev_samples, dev_counts = _extract_samples(
            model, dev_records, input_size=args.input_size, layer_index=args.layer_index,
            per_class=args.dev_pixels_per_class_per_record,
        )
    walkable = _stack(train_samples[0])
    fitted = fit_pca_reconstruction(walkable, components=args.components)
    dev_by_class = {class_id: _stack(rows) for class_id, rows in dev_samples.items()}
    walkable_scores = reconstruction_error(dev_by_class[0], fitted)
    auc_by_class: dict[str, float] = {}
    for class_id, name in enumerate(shared.CLASS_NAMES[1:], start=1):
        scores = np.concatenate([walkable_scores, reconstruction_error(dev_by_class[class_id], fitted)])
        labels = np.concatenate([np.zeros(len(walkable_scores), dtype=bool), np.ones(len(scores) - len(walkable_scores), dtype=bool)])
        auc_by_class[name] = binary_auc(scores, labels)
    all_nonwalkable = np.concatenate([dev_by_class[index] for index in range(1, len(shared.CLASS_NAMES))])
    composite_scores = np.concatenate([walkable_scores, reconstruction_error(all_nonwalkable, fitted)])
    composite_labels = np.concatenate([np.zeros(len(walkable_scores), dtype=bool), np.ones(len(all_nonwalkable), dtype=bool)])
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_only": True,
        "promotion": "do_not_replace_default_model",
        "risk_or_event_truth_present": False,
        "output_interpretation": "source-semantic anomaly diagnostic only; any future high score is unknown_motion_or_surface, never an alert",
        "access_contract": {
            "canonical_rows_read": ["train", "dev"],
            "blind_holdout_access": "not_accessed_by_probe",
            "weights_saved": False,
            "pixel_supervision_role": "auxiliary_source_geometry_only",
        },
        "dataset": {
            "root": str(dataset_root),
            "manifest_sha256": shared.sha256_file(dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST),
            "training_gate_report_sha256": gate["report_sha256"],
            "record_counts": {"train": len(train_records), "dev": len(dev_records)},
            "sample_counts": {
                "train_by_source_class": {shared.CLASS_NAMES[key]: value for key, value in train_counts.items()},
                "dev_by_source_class": {shared.CLASS_NAMES[key]: value for key, value in dev_counts.items()},
            },
        },
        "frozen_feature_source": {
            "model": "Depth Anything V2", "encoder": args.encoder, "layer_index": args.layer_index,
            "input_size": args.input_size, "checkpoint_sha256": shared.sha256_file(checkpoint), "trainable_parameters": 0,
        },
        "familiarity_fit": {
            "fit_population": "canonical_train source_semantic walkable only",
            "method": "centered PCA reconstruction error", "components": args.components,
            "fit_sample_count": len(walkable), "feature_dimension": int(walkable.shape[1]),
        },
        "dev_source_semantic_auroc": {
            "higher_score_is_outlier": True,
            "against_walkable": auc_by_class,
            "all_nonwalkable_against_walkable": binary_auc(composite_scores, composite_labels),
            "pre_registered_interpretation_gate": {
                "unknown_nonwalkable_gte": args.minimum_unknown_auc,
                "boundary_step_curb_gte": args.minimum_boundary_auc,
                "passed": bool(
                    auc_by_class.get("unknown_nonwalkable", 0.0) >= args.minimum_unknown_auc
                    and auc_by_class.get("boundary_step_curb", 0.0) >= args.minimum_boundary_auc
                ),
            },
        },
    }
    shared.write_json(report_path, report)
    Path(str(report_path) + ".sha256").write_text(shared.sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--src-root", default=DEFAULT_SRC)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vits")
    parser.add_argument("--layer-index", type=int, default=11)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--train-pixels-per-record", type=int, default=64)
    parser.add_argument("--dev-pixels-per-class-per-record", type=int, default=64)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--model-seed", type=int, default=20260715)
    parser.add_argument("--minimum-unknown-auc", type=float, default=0.80)
    parser.add_argument("--minimum-boundary-auc", type=float, default=0.65)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    if args.input_size <= 0 or args.input_size % 14 != 0:
        parser.error("--input-size must be a positive multiple of 14")
    if not 0 <= args.layer_index <= 11:
        parser.error("--layer-index must be in 0..11")
    if min(args.train_pixels_per_record, args.dev_pixels_per_class_per_record, args.components) <= 0:
        parser.error("sample counts and --components must be positive")
    if not 0.5 <= args.minimum_unknown_auc <= 1.0 or not 0.5 <= args.minimum_boundary_auc <= 1.0:
        parser.error("AUROC gates must be in [0.5, 1.0]")
    return args


def main() -> None:
    report = run(parse_args())
    metrics = report["dev_source_semantic_auroc"]
    print(json.dumps({"all_nonwalkable_auc": metrics["all_nonwalkable_against_walkable"], "gate_passed": metrics["pre_registered_interpretation_gate"]["passed"]}))


if __name__ == "__main__":
    main()
