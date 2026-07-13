#!/usr/bin/env python3
"""Run a deterministic four-class ridge probe on frozen SANPO P1-A features.

The probe consumes only the SHA256-authorized canonical train/dev manifest. It
never opens the benchmark-only blind manifest or any blind asset.  Its purpose
is diagnostic: determine whether the current frozen OS8/OS32 representation is
linearly separable before spending more GPU time on stochastic head training.
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

import sanpo_backend_equivalence
import sanpo_candidate_quality_gate as quality
import sanpo_training_gate as training_gate
import train_export_sanpo_segmentation as shared


SCHEMA = "blindassist_sanpo_deterministic_linear_probe_v1"
DEFAULT_DATASET = "test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713"
DEFAULT_FEATURE_WEIGHTS = (
    "test-artifacts.local/segmentation-candidate/"
    "p1-sigmoid-no-pooled-bn-20260713/candidate.weights.h5"
)
DEFAULT_REPORT = (
    "test-artifacts.local/segmentation-candidate/"
    "deterministic-linear-probe-20260713/probe_report.json"
)


def evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count < 0 or limit <= 0:
        raise ValueError("count must be non-negative and limit must be positive")
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=limit, dtype=np.int64)


def balance_samples(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    *,
    class_count: int,
    maximum_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(features) != len(labels) or len(labels) != len(sample_ids):
        raise ValueError("features, labels and sample_ids must be aligned")
    selected: list[int] = []
    for class_id in range(class_count):
        candidates = np.flatnonzero(labels == class_id)
        candidates = np.asarray(
            sorted(candidates.tolist(), key=lambda index: str(sample_ids[index])),
            dtype=np.int64,
        )
        if not len(candidates):
            raise ValueError(f"linear probe has no samples for class {class_id}")
        selected.extend(candidates[evenly_spaced_indices(len(candidates), maximum_per_class)].tolist())
    indices = np.asarray(selected, dtype=np.int64)
    return features[indices], labels[indices], sample_ids[indices]


def _coefficient_sha256(kernel: np.ndarray, bias: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(kernel, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(bias, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def fit_ridge_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    ridge: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y_ids = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y_ids) or not len(x):
        raise ValueError("ridge features must be a non-empty aligned matrix")
    if ridge <= 0 or np.any(y_ids < 0) or np.any(y_ids >= class_count):
        raise ValueError("ridge must be positive and labels must be in range")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (x - mean) / scale
    design = np.concatenate([standardized, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    targets = np.eye(class_count, dtype=np.float64)[y_ids]
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    standardized_kernel = coefficients[:-1]
    standardized_bias = coefficients[-1]
    raw_kernel = standardized_kernel / scale[:, None]
    raw_bias = standardized_bias - (mean / scale) @ standardized_kernel
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "standardized_kernel": standardized_kernel,
        "standardized_bias": standardized_bias,
        "kernel": raw_kernel,
        "bias": raw_bias,
        "coefficient_sha256": _coefficient_sha256(raw_kernel, raw_bias),
    }


def predict_labels(features: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(features) @ kernel + bias, axis=-1)


def repeat_consistency(
    repeats: Sequence[tuple[np.ndarray, np.ndarray, Sequence[np.ndarray]]],
) -> dict[str, Any]:
    if len(repeats) < 2:
        raise ValueError("repeat consistency requires at least two runs")
    reference_kernel, reference_bias, reference_predictions = repeats[0]
    coefficient_differences: list[float] = []
    agreements: list[float] = []
    exact = True
    for kernel, bias, predictions in repeats[1:]:
        exact = exact and np.array_equal(reference_kernel, kernel) and np.array_equal(reference_bias, bias)
        coefficient_differences.append(float(max(
            np.max(np.abs(reference_kernel - kernel)),
            np.max(np.abs(reference_bias - bias)),
        )))
        left = np.concatenate([np.asarray(item).reshape(-1) for item in reference_predictions])
        right = np.concatenate([np.asarray(item).reshape(-1) for item in predictions])
        agreements.append(float(np.mean(left == right)))
    return {
        "repeat_count": len(repeats),
        "exact_coefficient_match": exact,
        "maximum_coefficient_absolute_difference": max(coefficient_differences),
        "dev_argmax_agreement": min(agreements),
    }


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").resize(
            (width, height), Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )


def _fixed_pixels(
    feature_map: np.ndarray,
    target: np.ndarray,
    sample_id: str,
    per_class: int,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    flat_features = feature_map.reshape(-1, feature_map.shape[-1])
    flat_target = target.reshape(-1)
    for class_id in range(len(shared.CLASS_NAMES)):
        candidates = np.flatnonzero(flat_target == class_id)
        chosen = candidates[evenly_spaced_indices(len(candidates), per_class)]
        for index in chosen:
            vectors.append(flat_features[index])
            labels.append(class_id)
            ids.append(f"{sample_id}:{class_id}:{int(index):08d}")
    return vectors, labels, ids


def _extract(
    feature_model: Any,
    records: Sequence[shared.Record],
    *,
    input_size: int,
    per_class: int | None,
    batch_size: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[np.ndarray], list[np.ndarray]]:
    sampled_features: list[np.ndarray] = []
    sampled_labels: list[int] = []
    sampled_ids: list[str] = []
    feature_maps: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ordered = sorted(records, key=lambda item: item.sample_id)
    for start in range(0, len(ordered), batch_size):
        batch_records = ordered[start:start + batch_size]
        examples = [shared.load_example(record, input_size) for record in batch_records]
        images = np.stack([item[0] for item in examples])
        batch_targets = [item[1] for item in examples]
        batch_features = feature_model.predict(images, batch_size=batch_size, verbose=0)
        for record, target, raw_feature_map in zip(batch_records, batch_targets, batch_features):
            feature_map = np.asarray(raw_feature_map, dtype=np.float32)
            feature_maps.append(feature_map)
            targets.append(target)
            if per_class is not None:
                small_target = _resize_mask(target, feature_map.shape[1], feature_map.shape[0])
                vectors, labels, ids = _fixed_pixels(feature_map, small_target, record.sample_id, per_class)
                sampled_features.extend(vectors)
                sampled_labels.extend(labels)
                sampled_ids.extend(ids)
    if per_class is None:
        return None, None, None, feature_maps, targets
    return (
        np.asarray(sampled_features, dtype=np.float64),
        np.asarray(sampled_labels, dtype=np.int64),
        np.asarray(sampled_ids),
        feature_maps,
        targets,
    )


def _dev_predictions(
    feature_maps: Sequence[np.ndarray],
    kernel: np.ndarray,
    bias: np.ndarray,
    input_size: int,
) -> list[np.ndarray]:
    predictions: list[np.ndarray] = []
    for feature_map in feature_maps:
        logits = np.asarray(feature_map @ kernel + bias, dtype=np.float32)
        resized = np.stack([
            np.asarray(
                Image.fromarray(logits[..., class_id], mode="F").resize(
                    (input_size, input_size), Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
            for class_id in range(logits.shape[-1])
        ], axis=-1)
        predictions.append(np.argmax(resized, axis=-1).astype(np.uint8))
    return predictions


def _write_bootstrap(
    model: Any,
    kernel: np.ndarray,
    bias: np.ndarray,
    output: Path,
    report_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    layer = model.get_layer("semantic_logits")
    expected_kernel, expected_bias = layer.get_weights()
    candidate_kernel = np.asarray(kernel, dtype=np.float32).reshape(expected_kernel.shape)
    candidate_bias = np.asarray(bias, dtype=np.float32).reshape(expected_bias.shape)
    layer.set_weights([candidate_kernel, candidate_bias])
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(output)
    manifest = {
        "schema": "blindassist_sanpo_ridge_bootstrap_v1",
        "benchmark_only": True,
        "promotion": "do_not_replace_default_model",
        "weights": str(output),
        "weights_sha256": shared.sha256_file(output),
        "probe_report": str(report_path),
        "model_config": config,
        "initialized_layer": "semantic_logits",
        "format": "full_graph_keras_weights_h5",
        "suggested_seed_pairs": [
            "20260711:20260711", "20260712:20260711", "20260713:20260711",
            "20260711:20260712", "20260711:20260713",
        ],
        "consumer_boundary": (
            "Graph-compatible artifact only; the current trainer has no initial-weights CLI. "
            "A future isolated runner must explicitly load it before the five short runs."
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".bootstrap.json")
    shared.write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path), "manifest_sha256": shared.sha256_file(manifest_path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = shared.resolve(root, args.dataset_root).resolve()
    report_path = shared.resolve(root, args.report).resolve()
    weights_path = shared.resolve(root, args.feature_weights).resolve()
    gate_path = shared.resolve(dataset_root, args.training_gate_report).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"feature weights do not exist: {weights_path}")
    gate = training_gate.consume_training_authorization(dataset_root, gate_path)
    manifest = dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST
    records = sorted(shared.load_records(manifest), key=lambda item: item.sample_id)
    train_records = shared.records_by_split(records, "train")
    dev_records = shared.records_by_split(records, "dev")

    os.environ["KERAS_BACKEND"] = args.backend
    import keras

    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    keras.utils.set_random_seed(args.model_seed)
    if args.backend == "torch":
        import torch
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = shared.sanpo_segmentation_model.build_mobilenetv3_lraspp(
        keras,
        args.input_size,
        backbone_alpha=args.backbone_alpha,
        decoder_channels=args.decoder_channels,
        detail_output_stride=8,
        semantic_output_stride=32,
    )
    model.load_weights(weights_path)
    if args.feature_layer == "backbone_os8_os32":
        low = model.get_layer("activation_1").output
        high = model.get_layer("activation_17").output
        scale = int(low.shape[1]) // int(high.shape[1])
        high = keras.layers.UpSampling2D(
            size=(scale, scale), interpolation="bilinear", name="probe_high_up",
        )(high)
        feature_output = keras.layers.Concatenate(name="probe_backbone_os8_os32")([low, high])
        feature_layer_name = "activation_1+activation_17_bilinear_up"
    else:
        feature_output = model.get_layer(args.feature_layer).output
        feature_layer_name = args.feature_layer
    feature_model = keras.Model(model.input, feature_output)
    train_features, train_labels, train_ids, _, _ = _extract(
        feature_model,
        train_records,
        input_size=args.input_size,
        per_class=args.pixels_per_class_per_record,
        batch_size=args.feature_batch_size,
    )
    _, _, _, dev_features, dev_targets = _extract(
        feature_model,
        dev_records,
        input_size=args.input_size,
        per_class=None,
        batch_size=args.feature_batch_size,
    )
    assert train_features is not None and train_labels is not None and train_ids is not None
    balanced_features, balanced_labels, balanced_ids = balance_samples(
        train_features,
        train_labels,
        train_ids,
        class_count=len(shared.CLASS_NAMES),
        maximum_per_class=args.maximum_samples_per_class,
    )

    repeat_values: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []
    fitted: dict[str, Any] | None = None
    for _ in range(args.repeats):
        fitted = fit_ridge_probe(
            balanced_features,
            balanced_labels,
            class_count=len(shared.CLASS_NAMES),
            ridge=args.ridge,
        )
        predictions = _dev_predictions(
            dev_features, fitted["kernel"], fitted["bias"], args.input_size,
        )
        repeat_values.append((fitted["kernel"], fitted["bias"], predictions))
    assert fitted is not None
    predictions = repeat_values[0][2]
    metrics = quality.stratified_metrics(dev_records, predictions, dev_targets)
    global_metrics = metrics["global"]
    boundary_iou = global_metrics["per_class"]["boundary_step_curb"]["iou"] or 0.0
    separable = bool(
        (global_metrics["mean_iou"] or 0.0) >= args.separable_mean_iou
        and boundary_iou >= args.separable_boundary_iou
    )
    config = sanpo_backend_equivalence.model_config(
        args.backbone_alpha, args.decoder_channels, args.input_size, 8, 32,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_only": True,
        "promotion": "do_not_replace_default_model",
        "backend": args.backend,
        "blind_holdout_access": "not_accessed_by_probe",
        "dataset_root": str(dataset_root),
        "training_manifest": str(manifest),
        "training_manifest_sha256": shared.sha256_file(manifest),
        "training_gate_report": str(gate_path),
        "training_gate_report_sha256": gate["report_sha256"],
        "record_counts": {"train": len(train_records), "dev": len(dev_records)},
        "session_counts": {
            "train": len({item.session_id for item in train_records}),
            "dev": len({item.session_id for item in dev_records}),
        },
        "frozen_feature_source": {
            "layer": feature_layer_name,
            "layer_output_shape": [
                None if value is None else int(value) for value in feature_output.shape
            ],
            "weights": str(weights_path),
            "weights_sha256": shared.sha256_file(weights_path),
            "model_config": config,
            "model_config_sha256": sanpo_backend_equivalence.model_config_sha256(config),
            "trainable_parameters": 0,
        },
        "determinism": {
            "record_order": "sample_id_ascending",
            "pixel_order": "class_then_flat_index_even_spacing",
            "class_balance": "equal_maximum_per_class_after_sample_id_sort",
            "solver": "numpy_float64_closed_form_ridge",
            "model_seed_for_frozen_graph": args.model_seed,
            "repeats": repeat_consistency(repeat_values),
        },
        "sampling": {
            "pixels_per_class_per_record": args.pixels_per_class_per_record,
            "maximum_samples_per_class": args.maximum_samples_per_class,
            "selected_total": len(balanced_labels),
            "selected_per_class": {
                name: int(np.count_nonzero(balanced_labels == class_id))
                for class_id, name in enumerate(shared.CLASS_NAMES)
            },
            "selected_sample_id_sha256": hashlib.sha256(
                "\n".join(str(value) for value in balanced_ids).encode("utf-8")
            ).hexdigest(),
        },
        "ridge": {
            "lambda": args.ridge,
            "feature_dimension": int(balanced_features.shape[1]),
            "coefficient_sha256": fitted["coefficient_sha256"],
        },
        "dev_metrics": metrics,
        "linear_separability_gate": {
            "passed": separable,
            "thresholds": {
                "global_mean_iou_gte": args.separable_mean_iou,
                "boundary_iou_gte": args.separable_boundary_iou,
            },
        },
        "bootstrap": None,
    }
    if separable and args.bootstrap_output:
        if args.feature_layer != "lraspp_fuse":
            raise ValueError(
                "bootstrap-output currently requires --feature-layer lraspp_fuse; "
                "raw-backbone separability must be followed by an explicit head bootstrap design"
            )
        bootstrap_path = shared.resolve(root, args.bootstrap_output).resolve()
        report["bootstrap"] = _write_bootstrap(
            model, fitted["kernel"], fitted["bias"], bootstrap_path, report_path, config,
        )
    shared.write_json(report_path, report)
    Path(str(report_path) + ".sha256").write_text(shared.sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--feature-weights", default=DEFAULT_FEATURE_WEIGHTS)
    parser.add_argument("--backend", choices=("torch", "tensorflow"), default="torch")
    parser.add_argument(
        "--feature-layer",
        default="activation_1",
        help=(
            "Frozen feature layer to probe. The default is the raw OS8 MobileNetV3 "
            "endpoint; backbone_os8_os32 concatenates both raw backbone endpoints; "
            "lraspp_fuse is a secondary trained-head diagnostic only."
        ),
    )
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-output")
    parser.add_argument("--input-size", type=int, choices=(256, 384, 512), default=384)
    parser.add_argument("--backbone-alpha", type=float, choices=(0.75, 1.0), default=1.0)
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--pixels-per-class-per-record", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--maximum-samples-per-class", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--separable-mean-iou", type=float, default=0.35)
    parser.add_argument("--separable-boundary-iou", type=float, default=0.20)
    args = parser.parse_args(argv)
    if min(
        args.decoder_channels,
        args.pixels_per_class_per_record,
        args.feature_batch_size,
        args.maximum_samples_per_class,
        args.repeats,
    ) <= 0 or args.ridge <= 0:
        parser.error("positive decoder/sample/repeat counts and ridge are required")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    return args


def main() -> None:
    report = run(parse_args())
    metrics = report["dev_metrics"]
    print(f"global_mean_iou={metrics['global']['mean_iou']:.6f}")
    print(f"macro_session_mean_iou={metrics['macro_session_mean_iou']:.6f}")
    print(f"worst_session_mean_iou={metrics['worst_session_mean_iou']:.6f}")
    print(f"worst_scene_mean_iou={metrics['worst_scene_mean_iou']:.6f}")
    print(f"repeat_exact={report['determinism']['repeats']['exact_coefficient_match']}")
    print(f"linear_separable={report['linear_separability_gate']['passed']}")
    print("promotion=do_not_replace_default_model")


if __name__ == "__main__":
    main()
