#!/usr/bin/env python3
"""Diagnose SANPO class separability with frozen official Depth Anything V2 features.

This is deliberately a deterministic, train/dev-only diagnostic.  It does not
fine-tune Depth Anything, consume the blind holdout, manufacture event labels,
or authorize an application model change.  Its value is causal: compare a
general geometry-aware foundation representation with the existing MobileNet
backbone probe before treating the failure as a head-only optimization issue.
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

import sanpo_backend_equivalence
import sanpo_candidate_quality_gate as quality
import sanpo_deterministic_linear_probe as ridge
import sanpo_training_gate as training_gate
import smoke_depth_anything_v2_pytorch as depth_anything
import train_export_sanpo_segmentation as shared


SCHEMA = "blindassist_sanpo_depth_anything_linear_probe_v1"
DEFAULT_DATASET = "test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713"
DEFAULT_SRC = "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main"
DEFAULT_CHECKPOINT = "artifacts.local/downloads/depth-lab/checkpoints/depth_anything_v2_vits.pth"
DEFAULT_REPORT = "artifacts.local/evidence/sanpo-depth-anything-linear-probe-20260715/probe_report.json"


def tokens_to_feature_map(tokens: Any, *, patch_height: int, patch_width: int) -> np.ndarray:
    """Convert one DINO patch-token tensor to H×W×C without interpolation."""
    array = np.asarray(tokens.detach().cpu().numpy() if hasattr(tokens, "detach") else tokens, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] != patch_height * patch_width:
        raise ValueError(
            f"expected {patch_height}×{patch_width} patch tokens, got shape {tuple(array.shape)}"
        )
    return array.reshape(patch_height, patch_width, array.shape[-1])


def resize_feature_map(feature_map: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if feature_map.ndim != 3:
        raise ValueError("feature map must be H×W×C")
    if feature_map.shape[:2] == (height, width):
        return feature_map
    resized = cv2.resize(feature_map, (width, height), interpolation=cv2.INTER_LINEAR)
    if resized.ndim != 3 or resized.shape != (height, width, feature_map.shape[-1]):
        raise ValueError(f"feature resize produced unexpected shape {tuple(resized.shape)}")
    return np.asarray(resized, dtype=np.float32)


def _features_for_record(
    model: Any,
    record: shared.Record,
    *,
    input_size: int,
    layer_index: int,
    mobile_feature_model: Any | None,
    mobile_input_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot decode SANPO image: {record.image_path}")
    tensor, _original_size = model.image2tensor(image, input_size=input_size)
    patch_height, patch_width = tensor.shape[-2] // 14, tensor.shape[-1] // 14
    outputs = model.pretrained.get_intermediate_layers(tensor, [layer_index], return_class_token=True)
    tokens = outputs[0][0]
    feature_map = tokens_to_feature_map(tokens, patch_height=patch_height, patch_width=patch_width)
    target = shared.validate_binary_masks(record)
    if mobile_feature_model is not None:
        mobile_image, _unused_target = shared.load_example(record, mobile_input_size)
        mobile_map = np.asarray(mobile_feature_model.predict(mobile_image[None, ...], verbose=0)[0], dtype=np.float32)
        if mobile_map.ndim != 3:
            raise ValueError(f"MobileNet feature model must return H×W×C, got {tuple(mobile_map.shape)}")
        feature_map = np.concatenate(
            [resize_feature_map(feature_map, height=mobile_map.shape[0], width=mobile_map.shape[1]), mobile_map],
            axis=-1,
        )
    small_target = ridge._resize_mask(target, feature_map.shape[1], feature_map.shape[0])
    return feature_map, small_target


def _extract(
    model: Any,
    records: Sequence[shared.Record],
    *,
    input_size: int,
    layer_index: int,
    per_class: int | None,
    mobile_feature_model: Any | None,
    mobile_input_size: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[np.ndarray], list[np.ndarray]]:
    sampled_features: list[np.ndarray] = []
    sampled_labels: list[int] = []
    sampled_ids: list[str] = []
    feature_maps: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in sorted(records, key=lambda item: item.sample_id):
        feature_map, small_target = _features_for_record(
            model, record, input_size=input_size, layer_index=layer_index,
            mobile_feature_model=mobile_feature_model, mobile_input_size=mobile_input_size,
        )
        feature_maps.append(feature_map)
        targets.append(small_target)
        if per_class is not None:
            vectors, labels, ids = ridge._fixed_pixels(
                feature_map, small_target, record.sample_id, per_class,
            )
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


def _dev_predictions(feature_maps: Sequence[np.ndarray], kernel: np.ndarray, bias: np.ndarray) -> list[np.ndarray]:
    return [ridge.predict_labels(feature_map, kernel, bias).astype(np.uint8) for feature_map in feature_maps]


def _mobile_feature_model(args: argparse.Namespace) -> tuple[Any | None, dict[str, Any] | None]:
    if not args.append_mobile_os8_os32:
        return None, None
    root = shared.project_root()
    weights = shared.resolve(root, args.mobile_feature_weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"MobileNet feature weights are missing: {weights}")
    os.environ["KERAS_BACKEND"] = args.mobile_backend
    import keras

    keras.utils.set_random_seed(args.model_seed)
    model = shared.sanpo_segmentation_model.build_mobilenetv3_lraspp(
        keras,
        args.mobile_input_size,
        backbone_alpha=args.mobile_backbone_alpha,
        decoder_channels=args.mobile_decoder_channels,
        detail_output_stride=8,
        semantic_output_stride=32,
    )
    model.load_weights(weights)
    low = model.get_layer("activation_1").output
    high = model.get_layer("activation_17").output
    scale = int(low.shape[1]) // int(high.shape[1])
    high = keras.layers.UpSampling2D(size=(scale, scale), interpolation="bilinear")(high)
    output = keras.layers.Concatenate()([low, high])
    feature_model = keras.Model(model.input, output)
    config = sanpo_backend_equivalence.model_config(
        args.mobile_backbone_alpha, args.mobile_decoder_channels, args.mobile_input_size, 8, 32,
    )
    return feature_model, {
        "model": "P1-A MobileNetV3 LR-ASPP raw OS8+OS32",
        "weights": str(weights),
        "weights_sha256": shared.sha256_file(weights),
        "backend": args.mobile_backend,
        "model_config": config,
        "model_config_sha256": sanpo_backend_equivalence.model_config_sha256(config),
        "trainable_parameters": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = shared.resolve(root, args.dataset_root).resolve()
    source_root = shared.resolve(root, args.src_root).resolve()
    checkpoint = shared.resolve(root, args.checkpoint).resolve()
    report_path = shared.resolve(root, args.report).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Depth Anything source root is missing: {source_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Depth Anything checkpoint is missing: {checkpoint}")

    gate_path = shared.resolve(dataset_root, args.training_gate_report).resolve()
    gate = training_gate.consume_training_authorization(dataset_root, gate_path)
    manifest = dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST
    records = sorted(shared.load_records(manifest), key=lambda item: item.sample_id)
    train_records = shared.records_by_split(records, "train")
    dev_records = shared.records_by_split(records, "dev")

    import torch

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.use_deterministic_algorithms(True)
    model = depth_anything.load_model(source_root, checkpoint, args.encoder)
    model.eval()
    mobile_feature_model, mobile_source = _mobile_feature_model(args)
    with torch.no_grad():
        train_features, train_labels, train_ids, _, _ = _extract(
            model, train_records, input_size=args.input_size,
            layer_index=args.layer_index, per_class=args.pixels_per_class_per_record,
            mobile_feature_model=mobile_feature_model, mobile_input_size=args.mobile_input_size,
        )
        _, _, _, dev_features, dev_targets = _extract(
            model, dev_records, input_size=args.input_size,
            layer_index=args.layer_index, per_class=None,
            mobile_feature_model=mobile_feature_model, mobile_input_size=args.mobile_input_size,
        )
    assert train_features is not None and train_labels is not None and train_ids is not None
    balanced_features, balanced_labels, balanced_ids = ridge.balance_samples(
        train_features, train_labels, train_ids,
        class_count=len(shared.CLASS_NAMES), maximum_per_class=args.maximum_samples_per_class,
    )
    repeats: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []
    fitted: dict[str, Any] | None = None
    for _ in range(args.repeats):
        fitted = ridge.fit_ridge_probe(
            balanced_features, balanced_labels, class_count=len(shared.CLASS_NAMES), ridge=args.ridge,
        )
        predictions = _dev_predictions(dev_features, fitted["kernel"], fitted["bias"])
        repeats.append((fitted["kernel"], fitted["bias"], predictions))
    assert fitted is not None
    metrics = quality.stratified_metrics(dev_records, repeats[0][2], dev_targets)
    boundary_iou = metrics["global"]["per_class"]["boundary_step_curb"]["iou"] or 0.0
    separable = bool(
        (metrics["global"]["mean_iou"] or 0.0) >= args.separable_mean_iou
        and boundary_iou >= args.separable_boundary_iou
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_only": True,
        "promotion": "do_not_replace_default_model",
        "diagnostic_only": "No feature, decoder, or app weights are trained or written.",
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
            "model": "Depth Anything V2",
            "encoder": args.encoder,
            "intermediate_layer": args.layer_index,
            "input_size": args.input_size,
            "source_root": str(source_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": shared.sha256_file(checkpoint),
            "trainable_parameters": 0,
            "mobile_feature_append": mobile_source,
        },
        "determinism": {
            "record_order": "sample_id_ascending",
            "pixel_order": "class_then_flat_index_even_spacing",
            "class_balance": "equal_maximum_per_class_after_sample_id_sort",
            "solver": "numpy_float64_closed_form_ridge",
            "model_seed": args.model_seed,
            "repeats": ridge.repeat_consistency(repeats),
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
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--pixels-per-class-per-record", type=int, default=16)
    parser.add_argument("--maximum-samples-per-class", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--model-seed", type=int, default=20260715)
    parser.add_argument("--separable-mean-iou", type=float, default=0.35)
    parser.add_argument("--separable-boundary-iou", type=float, default=0.20)
    parser.add_argument("--append-mobile-os8-os32", action="store_true")
    parser.add_argument(
        "--mobile-feature-weights",
        default="test-artifacts.local/segmentation-candidate/p1-sigmoid-no-pooled-bn-20260713/candidate.weights.h5",
    )
    parser.add_argument("--mobile-backend", choices=("torch", "tensorflow"), default="torch")
    parser.add_argument("--mobile-input-size", type=int, choices=(256, 384, 512), default=384)
    parser.add_argument("--mobile-backbone-alpha", type=float, choices=(0.75, 1.0), default=1.0)
    parser.add_argument("--mobile-decoder-channels", type=int, default=96)
    args = parser.parse_args(argv)
    if args.input_size <= 0 or args.input_size % 14 != 0:
        parser.error("--input-size must be a positive multiple of 14")
    if not 0 <= args.layer_index <= 11:
        parser.error("vits layer index must be in 0..11")
    if min(args.pixels_per_class_per_record, args.maximum_samples_per_class, args.repeats, args.mobile_decoder_channels) <= 0 or args.ridge <= 0:
        parser.error("sample/repeat counts and ridge must be positive")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    return args


def main() -> None:
    report = run(parse_args())
    global_metrics = report["dev_metrics"]["global"]
    print(f"global_mean_iou={global_metrics['mean_iou']:.6f}")
    print(f"boundary_iou={global_metrics['per_class']['boundary_step_curb']['iou']:.6f}")
    print(f"separable={report['linear_separability_gate']['passed']}")


if __name__ == "__main__":
    main()
