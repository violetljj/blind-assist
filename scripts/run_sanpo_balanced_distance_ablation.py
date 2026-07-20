#!/usr/bin/env python3
"""Paired, train-only auxiliary-head ablation with a coverage-balanced session holdout.

This is deliberately a diagnostic runner, not the canonical trainer.  It uses
only rows already assigned to canonical ``train`` and holds out whole *train*
sessions for evaluation.  It refuses the run unless the boundary-pixel
fractions of its optimization and held-out partitions are sufficiently close.

The source masks remain pixel/geometry supervision only.  No risk, event, or
lifecycle label is constructed, canonical dev/blind are not read, no weights
are saved, and every report permanently says that production promotion is
unauthorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

import sanpo_boundary_distance_aux as distance_aux
import sanpo_training_gate as training_gate
import train_export_sanpo_segmentation as shared
import train_sanpo_segmentation_keras_torch as trainer


BOUNDARY_CLASS_ID = shared.CLASS_IDS["boundary_step_curb"]
DEFAULT_SEED_PAIRS = "2026071501:2026072501,2026071502:2026072502,2026071503:2026072503,2026071504:2026072504,2026071505:2026072505"


@dataclass(frozen=True)
class Partition:
    train_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    train_sessions: tuple[str, ...]
    evaluation_sessions: tuple[str, ...]
    train_boundary_fraction: float
    evaluation_boundary_fraction: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seed_pairs(value: str) -> tuple[tuple[int, int], ...]:
    return trainer.parse_seed_pairs(value)


def load_canonical_train_records_only(dataset_root: Path) -> list[shared.Record]:
    """Read canonical manifest rows but open image/mask assets only for ``train``.

    The manifest must be scanned to locate its train rows, but dev and blind
    image/mask assets are intentionally never resolved or read by this helper.
    The existing attested training gate is consumed separately before this is
    called.
    """
    manifest = dataset_root / "training_manifest.jsonl"
    records: list[shared.Record] = []
    seen_sessions: set[str] = set()
    for row in shared.json_lines(manifest):
        split = shared.split_for(row)
        if split != "train":
            continue
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError("canonical train row is missing id")
        label_authority = str(row.get("label_authority", "")).strip()
        if label_authority not in training_gate.validator.LABEL_AUTHORITIES:
            raise ValueError(f"{sample_id}: missing or unsupported label_authority")
        session_id = shared.session_for(row)
        if session_id in seen_sessions:
            # Repeated frames in a session are expected.  The set is retained
            # only as a cheap invariant that this path never mixes a dev row.
            pass
        seen_sessions.add(session_id)
        image_value = str(row.get("image_path", "")).strip()
        image_path = (dataset_root / image_value).resolve()
        try:
            image_path.relative_to(dataset_root.resolve())
        except ValueError as error:
            raise ValueError(f"{sample_id}: image path escapes dataset root") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"{sample_id}: missing train image {image_path}")
        masks, semantic_mask_path = shared.masks_for(row, dataset_root)
        records.append(shared.Record(
            sample_id=sample_id,
            split="train",
            session_id=session_id,
            image_path=image_path,
            masks=masks,
            semantic_mask_path=semantic_mask_path,
            scene_bucket=row.get("scene_bucket") if isinstance(row.get("scene_bucket"), str) else None,
            label_authority=label_authority,
        ))
    if not records:
        raise ValueError("canonical train contains no rows")
    return records


def _fraction(masks: Sequence[np.ndarray]) -> float:
    pixels = sum(int(mask.size) for mask in masks)
    if not pixels:
        raise ValueError("partition must contain at least one mask")
    return float(sum(int((mask == BOUNDARY_CLASS_ID).sum()) for mask in masks) / pixels)


def partition_train_sessions(
    records: Sequence[shared.Record],
    masks: Sequence[np.ndarray],
    evaluation_sessions: Sequence[str],
    *,
    minimum_ratio: float,
    maximum_ratio: float,
) -> Partition:
    """Make an all-train session holdout and prove boundary coverage is matched."""
    if len(records) != len(masks):
        raise ValueError("records and masks must have identical length")
    if not 0 < minimum_ratio <= maximum_ratio:
        raise ValueError("coverage ratio bounds must satisfy 0 < min <= max")
    requested = tuple(dict.fromkeys(str(value).strip() for value in evaluation_sessions if str(value).strip()))
    if len(requested) < 2:
        raise ValueError("at least two distinct train sessions are required for diagnostic evaluation")
    all_sessions = {record.session_id for record in records}
    missing = sorted(set(requested) - all_sessions)
    if missing:
        raise ValueError(f"evaluation sessions are not all canonical train sessions: {missing}")
    train_indices = tuple(index for index, record in enumerate(records) if record.session_id not in requested)
    evaluation_indices = tuple(index for index, record in enumerate(records) if record.session_id in requested)
    if not train_indices or not evaluation_indices:
        raise ValueError("both optimization and evaluation partitions must be non-empty")
    train_fraction = _fraction([masks[index] for index in train_indices])
    evaluation_fraction = _fraction([masks[index] for index in evaluation_indices])
    if train_fraction <= 0 or evaluation_fraction <= 0:
        raise ValueError("both partitions must contain boundary pixels")
    ratio = evaluation_fraction / train_fraction
    if not minimum_ratio <= ratio <= maximum_ratio:
        raise ValueError(
            "boundary coverage mismatch for diagnostic holdout: "
            f"evaluation/train={ratio:.6f}, required [{minimum_ratio:.6f}, {maximum_ratio:.6f}]"
        )
    train_sessions = tuple(sorted({records[index].session_id for index in train_indices}))
    return Partition(
        train_indices=train_indices,
        evaluation_indices=evaluation_indices,
        train_sessions=train_sessions,
        evaluation_sessions=tuple(sorted(requested)),
        train_boundary_fraction=train_fraction,
        evaluation_boundary_fraction=evaluation_fraction,
    )


def distance_targets(masks: np.ndarray, *, truncate: float, signed: bool) -> np.ndarray:
    """Build [target, deterministic loss weight] only after sampler crop/flip."""
    values: list[np.ndarray] = []
    for mask in masks:
        target, weight = distance_aux.smooth_l1_target_and_weight(
            mask == BOUNDARY_CLASS_ID,
            truncate=truncate,
            signed=signed,
        )
        values.append(np.stack((target, weight), axis=-1))
    return np.stack(values).astype(np.float32)


def build_distance_loss(keras: Any) -> Any:
    class WeightedSmoothL1(keras.losses.Loss):
        def call(self, y_true: Any, y_pred: Any) -> Any:
            target = keras.ops.cast(y_true[..., 0], "float32")
            weight = keras.ops.cast(y_true[..., 1], "float32")
            prediction = keras.ops.squeeze(keras.ops.cast(y_pred, "float32"), axis=-1)
            difference = keras.ops.abs(prediction - target)
            smooth_l1 = keras.ops.where(
                difference < 1.0,
                0.5 * keras.ops.square(difference),
                difference - 0.5,
            )
            return keras.ops.sum(smooth_l1 * weight) / keras.ops.maximum(keras.ops.sum(weight), 1e-7)

    return WeightedSmoothL1(name="boundary_distance_weighted_smooth_l1")


def boundary_probability_targets(masks: np.ndarray) -> np.ndarray:
    """Return a binary, full-resolution boundary target after augmentation."""
    return (masks == BOUNDARY_CLASS_ID).astype(np.float32)[..., np.newaxis]


def build_boundary_probability_loss(keras: Any) -> Any:
    """A stable positive-balanced BCE loss for a deliberately sparse boundary head."""
    class PositiveBalancedBce(keras.losses.Loss):
        def call(self, y_true: Any, y_pred: Any) -> Any:
            target = keras.ops.cast(y_true, "float32")
            logits = keras.ops.cast(y_pred, "float32")
            # max(x,0) - x*y + log(1 + exp(-abs(x))) is BCE from logits,
            # evaluated without the overflow of sigmoid/log composition.
            bce = (
                keras.ops.maximum(logits, 0.0)
                - logits * target
                + keras.ops.log(1.0 + keras.ops.exp(-keras.ops.abs(logits)))
            )
            positive_count = keras.ops.sum(target)
            negative_count = keras.ops.sum(1.0 - target)
            positive_weight = keras.ops.minimum(
                32.0,
                keras.ops.maximum(1.0, negative_count / keras.ops.maximum(positive_count, 1.0)),
            )
            weight = 1.0 + (positive_weight - 1.0) * target
            return keras.ops.sum(bce * weight) / keras.ops.maximum(keras.ops.sum(weight), 1e-7)

    return PositiveBalancedBce(name="boundary_probability_positive_balanced_bce")


def build_model(keras: Any, args: argparse.Namespace, *, auxiliary: str | None) -> Any:
    base = shared.sanpo_segmentation_model.build_mobilenetv3_lraspp(
        keras,
        args.input_size,
        len(shared.CLASS_NAMES),
        backbone_weights="imagenet",
        backbone_alpha=args.backbone_alpha,
        decoder_channels=args.decoder_channels,
        detail_output_stride=args.detail_output_stride,
        semantic_output_stride=args.semantic_output_stride,
    )
    if auxiliary is None:
        return base
    fused = base.get_layer("lraspp_fuse").output
    if auxiliary == "distance":
        head_name, field_name, model_name = (
            "distance_field_logits", "distance_field", "mobilenetv3_lraspp_distance_diagnostic",
        )
    elif auxiliary == "boundary_probability":
        head_name, field_name, model_name = (
            "boundary_probability_logits", "boundary_probability", "mobilenetv3_lraspp_boundary_probability_diagnostic",
        )
    else:
        raise ValueError(f"unknown auxiliary treatment: {auxiliary}")
    auxiliary_logits = keras.layers.Conv2D(1, 1, name=head_name)(fused)
    scale = args.input_size // int(fused.shape[1])
    auxiliary_field = keras.layers.UpSampling2D(
        size=(scale, scale), interpolation="bilinear", name=field_name,
    )(auxiliary_logits)
    return keras.Model(
        inputs=base.inputs,
        outputs=[base.output, auxiliary_field],
        name=model_name,
    )


def semantic_loss_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        focal_gamma=args.focal_gamma,
    )


def run_arm(
    *,
    keras: Any,
    torch: Any,
    args: argparse.Namespace,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    train_records: Sequence[shared.Record],
    evaluation_images: np.ndarray,
    evaluation_masks: np.ndarray,
    class_weights: np.ndarray,
    model_seed: int,
    sampler_seed: int,
    auxiliary: str | None,
) -> dict[str, Any]:
    keras.backend.clear_session()
    trainer.set_seed(keras, torch, model_seed)
    model = build_model(keras, args, auxiliary=auxiliary)
    trainer.configure_trainable_layers(
        model, keras, backbone_trainable=False, keep_batchnorm_frozen=True,
    )
    if auxiliary is not None:
        # The common configuration helper intentionally freezes every
        # non-semantic backbone layer.  This diagnostic head is the one
        # controlled treatment, so restore trainability explicitly.
        model.get_layer(
            "distance_field_logits" if auxiliary == "distance" else "boundary_probability_logits",
        ).trainable = True
    semantic_loss = trainer.build_composite_loss(keras, class_weights, semantic_loss_args(args))
    if auxiliary is not None:
        auxiliary_loss = (
            build_distance_loss(keras)
            if auxiliary == "distance"
            else build_boundary_probability_loss(keras)
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=[semantic_loss, auxiliary_loss],
            loss_weights=[1.0, args.distance_loss_weight],
            jit_compile=False,
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=semantic_loss,
            jit_compile=False,
        )
    sampler = trainer.SessionBalancedCropSampler(
        train_images,
        train_masks,
        train_records,
        batch_size=args.batch_size,
        seed=sampler_seed,
        guided_crop_fraction=args.guided_crop_fraction,
        crop_min_fraction=args.crop_min_fraction,
        crop_max_fraction=args.crop_max_fraction,
        horizontal_flip_probability=args.horizontal_flip_probability,
        boundary_guided_probability=args.boundary_guided_probability,
    )
    losses: list[float] = []
    for _ in range(args.optimizer_steps):
        batch_images, batch_masks = sampler.next_batch()
        if auxiliary is not None:
            auxiliary_target = (
                distance_targets(batch_masks, truncate=args.distance_truncate, signed=args.distance_signed)
                if auxiliary == "distance"
                else boundary_probability_targets(batch_masks)
            )
            logs = model.train_on_batch(
                batch_images,
                [batch_masks, auxiliary_target],
                return_dict=True,
            )
        else:
            logs = model.train_on_batch(batch_images, batch_masks, return_dict=True)
        losses.append(float(logs["loss"]))
        model.reset_metrics()
    predictions = model.predict(evaluation_images, batch_size=args.batch_size, verbose=0)
    semantic_logits = predictions[0] if auxiliary is not None else predictions
    metrics = shared.confusion_and_metrics(np.argmax(semantic_logits, axis=-1), evaluation_masks)
    result: dict[str, Any] = {
        "mean_training_loss": float(np.mean(losses)),
        "final_training_loss": float(losses[-1]),
        "mask_metrics": metrics,
        "selection_score": trainer.selection_score(metrics),
        "sampler": sampler.report(),
        "parameter_count": int(model.count_params()),
    }
    if auxiliary == "distance":
        prediction = np.asarray(predictions[1], dtype=np.float32)[..., 0]
        target = distance_targets(evaluation_masks, truncate=args.distance_truncate, signed=args.distance_signed)
        weight = target[..., 1]
        error = np.abs(prediction - target[..., 0])
        result["distance_evaluation"] = {
            "weighted_mae": float((error * weight).sum(dtype=np.float64) / max(float(weight.sum(dtype=np.float64)), 1e-7)),
            "mean_loss_weight": float(weight.mean(dtype=np.float64)),
        }
    elif auxiliary == "boundary_probability":
        logits = np.asarray(predictions[1], dtype=np.float32)[..., 0]
        prediction = logits >= 0.0
        target = boundary_probability_targets(evaluation_masks)[..., 0] > 0.5
        true_positive = int(np.logical_and(prediction, target).sum())
        false_positive = int(np.logical_and(prediction, ~target).sum())
        false_negative = int(np.logical_and(~prediction, target).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        result["boundary_probability_evaluation"] = {
            "threshold_logit": 0.0,
            "precision": float(true_positive / max(true_positive + false_positive, 1)),
            "recall": float(true_positive / max(true_positive + false_negative, 1)),
            "f1": float(2 * true_positive / max(denominator, 1)),
            "iou": float(true_positive / max(true_positive + false_positive + false_negative, 1)),
            "predicted_positive_pixels": int(prediction.sum()),
            "target_positive_pixels": int(target.sum()),
        }
    return result


def summarize_deltas(pairs: Sequence[dict[str, Any]], *, delta_key: str) -> dict[str, Any]:
    def values(path: Sequence[str]) -> list[float]:
        extracted: list[float] = []
        for pair in pairs:
            current: Any = pair[delta_key]
            for key in path:
                current = current[key]
            extracted.append(float(current))
        return extracted

    fields = {
        "mean_iou": ("mean_iou",),
        "boundary_iou": ("boundary_iou",),
        "selection_score": ("selection_score",),
        "unknown_iou": ("unknown_iou",),
    }
    summary: dict[str, Any] = {"pair_count": len(pairs)}
    for name, path in fields.items():
        array = np.asarray(values(path), dtype=np.float64)
        summary[name] = {
            "mean": float(array.mean()),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "values": array.tolist(),
        }
    return summary


def metric_delta(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, float]:
    baseline_metrics = baseline["mask_metrics"]
    treatment_metrics = treatment["mask_metrics"]
    return {
        "mean_iou": float(treatment_metrics["mean_iou"] - baseline_metrics["mean_iou"]),
        "boundary_iou": float(
            treatment_metrics["per_class"]["boundary_step_curb"]["iou"]
            - baseline_metrics["per_class"]["boundary_step_curb"]["iou"]
        ),
        "unknown_iou": float(
            treatment_metrics["per_class"]["unknown_nonwalkable"]["iou"]
            - baseline_metrics["per_class"]["unknown_nonwalkable"]["iou"]
        ),
        "selection_score": float(treatment["selection_score"] - baseline["selection_score"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = (root / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root).resolve()
    if "blind" in dataset_root.as_posix().lower():
        raise ValueError("refusing a dataset path containing blind")
    gate_path = (dataset_root / args.training_gate_report).resolve()
    gate_report = training_gate.consume_training_authorization(dataset_root, gate_path)
    records = load_canonical_train_records_only(dataset_root)
    examples = [shared.load_example(record, args.input_size) for record in records]
    images, masks = zip(*examples)
    images_array = np.stack(images).astype(np.uint8)
    masks_array = np.stack(masks).astype(np.uint8)
    partition = partition_train_sessions(
        records,
        masks_array,
        args.evaluation_session,
        minimum_ratio=args.minimum_boundary_coverage_ratio,
        maximum_ratio=args.maximum_boundary_coverage_ratio,
    )
    train_images = images_array[list(partition.train_indices)]
    train_masks = masks_array[list(partition.train_indices)]
    evaluation_images = images_array[list(partition.evaluation_indices)]
    evaluation_masks = masks_array[list(partition.evaluation_indices)]
    train_records = [records[index] for index in partition.train_indices]
    pixel_counts = np.bincount(train_masks.reshape(-1), minlength=len(shared.CLASS_NAMES))
    class_weights = trainer.class_loss_weights(
        {name: int(pixel_counts[index]) for index, name in enumerate(shared.CLASS_NAMES)},
        args.maximum_class_weight,
    )
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import torch

    if keras.backend.backend() != "torch" or not torch.cuda.is_available():
        raise RuntimeError("Keras torch backend with CUDA is required for this diagnostic")
    keras.mixed_precision.set_global_policy("mixed_float16")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paired_runs: list[dict[str, Any]] = []
    for model_seed, sampler_seed in args.seed_pairs:
        baseline = run_arm(
            keras=keras, torch=torch, args=args, train_images=train_images, train_masks=train_masks,
            train_records=train_records, evaluation_images=evaluation_images,
            evaluation_masks=evaluation_masks, class_weights=class_weights,
            model_seed=model_seed, sampler_seed=sampler_seed, auxiliary=None,
        )
        treatment = run_arm(
            keras=keras, torch=torch, args=args, train_images=train_images, train_masks=train_masks,
            train_records=train_records, evaluation_images=evaluation_images,
            evaluation_masks=evaluation_masks, class_weights=class_weights,
            model_seed=model_seed, sampler_seed=sampler_seed, auxiliary=args.auxiliary,
        )
        treatment_key = f"{args.auxiliary}_auxiliary"
        delta_key = f"delta_{args.auxiliary}_minus_baseline"
        paired_runs.append({
            "model_seed": model_seed,
            "sampler_seed": sampler_seed,
            "baseline": baseline,
            treatment_key: treatment,
            delta_key: metric_delta(baseline, treatment),
        })
    manifest = dataset_root / "training_manifest.jsonl"
    report = {
        "format": "blindassist_sanpo_balanced_auxiliary_ablation_v2",
        "purpose": "train_only_coverage_matched_pixel_geometry_diagnostic",
        "promotion": "do_not_replace_default_model",
        "training_execution_authorized": False,
        "production_model_replacement_authorized": False,
        "risk_or_event_truth_present": False,
        "source_mask_role": "auxiliary_pixel_geometry_only",
        "dataset_root": str(dataset_root),
        "training_manifest_sha256": sha256_file(manifest),
        "training_gate_report_sha256": gate_report["report_sha256"],
        "access_contract": {
            "canonical_rows_read": "train_only",
            "canonical_dev_access": "not_accessed",
            "blind_holdout_access": "not_accessed",
            "weights_saved": False,
        },
        "partition": {
            "optimization_sessions": list(partition.train_sessions),
            "evaluation_sessions": list(partition.evaluation_sessions),
            "optimization_frames": len(partition.train_indices),
            "evaluation_frames": len(partition.evaluation_indices),
            "optimization_boundary_fraction": partition.train_boundary_fraction,
            "evaluation_boundary_fraction": partition.evaluation_boundary_fraction,
            "evaluation_to_optimization_boundary_ratio": partition.evaluation_boundary_fraction / partition.train_boundary_fraction,
            "required_ratio_range": [args.minimum_boundary_coverage_ratio, args.maximum_boundary_coverage_ratio],
        },
        "protocol": {
            "input_size": args.input_size,
            "optimizer_steps": args.optimizer_steps,
            "batch_size": args.batch_size,
            "seed_pairs": [{"model_seed": a, "sampler_seed": b} for a, b in args.seed_pairs],
            "backbone": "MobileNetV3Small",
            "backbone_trainable": False,
            "sampler": "session_balanced_guided",
            "auxiliary": args.auxiliary,
            "auxiliary_loss_weight": args.distance_loss_weight,
            "distance_target": ({"signed": args.distance_signed, "truncate_pixels": args.distance_truncate}
                                if args.auxiliary == "distance" else None),
            "auxiliary_loss": ({"name": "weighted_smooth_l1"}
                               if args.auxiliary == "distance" else {"name": "positive_balanced_bce", "positive_weight_cap": 32.0}),
        },
        "paired_runs": paired_runs,
        "delta_summary": summarize_deltas(paired_runs, delta_key=f"delta_{args.auxiliary}_minus_baseline"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713")
    parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--evaluation-session", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-pairs", default=DEFAULT_SEED_PAIRS)
    parser.add_argument("--input-size", type=int, choices=(256, 384, 512), default=384)
    parser.add_argument("--optimizer-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--auxiliary", choices=("distance", "boundary_probability"), default="distance")
    parser.add_argument("--distance-loss-weight", type=float, default=0.20,
                        help="Loss weight for the selected auxiliary (legacy name retained for reproducibility).")
    parser.add_argument("--distance-truncate", type=float, default=16.0)
    parser.add_argument("--distance-signed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-boundary-coverage-ratio", type=float, default=0.80)
    parser.add_argument("--maximum-boundary-coverage-ratio", type=float, default=1.25)
    parser.add_argument("--guided-crop-fraction", type=float, default=0.70)
    parser.add_argument("--boundary-guided-probability", type=float, default=0.65)
    parser.add_argument("--crop-min-fraction", type=float, default=0.55)
    parser.add_argument("--crop-max-fraction", type=float, default=0.85)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.50)
    parser.add_argument("--ce-weight", type=float, default=0.50)
    parser.add_argument("--dice-weight", type=float, default=0.40)
    parser.add_argument("--focal-weight", type=float, default=0.10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--maximum-class-weight", type=float, default=4.0)
    parser.add_argument("--backbone-alpha", type=float, choices=(0.75, 1.0), default=1.0)
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--detail-output-stride", type=int, choices=(4, 8), default=8)
    parser.add_argument("--semantic-output-stride", type=int, choices=(16, 32), default=32)
    args = parser.parse_args(argv)
    args.seed_pairs = parse_seed_pairs(args.seed_pairs)
    args.evaluation_session = list(dict.fromkeys(args.evaluation_session))
    if len(args.evaluation_session) < 2:
        parser.error("at least two distinct --evaluation-session values are required")
    positive = (
        args.optimizer_steps, args.batch_size, args.learning_rate, args.distance_loss_weight,
        args.distance_truncate, args.focal_gamma, args.maximum_class_weight, args.decoder_channels,
    )
    if any(value <= 0 for value in positive):
        parser.error("steps, sizes, loss weights, learning rate, truncate, gamma and channels must be positive")
    if not 0 < args.minimum_boundary_coverage_ratio <= args.maximum_boundary_coverage_ratio:
        parser.error("coverage ratios must satisfy 0 < minimum <= maximum")
    if not 0 <= args.guided_crop_fraction <= 1 or not 0 <= args.boundary_guided_probability <= 1:
        parser.error("guided crop probabilities must be in [0, 1]")
    if not 0 < args.crop_min_fraction <= args.crop_max_fraction <= 1:
        parser.error("crop fractions must satisfy 0 < minimum <= maximum <= 1")
    if not 0 <= args.horizontal_flip_probability <= 1:
        parser.error("horizontal flip probability must be in [0, 1]")
    if min(args.ce_weight, args.dice_weight, args.focal_weight) < 0 or abs(args.ce_weight + args.dice_weight + args.focal_weight - 1.0) > 1e-6:
        parser.error("semantic loss weights must be non-negative and sum to one")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={args.output}")
    print(f"boundary_iou_delta_mean={report['delta_summary']['boundary_iou']['mean']:.6f}")
    print("promotion=do_not_replace_default_model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
