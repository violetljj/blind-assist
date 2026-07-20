#!/usr/bin/env python3
"""Probe a frozen dense DINO static-obstacle teacher learned from train-only masks.

The teacher is fit only on exact synthetic counterfactual patch pairs: a
composited obstacle patch is positive and the same location in the byte-bound
clear parent frame is negative. During each real leave-one-source-out fold,
synthetic records whose parent source is held out are excluded. The resulting
teacher produces deterministic lower-corridor frame statistics; a separate
real-episode ridge is fit on the remaining provisional episodes.

Synthetic masks are auxiliary train-only supervision, never event truth,
calibration data, blind truth, or production-promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import sanpo_depth_anything_linear_probe as depth_probe


SCHEMA = "blindassist_public_silver_synthetic_mask_teacher_probe_v1"


def reject_independent_direction(path: Path) -> None:
    mil.reject_independent_direction(path)


def mask_to_patch_grid(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values.max(axis=2)
    if values.ndim != 2 or not len(values) or height <= 0 or width <= 0:
        raise ValueError("mask and patch-grid dimensions must be valid")
    occupancy = cv2.resize(
        (values > 0).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    selected = occupancy >= 0.15
    if not selected.any():
        selected[np.unravel_index(int(np.argmax(occupancy)), occupancy.shape)] = True
    return selected


def corridor_masks(height: int, width: int) -> dict[str, np.ndarray]:
    if height < 2 or width < 2:
        raise ValueError("teacher corridor needs at least a 2x2 patch grid")
    yy, xx = np.mgrid[:height, :width]
    y = yy / max(height - 1, 1)
    x = xx / max(width - 1, 1)
    half_width = 0.14 + np.clip((y - 0.30) / 0.70, 0.0, 1.0) * 0.34
    corridor = (y >= 0.30) & (np.abs(x - 0.5) <= half_width)
    return {
        "corridor": corridor,
        "lower": corridor & (y >= 0.55),
        "core": (y >= 0.38) & (np.abs(x - 0.5) <= np.minimum(half_width, 0.24)),
        "terminal": corridor & (y >= 0.72),
    }


def _region_score_stats(score_map: np.ndarray, mask: np.ndarray) -> list[float]:
    values = np.asarray(score_map, dtype=np.float64)[mask]
    if not len(values):
        raise ValueError("teacher score region is empty")
    top_count = max(1, int(np.ceil(len(values) * 0.20)))
    top = np.partition(values, len(values) - top_count)[-top_count:]
    return [
        float(values.mean()),
        float(np.quantile(values, 0.75)),
        float(np.quantile(values, 0.90)),
        float(values.max()),
        float(top.mean()),
        float(np.mean(values > 0.0)),
    ]


def frame_teacher_vector(score_map: np.ndarray) -> np.ndarray:
    scores = np.asarray(score_map, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("teacher score map must be finite and two-dimensional")
    masks = corridor_masks(*scores.shape)
    values: list[float] = []
    for name in ("corridor", "lower", "core", "terminal"):
        values.extend(_region_score_stats(scores, masks[name]))
    values.extend([
        float(scores[masks["lower"]].mean() - scores[~masks["corridor"]].mean()),
        float(scores[masks["core"]].max() - scores[masks["corridor"]].mean()),
    ])
    return np.asarray(values, dtype=np.float64)


def episode_teacher_vector(frame_vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("teacher episode needs aligned finite multi-frame vectors")
    time = np.linspace(-1.0, 1.0, len(values))
    slope = (time[:, None] * values).sum(axis=0) / float(np.sum(time * time))
    return np.concatenate([
        values.mean(axis=0),
        values.max(axis=0),
        values[-1],
        values[-1] - values[0],
        slope,
    ])


def load_synthetic_patch_records(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    reject_independent_direction(root)
    records: list[dict[str, Any]] = []
    for source_path in sorted(root.glob("*/source_manifest_v2.json")):
        source = common.load_json(source_path)
        contract = source.get("synthetic_counterfactual")
        if not isinstance(contract, dict) or contract.get("train_only") is not True:
            raise ValueError(f"synthetic teacher package is not train-only: {source_path}")
        parent_source_id = contract.get("parent_source_id")
        if not isinstance(parent_source_id, str) or not parent_source_id:
            raise ValueError(f"synthetic teacher package lacks parent source: {source_path}")
        image_root = Path(source["promotion"]["image_root"]).resolve()
        reject_independent_direction(image_root)
        dataset_root = image_root.parent.parent
        frames = source.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"synthetic source has no frame list: {source_path}")
        clear_by_parent = {
            frame["parent_frame_sha256"]: frame
            for frame in frames
            if frame.get("synthetic_variant") == "clear_exact_copy"
        }
        composites = [
            frame for frame in frames
            if frame.get("synthetic_variant") == "static_obstacle_composite"
        ]
        if len(clear_by_parent) != len(composites) or not composites:
            raise ValueError(f"synthetic clear/composite alignment is incomplete: {source_path}")
        for composite in composites:
            parent_sha = composite.get("parent_frame_sha256")
            clear = clear_by_parent.get(parent_sha)
            if clear is None:
                raise ValueError(f"synthetic composite has no exact clear parent: {source_path}")
            positive_path = image_root / composite["file_name"]
            negative_path = image_root / clear["file_name"]
            mask_path = dataset_root / composite["mask_path"]
            for path in (positive_path, negative_path, mask_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
                reject_independent_direction(path)
            if common.sha256_file(positive_path) != composite["sha256"]:
                raise ValueError(f"synthetic positive hash mismatch: {positive_path}")
            if common.sha256_file(negative_path) != clear["sha256"] or clear["sha256"] != parent_sha:
                raise ValueError(f"synthetic exact-clear binding mismatch: {negative_path}")
            if common.sha256_file(mask_path) != composite["mask_sha256"]:
                raise ValueError(f"synthetic mask hash mismatch: {mask_path}")
            records.append({
                "source_id": source["source_id"],
                "parent_source_id": parent_source_id,
                "asset_name": contract.get("asset_name"),
                "positive_path": str(positive_path),
                "negative_path": str(negative_path),
                "mask_path": str(mask_path),
            })
    if len(records) < 2 or len({record["parent_source_id"] for record in records}) < 2:
        raise ValueError("synthetic teacher needs at least two parent sources")
    return records


def extract_dino_map(model: Any, path: str, *, input_size: int, layer_index: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    tensor, _ = model.image2tensor(image, input_size=input_size)
    patch_height, patch_width = tensor.shape[-2] // 14, tensor.shape[-1] // 14
    outputs = model.pretrained.get_intermediate_layers(
        tensor,
        [layer_index],
        return_class_token=True,
    )
    return depth_probe.tokens_to_feature_map(
        outputs[0][0],
        patch_height=patch_height,
        patch_width=patch_width,
    ).astype(np.float64)


def fit_patch_teacher(
    records: Sequence[dict[str, Any]],
    feature_maps: dict[str, np.ndarray],
    *,
    ridge: float,
) -> dict[str, Any]:
    samples: list[np.ndarray] = []
    labels: list[int] = []
    sample_counts: list[dict[str, Any]] = []
    for record in records:
        positive = feature_maps[record["positive_path"]]
        negative = feature_maps[record["negative_path"]]
        if positive.shape != negative.shape:
            raise ValueError("synthetic positive/negative DINO maps are misaligned")
        mask = cv2.imread(record["mask_path"], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(record["mask_path"])
        selected = mask_to_patch_grid(mask, height=positive.shape[0], width=positive.shape[1])
        positive_samples = positive[selected]
        negative_samples = negative[selected]
        samples.extend([positive_samples, negative_samples])
        labels.extend([1] * len(positive_samples))
        labels.extend([0] * len(negative_samples))
        sample_counts.append({
            "source_id": record["source_id"],
            "parent_source_id": record["parent_source_id"],
            "selected_patch_count": int(len(positive_samples)),
        })
    x = np.concatenate(samples, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    fitted = common.fit_episode_ridge(x, y, ridge=ridge, class_balanced=True)
    return {
        **fitted,
        "sample_count": int(len(y)),
        "positive_sample_count": int(np.sum(y == 1)),
        "negative_sample_count": int(np.sum(y == 0)),
        "record_sample_counts": sample_counts,
    }


def teacher_score_map(feature_map: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float64)
    logits = values @ np.asarray(kernel, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
    if logits.shape != (*values.shape[:2], 2):
        raise ValueError("teacher classifier produced an unexpected shape")
    return logits[..., 1] - logits[..., 0]


def evaluate(
    episodes: Sequence[dict[str, Any]],
    synthetic_records: Sequence[dict[str, Any]],
    feature_maps: dict[str, np.ndarray],
    *,
    teacher_ridge: float,
    episode_ridge: float,
) -> dict[str, Any]:
    labels = np.asarray([episode["label"] for episode in episodes], dtype=np.int64)
    source_ids = [episode["source_id"] for episode in episodes]
    episode_ids = [episode["episode_id"] for episode in episodes]
    predictions = np.full(len(episodes), -1, dtype=np.int64)
    margins = np.full(len(episodes), np.nan, dtype=np.float64)
    episode_features: list[np.ndarray | None] = [None] * len(episodes)
    folds: list[dict[str, Any]] = []
    source_array = np.asarray(source_ids, dtype=object)
    for held_out_source in dict.fromkeys(source_ids):
        holdout = np.flatnonzero(source_array == held_out_source)
        train = np.flatnonzero(source_array != held_out_source)
        eligible = [
            record for record in synthetic_records
            if record["parent_source_id"] != held_out_source
        ]
        teacher = fit_patch_teacher(eligible, feature_maps, ridge=teacher_ridge)
        fold_features: list[np.ndarray] = []
        fold_frame_scores: list[list[list[float]]] = []
        for episode in episodes:
            frame_vectors: list[np.ndarray] = []
            score_summaries: list[list[float]] = []
            for frame in episode["frames"]:
                scores = teacher_score_map(
                    feature_maps[frame["path"]],
                    teacher["kernel"],
                    teacher["bias"],
                )
                vector = frame_teacher_vector(scores)
                frame_vectors.append(vector)
                score_summaries.append(vector.tolist())
            fold_features.append(episode_teacher_vector(np.stack(frame_vectors)))
            fold_frame_scores.append(score_summaries)
        x = np.stack(fold_features)
        episode_head = common.fit_episode_ridge(
            x[train],
            labels[train],
            ridge=episode_ridge,
            class_balanced=True,
        )
        holdout_logits = x[holdout] @ episode_head["kernel"] + episode_head["bias"]
        fold_predictions = np.argmax(holdout_logits, axis=1).astype(np.int64)
        predictions[holdout] = fold_predictions
        margins[holdout] = holdout_logits[:, 1] - holdout_logits[:, 0]
        for index in holdout:
            episode_features[index] = x[index]
        folds.append({
            "held_out_source_id": held_out_source,
            "held_out_episode_ids": [episode_ids[index] for index in holdout],
            "expected": labels[holdout].tolist(),
            "predicted": fold_predictions.tolist(),
            "decision_margins": margins[holdout].tolist(),
            "eligible_synthetic_record_count": len(eligible),
            "excluded_parent_matched_record_count": len(synthetic_records) - len(eligible),
            "eligible_parent_source_ids": sorted({
                record["parent_source_id"] for record in eligible
            }),
            "teacher_patch_sample_count": teacher["sample_count"],
            "teacher_coefficient_sha256": teacher["coefficient_sha256"],
            "episode_coefficient_sha256": episode_head["coefficient_sha256"],
            "held_out_frame_teacher_vectors": [
                {
                    "episode_id": episode_ids[index],
                    "frame_vectors": fold_frame_scores[index],
                }
                for index in holdout
            ],
        })
    if np.any(predictions < 0) or np.any(~np.isfinite(margins)):
        raise RuntimeError("synthetic-mask teacher left real episodes unscored")
    pairs: list[dict[str, Any]] = []
    pair_ids = [episode.get("counterfactual_pair_id") for episode in episodes]
    for pair_id in sorted({value for value in pair_ids if value}):
        indices = [index for index, value in enumerate(pair_ids) if value == pair_id]
        if len(indices) == 2 and set(labels[indices].tolist()) == {0, 1}:
            negative = next(index for index in indices if labels[index] == 0)
            positive = next(index for index in indices if labels[index] == 1)
            pairs.append({
                "counterfactual_pair_id": pair_id,
                "no_alert_episode_id": episode_ids[negative],
                "alert_episode_id": episode_ids[positive],
                "no_alert_margin": float(margins[negative]),
                "alert_margin": float(margins[positive]),
                "correct_margin_order": bool(margins[positive] > margins[negative]),
            })
    feature_digest = hashlib.sha256()
    for values in episode_features:
        feature_digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    return {
        "predictions": predictions.tolist(),
        "decision_margins": margins.tolist(),
        "metrics": common.binary_metrics(labels, predictions),
        "counterfactual_pairs": pairs,
        "counterfactual_pair_order_rate": float(np.mean([
            pair["correct_margin_order"] for pair in pairs
        ])),
        "folds": folds,
        "held_out_episode_feature_sha256": feature_digest.hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.package_root,
        args.synthetic_package_root,
        args.src_root,
        args.checkpoint,
        args.output,
    ):
        reject_independent_direction(path)
    if (
        not args.package_root.is_dir()
        or not args.synthetic_package_root.is_dir()
        or not args.src_root.is_dir()
        or not args.checkpoint.is_file()
    ):
        raise FileNotFoundError("real packages, synthetic packages, DINO source, or checkpoint are missing")
    episodes, excluded = common.load_episode_specs(args.package_root)
    synthetic_records = load_synthetic_patch_records(args.synthetic_package_root)
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder)
    model.eval()
    paths = sorted({
        frame["path"]
        for episode in episodes
        for frame in episode["frames"]
    } | {
        record[key]
        for record in synthetic_records
        for key in ("positive_path", "negative_path")
    })
    feature_maps: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for path in paths:
            feature_maps[path] = extract_dino_map(
                model,
                path,
                input_size=args.input_size,
                layer_index=args.layer_index,
            )
    first = evaluate(
        episodes,
        synthetic_records,
        feature_maps,
        teacher_ridge=args.teacher_ridge,
        episode_ridge=args.episode_ridge,
    )
    second = evaluate(
        episodes,
        synthetic_records,
        feature_maps,
        teacher_ridge=args.teacher_ridge,
        episode_ridge=args.episode_ridge,
    )
    repeat_exact = first == second
    metrics = first["metrics"]
    gate = bool(
        repeat_exact
        and metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(args.package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "synthetic_teacher": {
            "package_root": str(args.synthetic_package_root),
            "record_count": len(synthetic_records),
            "parent_source_ids": sorted({
                record["parent_source_id"] for record in synthetic_records
            }),
            "asset_names": sorted({
                record["asset_name"] for record in synthetic_records
                if record["asset_name"]
            }),
            "supervision": "alpha-derived positive patch vs the same location in the exact clear parent frame",
            "train_only": True,
            "parent_matched_holdout_exclusion": True,
            "counted_in_real_metrics": False,
            "human_event_truth_present": False,
        },
        "frozen_input": {
            "model": "Depth Anything V2 frozen DINO encoder dense patch tokens",
            "encoder": args.encoder,
            "layer_index": args.layer_index,
            "input_size": args.input_size,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": common.sha256_file(args.checkpoint),
            "trainable_backbone_parameters": 0,
        },
        "evaluation": {
            "split": "leave_one_real_source_group_out",
            "teacher_ridge": args.teacher_ridge,
            "episode_ridge": args.episode_ridge,
            **first,
            "repeat_exact": repeat_exact,
        },
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {
                "balanced_accuracy_gte": args.minimum_balanced_accuracy,
                "each_class_recall_gte": args.minimum_class_recall,
            },
        },
        "evidence_limit": "Tiny provisional real episodes and synthetic train-only masks; diagnostic representation evidence only.",
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--synthetic-package-root", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits",), default="vits")
    parser.add_argument("--layer-index", type=int, choices=range(12), default=11)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--teacher-ridge", type=float, default=10.0)
    parser.add_argument("--episode-ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if (
        args.input_size <= 0
        or args.input_size % 14
        or args.teacher_ridge <= 0
        or args.episode_ridge <= 0
    ):
        parser.error("input size must be a positive multiple of 14 and ridge values must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    evaluation = report["evaluation"]
    print(json.dumps({
        "ok": True,
        "balanced_accuracy": evaluation["metrics"]["balanced_accuracy"],
        "candidate_no_alert_recall": evaluation["metrics"]["candidate_no_alert_recall"],
        "candidate_alert_recall": evaluation["metrics"]["candidate_alert_recall"],
        "pair_order_rate": evaluation["counterfactual_pair_order_rate"],
        "linear_separable": report["linear_separability_gate"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
