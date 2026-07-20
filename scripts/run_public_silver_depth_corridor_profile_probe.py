#!/usr/bin/env python3
"""Probe public-video episodes with deterministic relative-depth corridor profiles.

Unlike the frozen DINO embedding probe, this diagnostic consumes the predicted
relative depth map explicitly. It summarizes a fixed trapezoidal walking
corridor, row-relative protrusions, lateral imbalance, and temporal change.
Depth Anything remains frozen and supplies geometry features, never event
truth. Evaluation is leave-one-source-group-out.
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
import smoke_depth_anything_v2_pytorch as depth_anything


SCHEMA = "blindassist_public_silver_depth_corridor_profile_probe_v1"


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this probe's scope: {path}")


def normalized_relative_closeness(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 8 or not np.isfinite(values).all():
        raise ValueError("depth map must be a finite 2D array of at least 8x8")
    low, high = np.quantile(values, [0.02, 0.98])
    if high - low <= 1e-9:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def corridor_masks(height: int, width: int) -> dict[str, np.ndarray]:
    if height < 8 or width < 8:
        raise ValueError("corridor masks require at least 8x8")
    yy, xx = np.mgrid[:height, :width]
    y = yy / max(height - 1, 1)
    x = xx / max(width - 1, 1)
    half_width = 0.13 + np.clip((y - 0.34) / 0.66, 0.0, 1.0) * 0.31
    corridor = (y >= 0.34) & (np.abs(x - 0.5) <= half_width)
    lower = corridor & (y >= 0.58)
    core = (y >= 0.44) & (np.abs(x - 0.5) <= np.minimum(half_width, 0.21))
    left = corridor & (x < 0.5)
    right = corridor & (x >= 0.5)
    outside = (y >= 0.34) & ~corridor
    return {
        "corridor": corridor,
        "lower": lower,
        "core": core,
        "left": left,
        "right": right,
        "outside": outside,
    }


def _region_stats(closeness: np.ndarray, protrusion: np.ndarray, mask: np.ndarray) -> list[float]:
    close = closeness[mask]
    bump = protrusion[mask]
    if not len(close):
        raise ValueError("corridor region is empty")
    return [
        float(close.mean()),
        float(np.quantile(close, 0.90)),
        float(bump.mean()),
        float(np.quantile(bump, 0.90)),
        float(np.mean(bump >= 0.12)),
    ]


def depth_frame_vector(depth: np.ndarray) -> np.ndarray:
    closeness = normalized_relative_closeness(depth)
    height, width = closeness.shape
    masks = corridor_masks(height, width)
    row_baseline = np.median(closeness, axis=1, keepdims=True)
    protrusion = np.maximum(closeness - row_baseline, 0.0)

    values: list[float] = []
    for name in ("corridor", "lower", "core", "left", "right"):
        values.extend(_region_stats(closeness, protrusion, masks[name]))

    corridor_q90 = float(np.quantile(closeness[masks["corridor"]], 0.90))
    outside_q90 = float(np.quantile(closeness[masks["outside"]], 0.90))
    left_bump = float(protrusion[masks["left"]].mean())
    right_bump = float(protrusion[masks["right"]].mean())

    row_blockage: list[float] = []
    for row in range(int(height * 0.48), height):
        row_mask = masks["corridor"][row]
        if row_mask.any():
            row_blockage.append(float(np.mean(protrusion[row, row_mask] >= 0.12)))
    if not row_blockage:
        raise ValueError("lower corridor contains no sampled rows")
    values.extend([
        corridor_q90 - outside_q90,
        abs(left_bump - right_bump),
        float(np.mean(row_blockage)),
        float(np.max(row_blockage)),
        float(np.quantile(row_blockage, 0.90)),
    ])
    return np.asarray(values, dtype=np.float64)


def episode_vector(depth_maps: Sequence[np.ndarray]) -> np.ndarray:
    if len(depth_maps) < 2:
        raise ValueError("depth episode needs at least two frames")
    frame_vectors = np.stack([depth_frame_vector(depth) for depth in depth_maps])
    time = np.linspace(-1.0, 1.0, len(frame_vectors))
    denominator = float(np.sum(time * time))
    slope = (time[:, None] * frame_vectors).sum(axis=0) / denominator
    return np.concatenate([
        frame_vectors.mean(axis=0),
        frame_vectors.max(axis=0),
        frame_vectors[-1] - frame_vectors[0],
        slope,
    ])


def extract_features(model: Any, episodes: Sequence[dict[str, Any]], *, input_size: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for episode in episodes:
        depth_maps: list[np.ndarray] = []
        for frame in episode["frames"]:
            image = cv2.imread(frame["path"], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cannot decode image: {frame['path']}")
            depth_maps.append(model.infer_image(image, input_size=input_size))
        vectors.append(episode_vector(depth_maps))
    return np.stack(vectors)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.src_root, args.checkpoint, args.output):
        reject_independent_direction(path)
    if not args.package_root.is_dir() or not args.src_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("package root, Depth Anything source, or checkpoint is missing")

    episodes, excluded = common.load_episode_specs(args.package_root)
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("depth corridor probe requires both classes")

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_anything.load_model(args.src_root, args.checkpoint, args.encoder)
    model.eval()
    with torch.no_grad():
        features = extract_features(model, episodes, input_size=args.input_size)

    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]
    first = common.leave_one_source_group_out(
        features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True,
    )
    second = common.leave_one_source_group_out(
        features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True,
    )
    deterministic = first == second
    metrics = first["metrics"]
    gate = bool(
        deterministic
        and metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
    )
    feature_digest = hashlib.sha256(np.asarray(features, dtype="<f8").tobytes(order="C")).hexdigest()
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(args.package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "class_counts": {
            "candidate_no_alert": int(np.sum(labels == 0)),
            "candidate_alert": int(np.sum(labels == 1)),
        },
        "feature_source": {
            "model": "Depth Anything V2 frozen relative-depth output",
            "encoder": args.encoder,
            "input_size": args.input_size,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": common.sha256_file(args.checkpoint),
            "feature_dimension": int(features.shape[1]),
            "feature_matrix_sha256": feature_digest,
            "frame_contract": "fixed trapezoid; relative closeness; row-relative protrusion; lateral imbalance; lower-row blockage",
            "episode_contract": "frame mean + max + last-minus-first + deterministic temporal slope",
            "trainable_parameters": 0,
            "role": "frozen geometry feature only; predicted depth is not event truth",
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
        "counterfactual_delta_alignment": common.counterfactual_delta_alignment(features, episodes),
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {
                "balanced_accuracy_gte": args.minimum_balanced_accuracy,
                "each_class_recall_gte": args.minimum_class_recall,
            },
        },
        "episodes": [
            {key: value for key, value in row.items() if key != "frames"} | {"frame_count": len(row["frames"])}
            for row in episodes
        ],
        "excluded_abstentions": excluded,
        "isolation_contract": {
            "public_video_mainline_only": True,
            "independent_model_direction_data_used": False,
            "independent_model_direction_metrics_used_as_gate": False,
        },
        "evidence_limit": "Tiny GPT/VLM provisional labels and predicted relative depth; diagnostic only, not human truth, calibration, blind evaluation, or production promotion.",
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
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits",), default="vits")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    for field in ("package_root", "src_root", "checkpoint", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if args.input_size <= 0 or args.input_size % 14 != 0 or args.ridge <= 0:
        parser.error("input size must be a positive multiple of 14 and ridge must be positive")
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
        "prototype_direction_aligned": report["counterfactual_delta_alignment"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
