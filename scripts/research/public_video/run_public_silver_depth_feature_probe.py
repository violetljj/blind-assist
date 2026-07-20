#!/usr/bin/env python3
"""Repeat the public silver episode probe with frozen Depth Anything/DINO features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import sanpo_depth_anything_linear_probe as depth_probe


def extract_features(model: Any, episodes: Sequence[dict[str, Any]], *, input_size: int, layer_index: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for episode in episodes:
        frame_vectors: list[np.ndarray] = []
        for frame in episode["frames"]:
            image = cv2.imread(frame["path"], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cannot decode image: {frame['path']}")
            tensor, _original_size = model.image2tensor(image, input_size=input_size)
            patch_height, patch_width = tensor.shape[-2] // 14, tensor.shape[-1] // 14
            outputs = model.pretrained.get_intermediate_layers(tensor, [layer_index], return_class_token=True)
            feature_map = depth_probe.tokens_to_feature_map(outputs[0][0], patch_height=patch_height, patch_width=patch_width)
            frame_vectors.append(common.pool_frame_map(feature_map))
        vectors.append(common.pool_episode(np.stack(frame_vectors)))
    return np.stack(vectors)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.package_root.is_dir() or not args.src_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("package root, Depth Anything source, or checkpoint is missing")
    episodes, excluded = common.load_episode_specs(args.package_root)
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    if set(labels.tolist()) != {0, 1} or min(np.bincount(labels, minlength=2)) < 2:
        raise ValueError("probe requires at least two independent episodes per class")
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder)
    model.eval()
    with torch.no_grad():
        features = extract_features(model, episodes, input_size=args.input_size, layer_index=args.layer_index)
    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]
    first = common.leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True)
    second = common.leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True)
    deterministic = first == second
    metrics = first["metrics"]
    gate = bool(
        deterministic
        and metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
    )
    report = {
        "schema": "blindassist_public_silver_depth_feature_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(args.package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "class_counts": {"candidate_no_alert": int(np.sum(labels == 0)), "candidate_alert": int(np.sum(labels == 1))},
        "feature_source": {
            "model": "Depth Anything V2 frozen DINO encoder",
            "encoder": args.encoder,
            "layer_index": args.layer_index,
            "input_size": args.input_size,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": common.sha256_file(args.checkpoint),
            "feature_dimension": int(features.shape[1]),
            "frame_pool": "global_mean+global_max+center_mean+lower_center_mean",
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
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {"balanced_accuracy_gte": args.minimum_balanced_accuracy, "each_class_recall_gte": args.minimum_class_recall},
        },
        "episodes": [{key: value for key, value in row.items() if key != "frames"} | {"frame_count": len(row["frames"])} for row in episodes],
        "excluded_abstentions": excluded,
        "evidence_limit": "Tiny provisional GPT/VLM-labelled public set; diagnostic only, not human accuracy or promotion evidence.",
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
    parser.add_argument("--layer-index", type=int, choices=range(12), default=11)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260716)
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
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
