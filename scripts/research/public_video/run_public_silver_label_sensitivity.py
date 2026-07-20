#!/usr/bin/env python3
"""Measure post-hoc episode influence on the frozen trajectory probe.

This is a label-audit diagnostic, not a model-selection gate.  It extracts the
same frozen object-trajectory features once, then repeats source-isolated ridge
evaluation after quarantining each episode in turn.  A score improvement after
quarantine never authorizes deleting or relabeling the episode; it only routes
high-influence, semantically ambiguous examples to independent re-review.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_object_trajectory_probe as trajectory


SCHEMA = "blindassist_public_silver_label_sensitivity_v1"


def evaluate(features: np.ndarray, labels: np.ndarray, episodes: list[dict[str, Any]], *, ridge: float) -> dict[str, Any]:
    return common.leave_one_source_group_out(
        features,
        labels,
        [episode["episode_id"] for episode in episodes],
        [episode["source_id"] for episode in episodes],
        ridge=ridge,
        class_balanced=True,
    )


def rank_influence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["balanced_accuracy_delta"]),
            -float(row["minimum_class_recall_delta"]),
            row["episode_id"],
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    weights = args.detector_weights.resolve()
    if not package_root.is_dir() or not weights.is_file():
        raise FileNotFoundError("package root or detector weights are missing")
    episodes, excluded = common.load_episode_specs(package_root)
    labels = np.asarray([episode["label"] for episode in episodes], dtype=np.int64)
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    detector = YOLO(str(weights))
    features, summaries = trajectory.extract(detector, episodes, image_size=args.image_size, confidence=args.confidence)
    baseline = evaluate(features, labels, episodes, ridge=args.ridge)
    baseline_metrics = baseline["metrics"]
    baseline_min_recall = min(
        baseline_metrics["candidate_no_alert_recall"],
        baseline_metrics["candidate_alert_recall"],
    )
    rows: list[dict[str, Any]] = []
    for excluded_index, episode in enumerate(episodes):
        keep = np.arange(len(episodes)) != excluded_index
        kept_labels = labels[keep]
        kept_episodes = [row for index, row in enumerate(episodes) if index != excluded_index]
        if set(kept_labels.tolist()) != {0, 1}:
            continue
        result = evaluate(features[keep], kept_labels, kept_episodes, ridge=args.ridge)
        metrics = result["metrics"]
        minimum_recall = min(metrics["candidate_no_alert_recall"], metrics["candidate_alert_recall"])
        rows.append({
            "episode_id": episode["episode_id"],
            "source_id": episode["source_id"],
            "quarantined_label": int(episode["label"]),
            "quarantined_verdict": episode["verdict"],
            "quarantined_confidence": episode["confidence"],
            "remaining_episode_count": len(kept_episodes),
            "metrics": metrics,
            "balanced_accuracy_delta": float(metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]),
            "minimum_class_recall_delta": float(minimum_recall - baseline_min_recall),
            "predictions": result["predictions"],
        })
    ranked = rank_influence(rows)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "frozen_feature_source": {
            "extractor": "fixed yolo12n COCO proposals + deterministic object trajectories",
            "weights": str(weights),
            "weights_sha256": common.sha256_file(weights),
            "trainable_parameters": 0,
        },
        "baseline": baseline,
        "quarantine_sensitivity_ranked": ranked,
        "episode_detection_summaries": summaries,
        "interpretation": "Large positive deltas identify high-influence labels for independent semantic re-review; they do not justify deletion, flipping, or score promotion.",
        "post_hoc_analysis_only": True,
        "training_gate_authorized": False,
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
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-trajectory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    top = report["quarantine_sensitivity_ranked"][:3]
    print(json.dumps({
        "ok": True,
        "baseline_balanced_accuracy": report["baseline"]["metrics"]["balanced_accuracy"],
        "highest_influence": [
            {"episode_id": row["episode_id"], "balanced_accuracy_delta": row["balanced_accuracy_delta"]}
            for row in top
        ],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
