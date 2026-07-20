#!/usr/bin/env python3
"""Retrospectively probe static clearance with a frozen ADE20K SegFormer.

This does not rescue the frozen r7.17 result.  It measures soft walkable-class
support in fixed near-field regions and fits neither a head nor a threshold.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

import run_public_silver_background_normalized_static_probe as background
import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_free_space_topology_probe as topology


SCHEMA = "blindassist_public_silver_segformer_free_space_probe_v1"
WALKABLE_LABELS = ("floor", "road", "sidewalk", "path")
FEATURE_KEYS = (
    "lower_nonwalkable_mean",
    "core_nonwalkable_mean",
    "lower_nonwalkable_q75",
    "core_nonwalkable_q75",
    "lower_center_excess",
    "core_center_excess",
    "path_nonwalkable_mean",
    "path_nonwalkable_q90",
    "path_nonwalkable_maximum",
    "path_lower_nonwalkable_mean",
    "path_offset_mean",
    "path_offset_maximum",
)
AGGREGATIONS = ("median", "q75")


def clearance_masks(height: int, width: int) -> dict[str, np.ndarray]:
    if min(height, width) < 8:
        raise ValueError("clearance masks require at least 8x8")
    yy, xx = np.mgrid[:height, :width]
    y = yy / max(height - 1, 1)
    x = xx / max(width - 1, 1)
    depth = np.clip((y - 0.48) / 0.52, 0.0, 1.0)
    half_width = 0.10 + 0.25 * depth
    corridor = (y >= 0.48) & (np.abs(x - 0.5) <= half_width)
    lower = corridor & (y >= 0.62)
    core = corridor & (y >= 0.62) & (np.abs(x - 0.5) <= 0.18)
    peripheral = (y >= 0.62) & (y <= 0.94) & (np.abs(x - 0.5) >= 0.32)
    if any(not value.any() for value in (lower, core, peripheral)):
        raise ValueError("clearance mask is empty")
    return {"lower": lower, "core": core, "peripheral": peripheral}


def frame_descriptor(probabilities: np.ndarray, walkable_ids: Sequence[int]) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("SegFormer probabilities must be finite CxHxW")
    walkable = values[np.asarray(walkable_ids, dtype=np.int64)].sum(axis=0)
    nonwalkable = np.clip(1.0 - walkable, 0.0, 1.0)
    masks = clearance_masks(*walkable.shape)
    lower = nonwalkable[masks["lower"]]
    core = nonwalkable[masks["core"]]
    peripheral = nonwalkable[masks["peripheral"]]
    peripheral_mean = float(peripheral.mean())
    centers, path_walkable = topology.trace_adaptive_path(walkable)
    path_nonwalkable = 1.0 - path_walkable
    lower_start = int(len(path_nonwalkable) * 0.55)
    offsets = np.abs(centers / max(walkable.shape[1] - 1, 1) - 0.5)
    return {
        "lower_nonwalkable_mean": float(lower.mean()),
        "core_nonwalkable_mean": float(core.mean()),
        "lower_nonwalkable_q75": float(np.quantile(lower, 0.75)),
        "core_nonwalkable_q75": float(np.quantile(core, 0.75)),
        "lower_center_excess": max(0.0, float(lower.mean()) - peripheral_mean),
        "core_center_excess": max(0.0, float(core.mean()) - peripheral_mean),
        "path_nonwalkable_mean": float(path_nonwalkable.mean()),
        "path_nonwalkable_q90": float(np.quantile(path_nonwalkable, 0.90)),
        "path_nonwalkable_maximum": float(path_nonwalkable.max()),
        "path_lower_nonwalkable_mean": float(path_nonwalkable[lower_start:].mean()),
        "path_offset_mean": float(offsets.mean()),
        "path_offset_maximum": float(offsets.max()),
    }


def score(rows: Sequence[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("free-space window is empty")
    result: dict[str, float | int] = {"frame_count": len(rows)}
    for key in FEATURE_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"median_{key}"] = float(np.median(values))
        result[f"q75_{key}"] = float(np.quantile(values, 0.75))
    return result


class FrozenTeacher:
    def __init__(self, model_dir: Path) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            model_dir, local_files_only=True, use_fast=False
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir, local_files_only=True
        ).eval()
        labels = {int(key): value for key, value in self.model.config.id2label.items()}
        by_name = {value.lower(): key for key, value in labels.items()}
        missing = [name for name in WALKABLE_LABELS if name not in by_name]
        if missing:
            raise ValueError(f"SegFormer is missing walkable labels: {missing}")
        self.walkable_ids = [by_name[name] for name in WALKABLE_LABELS]

    def describe(self, images: Sequence[np.ndarray], *, batch_size: int) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            rgb = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in batch]
            inputs = self.processor(images=rgb, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            rows.extend(frame_descriptor(item, self.walkable_ids) for item in probabilities)
        return rows


def comparisons(
    static_rows: Sequence[dict[str, Any]], rice: dict[str, dict[str, float | int]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for aggregation in AGGREGATIONS:
        for feature in FEATURE_KEYS:
            key = f"{aggregation}_{feature}"
            pair_checks = [row["alert_scores"][key] > row["no_alert_scores"][key] for row in static_rows]
            rice_open = rice["risk"][key] > rice["pre_clear"][key]
            rice_close = rice["risk"][key] > rice["post_clear"][key]
            result[key] = {
                "legacy_static_pair_ordering": pair_checks,
                "legacy_static_pair_ordering_passed": all(pair_checks),
                "rice_open_ordering_passed": rice_open,
                "rice_close_ordering_passed": rice_close,
                "all_required_orderings_passed": all(pair_checks) and rice_open and rice_close,
            }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.mechanism_report, args.rice_video, args.rice_review, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    teacher = FrozenTeacher(args.model_dir.resolve())
    episodes, _ = common.load_episode_specs(args.package_root.resolve())
    qualified = temporal.load_qualified_pair_contract(args.mechanism_report.resolve())
    static_ids = set(qualified[temporal.STATIC])
    pairs: dict[str, list[dict[str, Any]]] = {pair_id: [] for pair_id in static_ids}
    for episode in episodes:
        if episode.get("counterfactual_pair_id") in pairs:
            pairs[episode["counterfactual_pair_id"]].append(episode)
    static_rows: list[dict[str, Any]] = []
    for pair_id in sorted(static_ids):
        no_alert = next(row for row in pairs[pair_id] if int(row["label"]) == 0)
        alert = next(row for row in pairs[pair_id] if int(row["label"]) == 1)
        no_images = background._episode_images(no_alert)
        alert_images = background._episode_images(alert)
        static_rows.append({
            "counterfactual_pair_id": pair_id,
            "no_alert_episode_id": no_alert["episode_id"],
            "alert_episode_id": alert["episode_id"],
            "no_alert_scores": score(teacher.describe(no_images, batch_size=args.batch_size)),
            "alert_scores": score(teacher.describe(alert_images, batch_size=args.batch_size)),
        })
    review = common.load_json(args.rice_review).get("review") or {}
    window_fields = {
        "pre_clear": "pre_risk_clear_window_ms",
        "risk": "risk_present_window_ms",
        "post_clear": "stable_post_clear_window_ms",
    }
    rice: dict[str, dict[str, float | int]] = {}
    for name, field in window_fields.items():
        window = review.get(field)
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"Rice review window is invalid: {name}")
        images = background.decode_video_window(args.rice_video.resolve(), int(window[0]), int(window[1]), interval_ms=1000)
        rice[name] = score(teacher.describe(images, batch_size=args.batch_size))
    candidates = comparisons(static_rows, rice)
    passing = sorted(key for key, value in candidates.items() if value["all_required_orderings_passed"])
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_frozen_teacher_diagnosis_after_r717_failure",
        "inputs": {
            "package_root": str(args.package_root.resolve()),
            "mechanism_report_sha256": common.sha256_file(args.mechanism_report),
            "rice_video_sha256": common.sha256_file(args.rice_video),
            "rice_review_sha256": common.sha256_file(args.rice_review),
            "model_dir": str(args.model_dir.resolve()),
            "model_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
        },
        "feature_contract": {
            "model": "nvidia/segformer-b2-finetuned-ade-512-512",
            "trainable_parameters": 0,
            "walkable_labels": list(WALKABLE_LABELS),
            "walkable_label_ids": teacher.walkable_ids,
            "fixed_geometry": True,
            "adaptive_path": "fixed greedy continuity path over soft walkable support; no threshold",
            "threshold_fitted": False,
            "feature_keys": list(FEATURE_KEYS),
            "window_aggregations": list(AGGREGATIONS),
        },
        "legacy_static_pairs": static_rows,
        "rice_street_windows": rice,
        "candidate_comparisons": candidates,
        "passing_candidate_keys": passing,
        "diagnostic_gate": {"minimum_required_orderings": 5, "passed": bool(passing)},
        "evidence_limit": "Retrospective frozen-teacher diagnosis after r7.17 failure; cannot rescue r7.17 or authorize training, calibration, blind evaluation, Android changes, or production.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--rice-video", type=Path, required=True)
    parser.add_argument("--rice-review", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "passing_candidate_keys": report["passing_candidate_keys"], "diagnostic_gate_passed": report["diagnostic_gate"]["passed"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
