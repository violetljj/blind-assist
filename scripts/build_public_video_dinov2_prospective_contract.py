#!/usr/bin/env python3
"""Freeze the DINOv2 regional prototype before reviewing a new source."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_background_normalized_static_probe as background
import run_public_silver_dinov2_regional_pair_probe as dino
import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_synthetic_pair_delta_probe as pair_probe


SCHEMA = "blindassist_public_video_dinov2_pair_contract_v1"


def direction_sha256(direction: np.ndarray) -> str:
    values = np.asarray(direction, dtype="<f8")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("direction must be a finite vector")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.mechanism_report, args.dataset_root, args.synthetic_audit, args.inverse_dataset_root, args.inverse_audit, args.model_dir, args.rice_video, args.rice_review, args.feature_gate_report, args.bootstrap_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    feature_gate = lifecycle.verify_json_sidecar(args.feature_gate_report.resolve())
    bootstrap = lifecycle.verify_json_sidecar(args.bootstrap_report.resolve())
    if feature_gate.get("fixed_teacher_gate", {}).get("passed") is not True:
        raise ValueError("DINOv2 feature gate did not pass")
    if bootstrap.get("bootstrap_stability_gate", {}).get("passed") is not True:
        raise ValueError("DINOv2 bootstrap stability gate did not pass")
    for audit_path in (args.synthetic_audit, args.inverse_audit):
        if lifecycle.verify_json_sidecar(audit_path.resolve()).get("parent_source_isolated_representation_short_run_authorized") is not True:
            raise ValueError("auxiliary audit did not authorize contract construction")

    teacher = dino.FrozenDinoV2(args.model_dir.resolve())
    episodes, _ = common.load_episode_specs(args.package_root.resolve())
    qualified = temporal.load_qualified_pair_contract(args.mechanism_report.resolve())
    pairs: list[dict[str, Any]] = []
    for pair_id in sorted(set(qualified[temporal.STATIC])):
        members = [row for row in episodes if row.get("counterfactual_pair_id") == pair_id]
        clear = next(row for row in members if int(row["label"]) == 0)
        risk = next(row for row in members if int(row["label"]) == 1)
        clear_vector = teacher.extract(background._episode_images(clear), batch_size=args.batch_size).mean(axis=0)
        risk_vector = teacher.extract(background._episode_images(risk), batch_size=args.batch_size).mean(axis=0)
        pairs.append({"kind": "real", "pair_id": pair_id, "parent_source_id": clear["source_id"], "delta": risk_vector - clear_vector})
    forward = dino.load_manifest_pairs(teacher, args.dataset_root.resolve(), kind="synthetic", batch_size=args.batch_size)
    inverse = dino.load_manifest_pairs(teacher, args.inverse_dataset_root.resolve(), kind="inverse", batch_size=args.batch_size)
    pairs.extend(forward)
    pairs.extend(inverse)
    direction = pair_probe.prototype_direction([row["delta"] for row in pairs])
    model_weights = args.model_dir / "model.safetensors"
    contract = {
        "schema": SCHEMA,
        "contract_id": "public-video-dinov2-regional-pair-r722",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "derivation_disclosure": "Selected after r7.17 Rice and ADE failures. Rice and all listed pairs are retrospective derivation evidence and cannot count as the next prospective source.",
        "frozen_feature_contract": {
            "model": "facebook/dinov2-small", "model_weights_sha256": common.sha256_file(model_weights),
            "model_config_sha256": common.sha256_file(args.model_dir / "config.json"),
            "preprocessor_config_sha256": common.sha256_file(args.model_dir / "preprocessor_config.json"),
            "input": "processor-default fixed 224 crop", "layer": "last_hidden_state only",
            "regions": ["cls", "global_patch_mean", "lower_half_mean", "lower_center_half_mean", "lower_center_minus_lower_peripheral"],
            "frame_vector_l2_normalized": True, "window_pool": "arithmetic mean of scheduled one-second frame vectors",
            "feature_dimension": int(len(direction)), "trainable_parameters": 0,
        },
        "frozen_prototype": {
            "construction": "unit-normalize every risk-minus-clear pair delta, average, unit-normalize",
            "direction": direction.tolist(), "direction_dtype": "little_endian_float64",
            "direction_sha256": direction_sha256(direction), "threshold_fitted": False,
            "open_rule": "dot(mean(risk)-mean(pre_clear), direction) > 0",
            "close_rule": "dot(mean(risk)-mean(post_clear), direction) > 0",
        },
        "derivation_inputs": {
            "feature_gate_report_sha256": common.sha256_file(args.feature_gate_report),
            "bootstrap_report_sha256": common.sha256_file(args.bootstrap_report),
            "mechanism_report_sha256": common.sha256_file(args.mechanism_report),
            "forward_manifest_sha256": common.sha256_file(args.dataset_root / "manifest.jsonl"),
            "inverse_manifest_sha256": common.sha256_file(args.inverse_dataset_root / "manifest.jsonl"),
            "forward_audit_sha256": common.sha256_file(args.synthetic_audit),
            "inverse_audit_sha256": common.sha256_file(args.inverse_audit),
            "rice_video_sha256": common.sha256_file(args.rice_video),
            "rice_review_sha256": common.sha256_file(args.rice_review),
            "training_pair_count": len(pairs),
            "training_parent_source_ids": sorted({row["parent_source_id"] for row in pairs}),
        },
        "prospective_source_requirements": {
            "source_id_not_in_training_parent_source_ids": True,
            "video_sha256_not_equal_rice_video_sha256": True,
            "item_level_reuse_license_required": True,
            "single_continuous_pedestrian_or_equivalent_ego_view": True,
            "full_video_features_frozen_before_original_order_review": True,
            "review_requires_pre_clear_risk_post_clear_windows": True,
            "hard_cut_in_or_between_windows_forbidden": True,
            "minimum_scheduled_samples_per_window": 3,
            "both_open_and_close_projection_strictly_positive": True
        },
        "authorizations": {
            "feature_extraction": True, "prospective_evaluation": True,
            "training": False, "calibration": False, "blind": False,
            "android_runtime_change": False, "production_model_replacement": False
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--synthetic-audit", type=Path, required=True)
    parser.add_argument("--inverse-dataset-root", type=Path, required=True)
    parser.add_argument("--inverse-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rice-video", type=Path, required=True)
    parser.add_argument("--rice-review", type=Path, required=True)
    parser.add_argument("--feature-gate-report", type=Path, required=True)
    parser.add_argument("--bootstrap-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "feature_dimension": payload["frozen_feature_contract"]["feature_dimension"], "direction_sha256": payload["frozen_prototype"]["direction_sha256"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
