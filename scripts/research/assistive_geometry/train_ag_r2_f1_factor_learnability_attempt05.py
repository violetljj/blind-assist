#!/usr/bin/env python3
"""Attempt 05: unchanged factor body plus learned uncertainty calibrators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt05_uncertainty_calibrators import (  # noqa: E402
    calibration_features,
    load_calibrator_checkpoint,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_ATTEMPT02_RESULT,
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_ATTEMPT02_RESULT_SHA256,
    EXPECTED_BASELINE_RESULT_SHA256,
    EXPECTED_DEPTHART_SHA256,
    apply_geometry,
    evaluate_cached,
    extract_features,
    gate,
    parent_vio_height_context,
    prepare,
    require,
    sha256_file,
)
from train_ag_r2_f1_factor_learnability_attempt04 import (  # noqa: E402
    GEOMETRY_CONFIG,
    load_caches,
)


DEFAULT_LOCK = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT05_UNCERTAINTY_CALIBRATION_AND_FRESH_CANARY_LOCK_2026-08-11.json"
)
DEFAULT_FRESH_LABEL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-fresh-ag-held-labels-r0/result.json"
DEFAULT_CALIBRATION_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-uncertainty-calibration-r3/result.json"
DEFAULT_CALIBRATOR_CHECKPOINT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-uncertainty-calibration-r3/uncertainty_calibrators.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-factor-learnability-attempt05-r0"


def calibrated_outputs(
    caches: list[list[dict[str, torch.Tensor]]],
    prepared: list[dict[str, Any]],
    depth_calibrator: torch.nn.Module,
    boundary_calibrator: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    rows = []
    with torch.no_grad():
        for frame_index, prepared_row in enumerate(prepared):
            members = [cache[frame_index] for cache in caches]
            result = dict(members[0])
            sample = prepared_row["sample"]
            base_depth = F.interpolate(
                sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
                sample.native_hw,
                mode="bilinear",
                align_corners=False,
            ).clamp_min(0.01)
            depth_features, boundary_features = calibration_features(members, base_depth)
            result["depth_log_sigma"] = depth_calibrator(depth_features)
            result["boundary_sigma_px"] = boundary_calibrator(boundary_features).exp()
            require(result["depth_log_sigma"].shape == members[0]["depth_log_sigma"].shape, "depth sigma shape drift")
            require(result["boundary_sigma_px"].shape == members[0]["boundary_sigma_px"].shape, "boundary sigma shape drift")
            for key in members[0]:
                if key not in {"depth_log_sigma", "boundary_sigma_px"}:
                    require(torch.equal(result[key], members[0][key]), f"point factor changed by calibrator: {key}")
            rows.append(result)
    return rows


def serialize_predictions(
    prepared: list[dict[str, Any]],
    outputs: list[dict[str, torch.Tensor]],
    output_dir: Path,
    calibrator_sha256: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    destination = output_dir / "canary_predictions"
    destination.mkdir(parents=True, exist_ok=False)
    context = parent_vio_height_context(prepared, outputs, GEOMETRY_CONFIG["height"], device)
    receipts = []
    with torch.no_grad():
        for prepared_row, raw in zip(prepared, outputs):
            sample = prepared_row["sample"]
            factors, geometry_receipt = apply_geometry(raw, sample, GEOMETRY_CONFIG, {}, device, context)
            log_depth = factors["predicted_log_depth"][0, 0]
            log_scale = log_depth.mean()
            with np.load(sample.label_path, allow_pickle=False) as source:
                camera_receipt = str(np.asarray(source["camera_geometry_receipt_sha256"]).item())
            payload = {
                "schema": np.asarray("blindassist_assistive_geometry_r2_f1_prediction_v1"),
                "sample_id": np.asarray(sample.sample_id),
                "factor_identity": np.asarray("AG_R2_F1_FACTORS_ATTEMPT05_LEARNED_UNCERTAINTY_CALIBRATION"),
                "uncertainty_calibrator_sha256": np.asarray(calibrator_sha256),
                "camera_geometry_receipt_sha256": np.asarray(camera_receipt),
                "depth_shape_positive_hw": torch.exp(log_depth - log_scale).cpu().numpy().astype(np.float32),
                "log_metric_scale_m_scalar": np.asarray(float(log_scale), dtype=np.float32),
                "depth_log_sigma_hw": factors["depth_log_sigma"][0, 0].cpu().numpy().astype(np.float32),
                "depth_valid_probability_hw": factors["depth_valid_probability"][0, 0].cpu().numpy().astype(np.float32),
                "metric_scale_valid": np.asarray(bool(factors["depth_valid_probability"].mean() >= 0.5)),
                "support_probability_hw": factors["support_probability"][0, 0].cpu().numpy().astype(np.float32),
                "support_plane_normal_camera_xyz": factors["support_plane_normal_camera_xyz"][0].cpu().numpy().astype(np.float32),
                "camera_height_m": np.asarray(float(factors["camera_height_m"][0]), dtype=np.float32),
                "support_residual_sigma_m": np.asarray(float(factors["support_residual_sigma_m"][0]), dtype=np.float32),
                "support_valid": np.asarray(bool(factors["support_valid_probability"][0] >= 0.5)),
                "obstacle_evidence_probability_hw": factors["obstacle_probability"][0, 0].cpu().numpy().astype(np.float32),
                "boundary_probability_hw": factors["boundary_probability"][0, 0].cpu().numpy().astype(np.float32),
                "boundary_localization_sigma_px_hw": factors["boundary_sigma_px"][0, 0].cpu().numpy().astype(np.float32),
                "evidence_valid_hw": (factors["evidence_valid_probability"][0, 0] >= 0.5).cpu().numpy().astype(np.bool_),
            }
            path = destination / f"{sample.sample_id}.npz"
            np.savez_compressed(path, **payload)
            receipts.append(
                {
                    "sample_id": sample.sample_id,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "camera_geometry_receipt_sha256": camera_receipt,
                    "uncertainty_calibrator_sha256": calibrator_sha256,
                    "geometry": geometry_receipt,
                }
            )
    return receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(lock["status"] == "ATTEMPT05_UNCERTAINTY_CALIBRATION_AND_FRESH_CANARY_EXECUTION_AUTHORIZED", "Attempt-05 lock invalid")
    require(sha256_file(args.fresh_label_result) == lock["bindings"]["fresh_label_result"]["sha256"], "fresh labels drift")
    require(sha256_file(args.calibration_result) == lock["bindings"]["calibration_result"]["sha256"], "calibration result drift")
    require(sha256_file(args.calibrator_checkpoint) == lock["bindings"]["calibrator_checkpoint"]["sha256"], "calibrator checkpoint drift")
    require(sha256_file(args.attempt02_result) == EXPECTED_ATTEMPT02_RESULT_SHA256, "Attempt-02 result drift")
    require(sha256_file(args.baseline_result) == EXPECTED_BASELINE_RESULT_SHA256, "baseline drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_result.read_text(encoding="utf-8"))
    require(bool(fresh["passed"]) and bool(calibration["passed"]), "Attempt-05 prerequisite not passed")
    require(
        calibration["attempt05_held_label_result_excluded"]["sha256"]
        == lock["bindings"]["fresh_label_result"]["sha256"],
        "held-label exclusion receipt drift",
    )
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    rows_by_role = {
        role: [{**row, "role": role} for row in fresh["frames"] if row["role"] == role]
        for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    require(
        {key: len(value) for key, value in rows_by_role.items()}
        == {"CHECKPOINT_SELECTION": 6, "TRAIN_CANARY": 6},
        "Attempt-05 role roster drift",
    )
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    depth_calibrator, boundary_calibrator, calibrator_metadata = load_calibrator_checkpoint(
        args.calibrator_checkpoint, device
    )
    require(
        calibrator_metadata["attempt05_held_label_result_excluded"]["sha256"]
        == lock["bindings"]["fresh_label_result"]["sha256"],
        "calibrator held-label firewall drift",
    )

    selection_samples, selection_feature_receipt = extract_features(
        sorted(rows_by_role["CHECKPOINT_SELECTION"], key=lambda row: row["sample_id"]),
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    selection_prepared = prepare(selection_samples, device)
    selection_baseline = evaluate_cached(selection_prepared, None, baseline, None, device)
    selection_caches, checkpoint_receipts = load_caches(attempt02, baseline, selection_prepared, device)
    selection_outputs = calibrated_outputs(
        selection_caches, selection_prepared, depth_calibrator, boundary_calibrator, device
    )
    selection_evaluation = evaluate_cached(
        selection_prepared, selection_outputs, baseline, GEOMETRY_CONFIG, device
    )
    selection_gate = gate(selection_evaluation, selection_baseline, 83)
    selection_passed = bool(
        selection_gate["all_primary_metrics_passed"]
        and selection_gate["all_uncertainty_families_passed"]
    )

    canary_feature_receipt = None
    canary_prepared: list[dict[str, Any]] = []
    canary_baseline = None
    canary_evaluation = None
    canary_gate_result = None
    canary_outputs: list[dict[str, torch.Tensor]] = []
    if selection_passed:
        canary_samples, canary_feature_receipt = extract_features(
            sorted(rows_by_role["TRAIN_CANARY"], key=lambda row: row["sample_id"]),
            args.depthart_source,
            args.depthart_checkpoint,
            args.depthart_extension,
            device,
        )
        canary_prepared = prepare(canary_samples, device)
        canary_baseline = evaluate_cached(canary_prepared, None, baseline, None, device)
        canary_caches, _ = load_caches(attempt02, baseline, canary_prepared, device)
        canary_outputs = calibrated_outputs(
            canary_caches, canary_prepared, depth_calibrator, boundary_calibrator, device
        )
        canary_evaluation = evaluate_cached(
            canary_prepared, canary_outputs, baseline, GEOMETRY_CONFIG, device
        )
        canary_gate_result = gate(canary_evaluation, canary_baseline, 83)
    passed = bool(
        selection_passed
        and canary_gate_result
        and canary_gate_result["all_primary_metrics_passed"]
        and canary_gate_result["all_uncertainty_families_passed"]
    )
    calibrator_sha = lock["bindings"]["calibrator_checkpoint"]["sha256"]
    predictions = (
        serialize_predictions(canary_prepared, canary_outputs, args.output_dir, calibrator_sha, device)
        if passed
        else []
    )
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_factor_learnability_attempt05_result_v1",
        "status": (
            "R2_F1_FACTOR_LEARNABILITY_ATTEMPT05_PASS_FACTOR_TENSORS_SERIALIZED"
            if passed
            else "R2_F1_FACTOR_LEARNABILITY_ATTEMPT05_FAIL_STOP"
        ),
        "passed": passed,
        "execution_lock": str(args.lock.resolve()),
        "execution_lock_sha256": sha256_file(args.lock),
        "geometry_config": GEOMETRY_CONFIG,
        "calibration_result": {
            "path": str(args.calibration_result.resolve()),
            "sha256": lock["bindings"]["calibration_result"]["sha256"],
        },
        "calibrator_checkpoint": {
            "path": str(args.calibrator_checkpoint.resolve()),
            "sha256": calibrator_sha,
        },
        "checkpoint_receipts": checkpoint_receipts,
        "feature_receipt": {
            "selection": selection_feature_receipt,
            "canary_after_selection_pass": canary_feature_receipt,
        },
        "role_frame_counts": {
            "CHECKPOINT_SELECTION": len(selection_prepared),
            "TRAIN_CANARY": len(canary_prepared),
        },
        "selection_baseline": selection_baseline,
        "selection_evaluation": selection_evaluation,
        "selection_gate": selection_gate,
        "selection_passed_before_canary_open": selection_passed,
        "canary_baseline": canary_baseline,
        "canary_evaluation": canary_evaluation,
        "canary_gate": canary_gate_result,
        "prediction_receipts": predictions,
        "decision": {
            "canonical_point_model": "seed17 Attempt-02 composite, byte-for-byte unchanged by calibrators",
            "support_geometry_and_uncertainty": "Attempt-04 frozen geometry configuration, unchanged",
            "seed29_and_seed43_use": "depth epistemic disagreement feature only",
            "uncertainty_calibrator_optimizer_steps_in_this_execution": 0,
            "reducer_or_task_outcome_read": False,
            "next_action_if_pass": "Run FactorTensorAdapter on the six serialized real factor tensors, then execute the deterministic reducer seam canary.",
        },
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--fresh-label-result", type=Path, default=DEFAULT_FRESH_LABEL_RESULT)
    parser.add_argument("--calibration-result", type=Path, default=DEFAULT_CALIBRATION_RESULT)
    parser.add_argument("--calibrator-checkpoint", type=Path, default=DEFAULT_CALIBRATOR_CHECKPOINT)
    parser.add_argument("--attempt02-result", type=Path, default=DEFAULT_ATTEMPT02_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "selection_passed_before_canary_open": result["selection_passed_before_canary_open"],
                "selection_gate": result["selection_gate"],
                "canary_gate": result["canary_gate"],
                "prediction_count": len(result["prediction_receipts"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
