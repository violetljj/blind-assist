#!/usr/bin/env python3
"""Attempt 04: canonical seed17 factors plus frozen ensemble uncertainty calibration."""

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

from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_ATTEMPT02_RESULT,
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    DEFAULT_FRESH_LABEL_RESULT,
    EXPECTED_ATTEMPT02_RESULT_SHA256,
    EXPECTED_BASELINE_RESULT_SHA256,
    EXPECTED_COMPOSITES,
    EXPECTED_DEPTHART_SHA256,
    FactorSplitHead,
    apply_geometry,
    cache_model_outputs,
    evaluate_cached,
    extract_features,
    gate,
    parent_vio_height_context,
    prepare,
    require,
    sha256_file,
)

DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT04_CALIBRATED_FACTOR_EXECUTION_LOCK_2026-08-11.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-factor-learnability-attempt04-r0"

GEOMETRY_CONFIG = {
    "height": {
        "depth_source": "base_depthart",
        "estimator": "weighted_quantile",
        "metric_scale_calibration": "none",
        "mode_bin_m": 0.04,
        "mode_radius_m": 0.12,
        "quantile": 0.10,
        "scope": "parent_vio_world_plane",
        "support_power": 2.0,
        "support_threshold": 0.05,
    },
    "support_sigma": {"multiplier": 1.5, "source": "coverage_complement"},
}
DEPTH_UNCERTAINTY_CONFIG = {
    "aleatoric_weight": 1.0,
    "epistemic_weight": 1.0,
    "base_disagreement_weight": 2.0,
    "invalidity_weight": 0.0,
    "scale": 0.3375950044697517,
}


def calibrated_depth_sigma(
    seed17_log_sigma: torch.Tensor,
    seed_logs: torch.Tensor,
    seed17_log_depth: torch.Tensor,
    base_log_depth: torch.Tensor,
    depth_valid_probability: torch.Tensor,
) -> torch.Tensor:
    aleatoric = seed17_log_sigma.exp()
    epistemic = seed_logs.std(dim=0, correction=0)
    base_disagreement = (seed17_log_depth - base_log_depth).abs()
    invalidity = 1.0 - depth_valid_probability
    raw = (
        DEPTH_UNCERTAINTY_CONFIG["aleatoric_weight"] * aleatoric
        + DEPTH_UNCERTAINTY_CONFIG["epistemic_weight"] * epistemic
        + DEPTH_UNCERTAINTY_CONFIG["base_disagreement_weight"] * base_disagreement
        + DEPTH_UNCERTAINTY_CONFIG["invalidity_weight"] * invalidity
    )
    return (raw * DEPTH_UNCERTAINTY_CONFIG["scale"]).clamp_min(1.0e-3)


def calibrated_outputs(
    caches: list[list[dict[str, torch.Tensor]]],
    prepared: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    rows = []
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
        seed_logs = torch.stack([row["predicted_log_depth"] for row in members])
        sigma = calibrated_depth_sigma(
            members[0]["depth_log_sigma"],
            seed_logs,
            members[0]["predicted_log_depth"],
            base_depth.log(),
            members[0]["depth_valid_probability"],
        )
        result["depth_log_sigma"] = sigma.log()
        rows.append(result)
    return rows


def load_caches(
    attempt02: dict[str, Any],
    baseline: dict[str, Any],
    prepared: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[list[dict[str, torch.Tensor]]], list[dict[str, Any]]]:
    caches = []
    receipts = []
    for seed_row in attempt02["seed_results"]:
        seed = int(seed_row["seed"])
        checkpoint = Path(seed_row["composite_checkpoint"]["path"])
        require(sha256_file(checkpoint) == EXPECTED_COMPOSITES[seed], f"composite drift: {seed}")
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        caches.append(cache_model_outputs(model, prepared, device))
        receipts.append({"seed": seed, "path": str(checkpoint.resolve()), "sha256": EXPECTED_COMPOSITES[seed]})
        del model
        torch.cuda.empty_cache()
    require([row["seed"] for row in receipts] == [17, 29, 43], "seed order drift")
    return caches, receipts


def serialize_predictions(
    prepared: list[dict[str, Any]],
    outputs: list[dict[str, torch.Tensor]],
    output_dir: Path,
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
                "factor_identity": np.asarray("AG_R2_F1_FACTORS_RGB_K_IMU_VIO_CALIBRATED"),
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
            receipts.append({"sample_id": sample.sample_id, "path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size, "camera_geometry_receipt_sha256": camera_receipt, "geometry": geometry_receipt})
    return receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(lock["status"] == "ATTEMPT04_CALIBRATED_FACTOR_EXECUTION_AUTHORIZED", "Attempt-04 lock invalid")
    require(sha256_file(args.fresh_label_result) == lock["bindings"]["fresh_label_result_sha256"], "fresh labels drift")
    require(sha256_file(args.attempt02_result) == EXPECTED_ATTEMPT02_RESULT_SHA256, "Attempt-02 result drift")
    require(sha256_file(args.baseline_result) == EXPECTED_BASELINE_RESULT_SHA256, "baseline drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    rows_by_role = {role: [{**row, "role": role} for row in fresh["frames"] if row["role"] == role] for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY")}
    require({key: len(value) for key, value in rows_by_role.items()} == {"CHECKPOINT_SELECTION": 6, "TRAIN_CANARY": 6}, "role roster drift")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    selection_samples, selection_feature_receipt = extract_features(sorted(rows_by_role["CHECKPOINT_SELECTION"], key=lambda row: row["sample_id"]), args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device)
    selection_prepared = prepare(selection_samples, device)
    selection_baseline = evaluate_cached(selection_prepared, None, baseline, None, device)
    selection_caches, checkpoint_receipts = load_caches(attempt02, baseline, selection_prepared, device)
    selection_outputs = calibrated_outputs(selection_caches, selection_prepared, device)
    selection_evaluation = evaluate_cached(selection_prepared, selection_outputs, baseline, GEOMETRY_CONFIG, device)
    selection_gate = gate(selection_evaluation, selection_baseline, 71)
    selection_passed = selection_gate["all_primary_metrics_passed"] and selection_gate["all_uncertainty_families_passed"]

    canary_feature_receipt = None
    canary_prepared: list[dict[str, Any]] = []
    canary_baseline = None
    canary_evaluation = None
    canary_gate_result = None
    canary_outputs: list[dict[str, torch.Tensor]] = []
    if selection_passed:
        canary_samples, canary_feature_receipt = extract_features(sorted(rows_by_role["TRAIN_CANARY"], key=lambda row: row["sample_id"]), args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device)
        canary_prepared = prepare(canary_samples, device)
        canary_baseline = evaluate_cached(canary_prepared, None, baseline, None, device)
        canary_caches, _ = load_caches(attempt02, baseline, canary_prepared, device)
        canary_outputs = calibrated_outputs(canary_caches, canary_prepared, device)
        canary_evaluation = evaluate_cached(canary_prepared, canary_outputs, baseline, GEOMETRY_CONFIG, device)
        canary_gate_result = gate(canary_evaluation, canary_baseline, 71)
    passed = bool(selection_passed and canary_gate_result and canary_gate_result["all_primary_metrics_passed"] and canary_gate_result["all_uncertainty_families_passed"])
    predictions = serialize_predictions(canary_prepared, canary_outputs, args.output_dir, device) if passed else []
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_factor_learnability_attempt04_result_v1",
        "status": "R2_F1_FACTOR_LEARNABILITY_ATTEMPT04_PASS" if passed else "R2_F1_FACTOR_LEARNABILITY_ATTEMPT04_FAIL_STOP",
        "passed": passed,
        "execution_lock": str(args.lock.resolve()),
        "execution_lock_sha256": sha256_file(args.lock),
        "geometry_config": GEOMETRY_CONFIG,
        "depth_uncertainty_config": DEPTH_UNCERTAINTY_CONFIG,
        "checkpoint_receipts": checkpoint_receipts,
        "feature_receipt": {"selection": selection_feature_receipt, "canary_after_selection_pass": canary_feature_receipt},
        "role_frame_counts": {"CHECKPOINT_SELECTION": len(selection_prepared), "TRAIN_CANARY": len(canary_prepared)},
        "selection_baseline": selection_baseline,
        "selection_evaluation": selection_evaluation,
        "selection_gate": selection_gate,
        "selection_passed_before_canary_open": selection_passed,
        "canary_baseline": canary_baseline,
        "canary_evaluation": canary_evaluation,
        "canary_gate": canary_gate_result,
        "prediction_receipts": predictions,
        "decision": {
            "canonical_point_model": "seed17 Attempt-02 composite",
            "seed29_and_seed43_use": "depth epistemic disagreement only",
            "optimizer_steps": 0,
            "reducer_or_task_outcome_read": False,
            "next_action_if_pass": "Run FactorTensorAdapter on the six serialized real factor tensors, then the deterministic reducer seam canary.",
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
    parser.add_argument("--attempt02-result", type=Path, default=DEFAULT_ATTEMPT02_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("lock", "fresh_label_result", "attempt02_result", "baseline_result", "depthart_source", "depthart_checkpoint", "depthart_extension", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "selection_passed_before_canary_open": result["selection_passed_before_canary_open"], "canary_opened": result["canary_evaluation"] is not None}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
