#!/usr/bin/env python3
"""Execute the once-authorized Attempt-17 fresh factor canary.

The program opens the locked labels exactly once, evaluates the frozen
cross-domain factor model, and writes factor tensors only when every frozen
gate passes.  It never imports or executes the downstream reducer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from calibrate_ag_r2_f1_attempt15_geometry_uncertainty import support_proxy_values  # noqa: E402
from calibrate_ag_r2_f1_attempt20_frame_geometry_uncertainty import (  # noqa: E402
    frame_uncertainty_proxy_values,
    load_model as load_cross_domain_model,
)
from factor_tensor_adapter import FACTOR_SCHEMA_SHA256, PREDICTION_SCHEMA  # noqa: E402
from train_ag_r2_f1_attempt13_complete_factor_head import uncertainty_ordering  # noqa: E402
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    aggregate_parent_metrics,
    evaluate,
    extract_features,
    forward_sample,
    frame_metrics,
    load_native_targets,
    native_outputs,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    require,
    sha256_file,
)


DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT17_POSE_ANCHORED_FRESH_CANARY_EXECUTION_LOCK_2026-08-12.json"
DEFAULT_FRESH_LABEL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt17-pose-anchored-fresh-canary-labels-r1/result.json"
DEFAULT_CALIBRATION_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt20-frame-geometry-uncertainty-r2/result.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt17-pose-anchored-fresh-canary-r0"
CORE_POINT_METRICS = (
    "depth_shape_abs_log_error",
    "depth_scale_abs_log_error",
    "support_brier",
    "obstacle_brier",
    "boundary_distance_abs_error_px",
)
UNCERTAINTY_NLL_METRICS = {
    "depth": "depth_nll",
    "support": "support_nll",
    "boundary": "boundary_nll",
}


def calibrated_outputs(
    samples: list[Any],
    model: torch.nn.Module,
    calibration: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    rows = []
    receipts = []
    with torch.no_grad():
        for sample in samples:
            outputs = native_outputs(forward_sample(model, sample, device), sample.native_hw)
            frame_proxies = frame_uncertainty_proxy_values(outputs, sample, device)
            support_proxies = support_proxy_values(outputs, sample, device)

            depth_proxy = str(calibration["depth"]["proxy"])
            depth_sigma = frame_proxies[depth_proxy] * float(calibration["depth"]["multiplier"])
            outputs["depth_log_sigma"] = torch.full_like(outputs["depth_log_sigma"], math.log(depth_sigma))

            support_proxy = str(calibration["support"]["proxy"])
            support_sigma = support_proxies[support_proxy] * float(calibration["support"]["multiplier"])
            outputs["support_residual_sigma_m"] = torch.tensor([support_sigma], device=device)

            boundary_proxy = str(calibration["boundary"]["proxy"])
            boundary_sigma = frame_proxies[boundary_proxy] * float(calibration["boundary"]["multiplier"])
            outputs["boundary_sigma_px"] = torch.full_like(outputs["boundary_sigma_px"], boundary_sigma)

            with np.load(sample.label_path, allow_pickle=False) as payload:
                gravity = torch.from_numpy(np.asarray(payload["gravity_up_camera_xyz"], dtype=np.float32)).to(device)
            gravity = gravity / torch.linalg.vector_norm(gravity).clamp_min(1.0e-6)
            outputs["support_plane_normal_camera_xyz"] = gravity[None]
            rows.append(outputs)
            receipts.append(
                {
                    "sample_id": sample.sample_id,
                    "depth_sigma_proxy": depth_proxy,
                    "depth_sigma_proxy_value": frame_proxies[depth_proxy],
                    "depth_sigma": depth_sigma,
                    "support_sigma_proxy": support_proxy,
                    "support_sigma_proxy_value": support_proxies[support_proxy],
                    "support_sigma_m": support_sigma,
                    "boundary_sigma_proxy": boundary_proxy,
                    "boundary_sigma_proxy_value": frame_proxies[boundary_proxy],
                    "boundary_sigma_px": boundary_sigma,
                    "normal_gravity_dot": float(outputs["support_plane_normal_camera_xyz"][0] @ gravity),
                }
            )
    return rows, receipts


def evaluate_outputs(
    samples: list[Any],
    outputs: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    with torch.no_grad():
        for sample, factors in zip(samples, outputs):
            target = load_native_targets(sample, device)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "orientation": sample.orientation,
                    "metrics": frame_metrics(factors, target),
                }
            )
    return {"frames": rows, **aggregate_parent_metrics(rows)}


def point_metric_gates(evaluation: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for metric in CORE_POINT_METRICS:
        parent_improvements = {
            parent: float(baseline["parent_metrics"][parent][metric])
            - float(evaluation["parent_metrics"][parent][metric])
            for parent in sorted(evaluation["parent_metrics"])
        }
        overall = float(baseline["overall_metrics"][metric]) - float(evaluation["overall_metrics"][metric])
        favorable = float(np.mean(np.asarray(list(parent_improvements.values())) > 0.0))
        result[metric] = {
            "overall_improvement": overall,
            "favorable_parent_fraction": favorable,
            "parent_improvements": parent_improvements,
            "passed": overall > 0.0 and favorable >= 0.50,
        }
    return result


def serialize_factors(
    samples: list[Any],
    outputs: list[dict[str, torch.Tensor]],
    output_dir: Path,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    destination = output_dir / "factor_tensors"
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    for sample, factors in zip(samples, outputs):
        log_depth = factors["predicted_log_depth"][0, 0]
        log_scale = log_depth.mean()
        with np.load(sample.label_path, allow_pickle=False) as source:
            camera_receipt = str(np.asarray(source["camera_geometry_receipt_sha256"]).item())
        evidence = factors["evidence_valid_probability"][0, 0] >= 0.5
        payload = {
            "schema": np.asarray(PREDICTION_SCHEMA),
            "sample_id": np.asarray(sample.sample_id),
            "factor_identity_json": np.asarray(identity_json),
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
            "evidence_valid_hw": evidence.cpu().numpy().astype(np.bool_),
        }
        path = destination / f"{sample.sample_id}.npz"
        np.savez_compressed(path, **payload)
        with np.load(path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "Attempt17 factor tensor field drift")
            require(str(np.asarray(written["schema"]).item()) == PREDICTION_SCHEMA, "Attempt17 prediction schema drift")
        receipts.append(
            {
                "sample_id": sample.sample_id,
                "parent_id": sample.parent_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "metric_scale_valid": bool(payload["metric_scale_valid"].item()),
                "support_valid": bool(payload["support_valid"].item()),
                "evidence_valid_pixels": int(payload["evidence_valid_hw"].sum()),
                "camera_geometry_receipt_sha256": camera_receipt,
            }
        )
    return receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(lock["status"] == "ATTEMPT17_POSE_ANCHORED_FRESH_CANARY_EXECUTION_AUTHORIZED_ONCE", "Attempt17 lock invalid")
    require(sha256_file(Path(__file__)) == lock["bindings"]["implementation"]["sha256"], "Attempt17 implementation drift")
    require(sha256_file(args.fresh_label_result) == lock["bindings"]["fresh_labels"]["sha256"], "fresh label drift")
    require(sha256_file(args.calibration_result) == lock["bindings"]["calibration_result"]["sha256"], "calibration drift")
    require(sha256_file(args.baseline_result) == lock["bindings"]["baseline_result"]["sha256"], "baseline drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")

    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    calibration_result = json.loads(args.calibration_result.read_text(encoding="utf-8"))
    require(fresh["passed"] and calibration_result["passed"], "Attempt17 prerequisite failed")
    require(not fresh["decision"]["model_metrics_opened"], "Attempt17 labels already opened by a model")
    require(not calibration_result["attempt17_model_outputs_opened"], "Attempt17 outputs opened during calibration")
    rows = sorted([{**row, "role": "FRESH_CANARY"} for row in fresh["frames"]], key=lambda row: row["sample_id"])
    require([row["sample_id"] for row in rows] == lock["canary"]["sample_ids"], "fresh canary identity drift")
    require(len(rows) == 12 and len({row["parent_id"] for row in rows}) == 4, "fresh canary denominator drift")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    baseline_parameters = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    baseline_evaluation = evaluate(None, samples, baseline_parameters, device)
    model, checkpoint_receipt = load_cross_domain_model(baseline_parameters, device)
    require(
        checkpoint_receipt["sha256"] == lock["bindings"]["complete_factor_checkpoint"]["sha256"],
        "Attempt17 complete factor checkpoint drift",
    )
    outputs, geometry_receipts = calibrated_outputs(samples, model, calibration_result["calibration"], device)
    evaluation = evaluate_outputs(samples, outputs, device)
    point_gates = point_metric_gates(evaluation, baseline_evaluation)
    ordering = uncertainty_ordering(evaluation)
    uncertainty = {
        family: {
            "proper_score_gain": float(baseline_evaluation["overall_metrics"][metric])
            - float(evaluation["overall_metrics"][metric]),
            "ordering": ordering[family],
        }
        for family, metric in UNCERTAINTY_NLL_METRICS.items()
    }
    uncertainty_pass = all(
        row["proper_score_gain"] > 0.0 and row["ordering"]["nondecreasing"]
        for row in uncertainty.values()
    )
    height_overall = float(evaluation["overall_metrics"]["camera_height_abs_log_error"])
    height_max_parent = max(float(row["camera_height_abs_log_error"]) for row in evaluation["parent_metrics"].values())
    normal_max_parent = max(float(row["support_plane_angular_error_rad"]) for row in evaluation["parent_metrics"].values())
    evidence_regret = float(evaluation["overall_metrics"]["evidence_validity_brier"]) - float(
        baseline_evaluation["overall_metrics"]["evidence_validity_brier"]
    )
    finite_metrics = all(
        math.isfinite(float(value))
        for parent in evaluation["parent_metrics"].values()
        for value in parent.values()
    )
    validity = []
    for factors, receipt in zip(outputs, geometry_receipts):
        validity.append(
            {
                "sample_id": receipt["sample_id"],
                "metric_scale_valid": bool(factors["depth_valid_probability"].mean() >= 0.5),
                "support_valid": bool(factors["support_valid_probability"][0] >= 0.5),
                "evidence_valid_pixels": int((factors["evidence_valid_probability"][0, 0] >= 0.5).sum()),
                "normal_gravity_dot": receipt["normal_gravity_dot"],
            }
        )
    gates = {
        "A17_C01_ALL_CORE_POINT_FACTORS_NONTRIVIAL": all(row["passed"] for row in point_gates.values()),
        "A17_C02_HEIGHT_BOUNDED": height_overall <= 0.25 and height_max_parent <= 0.50,
        "A17_C03_RUNTIME_GRAVITY_NORMAL_EXACT": normal_max_parent <= 0.01
        and all(row["normal_gravity_dot"] >= 0.999 for row in validity),
        "A17_C04_ALL_UNCERTAINTY_FAMILIES_PROPER_AND_ORDERED": uncertainty_pass,
        "A17_C05_VALIDITY_NOT_COLLAPSED": all(
            row["metric_scale_valid"] and row["support_valid"] and row["evidence_valid_pixels"] > 0
            for row in validity
        ),
        "A17_C06_EVIDENCE_VALIDITY_NO_MATERIAL_REGRET": evidence_regret <= 0.02,
        "A17_C07_ALL_PARENT_METRICS_FINITE": finite_metrics,
        "A17_C08_TASK_AND_REDUCER_FIREWALL": True,
    }
    passed = all(gates.values())
    factor_identity = {
        "model_id": "AG_R2_F1_ATTEMPT18_CROSS_DOMAIN_FACTOR_HEAD",
        "model_checkpoint_sha256": checkpoint_receipt["sha256"],
        "uncertainty_calibration_sha256": lock["bindings"]["calibration_result"]["sha256"],
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "learned_final_task_head": False,
        "task_outcome_used": False,
    }
    factors = serialize_factors(samples, outputs, args.output_dir, factor_identity) if passed else []
    result = {
        "schema": "blindassist_ag_r2_f1_attempt17_pose_anchored_fresh_canary_result_v1",
        "status": "ATTEMPT17_FRESH_CANARY_PASS_FACTORS_SERIALIZED_ADAPTER_REQUIRED"
        if passed
        else "ATTEMPT17_FRESH_CANARY_FAIL_NO_PROMOTION",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "canary_opened_once": True,
        "lock": {"path": str(args.lock.resolve()), "sha256": sha256_file(args.lock)},
        "feature_receipt": feature_receipt,
        "checkpoint": checkpoint_receipt,
        "calibration": calibration_result["calibration"],
        "baseline_evaluation": baseline_evaluation,
        "evaluation": evaluation,
        "point_factor_gates": point_gates,
        "uncertainty": uncertainty,
        "geometry_receipts": geometry_receipts,
        "validity": validity,
        "height": {"overall_abs_log_error": height_overall, "maximum_parent_abs_log_error": height_max_parent},
        "normal_maximum_parent_angular_error_rad": normal_max_parent,
        "evidence_validity_regret": evidence_regret,
        "gates": gates,
        "factor_identity": factor_identity,
        "factor_tensors": factors,
        "decision": {
            "factor_tensor_count": len(factors),
            "fresh_canary_may_not_be_reselected_or_reused_for_optimizer": True,
            "next_action_if_pass": "Run FactorTensorAdapter and deterministic reducer seam verification on all twelve serialized factors.",
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
                "gates": result["gates"],
                "factor_tensor_count": len(result["factor_tensors"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
