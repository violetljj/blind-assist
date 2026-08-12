#!/usr/bin/env python3
"""Calibrate frame-level depth/boundary sigma scale from deployable geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from calibrate_ag_r2_f1_attempt15_geometry_uncertainty import (  # noqa: E402
    choose_multiplier,
    support_proxy_values,
)
from train_ag_r2_f1_attempt13_complete_factor_head import uncertainty_ordering  # noqa: E402
from train_ag_r2_f1_attempt18_consumed_cross_domain_adaptation import (  # noqa: E402
    CrossDomainFactorHead,
    adapted_rows,
)
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    BOUNDARY_DISTANCE_SCALE_PX,
    aggregate_parent_metrics,
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
    quantile_residual_summary,
    require,
    sha256_file,
)


ATTEMPT18_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt18-consumed-cross-domain-adaptation-r0/result.json"
EXPECTED_ATTEMPT18_RESULT_SHA256 = "5898A02FC2B475EB6670BFBA8BF9717C1434DCE5F3756DED9698887229DB4BCA"
ATTEMPT18_CHECKPOINT_SHA256 = "22F9E14C33EAE450874E097A4C8AC3DDFAFBAA6E50FE6EBB4A8415C5AC0613DE"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt20-frame-geometry-uncertainty-r2"
ATTEMPT16_PARENTS = {
    "rgbd_dataset_freiburg3_cabinet",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_teddy",
    "rgbd_dataset_freiburg3_walking_static",
}


def load_model(baseline: dict[str, Any], device: torch.device) -> tuple[CrossDomainFactorHead, dict[str, Any]]:
    require(sha256_file(ATTEMPT18_RESULT) == EXPECTED_ATTEMPT18_RESULT_SHA256, "Attempt18 result drift")
    result = json.loads(ATTEMPT18_RESULT.read_text(encoding="utf-8"))
    checkpoint = Path(result["composite_checkpoint"]["path"])
    require(sha256_file(checkpoint) == ATTEMPT18_CHECKPOINT_SHA256, "Attempt18 checkpoint drift")
    model = CrossDomainFactorHead(baseline).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    model.eval()
    return model, {"path": str(checkpoint.resolve()), "sha256": ATTEMPT18_CHECKPOINT_SHA256}


def probability_entropy(value: torch.Tensor) -> torch.Tensor:
    bounded = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return -(bounded * bounded.log() + (1.0 - bounded) * (1.0 - bounded).log())


def frame_uncertainty_proxy_values(
    outputs: dict[str, torch.Tensor],
    sample: Any,
    device: torch.device,
) -> dict[str, float]:
    support = support_proxy_values(outputs, sample, device)
    predicted_log = outputs["predicted_log_depth"][0, 0]
    base_depth = F.interpolate(
        sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
        sample.native_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.05, 20.0)
    correction = (predicted_log - base_depth.log()).abs()
    dx = (predicted_log[:, 1:] - predicted_log[:, :-1]).abs().mean()
    dy = (predicted_log[1:, :] - predicted_log[:-1, :]).abs().mean()
    depth_validity = outputs["depth_valid_probability"][0, 0]
    evidence_validity = outputs["evidence_valid_probability"][0, 0]
    boundary = outputs["boundary_probability"][0, 0]
    depth_sigma = outputs["depth_log_sigma"][0, 0].exp()
    boundary_sigma = outputs["boundary_sigma_px"][0, 0]
    values = {
        "raw_depth_sigma_mean": depth_sigma.mean(),
        "raw_boundary_sigma_mean": boundary_sigma.mean(),
        "depth_correction_mean": correction.mean() + 0.01,
        "depth_correction_p90": torch.quantile(correction, 0.90) + 0.01,
        "depth_correction_scale": (predicted_log.mean() - base_depth.log().mean()).abs() + 0.01,
        "depth_valid_unknown_mean": (1.0 - depth_validity).mean() + 0.01,
        "depth_valid_unknown_p90": torch.quantile(1.0 - depth_validity, 0.90) + 0.01,
        "depth_log_std": predicted_log.std().clamp_min(0.01),
        "depth_gradient_mean": dx + dy + 0.01,
        "boundary_probability_mean": boundary.mean() + 0.01,
        "boundary_entropy_mean": probability_entropy(boundary).mean() + 0.01,
        "boundary_uncertain_probability": (1.0 - (2.0 * boundary - 1.0).abs()).mean() + 0.01,
        "evidence_unknown_mean": (1.0 - evidence_validity).mean() + 0.01,
        "camera_height": outputs["camera_height_m"][0],
        "support_geometry_rms": torch.tensor(support["geometry_rms"], device=device),
        "support_geometry_plus_depth": torch.tensor(support["geometry_plus_depth"], device=device),
        "support_geometry_abs_mean": torch.tensor(support["geometry_abs_mean"], device=device),
    }
    # A few low-capacity interactions capture scale-dependent residuals.
    values["correction_x_unknown"] = values["depth_correction_mean"] * values["depth_valid_unknown_mean"]
    values["gradient_x_unknown"] = values["depth_gradient_mean"] * values["evidence_unknown_mean"]
    values["boundary_entropy_x_geometry"] = values["boundary_entropy_mean"] * values["support_geometry_rms"]
    result = {}
    for key, value in values.items():
        scalar = float(value)
        result[key] = max(scalar, 0.005) if math.isfinite(scalar) else 0.005
    return result


def choose_family(
    observations: list[dict[str, Any]],
    family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for name in sorted(observations[0]["proxies"]):
        ordering = quantile_residual_summary([
            (np.asarray([row["proxies"][name]]), np.asarray([row["target_abs_residual_mean"]]))
            for row in observations
        ])
        nll_rows = []
        for row in observations:
            proxy = row["proxies"][name]
            nll_rows.append({
                "parent_id": row["parent_id"],
                "normalized_squared_residual": row["target_squared_residual_mean"] / (proxy * proxy),
                "mean_log_sigma": math.log(proxy),
            })
        scale = choose_multiplier(nll_rows)
        candidates.append({
            "name": name,
            "family": family,
            "ordering": ordering,
            "calibration": scale,
            "eligible": bool(ordering["nondecreasing"]),
        })
    eligible = [row for row in candidates if row["eligible"]]
    selected = min(
        eligible or candidates,
        key=lambda row: (row["calibration"]["selected"]["parent_macro_nll"] + (0 if row["eligible"] else 1000), row["name"]),
    )
    return selected, candidates


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    rows, data_receipt = adapted_rows()
    attempt18 = json.loads(ATTEMPT18_RESULT.read_text(encoding="utf-8"))
    calibration_parents = set(attempt18["internal_validation_parents"]) | ATTEMPT16_PARENTS
    calibration_rows = [row for row in rows if row["parent_id"] in calibration_parents]
    require(len(calibration_rows) == 36 and len(calibration_parents) == 9, "frame calibration roster drift")
    samples, feature_receipt = extract_features(
        calibration_rows, args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device
    )
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    model, checkpoint_receipt = load_model(baseline, device)
    cached_outputs = {}
    depth_observations = []
    boundary_observations = []
    support_observations = []
    with torch.no_grad():
        for sample in samples:
            outputs = native_outputs(forward_sample(model, sample, device), sample.native_hw)
            target = load_native_targets(sample, device)
            cached_outputs[sample.sample_id] = {key: value.detach() for key, value in outputs.items()}
            proxies = frame_uncertainty_proxy_values(outputs, sample, device)
            depth_residual = outputs["predicted_log_depth"] - target["depth"].clamp_min(0.01).log()
            depth_sigma = outputs["depth_log_sigma"].exp()
            valid = target["depth_valid"]
            depth_observations.append({
                "sample_id": sample.sample_id,
                "parent_id": sample.parent_id,
                "target_abs_residual_mean": float(depth_residual[valid].abs().mean()),
                "target_squared_residual_mean": float(depth_residual[valid].square().mean()),
                "raw_sigma_mean": float(depth_sigma[valid].mean()),
                "raw_normalized_squared_residual": float(((depth_residual[valid] / depth_sigma[valid]) ** 2).mean()),
                "raw_mean_log_sigma": float(depth_sigma[valid].log().mean()),
                "proxies": proxies,
            })
            boundary_distance = (-BOUNDARY_DISTANCE_SCALE_PX * outputs["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
            boundary_residual = boundary_distance - target["boundary_distance"]
            boundary_sigma = outputs["boundary_sigma_px"]
            evidence_valid = target["evidence_valid"]
            if bool(evidence_valid.any()):
                boundary_observations.append({
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "target_abs_residual_mean": float(boundary_residual[evidence_valid].abs().mean()),
                    "target_squared_residual_mean": float(boundary_residual[evidence_valid].square().mean()),
                    "raw_sigma_mean": float(boundary_sigma[evidence_valid].mean()),
                    "raw_normalized_squared_residual": float(((boundary_residual[evidence_valid] / boundary_sigma[evidence_valid]) ** 2).mean()),
                    "raw_mean_log_sigma": float(boundary_sigma[evidence_valid].log().mean()),
                    "proxies": proxies,
                })
            support_residual = target["support_residual"][target["support_valid"]].abs()
            if support_residual.numel() > 0:
                support_values = support_proxy_values(outputs, sample, device)
                support_observations.append({
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "target_abs_residual_mean": float(support_residual.mean()),
                    "target_squared_residual_mean": float(support_residual.square().mean()),
                    "proxies": support_values,
                })

    selected_depth, depth_candidates = choose_family(depth_observations, "depth")
    selected_boundary, boundary_candidates = choose_family(boundary_observations, "boundary")
    support_candidates = []
    for name in sorted(support_observations[0]["proxies"]):
        ordering = quantile_residual_summary([(np.asarray([row["proxies"][name]]), np.asarray([row["target_abs_residual_mean"]])) for row in support_observations])
        nll_rows = [{
            "parent_id": row["parent_id"],
            "normalized_squared_residual": row["target_squared_residual_mean"] / (row["proxies"][name] ** 2),
            "mean_log_sigma": math.log(row["proxies"][name]),
        } for row in support_observations]
        scale = choose_multiplier(nll_rows)
        support_candidates.append({"name": name, "ordering": ordering, "calibration": scale, "eligible": bool(ordering["nondecreasing"])})
    eligible_support = [row for row in support_candidates if row["eligible"]]
    selected_support = min(eligible_support or support_candidates, key=lambda row: (row["calibration"]["selected"]["parent_macro_nll"] + (0 if row["eligible"] else 1000), row["name"]))
    calibration = {
        "depth": {"source": "frame_proxy_constant_dense_sigma", "proxy": selected_depth["name"], **selected_depth["calibration"]["selected"]},
        "support": {"source": "deployment_geometry_proxy", "proxy": selected_support["name"], **selected_support["calibration"]["selected"]},
        "boundary": {"source": "frame_proxy_constant_dense_sigma", "proxy": selected_boundary["name"], **selected_boundary["calibration"]["selected"]},
    }

    metric_rows = []
    with torch.no_grad():
        for sample in samples:
            target = load_native_targets(sample, device)
            outputs = dict(cached_outputs[sample.sample_id])
            proxies = frame_uncertainty_proxy_values(outputs, sample, device)
            support_values = support_proxy_values(outputs, sample, device)
            depth_sigma = calibration["depth"]["multiplier"] * proxies[calibration["depth"]["proxy"]]
            outputs["depth_log_sigma"] = torch.full_like(outputs["depth_log_sigma"], math.log(depth_sigma))
            outputs["support_residual_sigma_m"] = torch.tensor([support_values[calibration["support"]["proxy"]] * calibration["support"]["multiplier"]], device=device)
            boundary_sigma = calibration["boundary"]["multiplier"] * proxies[calibration["boundary"]["proxy"]]
            outputs["boundary_sigma_px"] = torch.full_like(outputs["boundary_sigma_px"], boundary_sigma)
            metric_rows.append({"sample_id": sample.sample_id, "parent_id": sample.parent_id, "orientation": sample.orientation, "metrics": frame_metrics(outputs, target)})
    evaluation = {"frames": metric_rows, **aggregate_parent_metrics(metric_rows)}
    ordering = uncertainty_ordering(evaluation)
    passed = all(row["nondecreasing"] for row in ordering.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt20_frame_geometry_uncertainty_result_v1",
        "status": "ATTEMPT20_FRAME_GEOMETRY_UNCERTAINTY_PASS_ATTEMPT17_LOCK_REQUIRED" if passed else "ATTEMPT20_FRAME_GEOMETRY_UNCERTAINTY_FAIL_NO_NEW_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "attempt17_model_outputs_opened": False,
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "calibration_parents": sorted(calibration_parents),
        "checkpoint": checkpoint_receipt,
        "depth_proxy_candidates": depth_candidates,
        "support_proxy_candidates": support_candidates,
        "boundary_proxy_candidates": boundary_candidates,
        "calibration": calibration,
        "calibrated_evaluation": evaluation,
        "calibrated_ordering": ordering,
        "decision": {
            "direct_sigma_pseudo_truth_used": False,
            "deployment_available_inputs_only": True,
            "spatial_sigma_shape_preserved": False,
            "constant_dense_sigma_reason": "Cross-domain frame ordering is supported; spatial ordering is not yet evidenced and is not fabricated.",
            "attempt17_model_outputs_opened": False,
            "next_action_if_pass": "Materialize/freeze Attempt17 labels and authorize one execution.",
        },
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "calibration": result["calibration"], "ordering": {family: row["nondecreasing"] for family, row in result["calibrated_ordering"].items()}}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
