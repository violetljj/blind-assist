#!/usr/bin/env python3
"""Freeze deployable uncertainty calibration before the fresh canary.

Depth and boundary keep their residual-ranked spatial sigma maps with a positive
scale calibration.  Support sigma is selected from deployment-available
predicted depth, K, gravity, support and validity geometry; no direct sigma
pseudo-label is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from train_ag_r2_f1_attempt13_complete_factor_head import (  # noqa: E402
    consumed_rows,
    parent_split,
    uncertainty_ordering,
)
from train_ag_r2_f1_attempt14_residual_rank_uncertainty import ResidualRankUncertaintyHead  # noqa: E402
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    BOUNDARY_DISTANCE_SCALE_PX,
    GAUSSIAN_CONSTANT,
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
    quantile_residual_summary,
    require,
    sha256_file,
)


ATTEMPT14_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt14-residual-rank-uncertainty-r1/result.json"
EXPECTED_ATTEMPT14_RESULT_SHA256 = "1F8F78E9EBBD3E42BD1F32032F4597DA4A1D34203E1810520C26EC979CA61D4A"
ATTEMPT14_CHECKPOINT_SHA256 = "A0F8A99E65A83E88A19AAE59A19AACCF8B01437FFD0B87BB89D0A53B4345E842"
ATTEMPT13_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt13-complete-factor-head-r1/result.json"
EXPECTED_ATTEMPT13_RESULT_SHA256 = "C1862DE977F2555FF7AE86197970E99901292095A493D361547B9F0751F001BD"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt15-geometry-uncertainty-calibration-r1"


def load_model(baseline: dict[str, Any], device: torch.device) -> tuple[ResidualRankUncertaintyHead, dict[str, Any]]:
    require(sha256_file(ATTEMPT14_RESULT) == EXPECTED_ATTEMPT14_RESULT_SHA256, "Attempt14 result drift")
    result = json.loads(ATTEMPT14_RESULT.read_text(encoding="utf-8"))
    checkpoint = Path(result["composite_checkpoint"]["path"])
    require(sha256_file(checkpoint) == ATTEMPT14_CHECKPOINT_SHA256, "Attempt14 checkpoint drift")
    model = ResidualRankUncertaintyHead(baseline).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    model.eval()
    return model, {"path": str(checkpoint.resolve()), "sha256": ATTEMPT14_CHECKPOINT_SHA256}


def support_proxy_values(
    outputs: dict[str, torch.Tensor],
    sample: Any,
    device: torch.device,
) -> dict[str, float]:
    with np.load(sample.label_path, allow_pickle=False) as payload:
        intrinsics = torch.from_numpy(np.asarray(payload["intrinsics_output"], dtype=np.float32)).to(device)
        gravity = torch.from_numpy(np.asarray(payload["gravity_up_camera_xyz"], dtype=np.float32)).to(device)
    gravity = gravity / torch.linalg.vector_norm(gravity).clamp_min(1.0e-6)
    depth = outputs["predicted_log_depth"][0, 0].exp()
    support = outputs["support_probability"][0, 0]
    depth_validity = outputs["depth_valid_probability"][0, 0]
    evidence_validity = outputs["evidence_valid_probability"][0, 0]
    depth_sigma_log = outputs["depth_log_sigma"][0, 0].exp()
    height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ray_x = (xx - intrinsics[0, 2]) / intrinsics[0, 0]
    ray_y = (yy - intrinsics[1, 2]) / intrinsics[1, 1]
    ray_dot_up = gravity[0] * ray_x + gravity[1] * ray_y + gravity[2]
    signed_plane_residual = outputs["camera_height_m"][0] + depth * ray_dot_up
    weights = (
        support.clamp(0.0, 1.0)
        * depth_validity.clamp(0.0, 1.0)
        * evidence_validity.clamp(0.0, 1.0)
    )
    valid = torch.isfinite(signed_plane_residual) & torch.isfinite(depth) & (depth >= 0.10) & (depth <= 10.0)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    denominator = weights.sum().clamp_min(1.0e-6)
    coverage = (weights > 0.10).float().mean().clamp_min(1.0e-4)
    geometry_abs = (weights * signed_plane_residual.abs()).sum() / denominator
    geometry_rms = torch.sqrt((weights * signed_plane_residual.square()).sum() / denominator)
    vertical_depth_sigma = depth * depth_sigma_log * ray_dot_up.abs()
    propagated = torch.sqrt((weights * vertical_depth_sigma.square()).sum() / denominator)
    support_entropy = (
        -(support.clamp(1.0e-5, 1.0 - 1.0e-5) * support.clamp(1.0e-5, 1.0 - 1.0e-5).log()
          + (1.0 - support).clamp(1.0e-5, 1.0) * (1.0 - support).clamp(1.0e-5, 1.0).log())
    ).mean()
    geometry_plus_depth = torch.sqrt(geometry_rms.square() + propagated.square())
    raw = outputs["support_residual_sigma_m"][0]
    candidates = {
        "raw_learned_sigma": raw,
        "geometry_abs_mean": geometry_abs,
        "geometry_rms": geometry_rms,
        "vertical_depth_sigma_rms": propagated,
        "geometry_plus_depth": geometry_plus_depth,
        "coverage_inverse_sqrt": torch.rsqrt(coverage),
        "geometry_rms_coverage": geometry_rms * torch.rsqrt(coverage),
        "geometry_plus_depth_coverage": geometry_plus_depth * torch.rsqrt(coverage),
        "support_entropy": support_entropy + 0.05,
        "depth_sigma_mean": depth_sigma_log.mean(),
        "boundary_sigma_mean": outputs["boundary_sigma_px"].mean() / 10.0,
        "evidence_unknown_mean": (1.0 - evidence_validity).mean() + 0.05,
    }
    result = {}
    for key, value in candidates.items():
        scalar = float(value)
        result[key] = max(scalar, 0.005) if math.isfinite(scalar) else 0.005
    return result


def parent_macro_nll(rows: list[dict[str, Any]], multiplier: float) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = (
            0.5 * float(row["normalized_squared_residual"]) / (multiplier * multiplier)
            + float(row["mean_log_sigma"])
            + math.log(multiplier)
            + GAUSSIAN_CONSTANT
        )
        grouped[str(row["parent_id"])].append(value)
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def choose_multiplier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grid = np.exp(np.linspace(math.log(0.20), math.log(5.0), 321))
    candidates = [
        {"multiplier": float(value), "parent_macro_nll": parent_macro_nll(rows, float(value))}
        for value in grid
    ]
    selected = min(candidates, key=lambda row: (row["parent_macro_nll"], row["multiplier"]))
    return {"selected": selected, "candidate_count": len(candidates)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    require(sha256_file(ATTEMPT13_RESULT) == EXPECTED_ATTEMPT13_RESULT_SHA256, "Attempt13 result drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()
    rows, data_receipt = consumed_rows()
    _, validation_parents = parent_split(rows)
    validation_rows = [row for row in rows if row["parent_id"] in set(validation_parents)]
    samples, feature_receipt = extract_features(
        validation_rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline = baseline_result["baseline_parameters"]
    model, checkpoint_receipt = load_model(baseline, device)
    attempt13 = json.loads(ATTEMPT13_RESULT.read_text(encoding="utf-8"))
    base_nll = {
        family: float(attempt13["final_evaluation"]["overall_metrics"][metric])
        for family, metric in (("depth", "depth_nll"), ("support", "support_nll"), ("boundary", "boundary_nll"))
    }

    cached_outputs: dict[str, dict[str, torch.Tensor]] = {}
    support_observations = []
    depth_rows = []
    boundary_rows = []
    with torch.no_grad():
        for sample in samples:
            outputs = native_outputs(forward_sample(model, sample, device), sample.native_hw)
            target = load_native_targets(sample, device)
            cached_outputs[sample.sample_id] = {key: value.detach() for key, value in outputs.items()}
            proxies = support_proxy_values(outputs, sample, device)
            support_residual = target["support_residual"][target["support_valid"]].abs()
            if support_residual.numel() > 0:
                support_observations.append({
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "target_abs_residual_mean": float(support_residual.mean()),
                    "target_squared_residual_mean": float(support_residual.square().mean()),
                    "proxies": proxies,
                })
            depth_residual = outputs["predicted_log_depth"] - target["depth"].clamp_min(0.01).log()
            depth_sigma = outputs["depth_log_sigma"].exp()
            depth_valid = target["depth_valid"]
            depth_rows.append({
                "parent_id": sample.parent_id,
                "normalized_squared_residual": float(((depth_residual[depth_valid] / depth_sigma[depth_valid]) ** 2).mean()),
                "mean_log_sigma": float(depth_sigma[depth_valid].log().mean()),
            })
            boundary_distance = (-BOUNDARY_DISTANCE_SCALE_PX * outputs["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
            boundary_residual = boundary_distance - target["boundary_distance"]
            boundary_sigma = outputs["boundary_sigma_px"]
            evidence_valid = target["evidence_valid"]
            if bool(evidence_valid.any()):
                boundary_rows.append({
                    "parent_id": sample.parent_id,
                    "normalized_squared_residual": float(((boundary_residual[evidence_valid] / boundary_sigma[evidence_valid]) ** 2).mean()),
                    "mean_log_sigma": float(boundary_sigma[evidence_valid].log().mean()),
                })

    support_candidates = []
    proxy_names = sorted(support_observations[0]["proxies"])
    for name in proxy_names:
        pairs = [
            (np.asarray([row["proxies"][name]]), np.asarray([row["target_abs_residual_mean"]]))
            for row in support_observations
        ]
        ordering = quantile_residual_summary(pairs)
        nll_rows = [
            {
                "parent_id": row["parent_id"],
                "normalized_squared_residual": row["target_squared_residual_mean"] / (row["proxies"][name] ** 2),
                "mean_log_sigma": math.log(row["proxies"][name]),
            }
            for row in support_observations
        ]
        calibration = choose_multiplier(nll_rows)
        support_candidates.append({
            "name": name,
            "ordering": ordering,
            "calibration": calibration,
            "eligible": bool(ordering["nondecreasing"]),
        })
    eligible_support = [row for row in support_candidates if row["eligible"]]
    selected_support = min(
        eligible_support or support_candidates,
        key=lambda row: (
            row["calibration"]["selected"]["parent_macro_nll"] + (0.0 if row["eligible"] else 1000.0),
            row["name"],
        ),
    )
    depth_calibration = choose_multiplier(depth_rows)
    boundary_calibration = choose_multiplier(boundary_rows)
    calibration = {
        "depth": {"source": "attempt14_depth_sigma", **depth_calibration["selected"]},
        "support": {
            "source": "deployment_geometry_proxy",
            "proxy": selected_support["name"],
            **selected_support["calibration"]["selected"],
        },
        "boundary": {"source": "attempt14_boundary_sigma", **boundary_calibration["selected"]},
    }

    metric_rows = []
    with torch.no_grad():
        for sample in samples:
            target = load_native_targets(sample, device)
            outputs = dict(cached_outputs[sample.sample_id])
            proxies = support_proxy_values(outputs, sample, device)
            outputs["depth_log_sigma"] = outputs["depth_log_sigma"] + math.log(calibration["depth"]["multiplier"])
            outputs["support_residual_sigma_m"] = torch.tensor(
                [proxies[calibration["support"]["proxy"]] * calibration["support"]["multiplier"]],
                device=device,
            )
            outputs["boundary_sigma_px"] = outputs["boundary_sigma_px"] * calibration["boundary"]["multiplier"]
            metric_rows.append({
                "sample_id": sample.sample_id,
                "parent_id": sample.parent_id,
                "orientation": sample.orientation,
                "metrics": frame_metrics(outputs, target),
            })
    calibrated_evaluation = {"frames": metric_rows, **aggregate_parent_metrics(metric_rows)}
    ordering = uncertainty_ordering(calibrated_evaluation)
    calibrated_nll = {
        family: float(calibrated_evaluation["overall_metrics"][metric])
        for family, metric in (("depth", "depth_nll"), ("support", "support_nll"), ("boundary", "boundary_nll"))
    }
    nll_ratios = {family: calibrated_nll[family] / max(abs(base_nll[family]), 1.0e-8) for family in base_nll}
    passed = all(row["nondecreasing"] for row in ordering.values()) and all(value <= 1.05 for value in nll_ratios.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt15_geometry_uncertainty_calibration_result_v1",
        "status": "ATTEMPT15_GEOMETRY_UNCERTAINTY_INTERNAL_PASS_FINAL_FRESH_CANARY_AUTHORIZATION_REQUIRED" if passed else "ATTEMPT15_GEOMETRY_UNCERTAINTY_INTERNAL_FAIL_NO_FRESH_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "fresh_canary_model_metrics_opened": False,
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "internal_validation_parents": validation_parents,
        "checkpoint": checkpoint_receipt,
        "support_proxy_candidates": support_candidates,
        "calibration": calibration,
        "attempt13_base_nll": base_nll,
        "calibrated_nll": calibrated_nll,
        "nll_ratios_to_attempt13": nll_ratios,
        "calibrated_ordering": ordering,
        "calibrated_evaluation": calibrated_evaluation,
        "decision": {
            "point_predictions_changed": False,
            "direct_sigma_pseudo_truth_used": False,
            "deployment_available_inputs_only": True,
            "fresh_canary_model_outputs_opened": False,
            "next_action_if_pass": "Bind exact checkpoint and calibration in one-shot fresh-canary execution lock.",
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
    print(json.dumps({"status": result["status"], "passed": result["passed"], "calibration": result["calibration"], "ordering": {family: row["nondecreasing"] for family, row in result["calibrated_ordering"].items()}, "nll_ratios": result["nll_ratios_to_attempt13"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
