#!/usr/bin/env python3
"""Fit a compact frame-level metric-scale calibrator for the expanded AG head."""

from __future__ import annotations

import argparse
import json
import math
import time
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
    DEPTH_FEATURE_NAMES,
    calibration_features,
)
from train_ag_r2_f1_attempt07_point_factor_expansion import load_rows  # noqa: E402
from train_ag_r2_f1_factor_learnability import masked_mean  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    FactorSplitHead,
    cache_model_outputs,
    evaluate_cached,
    extract_features,
    gate,
    prepare,
    require,
    sha256_file,
)
from train_ag_r2_f1_factor_learnability_attempt04 import GEOMETRY_CONFIG  # noqa: E402


POINT_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-point-factor-expansion-r0/result.json"
EXPECTED_POINT_RESULT_SHA256 = "580A94AD71B9C86A706D8FB233BF87AAD9376B503C8E7C35DF8F1102A34AE946"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt08-global-scale-calibration-r0"
RIDGE_LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SHRINKAGES = (0.5, 0.75, 1.0)
UNCERTAINTY_GAMMAS = (0.0, 0.25, 0.5, 1.0)

GLOBAL_FEATURE_NAMES = tuple(
    [f"mean_{name}" for name in DEPTH_FEATURE_NAMES]
    + [f"std_{name}" for name in DEPTH_FEATURE_NAMES]
    + [
        "model_vs_depthart_weighted_log_shift",
        "depth_gate",
        "orientation_landscape",
        "orientation_portrait",
        "orientation_square",
    ]
)


def global_features(
    members: list[dict[str, torch.Tensor]],
    base_depth: torch.Tensor,
    orientation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth_features, _ = calibration_features(members, base_depth)
    flattened = depth_features.flatten(2)
    mean = flattened.mean(dim=2)[0]
    std = flattened.std(dim=2, correction=0)[0]
    weight = members[0]["depth_valid_probability"].clamp_min(1.0e-3)
    shift = ((members[0]["predicted_log_depth"] - base_depth.clamp_min(0.01).log()) * weight).sum() / weight.sum().clamp_min(1.0e-6)
    flags = torch.tensor(
        [
            float(orientation == "landscape"),
            float(orientation == "portrait"),
            float(orientation not in {"landscape", "portrait"}),
        ],
        device=mean.device,
        dtype=mean.dtype,
    )
    vector = torch.cat((mean, std, shift[None], members[0]["depth_gate"].reshape(1), flags))
    require(vector.numel() == len(GLOBAL_FEATURE_NAMES), "global scale feature count drift")
    return vector, shift


def fit_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    std = features.std(dim=0, correction=0).clamp_min(1.0e-4)
    normalized = (features - mean) / std
    design = torch.cat((normalized, torch.ones((normalized.shape[0], 1), dtype=normalized.dtype)), dim=1)
    penalty = torch.eye(design.shape[1], dtype=design.dtype) * float(ridge_lambda)
    penalty[-1, -1] = 0.0
    solution = torch.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return mean, std, solution[:-1], solution[-1]


def apply_candidate(
    cached: list[dict[str, torch.Tensor]],
    features: torch.Tensor,
    shifts: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shrinkage: float,
    uncertainty_gamma: float,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, float]]]:
    prediction = ((features - mean) / std) @ weight + bias
    correction = (float(shrinkage) * prediction).clamp(-0.75, 0.75)
    rows: list[dict[str, torch.Tensor]] = []
    receipts: list[dict[str, float]] = []
    for index, original in enumerate(cached):
        output = dict(original)
        output["predicted_log_depth"] = original["predicted_log_depth"] - correction[index]
        raw_sigma = original["depth_log_sigma"].exp()
        sigma = torch.sqrt(raw_sigma.square() + float(uncertainty_gamma) * shifts[index].abs().square()).clamp(0.01, 3.0)
        output["depth_log_sigma"] = sigma.log()
        rows.append(output)
        receipts.append(
            {
                "predicted_log_scale_correction": float(prediction[index]),
                "applied_log_scale_correction": float(correction[index]),
                "model_vs_depthart_weighted_log_shift": float(shifts[index]),
            }
        )
    return rows, receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.point_result) == EXPECTED_POINT_RESULT_SHA256, "point result drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    point = json.loads(args.point_result.read_text(encoding="utf-8"))
    fit_parents = set(point["fit_parents"])
    validation_parents = set(point["internal_validation_parents"])
    rows, data_receipt = load_rows()
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    prepared = prepare(samples, device)
    caches: list[list[dict[str, torch.Tensor]]] = []
    checkpoint_receipts = []
    for seed_row in point["seed_results"]:
        checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        expected = str(seed_row["selected_checkpoint"]["sha256"])
        require(sha256_file(checkpoint) == expected, "point checkpoint drift")
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        caches.append(cache_model_outputs(model, prepared, device))
        checkpoint_receipts.append({"seed": seed_row["seed"], "path": str(checkpoint.resolve()), "sha256": expected})

    vectors = []
    shifts = []
    targets = []
    for index, (sample, prepared_row) in enumerate(zip(samples, prepared)):
        members = [cache[index] for cache in caches]
        base_depth = F.interpolate(
            sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
            sample.native_hw,
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0.01)
        vector, shift = global_features(members, base_depth, sample.orientation)
        target = prepared_row["target"]
        signed_scale_error = masked_mean(
            members[0]["predicted_log_depth"] - target["depth"].clamp_min(0.01).log(),
            target["depth_valid"],
        )
        vectors.append(vector.detach().cpu().double())
        shifts.append(shift.detach().cpu().double())
        targets.append(signed_scale_error.detach().cpu().double())
    all_features = torch.stack(vectors)
    all_shifts = torch.stack(shifts)
    all_targets = torch.stack(targets)
    fit_mask = torch.tensor([sample.parent_id in fit_parents for sample in samples], dtype=torch.bool)
    validation_mask = torch.tensor([sample.parent_id in validation_parents for sample in samples], dtype=torch.bool)
    require(int(fit_mask.sum()) == 60 and int(validation_mask.sum()) == 15, "global scale split drift")
    validation_samples = [sample for sample in samples if sample.parent_id in validation_parents]
    validation_prepared = [row for sample, row in zip(samples, prepared) if sample.parent_id in validation_parents]
    validation_cached = [row for sample, row in zip(samples, caches[0]) if sample.parent_id in validation_parents]
    baseline_evaluation = evaluate_cached(validation_prepared, None, baseline, None, device)
    candidates = []
    models: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for ridge_lambda in RIDGE_LAMBDAS:
        model_state = fit_ridge(all_features[fit_mask], all_targets[fit_mask], ridge_lambda)
        models[ridge_lambda] = model_state
        mean, std, weight, bias = model_state
        for shrinkage in SHRINKAGES:
            for uncertainty_gamma in UNCERTAINTY_GAMMAS:
                candidate_cache, receipts = apply_candidate(
                    validation_cached,
                    all_features[validation_mask],
                    all_shifts[validation_mask],
                    mean,
                    std,
                    weight,
                    bias,
                    shrinkage,
                    uncertainty_gamma,
                )
                evaluation = evaluate_cached(validation_prepared, candidate_cache, baseline, GEOMETRY_CONFIG, device)
                candidate_gate = gate(evaluation, baseline_evaluation, 208)
                primary_pass_count = sum(row["passed"] for row in candidate_gate["metric_improvements"].values())
                uncertainty_pass_count = sum(row["passed"] for row in candidate_gate["uncertainty"].values())
                candidates.append(
                    {
                        "ridge_lambda": ridge_lambda,
                        "shrinkage": shrinkage,
                        "uncertainty_gamma": uncertainty_gamma,
                        "primary_pass_count": primary_pass_count,
                        "uncertainty_pass_count": uncertainty_pass_count,
                        "evaluation": evaluation,
                        "gate": candidate_gate,
                        "receipts": receipts,
                    }
                )
    selected = min(
        candidates,
        key=lambda row: (
            -row["primary_pass_count"],
            -row["uncertainty_pass_count"],
            -float(row["gate"]["metric_improvements"]["depth_scale_abs_log_error"]["bootstrap_95_lower"]),
            -float(row["gate"]["uncertainty"]["depth"]["proper_score_gain"]),
            row["ridge_lambda"],
            row["shrinkage"],
            row["uncertainty_gamma"],
        ),
    )
    mean, std, weight, bias = models[float(selected["ridge_lambda"])]
    checkpoint_path = args.output_dir / "global_scale_calibrator.pt"
    checkpoint_payload = {
        "schema": "blindassist_ag_r2_f1_attempt08_global_scale_calibrator_v1",
        "feature_names": list(GLOBAL_FEATURE_NAMES),
        "feature_mean": mean.float(),
        "feature_std": std.float(),
        "weight": weight.float(),
        "bias": bias.float(),
        "ridge_lambda": float(selected["ridge_lambda"]),
        "shrinkage": float(selected["shrinkage"]),
        "uncertainty_gamma": float(selected["uncertainty_gamma"]),
        "maximum_abs_log_correction": 0.75,
        "point_result_sha256": EXPECTED_POINT_RESULT_SHA256,
        "checkpoint_receipts": checkpoint_receipts,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    passed = bool(
        selected["gate"]["metric_improvements"]["depth_scale_abs_log_error"]["passed"]
        and selected["gate"]["uncertainty"]["depth"]["passed"]
        and selected["primary_pass_count"] >= 9
        and selected["uncertainty_pass_count"] == 3
    )
    result = {
        "schema": "blindassist_ag_r2_f1_attempt08_global_scale_calibration_result_v1",
        "status": "ATTEMPT08_GLOBAL_SCALE_INTERNAL_PASS_HEIGHT_RECALIBRATION_REQUIRED" if passed else "ATTEMPT08_GLOBAL_SCALE_INTERNAL_FAIL_NO_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "preserved_canary_metrics_opened": False,
        "fit_parents": sorted(fit_parents),
        "internal_validation_parents": sorted(validation_parents),
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "point_result": {"path": str(args.point_result.resolve()), "sha256": EXPECTED_POINT_RESULT_SHA256},
        "point_checkpoint_receipts": checkpoint_receipts,
        "feature_names": list(GLOBAL_FEATURE_NAMES),
        "selection_order": ["primary_pass_count_desc", "uncertainty_pass_count_desc", "depth_scale_bootstrap_lower_desc", "depth_proper_gain_desc"],
        "selected": selected,
        "candidates": candidates,
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha, "bytes": checkpoint_path.stat().st_size},
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-result", type=Path, default=POINT_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    selected = result["selected"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "ridge_lambda": selected["ridge_lambda"],
                "shrinkage": selected["shrinkage"],
                "uncertainty_gamma": selected["uncertainty_gamma"],
                "primary_pass_count": selected["primary_pass_count"],
                "uncertainty_pass_count": selected["uncertainty_pass_count"],
                "depth_scale_gate": selected["gate"]["metric_improvements"]["depth_scale_abs_log_error"],
                "depth_uncertainty": selected["gate"]["uncertainty"]["depth"],
                "checkpoint": result["checkpoint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
