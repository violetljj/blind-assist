#!/usr/bin/env python3
"""Calibrate all three uncertainty fields for the expanded Attempt-07 heads."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt05_uncertainty_calibrators import (  # noqa: E402
    BOUNDARY_FEATURE_NAMES,
    DEPTH_FEATURE_NAMES,
    PixelScaleCalibrator,
    _sample_valid,
    _train_one,
    calibration_features,
)
from train_ag_r2_f1_attempt06_uncertainty_calibrators import (  # noqa: E402
    SUPPORT_FEATURE_NAMES,
    _train_support,
    support_features,
)
from train_ag_r2_f1_attempt07_point_factor_expansion import load_rows  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    FactorSplitHead,
    apply_geometry,
    cache_model_outputs,
    extract_features,
    parent_vio_height_context,
    prepare,
    require,
    sha256_file,
)
from train_ag_r2_f1_factor_learnability_attempt04 import GEOMETRY_CONFIG  # noqa: E402


DEFAULT_POINT_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-point-factor-expansion-r0/result.json"
EXPECTED_POINT_RESULT_SHA256 = "580A94AD71B9C86A706D8FB233BF87AAD9376B503C8E7C35DF8F1102A34AE946"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-uncertainty-calibration-r0"
TRAINING_SEED = 1907


def _gaussian_nll(sigma: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp_min(1.0e-4)
    return sigma.log() + 0.5 * residual.square() / sigma.square()


def load_attempt07_calibrators(
    path: Path, device: torch.device
) -> tuple[PixelScaleCalibrator, PixelScaleCalibrator, PixelScaleCalibrator, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    models = []
    for family in ("depth", "boundary", "support"):
        row = payload["models"][family]
        config = row["config"]
        model = PixelScaleCalibrator(
            row["feature_mean"],
            row["feature_std"],
            int(config["hidden_channels"]),
            float(config["minimum_sigma"]),
            float(config["maximum_sigma"]),
        ).to(device)
        model.load_state_dict(row["state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models[0], models[1], models[2], payload["metadata"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    require(sha256_file(args.point_result) == EXPECTED_POINT_RESULT_SHA256, "Attempt-07 point result drift")
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    rows, data_receipt = load_rows()
    point_result = json.loads(args.point_result.read_text(encoding="utf-8"))
    require(bool(point_result["passed"]), "Attempt-07 point expansion not passed")
    fit_parents = list(point_result["fit_parents"])
    validation_parents = list(point_result["internal_validation_parents"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()
    samples, feature_receipt = extract_features(
        rows, args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device
    )
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    factor_models = []
    checkpoint_receipts = []
    for seed_row in point_result["seed_results"]:
        seed = int(seed_row["seed"])
        checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        expected = str(seed_row["selected_checkpoint"]["sha256"])
        require(sha256_file(checkpoint) == expected, f"Attempt-07 point checkpoint drift: {seed}")
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        model.eval()
        factor_models.append(model)
        checkpoint_receipts.append({"seed": seed, "path": str(checkpoint.resolve()), "sha256": expected})
    require([row["seed"] for row in checkpoint_receipts] == [17, 29, 43], "Attempt-07 seed order drift")

    by_parent: dict[str, list[Any]] = {}
    for sample in samples:
        by_parent.setdefault(sample.parent_id, []).append(sample)
    sampled = {
        family: {
            "fit_features": [],
            "fit_residual": [],
            "fit_source_sigma": [],
            "validation_features": [],
            "validation_residual": [],
            "validation_source_sigma": [],
        }
        for family in ("depth", "boundary")
    }
    support_rows = {
        "fit_features": [],
        "fit_residual": [],
        "fit_source_sigma": [],
        "validation_features": [],
        "validation_residual": [],
        "validation_source_sigma": [],
    }
    frame_receipts = []
    processed = 0
    for parent in sorted(by_parent):
        parent_samples = sorted(by_parent[parent], key=lambda sample: sample.sample_id)
        require(len(parent_samples) == 3, f"Attempt-07 parent frame count drift: {parent}")
        prepared = prepare(parent_samples, device)
        caches = [cache_model_outputs(model, prepared, device) for model in factor_models]
        context = parent_vio_height_context(prepared, caches[0], GEOMETRY_CONFIG["height"], device)
        role = "fit" if parent in fit_parents else "validation"
        for index, prepared_row in enumerate(prepared):
            sample = prepared_row["sample"]
            members = [cache[index] for cache in caches]
            base_depth = F.interpolate(
                sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
                sample.native_hw,
                mode="bilinear",
                align_corners=False,
            ).clamp_min(0.01)
            depth_features, boundary_features = calibration_features(members, base_depth)
            target = prepared_row["target"]
            depth_residual = (members[0]["predicted_log_depth"] - target["depth"].clamp_min(0.01).log()).abs()
            boundary_distance = (-3.0 * members[0]["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
            boundary_residual = (boundary_distance - target["boundary_distance"]).abs()
            depth_x, depth_y = _sample_valid(depth_features, depth_residual, target["depth_valid"], args.samples_per_frame, f"attempt07-depth:{sample.sample_id}")
            depth_source, _ = _sample_valid(members[0]["depth_log_sigma"].exp(), depth_residual, target["depth_valid"], args.samples_per_frame, f"attempt07-depth:{sample.sample_id}")
            boundary_x, boundary_y = _sample_valid(boundary_features, boundary_residual, target["evidence_valid"], args.samples_per_frame, f"attempt07-boundary:{sample.sample_id}")
            boundary_source, _ = _sample_valid(members[0]["boundary_sigma_px"], boundary_residual, target["evidence_valid"], args.samples_per_frame, f"attempt07-boundary:{sample.sample_id}")
            for family, features, residual, source_sigma in (
                ("depth", depth_x, depth_y, depth_source[:, 0]),
                ("boundary", boundary_x, boundary_y, boundary_source[:, 0]),
            ):
                sampled[family][f"{role}_features"].append(features)
                sampled[family][f"{role}_residual"].append(residual)
                sampled[family][f"{role}_source_sigma"].append(source_sigma)
            factors, geometry_receipt = apply_geometry(members[0], sample, GEOMETRY_CONFIG, baseline, device, context)
            support_valid = target["support_valid"]
            support_count = int(support_valid.sum())
            if bool(target["plane_valid"]) and support_count > 0:
                residual_rms = target["support_residual"][support_valid].square().mean().sqrt()[None]
                feature = support_features(members, base_depth, factors, geometry_receipt, sample.orientation)
                support_rows[f"{role}_features"].append(feature.detach().cpu())
                support_rows[f"{role}_residual"].append(residual_rms.detach().cpu())
                support_rows[f"{role}_source_sigma"].append(factors["support_residual_sigma_m"].detach().cpu())
            frame_receipts.append({"sample_id": sample.sample_id, "parent_id": parent, "calibration_role": role.upper(), "depth_samples": int(depth_y.numel()), "boundary_samples": int(boundary_y.numel()), "support_valid_pixels": support_count})
            processed += 1
        del prepared, caches
        torch.cuda.empty_cache()
        if processed % 6 == 0:
            print(json.dumps({"feature_sampling_frames": processed, "total_frames": len(samples)}), flush=True)
    del factor_models, samples
    torch.cuda.empty_cache()

    tensors = {
        family: {key: torch.cat(value, dim=0) for key, value in rows_by_key.items()}
        for family, rows_by_key in sampled.items()
    }
    support_tensors = {key: torch.cat(value, dim=0) for key, value in support_rows.items()}
    depth_model, depth_training = _train_one("depth", tensors["depth"]["fit_features"], tensors["depth"]["fit_residual"], tensors["depth"]["validation_features"], tensors["depth"]["validation_residual"], device, args.optimizer_steps)
    boundary_model, boundary_training = _train_one("boundary", tensors["boundary"]["fit_features"], tensors["boundary"]["fit_residual"], tensors["boundary"]["validation_features"], tensors["boundary"]["validation_residual"], device, args.optimizer_steps)
    support_model, support_training = _train_support(support_tensors["fit_features"], support_tensors["fit_residual"], support_tensors["validation_features"], support_tensors["validation_residual"], support_tensors["validation_source_sigma"], device, args.support_optimizer_steps)

    for family, training in (("depth", depth_training), ("boundary", boundary_training)):
        baseline_nll = float(_gaussian_nll(tensors[family]["validation_source_sigma"], tensors[family]["validation_residual"]).mean())
        training["internal_validation"]["baseline_mean_gaussian_nll"] = baseline_nll
        training["internal_validation"]["proper_score_gain"] = baseline_nll - training["internal_validation"]["mean_gaussian_nll"]
    passed = all(
        training["internal_validation"]["nondecreasing"]
        and training["internal_validation"]["proper_score_gain"] > 0.0
        for training in (depth_training, boundary_training, support_training)
    )
    checkpoint_path = args.output_dir / "uncertainty_calibrators.pt"
    checkpoint_payload = {
        "schema": "blindassist_ag_r2_f1_attempt07_uncertainty_calibrators_v1",
        "models": {
            "depth": {"config": {"hidden_channels": 32, "minimum_sigma": 0.01, "maximum_sigma": 3.0}, "feature_names": list(DEPTH_FEATURE_NAMES), "feature_mean": depth_model.feature_mean.detach().cpu(), "feature_std": depth_model.feature_std.detach().cpu(), "state_dict": {key: value.detach().cpu() for key, value in depth_model.state_dict().items()}},
            "boundary": {"config": {"hidden_channels": 32, "minimum_sigma": 0.25, "maximum_sigma": 32.0}, "feature_names": list(BOUNDARY_FEATURE_NAMES), "feature_mean": boundary_model.feature_mean.detach().cpu(), "feature_std": boundary_model.feature_std.detach().cpu(), "state_dict": {key: value.detach().cpu() for key, value in boundary_model.state_dict().items()}},
            "support": {"config": {"hidden_channels": 16, "minimum_sigma": 0.03, "maximum_sigma": 2.5}, "feature_names": list(SUPPORT_FEATURE_NAMES), "feature_mean": support_model.feature_mean.detach().cpu(), "feature_std": support_model.feature_std.detach().cpu(), "state_dict": {key: value.detach().cpu() for key, value in support_model.state_dict().items()}},
        },
        "metadata": {"training_seed": TRAINING_SEED, "fit_parents": fit_parents, "internal_validation_parents": validation_parents, "data_receipt": data_receipt, "point_result": {"path": str(args.point_result.resolve()), "sha256": EXPECTED_POINT_RESULT_SHA256}, "point_checkpoint_receipts": checkpoint_receipts},
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    _, _, _, metadata = load_attempt07_calibrators(checkpoint_path, device)
    require(metadata["point_result"]["sha256"] == EXPECTED_POINT_RESULT_SHA256, "Attempt-07 calibrator roundtrip drift")
    result = {
        "schema": "blindassist_ag_r2_f1_attempt07_uncertainty_calibration_result_v1",
        "status": "ATTEMPT07_UNCERTAINTY_CALIBRATION_INTERNAL_PASS_FINAL_CANARY_LOCK_REQUIRED" if passed else "ATTEMPT07_UNCERTAINTY_CALIBRATION_INTERNAL_FAIL_NO_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch": torch.__version__,
        "data_receipt": data_receipt,
        "point_result": {"path": str(args.point_result.resolve()), "sha256": EXPECTED_POINT_RESULT_SHA256},
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "frame_count": len(frame_receipts),
        "parent_count": len(set(fit_parents) | set(validation_parents)),
        "frame_sample_receipts": frame_receipts,
        "feature_receipt": feature_receipt,
        "point_checkpoint_receipts": checkpoint_receipts,
        "calibrator_checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha, "bytes": checkpoint_path.stat().st_size},
        "depth": depth_training,
        "boundary": boundary_training,
        "support": support_training,
        "decision": {"point_factor_parameters_changed_during_calibration": False, "preserved_canary_metrics_opened": False, "next_action_if_pass": "Freeze one final canary-only execution lock, then serialize real factor tensors only if all gates pass."},
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-result", type=Path, default=DEFAULT_POINT_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-frame", type=int, default=4096)
    parser.add_argument("--optimizer-steps", type=int, default=1200)
    parser.add_argument("--support-optimizer-steps", type=int, default=1600)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "checkpoint": result["calibrator_checkpoint"], "depth": result["depth"]["internal_validation"], "boundary": result["boundary"]["internal_validation"], "support": result["support"]["internal_validation"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
