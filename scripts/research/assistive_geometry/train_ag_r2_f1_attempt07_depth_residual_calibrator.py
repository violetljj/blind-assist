#!/usr/bin/env python3
"""Fit a bounded residual correction on the expanded head's raw depth sigma."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt05_uncertainty_calibrators import (  # noqa: E402
    DEPTH_FEATURE_NAMES,
    _sample_valid,
    calibration_features,
)
from train_ag_r2_f1_attempt07_point_factor_expansion import load_rows  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    FactorSplitHead,
    cache_model_outputs,
    extract_features,
    prepare,
    require,
    sha256_file,
)


POINT_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-point-factor-expansion-r0/result.json"
EXPECTED_POINT_RESULT_SHA256 = "580A94AD71B9C86A706D8FB233BF87AAD9376B503C8E7C35DF8F1102A34AE946"
ATTEMPT07_CALIBRATOR_R0 = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-uncertainty-calibration-r0/uncertainty_calibrators.pt"
EXPECTED_ATTEMPT07_CALIBRATOR_R0_SHA256 = "56DACC1B75AC0FF2BEFA86F2323FBFDDF918538F48B0E1E0F58B21A25E31D7D4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-uncertainty-calibration-r1"
TRAINING_SEED = 2007


class ResidualDepthCalibrator(nn.Module):
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        hidden_channels: int = 32,
        maximum_log_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1.0e-4))
        self.maximum_log_delta = float(maximum_log_delta)
        self.net = nn.Sequential(
            nn.Linear(int(feature_mean.numel()), hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            batch, channels, height, width = features.shape
            flattened = features.permute(0, 2, 3, 1).reshape(-1, channels)
            output = self.forward(flattened).reshape(batch, height, width)
            return output[:, None]
        require(features.ndim == 2, "residual depth feature rank drift")
        normalized = (features - self.feature_mean) / self.feature_std
        delta = self.maximum_log_delta * torch.tanh(self.net(normalized).squeeze(-1))
        raw_log_sigma = features[:, 0]
        return (raw_log_sigma + delta).clamp(math.log(0.01), math.log(3.0))


def _nll(log_sigma: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return log_sigma + 0.5 * residual.square() * torch.exp(-2.0 * log_sigma)


def _summary(log_sigma: torch.Tensor, residual: torch.Tensor) -> dict[str, Any]:
    sigma = log_sigma.exp().detach().cpu().numpy().astype(np.float64, copy=False)
    target = residual.detach().cpu().numpy().astype(np.float64, copy=False)
    order = np.argsort(sigma, kind="stable")
    groups = np.array_split(order, 4)
    means = [float(target[group].mean()) for group in groups]
    return {
        "mean_gaussian_nll": float(_nll(log_sigma, residual).mean()),
        "sigma_quantile_means": [float(sigma[group].mean()) for group in groups],
        "quantile_residual_means": means,
        "nondecreasing": all(a <= b + 1.0e-6 for a, b in zip(means, means[1:])),
        "observation_count": int(target.size),
    }


def _evaluate(
    model: ResidualDepthCalibrator,
    features: torch.Tensor,
    residual: torch.Tensor,
    batch_size: int = 65536,
) -> dict[str, Any]:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            outputs.append(model(features[start : start + batch_size]))
    return _summary(torch.cat(outputs), residual)


def load_attempt07_final_calibration(
    path: Path, device: torch.device
) -> tuple[ResidualDepthCalibrator, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    row = payload["depth_residual_calibrator"]
    model = ResidualDepthCalibrator(
        row["feature_mean"],
        row["feature_std"],
        int(row["hidden_channels"]),
        float(row["maximum_log_delta"]),
    ).to(device)
    model.load_state_dict(row["state_dict"], strict=True)
    model.eval()
    return model, payload["support_model"], payload["metadata"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    require(sha256_file(args.point_result) == EXPECTED_POINT_RESULT_SHA256, "point result drift")
    require(sha256_file(args.attempt07_calibrator_r0) == EXPECTED_ATTEMPT07_CALIBRATOR_R0_SHA256, "support calibrator source drift")
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    rows, data_receipt = load_rows()
    point = json.loads(args.point_result.read_text(encoding="utf-8"))
    fit_parents = set(point["fit_parents"])
    validation_parents = set(point["internal_validation_parents"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()
    samples, feature_receipt = extract_features(rows, args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device)
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    factor_models = []
    checkpoint_receipts = []
    for seed_row in point["seed_results"]:
        checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        expected = str(seed_row["selected_checkpoint"]["sha256"])
        require(sha256_file(checkpoint) == expected, "point checkpoint drift")
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        model.eval()
        factor_models.append(model)
        checkpoint_receipts.append({"seed": seed_row["seed"], "path": str(checkpoint.resolve()), "sha256": expected})

    sampled = {"fit_features": [], "fit_residual": [], "validation_features": [], "validation_residual": []}
    for index, sample in enumerate(samples):
        prepared = prepare([sample], device)
        members = [cache_model_outputs(model, prepared, device)[0] for model in factor_models]
        base_depth = F.interpolate(sample.base_depth_feature[None].to(device=device, dtype=torch.float32), sample.native_hw, mode="bilinear", align_corners=False).clamp_min(0.01)
        depth_features, _ = calibration_features(members, base_depth)
        target = prepared[0]["target"]
        residual = (members[0]["predicted_log_depth"] - target["depth"].clamp_min(0.01).log()).abs()
        role = "fit" if sample.parent_id in fit_parents else "validation"
        features, sampled_residual = _sample_valid(depth_features, residual, target["depth_valid"], args.samples_per_frame, f"attempt07-residual-depth:{sample.sample_id}")
        sampled[f"{role}_features"].append(features)
        sampled[f"{role}_residual"].append(sampled_residual)
        if (index + 1) % 6 == 0:
            print(json.dumps({"feature_sampling_frames": index + 1, "total_frames": len(samples)}), flush=True)
    tensors = {key: torch.cat(value, dim=0).to(device) for key, value in sampled.items()}
    mean = tensors["fit_features"].mean(dim=0)
    std = tensors["fit_features"].std(dim=0, correction=0).clamp_min(1.0e-4)
    model = ResidualDepthCalibrator(mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)
    raw_validation = _summary(tensors["validation_features"][:, 0], tensors["validation_residual"])
    best_score = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    trace = []
    batch_size = min(8192, tensors["fit_features"].shape[0])
    for step in range(1, args.optimizer_steps + 1):
        indices = torch.randint(tensors["fit_features"].shape[0], (batch_size,), generator=generator, device=device)
        features = tensors["fit_features"][indices]
        residual = tensors["fit_residual"][indices]
        log_sigma = model(features)
        target_log = residual.clamp_min(0.01).log()
        permutation = torch.randperm(batch_size, generator=generator, device=device)
        target_difference = target_log - target_log[permutation]
        ranking_mask = target_difference.abs() >= 0.20
        ranking = F.softplus(-torch.sign(target_difference[ranking_mask]) * (log_sigma[ranking_mask] - log_sigma[permutation][ranking_mask])).mean()
        delta = log_sigma - features[:, 0]
        loss = _nll(log_sigma, residual).mean() + 0.20 * F.smooth_l1_loss(log_sigma, target_log) + 0.50 * ranking + 0.05 * delta.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 25 == 0 or step == args.optimizer_steps:
            evaluation = _evaluate(model, tensors["validation_features"], tensors["validation_residual"])
            eligible = evaluation["nondecreasing"] and evaluation["mean_gaussian_nll"] <= raw_validation["mean_gaussian_nll"] + 0.02
            score = evaluation["mean_gaussian_nll"] + (0.0 if eligible else 1.0e6)
            trace.append({"step": step, "eligible": eligible, "selection_score": score, **evaluation})
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    final_validation = _evaluate(model, tensors["validation_features"], tensors["validation_residual"])
    final_fit = _evaluate(model, tensors["fit_features"], tensors["fit_residual"])
    passed = bool(final_validation["nondecreasing"] and final_validation["mean_gaussian_nll"] <= raw_validation["mean_gaussian_nll"] + 0.02)
    source_payload = torch.load(args.attempt07_calibrator_r0, map_location="cpu", weights_only=True)
    checkpoint_path = args.output_dir / "final_uncertainty_calibration.pt"
    checkpoint_payload = {
        "schema": "blindassist_ag_r2_f1_attempt07_final_uncertainty_calibration_v1",
        "depth_residual_calibrator": {
            "feature_names": list(DEPTH_FEATURE_NAMES),
            "feature_mean": model.feature_mean.detach().cpu(),
            "feature_std": model.feature_std.detach().cpu(),
            "hidden_channels": 32,
            "maximum_log_delta": 0.5,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        "boundary_mode": "expanded_seed17_raw_sigma",
        "support_model": source_payload["models"]["support"],
        "metadata": {
            "point_result": {"path": str(args.point_result.resolve()), "sha256": EXPECTED_POINT_RESULT_SHA256},
            "point_checkpoint_receipts": checkpoint_receipts,
            "data_receipt": data_receipt,
            "attempt07_support_calibrator_source_sha256": EXPECTED_ATTEMPT07_CALIBRATOR_R0_SHA256,
            "fit_parents": sorted(fit_parents),
            "internal_validation_parents": sorted(validation_parents),
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    reload_model, _, metadata = load_attempt07_final_calibration(checkpoint_path, device)
    require(metadata["point_result"]["sha256"] == EXPECTED_POINT_RESULT_SHA256, "final calibration checkpoint drift")
    require(all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reload_model.state_dict().values())), "residual calibrator roundtrip drift")
    result = {
        "schema": "blindassist_ag_r2_f1_attempt07_depth_residual_calibration_result_v1",
        "status": "ATTEMPT07_FINAL_UNCERTAINTY_INTERNAL_PASS_CANARY_LOCK_REQUIRED" if passed else "ATTEMPT07_FINAL_UNCERTAINTY_INTERNAL_FAIL_NO_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "point_result": {"path": str(args.point_result.resolve()), "sha256": EXPECTED_POINT_RESULT_SHA256},
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "point_checkpoint_receipts": checkpoint_receipts,
        "raw_depth_internal_validation": raw_validation,
        "depth_residual_calibrator": {"best_step": best_step, "best_selection_score": best_score, "fit": final_fit, "internal_validation": final_validation, "trace": trace},
        "boundary_internal_evidence": {"mode": "expanded_seed17_raw_sigma", "diagnostic_path": "artifacts.local/experiments/ag-r2-f1-attempt07-raw-uncertainty-diagnostic-r0/result.json"},
        "support_internal_evidence": {"mode": "attempt07_scalar_support_calibrator", "source_checkpoint_sha256": EXPECTED_ATTEMPT07_CALIBRATOR_R0_SHA256},
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha, "bytes": checkpoint_path.stat().st_size},
        "decision": {"point_factor_parameters_changed": False, "preserved_canary_metrics_opened": False, "next_action_if_pass": "Freeze and run the one-shot preserved canary; serialize real factor tensors only on complete pass."},
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-result", type=Path, default=POINT_RESULT)
    parser.add_argument("--attempt07-calibrator-r0", type=Path, default=ATTEMPT07_CALIBRATOR_R0)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-frame", type=int, default=4096)
    parser.add_argument("--optimizer-steps", type=int, default=1600)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "raw": result["raw_depth_internal_validation"], "calibrated": result["depth_residual_calibrator"]["internal_validation"], "best_step": result["depth_residual_calibrator"]["best_step"], "checkpoint": result["checkpoint"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
