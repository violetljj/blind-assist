#!/usr/bin/env python3
"""Train compact depth and boundary uncertainty calibrators on consumed AG-F1 evidence.

The calibrators are small pointwise MLPs. They do not alter any point factor;
they estimate conditional residual scale from frozen factor-model signals. New
Attempt-05 selection and canary labels are explicitly excluded from this run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_ATTEMPT02_RESULT,
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_COMPOSITES,
    EXPECTED_DEPTHART_SHA256,
    FactorSplitHead,
    cache_model_outputs,
    extract_features,
    prepare,
    require,
    sha256_file,
)


CONSUMED_LABEL_RESULTS = (
    (
        "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0/result.json",
        "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422",
    ),
    (
        "artifacts.local/experiments/ag-r2-f1-attempt02-fresh-canary-labels-r2/result.json",
        "7AFB581F30779A53CF1A54C15B06CFE3176D45D3CD5B92F52544283174188CCD",
    ),
    (
        "artifacts.local/experiments/ag-r2-f1-attempt02-corrected-canary-labels-r0/result.json",
        "E153A0E97678676033F6BF068C15439CDCC895FC97926966C8FAD82CD10ABB66",
    ),
    (
        "artifacts.local/experiments/ag-r2-f1-attempt03-fresh-held-labels-r1/result.json",
        "5FEFA4400657028FA489DB74D641786EB03BA9113C85893BDD7B65594FAD747C",
    ),
)
FORBIDDEN_ATTEMPT05_LABEL_RESULT = (
    "artifacts.local/experiments/ag-r2-f1-attempt05-fresh-ag-held-labels-r0/result.json",
    "4DBF0E85F45357C613221DF9F2C5A5E3B0971C314EB29D1967C02E0D6FAEB7CC",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-uncertainty-calibration-r3"
SPLIT_TOKEN = "AG_R2_F1_ATTEMPT05_UNCERTAINTY_CALIBRATION_2026-08-11"
TRAINING_SEED = 1705

DEPTH_FEATURE_NAMES = (
    "seed17_log_sigma",
    "three_seed_log_depth_std",
    "seed17_vs_base_abs_log_disagreement",
    "seed17_log_depth",
    "base_log_depth",
    "depth_valid_probability",
    "evidence_valid_probability",
    "support_probability",
    "obstacle_probability",
    "boundary_probability",
    "seed17_log_depth_gradient",
    "base_log_depth_gradient",
    "normalized_x",
    "normalized_y",
)
BOUNDARY_FEATURE_NAMES = (
    "seed17_log_boundary_sigma",
    "predicted_boundary_distance_px",
    "boundary_probability",
    "boundary_probability_gradient",
    "evidence_valid_probability",
    "depth_valid_probability",
    "support_probability",
    "support_probability_gradient",
    "obstacle_probability",
    "obstacle_probability_gradient",
    "seed17_log_depth",
    "seed17_log_depth_gradient",
    "three_seed_log_depth_std",
    "seed17_vs_base_abs_log_disagreement",
    "normalized_x",
    "normalized_y",
)


class PixelScaleCalibrator(nn.Module):
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        hidden_channels: int,
        minimum_sigma: float,
        maximum_sigma: float,
    ) -> None:
        super().__init__()
        require(feature_mean.ndim == feature_std.ndim == 1, "calibrator normalization rank drift")
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1.0e-4))
        self.minimum_log_sigma = math.log(float(minimum_sigma))
        self.maximum_log_sigma = math.log(float(maximum_sigma))
        feature_count = int(feature_mean.numel())
        self.net = nn.Sequential(
            nn.Linear(feature_count, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            normalized = (features - self.feature_mean) / self.feature_std
            return self.net(normalized).squeeze(-1).clamp(self.minimum_log_sigma, self.maximum_log_sigma)
        require(features.ndim == 4, "calibrator feature rank drift")
        batch, channels, height, width = features.shape
        require(channels == self.feature_mean.numel(), "calibrator feature channel drift")
        flattened = features.permute(0, 2, 3, 1).reshape(-1, channels)
        output = self.forward(flattened).reshape(batch, height, width)
        return output[:, None]


def _gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
    require(value.ndim == 4 and value.shape[1] == 1, "gradient input shape drift")
    dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1.0e-12)


def _coordinate_features(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = reference.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx[None, None], yy[None, None]


def calibration_features(
    members: list[dict[str, torch.Tensor]],
    base_depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    require(len(members) == 3, "three-seed calibration members required")
    seed17 = members[0]
    seed_logs = torch.stack([row["predicted_log_depth"] for row in members])
    epistemic = seed_logs.std(dim=0, correction=0)
    base_log = base_depth.clamp_min(0.01).log()
    base_disagreement = (seed17["predicted_log_depth"] - base_log).abs()
    x, y = _coordinate_features(seed17["predicted_log_depth"])
    depth_features = torch.cat(
        (
            seed17["depth_log_sigma"],
            epistemic,
            base_disagreement,
            seed17["predicted_log_depth"],
            base_log,
            seed17["depth_valid_probability"],
            seed17["evidence_valid_probability"],
            seed17["support_probability"],
            seed17["obstacle_probability"],
            seed17["boundary_probability"],
            _gradient_magnitude(seed17["predicted_log_depth"]),
            _gradient_magnitude(base_log),
            x,
            y,
        ),
        dim=1,
    )
    boundary_distance = (-3.0 * seed17["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
    boundary_features = torch.cat(
        (
            seed17["boundary_sigma_px"].clamp_min(1.0e-3).log(),
            boundary_distance,
            seed17["boundary_probability"],
            _gradient_magnitude(seed17["boundary_probability"]),
            seed17["evidence_valid_probability"],
            seed17["depth_valid_probability"],
            seed17["support_probability"],
            _gradient_magnitude(seed17["support_probability"]),
            seed17["obstacle_probability"],
            _gradient_magnitude(seed17["obstacle_probability"]),
            seed17["predicted_log_depth"],
            _gradient_magnitude(seed17["predicted_log_depth"]),
            epistemic,
            base_disagreement,
            x,
            y,
        ),
        dim=1,
    )
    require(depth_features.shape[1] == len(DEPTH_FEATURE_NAMES), "depth calibration feature drift")
    require(boundary_features.shape[1] == len(BOUNDARY_FEATURE_NAMES), "boundary calibration feature drift")
    return depth_features, boundary_features


def _sample_valid(
    features: torch.Tensor,
    residual: torch.Tensor,
    valid: torch.Tensor,
    maximum: int,
    token: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        return (
            torch.empty((0, features.shape[1]), dtype=features.dtype),
            torch.empty((0,), dtype=residual.dtype),
        )
    count = min(int(valid_indices.numel()), int(maximum))
    offset = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % int(valid_indices.numel())
    positions = (torch.arange(count, device=valid_indices.device, dtype=torch.long) * int(valid_indices.numel()) // count + offset) % int(valid_indices.numel())
    selected = valid_indices[positions]
    flattened = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    return flattened[selected].detach().cpu(), residual.reshape(-1)[selected].detach().cpu()


def _load_consumed_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows_by_sample: dict[str, dict[str, Any]] = {}
    receipts = []
    for relative, expected_sha in CONSUMED_LABEL_RESULTS:
        path = REPO_ROOT / relative
        require(path.is_file() and sha256_file(path) == expected_sha, f"consumed label result drift: {relative}")
        result = json.loads(path.read_text(encoding="utf-8"))
        require(bool(result["passed"]), f"consumed label result not passed: {relative}")
        receipts.append({"path": relative, "sha256": expected_sha})
        for row in result["frames"]:
            sample_id = str(row["sample_id"])
            if sample_id in rows_by_sample:
                require(rows_by_sample[sample_id]["output_sha256"] == row["output_sha256"], "duplicate label payload drift")
                continue
            rows_by_sample[sample_id] = {**row, "role": "CONSUMED_CALIBRATION"}
    forbidden_path = REPO_ROOT / FORBIDDEN_ATTEMPT05_LABEL_RESULT[0]
    require(forbidden_path.is_file(), "Attempt-05 label result receipt missing")
    require(sha256_file(forbidden_path) == FORBIDDEN_ATTEMPT05_LABEL_RESULT[1], "Attempt-05 label result receipt drift")
    forbidden = json.loads(forbidden_path.read_text(encoding="utf-8"))
    forbidden_samples = {str(row["sample_id"]) for row in forbidden["frames"]}
    require(not (forbidden_samples & set(rows_by_sample)), "Attempt-05 held sample leaked into calibration")
    rows = sorted(rows_by_sample.values(), key=lambda row: row["sample_id"])
    require(len(rows) == 63 and len({row["parent_id"] for row in rows}) == 21, "consumed calibration roster drift")
    return rows, receipts


def _parent_split(parents: set[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(
        parents,
        key=lambda parent: hashlib.sha256(f"{SPLIT_TOKEN}:{parent}".encode("utf-8")).hexdigest(),
    )
    require(len(ordered) == 21, "calibration parent count drift")
    return ordered[:17], ordered[17:]


def _quantile_summary(sigma: torch.Tensor, residual: torch.Tensor) -> dict[str, Any]:
    sigma_np = sigma.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    residual_np = residual.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    order = np.argsort(sigma_np, kind="stable")
    groups = np.array_split(order, 4)
    means = [float(residual_np[group].mean()) for group in groups]
    return {
        "observation_count": int(sigma_np.size),
        "sigma_quantile_means": [float(sigma_np[group].mean()) for group in groups],
        "quantile_residual_means": means,
        "nondecreasing": all(a <= b + 1.0e-6 for a, b in zip(means, means[1:])),
    }


def _gaussian_nll(log_sigma: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return log_sigma + 0.5 * residual.square() * torch.exp(-2.0 * log_sigma)


def _evaluate_calibrator(
    model: PixelScaleCalibrator,
    features: torch.Tensor,
    residual: torch.Tensor,
    batch_size: int = 65536,
) -> dict[str, Any]:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            outputs.append(model(features[start : start + batch_size]))
    log_sigma = torch.cat(outputs)
    summary = _quantile_summary(log_sigma.exp(), residual)
    violations = sum(max(0.0, a - b) for a, b in zip(summary["quantile_residual_means"], summary["quantile_residual_means"][1:]))
    return {
        "mean_gaussian_nll": float(_gaussian_nll(log_sigma, residual).mean()),
        "mean_abs_residual": float(residual.mean()),
        "ordering_violation_sum": float(violations),
        **summary,
    }


def _train_one(
    family: str,
    fit_features_cpu: torch.Tensor,
    fit_residual_cpu: torch.Tensor,
    validation_features_cpu: torch.Tensor,
    validation_residual_cpu: torch.Tensor,
    device: torch.device,
    steps: int,
) -> tuple[PixelScaleCalibrator, dict[str, Any]]:
    config = {
        "depth": {"minimum_sigma": 0.01, "maximum_sigma": 3.0, "target_floor": 0.01},
        "boundary": {"minimum_sigma": 0.25, "maximum_sigma": 32.0, "target_floor": 0.25},
    }[family]
    feature_mean = fit_features_cpu.mean(dim=0)
    feature_std = fit_features_cpu.std(dim=0, correction=0).clamp_min(1.0e-4)
    model = PixelScaleCalibrator(feature_mean, feature_std, 32, config["minimum_sigma"], config["maximum_sigma"]).to(device)
    nn.init.constant_(model.net[-1].bias, float(fit_residual_cpu.clamp_min(config["target_floor"]).log().median()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    fit_features = fit_features_cpu.to(device)
    fit_residual = fit_residual_cpu.to(device)
    validation_features = validation_features_cpu.to(device)
    validation_residual = validation_residual_cpu.to(device)
    generator = torch.Generator(device=device).manual_seed(TRAINING_SEED + (0 if family == "depth" else 1))
    best_score = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    validation_trace = []
    batch_size = min(8192, fit_features.shape[0])
    model.train()
    for step in range(1, steps + 1):
        indices = torch.randint(fit_features.shape[0], (batch_size,), generator=generator, device=device)
        features = fit_features[indices]
        residual = fit_residual[indices]
        log_sigma = model(features)
        nll = _gaussian_nll(log_sigma, residual).mean()
        target_log = residual.clamp_min(config["target_floor"]).log()
        regression = F.smooth_l1_loss(log_sigma, target_log)
        permutation = torch.randperm(batch_size, generator=generator, device=device)
        target_difference = target_log - target_log[permutation]
        ranking_mask = target_difference.abs() >= 0.25
        if bool(ranking_mask.any()):
            ranking = F.softplus(
                -torch.sign(target_difference[ranking_mask])
                * (log_sigma[ranking_mask] - log_sigma[permutation][ranking_mask])
            ).mean()
        else:
            ranking = log_sigma.sum() * 0.0
        loss = nll + 0.25 * regression + 0.10 * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 50 == 0 or step == steps:
            validation = _evaluate_calibrator(model, validation_features, validation_residual)
            scale = max(validation["mean_abs_residual"], 1.0e-6)
            score = (
                validation["mean_gaussian_nll"]
                + 4.0 * validation["ordering_violation_sum"] / scale
                + (0.0 if validation["nondecreasing"] else 1.0e6)
            )
            validation_trace.append({"step": step, "selection_score": score, **validation})
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            model.train()
    model.load_state_dict(best_state)
    fit_evaluation = _evaluate_calibrator(model, fit_features, fit_residual)
    validation_evaluation = _evaluate_calibrator(model, validation_features, validation_residual)
    return model, {
        "family": family,
        "hidden_channels": 32,
        "optimizer_steps": steps,
        "best_step": best_step,
        "best_selection_score": best_score,
        "fit": fit_evaluation,
        "internal_validation": validation_evaluation,
        "validation_trace": validation_trace,
    }


def load_calibrator_checkpoint(path: Path, device: torch.device) -> tuple[PixelScaleCalibrator, PixelScaleCalibrator, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    models = []
    for family in ("depth", "boundary"):
        config = payload["models"][family]["config"]
        model = PixelScaleCalibrator(
            payload["models"][family]["feature_mean"],
            payload["models"][family]["feature_std"],
            int(config["hidden_channels"]),
            float(config["minimum_sigma"]),
            float(config["maximum_sigma"]),
        ).to(device)
        model.load_state_dict(payload["models"][family]["state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models[0], models[1], payload["metadata"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART checkpoint drift")
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    rows, label_receipts = _load_consumed_rows()
    parents = {str(row["parent_id"]) for row in rows}
    fit_parents, validation_parents = _parent_split(parents)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()

    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    models = []
    checkpoint_receipts = []
    for seed_row in attempt02["seed_results"]:
        seed = int(seed_row["seed"])
        checkpoint = Path(seed_row["composite_checkpoint"]["path"])
        require(sha256_file(checkpoint) == EXPECTED_COMPOSITES[seed], f"composite drift: {seed}")
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        model.eval()
        models.append(model)
        checkpoint_receipts.append({"seed": seed, "path": str(checkpoint.resolve()), "sha256": EXPECTED_COMPOSITES[seed]})
    require([row["seed"] for row in checkpoint_receipts] == [17, 29, 43], "seed order drift")

    sampled: dict[str, dict[str, list[torch.Tensor]]] = {
        family: {"fit_features": [], "fit_residual": [], "validation_features": [], "validation_residual": []}
        for family in ("depth", "boundary")
    }
    frame_sample_receipts = []
    for index, sample in enumerate(samples):
        prepared = prepare([sample], device)
        members = [cache_model_outputs(model, prepared, device)[0] for model in models]
        base_depth = F.interpolate(
            sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
            sample.native_hw,
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0.01)
        depth_features, boundary_features = calibration_features(members, base_depth)
        target = prepared[0]["target"]
        depth_residual = (members[0]["predicted_log_depth"] - target["depth"].clamp_min(0.01).log()).abs()
        boundary_distance = (-3.0 * members[0]["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
        boundary_residual = (boundary_distance - target["boundary_distance"]).abs()
        role = "fit" if sample.parent_id in fit_parents else "validation"
        depth_x, depth_y = _sample_valid(
            depth_features,
            depth_residual,
            target["depth_valid"],
            args.samples_per_frame,
            f"depth:{sample.sample_id}",
        )
        boundary_x, boundary_y = _sample_valid(
            boundary_features,
            boundary_residual,
            target["evidence_valid"],
            args.samples_per_frame,
            f"boundary:{sample.sample_id}",
        )
        sampled["depth"][f"{role}_features"].append(depth_x)
        sampled["depth"][f"{role}_residual"].append(depth_y)
        sampled["boundary"][f"{role}_features"].append(boundary_x)
        sampled["boundary"][f"{role}_residual"].append(boundary_y)
        frame_sample_receipts.append(
            {
                "sample_id": sample.sample_id,
                "parent_id": sample.parent_id,
                "calibration_role": role.upper(),
                "depth_samples": int(depth_y.numel()),
                "boundary_samples": int(boundary_y.numel()),
            }
        )
        del prepared, members, base_depth, depth_features, boundary_features
        torch.cuda.empty_cache()
        if (index + 1) % 6 == 0:
            print(json.dumps({"feature_sampling_frames": index + 1, "total_frames": len(samples)}), flush=True)
    del models, samples
    torch.cuda.empty_cache()

    tensors: dict[str, dict[str, torch.Tensor]] = {}
    family_parent_coverage: dict[str, dict[str, list[str]]] = {}
    for family in ("depth", "boundary"):
        tensors[family] = {
            key: torch.cat(value, dim=0)
            for key, value in sampled[family].items()
        }
        require(
            tensors[family]["fit_residual"].numel() > 0
            and tensors[family]["validation_residual"].numel() > 0,
            f"{family} calibration role denominator missing",
        )
        covered_parents = {
            row["parent_id"]
            for row in frame_sample_receipts
            if int(row[f"{family}_samples"]) > 0
        }
        minimum_parent_count = len(parents) if family == "depth" else len(parents) - 1
        require(len(covered_parents) >= minimum_parent_count, f"{family} calibration parent denominator insufficient")
        family_parent_coverage[family] = {
            "included": sorted(covered_parents),
            "excluded_unknown": sorted(parents - covered_parents),
        }
    depth_model, depth_training = _train_one(
        "depth",
        tensors["depth"]["fit_features"],
        tensors["depth"]["fit_residual"],
        tensors["depth"]["validation_features"],
        tensors["depth"]["validation_residual"],
        device,
        args.optimizer_steps,
    )
    boundary_model, boundary_training = _train_one(
        "boundary",
        tensors["boundary"]["fit_features"],
        tensors["boundary"]["fit_residual"],
        tensors["boundary"]["validation_features"],
        tensors["boundary"]["validation_residual"],
        device,
        args.optimizer_steps,
    )
    checkpoint_path = args.output_dir / "uncertainty_calibrators.pt"
    checkpoint_payload = {
        "schema": "blindassist_ag_r2_f1_attempt05_uncertainty_calibrators_v1",
        "models": {
            "depth": {
                "config": {"hidden_channels": 32, "minimum_sigma": 0.01, "maximum_sigma": 3.0},
                "feature_names": list(DEPTH_FEATURE_NAMES),
                "feature_mean": depth_model.feature_mean.detach().cpu(),
                "feature_std": depth_model.feature_std.detach().cpu(),
                "state_dict": {key: value.detach().cpu() for key, value in depth_model.state_dict().items()},
            },
            "boundary": {
                "config": {"hidden_channels": 32, "minimum_sigma": 0.25, "maximum_sigma": 32.0},
                "feature_names": list(BOUNDARY_FEATURE_NAMES),
                "feature_mean": boundary_model.feature_mean.detach().cpu(),
                "feature_std": boundary_model.feature_std.detach().cpu(),
                "state_dict": {key: value.detach().cpu() for key, value in boundary_model.state_dict().items()},
            },
        },
        "metadata": {
            "training_seed": TRAINING_SEED,
            "split_token": SPLIT_TOKEN,
            "fit_parents": fit_parents,
            "internal_validation_parents": validation_parents,
            "consumed_label_receipts": label_receipts,
            "attempt05_held_label_result_excluded": {
                "path": FORBIDDEN_ATTEMPT05_LABEL_RESULT[0],
                "sha256": FORBIDDEN_ATTEMPT05_LABEL_RESULT[1],
            },
            "checkpoint_receipts": checkpoint_receipts,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    reload_depth, reload_boundary, reload_metadata = load_calibrator_checkpoint(checkpoint_path, device)
    require(reload_metadata["fit_parents"] == fit_parents, "calibrator checkpoint metadata drift")
    require(
        all(torch.equal(a, b) for a, b in zip(depth_model.state_dict().values(), reload_depth.state_dict().values()))
        and all(torch.equal(a, b) for a, b in zip(boundary_model.state_dict().values(), reload_boundary.state_dict().values())),
        "calibrator checkpoint roundtrip drift",
    )
    internal_pass = bool(
        depth_training["internal_validation"]["nondecreasing"]
        and boundary_training["internal_validation"]["nondecreasing"]
    )
    result = {
        "schema": "blindassist_ag_r2_f1_attempt05_uncertainty_calibration_result_v1",
        "status": (
            "ATTEMPT05_UNCERTAINTY_CALIBRATION_INTERNAL_PASS_FRESH_EXECUTION_LOCK_REQUIRED"
            if internal_pass
            else "ATTEMPT05_UNCERTAINTY_CALIBRATION_INTERNAL_FAIL_NO_FRESH_METRICS"
        ),
        "passed": internal_pass,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch": torch.__version__,
        "consumed_label_receipts": label_receipts,
        "attempt05_held_label_result_excluded": {
            "path": FORBIDDEN_ATTEMPT05_LABEL_RESULT[0],
            "sha256": FORBIDDEN_ATTEMPT05_LABEL_RESULT[1],
        },
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "frame_count": len(frame_sample_receipts),
        "parent_count": len(parents),
        "samples_per_frame_maximum": args.samples_per_frame,
        "family_parent_coverage": family_parent_coverage,
        "frame_sample_receipts": frame_sample_receipts,
        "feature_receipt": feature_receipt,
        "checkpoint_receipts": checkpoint_receipts,
        "calibrator_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
        },
        "depth": depth_training,
        "boundary": boundary_training,
        "decision": {
            "point_factor_parameters_changed": False,
            "support_uncertainty_changed": False,
            "fresh_attempt05_selection_or_canary_metrics_opened": False,
            "next_action_if_pass": "Freeze the Attempt-05 execution lock, reproduce primary factors, and open fresh CHECKPOINT_SELECTION before TRAIN_CANARY.",
        },
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt02-result", type=Path, default=DEFAULT_ATTEMPT02_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-frame", type=int, default=4096)
    parser.add_argument("--optimizer-steps", type=int, default=1200)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "checkpoint": result["calibrator_checkpoint"],
                "depth_internal_validation": result["depth"]["internal_validation"],
                "boundary_internal_validation": result["boundary"]["internal_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
