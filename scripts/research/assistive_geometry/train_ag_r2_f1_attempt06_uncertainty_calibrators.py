#!/usr/bin/env python3
"""Recalibrate boundary and scalar support uncertainty on consumed evidence."""

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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt05_uncertainty_calibrators import (  # noqa: E402
    BOUNDARY_FEATURE_NAMES,
    PixelScaleCalibrator,
    _evaluate_calibrator,
    _gradient_magnitude,
    _load_consumed_rows,
    _sample_valid,
    _train_one,
    calibration_features,
    load_calibrator_checkpoint,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_ATTEMPT02_RESULT,
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_COMPOSITES,
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


ATTEMPT05_LABEL_RESULT = (
    "artifacts.local/experiments/ag-r2-f1-attempt05-fresh-ag-held-labels-r0/result.json",
    "4DBF0E85F45357C613221DF9F2C5A5E3B0971C314EB29D1967C02E0D6FAEB7CC",
)
ATTEMPT06_SELECTION_LABEL_RESULT = (
    "artifacts.local/experiments/ag-r2-f1-attempt06-fresh-selection-labels-r0/result.json",
    "F67A9A000A4A82C180B9E875DEE976C80E00584AB8A6CEBCC603DFAAEE1E90A5",
)
ATTEMPT05_CALIBRATOR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-uncertainty-calibration-r3/uncertainty_calibrators.pt"
EXPECTED_ATTEMPT05_CALIBRATOR_SHA256 = "37116A6E115915C082368FE9EC30AB41D3C8B2EB24F24A47D5D158299C6B107E"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt06-uncertainty-calibration-r0"
SPLIT_TOKEN = "AG_R2_F1_ATTEMPT06_UNCERTAINTY_RECALIBRATION_2026-08-11"
TRAINING_SEED = 1806

SUPPORT_FEATURE_NAMES = (
    "attempt04_log_support_sigma",
    "support_geometry_coverage",
    "support_probability_mean",
    "support_probability_std",
    "support_probability_q25",
    "support_probability_q75",
    "support_probability_gradient_mean",
    "depth_valid_probability_mean",
    "evidence_valid_probability_mean",
    "predicted_log_depth_std",
    "base_log_disagreement_mean",
    "base_log_disagreement_std",
    "three_seed_epistemic_mean",
    "three_seed_epistemic_std",
    "log_camera_height",
    "portrait_orientation",
)


def _load_calibration_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows, base_receipts = _load_consumed_rows()
    attempt05_path = REPO_ROOT / ATTEMPT05_LABEL_RESULT[0]
    require(sha256_file(attempt05_path) == ATTEMPT05_LABEL_RESULT[1], "Attempt-05 label result drift")
    attempt05 = json.loads(attempt05_path.read_text(encoding="utf-8"))
    selection_rows = [
        {**row, "role": "CONSUMED_CALIBRATION"}
        for row in attempt05["frames"]
        if row["role"] == "CHECKPOINT_SELECTION"
    ]
    canary_rows = [row for row in attempt05["frames"] if row["role"] == "TRAIN_CANARY"]
    require(len(selection_rows) == len(canary_rows) == 6, "Attempt-05 role drift")
    attempt06_path = REPO_ROOT / ATTEMPT06_SELECTION_LABEL_RESULT[0]
    require(sha256_file(attempt06_path) == ATTEMPT06_SELECTION_LABEL_RESULT[1], "Attempt-06 selection labels drift")
    attempt06 = json.loads(attempt06_path.read_text(encoding="utf-8"))
    forbidden_samples = {
        str(row["sample_id"])
        for row in canary_rows + list(attempt06["frames"])
    }
    rows = sorted(base_rows + selection_rows, key=lambda row: row["sample_id"])
    require(len(rows) == 69 and len({row["parent_id"] for row in rows}) == 23, "Attempt-06 calibration roster drift")
    require(not ({row["sample_id"] for row in rows} & forbidden_samples), "Attempt-06 held sample leaked")
    return rows, {
        "base_consumed_label_receipts": base_receipts,
        "attempt05_consumed_selection": {
            "path": ATTEMPT05_LABEL_RESULT[0],
            "sha256": ATTEMPT05_LABEL_RESULT[1],
            "parents": sorted({row["parent_id"] for row in selection_rows}),
            "frames": len(selection_rows),
        },
        "preserved_canary_excluded": {
            "path": ATTEMPT05_LABEL_RESULT[0],
            "sha256": ATTEMPT05_LABEL_RESULT[1],
            "parents": sorted({row["parent_id"] for row in canary_rows}),
            "sample_ids": sorted(row["sample_id"] for row in canary_rows),
        },
        "attempt06_selection_excluded": {
            "path": ATTEMPT06_SELECTION_LABEL_RESULT[0],
            "sha256": ATTEMPT06_SELECTION_LABEL_RESULT[1],
            "parents": sorted({row["parent_id"] for row in attempt06["frames"]}),
        },
    }


def _parent_split(parents: set[str]) -> tuple[list[str], list[str]]:
    forced_fit = {"rgbd_dataset_freiburg1_xyz", "rgbd_dataset_freiburg1_room"}
    require(forced_fit <= parents, "Attempt-05 selection parents missing")
    remaining = sorted(
        parents - forced_fit,
        key=lambda parent: hashlib.sha256(f"{SPLIT_TOKEN}:{parent}".encode("utf-8")).hexdigest(),
    )
    validation = remaining[-4:]
    fit = sorted(parents - set(validation))
    require(len(fit) == 19 and len(validation) == 4, "Attempt-06 calibration split drift")
    return fit, validation


def support_features(
    members: list[dict[str, torch.Tensor]],
    base_depth: torch.Tensor,
    factors: dict[str, torch.Tensor],
    geometry_receipt: dict[str, Any],
    orientation: str,
) -> torch.Tensor:
    seed17 = members[0]
    seed_logs = torch.stack([row["predicted_log_depth"] for row in members])
    epistemic = seed_logs.std(dim=0, correction=0)
    base_log = base_depth.clamp_min(0.01).log()
    disagreement = (seed17["predicted_log_depth"] - base_log).abs()
    support = seed17["support_probability"]
    support_flat = support.reshape(-1)
    coverage = float(geometry_receipt["geometry_pixel_count"]) / float(support_flat.numel())
    values = torch.stack(
        (
            factors["support_residual_sigma_m"][0].clamp_min(0.03).log(),
            torch.tensor(coverage, device=support.device),
            support.mean(),
            support.std(correction=0),
            torch.quantile(support_flat, 0.25),
            torch.quantile(support_flat, 0.75),
            _gradient_magnitude(support).mean(),
            seed17["depth_valid_probability"].mean(),
            seed17["evidence_valid_probability"].mean(),
            seed17["predicted_log_depth"].std(correction=0),
            disagreement.mean(),
            disagreement.std(correction=0),
            epistemic.mean(),
            epistemic.std(correction=0),
            factors["camera_height_m"][0].clamp_min(0.03).log(),
            torch.tensor(float(orientation != "LANDSCAPE_IDENTITY"), device=support.device),
        )
    )[None]
    require(values.shape[1] == len(SUPPORT_FEATURE_NAMES), "support calibration feature drift")
    return values


def _gaussian_nll(log_sigma: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return log_sigma + 0.5 * residual.square() * torch.exp(-2.0 * log_sigma)


def _train_support(
    fit_features_cpu: torch.Tensor,
    fit_residual_cpu: torch.Tensor,
    validation_features_cpu: torch.Tensor,
    validation_residual_cpu: torch.Tensor,
    validation_source_sigma_cpu: torch.Tensor,
    device: torch.device,
    steps: int,
) -> tuple[PixelScaleCalibrator, dict[str, Any]]:
    mean = fit_features_cpu.mean(dim=0)
    std = fit_features_cpu.std(dim=0, correction=0).clamp_min(1.0e-4)
    model = PixelScaleCalibrator(mean, std, 16, 0.03, 2.5).to(device)
    torch.nn.init.constant_(model.net[-1].bias, float(fit_residual_cpu.clamp_min(0.03).log().median()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    fit_features = fit_features_cpu.to(device)
    fit_residual = fit_residual_cpu.to(device)
    validation_features = validation_features_cpu.to(device)
    validation_residual = validation_residual_cpu.to(device)
    source_sigma = validation_source_sigma_cpu.to(device).clamp_min(0.03)
    baseline_nll = float(_gaussian_nll(source_sigma.log(), validation_residual).mean())
    generator = torch.Generator(device=device).manual_seed(TRAINING_SEED + 2)
    best_score = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    trace = []
    batch_size = min(64, fit_features.shape[0])
    for step in range(1, steps + 1):
        indices = torch.randint(fit_features.shape[0], (batch_size,), generator=generator, device=device)
        features = fit_features[indices]
        residual = fit_residual[indices]
        log_sigma = model(features)
        target_log = residual.clamp_min(0.03).log()
        nll = _gaussian_nll(log_sigma, residual).mean()
        regression = F.smooth_l1_loss(log_sigma, target_log)
        permutation = torch.randperm(batch_size, generator=generator, device=device)
        difference = target_log - target_log[permutation]
        ranking_mask = difference.abs() >= 0.15
        ranking = (
            F.softplus(
                -torch.sign(difference[ranking_mask])
                * (log_sigma[ranking_mask] - log_sigma[permutation][ranking_mask])
            ).mean()
            if bool(ranking_mask.any())
            else log_sigma.sum() * 0.0
        )
        loss = nll + 0.35 * regression + 0.20 * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 25 == 0 or step == steps:
            evaluation = _evaluate_calibrator(model, validation_features, validation_residual)
            proper_gain = baseline_nll - evaluation["mean_gaussian_nll"]
            eligible = evaluation["nondecreasing"] and proper_gain > 0.0
            score = evaluation["mean_gaussian_nll"] + (0.0 if eligible else 1.0e6)
            trace.append({"step": step, "baseline_nll": baseline_nll, "proper_score_gain": proper_gain, "selection_score": score, **evaluation})
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    fit_evaluation = _evaluate_calibrator(model, fit_features, fit_residual)
    validation_evaluation = _evaluate_calibrator(model, validation_features, validation_residual)
    validation_evaluation["baseline_mean_gaussian_nll"] = baseline_nll
    validation_evaluation["proper_score_gain"] = baseline_nll - validation_evaluation["mean_gaussian_nll"]
    return model, {
        "family": "support",
        "hidden_channels": 16,
        "optimizer_steps": steps,
        "best_step": best_step,
        "best_selection_score": best_score,
        "fit": fit_evaluation,
        "internal_validation": validation_evaluation,
        "validation_trace": trace,
    }


def load_attempt06_calibrators(
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
    require(sha256_file(args.attempt05_calibrator) == EXPECTED_ATTEMPT05_CALIBRATOR_SHA256, "Attempt-05 calibrator drift")
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    rows, data_receipt = _load_calibration_rows()
    parents = {str(row["parent_id"]) for row in rows}
    fit_parents, validation_parents = _parent_split(parents)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()

    samples, feature_receipt = extract_features(
        rows, args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device
    )
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    factor_models = []
    checkpoint_receipts = []
    for seed_row in attempt02["seed_results"]:
        seed = int(seed_row["seed"])
        checkpoint = Path(seed_row["composite_checkpoint"]["path"])
        require(sha256_file(checkpoint) == EXPECTED_COMPOSITES[seed], f"composite drift: {seed}")
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        model.eval()
        factor_models.append(model)
        checkpoint_receipts.append({"seed": seed, "path": str(checkpoint.resolve()), "sha256": EXPECTED_COMPOSITES[seed]})

    by_parent: dict[str, list[Any]] = {}
    for sample in samples:
        by_parent.setdefault(sample.parent_id, []).append(sample)
    sampled = {"fit_features": [], "fit_residual": [], "validation_features": [], "validation_residual": []}
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
        require(len(parent_samples) == 3, f"parent frame count drift: {parent}")
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
            _, boundary_features = calibration_features(members, base_depth)
            target = prepared_row["target"]
            boundary_distance = (-3.0 * members[0]["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
            boundary_residual = (boundary_distance - target["boundary_distance"]).abs()
            boundary_x, boundary_y = _sample_valid(
                boundary_features,
                boundary_residual,
                target["evidence_valid"],
                args.samples_per_frame,
                f"attempt06-boundary:{sample.sample_id}",
            )
            sampled[f"{role}_features"].append(boundary_x)
            sampled[f"{role}_residual"].append(boundary_y)
            factors, geometry_receipt = apply_geometry(
                members[0], sample, GEOMETRY_CONFIG, baseline, device, context
            )
            support_valid = target["support_valid"]
            support_sample_count = int(support_valid.sum())
            if bool(target["plane_valid"]) and support_sample_count > 0:
                residual_rms = target["support_residual"][support_valid].square().mean().sqrt()[None]
                feature = support_features(members, base_depth, factors, geometry_receipt, sample.orientation)
                support_rows[f"{role}_features"].append(feature.detach().cpu())
                support_rows[f"{role}_residual"].append(residual_rms.detach().cpu())
                support_rows[f"{role}_source_sigma"].append(factors["support_residual_sigma_m"].detach().cpu())
            frame_receipts.append(
                {
                    "sample_id": sample.sample_id,
                    "parent_id": parent,
                    "calibration_role": role.upper(),
                    "boundary_samples": int(boundary_y.numel()),
                    "support_valid_pixels": support_sample_count,
                }
            )
            processed += 1
        del prepared, caches
        torch.cuda.empty_cache()
        if processed % 6 == 0:
            print(json.dumps({"feature_sampling_frames": processed, "total_frames": len(samples)}), flush=True)
    del factor_models, samples
    torch.cuda.empty_cache()

    boundary_tensors = {key: torch.cat(value, dim=0) for key, value in sampled.items()}
    support_tensors = {key: torch.cat(value, dim=0) for key, value in support_rows.items()}
    boundary_model, boundary_training = _train_one(
        "boundary",
        boundary_tensors["fit_features"],
        boundary_tensors["fit_residual"],
        boundary_tensors["validation_features"],
        boundary_tensors["validation_residual"],
        device,
        args.optimizer_steps,
    )
    support_model, support_training = _train_support(
        support_tensors["fit_features"],
        support_tensors["fit_residual"],
        support_tensors["validation_features"],
        support_tensors["validation_residual"],
        support_tensors["validation_source_sigma"],
        device,
        args.support_optimizer_steps,
    )

    previous_payload = torch.load(args.attempt05_calibrator, map_location="cpu", weights_only=True)
    checkpoint_path = args.output_dir / "uncertainty_calibrators.pt"
    checkpoint_payload = {
        "schema": "blindassist_ag_r2_f1_attempt06_uncertainty_calibrators_v1",
        "models": {
            "depth": previous_payload["models"]["depth"],
            "boundary": {
                "config": {"hidden_channels": 32, "minimum_sigma": 0.25, "maximum_sigma": 32.0},
                "feature_names": list(BOUNDARY_FEATURE_NAMES),
                "feature_mean": boundary_model.feature_mean.detach().cpu(),
                "feature_std": boundary_model.feature_std.detach().cpu(),
                "state_dict": {key: value.detach().cpu() for key, value in boundary_model.state_dict().items()},
            },
            "support": {
                "config": {"hidden_channels": 16, "minimum_sigma": 0.03, "maximum_sigma": 2.5},
                "feature_names": list(SUPPORT_FEATURE_NAMES),
                "feature_mean": support_model.feature_mean.detach().cpu(),
                "feature_std": support_model.feature_std.detach().cpu(),
                "state_dict": {key: value.detach().cpu() for key, value in support_model.state_dict().items()},
            },
        },
        "metadata": {
            "training_seed": TRAINING_SEED,
            "split_token": SPLIT_TOKEN,
            "fit_parents": fit_parents,
            "internal_validation_parents": validation_parents,
            "data_receipt": data_receipt,
            "attempt05_depth_calibrator_preserved_sha256": EXPECTED_ATTEMPT05_CALIBRATOR_SHA256,
            "checkpoint_receipts": checkpoint_receipts,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    _, reload_boundary, reload_support, reload_metadata = load_attempt06_calibrators(checkpoint_path, device)
    require(reload_metadata["fit_parents"] == fit_parents, "Attempt-06 checkpoint metadata drift")
    require(
        all(torch.equal(a, b) for a, b in zip(boundary_model.state_dict().values(), reload_boundary.state_dict().values()))
        and all(torch.equal(a, b) for a, b in zip(support_model.state_dict().values(), reload_support.state_dict().values())),
        "Attempt-06 checkpoint roundtrip drift",
    )
    passed = bool(
        boundary_training["internal_validation"]["nondecreasing"]
        and support_training["internal_validation"]["nondecreasing"]
        and support_training["internal_validation"]["proper_score_gain"] > 0.0
    )
    result = {
        "schema": "blindassist_ag_r2_f1_attempt06_uncertainty_calibration_result_v1",
        "status": (
            "ATTEMPT06_UNCERTAINTY_RECALIBRATION_INTERNAL_PASS_FRESH_EXECUTION_LOCK_REQUIRED"
            if passed
            else "ATTEMPT06_UNCERTAINTY_RECALIBRATION_INTERNAL_FAIL_NO_FRESH_METRICS"
        ),
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch": torch.__version__,
        "data_receipt": data_receipt,
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "frame_count": len(frame_receipts),
        "parent_count": len(parents),
        "frame_sample_receipts": frame_receipts,
        "feature_receipt": feature_receipt,
        "checkpoint_receipts": checkpoint_receipts,
        "calibrator_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
        },
        "depth": {
            "preserved_attempt05_checkpoint_sha256": EXPECTED_ATTEMPT05_CALIBRATOR_SHA256,
            "attempt05_selection_passed": True,
        },
        "boundary": boundary_training,
        "support": support_training,
        "decision": {
            "point_factor_parameters_changed": False,
            "depth_calibrator_changed": False,
            "attempt06_selection_metrics_opened": False,
            "preserved_canary_metrics_opened": False,
            "next_action_if_pass": "Freeze Attempt-06 execution; pass new selection before opening the preserved canary.",
        },
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt05-calibrator", type=Path, default=ATTEMPT05_CALIBRATOR)
    parser.add_argument("--attempt02-result", type=Path, default=DEFAULT_ATTEMPT02_RESULT)
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
    print(json.dumps({"status": result["status"], "passed": result["passed"], "checkpoint": result["calibrator_checkpoint"], "boundary_internal_validation": result["boundary"]["internal_validation"], "support_internal_validation": result["support"]["internal_validation"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
