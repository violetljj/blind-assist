#!/usr/bin/env python3
"""Train an explicit DepthART-skip depth shape/scale factor head."""

from __future__ import annotations

import argparse
import json
import math
import random
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

from train_ag_r2_f1_attempt07_point_factor_expansion import (  # noqa: E402
    load_rows,
    parent_split,
)
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    CHARBONNIER_EPSILON,
    bootstrap_lower,
    evaluate,
    extract_features,
    gaussian_nll_tensor,
    huber_tensor,
    masked_mean,
    move_targets,
    save_checkpoint,
)
from train_ag_r2_f1_factor_learnability_attempt02 import (  # noqa: E402
    DEPTHART_PYRAMID_CHANNELS,
    GlobalBlock,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_ATTEMPT02_RESULT,
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    FactorSplitHead,
    require,
    sha256_file,
)


POINT_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-point-factor-expansion-r0/result.json"
EXPECTED_POINT_RESULT_SHA256 = "580A94AD71B9C86A706D8FB233BF87AAD9376B503C8E7C35DF8F1102A34AE946"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt08-depthart-residual-head-r0"
DEPTH_METRICS = ("depth_shape_abs_log_error", "depth_scale_abs_log_error", "depth_nll")
MAXIMUM_LOCAL_LOG_RESIDUAL = 0.75
MAXIMUM_GLOBAL_LOG_RESIDUAL = 0.75


class DepthArtResidualFactorHead(FactorSplitHead):
    """Separate zero-mean local shape from a pooled global metric correction."""

    def __init__(self, baseline: dict[str, Any]) -> None:
        super().__init__(baseline)
        inputs = DEPTHART_PYRAMID_CHANNELS + 1
        self.depth_scale_adapter = GlobalBlock(inputs, 64, 1)

    def reset_depth_prediction(self) -> None:
        final = self.depth.block.net[-1]
        require(isinstance(final, nn.Conv2d) and final.out_channels == 3, "depth final layer drift")
        with torch.no_grad():
            final.weight[0].zero_()
            final.bias[0].zero_()
        for module in self.depth_scale_adapter.modules():
            if isinstance(module, nn.Linear) and module.out_features == 1:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feature: torch.Tensor, base_depth_m: torch.Tensor) -> dict[str, torch.Tensor]:
        base_log = base_depth_m.clamp(0.05, 20.0).log()
        value = torch.cat([feature, base_log], dim=1)
        raw = self.depth.block(value)
        local = torch.tanh(raw[:, 0:1])
        local = local - local.mean(dim=(2, 3), keepdim=True)
        global_residual = torch.tanh(self.depth_scale_adapter(value)[:, 0])[:, None, None, None]
        log_depth = (
            base_log
            + MAXIMUM_LOCAL_LOG_RESIDUAL * local
            + MAXIMUM_GLOBAL_LOG_RESIDUAL * global_residual
        )
        log_sigma = (self.depth.baseline_log_sigma + raw[:, 1:2]).clamp(math.log(0.005), math.log(5.0))
        depth_valid = torch.sigmoid(self.depth.baseline_valid_logit + raw[:, 2:3])
        activity = local.abs().mean(dim=(1, 2, 3))
        boundary_probability, boundary_sigma = self.boundary(value)
        return {
            "predicted_log_depth": log_depth,
            "depth_log_sigma": log_sigma,
            "depth_valid_probability": depth_valid,
            "support_probability": self.support_probability(value),
            "obstacle_probability": self.obstacle(value),
            "boundary_probability": boundary_probability,
            "boundary_sigma_px": boundary_sigma,
            "evidence_valid_probability": self.evidence_validity(value),
            "support_plane_normal_camera_xyz": self.support_normal(value),
            "camera_height_m": self.camera_height(value),
            "support_residual_sigma_m": self.support_uncertainty(value),
            "support_valid_probability": self.support_validity(value),
            "depth_gate": activity,
        }


def freeze_non_depth(model: DepthArtResidualFactorHead) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.depth.requires_grad_(True)
    model.depth_scale_adapter.requires_grad_(True)


def training_forward(
    model: DepthArtResidualFactorHead,
    sample: Any,
    device: torch.device,
    horizontal_flip: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    feature = sample.feature[None].to(device=device, dtype=torch.float32)
    base = sample.base_depth_feature[None].to(device=device, dtype=torch.float32)
    targets = move_targets(sample.targets, device)
    if horizontal_flip:
        feature = torch.flip(feature, dims=(-1,))
        base = torch.flip(base, dims=(-1,))
        targets = {
            key: torch.flip(value, dims=(-1,)) if value.ndim >= 2 and value.shape[-1] == feature.shape[-1] else value
            for key, value in targets.items()
        }
    return model(feature, base), targets


def depth_objective(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    normalization: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = targets["metric_valid"]
    target_log = targets["metric_depth_m"].clamp_min(0.01).log()
    predicted_log = outputs["predicted_log_depth"]
    target_scale = masked_mean(target_log, valid)
    predicted_scale = masked_mean(predicted_log, valid)
    target_shape = target_log - target_scale
    predicted_shape = predicted_log - predicted_scale
    residual = predicted_log - target_log
    losses = {
        "depth_shape_log_charbonnier": masked_mean(
            torch.sqrt((predicted_shape - target_shape).square() + CHARBONNIER_EPSILON**2) - CHARBONNIER_EPSILON,
            valid,
        ),
        "metric_scale_log_huber": huber_tensor(predicted_scale - target_scale),
        "depth_heteroscedastic_nll": masked_mean(
            gaussian_nll_tensor(residual, outputs["depth_log_sigma"].exp()), valid
        ),
        "depth_validity_brier": (outputs["depth_valid_probability"] - valid.float()).square().mean(),
    }
    objective = (
        2.0 * losses["depth_shape_log_charbonnier"] / float(normalization["depth_shape_log_charbonnier"])
        + 5.0 * losses["metric_scale_log_huber"] / float(normalization["metric_scale_log_huber"])
        + 1.0 * losses["depth_heteroscedastic_nll"] / float(normalization["depth_heteroscedastic_nll"])
        + 0.25 * losses["depth_validity_brier"] / float(normalization["depth_validity_brier"])
    ) / 8.25
    return objective, losses


def depth_evidence(
    evaluation: dict[str, Any],
    baseline: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    metrics = {}
    normalized_regrets = []
    for metric in DEPTH_METRICS:
        improvements = [
            float(baseline["parent_metrics"][parent][metric])
            - float(evaluation["parent_metrics"][parent][metric])
            for parent in sorted(baseline["parent_metrics"])
        ]
        favorable = float(np.mean(np.asarray(improvements) > 0.0))
        lower = bootstrap_lower(improvements, seed + sum(ord(char) for char in metric))
        overall_base = float(baseline["overall_metrics"][metric])
        overall_model = float(evaluation["overall_metrics"][metric])
        normalized_regret = (overall_model - overall_base) / max(abs(overall_base), 1.0e-8)
        normalized_regrets.append(normalized_regret)
        metrics[metric] = {
            "parent_improvements": improvements,
            "favorable_parent_fraction": favorable,
            "bootstrap_95_lower": lower,
            "overall_improvement": overall_base - overall_model,
            "normalized_regret": normalized_regret,
            "passed": lower > 0.0 and favorable >= 0.75,
        }
    eligible = all(row["passed"] for row in metrics.values())
    score = max(normalized_regrets) + 0.25 * float(np.mean(normalized_regrets)) + (0.0 if eligible else 1.0e6)
    return {"metrics": metrics, "eligible": eligible, "selection_score": score}


def train_seed(
    seed: int,
    fit: list[Any],
    validation: list[Any],
    baseline_parameters: dict[str, Any],
    normalization: dict[str, float],
    validation_baseline: dict[str, Any],
    source_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    random.seed(seed + 8000)
    np.random.seed(seed + 8000)
    torch.manual_seed(seed + 8000)
    torch.cuda.manual_seed_all(seed + 8000)
    model = DepthArtResidualFactorHead(baseline_parameters).to(device)
    source_state = torch.load(source_checkpoint, map_location=device, weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(source_state, strict=False)
    require(set(missing) == {"depth_scale_adapter.net.0.weight", "depth_scale_adapter.net.0.bias", "depth_scale_adapter.net.2.weight", "depth_scale_adapter.net.2.bias"}, "residual adapter missing-key drift")
    require(not unexpected, "residual adapter unexpected source keys")
    model.reset_depth_prediction()
    freeze_non_depth(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=4.0e-4,
        weight_decay=1.0e-4,
    )
    rng = random.Random(seed + 9000)
    checkpoints = []
    loss_trace = []

    def capture(step: int) -> None:
        model.eval()
        evaluation = evaluate(model, validation, baseline_parameters, device)
        receipt = save_checkpoint(output_dir / f"seed-{seed}-step-{step}.pt", model, seed, step)
        checkpoints.append(
            {
                "step": step,
                "checkpoint": receipt,
                "evaluation": evaluation,
                "evidence": depth_evidence(evaluation, validation_baseline, seed + step),
            }
        )

    capture(0)
    for step in range(1, steps + 1):
        model.train()
        sample = fit[rng.randrange(len(fit))]
        outputs, targets = training_forward(model, sample, device, rng.random() < 0.5)
        objective, losses = depth_objective(outputs, targets, normalization)
        require(bool(torch.isfinite(objective)), "residual depth objective non-finite")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 5.0)
        progress = step / steps
        optimizer.param_groups[0]["lr"] = 4.0e-4 * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        optimizer.step()
        if step == 1 or step % 100 == 0:
            loss_trace.append(
                {
                    "step": step,
                    "objective": float(objective.detach()),
                    "components": {key: float(value.detach()) for key, value in losses.items()},
                }
            )
        if step % 200 == 0 or step == steps:
            capture(step)
    selected = min(
        checkpoints,
        key=lambda row: (row["evidence"]["selection_score"], row["step"], row["checkpoint"]["sha256"]),
    )
    return {
        "seed": seed,
        "source_checkpoint": {"path": str(source_checkpoint.resolve()), "sha256": sha256_file(source_checkpoint)},
        "checkpoints": checkpoints,
        "selected_step": selected["step"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_evaluation": selected["evaluation"],
        "selected_evidence": selected["evidence"],
        "loss_trace": loss_trace,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.point_result) == EXPECTED_POINT_RESULT_SHA256, "point result drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    rows, data_receipt = load_rows()
    parents = {str(row["parent_id"]) for row in rows}
    fit_parents, validation_parents = parent_split(parents)
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    fit = [sample for sample in samples if sample.parent_id in fit_parents]
    validation = [sample for sample in samples if sample.parent_id in validation_parents]
    require(len(fit) == 60 and len(validation) == 15, "residual depth split drift")
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline_parameters = baseline_result["baseline_parameters"]
    normalization = baseline_result["optimizer_normalization"]
    validation_baseline = evaluate(None, validation, baseline_parameters, device)
    point = json.loads(args.point_result.read_text(encoding="utf-8"))
    seed_results = []
    for seed_row in point["seed_results"]:
        seed = int(seed_row["seed"])
        source_checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        require(sha256_file(source_checkpoint) == seed_row["selected_checkpoint"]["sha256"], "source point checkpoint drift")
        seed_result = train_seed(
            seed,
            fit,
            validation,
            baseline_parameters,
            normalization,
            validation_baseline,
            source_checkpoint,
            args.output_dir,
            device,
            args.optimizer_steps,
        )
        seed_results.append(seed_result)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "selected_step": seed_result["selected_step"],
                    "eligible": seed_result["selected_evidence"]["eligible"],
                    "metrics": seed_result["selected_evidence"]["metrics"],
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
    eligible = [row for row in seed_results if row["selected_evidence"]["eligible"]]
    canonical = min(
        eligible or seed_results,
        key=lambda row: (
            row["selected_evidence"]["selection_score"],
            row["seed"],
            row["selected_checkpoint"]["sha256"],
        ),
    )
    passed = bool(eligible)
    result = {
        "schema": "blindassist_ag_r2_f1_attempt08_depthart_residual_head_result_v1",
        "status": "ATTEMPT08_RESIDUAL_DEPTH_INTERNAL_PASS_FINAL_UNCERTAINTY_REQUIRED" if passed else "ATTEMPT08_RESIDUAL_DEPTH_INTERNAL_FAIL_NO_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "preserved_canary_metrics_opened": False,
        "data_receipt": data_receipt,
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "feature_receipt": feature_receipt,
        "architecture": {
            "base": "DepthART metric log-depth skip",
            "local_shape": "zero-mean bounded spatial residual",
            "global_scale": "bounded pooled residual",
            "maximum_local_log_residual": MAXIMUM_LOCAL_LOG_RESIDUAL,
            "maximum_global_log_residual": MAXIMUM_GLOBAL_LOG_RESIDUAL,
        },
        "seed_results": seed_results,
        "canonical_seed": canonical["seed"],
        "canonical_checkpoint": canonical["selected_checkpoint"],
        "decision": {
            "non_depth_factor_components_changed": False,
            "preserved_canary_metrics_opened": False,
            "next_action_if_pass": "Re-evaluate geometry and uncertainty with the residual depth ensemble, freeze final canary identity, then execute once.",
        },
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
    parser.add_argument("--optimizer-steps", type=int, default=1600)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "canonical_seed": result["canonical_seed"],
                "canonical_checkpoint": result["canonical_checkpoint"],
                "seeds": [
                    {
                        "seed": row["seed"],
                        "selected_step": row["selected_step"],
                        "eligible": row["selected_evidence"]["eligible"],
                    }
                    for row in result["seed_results"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
