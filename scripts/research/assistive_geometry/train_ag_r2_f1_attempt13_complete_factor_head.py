#!/usr/bin/env python3
"""Retrain every non-geometric AG factor on all consumed SuperTeacher labels.

The successful residual metric-depth and learned camera-height components are
kept fixed.  Support normal remains a runtime gravity factor.  The remaining
probability, validity and uncertainty heads are retrained parent-uniformly.
The still-unopened Attempt-12 Freiburg3 cohort is never loaded here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt08_depthart_residual_head import DepthArtResidualFactorHead  # noqa: E402
from train_ag_r2_f1_attempt09_expanded_residual_depth import expanded_rows  # noqa: E402
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    BOUNDARY_DISTANCE_SCALE_PX,
    DEPTHART_PYRAMID_CHANNELS,
    evaluate,
    extract_features,
    gaussian_nll_tensor,
    masked_mean,
    move_targets,
    save_checkpoint,
)
from train_ag_r2_f1_factor_learnability_attempt02 import GlobalBlock, SpatialBlock  # noqa: E402
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


ATTEMPT05_LABEL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-fresh-ag-held-labels-r0/result.json"
EXPECTED_ATTEMPT05_LABEL_SHA256 = "4DBF0E85F45357C613221DF9F2C5A5E3B0971C314EB29D1967C02E0D6FAEB7CC"
ATTEMPT10_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt10-camera-height-r0/result.json"
EXPECTED_ATTEMPT10_RESULT_SHA256 = "963CB8D429962F8649B0CA0118C89D66A7C8698A851E0803E5D8958A64D9296A"
BASE_CHECKPOINT_SHA256 = "03C10530C729916CAB8A3253F1826B8E77FC3FF59AFC568766FCB76644DFD627"
FRESH_LABEL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt12-pose-anchored-fresh-canary-labels-r1/result.json"
EXPECTED_FRESH_LABEL_RESULT_SHA256 = "659FAD8A11EDB8D148FF5DC26E2BA43106BAD87EB5C23AEB7D4202832673C892"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt13-complete-factor-head-r1"
SPLIT_TOKEN = "AG_R2_F1_ATTEMPT13_COMPLETE_FACTOR_HEAD_2026-08-11"
TRAINING_SEED = 1317

COMPONENT_PREFIXES = {
    "depth_uncertainty": ("depth_sigma_refiner.",),
    "support_probability": ("support_probability.",),
    "support_uncertainty": ("support_uncertainty.", "support_sigma_refiner."),
    "obstacle": ("obstacle.",),
    "boundary": ("boundary.",),
    "evidence_validity": ("evidence_validity.",),
    "support_validity": ("support_validity.",),
}
COMPONENT_METRICS = {
    "depth_uncertainty": ("depth_nll",),
    "support_probability": ("support_brier",),
    "support_uncertainty": ("support_nll",),
    "obstacle": ("obstacle_brier",),
    "boundary": ("boundary_distance_abs_error_px", "boundary_nll"),
    "evidence_validity": ("evidence_validity_brier",),
    "support_validity": ("support_validity_brier",),
}


class CompleteFactorHead(DepthArtResidualFactorHead):
    """Residual-depth head plus independently trainable uncertainty refiners."""

    def __init__(self, baseline: dict[str, Any]) -> None:
        super().__init__(baseline)
        inputs = DEPTHART_PYRAMID_CHANNELS + 1
        self.depth_sigma_refiner = SpatialBlock(inputs, 32, 1)
        self.support_sigma_refiner = GlobalBlock(inputs, 32, 1)

    def forward(self, feature: torch.Tensor, base_depth_m: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(feature, base_depth_m)
        value = torch.cat([feature, base_depth_m.clamp(0.05, 20.0).log()], dim=1)
        outputs["depth_log_sigma"] = (
            outputs["depth_log_sigma"] + 0.75 * torch.tanh(self.depth_sigma_refiner(value))
        ).clamp(math.log(0.005), math.log(5.0))
        support_log_sigma = outputs["support_residual_sigma_m"].clamp_min(0.005).log()
        support_refinement = 1.5 * torch.tanh(self.support_sigma_refiner(value)[:, 0])
        outputs["support_residual_sigma_m"] = torch.exp(
            (support_log_sigma + support_refinement).clamp(math.log(0.005), math.log(5.0))
        )
        return outputs


def consumed_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, receipt = expanded_rows()
    require(sha256_file(ATTEMPT05_LABEL_RESULT) == EXPECTED_ATTEMPT05_LABEL_SHA256, "Attempt05 labels drift")
    attempt05 = json.loads(ATTEMPT05_LABEL_RESULT.read_text(encoding="utf-8"))
    consumed_canary = [
        {**row, "role": "CONSUMED_ATTEMPT11_FAILURE_TRAINING"}
        for row in attempt05["frames"]
        if row["role"] == "TRAIN_CANARY"
    ]
    require(len(consumed_canary) == 6 and len({row["parent_id"] for row in consumed_canary}) == 2, "consumed Attempt11 roster drift")
    by_sample = {str(row["sample_id"]): row for row in rows}
    for row in consumed_canary:
        require(row["sample_id"] not in by_sample, "consumed Attempt11 sample collision")
        by_sample[row["sample_id"]] = row
    combined = sorted(by_sample.values(), key=lambda row: (str(row["parent_id"]), str(row["sample_id"])))
    require(len(combined) == 144 and len({row["parent_id"] for row in combined}) == 27, "complete-factor consumed roster drift")
    return combined, {
        "expanded": receipt,
        "consumed_attempt11_labels": {
            "path": str(ATTEMPT05_LABEL_RESULT.resolve()),
            "sha256": EXPECTED_ATTEMPT05_LABEL_SHA256,
            "parents": sorted({str(row["parent_id"]) for row in consumed_canary}),
            "frame_count": len(consumed_canary),
        },
    }


def parent_split(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)
    joint = {
        parent
        for parent, members in by_parent.items()
        if any(bool(row["support_plane_valid"]) for row in members)
        and sum(int(row["support_valid_pixels"]) for row in members) > 0
        and sum(int(row["evidence_valid_pixels"]) for row in members) > 0
    }
    require(len(by_parent) == 27 and len(joint) == 26, "joint consumed parent coverage drift")
    forced_fit = {
        "rgbd_dataset_freiburg1_desk2",
        "rgbd_dataset_freiburg2_xyz",
        "rgbd_dataset_freiburg2_pioneer_slam2",
    }
    require(forced_fit <= joint, "consumed failure parent missing")
    candidates = joint - forced_fit
    by_family = {
        "freiburg1": sorted(
            (parent for parent in candidates if "freiburg1" in parent),
            key=lambda parent: hashlib.sha256(f"{SPLIT_TOKEN}:{parent}".encode()).hexdigest(),
        ),
        "freiburg2": sorted(
            (parent for parent in candidates if "freiburg2" in parent),
            key=lambda parent: hashlib.sha256(f"{SPLIT_TOKEN}:{parent}".encode()).hexdigest(),
        ),
    }
    validation = sorted(by_family["freiburg1"][-2:] + by_family["freiburg2"][-3:])
    fit = sorted(set(by_parent) - set(validation))
    require(len(fit) == 22 and len(validation) == 5 and forced_fit <= set(fit), "complete-factor split drift")
    return fit, validation


def load_base_model(baseline: dict[str, Any], device: torch.device) -> tuple[CompleteFactorHead, dict[str, Any]]:
    require(sha256_file(ATTEMPT10_RESULT) == EXPECTED_ATTEMPT10_RESULT_SHA256, "Attempt10 result drift")
    attempt10 = json.loads(ATTEMPT10_RESULT.read_text(encoding="utf-8"))
    checkpoint = Path(attempt10["selected_checkpoint"]["path"])
    require(sha256_file(checkpoint) == BASE_CHECKPOINT_SHA256, "Attempt10 checkpoint drift")
    model = CompleteFactorHead(baseline).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {
        "depth_sigma_refiner.net.0.weight",
        "depth_sigma_refiner.net.0.bias",
        "depth_sigma_refiner.net.1.weight",
        "depth_sigma_refiner.net.1.bias",
        "depth_sigma_refiner.net.3.weight",
        "depth_sigma_refiner.net.3.bias",
        "depth_sigma_refiner.net.4.weight",
        "depth_sigma_refiner.net.4.bias",
        "depth_sigma_refiner.net.5.weight",
        "depth_sigma_refiner.net.5.bias",
        "depth_sigma_refiner.net.7.weight",
        "depth_sigma_refiner.net.7.bias",
        "support_sigma_refiner.net.0.weight",
        "support_sigma_refiner.net.0.bias",
        "support_sigma_refiner.net.2.weight",
        "support_sigma_refiner.net.2.bias",
    }
    require(set(missing) == expected_missing and not unexpected, "complete-factor base state drift")
    return model, {"path": str(checkpoint.resolve()), "sha256": BASE_CHECKPOINT_SHA256}


def freeze_fixed_components(model: CompleteFactorHead) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for component in COMPONENT_PREFIXES:
        for prefix in COMPONENT_PREFIXES[component]:
            module_name = prefix[:-1]
            getattr(model, module_name).requires_grad_(True)


def training_forward(
    model: CompleteFactorHead,
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


def factor_objective(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    normalization: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metric_valid = targets["metric_valid"]
    depth_residual = outputs["predicted_log_depth"] - targets["metric_depth_m"].clamp_min(0.01).log()
    evidence_valid = targets["evidence_valid"]
    support_valid = targets["support_valid"]
    plane_valid = targets["support_plane_valid"].bool()
    boundary_distance_target = targets["boundary_distance"].unsqueeze(0)
    boundary_target = torch.exp(-boundary_distance_target / BOUNDARY_DISTANCE_SCALE_PX)
    boundary_distance = (
        -BOUNDARY_DISTANCE_SCALE_PX * outputs["boundary_probability"].clamp_min(1.0e-8).log()
    ).clamp_max(32.0)
    if bool(plane_valid):
        support_nll = masked_mean(
            gaussian_nll_tensor(targets["support_residual"], outputs["support_residual_sigma_m"][0]),
            support_valid,
        )
    else:
        support_nll = outputs["support_residual_sigma_m"].sum() * 0.0
    losses = {
        "depth_heteroscedastic_nll": masked_mean(
            gaussian_nll_tensor(depth_residual, outputs["depth_log_sigma"].exp()), metric_valid
        ),
        "support_probability_brier": masked_mean(
            (outputs["support_probability"] - targets["support"]).square(), support_valid
        ),
        "support_residual_heteroscedastic_nll": support_nll,
        "support_validity_brier": (
            outputs["support_valid_probability"][0] - plane_valid.float()
        ).square(),
        "obstacle_evidence_brier": masked_mean(
            (outputs["obstacle_probability"] - targets["obstacle"]).square(), evidence_valid
        ),
        "boundary_probability_brier": masked_mean(
            (outputs["boundary_probability"] - boundary_target).square(), evidence_valid
        ),
        "boundary_localization_heteroscedastic_nll": masked_mean(
            gaussian_nll_tensor(
                boundary_distance - boundary_distance_target, outputs["boundary_sigma_px"]
            ),
            evidence_valid,
        ),
        "boundary_distance_smooth_l1": masked_mean(
            F.smooth_l1_loss(boundary_distance, boundary_distance_target, reduction="none", beta=2.0),
            evidence_valid,
        ),
        "evidence_validity_brier": (
            outputs["evidence_valid_probability"] - evidence_valid.float()
        ).square().mean(),
    }
    weighted = [
        losses["depth_heteroscedastic_nll"] / float(normalization["depth_heteroscedastic_nll"]),
        losses["support_probability_brier"] / float(normalization["support_probability_brier"]),
        losses["support_residual_heteroscedastic_nll"] / float(normalization["support_residual_heteroscedastic_nll"]),
        0.25 * losses["support_validity_brier"] / float(normalization["support_validity_brier"]),
        losses["obstacle_evidence_brier"] / float(normalization["obstacle_evidence_brier"]),
        losses["boundary_probability_brier"] / float(normalization["boundary_probability_brier"]),
        losses["boundary_localization_heteroscedastic_nll"] / float(normalization["boundary_localization_heteroscedastic_nll"]),
        0.5 * losses["boundary_distance_smooth_l1"] / BOUNDARY_DISTANCE_SCALE_PX,
        0.25 * losses["evidence_validity_brier"] / float(normalization["evidence_validity_brier"]),
    ]
    return torch.stack(weighted).mean(), losses


def component_score(
    evaluation: dict[str, Any],
    base: dict[str, Any],
    metrics: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    ratios = {
        metric: float(evaluation["overall_metrics"][metric])
        / max(abs(float(base["overall_metrics"][metric])), 1.0e-8)
        for metric in metrics
    }
    return float(np.mean(list(ratios.values()))), ratios


def uncertainty_ordering(evaluation: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "depth": ("depth_sigma_mean", "depth_abs_residual_mean"),
        "support": ("support_sigma_mean", "support_abs_residual_mean"),
        "boundary": ("boundary_sigma_mean", "boundary_abs_residual_mean"),
    }
    result = {}
    for family, (sigma_key, residual_key) in fields.items():
        pairs = []
        for row in evaluation["frames"]:
            metrics = row["metrics"]
            sigma = float(metrics[sigma_key])
            residual = float(metrics[residual_key])
            if math.isfinite(sigma) and math.isfinite(residual) and sigma > 0.0:
                pairs.append((np.asarray([sigma]), np.asarray([residual])))
        result[family] = quantile_residual_summary(pairs)
    return result


def compose_components(
    model: CompleteFactorHead,
    candidates: list[dict[str, Any]],
    base_evaluation: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected: dict[str, Any] = {}
    destination = model.state_dict()
    for component, metrics in COMPONENT_METRICS.items():
        rows = []
        for candidate in candidates:
            score, ratios = component_score(candidate["evaluation"], base_evaluation, metrics)
            rows.append({"step": candidate["step"], "checkpoint": candidate["checkpoint"], "score": score, "metric_ratios_to_base": ratios})
        chosen = min(rows, key=lambda row: (row["score"], row["step"], row["checkpoint"]["sha256"]))
        state = torch.load(chosen["checkpoint"]["path"], map_location=device, weights_only=True)["model"]
        for prefix in COMPONENT_PREFIXES[component]:
            for key, value in state.items():
                if key.startswith(prefix):
                    destination[key] = value
        selected[component] = {"selected": chosen, "candidates": rows}
    model.load_state_dict(destination, strict=True)
    receipt = save_checkpoint(output_dir / "complete-factor-composite.pt", model, TRAINING_SEED, -1)
    return receipt, selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    require(sha256_file(FRESH_LABEL_RESULT) == EXPECTED_FRESH_LABEL_RESULT_SHA256, "fresh canary label drift")
    fresh = json.loads(FRESH_LABEL_RESULT.read_text(encoding="utf-8"))
    require(fresh["passed"] and not fresh["decision"]["model_metrics_opened"], "fresh canary not sealed")
    fresh_parents = set(fresh["parent_joint"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    rows, data_receipt = consumed_rows()
    require(fresh_parents.isdisjoint(str(row["parent_id"]) for row in rows), "fresh canary leaked into training")
    fit_parents, validation_parents = parent_split(rows)
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    fit = [sample for sample in samples if sample.parent_id in set(fit_parents)]
    validation = [sample for sample in samples if sample.parent_id in set(validation_parents)]
    require(len(fit) + len(validation) == 144 and len(validation) >= 15, "complete-factor sample split drift")
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline = baseline_result["baseline_parameters"]
    normalization = baseline_result["optimizer_normalization"]
    model, base_checkpoint = load_base_model(baseline, device)
    base_evaluation = evaluate(model, validation, baseline, device)
    freeze_fixed_components(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    require(trainable and all(parameter.requires_grad for parameter in trainable), "no trainable factor parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-4)
    fit_by_parent: dict[str, list[Any]] = defaultdict(list)
    for sample in fit:
        fit_by_parent[sample.parent_id].append(sample)
    parents = sorted(fit_by_parent)
    require(len(parents) == 22, "FIT parent count drift")
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    rng = random.Random(TRAINING_SEED)
    candidates = []
    trace = []

    def capture(step: int) -> None:
        model.eval()
        evaluation = evaluate(model, validation, baseline, device)
        receipt = save_checkpoint(args.output_dir / f"complete-factor-step-{step}.pt", model, TRAINING_SEED, step)
        candidates.append({"step": step, "checkpoint": receipt, "evaluation": evaluation})
        print(json.dumps({"step": step, "metrics": {metric: evaluation["overall_metrics"][metric] for metric in sorted({value for metrics in COMPONENT_METRICS.values() for value in metrics})}}), flush=True)

    capture(0)
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        parent = parents[rng.randrange(len(parents))]
        sample = fit_by_parent[parent][rng.randrange(len(fit_by_parent[parent]))]
        outputs, targets = training_forward(model, sample, device, rng.random() < 0.5)
        objective, losses = factor_objective(outputs, targets, normalization)
        require(bool(torch.isfinite(objective)), "complete-factor objective non-finite")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        progress = step / args.optimizer_steps
        optimizer.param_groups[0]["lr"] = args.learning_rate * (
            0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append({"step": step, "parent_id": parent, "objective": float(objective.detach()), "losses": {key: float(value.detach()) for key, value in losses.items()}})
        if step % args.checkpoint_interval == 0 or step == args.optimizer_steps:
            capture(step)

    composite_checkpoint, component_selection = compose_components(
        model, candidates, base_evaluation, args.output_dir, device
    )
    model.eval()
    final_evaluation = evaluate(model, validation, baseline, device)
    component_improved = {
        component: component_score(final_evaluation, base_evaluation, metrics)[0] < 1.0
        for component, metrics in COMPONENT_METRICS.items()
    }
    uncertainty = uncertainty_ordering(final_evaluation)
    uncertainty_nondecreasing = {
        family: bool(uncertainty[family]["nondecreasing"])
        for family in ("depth", "support", "boundary")
    }
    passed = (
        sum(component_improved.values()) >= 6
        and component_improved["support_probability"]
        and component_improved["obstacle"]
        and component_improved["boundary"]
        and all(uncertainty_nondecreasing.values())
    )
    result = {
        "schema": "blindassist_ag_r2_f1_attempt13_complete_factor_head_result_v1",
        "status": "ATTEMPT13_COMPLETE_FACTOR_HEAD_INTERNAL_PASS_FRESH_CANARY_LOCK_REQUIRED" if passed else "ATTEMPT13_COMPLETE_FACTOR_HEAD_INTERNAL_FAIL_NO_FRESH_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "fresh_canary_model_metrics_opened": False,
        "fresh_canary_labels": {"path": str(FRESH_LABEL_RESULT.resolve()), "sha256": EXPECTED_FRESH_LABEL_RESULT_SHA256, "parents": sorted(fresh_parents)},
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "role_frame_counts": {"FIT": len(fit), "INTERNAL_VALIDATION": len(validation), "FRESH_CANARY": 0},
        "base_checkpoint": base_checkpoint,
        "base_evaluation": base_evaluation,
        "component_selection": component_selection,
        "composite_checkpoint": composite_checkpoint,
        "final_evaluation": final_evaluation,
        "component_improved_over_consumed_base": component_improved,
        "uncertainty_nondecreasing": uncertainty_nondecreasing,
        "loss_trace": trace,
        "decision": {
            "fixed_depth_and_height_preserved": True,
            "runtime_gravity_support_normal_preserved": True,
            "all_other_factor_families_retrained": True,
            "fresh_canary_model_outputs_opened": False,
            "next_action_if_pass": "Freeze exact composite and execute the four-parent pose-anchored fresh canary once.",
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
    parser.add_argument("--optimizer-steps", type=int, default=3200)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "component_improved": result["component_improved_over_consumed_base"], "uncertainty_nondecreasing": result["uncertainty_nondecreasing"], "checkpoint": result["composite_checkpoint"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
