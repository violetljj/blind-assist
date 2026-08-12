#!/usr/bin/env python3
"""Train uncertainty from residual magnitude and pairwise ordering.

Attempt-13 proved the point factors learn on expanded SuperTeacher labels, but
NLL-only sigma heads did not rank residuals.  This stage freezes every point
prediction and trains only deployable sigma refiners with proper-score,
residual-regression and pairwise ranking losses.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt13_complete_factor_head import (  # noqa: E402
    CompleteFactorHead,
    consumed_rows,
    parent_split,
    training_forward,
    uncertainty_ordering,
)
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    BOUNDARY_DISTANCE_SCALE_PX,
    DEPTHART_PYRAMID_CHANNELS,
    evaluate,
    extract_features,
    gaussian_nll_tensor,
    masked_mean,
    save_checkpoint,
)
from train_ag_r2_f1_factor_learnability_attempt02 import SpatialBlock  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    require,
    sha256_file,
)


ATTEMPT13_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt13-complete-factor-head-r1/result.json"
EXPECTED_ATTEMPT13_RESULT_SHA256 = "C1862DE977F2555FF7AE86197970E99901292095A493D361547B9F0751F001BD"
ATTEMPT13_COMPOSITE_SHA256 = "394B79BF5DA6F3350050A932D63D4BF20A77811A969EE912AF4FA8BBA03B6A04"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt14-residual-rank-uncertainty-r1"
TRAINING_SEED = 1417
FAMILY_PREFIXES = {
    "depth": ("depth_sigma_refiner.",),
    "support": ("support_uncertainty.", "support_sigma_refiner."),
    "boundary": ("boundary_sigma_refiner.",),
}
FAMILY_NLL_METRIC = {
    "depth": "depth_nll",
    "support": "support_nll",
    "boundary": "boundary_nll",
}


class ResidualRankUncertaintyHead(CompleteFactorHead):
    def __init__(self, baseline: dict[str, Any]) -> None:
        super().__init__(baseline)
        self.boundary_sigma_refiner = SpatialBlock(DEPTHART_PYRAMID_CHANNELS + 1, 32, 1)

    def forward(self, feature: torch.Tensor, base_depth_m: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(feature, base_depth_m)
        value = torch.cat([feature, base_depth_m.clamp(0.05, 20.0).log()], dim=1)
        boundary_log_sigma = outputs["boundary_sigma_px"].clamp_min(0.05).log()
        outputs["boundary_sigma_px"] = torch.exp(
            (boundary_log_sigma + 1.5 * torch.tanh(self.boundary_sigma_refiner(value))).clamp(
                math.log(0.05), math.log(64.0)
            )
        )
        return outputs


def load_model(baseline: dict[str, Any], device: torch.device) -> tuple[ResidualRankUncertaintyHead, dict[str, Any]]:
    require(sha256_file(ATTEMPT13_RESULT) == EXPECTED_ATTEMPT13_RESULT_SHA256, "Attempt13 result drift")
    attempt13 = json.loads(ATTEMPT13_RESULT.read_text(encoding="utf-8"))
    checkpoint = Path(attempt13["composite_checkpoint"]["path"])
    require(sha256_file(checkpoint) == ATTEMPT13_COMPOSITE_SHA256, "Attempt13 composite drift")
    model = ResidualRankUncertaintyHead(baseline).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {
        "boundary_sigma_refiner.net.0.weight",
        "boundary_sigma_refiner.net.0.bias",
        "boundary_sigma_refiner.net.1.weight",
        "boundary_sigma_refiner.net.1.bias",
        "boundary_sigma_refiner.net.3.weight",
        "boundary_sigma_refiner.net.3.bias",
        "boundary_sigma_refiner.net.4.weight",
        "boundary_sigma_refiner.net.4.bias",
        "boundary_sigma_refiner.net.5.weight",
        "boundary_sigma_refiner.net.5.bias",
        "boundary_sigma_refiner.net.7.weight",
        "boundary_sigma_refiner.net.7.bias",
    }
    require(set(missing) == expected_missing and not unexpected, "rank-uncertainty base state drift")
    return model, {"path": str(checkpoint.resolve()), "sha256": ATTEMPT13_COMPOSITE_SHA256}


def freeze_point_factors(model: ResidualRankUncertaintyHead) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for prefixes in FAMILY_PREFIXES.values():
        for prefix in prefixes:
            getattr(model, prefix[:-1]).requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def ordered_residual_loss(
    predicted_log_sigma: torch.Tensor,
    absolute_residual: torch.Tensor,
    valid: torch.Tensor,
    epsilon: float,
    maximum_points: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = valid.bool() & torch.isfinite(predicted_log_sigma) & torch.isfinite(absolute_residual)
    prediction = predicted_log_sigma[selected]
    residual = absolute_residual[selected].detach()
    if prediction.numel() == 0:
        zero = predicted_log_sigma.sum() * 0.0
        return zero, zero
    if prediction.numel() > maximum_points:
        indices = torch.randperm(prediction.numel(), device=prediction.device)[:maximum_points]
        prediction = prediction[indices]
        residual = residual[indices]
    target = (residual + epsilon).log()
    regression = F.smooth_l1_loss(prediction, target, beta=0.35)
    if prediction.numel() < 4:
        return regression, regression * 0.0
    order = torch.randperm(prediction.numel(), device=prediction.device)
    half = prediction.numel() // 2
    left, right = order[:half], order[-half:]
    target_delta = residual[left] - residual[right]
    informative = target_delta.abs() > epsilon
    if not bool(informative.any()):
        return regression, regression * 0.0
    direction = torch.sign(target_delta[informative])
    predicted_delta = prediction[left][informative] - prediction[right][informative]
    ranking = F.softplus(-2.0 * direction * predicted_delta).mean()
    return regression, ranking


def uncertainty_objective(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    depth_residual = (
        outputs["predicted_log_depth"] - targets["metric_depth_m"].clamp_min(0.01).log()
    ).detach()
    depth_valid = targets["metric_valid"]
    depth_sigma = outputs["depth_log_sigma"].exp()
    depth_regression, depth_ranking = ordered_residual_loss(
        outputs["depth_log_sigma"], depth_residual.abs(), depth_valid, 0.01
    )
    depth_nll = masked_mean(gaussian_nll_tensor(depth_residual, depth_sigma), depth_valid)

    boundary_target = targets["boundary_distance"].unsqueeze(0)
    boundary_distance = (
        -BOUNDARY_DISTANCE_SCALE_PX * outputs["boundary_probability"].clamp_min(1.0e-8).log()
    ).clamp_max(32.0)
    boundary_residual = (boundary_distance - boundary_target).detach()
    boundary_valid = targets["evidence_valid"].unsqueeze(0)
    boundary_regression, boundary_ranking = ordered_residual_loss(
        outputs["boundary_sigma_px"].log(), boundary_residual.abs(), boundary_valid, 0.10
    )
    boundary_nll = masked_mean(
        gaussian_nll_tensor(boundary_residual, outputs["boundary_sigma_px"]), boundary_valid
    )

    support_valid = targets["support_valid"]
    if bool(targets["support_plane_valid"]):
        support_residual = targets["support_residual"].abs()
        support_target = masked_mean(support_residual, support_valid).detach().clamp_min(0.005)
        support_sigma = outputs["support_residual_sigma_m"][0]
        support_regression = F.smooth_l1_loss(
            support_sigma.log(), support_target.log(), beta=0.35
        )
        support_nll = masked_mean(
            gaussian_nll_tensor(targets["support_residual"], support_sigma), support_valid
        )
    else:
        support_regression = outputs["support_residual_sigma_m"].sum() * 0.0
        support_nll = support_regression
    losses = {
        "depth_nll": depth_nll,
        "depth_residual_regression": depth_regression,
        "depth_pairwise_ranking": depth_ranking,
        "support_nll": support_nll,
        "support_residual_regression": support_regression,
        "boundary_nll": boundary_nll,
        "boundary_residual_regression": boundary_regression,
        "boundary_pairwise_ranking": boundary_ranking,
    }
    objective = (
        depth_nll
        + 0.75 * depth_regression
        + 0.50 * depth_ranking
        + support_nll
        + 0.75 * support_regression
        + boundary_nll / 4.0
        + 0.75 * boundary_regression
        + 0.50 * boundary_ranking
    ) / 5.5
    return objective, losses


def family_candidates(
    candidates: list[dict[str, Any]],
    base_evaluation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = {}
    tables = {}
    for family, metric in FAMILY_NLL_METRIC.items():
        rows = []
        base_nll = float(base_evaluation["overall_metrics"][metric])
        for candidate in candidates:
            ordering = uncertainty_ordering(candidate["evaluation"])[family]
            nll = float(candidate["evaluation"]["overall_metrics"][metric])
            ratio = nll / max(abs(base_nll), 1.0e-8)
            eligible = bool(ordering["nondecreasing"]) and ratio <= 1.05
            rows.append({
                "step": candidate["step"],
                "checkpoint": candidate["checkpoint"],
                "nll": nll,
                "nll_ratio_to_base": ratio,
                "ordering": ordering,
                "eligible": eligible,
                "score": ratio + (0.0 if ordering["nondecreasing"] else 1000.0),
            })
        eligible_rows = [row for row in rows if row["eligible"]]
        chosen = min(eligible_rows or rows, key=lambda row: (row["score"], row["step"], row["checkpoint"]["sha256"]))
        selected[family] = chosen
        tables[family] = rows
    return selected, tables


def compose(
    model: ResidualRankUncertaintyHead,
    selected: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    destination = model.state_dict()
    for family, row in selected.items():
        state = torch.load(row["checkpoint"]["path"], map_location=device, weights_only=True)["model"]
        for prefix in FAMILY_PREFIXES[family]:
            for key, value in state.items():
                if key.startswith(prefix):
                    destination[key] = value
    model.load_state_dict(destination, strict=True)
    return save_checkpoint(output_dir / "residual-rank-uncertainty-composite.pt", model, TRAINING_SEED, -1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    rows, data_receipt = consumed_rows()
    fit_parents, validation_parents = parent_split(rows)
    samples, feature_receipt = extract_features(
        rows, args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device
    )
    fit = [sample for sample in samples if sample.parent_id in set(fit_parents)]
    validation = [sample for sample in samples if sample.parent_id in set(validation_parents)]
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline = baseline_result["baseline_parameters"]
    model, base_checkpoint = load_model(baseline, device)
    base_evaluation = evaluate(model, validation, baseline, device)
    trainable = freeze_point_factors(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-4)
    by_parent: dict[str, list[Any]] = defaultdict(list)
    for sample in fit:
        by_parent[sample.parent_id].append(sample)
    parents = sorted(by_parent)
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
        ordering = uncertainty_ordering(evaluation)
        receipt = save_checkpoint(args.output_dir / f"uncertainty-step-{step}.pt", model, TRAINING_SEED, step)
        candidates.append({"step": step, "checkpoint": receipt, "evaluation": evaluation})
        print(json.dumps({"step": step, "nll": {family: evaluation["overall_metrics"][metric] for family, metric in FAMILY_NLL_METRIC.items()}, "ordering": {family: row["nondecreasing"] for family, row in ordering.items()}}), flush=True)

    capture(0)
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        parent = parents[rng.randrange(len(parents))]
        sample = by_parent[parent][rng.randrange(len(by_parent[parent]))]
        outputs, targets = training_forward(model, sample, device, rng.random() < 0.5)
        objective, losses = uncertainty_objective(outputs, targets)
        require(bool(torch.isfinite(objective)), "rank uncertainty objective non-finite")
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

    selected, candidate_tables = family_candidates(candidates, base_evaluation)
    composite = compose(model, selected, args.output_dir, device)
    final_evaluation = evaluate(model, validation, baseline, device)
    final_ordering = uncertainty_ordering(final_evaluation)
    nll_ratios = {
        family: float(final_evaluation["overall_metrics"][metric])
        / max(abs(float(base_evaluation["overall_metrics"][metric])), 1.0e-8)
        for family, metric in FAMILY_NLL_METRIC.items()
    }
    passed = all(row["nondecreasing"] for row in final_ordering.values()) and all(value <= 1.05 for value in nll_ratios.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt14_residual_rank_uncertainty_result_v1",
        "status": "ATTEMPT14_RESIDUAL_RANK_UNCERTAINTY_INTERNAL_PASS_FRESH_CANARY_LOCK_REQUIRED" if passed else "ATTEMPT14_RESIDUAL_RANK_UNCERTAINTY_INTERNAL_FAIL_NO_FRESH_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "fresh_canary_model_metrics_opened": False,
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "base_checkpoint": base_checkpoint,
        "base_evaluation": base_evaluation,
        "candidate_tables": candidate_tables,
        "selected_by_family": selected,
        "composite_checkpoint": composite,
        "final_evaluation": final_evaluation,
        "final_ordering": final_ordering,
        "nll_ratios_to_attempt13": nll_ratios,
        "loss_trace": trace,
        "decision": {
            "point_predictions_changed": False,
            "uncertainty_trained_from_observed_residuals": True,
            "direct_sigma_pseudo_truth_used": False,
            "fresh_canary_model_outputs_opened": False,
            "next_action_if_pass": "Freeze the complete point-factor plus residual-ranked uncertainty composite and execute the fresh canary exactly once.",
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
    parser.add_argument("--optimizer-steps", type=int, default=2400)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "ordering": {family: row["nondecreasing"] for family, row in result["final_ordering"].items()}, "nll_ratios": result["nll_ratios_to_attempt13"], "checkpoint": result["composite_checkpoint"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
