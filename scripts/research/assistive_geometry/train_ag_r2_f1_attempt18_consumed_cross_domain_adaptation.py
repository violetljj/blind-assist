#!/usr/bin/env python3
"""Adapt height, validity and uncertainty using the consumed Attempt-16 evidence."""

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
    consumed_rows,
    parent_split,
    training_forward,
    uncertainty_ordering,
)
from train_ag_r2_f1_attempt14_residual_rank_uncertainty import (  # noqa: E402
    ResidualRankUncertaintyHead,
    ordered_residual_loss,
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


ATTEMPT14_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt14-residual-rank-uncertainty-r1/result.json"
EXPECTED_ATTEMPT14_RESULT_SHA256 = "1F8F78E9EBBD3E42BD1F32032F4597DA4A1D34203E1810520C26EC979CA61D4A"
ATTEMPT14_CHECKPOINT_SHA256 = "A0F8A99E65A83E88A19AAE59A19AACCF8B01437FFD0B87BB89D0A53B4345E842"
ATTEMPT16_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt16-pose-anchored-fresh-canary-r0/result.json"
EXPECTED_ATTEMPT16_RESULT_SHA256 = "1483816AAB3AA371F7AA923B911D62D43E171C7FDC9934BEE39DE1BC603CE093"
ATTEMPT16_LABELS = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt12-pose-anchored-fresh-canary-labels-r1/result.json"
EXPECTED_ATTEMPT16_LABELS_SHA256 = "659FAD8A11EDB8D148FF5DC26E2BA43106BAD87EB5C23AEB7D4202832673C892"
ATTEMPT17_SOURCE_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT17_FRESH_CANARY_SOURCE_LOCK_2026-08-12.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt18-consumed-cross-domain-adaptation-r0"
TRAINING_SEED = 1817
COMPONENT_PREFIXES = {
    "camera_height": ("camera_height.",),
    "depth_validity": ("depth_validity_refiner.",),
    "depth_uncertainty": ("depth_sigma_refiner.",),
    "boundary_uncertainty": ("boundary_sigma_refiner.",),
}


class CrossDomainFactorHead(ResidualRankUncertaintyHead):
    def __init__(self, baseline: dict[str, Any]) -> None:
        super().__init__(baseline)
        self.depth_validity_refiner = SpatialBlock(DEPTHART_PYRAMID_CHANNELS + 1, 16, 1)

    def forward(self, feature: torch.Tensor, base_depth_m: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(feature, base_depth_m)
        value = torch.cat([feature, base_depth_m.clamp(0.05, 20.0).log()], dim=1)
        probability = outputs["depth_valid_probability"].clamp(1.0e-5, 1.0 - 1.0e-5)
        logit = torch.log(probability) - torch.log1p(-probability)
        outputs["depth_valid_probability"] = torch.sigmoid(
            logit + 1.5 * torch.tanh(self.depth_validity_refiner(value))
        )
        return outputs


def adapted_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, receipt = consumed_rows()
    require(sha256_file(ATTEMPT16_RESULT) == EXPECTED_ATTEMPT16_RESULT_SHA256, "Attempt16 result drift")
    require(sha256_file(ATTEMPT16_LABELS) == EXPECTED_ATTEMPT16_LABELS_SHA256, "Attempt16 labels drift")
    attempt16 = json.loads(ATTEMPT16_RESULT.read_text(encoding="utf-8"))
    require(attempt16["canary_opened_once"] and not attempt16["passed"] and not attempt16["factor_tensors"], "Attempt16 consumption state drift")
    labels = json.loads(ATTEMPT16_LABELS.read_text(encoding="utf-8"))
    consumed = [{**row, "role": "CONSUMED_ATTEMPT16_CROSS_DOMAIN_FIT"} for row in labels["frames"]]
    by_sample = {str(row["sample_id"]): row for row in rows}
    for row in consumed:
        require(row["sample_id"] not in by_sample, "Attempt16 sample collision")
        by_sample[row["sample_id"]] = row
    combined = sorted(by_sample.values(), key=lambda row: (str(row["parent_id"]), str(row["sample_id"])))
    require(len(combined) == 156 and len({row["parent_id"] for row in combined}) == 31, "cross-domain roster drift")
    return combined, {
        "prior_consumed": receipt,
        "attempt16_result": {"path": str(ATTEMPT16_RESULT.resolve()), "sha256": EXPECTED_ATTEMPT16_RESULT_SHA256},
        "attempt16_labels": {"path": str(ATTEMPT16_LABELS.resolve()), "sha256": EXPECTED_ATTEMPT16_LABELS_SHA256, "frame_count": len(consumed)},
        "attempt17_source_lock": {"path": str(ATTEMPT17_SOURCE_LOCK.resolve()), "sha256": sha256_file(ATTEMPT17_SOURCE_LOCK)},
    }


def load_model(baseline: dict[str, Any], device: torch.device) -> tuple[CrossDomainFactorHead, dict[str, Any]]:
    require(sha256_file(ATTEMPT14_RESULT) == EXPECTED_ATTEMPT14_RESULT_SHA256, "Attempt14 result drift")
    attempt14 = json.loads(ATTEMPT14_RESULT.read_text(encoding="utf-8"))
    checkpoint = Path(attempt14["composite_checkpoint"]["path"])
    require(sha256_file(checkpoint) == ATTEMPT14_CHECKPOINT_SHA256, "Attempt14 checkpoint drift")
    model = CrossDomainFactorHead(baseline).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {
        "depth_validity_refiner.net.0.weight",
        "depth_validity_refiner.net.0.bias",
        "depth_validity_refiner.net.1.weight",
        "depth_validity_refiner.net.1.bias",
        "depth_validity_refiner.net.3.weight",
        "depth_validity_refiner.net.3.bias",
        "depth_validity_refiner.net.4.weight",
        "depth_validity_refiner.net.4.bias",
        "depth_validity_refiner.net.5.weight",
        "depth_validity_refiner.net.5.bias",
        "depth_validity_refiner.net.7.weight",
        "depth_validity_refiner.net.7.bias",
    }
    require(set(missing) == expected_missing and not unexpected, "cross-domain model state drift")
    return model, {"path": str(checkpoint.resolve()), "sha256": ATTEMPT14_CHECKPOINT_SHA256}


def freeze_components(model: CrossDomainFactorHead) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for prefixes in COMPONENT_PREFIXES.values():
        for prefix in prefixes:
            getattr(model, prefix[:-1]).requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def adaptation_objective(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metric_valid = targets["metric_valid"]
    depth_residual = (
        outputs["predicted_log_depth"] - targets["metric_depth_m"].clamp_min(0.01).log()
    ).detach()
    depth_regression, depth_ranking = ordered_residual_loss(
        outputs["depth_log_sigma"], depth_residual.abs(), metric_valid, 0.01
    )
    depth_nll = masked_mean(
        gaussian_nll_tensor(depth_residual, outputs["depth_log_sigma"].exp()), metric_valid
    )
    depth_validity = (
        outputs["depth_valid_probability"] - metric_valid.float()
    ).square().mean()

    boundary_target = targets["boundary_distance"].unsqueeze(0)
    boundary_distance = (
        -BOUNDARY_DISTANCE_SCALE_PX * outputs["boundary_probability"].clamp_min(1.0e-8).log()
    ).clamp_max(32.0)
    boundary_residual = (boundary_distance - boundary_target).detach()
    evidence_valid = targets["evidence_valid"].unsqueeze(0)
    boundary_regression, boundary_ranking = ordered_residual_loss(
        outputs["boundary_sigma_px"].log(), boundary_residual.abs(), evidence_valid, 0.10
    )
    boundary_nll = masked_mean(
        gaussian_nll_tensor(boundary_residual, outputs["boundary_sigma_px"]), evidence_valid
    )

    if bool(targets["support_plane_valid"]):
        height_error = outputs["camera_height_m"][0].clamp_min(0.1).log() - targets["camera_height_m"].clamp_min(0.1).log()
        height = F.smooth_l1_loss(height_error, torch.zeros_like(height_error), beta=0.10)
    else:
        height = outputs["camera_height_m"].sum() * 0.0
    losses = {
        "camera_height_log_huber": height,
        "depth_validity_brier": depth_validity,
        "depth_nll": depth_nll,
        "depth_residual_regression": depth_regression,
        "depth_pairwise_ranking": depth_ranking,
        "boundary_nll": boundary_nll,
        "boundary_residual_regression": boundary_regression,
        "boundary_pairwise_ranking": boundary_ranking,
    }
    objective = (
        2.0 * height
        + 0.5 * depth_validity
        + depth_nll
        + 0.75 * depth_regression
        + 0.5 * depth_ranking
        + boundary_nll / 4.0
        + 0.75 * boundary_regression
        + 0.5 * boundary_ranking
    ) / 6.0
    return objective, losses


def component_score(component: str, evaluation: dict[str, Any], base: dict[str, Any]) -> float:
    metric = {
        "camera_height": "camera_height_abs_log_error",
        "depth_validity": "depth_validity_brier",
        "depth_uncertainty": "depth_nll",
        "boundary_uncertainty": "boundary_nll",
    }[component]
    ratio = float(evaluation["overall_metrics"][metric]) / max(abs(float(base["overall_metrics"][metric])), 1.0e-8)
    if component in {"depth_uncertainty", "boundary_uncertainty"}:
        family = component.split("_")[0]
        ordered = uncertainty_ordering(evaluation)[family]["nondecreasing"]
        return ratio + (0.0 if ordered else 1000.0)
    return ratio


def compose(
    model: CrossDomainFactorHead,
    candidates: list[dict[str, Any]],
    base_evaluation: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    destination = model.state_dict()
    selected = {}
    for component, prefixes in COMPONENT_PREFIXES.items():
        rows = [
            {"step": row["step"], "checkpoint": row["checkpoint"], "score": component_score(component, row["evaluation"], base_evaluation)}
            for row in candidates
        ]
        chosen = min(rows, key=lambda row: (row["score"], row["step"], row["checkpoint"]["sha256"]))
        state = torch.load(chosen["checkpoint"]["path"], map_location=device, weights_only=True)["model"]
        for prefix in prefixes:
            for key, value in state.items():
                if key.startswith(prefix):
                    destination[key] = value
        selected[component] = {"selected": chosen, "candidates": rows}
    model.load_state_dict(destination, strict=True)
    receipt = save_checkpoint(output_dir / "cross-domain-composite.pt", model, TRAINING_SEED, -1)
    return receipt, selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    rows, data_receipt = adapted_rows()
    prior_rows, _ = consumed_rows()
    prior_fit, validation_parents = parent_split(prior_rows)
    attempt16_parents = {
        "rgbd_dataset_freiburg3_cabinet",
        "rgbd_dataset_freiburg3_sitting_halfsphere",
        "rgbd_dataset_freiburg3_teddy",
        "rgbd_dataset_freiburg3_walking_static",
    }
    fit_parents = sorted(set(prior_fit) | attempt16_parents)
    require(set(fit_parents).isdisjoint(validation_parents), "adaptation split overlap")
    samples, feature_receipt = extract_features(
        rows, args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device
    )
    fit = [sample for sample in samples if sample.parent_id in set(fit_parents)]
    validation = [sample for sample in samples if sample.parent_id in set(validation_parents)]
    require(len(fit) == 132 and len(validation) == 24, "adaptation frame split drift")
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline = baseline_result["baseline_parameters"]
    model, source_checkpoint = load_model(baseline, device)
    base_evaluation = evaluate(model, validation, baseline, device)
    trainable = freeze_components(model)
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
        receipt = save_checkpoint(args.output_dir / f"adapt-step-{step}.pt", model, TRAINING_SEED, step)
        candidates.append({"step": step, "checkpoint": receipt, "evaluation": evaluation})
        print(json.dumps({"step": step, "height": evaluation["overall_metrics"]["camera_height_abs_log_error"], "depth_validity": evaluation["overall_metrics"]["depth_validity_brier"], "depth_nll": evaluation["overall_metrics"]["depth_nll"], "boundary_nll": evaluation["overall_metrics"]["boundary_nll"]}), flush=True)

    capture(0)
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        parent = parents[rng.randrange(len(parents))]
        sample = by_parent[parent][rng.randrange(len(by_parent[parent]))]
        outputs, targets = training_forward(model, sample, device, rng.random() < 0.5)
        objective, losses = adaptation_objective(outputs, targets)
        require(bool(torch.isfinite(objective)), "adaptation objective non-finite")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        progress = step / args.optimizer_steps
        optimizer.param_groups[0]["lr"] = args.learning_rate * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append({"step": step, "parent_id": parent, "objective": float(objective.detach()), "losses": {key: float(value.detach()) for key, value in losses.items()}})
        if step % args.checkpoint_interval == 0 or step == args.optimizer_steps:
            capture(step)

    composite, selection = compose(model, candidates, base_evaluation, args.output_dir, device)
    final_evaluation = evaluate(model, validation, baseline, device)
    ratios = {
        component: component_score(component, final_evaluation, base_evaluation)
        for component in COMPONENT_PREFIXES
    }
    passed = all(value <= 1.05 for value in ratios.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt18_consumed_cross_domain_adaptation_result_v1",
        "status": "ATTEMPT18_CROSS_DOMAIN_ADAPTATION_INTERNAL_PASS_RECALIBRATION_REQUIRED" if passed else "ATTEMPT18_CROSS_DOMAIN_ADAPTATION_INTERNAL_FAIL_NO_NEW_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "attempt17_model_outputs_opened": False,
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "role_frame_counts": {"FIT": len(fit), "INTERNAL_VALIDATION": len(validation), "ATTEMPT17_FRESH_CANARY": 0},
        "source_checkpoint": source_checkpoint,
        "base_evaluation": base_evaluation,
        "component_selection": selection,
        "composite_checkpoint": composite,
        "final_evaluation": final_evaluation,
        "component_scores": ratios,
        "loss_trace": trace,
        "decision": {
            "core_point_factors_changed": False,
            "consumed_attempt16_used_for_fit": True,
            "attempt17_model_outputs_opened": False,
            "next_action_if_pass": "Recalibrate uncertainty on consumed evidence, then freeze one Attempt17 execution.",
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
    parser.add_argument("--optimizer-steps", type=int, default=2600)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "scores": result["component_scores"], "checkpoint": result["composite_checkpoint"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
