#!/usr/bin/env python3
"""Attempt 02: component-isolated no-regret AG R2 F1 factor learning."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    DEFAULT_LABEL_RESULT,
    DEPTHART_PYRAMID_CHANNELS,
    PRIMARY_METRICS,
    canary_gate,
    compute_losses,
    evaluate,
    extract_features,
    move_targets,
    save_checkpoint,
    serialize_predictions,
    sha256_file,
)
from validate_ag_r2_f1_attempt02_execution_lock import validate as validate_execution_lock  # noqa: E402

DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT02_FACTOR_SPLIT_EXECUTION_LOCK_2026-08-11.json"
DEFAULT_SELECTION_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt02-fresh-canary-labels-r2/result.json"
DEFAULT_CANARY_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt02-corrected-canary-labels-r0/result.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-factor-learnability-attempt02-r0"
EXPECTED_LABEL_RESULT_SHA256 = "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422"
EXPECTED_BASELINE_RESULT_SHA256 = "EECD5C9244C6A8A467B7890AF79D7871374AC8B325A87B48AE9E052089908F44"
EXPECTED_SELECTION_RESULT_SHA256 = "7AFB581F30779A53CF1A54C15B06CFE3176D45D3CD5B92F52544283174188CCD"
EXPECTED_DEPTHART_SHA256 = "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65"
COMPONENT_METRICS = {
    "depth": (
        "depth_shape_abs_log_error",
        "depth_scale_abs_log_error",
        "depth_nll",
    ),
    "support_probability": ("support_brier",),
    "support_normal": ("support_plane_angular_error_rad",),
    "camera_height": ("camera_height_abs_log_error",),
    "support_uncertainty": ("support_nll",),
    "obstacle": ("obstacle_brier",),
    "boundary": ("boundary_distance_abs_error_px", "boundary_nll"),
}
COMPONENT_PREFIXES = {
    "depth": ("depth.",),
    "support_probability": ("support_probability.",),
    "support_normal": ("support_normal.",),
    "camera_height": ("camera_height.",),
    "support_uncertainty": ("support_uncertainty.",),
    "obstacle": ("obstacle.",),
    "boundary": ("boundary.",),
    "support_validity": ("support_validity.",),
    "evidence_validity": ("evidence_validity.",),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def logit(probability: float) -> float:
    bounded = min(max(float(probability), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(bounded / (1.0 - bounded))


def inverse_softplus(value: float) -> float:
    bounded = max(float(value), 1.0e-5)
    return math.log(math.expm1(bounded)) if bounded < 20.0 else bounded


class SpatialBlock(nn.Module):
    def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
        super().__init__()
        groups = 8 if hidden % 8 == 0 else 4
        self.net = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, outputs, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class GlobalBlock(nn.Module):
    def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(F.adaptive_avg_pool2d(value, 1).flatten(1))


class DepthComponent(nn.Module):
    def __init__(self, inputs: int, baseline: dict[str, Any]) -> None:
        super().__init__()
        self.block = SpatialBlock(inputs, 48, 3)
        self.register_buffer("baseline_log_scale", torch.tensor(float(baseline["depth_log_scale"])))
        self.register_buffer("baseline_log_sigma", torch.tensor(float(baseline["depth_log_sigma"])))
        self.register_buffer("baseline_valid_logit", torch.tensor(logit(baseline["depth_valid_probability"])))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.block(value)
        log_depth = self.baseline_log_scale + 2.0 * torch.tanh(raw[:, 0:1])
        log_sigma = (self.baseline_log_sigma + raw[:, 1:2]).clamp(math.log(0.005), math.log(5.0))
        valid = torch.sigmoid(self.baseline_valid_logit + raw[:, 2:3])
        activity = torch.tanh(raw[:, 0:1]).abs().mean(dim=(1, 2, 3))
        return log_depth, log_sigma, valid, activity


class ProbabilityComponent(nn.Module):
    def __init__(self, inputs: int, prior: float, hidden: int = 32) -> None:
        super().__init__()
        self.block = SpatialBlock(inputs, hidden, 1)
        self.register_buffer("baseline_logit", torch.tensor(logit(prior)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.baseline_logit + self.block(value))


class BoundaryComponent(nn.Module):
    def __init__(self, inputs: int, baseline: dict[str, Any]) -> None:
        super().__init__()
        self.block = SpatialBlock(inputs, 40, 2)
        self.register_buffer("baseline_logit", torch.tensor(logit(baseline["boundary_probability"])))
        self.register_buffer(
            "baseline_sigma_raw",
            torch.tensor(inverse_softplus(baseline["boundary_localization_sigma_px"])),
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.block(value)
        return (
            torch.sigmoid(self.baseline_logit + raw[:, 0:1]),
            F.softplus(self.baseline_sigma_raw + raw[:, 1:2]).clamp(0.05, 64.0),
        )


class NormalComponent(nn.Module):
    def __init__(self, inputs: int, baseline: dict[str, Any]) -> None:
        super().__init__()
        self.block = GlobalBlock(inputs, 64, 3)
        self.register_buffer(
            "baseline",
            torch.tensor(baseline["support_plane_normal_camera_xyz"], dtype=torch.float32),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.baseline[None] + self.block(value), dim=1, eps=1.0e-6)


class PositiveGlobalComponent(nn.Module):
    def __init__(self, inputs: int, baseline: float, *, softplus: bool) -> None:
        super().__init__()
        self.block = GlobalBlock(inputs, 64, 1)
        self.softplus = softplus
        initial = inverse_softplus(baseline) if softplus else math.log(baseline)
        self.register_buffer("baseline_raw", torch.tensor(float(initial)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        raw = self.baseline_raw + self.block(value)[:, 0]
        return F.softplus(raw).clamp(0.005, 5.0) if self.softplus else torch.exp(raw.clamp(math.log(0.3), math.log(3.0)))


class GlobalProbabilityComponent(nn.Module):
    def __init__(self, inputs: int, prior: float) -> None:
        super().__init__()
        self.block = GlobalBlock(inputs, 48, 1)
        self.register_buffer("baseline_logit", torch.tensor(logit(prior)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.baseline_logit + self.block(value)[:, 0])


class FactorSplitHead(nn.Module):
    def __init__(self, baseline: dict[str, Any]) -> None:
        super().__init__()
        inputs = DEPTHART_PYRAMID_CHANNELS + 1
        self.depth = DepthComponent(inputs, baseline)
        self.support_probability = ProbabilityComponent(inputs, baseline["support_probability"])
        self.obstacle = ProbabilityComponent(inputs, baseline["obstacle_evidence_probability"])
        self.boundary = BoundaryComponent(inputs, baseline)
        self.evidence_validity = ProbabilityComponent(inputs, baseline["evidence_valid_probability"], hidden=16)
        self.support_normal = NormalComponent(inputs, baseline)
        self.camera_height = PositiveGlobalComponent(inputs, baseline["camera_height_m"], softplus=False)
        self.support_uncertainty = PositiveGlobalComponent(
            inputs, baseline["support_residual_sigma_m"], softplus=True
        )
        self.support_validity = GlobalProbabilityComponent(inputs, baseline["support_valid_probability"])

    def forward(self, feature: torch.Tensor, base_depth_m: torch.Tensor) -> dict[str, torch.Tensor]:
        value = torch.cat([feature, base_depth_m.clamp(0.05, 20.0).log()], dim=1)
        log_depth, log_sigma, depth_valid, activity = self.depth(value)
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


def forward_sample(model: FactorSplitHead, sample: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        sample.feature[None].to(device=device, dtype=torch.float32),
        sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
    )


def normalized_regret(
    evaluation: dict[str, Any], baseline: dict[str, Any], metrics: tuple[str, ...]
) -> dict[str, float]:
    return {
        metric: (
            float(evaluation["overall_metrics"][metric])
            - float(baseline["overall_metrics"][metric])
        )
        / max(abs(float(baseline["overall_metrics"][metric])), 1.0e-8)
        for metric in metrics
    }


def choose_components(
    candidates: list[dict[str, Any]], baseline_eval: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    table: dict[str, Any] = {}
    for component, metrics in COMPONENT_METRICS.items():
        rows = []
        for candidate in candidates:
            regret = normalized_regret(candidate["evaluation"], baseline_eval, metrics)
            rows.append(
                {
                    "step": candidate["step"],
                    "checkpoint": candidate["checkpoint"],
                    "normalized_regret": regret,
                    "maximum_normalized_regret": max(regret.values()),
                    "no_worse": all(value <= 1.0e-4 for value in regret.values()),
                }
            )
        eligible = [row for row in rows if row["no_worse"]]
        require(eligible, f"component has no baseline-safe checkpoint: {component}")
        chosen = min(
            eligible,
            key=lambda row: (
                row["maximum_normalized_regret"],
                row["step"],
                row["checkpoint"]["sha256"],
            ),
        )
        selected[component] = chosen
        table[component] = rows
    return selected, table


def compose_checkpoint(
    model: FactorSplitHead,
    selected: dict[str, dict[str, Any]],
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    destination = model.state_dict()
    for component, row in selected.items():
        state = torch.load(row["checkpoint"]["path"], map_location=device, weights_only=True)["model"]
        for prefix in COMPONENT_PREFIXES[component]:
            for key, value in state.items():
                if key.startswith(prefix):
                    destination[key] = value
    # Validity components use the final candidate; they are diagnostics, not primary selectors.
    final_state = torch.load(selected["depth"]["checkpoint"]["path"], map_location=device, weights_only=True)["model"]
    for component in ("support_validity", "evidence_validity"):
        for prefix in COMPONENT_PREFIXES[component]:
            for key, value in final_state.items():
                if key.startswith(prefix):
                    destination[key] = value
    model.load_state_dict(destination, strict=True)
    path = output_dir / f"seed-{seed}-composite.pt"
    return save_checkpoint(path, model, seed, -1)


def train_seed(
    seed: int,
    fit: list[Any],
    selection: list[Any],
    baseline: dict[str, Any],
    normalization: dict[str, float],
    lock: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FactorSplitHead(baseline).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lock["training"]["learning_rate"]),
        weight_decay=float(lock["training"]["weight_decay"]),
    )
    total_steps = int(lock["training"]["optimizer_steps"])
    warmup = int(lock["training"]["warmup_steps"])
    schedule = [int(value) for value in lock["training"]["checkpoint_steps"]]
    rng = random.Random(seed)
    baseline_eval = evaluate(None, selection, baseline, device)
    candidates = []
    loss_trace = []

    def candidate(step: int) -> None:
        evaluation = evaluate(model, selection, baseline, device)
        receipt = save_checkpoint(output_dir / f"seed-{seed}-step-{step}.pt", model, seed, step)
        candidates.append({"step": step, "evaluation": evaluation, "checkpoint": receipt})

    candidate(0)
    model.train()
    for step in range(1, total_steps + 1):
        sample = fit[rng.randrange(len(fit))]
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_sample(model, sample, device)
        objective, losses = compute_losses(outputs, move_targets(sample.targets, device), normalization)
        require(bool(torch.isfinite(objective)), "Attempt-02 objective non-finite")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if step <= warmup:
            factor = step / warmup
        else:
            progress = (step - warmup) / (total_steps - warmup)
            factor = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = float(lock["training"]["learning_rate"]) * factor
        optimizer.step()
        if step == 1 or step % 100 == 0:
            loss_trace.append(
                {
                    "step": step,
                    "objective": float(objective.detach()),
                    "components": {key: float(value.detach()) for key, value in losses.items()},
                }
            )
        if step in schedule:
            model.eval()
            candidate(step)
            model.train()
    selected, component_table = choose_components(candidates, baseline_eval)
    composite = FactorSplitHead(baseline).to(device)
    composite_receipt = compose_checkpoint(composite, selected, seed, output_dir, device)
    composite_eval = evaluate(composite, selection, baseline, device)
    full_regret = normalized_regret(composite_eval, baseline_eval, PRIMARY_METRICS)
    eligible = all(value <= 1.0e-4 for value in full_regret.values()) and any(
        value < -1.0e-4 for value in full_regret.values()
    )
    return {
        "seed": seed,
        "selection_baseline": baseline_eval,
        "candidates": candidates,
        "component_selection_table": component_table,
        "selected_component_steps": {key: row["step"] for key, row in selected.items()},
        "composite_checkpoint": composite_receipt,
        "composite_selection_evaluation": composite_eval,
        "composite_normalized_regret": full_regret,
        "composite_eligible": eligible,
        "loss_trace": loss_trace,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA required")
    validation = validate_execution_lock(args.lock)
    require(validation["passed"], "Attempt-02 execution lock validation failed")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(lock["status"] == "ATTEMPT02_FACTOR_SPLIT_EXECUTION_AUTHORIZED", "Attempt-02 lock invalid")
    require(sha256_file(args.label_result) == EXPECTED_LABEL_RESULT_SHA256, "FIT label result drift")
    require(sha256_file(args.baseline_result) == EXPECTED_BASELINE_RESULT_SHA256, "baseline result drift")
    require(sha256_file(args.selection_result) == EXPECTED_SELECTION_RESULT_SHA256, "selection result drift")
    require(sha256_file(args.canary_result) == lock["bindings"]["canary_result_sha256"], "canary result drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART checkpoint drift")
    device = torch.device(args.device)
    fit_result = json.loads(args.label_result.read_text(encoding="utf-8"))
    selection_result = json.loads(args.selection_result.read_text(encoding="utf-8"))
    canary_result = json.loads(args.canary_result.read_text(encoding="utf-8"))
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    fit_rows = [{**row, "role": "FIT"} for row in fit_result["frames"] if row["role"] == "FIT"]
    selection_rows = [{**row, "role": "CHECKPOINT_SELECTION"} for row in selection_result["frames"]]
    canary_rows = [{**row, "role": "TRAIN_CANARY"} for row in canary_result["frames"]]
    pre_canary_rows = sorted(fit_rows + selection_rows, key=lambda row: row["sample_id"])
    require(
        len(pre_canary_rows) == 33
        and len({row["sample_id"] for row in pre_canary_rows}) == 33
        and len(canary_rows) == 6,
        "Attempt-02 roster drift",
    )
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    cached, pre_canary_feature_receipt = extract_features(
        pre_canary_rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    by_role = {
        role: [sample for sample in cached if sample.role == role]
        for role in ("FIT", "CHECKPOINT_SELECTION")
    }
    require(
        {key: len(value) for key, value in by_role.items()}
        == {"FIT": 27, "CHECKPOINT_SELECTION": 6},
        "Attempt-02 pre-canary role drift",
    )
    baseline = baseline_result["baseline_parameters"]
    normalization = baseline_result["optimizer_normalization"]
    seeds = [int(value) for value in lock["training"]["seeds"]]
    seed_results = []
    for seed in seeds:
        training = train_seed(
            seed,
            by_role["FIT"],
            by_role["CHECKPOINT_SELECTION"],
            baseline,
            normalization,
            lock,
            args.output_dir,
            device,
        )
        seed_results.append(training)
        torch.cuda.empty_cache()
    all_composites_eligible = all(row["composite_eligible"] for row in seed_results)
    canary_feature_receipt = None
    canary_samples: list[Any] = []
    if all_composites_eligible:
        canary_samples, canary_feature_receipt = extract_features(
            sorted(canary_rows, key=lambda row: row["sample_id"]),
            args.depthart_source,
            args.depthart_checkpoint,
            args.depthart_extension,
            device,
        )
        require(
            len(canary_samples) == 6
            and all(sample.role == "TRAIN_CANARY" for sample in canary_samples),
            "Attempt-02 canary role drift",
        )
        # Baseline truth is opened only after all three composites are selection-eligible.
        canary_baseline = evaluate(None, canary_samples, baseline, device)
        for row in seed_results:
            model = FactorSplitHead(baseline).to(device)
            state = torch.load(
                row["composite_checkpoint"]["path"],
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state["model"], strict=True)
            canary_eval = evaluate(model, canary_samples, baseline, device)
            row["canary_evaluation"] = canary_eval
            row["canary_baseline"] = canary_baseline
            row["canary_gate"] = canary_gate(canary_eval, canary_baseline, row["seed"])
            del model
    else:
        for row in seed_results:
            row["canary_evaluation"] = None
            row["canary_baseline"] = None
            row["canary_gate"] = None
    passed = all_composites_eligible and all(
        row["canary_gate"]["all_primary_metrics_passed"]
        and row["canary_gate"]["all_uncertainty_families_passed"]
        for row in seed_results
    )
    predictions = []
    if passed:
        model = FactorSplitHead(baseline).to(device)
        state = torch.load(seed_results[0]["composite_checkpoint"]["path"], map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        predictions = serialize_predictions(model, canary_samples, args.output_dir, device)
        del model
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_factor_learnability_attempt02_result_v1",
        "status": "R2_F1_FACTOR_LEARNABILITY_ATTEMPT02_PASS" if passed else "R2_F1_FACTOR_LEARNABILITY_ATTEMPT02_FAIL_STOP",
        "passed": passed,
        "execution_lock": str(args.lock.resolve()),
        "execution_lock_sha256": sha256_file(args.lock),
        "feature_receipt": {
            "fit_and_selection": pre_canary_feature_receipt,
            "canary_after_all_composites_eligible": canary_feature_receipt,
        },
        "role_frame_counts": {
            "FIT": len(by_role["FIT"]),
            "CHECKPOINT_SELECTION": len(by_role["CHECKPOINT_SELECTION"]),
            "TRAIN_CANARY": len(canary_samples),
        },
        "seed_results": seed_results,
        "prediction_receipts_seed17": predictions,
        "decision": {
            "all_seeds_passed": passed,
            "all_composites_selection_eligible_before_canary_open": all_composites_eligible,
            "old_attempt01_canary_used": False,
            "reducer_or_task_outcome_read": False,
            "next_action_if_pass": "Execute the preexisting FactorTensorAdapter on the serialized real factor tensors, then the deterministic reducer seam canary.",
        },
    }
    result_path = args.output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--label-result", type=Path, default=DEFAULT_LABEL_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--selection-result", type=Path, default=DEFAULT_SELECTION_RESULT)
    parser.add_argument("--canary-result", type=Path, default=DEFAULT_CANARY_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "lock",
        "label_result",
        "baseline_result",
        "selection_result",
        "canary_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "component_steps": {
                    str(row["seed"]): row["selected_component_steps"]
                    for row in result["seed_results"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
