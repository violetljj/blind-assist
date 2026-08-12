#!/usr/bin/env python3
"""Train the final camera-height factor on expanded source-native FIT labels."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt08_depthart_residual_head import DepthArtResidualFactorHead  # noqa: E402
from train_ag_r2_f1_attempt09_expanded_residual_depth import expanded_rows  # noqa: E402
from train_ag_r2_f1_factor_learnability import evaluate, extract_features, move_targets, save_checkpoint  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    bootstrap_lower,
    require,
    sha256_file,
)


RESIDUAL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-expanded-residual-depth-r2/result.json"
EXPECTED_RESIDUAL_RESULT_SHA256 = "9B022091A409D821AE01779FD8E5266A31C619CE1739A85CAC93166E82F37FAE"
CANONICAL_CHECKPOINT_SHA256 = "517C2CDDE33674CB94E33808C12B263527AF17E6CB260368F009FEB1DD9B9936"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt10-camera-height-r0"
TRAINING_SEED = 1029


def height_evidence(evaluation: dict[str, Any], baseline: dict[str, Any], seed: int) -> dict[str, Any]:
    metric = "camera_height_abs_log_error"
    improvements = [
        float(baseline["parent_metrics"][parent][metric])
        - float(evaluation["parent_metrics"][parent][metric])
        for parent in sorted(baseline["parent_metrics"])
    ]
    favorable = float(np.mean(np.asarray(improvements) > 0.0))
    lower = bootstrap_lower(improvements, seed)
    overall_improvement = float(baseline["overall_metrics"][metric] - evaluation["overall_metrics"][metric])
    passed = lower > 0.0 and favorable >= 0.75
    return {
        "parent_improvements": improvements,
        "favorable_parent_fraction": favorable,
        "bootstrap_95_lower": lower,
        "overall_improvement": overall_improvement,
        "passed": passed,
        "selection_score": -lower - 0.25 * overall_improvement + (0.0 if passed else 1.0e6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.residual_result) == EXPECTED_RESIDUAL_RESULT_SHA256, "residual result drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    device = torch.device(args.device)
    rows, data_receipt = expanded_rows()
    residual = json.loads(args.residual_result.read_text(encoding="utf-8"))
    fit_parents = set(residual["fit_parents"])
    validation_parents = set(residual["internal_validation_parents"])
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    fit = [sample for sample in samples if sample.parent_id in fit_parents]
    validation = [sample for sample in samples if sample.parent_id in validation_parents]
    require(len(fit) == 123 and len(validation) == 15, "camera height split drift")
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline_parameters = baseline_result["baseline_parameters"]
    validation_baseline = evaluate(None, validation, baseline_parameters, device)
    checkpoint = Path(residual["canonical_checkpoint"]["path"])
    require(sha256_file(checkpoint) == CANONICAL_CHECKPOINT_SHA256, "canonical residual checkpoint drift")
    model = DepthArtResidualFactorHead(baseline_parameters).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.camera_height.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.camera_height.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    fit_by_parent: dict[str, list[Any]] = defaultdict(list)
    for sample in fit:
        if bool(sample.targets["support_plane_valid"]):
            fit_by_parent[sample.parent_id].append(sample)
    require(len(fit_by_parent) >= 18, "camera height FIT parent coverage insufficient")
    parents = sorted(fit_by_parent)
    rng = random.Random(TRAINING_SEED)
    trace = []
    checkpoints = []

    def capture(step: int) -> None:
        model.eval()
        evaluation = evaluate(model, validation, baseline_parameters, device)
        receipt = save_checkpoint(args.output_dir / f"camera-height-step-{step}.pt", model, TRAINING_SEED, step)
        checkpoints.append({"step": step, "checkpoint": receipt, "evaluation": evaluation, "evidence": height_evidence(evaluation, validation_baseline, TRAINING_SEED + step)})

    capture(0)
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        parent = parents[rng.randrange(len(parents))]
        sample = fit_by_parent[parent][rng.randrange(len(fit_by_parent[parent]))]
        feature = sample.feature[None].to(device=device, dtype=torch.float32)
        base = sample.base_depth_feature[None].to(device=device, dtype=torch.float32)
        if rng.random() < 0.5:
            feature = torch.flip(feature, dims=(-1,))
            base = torch.flip(base, dims=(-1,))
        targets = move_targets(sample.targets, device)
        predicted = model(feature, base)["camera_height_m"][0].clamp_min(0.1)
        target = targets["camera_height_m"].clamp_min(0.1)
        log_error = predicted.log() - target.log()
        loss = torch.where(log_error.abs() <= 0.10, 0.5 * log_error.square() / 0.10, log_error.abs() - 0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.camera_height.parameters(), 5.0)
        progress = step / args.optimizer_steps
        optimizer.param_groups[0]["lr"] = 1.0e-3 * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append({"step": step, "loss": float(loss.detach()), "predicted_height_m": float(predicted.detach()), "target_height_m": float(target.detach()), "parent_id": parent})
        if step % 200 == 0 or step == args.optimizer_steps:
            capture(step)
    selected = min(checkpoints, key=lambda row: (row["evidence"]["selection_score"], row["step"], row["checkpoint"]["sha256"]))
    passed = bool(selected["evidence"]["passed"])
    result = {
        "schema": "blindassist_ag_r2_f1_attempt10_camera_height_result_v1",
        "status": "ATTEMPT10_CAMERA_HEIGHT_INTERNAL_PASS_FINAL_CANARY_LOCK_REQUIRED" if passed else "ATTEMPT10_CAMERA_HEIGHT_INTERNAL_FAIL_NO_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "preserved_canary_metrics_opened": False,
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "fit_parent_count": len(fit_by_parent),
        "fit_frame_count": sum(len(value) for value in fit_by_parent.values()),
        "validation_parent_count": len(validation_parents),
        "validation_frame_count": len(validation),
        "source_residual_checkpoint": {"path": str(checkpoint.resolve()), "sha256": CANONICAL_CHECKPOINT_SHA256},
        "checkpoints": checkpoints,
        "selected_step": selected["step"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_evaluation": selected["evaluation"],
        "selected_evidence": selected["evidence"],
        "loss_trace": trace,
        "decision": {"only_camera_height_parameters_changed": True, "preserved_canary_metrics_opened": False, "next_action_if_pass": "Bind learned height with frozen final factor calibration and execute the preserved canary once."},
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-result", type=Path, default=RESIDUAL_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer-steps", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "selected_step": result["selected_step"], "evidence": result["selected_evidence"], "checkpoint": result["selected_checkpoint"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
