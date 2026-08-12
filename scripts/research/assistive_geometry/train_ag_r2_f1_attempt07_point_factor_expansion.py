#!/usr/bin/env python3
"""Expand point-factor training to all consumed AG-F1 parents.

The still-sealed Attempt-05 canary is excluded. Sensor-conditioned normal,
height, and support sigma remain outside this point-head optimization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt05_uncertainty_calibrators import _load_consumed_rows  # noqa: E402
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    compute_losses,
    evaluate,
    extract_features,
    move_targets,
    save_checkpoint,
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
    require,
    sha256_file,
)


ATTEMPT05_LABEL_RESULT = (
    "artifacts.local/experiments/ag-r2-f1-attempt05-fresh-ag-held-labels-r0/result.json",
    "4DBF0E85F45357C613221DF9F2C5A5E3B0971C314EB29D1967C02E0D6FAEB7CC",
)
ATTEMPT06_LABEL_RESULT = (
    "artifacts.local/experiments/ag-r2-f1-attempt06-fresh-selection-labels-r0/result.json",
    "F67A9A000A4A82C180B9E875DEE976C80E00584AB8A6CEBCC603DFAAEE1E90A5",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-point-factor-expansion-r0"
SPLIT_TOKEN = "AG_R2_F1_ATTEMPT07_POINT_FACTOR_EXPANSION_2026-08-11"
POINT_METRICS = (
    "depth_shape_abs_log_error",
    "depth_scale_abs_log_error",
    "support_brier",
    "obstacle_brier",
    "boundary_distance_abs_error_px",
)
FROZEN_COMPONENTS = ("support_normal", "camera_height", "support_uncertainty")


def load_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows, base_receipts = _load_consumed_rows()
    attempt05_path = REPO_ROOT / ATTEMPT05_LABEL_RESULT[0]
    attempt06_path = REPO_ROOT / ATTEMPT06_LABEL_RESULT[0]
    require(sha256_file(attempt05_path) == ATTEMPT05_LABEL_RESULT[1], "Attempt-05 labels drift")
    require(sha256_file(attempt06_path) == ATTEMPT06_LABEL_RESULT[1], "Attempt-06 labels drift")
    attempt05 = json.loads(attempt05_path.read_text(encoding="utf-8"))
    attempt06 = json.loads(attempt06_path.read_text(encoding="utf-8"))
    attempt05_selection = [row for row in attempt05["frames"] if row["role"] == "CHECKPOINT_SELECTION"]
    preserved_canary = [row for row in attempt05["frames"] if row["role"] == "TRAIN_CANARY"]
    require(len(attempt05_selection) == len(preserved_canary) == 6, "Attempt-05 role drift")
    additional = attempt05_selection + list(attempt06["frames"])
    rows_by_sample = {row["sample_id"]: {**row, "role": "CONSUMED_POINT_TRAINING"} for row in base_rows}
    for row in additional:
        require(row["sample_id"] not in rows_by_sample, "Attempt-07 duplicate sample")
        rows_by_sample[row["sample_id"]] = {**row, "role": "CONSUMED_POINT_TRAINING"}
    forbidden_samples = {row["sample_id"] for row in preserved_canary}
    require(not (forbidden_samples & set(rows_by_sample)), "preserved canary leaked into point training")
    rows = sorted(rows_by_sample.values(), key=lambda row: row["sample_id"])
    require(len(rows) == 75 and len({row["parent_id"] for row in rows}) == 25, "Attempt-07 roster drift")
    return rows, {
        "base_consumed_label_receipts": base_receipts,
        "attempt05_consumed_selection": {
            "path": ATTEMPT05_LABEL_RESULT[0],
            "sha256": ATTEMPT05_LABEL_RESULT[1],
            "parents": sorted({row["parent_id"] for row in attempt05_selection}),
        },
        "attempt06_consumed_selection": {
            "path": ATTEMPT06_LABEL_RESULT[0],
            "sha256": ATTEMPT06_LABEL_RESULT[1],
            "parents": sorted({row["parent_id"] for row in attempt06["frames"]}),
        },
        "preserved_canary_excluded": {
            "path": ATTEMPT05_LABEL_RESULT[0],
            "sha256": ATTEMPT05_LABEL_RESULT[1],
            "parents": sorted({row["parent_id"] for row in preserved_canary}),
            "sample_ids": sorted(row["sample_id"] for row in preserved_canary),
        },
    }


def parent_split(parents: set[str]) -> tuple[list[str], list[str]]:
    forced_fit = {
        "rgbd_dataset_freiburg1_xyz",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg1_desk",
        "rgbd_dataset_freiburg1_rpy",
    }
    require(forced_fit <= parents, "failed-selection parent missing")
    remaining = sorted(
        parents - forced_fit,
        key=lambda parent: hashlib.sha256(f"{SPLIT_TOKEN}:{parent}".encode("utf-8")).hexdigest(),
    )
    validation = remaining[-5:]
    fit = sorted(parents - set(validation))
    require(len(fit) == 20 and len(validation) == 5, "Attempt-07 parent split drift")
    return fit, validation


def candidate_evidence(
    evaluation: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    metrics = {}
    normalized_regrets = []
    for metric in POINT_METRICS:
        parent_improvements = [
            float(baseline["parent_metrics"][parent][metric])
            - float(evaluation["parent_metrics"][parent][metric])
            for parent in sorted(baseline["parent_metrics"])
        ]
        overall_base = float(baseline["overall_metrics"][metric])
        overall_model = float(evaluation["overall_metrics"][metric])
        normalized_regret = (overall_model - overall_base) / max(abs(overall_base), 1.0e-8)
        normalized_regrets.append(normalized_regret)
        metrics[metric] = {
            "overall_improvement": overall_base - overall_model,
            "normalized_regret": normalized_regret,
            "parent_improvements": parent_improvements,
            "favorable_parent_fraction": float(np.mean(np.asarray(parent_improvements) > 0.0)),
        }
    eligible = all(
        row["overall_improvement"] > 0.0 and row["favorable_parent_fraction"] >= 0.60
        for row in metrics.values()
    )
    score = max(normalized_regrets) + 0.25 * float(np.mean(normalized_regrets)) + (0.0 if eligible else 1.0e6)
    return {"metrics": metrics, "eligible": eligible, "selection_score": score}


def freeze_sensor_global_components(model: FactorSplitHead) -> None:
    for name in FROZEN_COMPONENTS:
        module = getattr(model, name)
        module.requires_grad_(False)


def training_forward(
    model: FactorSplitHead,
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
            key: (
                torch.flip(value, dims=(-1,))
                if value.ndim >= 2 and value.shape[-1] == feature.shape[-1]
                else value
            )
            for key, value in targets.items()
        }
    return model(feature, base), targets


def train_seed(
    seed: int,
    fit: list[Any],
    validation: list[Any],
    baseline: dict[str, Any],
    normalization: dict[str, float],
    source_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FactorSplitHead(baseline).to(device)
    model.load_state_dict(torch.load(source_checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    freeze_sensor_global_components(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5.0e-4,
        weight_decay=1.0e-4,
    )
    validation_baseline = evaluate(None, validation, baseline, device)
    rng = random.Random(seed + 7000)
    checkpoints = []
    loss_trace = []

    def capture(step: int) -> None:
        model.eval()
        evaluation = evaluate(model, validation, baseline, device)
        receipt = save_checkpoint(output_dir / f"seed-{seed}-step-{step}.pt", model, seed, step)
        checkpoints.append({"step": step, "checkpoint": receipt, "evaluation": evaluation, "evidence": candidate_evidence(evaluation, validation_baseline)})

    capture(0)
    for step in range(1, steps + 1):
        model.train()
        sample = fit[rng.randrange(len(fit))]
        outputs, targets = training_forward(model, sample, device, rng.random() < 0.5)
        objective, losses = compute_losses(outputs, targets, normalization)
        require(bool(torch.isfinite(objective)), "Attempt-07 objective non-finite")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        progress = step / steps
        optimizer.param_groups[0]["lr"] = 5.0e-4 * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        optimizer.step()
        if step == 1 or step % 100 == 0:
            loss_trace.append({"step": step, "objective": float(objective.detach()), "components": {key: float(value.detach()) for key, value in losses.items()}})
        if step % 300 == 0 or step == steps:
            capture(step)
    selected = min(
        checkpoints,
        key=lambda row: (
            row["evidence"]["selection_score"],
            row["step"],
            row["checkpoint"]["sha256"],
        ),
    )
    return {
        "seed": seed,
        "source_checkpoint": {"path": str(source_checkpoint.resolve()), "sha256": sha256_file(source_checkpoint)},
        "validation_baseline": validation_baseline,
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
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    rows, data_receipt = load_rows()
    parents = {str(row["parent_id"]) for row in rows}
    fit_parents, validation_parents = parent_split(parents)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()
    samples, feature_receipt = extract_features(
        rows, args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device
    )
    fit = [sample for sample in samples if sample.parent_id in fit_parents]
    validation = [sample for sample in samples if sample.parent_id in validation_parents]
    require(len(fit) == 60 and len(validation) == 15, "Attempt-07 frame split drift")
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline = baseline_result["baseline_parameters"]
    normalization = baseline_result["optimizer_normalization"]
    seed_results = []
    for seed_row in attempt02["seed_results"]:
        seed = int(seed_row["seed"])
        source_checkpoint = Path(seed_row["composite_checkpoint"]["path"])
        require(sha256_file(source_checkpoint) == EXPECTED_COMPOSITES[seed], f"source checkpoint drift: {seed}")
        result = train_seed(
            seed,
            fit,
            validation,
            baseline,
            normalization,
            source_checkpoint,
            args.output_dir,
            device,
            args.optimizer_steps,
        )
        seed_results.append(result)
        print(json.dumps({"seed": seed, "selected_step": result["selected_step"], "eligible": result["selected_evidence"]["eligible"], "selection_score": result["selected_evidence"]["selection_score"]}), flush=True)
        torch.cuda.empty_cache()
    passed = all(row["selected_evidence"]["eligible"] for row in seed_results)
    result = {
        "schema": "blindassist_ag_r2_f1_attempt07_point_factor_expansion_result_v1",
        "status": (
            "ATTEMPT07_POINT_FACTOR_EXPANSION_INTERNAL_PASS_UNCERTAINTY_RECALIBRATION_REQUIRED"
            if passed
            else "ATTEMPT07_POINT_FACTOR_EXPANSION_INTERNAL_FAIL_NO_CANARY"
        ),
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch": torch.__version__,
        "data_receipt": data_receipt,
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "role_frame_counts": {"FIT": len(fit), "INTERNAL_VALIDATION": len(validation), "PRESERVED_CANARY": 0},
        "feature_receipt": feature_receipt,
        "point_metrics": list(POINT_METRICS),
        "frozen_components": list(FROZEN_COMPONENTS),
        "seed_results": seed_results,
        "decision": {
            "preserved_canary_metrics_opened": False,
            "sensor_geometry_or_support_sigma_changed": False,
            "next_action_if_pass": "Recalibrate uncertainty against the expanded point checkpoints, freeze one final canary lock, then open the preserved canary once.",
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
    parser.add_argument("--optimizer-steps", type=int, default=1800)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "seeds": [{"seed": row["seed"], "selected_step": row["selected_step"], "eligible": row["selected_evidence"]["eligible"], "metrics": row["selected_evidence"]["metrics"]} for row in result["seed_results"]]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
