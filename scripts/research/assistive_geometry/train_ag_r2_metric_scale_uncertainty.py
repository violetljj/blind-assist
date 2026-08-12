#!/usr/bin/env python3
"""Train a factor-only per-frame global metric-scale uncertainty head."""

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
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_factor_learnability import extract_features  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    require,
    sha256_file,
)
from train_ag_r2_metric_depth_student import MetricDepthStudentHead  # noqa: E402


DEFAULT_CORPUS_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-multisource-distillation-corpus-r0/result.json"
)
EXPECTED_CORPUS_RESULT_SHA256 = (
    "0D948F8D582F132BD941CAFBDBC7E60E8C11D9C40CC301B0A0AD4F59F369CD6E"
)
DEFAULT_METRIC_STUDENT_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-multisource-metric-depth-student-r0/result.json"
)
EXPECTED_METRIC_STUDENT_RESULT_SHA256 = (
    "F0703357B0F25C7ABF209EE53DE9B04E588BEDE3629C1B1F5273D9E31D41BFF3"
)
EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256 = (
    "980B26D16659BF1AAF47C64C5CBAC63A5E91D60573E93AD23190DFF3BB67E4B7"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-metric-scale-uncertainty-r0"
)
TRAINING_SEED = 2026081203
SIGMA_FLOOR = 0.02
SIGMA_CAP = 1.00
CALIBRATION_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00)


class MetricScaleUncertaintyHead(nn.Module):
    """Predict global log-scale sigma from frozen deployable features."""

    def __init__(self, input_dim: int = 390, hidden: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        initial = 0.25 - SIGMA_FLOOR
        nn.init.constant_(self.net[-1].bias, math.log(math.expm1(initial)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        sigma = SIGMA_FLOOR + F.softplus(self.net(value)[:, 0])
        return sigma.clamp(SIGMA_FLOOR, SIGMA_CAP)


def pooled_uncertainty_feature(
    feature: torch.Tensor,
    base_depth: torch.Tensor,
    predicted_log_depth: torch.Tensor,
) -> torch.Tensor:
    feature = feature.float()
    base_log = base_depth.float().clamp(0.05, 20.0).log()
    predicted = predicted_log_depth.float()
    correction = predicted - base_log
    value = torch.cat(
        [
            feature.mean(dim=(-2, -1)),
            feature.std(dim=(-2, -1), unbiased=False),
            base_log.mean(dim=(-2, -1)),
            base_log.std(dim=(-2, -1), unbiased=False),
            predicted.mean(dim=(-2, -1)),
            predicted.std(dim=(-2, -1), unbiased=False),
            correction.mean(dim=(-2, -1)),
            correction.std(dim=(-2, -1), unbiased=False),
        ],
        dim=1,
    )
    require(value.shape[1] == 390, "uncertainty feature dimension drift")
    return value


def load_metric_student(
    result: dict[str, Any], device: torch.device
) -> tuple[MetricDepthStudentHead, Path]:
    checkpoint = Path(result["checkpoint"]["path"])
    require(
        sha256_file(checkpoint) == EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256,
        "multi-source metric student checkpoint drift",
    )
    model = MetricDepthStudentHead(
        hidden=int(result["architecture"]["hidden_channels"]),
        global_hidden=int(result["architecture"]["global_hidden_channels"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, checkpoint


def build_examples(
    samples: list[Any],
    metric_model: MetricDepthStudentHead,
    device: torch.device,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            feature = sample.feature[None].to(device=device, dtype=torch.float32)
            base_depth = sample.base_depth_feature[None].to(
                device=device, dtype=torch.float32
            )
            output = metric_model(feature, base_depth)
            predicted = output["predicted_log_depth"]
            target_depth = sample.targets["metric_depth_m"][None].to(device)
            valid = sample.targets["metric_valid"][None].to(device).bool()
            require(bool(valid.any()), "metric scale residual denominator empty")
            residual = float(
                (predicted - target_depth.clamp_min(0.05).log())[valid].mean()
            )
            pooled = pooled_uncertainty_feature(feature, base_depth, predicted)
            examples.append(
                {
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "role": sample.role,
                    "feature": pooled[0].detach().cpu(),
                    "signed_scale_residual": residual,
                    "absolute_scale_residual": abs(residual),
                }
            )
    return examples


def gaussian_nll(residual: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp(SIGMA_FLOOR, SIGMA_CAP)
    return 0.5 * (residual / sigma).square() + sigma.log()


def evaluate_examples(
    model: MetricScaleUncertaintyHead,
    examples: list[dict[str, Any]],
    device: torch.device,
    multiplier: float,
) -> dict[str, Any]:
    require(bool(examples), "uncertainty evaluation roster empty")
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for example in examples:
            value = example["feature"][None].to(device=device, dtype=torch.float32)
            raw_sigma = float(model(value)[0])
            sigma = float(np.clip(raw_sigma * multiplier, SIGMA_FLOOR, SIGMA_CAP))
            residual = float(example["signed_scale_residual"])
            nll = 0.5 * (residual / sigma) ** 2 + math.log(sigma)
            rows.append(
                {
                    "sample_id": example["sample_id"],
                    "parent_id": example["parent_id"],
                    "role": example["role"],
                    "signed_scale_residual": residual,
                    "absolute_scale_residual": abs(residual),
                    "raw_predicted_sigma": raw_sigma,
                    "calibrated_predicted_sigma": sigma,
                    "covered_at_one_sigma": abs(residual) <= sigma,
                    "gaussian_nll": nll,
                }
            )
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)
    parent_metrics = {
        parent: {
            "frame_count": len(parent_rows),
            "gaussian_nll": float(
                np.mean([row["gaussian_nll"] for row in parent_rows])
            ),
            "one_sigma_coverage": float(
                np.mean([row["covered_at_one_sigma"] for row in parent_rows])
            ),
            "mean_predicted_sigma": float(
                np.mean([row["calibrated_predicted_sigma"] for row in parent_rows])
            ),
            "mean_absolute_scale_residual": float(
                np.mean([row["absolute_scale_residual"] for row in parent_rows])
            ),
        }
        for parent, parent_rows in sorted(by_parent.items())
    }
    metric_names = (
        "gaussian_nll",
        "one_sigma_coverage",
        "mean_predicted_sigma",
        "mean_absolute_scale_residual",
    )
    parent_macro = {
        name: float(np.mean([values[name] for values in parent_metrics.values()]))
        for name in metric_names
    }
    return {
        "frame_count": len(rows),
        "parent_count": len(parent_metrics),
        "multiplier": float(multiplier),
        "parent_macro_metrics": parent_macro,
        "parent_metrics": parent_metrics,
        "frames": rows,
    }


def choose_multiplier(
    model: MetricScaleUncertaintyHead,
    examples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        evaluate_examples(model, examples, device, multiplier)
        for multiplier in CALIBRATION_MULTIPLIERS
    ]
    selected = min(
        candidates,
        key=lambda row: (
            row["parent_macro_metrics"]["gaussian_nll"],
            row["multiplier"],
        ),
    )
    return selected, candidates


def save_checkpoint(
    path: Path,
    model: MetricScaleUncertaintyHead,
    step: int,
    multiplier: float,
) -> dict[str, Any]:
    torch.save(
        {
            "schema": "blindassist_ag_r2_metric_scale_uncertainty_checkpoint_v1",
            "step": step,
            "seed": TRAINING_SEED,
            "input_dim": model.input_dim,
            "calibration_multiplier": float(multiplier),
            "model": model.state_dict(),
        },
        path,
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "step": step,
        "calibration_multiplier": float(multiplier),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(
        torch.cuda.is_available() and str(args.device).startswith("cuda"),
        "CUDA required",
    )
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(
        sha256_file(args.corpus_result) == EXPECTED_CORPUS_RESULT_SHA256,
        "multi-source corpus drift",
    )
    require(
        sha256_file(args.metric_student_result)
        == EXPECTED_METRIC_STUDENT_RESULT_SHA256,
        "multi-source metric student result drift",
    )
    require(
        sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256,
        "DepthART drift",
    )
    corpus = json.loads(args.corpus_result.read_text(encoding="utf-8"))
    metric_result = json.loads(args.metric_student_result.read_text(encoding="utf-8"))
    require(corpus["passed"] and metric_result["passed"], "prerequisite failed")
    rows = sorted(
        [
            row
            for row in corpus["frames"]
            if int(row["metric_depth_valid_pixels"]) > 0
        ],
        key=lambda row: str(row["sample_id"]),
    )
    require(len(rows) == 179, "uncertainty corpus roster drift")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    metric_model, metric_checkpoint = load_metric_student(metric_result, device)
    examples = build_examples(samples, metric_model, device)
    del metric_model
    torch.cuda.empty_cache()
    by_role = {
        role: [row for row in examples if row["role"] == role]
        for role in ("FIT", "CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    require(
        {role: len(values) for role, values in by_role.items()}
        == {"FIT": 131, "CHECKPOINT_SELECTION": 24, "TRAIN_CANARY": 24},
        "uncertainty role split drift",
    )
    role_parents = {
        role: {row["parent_id"] for row in values}
        for role, values in by_role.items()
    }
    require(
        role_parents["FIT"].isdisjoint(role_parents["CHECKPOINT_SELECTION"])
        and role_parents["FIT"].isdisjoint(role_parents["TRAIN_CANARY"])
        and role_parents["CHECKPOINT_SELECTION"].isdisjoint(
            role_parents["TRAIN_CANARY"]
        ),
        "uncertainty parent overlap",
    )

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    model = MetricScaleUncertaintyHead(390, args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-3
    )
    fit_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in by_role["FIT"]:
        fit_by_parent[str(example["parent_id"])].append(example)
    fit_parents = sorted(fit_by_parent)
    rng = random.Random(TRAINING_SEED)
    candidates: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    def capture(step: int) -> None:
        selected_multiplier, multiplier_candidates = choose_multiplier(
            model, by_role["CHECKPOINT_SELECTION"], device
        )
        checkpoint = save_checkpoint(
            args.output_dir / f"scale-uncertainty-step-{step}.pt",
            model,
            step,
            float(selected_multiplier["multiplier"]),
        )
        row = {
            "step": step,
            "checkpoint": checkpoint,
            "selected_multiplier": float(selected_multiplier["multiplier"]),
            "selection": selected_multiplier,
            "multiplier_candidates": multiplier_candidates,
            "score": selected_multiplier["parent_macro_metrics"]["gaussian_nll"],
        }
        candidates.append(row)
        print(
            json.dumps(
                {
                    "step": step,
                    "multiplier": row["selected_multiplier"],
                    "selection_parent_macro_nll": row["score"],
                    "selection_one_sigma_coverage": selected_multiplier[
                        "parent_macro_metrics"
                    ]["one_sigma_coverage"],
                }
            ),
            flush=True,
        )

    capture(0)
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        parent = fit_parents[rng.randrange(len(fit_parents))]
        example = fit_by_parent[parent][rng.randrange(len(fit_by_parent[parent]))]
        value = example["feature"][None].to(device=device, dtype=torch.float32)
        residual = torch.tensor(
            [float(example["signed_scale_residual"])], device=device
        )
        sigma = model(value)
        loss = gaussian_nll(residual, sigma).mean()
        require(bool(torch.isfinite(loss)), "uncertainty objective non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        progress = step / args.optimizer_steps
        optimizer.param_groups[0]["lr"] = args.learning_rate * (
            0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append(
                {
                    "step": step,
                    "parent_id": parent,
                    "signed_scale_residual": float(residual[0]),
                    "raw_predicted_sigma": float(sigma[0].detach()),
                    "gaussian_nll": float(loss.detach()),
                }
            )
        if step % args.checkpoint_interval == 0 or step == args.optimizer_steps:
            capture(step)

    selected = min(candidates, key=lambda row: (row["score"], row["step"]))
    state = torch.load(
        selected["checkpoint"]["path"], map_location=device, weights_only=True
    )
    model.load_state_dict(state["model"], strict=True)
    multiplier = float(selected["selected_multiplier"])
    final_checkpoint = save_checkpoint(
        args.output_dir / "metric-scale-uncertainty.pt",
        model,
        int(selected["step"]),
        multiplier,
    )
    fit_evaluation = evaluate_examples(
        model, by_role["FIT"], device, multiplier
    )
    selection_evaluation = evaluate_examples(
        model, by_role["CHECKPOINT_SELECTION"], device, multiplier
    )
    # Held TRAIN_CANARY is opened exactly once after checkpoint and multiplier selection.
    canary_evaluation = evaluate_examples(
        model, by_role["TRAIN_CANARY"], device, multiplier
    )
    all_metrics = [
        value
        for evaluation in (fit_evaluation, selection_evaluation, canary_evaluation)
        for value in evaluation["parent_macro_metrics"].values()
    ]
    finite = all(math.isfinite(float(value)) for value in all_metrics)
    fit_parent_metrics = fit_evaluation["parent_metrics"]
    gates = {
        "SCALEUNC_C01_EXACT_CORPUS_STUDENT_AND_DEPTHART": True,
        "SCALEUNC_C02_PARENT_DISJOINT_131_24_24": True,
        "SCALEUNC_C03_FIT_ONLY_HEAD_OPTIMIZATION": True,
        "SCALEUNC_C04_SELECTION_ONLY_CHECKPOINT_AND_MULTIPLIER": True,
        "SCALEUNC_C05_TRAIN_CANARY_OPENED_ONCE_AFTER_SELECTION": True,
        "SCALEUNC_C06_NONZERO_HEAD_CHECKPOINT_SELECTED": bool(
            selected["step"] > 0
        ),
        "SCALEUNC_C07_FINITE_POSITIVE_UNCERTAINTY": bool(
            finite
            and all(
                SIGMA_FLOOR
                <= row["calibrated_predicted_sigma"]
                <= SIGMA_CAP
                for evaluation in (
                    fit_evaluation,
                    selection_evaluation,
                    canary_evaluation,
                )
                for row in evaluation["frames"]
            )
        ),
        "SCALEUNC_C08_CONSUMED_ICL_ABSTAINS_WIDER_THAN_REAL_TUM": bool(
            fit_parent_metrics["icl_living_room_kt1"]["mean_predicted_sigma"]
            > fit_parent_metrics["rgbd_dataset_freiburg3_sitting_static"][
                "mean_predicted_sigma"
            ]
        ),
        "SCALEUNC_C09_FACTOR_ONLY_NO_TASK_OR_REDUCER_OUTPUT": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_metric_scale_uncertainty_result_v1",
        "status": "AG_R2_METRIC_SCALE_UNCERTAINTY_PASS_READY_FOR_V2_SEAM"
        if passed
        else "AG_R2_METRIC_SCALE_UNCERTAINTY_FAIL",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "corpus": {
            "path": str(args.corpus_result.resolve()),
            "sha256": EXPECTED_CORPUS_RESULT_SHA256,
        },
        "metric_depth_student": {
            "result": str(args.metric_student_result.resolve()),
            "result_sha256": EXPECTED_METRIC_STUDENT_RESULT_SHA256,
            "checkpoint": str(metric_checkpoint.resolve()),
            "checkpoint_sha256": EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256,
        },
        "feature_receipt": feature_receipt,
        "roles": {
            role: {
                "frame_count": len(values),
                "parents": sorted(role_parents[role]),
            }
            for role, values in by_role.items()
        },
        "architecture": {
            "input": "pooled frozen DepthART features plus base/predicted/correction log-depth statistics",
            "input_dim": 390,
            "hidden_channels": args.hidden_channels,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "output": "per-frame global metric-scale relative sigma",
            "sigma_floor": SIGMA_FLOOR,
            "sigma_cap": SIGMA_CAP,
            "learned_final_task_head": False,
        },
        "training": {
            "seed": TRAINING_SEED,
            "optimizer_steps": args.optimizer_steps,
            "checkpoint_interval": args.checkpoint_interval,
            "learning_rate": args.learning_rate,
            "parent_balanced_sampling": True,
            "objective": "Gaussian NLL of signed per-frame log metric-scale residual",
            "trace": trace,
        },
        "selection_candidates": candidates,
        "selected_step": selected["step"],
        "selected_calibration_multiplier": multiplier,
        "checkpoint": final_checkpoint,
        "fit_evaluation": fit_evaluation,
        "selection_evaluation": selection_evaluation,
        "canary_evaluation": canary_evaluation,
        "gates": gates,
        "decision": {
            "per_frame_metric_scale_uncertainty_learned": True,
            "local_shape_uncertainty_separate": True,
            "task_or_reducer_output_used_for_training_or_selection": False,
            "next_action": "Serialize scale sigma separately from local shape sigma through FactorTensorAdapter v2, then run a consumed seam before opening the third real parent.",
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-result", type=Path, default=DEFAULT_CORPUS_RESULT)
    parser.add_argument(
        "--metric-student-result",
        type=Path,
        default=DEFAULT_METRIC_STUDENT_RESULT,
    )
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument(
        "--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer-steps", type=int, default=2400)
    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    args = parser.parse_args()
    for name in (
        "corpus_result",
        "metric_student_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "selected_step": result["selected_step"],
                "selected_calibration_multiplier": result[
                    "selected_calibration_multiplier"
                ],
                "checkpoint": result["checkpoint"],
                "selection": result["selection_evaluation"][
                    "parent_macro_metrics"
                ],
                "canary": result["canary_evaluation"]["parent_macro_metrics"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
