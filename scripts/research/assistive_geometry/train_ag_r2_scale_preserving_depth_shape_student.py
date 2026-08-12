#!/usr/bin/env python3
"""Train a factor-only depth shape head without changing metric scale."""

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

from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    DEPTHART_PYRAMID_CHANNELS,
    extract_features,
)
from train_ag_r2_f1_factor_learnability_attempt02 import SpatialBlock  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    require,
    sha256_file,
)


DEFAULT_CORPUS_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-superteacher-distillation-corpus-tum13-r0/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-scale-preserving-depth-shape-student-r1"
)
TRAINING_SEED = 20260812
LOG_CORRECTION_RANGE = 1.25


class ScalePreservingDepthShapeHead(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.residual = SpatialBlock(DEPTHART_PYRAMID_CHANNELS + 1, hidden, 1)

    def forward(
        self,
        feature: torch.Tensor,
        base_depth_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_log = base_depth_m.clamp(0.05, 20.0).log()
        raw = LOG_CORRECTION_RANGE * torch.tanh(
            self.residual(torch.cat([feature, base_log], dim=1))
        )
        correction = raw - raw.mean(dim=(-2, -1), keepdim=True)
        predicted_log_depth = base_log + correction
        return {
            "predicted_log_depth": predicted_log_depth,
            "log_shape_correction": correction,
            "removed_log_correction_center": raw.mean(dim=(-2, -1)),
        }


def forward_sample(
    model: ScalePreservingDepthShapeHead,
    sample: Any,
    device: torch.device,
    flip: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    feature = sample.feature[None].to(device=device, dtype=torch.float32)
    base = sample.base_depth_feature[None].to(device=device, dtype=torch.float32)
    target = {
        key: value[None].to(device=device)
        for key, value in sample.targets.items()
        if key in {"metric_depth_m", "metric_valid"}
    }
    if flip:
        feature = torch.flip(feature, dims=(-1,))
        base = torch.flip(base, dims=(-1,))
        target = {key: torch.flip(value, dims=(-1,)) for key, value in target.items()}
    return model(feature, base), target


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = valid.bool() & torch.isfinite(value)
    require(bool(selected.any()), "depth supervision denominator empty")
    return value[selected].mean()


def depth_shape_loss(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    predicted = outputs["predicted_log_depth"]
    truth = target["metric_depth_m"].clamp_min(0.05).log()
    valid = target["metric_valid"].bool()
    point = masked_mean(F.smooth_l1_loss(predicted, truth, reduction="none", beta=0.10), valid)
    horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
    vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
    predicted_dx = predicted[..., :, 1:] - predicted[..., :, :-1]
    truth_dx = truth[..., :, 1:] - truth[..., :, :-1]
    predicted_dy = predicted[..., 1:, :] - predicted[..., :-1, :]
    truth_dy = truth[..., 1:, :] - truth[..., :-1, :]
    horizontal = masked_mean(
        F.smooth_l1_loss(predicted_dx, truth_dx, reduction="none", beta=0.05),
        horizontal_valid,
    )
    vertical = masked_mean(
        F.smooth_l1_loss(predicted_dy, truth_dy, reduction="none", beta=0.05),
        vertical_valid,
    )
    correction_regularizer = outputs["log_shape_correction"].square().mean()
    total = point + 0.25 * (horizontal + vertical) + 0.01 * correction_regularizer
    return {
        "total": total,
        "point_log_huber": point,
        "horizontal_log_gradient": horizontal,
        "vertical_log_gradient": vertical,
        "correction_l2": correction_regularizer,
    }


def frame_metrics(
    predicted_log: torch.Tensor,
    base_log: torch.Tensor,
    truth_depth: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    truth_log = truth_depth.clamp_min(0.05).log()
    selected = valid.bool()
    predicted_error = (predicted_log - truth_log).abs()[selected]
    base_error = (base_log - truth_log).abs()[selected]
    predicted_depth = predicted_log.exp()
    scale_delta = abs(float(predicted_log.mean() - base_log.mean()))
    return {
        "depth_abs_log_error": float(predicted_error.mean()),
        "depth_log_rmse": float(torch.sqrt(predicted_error.square().mean())),
        "depth_abs_log_error_q90": float(torch.quantile(predicted_error, 0.90)),
        "depth_mae_m": float((predicted_depth - truth_depth).abs()[selected].mean()),
        "base_depth_abs_log_error": float(base_error.mean()),
        "base_depth_log_rmse": float(torch.sqrt(base_error.square().mean())),
        "base_depth_abs_log_error_q90": float(torch.quantile(base_error, 0.90)),
        "base_depth_mae_m": float((base_log.exp() - truth_depth).abs()[selected].mean()),
        "all_pixel_log_scale_delta": scale_delta,
    }


def evaluate(
    model: ScalePreservingDepthShapeHead,
    samples: list[Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            outputs, target = forward_sample(model, sample, device)
            base_log = sample.base_depth_feature[None].to(device=device, dtype=torch.float32).clamp(0.05, 20.0).log()
            metrics = frame_metrics(
                outputs["predicted_log_depth"],
                base_log,
                target["metric_depth_m"],
                target["metric_valid"],
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "role": sample.role,
                    "metrics": metrics,
                }
            )
    metric_names = tuple(rows[0]["metrics"])
    overall = {
        name: float(np.mean([row["metrics"][name] for row in rows]))
        for name in metric_names
    }
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[row["parent_id"]].append(row)
    parent_metrics = {
        parent: {
            name: float(np.mean([row["metrics"][name] for row in parent_rows]))
            for name in metric_names
        }
        for parent, parent_rows in sorted(by_parent.items())
    }
    parent_macro = {
        name: float(np.mean([value[name] for value in parent_metrics.values()]))
        for name in metric_names
    }
    return {
        "frame_count": len(rows),
        "parent_count": len(parent_metrics),
        "overall_metrics": overall,
        "parent_macro_metrics": parent_macro,
        "parent_metrics": parent_metrics,
        "frames": rows,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    step: int,
) -> dict[str, Any]:
    torch.save(
        {
            "schema": "blindassist_ag_r2_scale_preserving_depth_shape_checkpoint_v1",
            "step": step,
            "seed": TRAINING_SEED,
            "model": model.state_dict(),
        },
        path,
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "step": step,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and str(args.device).startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    corpus_sha = sha256_file(args.corpus_result)
    corpus = json.loads(args.corpus_result.read_text(encoding="utf-8"))
    require(corpus["passed"] and corpus["frame_count"] == 156, "distillation corpus drift")
    unsupported_rows = [
        row for row in corpus["frames"] if int(row["metric_depth_valid_pixels"]) == 0
    ]
    rows = sorted(
        [row for row in corpus["frames"] if int(row["metric_depth_valid_pixels"]) > 0],
        key=lambda row: str(row["sample_id"]),
    )
    require(
        len(unsupported_rows) == 1
        and unsupported_rows[0]["sample_id"]
        == "rgbd_dataset_freiburg2_360_kidnap__rgb000530",
        "unsupported depth roster drift",
    )
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
    by_role = {
        role: [sample for sample in samples if sample.role == role]
        for role in ("FIT", "CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    require(
        {role: len(values) for role, values in by_role.items()}
        == {"FIT": 107, "CHECKPOINT_SELECTION": 24, "TRAIN_CANARY": 24},
        "student role split drift",
    )
    require(
        not ({sample.parent_id for sample in by_role["FIT"]} & {sample.parent_id for sample in by_role["CHECKPOINT_SELECTION"]})
        and not ({sample.parent_id for sample in by_role["FIT"]} & {sample.parent_id for sample in by_role["TRAIN_CANARY"]})
        and not ({sample.parent_id for sample in by_role["CHECKPOINT_SELECTION"]} & {sample.parent_id for sample in by_role["TRAIN_CANARY"]}),
        "student parent split overlap",
    )

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    model = ScalePreservingDepthShapeHead(args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    fit_by_parent: dict[str, list[Any]] = defaultdict(list)
    for sample in by_role["FIT"]:
        fit_by_parent[sample.parent_id].append(sample)
    parents = sorted(fit_by_parent)
    rng = random.Random(TRAINING_SEED)
    trace = []
    candidates = []

    def capture(step: int) -> None:
        selection = evaluate(model, by_role["CHECKPOINT_SELECTION"], device)
        checkpoint = save_checkpoint(args.output_dir / f"shape-step-{step}.pt", model, step)
        candidates.append(
            {
                "step": step,
                "checkpoint": checkpoint,
                "selection": selection,
                "score": selection["parent_macro_metrics"]["depth_log_rmse"],
            }
        )
        print(
            json.dumps(
                {
                    "step": step,
                    "selection_depth_log_rmse": selection["parent_macro_metrics"]["depth_log_rmse"],
                    "selection_base_depth_log_rmse": selection["parent_macro_metrics"]["base_depth_log_rmse"],
                }
            ),
            flush=True,
        )

    capture(0)
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        parent = parents[rng.randrange(len(parents))]
        sample = fit_by_parent[parent][rng.randrange(len(fit_by_parent[parent]))]
        outputs, target = forward_sample(model, sample, device, flip=rng.random() < 0.5)
        losses = depth_shape_loss(outputs, target)
        require(bool(torch.isfinite(losses["total"])), "shape objective non-finite")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
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
                    "losses": {name: float(value.detach()) for name, value in losses.items()},
                }
            )
        if step % args.checkpoint_interval == 0 or step == args.optimizer_steps:
            capture(step)

    selected = min(candidates, key=lambda row: (row["score"], row["step"]))
    state = torch.load(selected["checkpoint"]["path"], map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    final_checkpoint = save_checkpoint(args.output_dir / "scale-preserving-shape-head.pt", model, selected["step"])
    fit_evaluation = evaluate(model, by_role["FIT"], device)
    selection_evaluation = evaluate(model, by_role["CHECKPOINT_SELECTION"], device)
    canary_evaluation = evaluate(model, by_role["TRAIN_CANARY"], device)

    selection_macro = selection_evaluation["parent_macro_metrics"]
    canary_macro = canary_evaluation["parent_macro_metrics"]
    favorable_canary_parents = sum(
        metrics["depth_log_rmse"] < metrics["base_depth_log_rmse"]
        for metrics in canary_evaluation["parent_metrics"].values()
    )
    gates = {
        "SHAPE_C01_PARENT_DISJOINT_107_24_24": True,
        "SHAPE_C02_SELECTION_RMSE_IMPROVES_BASE": selection_macro["depth_log_rmse"]
        < selection_macro["base_depth_log_rmse"],
        "SHAPE_C03_CANARY_RMSE_IMPROVES_BASE": canary_macro["depth_log_rmse"]
        < canary_macro["base_depth_log_rmse"],
        "SHAPE_C04_CANARY_PARENT_FAVORABLE_GE_1_OF_2": favorable_canary_parents >= 1,
        "SHAPE_C05_SCALE_IDENTITY_EXACT": max(
            selection_macro["all_pixel_log_scale_delta"],
            canary_macro["all_pixel_log_scale_delta"],
        )
        <= 1.0e-6,
        "SHAPE_C06_FACTOR_ONLY_NO_TASK_OR_REDUCER_OUTPUT": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_scale_preserving_depth_shape_student_result_v1",
        "status": "AG_R2_SCALE_PRESERVING_DEPTH_SHAPE_STUDENT_PASS"
        if passed
        else "AG_R2_SCALE_PRESERVING_DEPTH_SHAPE_STUDENT_FAIL_NO_SEAM_PROMOTION",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "corpus": {
            "path": str(args.corpus_result.resolve()),
            "sha256": corpus_sha,
            "parent_count": 13,
            "frame_count": 156,
            "optimizer_supported_frame_count": 155,
            "unsupported_unknown_frames": [
                {
                    "sample_id": row["sample_id"],
                    "reason": "ZERO_SOURCE_NATIVE_DEPTH_VALID_PIXELS",
                }
                for row in unsupported_rows
            ],
        },
        "feature_receipt": feature_receipt,
        "architecture": {
            "frozen_encoder": "DepthART-S metric indoor",
            "trainable_head": "scale-preserving dense log-depth shape residual",
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "log_correction_range": LOG_CORRECTION_RANGE,
            "learned_final_task_head": False,
        },
        "training": {
            "seed": TRAINING_SEED,
            "optimizer_steps": args.optimizer_steps,
            "learning_rate": args.learning_rate,
            "checkpoint_interval": args.checkpoint_interval,
            "trace": trace,
        },
        "selection_candidates": candidates,
        "selected_step": selected["step"],
        "checkpoint": final_checkpoint,
        "fit_evaluation": fit_evaluation,
        "selection_evaluation": selection_evaluation,
        "canary_evaluation": canary_evaluation,
        "favorable_canary_parent_count": favorable_canary_parents,
        "gates": gates,
        "decision": {
            "factor_student_shape_head_promoted_to_consumed_seam": passed,
            "task_or_reducer_output_used_for_training_or_selection": False,
            "metric_scale_learned_or_modified": False,
            "fresh_generalization_claim": False,
            "next_action_if_pass": "Compose this head with support/boundary factor heads and rerun the consumed factor-to-adapter-to-reducer seam.",
        },
        "claim_boundary": "Parent-disjoint TUM TRAIN factor-distillation learnability. Not fresh-source, task utility, deployment, product, or safety proof.",
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-result", type=Path, default=DEFAULT_CORPUS_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--optimizer-steps", type=int, default=2400)
    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    for name in (
        "corpus_result",
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
                "checkpoint": result["checkpoint"],
                "selection": result["selection_evaluation"]["parent_macro_metrics"],
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
