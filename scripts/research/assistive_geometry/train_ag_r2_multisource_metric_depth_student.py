#!/usr/bin/env python3
"""Adapt the frozen metric-depth student on consumed multi-source factor labels."""

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

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

import train_ag_r2_metric_depth_student as base  # noqa: E402
from train_ag_r2_f1_factor_learnability import extract_features  # noqa: E402
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
    / "artifacts.local/experiments/ag-r2-multisource-distillation-corpus-r0/result.json"
)
EXPECTED_CORPUS_RESULT_SHA256 = (
    "0D948F8D582F132BD941CAFBDBC7E60E8C11D9C40CC301B0A0AD4F59F369CD6E"
)
SOURCE_STUDENT_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-metric-depth-student-r0/result.json"
)
EXPECTED_SOURCE_STUDENT_RESULT_SHA256 = (
    "49549F4D46A70AA56EC55695C1822060A5B8A73709534BB2F045BC2353107DB6"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "9B990AA0D8BA136B1789A70F8BB939D3D0F00ABD6FDE210B00A9EC2357AC1CBD"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-multisource-metric-depth-student-r0"
)
TRAINING_SEED = 2026081202


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    step: int,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    torch.save(
        {
            "schema": "blindassist_ag_r2_multisource_metric_depth_student_checkpoint_v1",
            "step": step,
            "seed": TRAINING_SEED,
            "source_checkpoint_sha256": source_checkpoint_sha256,
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
    require(
        torch.cuda.is_available() and str(args.device).startswith("cuda"),
        "CUDA required",
    )
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(
        sha256_file(args.corpus_result) == args.expected_corpus_sha256,
        "multi-source corpus drift",
    )
    require(
        sha256_file(args.source_student_result)
        == args.expected_source_student_result_sha256,
        "source metric student result drift",
    )
    require(
        sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256,
        "DepthART drift",
    )
    corpus = json.loads(args.corpus_result.read_text(encoding="utf-8"))
    require(
        corpus["passed"]
        and corpus["frame_count"] == len(corpus["frames"])
        and corpus["optimizer_supported_frame_count"]
        == sum(int(row["metric_depth_valid_pixels"]) > 0 for row in corpus["frames"]),
        "multi-source corpus invalid",
    )
    unsupported = [
        row for row in corpus["frames"] if int(row["metric_depth_valid_pixels"]) == 0
    ]
    rows = sorted(
        [row for row in corpus["frames"] if int(row["metric_depth_valid_pixels"]) > 0],
        key=lambda row: str(row["sample_id"]),
    )
    require(
        len(rows) == int(corpus["optimizer_supported_frame_count"]),
        "optimizer-supported roster drift",
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
        == {
            role: int(corpus["roles"][role]["frame_count"])
            - sum(
                1
                for row in unsupported
                if str(row["role"]) == role
            )
            for role in by_role
        },
        "multi-source student role split drift",
    )
    role_parents = {
        role: {sample.parent_id for sample in values}
        for role, values in by_role.items()
    }
    require(
        {
            role: sorted(values)
            for role, values in role_parents.items()
        }
        == {
            role: sorted(str(value) for value in corpus["roles"][role]["parents"])
            for role in role_parents
        },
        "multi-source student parent split drift",
    )
    require(
        role_parents["FIT"].isdisjoint(role_parents["CHECKPOINT_SELECTION"])
        and role_parents["FIT"].isdisjoint(role_parents["TRAIN_CANARY"])
        and role_parents["CHECKPOINT_SELECTION"].isdisjoint(
            role_parents["TRAIN_CANARY"]
        ),
        "multi-source student parent overlap",
    )

    source_result = json.loads(
        args.source_student_result.read_text(encoding="utf-8")
    )
    source_checkpoint = Path(source_result["checkpoint"]["path"])
    require(
        sha256_file(source_checkpoint) == args.expected_source_checkpoint_sha256,
        "source metric student checkpoint drift",
    )
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    model = base.MetricDepthStudentHead(
        args.hidden_channels, args.global_hidden_channels
    ).to(device)
    source_state = torch.load(source_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(source_state["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    fit_by_parent: dict[str, list[Any]] = defaultdict(list)
    for sample in by_role["FIT"]:
        fit_by_parent[sample.parent_id].append(sample)
    fit_parents = sorted(fit_by_parent)
    rng = random.Random(TRAINING_SEED)
    candidates: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    def capture(step: int) -> None:
        selection = base.evaluate(model, by_role["CHECKPOINT_SELECTION"], device)
        checkpoint = save_checkpoint(
            args.output_dir / f"multisource-metric-depth-step-{step}.pt",
            model,
            step,
            args.expected_source_checkpoint_sha256,
        )
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
                    "selection_depth_log_rmse": selection["parent_macro_metrics"][
                        "depth_log_rmse"
                    ],
                    "selection_depth_scale_abs_log_error": selection[
                        "parent_macro_metrics"
                    ]["depth_scale_abs_log_error"],
                }
            ),
            flush=True,
        )

    capture(0)
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        parent = fit_parents[rng.randrange(len(fit_parents))]
        sample = fit_by_parent[parent][rng.randrange(len(fit_by_parent[parent]))]
        outputs, target = base.forward_sample(
            model, sample, device, flip=rng.random() < 0.5
        )
        losses = base.metric_depth_loss(outputs, target)
        require(bool(torch.isfinite(losses["total"])), "metric objective non-finite")
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
                    "losses": {
                        name: float(value.detach()) for name, value in losses.items()
                    },
                }
            )
        if step % args.checkpoint_interval == 0 or step == args.optimizer_steps:
            capture(step)

    selected = min(candidates, key=lambda row: (row["score"], row["step"]))
    state = torch.load(
        selected["checkpoint"]["path"], map_location=device, weights_only=True
    )
    model.load_state_dict(state["model"], strict=True)
    final_checkpoint = save_checkpoint(
        args.output_dir / "multisource-metric-depth-student.pt",
        model,
        selected["step"],
        args.expected_source_checkpoint_sha256,
    )
    fit_evaluation = base.evaluate(model, by_role["FIT"], device)
    selection_evaluation = base.evaluate(
        model, by_role["CHECKPOINT_SELECTION"], device
    )
    # This is the only post-selection opening of the held TRAIN_CANARY role.
    canary_evaluation = base.evaluate(model, by_role["TRAIN_CANARY"], device)
    finite_metrics = all(
        math.isfinite(float(value))
        for evaluation in (fit_evaluation, selection_evaluation, canary_evaluation)
        for value in evaluation["parent_macro_metrics"].values()
    )
    gates = {
        "MSMETRIC_C01_EXACT_CORPUS_SOURCE_STUDENT_AND_DEPTHART": True,
        "MSMETRIC_C02_CORPUS_DECLARED_PARENT_DISJOINT_ROLES": True,
        "MSMETRIC_C03_SELECTION_ONLY_CHECKPOINT_CHOICE": True,
        "MSMETRIC_C04_TRAIN_CANARY_OPENED_ONCE_AFTER_SELECTION": True,
        "MSMETRIC_C05_NONZERO_ADAPTATION_CHECKPOINT_SELECTED": bool(
            selected["step"] > 0
        ),
        "MSMETRIC_C06_FINITE_FACTOR_METRICS_AND_CHECKPOINT": finite_metrics,
        "MSMETRIC_C07_FACTOR_ONLY_NO_TASK_OR_REDUCER_OUTPUT": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_multisource_metric_depth_student_result_v1",
        "status": "AG_R2_MULTISOURCE_METRIC_DEPTH_STUDENT_PASS"
        if passed
        else "AG_R2_MULTISOURCE_METRIC_DEPTH_STUDENT_FAIL",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "corpus": {
            "path": str(args.corpus_result.resolve()),
            "sha256": args.expected_corpus_sha256,
            "parent_count": int(corpus["parent_count"]),
            "frame_count": int(corpus["frame_count"]),
            "optimizer_supported_frame_count": int(
                corpus["optimizer_supported_frame_count"]
            ),
            "unsupported_unknown_sample_ids": [
                row["sample_id"] for row in unsupported
            ],
        },
        "source_student": {
            "result": str(args.source_student_result.resolve()),
            "result_sha256": args.expected_source_student_result_sha256,
            "checkpoint": str(source_checkpoint.resolve()),
            "checkpoint_sha256": args.expected_source_checkpoint_sha256,
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
            "frozen_encoder": "DepthART-S metric indoor",
            "trainable_head": "bounded global metric scale plus zero-mean local log-depth shape",
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "hidden_channels": args.hidden_channels,
            "global_hidden_channels": args.global_hidden_channels,
            "local_log_correction_range": base.LOCAL_LOG_CORRECTION_RANGE,
            "global_log_correction_range": base.GLOBAL_LOG_CORRECTION_RANGE,
            "learned_final_task_head": False,
        },
        "training": {
            "seed": TRAINING_SEED,
            "optimizer_steps": args.optimizer_steps,
            "checkpoint_interval": args.checkpoint_interval,
            "learning_rate": args.learning_rate,
            "parent_balanced_sampling": True,
            "trace": trace,
        },
        "selection_candidates": candidates,
        "selected_step": selected["step"],
        "checkpoint": final_checkpoint,
        "fit_evaluation": fit_evaluation,
        "selection_evaluation": selection_evaluation,
        "canary_evaluation": canary_evaluation,
        "gates": gates,
        "decision": {
            "metric_scale_learned_from_superteacher_depth": True,
            "consumed_icl_and_tum_labels_used_for_fit": True,
            "task_or_reducer_output_used_for_training_or_selection": False,
            "quality_claim_deferred_to_new_ag_seam": True,
            "next_action": "Recompute factor-only uncertainty on the consumed fit/selection corpus, freeze it, then open one third checkpoint-unseen real parent.",
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
        "--expected-corpus-sha256",
        default=EXPECTED_CORPUS_RESULT_SHA256,
    )
    parser.add_argument(
        "--source-student-result",
        type=Path,
        default=SOURCE_STUDENT_RESULT,
    )
    parser.add_argument(
        "--expected-source-student-result-sha256",
        default=EXPECTED_SOURCE_STUDENT_RESULT_SHA256,
    )
    parser.add_argument(
        "--expected-source-checkpoint-sha256",
        default=EXPECTED_SOURCE_CHECKPOINT_SHA256,
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
    parser.add_argument("--optimizer-steps", type=int, default=3000)
    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--global-hidden-channels", type=int, default=128)
    args = parser.parse_args()
    for name in (
        "corpus_result",
        "source_student_result",
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
