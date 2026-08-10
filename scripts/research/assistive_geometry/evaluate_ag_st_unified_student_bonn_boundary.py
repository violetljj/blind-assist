#!/usr/bin/env python3
"""Evaluate a frozen unified AG-ST checkpoint on Bonn source-depth boundaries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_ag_st_student_bonn_depth import (
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    build_students,
    checkpoint_architecture,
    checkpoint_parent_ids,
    extract_rgb_only_feature,
    fixed_frame_pairs,
    load_cohort_indices,
    load_depth_native,
    load_rgb_native,
    validate_source_receipts,
)
from materialize_ag_st_source_native_boundary_corpus import conservative_source_boundary
from train_ag_st_soft_boundary_bonn_canary import boundary_metrics
from train_ag_st_masked_student import load_depthart_backbone
from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COHORT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_BONN_MIXED_DOMAIN_COHORT_R0_2026-08-10.json"
)
BONN_INTRINSICS = np.asarray(
    [[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def parent_macro(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    require(rows, "boundary parent metrics empty")
    names = (
        "student_average_precision",
        "precision_within_tolerance",
        "recall_within_tolerance",
        "f1_within_tolerance",
    )
    return {
        name: float(np.mean([float(row[name]) for row in rows.values()]))
        for name in names
    }


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output = args.output.resolve()
    checkpoint_path = args.student_checkpoint.resolve()
    cohort_path = args.cohort_manifest.resolve()
    require(not output.exists(), f"boundary output collision: {output}")
    require(checkpoint_path.is_file() and cohort_path.is_file(), "boundary evaluator input missing")
    require(torch.cuda.is_available(), "Bonn boundary evaluation requires CUDA")
    frame_indices = load_cohort_indices(cohort_path, "evaluation")
    _, _, source_provenance = validate_source_receipts(
        args.dataset_root.resolve(),
        args.archive.resolve(),
        args.catalog.resolve(),
        args.receipt.resolve(),
        set(frame_indices),
    )
    pairs_by_parent = fixed_frame_pairs(args.dataset_root.resolve(), frame_indices)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "student checkpoint invalid")
    architecture = checkpoint_architecture(checkpoint)
    require(
        not (checkpoint_parent_ids(checkpoint) & set(frame_indices)),
        "checkpoint/Bonn parent overlap",
    )
    device = torch.device("cuda")
    extractor, scan = load_depthart_backbone(
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(checkpoint["seed"]),
    )
    baseline, student = build_students(checkpoint, architecture, device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    baseline_probability_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    student_probability_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    student_distance_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    by_parent_values: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    frames: list[dict[str, Any]] = []
    for parent_id, pairs in pairs_by_parent.items():
        grouped = {
            "baseline_probability": [],
            "student_probability": [],
            "student_distance": [],
        }
        by_parent_values[parent_id] = grouped
        for pair in pairs:
            rgb = load_rgb_native(pair.rgb.absolute_path)
            feature, base_depth = extract_rgb_only_feature(
                extractor,
                rgb,
                architecture["feature_profile"],
                device,
                amp_dtype,
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                baseline_output = baseline(feature, base_depth, (480, 640))
                student_output = student(feature, base_depth, (480, 640))
            baseline_probability = torch.sigmoid(
                baseline_output["boundary_logits"]
            )[0, 0].float().cpu().numpy()
            student_probability = torch.sigmoid(
                student_output["boundary_logits"]
            )[0, 0].float().cpu().numpy()
            student_distance = student_output["boundary_distance_px"][0, 0].float().cpu().numpy()
            student_distance_score = np.exp(
                -0.5 * np.square(student_distance / 3.0)
            ).astype(np.float32)
            # Source depth is opened only after all RGB/K predictions exist.
            depth, source_valid = load_depth_native(pair.depth.absolute_path)
            truth_probability, boundary_valid = conservative_source_boundary(
                depth,
                source_valid,
                BONN_INTRINSICS,
            )
            truth = boundary_valid & (truth_probability >= 0.5)
            items = {
                "baseline_probability": (baseline_probability, truth, boundary_valid),
                "student_probability": (student_probability, truth, boundary_valid),
                "student_distance": (student_distance_score, truth, boundary_valid),
            }
            baseline_probability_values.append(items["baseline_probability"])
            student_probability_values.append(items["student_probability"])
            student_distance_values.append(items["student_distance"])
            for name, item in items.items():
                grouped[name].append(item)
            frames.append(
                {
                    "parent_id": parent_id,
                    "rgb_row_index_zero_based": pair.rgb.row_index,
                    "rgb_sha256": sha256_file(pair.rgb.absolute_path),
                    "depth_sha256": sha256_file(pair.depth.absolute_path),
                    "association_delta_seconds": pair.association_delta_seconds,
                    "boundary_valid_pixels": int(np.sum(boundary_valid)),
                    "boundary_positive_pixels": int(np.sum(truth)),
                }
            )
    threshold = 0.5
    tolerance = 4
    overall = {
        "baseline_probability": boundary_metrics(
            baseline_probability_values, threshold, tolerance
        ),
        "student_probability": boundary_metrics(
            student_probability_values, threshold, tolerance
        ),
        "student_distance": boundary_metrics(
            student_distance_values, threshold, tolerance
        ),
    }
    by_parent: dict[str, dict[str, Any]] = {}
    for parent_id, grouped in sorted(by_parent_values.items()):
        by_parent[parent_id] = {
            name: boundary_metrics(values, threshold, tolerance)
            for name, values in grouped.items()
        }
    macros = {
        name: parent_macro(
            {parent: metrics[name] for parent, metrics in by_parent.items()}
        )
        for name in overall
    }
    result = {
        "schema": "blindassist_ag_st_unified_student_bonn_boundary_evaluation_v1",
        "status": "EXTERNAL_BONN_BOUNDARY_DIAGNOSTIC_COMPLETE",
        "question": "Does the frozen R11 unified checkpoint transfer its boundary factor to source-native Bonn geometry without fitting or threshold selection?",
        "inputs": {
            "student_checkpoint": str(checkpoint_path),
            "student_checkpoint_sha256": sha256_file(checkpoint_path),
            "cohort": str(cohort_path),
            "cohort_sha256": sha256_file(cohort_path),
            "source_provenance": source_provenance,
            "depthart_checkpoint_sha256": sha256_file(args.depthart_checkpoint.resolve()),
        },
        "protocol": {
            "parent_count": len(pairs_by_parent),
            "frame_count": len(frames),
            "threshold": threshold,
            "tolerance_px": tolerance,
            "source_depth_opened_after_rgb_k_predictions": True,
            "fitting_or_threshold_selection": False,
        },
        "overall": overall,
        "parent_macro": macros,
        "by_parent": by_parent,
        "frames": frames,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "scan_backend": scan,
        },
        "decision": {
            "formal_f1_authority_changed": False,
            "task_utility_evaluated": False,
            "support_or_obstacle_evaluated": False,
        },
        "claim_boundary": "External Bonn source-depth boundary diagnostic of one frozen checkpoint; no support, obstacle, task, safety, deployment, product, license, or formal F1 claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output), "overall": overall, "parent_macro": macros, "execution": result["execution"]}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
