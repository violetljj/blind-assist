#!/usr/bin/env python3
"""Calibrate frozen obstacle logits into POSITIVE/NEGATIVE/UNKNOWN evidence.

The six ARKit/TUM parents held out by the R14/R21 factor checkpoint are used in
leave-one-parent-out Development.  RGB/K predictions are completed before
source obstacle truth or validity is opened.  Low and high thresholds are
selected on the other five parents only; the middle interval and every
source-invalid pixel remain UNKNOWN.

This canary does not train a model or execute the reducer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.assistive_geometry import (
    run_ag_angular_boundary_body_swept_task_canary as boundary_base,
)
from scripts.research.assistive_geometry.run_ag_st_direct_teacher_to_ag_real_seam import (
    require,
    sha256_file,
    write_json,
)
from scripts.research.assistive_geometry.train_ag_st_masked_student import (
    FrameDescriptor,
    build_frame_descriptor_batches,
    build_tum_bound_frame_descriptors,
    load_depthart_backbone,
    parse_trajectory,
    preprocess_rgb,
    resolve_trajectory_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTE_RESULT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_POSITIVE_OBSTACLE_SUPPORT_TASK_EFFECT_AUDIT_RESULT_2026-08-13.json"
)
DEFAULT_R21_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-depthart-angular-boundary-trisource-massnorm-r1/result.json"
)
EXPECTED_R21_RESULT_SHA256 = (
    "48ECB205F8806FE73530305FA391E6AF7A2A6116DF96B5B4AEA1FD6BD0CD9DA0"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-obstacle-evidence-tristate-calibration-canary-r0"
)
LOW_THRESHOLDS = tuple(float(value) for value in np.arange(0.01, 0.50, 0.02))
HIGH_THRESHOLDS = tuple(float(value) for value in np.arange(0.51, 1.00, 0.02))
FALSE_NEGATIVE_RATE_CAP = 0.01
FALSE_POSITIVE_RATE_CAP = 0.05
MIN_SIDE_COVERAGE = 0.01
MIN_NONZERO_PARENT_FRACTION = 0.5


def held_parent_ids(checkpoint: dict[str, Any]) -> tuple[str, ...]:
    by_source = checkpoint["split"]["by_source"]
    parents: list[str] = []
    for source in ("arkitscenes", "tum_rgbd"):
        for role in ("selection_parents", "canary_parents"):
            parents.extend(str(value) for value in by_source[source][role])
    require(len(parents) == len(set(parents)) == 6, "held parent roster drift")
    return tuple(sorted(parents))


def load_descriptors(result: dict[str, Any]) -> list[FrameDescriptor]:
    batches = result["inputs"]["source_batches"]
    arkit = next(row for row in batches if row.get("source_role"))
    tum = next(row for row in batches if row.get("source") == "tum_rgbd")
    arkit_descriptors, _ = build_frame_descriptor_batches(
        [Path(arkit["stage0a_result_path"])],
        [Path(arkit["factor_label_result_path"]).parent],
    )
    tum_descriptors, _ = build_tum_bound_frame_descriptors(
        Path(tum["factor_label_result_path"]).parent,
        Path(tum["rgb_binding_path"]),
    )
    descriptors = arkit_descriptors + tum_descriptors
    require(
        len(descriptors) == 69
        and len({row.frame_stem for row in descriptors}) == 69,
        "ARKit/TUM descriptor roster drift",
    )
    return descriptors


def extract_obstacle_predictions(
    descriptors: list[FrameDescriptor],
    student: torch.nn.Module,
    depthart_source: Path,
    depthart_checkpoint: Path,
    feature_profile: str,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    require(device.type == "cuda", "DepthART prediction requires CUDA")
    started = time.perf_counter()
    extractor, scan = load_depthart_backbone(
        depthart_source, depthart_checkpoint, device, seed
    )
    trajectories: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    padded_shapes: set[tuple[int, int]] = set()
    student.eval()
    for descriptor in descriptors:
        with np.load(descriptor.label_path, allow_pickle=False) as payload:
            intrinsics = np.asarray(payload["intrinsics_output"], dtype=np.float64)
        if descriptor.source_id == "arkitscenes" and descriptor.parent_id not in trajectories:
            trajectories[descriptor.parent_id] = parse_trajectory(
                resolve_trajectory_path(descriptor.video)
            )
        image, intrinsics_tensor = preprocess_rgb(
            descriptor,
            trajectories.get(descriptor.parent_id),
            intrinsics,
        )
        height, width = descriptor.output_hw
        padded_height = int(math.ceil(height / 32.0) * 32)
        padded_width = int(math.ceil(width / 32.0) * 32)
        padded_shapes.add((padded_height, padded_width))
        image_batch = F.pad(
            image[None].to(device),
            (0, padded_width - width, 0, padded_height - height),
            mode="constant",
            value=0.0,
        )
        intrinsics_batch = intrinsics_tensor[None].to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            cameras = extractor.metric_depthart.cam_embedder(
                intrinsics_batch,
                padded_height,
                padded_width,
                device,
            )
            features = extractor.metric_depthart.pretrained.forward_with_adapters(
                image_batch,
                adapters=[
                    extractor.metric_depthart.daa1,
                    extractor.metric_depthart.daa2,
                    extractor.metric_depthart.daa3,
                    extractor.metric_depthart.daa4,
                ],
                cams=list(cameras),
            )
            relative_depth, shared, pyramid = extractor.decode(
                list(features), (padded_height, padded_width)
            )
            scale = extractor.metric_depthart.sfh(features[3], cameras[3])
            base_depth = (
                relative_depth
                * scale.view(-1, 1, 1, 1)
                * extractor.metric_depthart.max_depth
            )[..., :height, :width].float().clamp(0.05, 20.0)
            selected = shared if feature_profile == "shared" else pyramid
            content_height = int(round(selected.shape[-2] * height / padded_height))
            content_width = int(round(selected.shape[-1] * width / padded_width))
            require(
                content_height * padded_height == selected.shape[-2] * height
                and content_width * padded_width == selected.shape[-1] * width,
                "DepthART feature/content ratio drift",
            )
            selected = selected[..., :content_height, :content_width]
            output = student(selected, base_depth, descriptor.output_hw)
            probability = torch.sigmoid(output["obstacle_logits"])[0, 0]
        require(
            bool(torch.isfinite(probability).all()),
            "obstacle probability non-finite",
        )
        predictions[descriptor.frame_stem] = probability.float().cpu().numpy()
    del extractor
    torch.cuda.empty_cache()
    return predictions, {
        "elapsed_seconds": time.perf_counter() - started,
        "frame_count": len(predictions),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "padded_shapes_hw": [list(value) for value in sorted(padded_shapes)],
        "scan_backend": scan,
    }


def open_parent_outcomes(
    descriptors: list[FrameDescriptor],
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    grouped_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_truth: dict[str, list[np.ndarray]] = defaultdict(list)
    for descriptor in descriptors:
        probability = predictions[descriptor.frame_stem]
        with np.load(descriptor.label_path, allow_pickle=False) as payload:
            truth = np.asarray(payload["obstacle_evidence_truth_hw"], dtype=np.float32)
            valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
        require(
            probability.shape == truth.shape == valid.shape == descriptor.output_hw,
            "obstacle prediction/outcome shape drift",
        )
        grouped_scores[descriptor.parent_id].append(probability[valid])
        grouped_truth[descriptor.parent_id].append(truth[valid] >= 0.5)
    result: dict[str, dict[str, np.ndarray]] = {}
    for parent in sorted(grouped_scores):
        scores = np.concatenate(grouped_scores[parent]).astype(np.float32)
        truth = np.concatenate(grouped_truth[parent]).astype(np.bool_)
        require(scores.size == truth.size > 0, "obstacle parent denominator zero")
        require(bool(truth.any()) and bool((~truth).any()), "obstacle parent class missing")
        result[parent] = {"scores": scores, "truth": truth}
    return result


def threshold_stats(
    scores: np.ndarray,
    truth: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    require(0.0 <= low_threshold < high_threshold <= 1.0, "tri-state thresholds invalid")
    positive = scores >= high_threshold
    verified_negative = scores <= low_threshold
    unknown = ~(positive | verified_negative)
    positive_truth = truth
    negative_truth = ~truth
    positive_count = int(positive_truth.sum())
    negative_count = int(negative_truth.sum())
    false_negative = int((verified_negative & positive_truth).sum())
    false_positive = int((positive & negative_truth).sum())
    true_positive = int((positive & positive_truth).sum())
    true_negative = int((verified_negative & negative_truth).sum())
    return {
        "pixel_count": int(scores.size),
        "truth_positive_count": positive_count,
        "truth_negative_count": negative_count,
        "predicted_positive_count": int(positive.sum()),
        "verified_negative_count": int(verified_negative.sum()),
        "unknown_count": int(unknown.sum()),
        "false_negative_count": false_negative,
        "false_positive_count": false_positive,
        "false_negative_rate": false_negative / positive_count,
        "false_positive_rate": false_positive / negative_count,
        "positive_recall": true_positive / positive_count,
        "verified_negative_coverage": true_negative / negative_count,
        "known_coverage": float((positive | verified_negative).mean()),
    }


def choose_low_threshold(
    parents: dict[str, dict[str, np.ndarray]],
) -> tuple[float | None, dict[str, Any]]:
    minimum_nonzero = int(math.ceil(MIN_NONZERO_PARENT_FRACTION * len(parents)))
    tested: list[dict[str, Any]] = []
    for threshold in LOW_THRESHOLDS:
        rows = {
            parent: threshold_stats(value["scores"], value["truth"], threshold, 1.0)
            for parent, value in parents.items()
        }
        safe = all(row["false_negative_rate"] <= FALSE_NEGATIVE_RATE_CAP for row in rows.values())
        covered = sum(
            row["verified_negative_coverage"] >= MIN_SIDE_COVERAGE
            for row in rows.values()
        )
        tested.append(
            {
                "threshold": threshold,
                "safe": safe,
                "covered_parent_count": covered,
                "mean_verified_negative_coverage": float(
                    np.mean([row["verified_negative_coverage"] for row in rows.values()])
                ),
            }
        )
    admissible = [
        row
        for row in tested
        if row["safe"] and row["covered_parent_count"] >= minimum_nonzero
    ]
    selected = max(admissible, key=lambda row: (row["threshold"], row["mean_verified_negative_coverage"])) if admissible else None
    return (None if selected is None else float(selected["threshold"])), {
        "minimum_nonzero_parent_count": minimum_nonzero,
        "tested": tested,
        "selected": selected,
    }


def choose_high_threshold(
    parents: dict[str, dict[str, np.ndarray]],
) -> tuple[float | None, dict[str, Any]]:
    minimum_nonzero = int(math.ceil(MIN_NONZERO_PARENT_FRACTION * len(parents)))
    tested: list[dict[str, Any]] = []
    for threshold in HIGH_THRESHOLDS:
        rows = {
            parent: threshold_stats(value["scores"], value["truth"], 0.0, threshold)
            for parent, value in parents.items()
        }
        safe = all(row["false_positive_rate"] <= FALSE_POSITIVE_RATE_CAP for row in rows.values())
        covered = sum(row["positive_recall"] >= MIN_SIDE_COVERAGE for row in rows.values())
        tested.append(
            {
                "threshold": threshold,
                "safe": safe,
                "covered_parent_count": covered,
                "mean_positive_recall": float(
                    np.mean([row["positive_recall"] for row in rows.values()])
                ),
            }
        )
    admissible = [
        row
        for row in tested
        if row["safe"] and row["covered_parent_count"] >= minimum_nonzero
    ]
    selected = min(admissible, key=lambda row: (row["threshold"], -row["mean_positive_recall"])) if admissible else None
    return (None if selected is None else float(selected["threshold"])), {
        "minimum_nonzero_parent_count": minimum_nonzero,
        "tested": tested,
        "selected": selected,
    }


def leave_one_parent_out(
    parents: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for held_parent in sorted(parents):
        training = {
            parent: value for parent, value in parents.items() if parent != held_parent
        }
        low, low_receipt = choose_low_threshold(training)
        high, high_receipt = choose_high_threshold(training)
        valid = low is not None and high is not None and low < high
        held = (
            threshold_stats(
                parents[held_parent]["scores"],
                parents[held_parent]["truth"],
                float(low),
                float(high),
            )
            if valid
            else None
        )
        folds.append(
            {
                "held_parent": held_parent,
                "training_parent_count": len(training),
                "low_threshold": low,
                "high_threshold": high,
                "threshold_pair_valid": valid,
                "held_metrics": held,
                "low_selection": low_receipt,
                "high_selection": high_receipt,
            }
        )
    return folds


def gate_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in folds if row["held_metrics"] is not None]
    minimum_nonzero = int(math.ceil(MIN_NONZERO_PARENT_FRACTION * len(folds)))
    positive_covered = [
        row["held_parent"]
        for row in evaluable
        if row["held_metrics"]["positive_recall"] >= MIN_SIDE_COVERAGE
    ]
    negative_covered = [
        row["held_parent"]
        for row in evaluable
        if row["held_metrics"]["verified_negative_coverage"] >= MIN_SIDE_COVERAGE
    ]
    checks = {
        "all_folds_threshold_pair_valid": len(evaluable) == len(folds),
        "all_parents_false_negative_rate_le_0p01": bool(evaluable)
        and all(
            row["held_metrics"]["false_negative_rate"] <= FALSE_NEGATIVE_RATE_CAP
            for row in evaluable
        ),
        "all_parents_false_positive_rate_le_0p05": bool(evaluable)
        and all(
            row["held_metrics"]["false_positive_rate"] <= FALSE_POSITIVE_RATE_CAP
            for row in evaluable
        ),
        "at_least_half_parents_positive_recall_ge_0p01": len(positive_covered)
        >= minimum_nonzero,
        "at_least_half_parents_verified_negative_coverage_ge_0p01": len(
            negative_covered
        )
        >= minimum_nonzero,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "evaluable_parent_count": len(evaluable),
        "minimum_nonzero_parent_count": minimum_nonzero,
        "positive_covered_parents": positive_covered,
        "verified_negative_covered_parents": negative_covered,
        "parent_macro": {
            name: (
                float(np.mean([row["held_metrics"][name] for row in evaluable]))
                if evaluable
                else None
            )
            for name in (
                "false_negative_rate",
                "false_positive_rate",
                "positive_recall",
                "verified_negative_coverage",
                "known_coverage",
            )
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(torch.cuda.is_available(), "CUDA required")
    route = json.loads(args.route_result.read_text(encoding="utf-8"))
    require(
        route.get("decision", {}).get("next_successor")
        == "AG_OBSTACLE_EVIDENCE_TRISTATE_CALIBRATION_CANARY_R0",
        "obstacle tri-state canary not authorized",
    )
    require(
        sha256_file(args.r21_result.resolve()) == EXPECTED_R21_RESULT_SHA256,
        "R21 result SHA drift",
    )
    result21 = json.loads(args.r21_result.read_text(encoding="utf-8"))
    checkpoint_path = args.r21_checkpoint.resolve()
    require(
        sha256_file(checkpoint_path) == boundary_base.EXPECTED_R21_CHECKPOINT_SHA256,
        "R21 checkpoint SHA drift",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    held = held_parent_ids(checkpoint)
    descriptors = [
        row for row in load_descriptors(result21) if row.parent_id in set(held)
    ]
    require(
        len(descriptors) == 18
        and {row.parent_id for row in descriptors} == set(held),
        "held descriptor roster drift",
    )
    device = torch.device(args.device)
    student, loaded = boundary_base.load_student(
        checkpoint_path, boundary_base.EXPECTED_R21_CHECKPOINT_SHA256, device
    )
    predictions, inference = extract_obstacle_predictions(
        descriptors,
        student,
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        loaded["architecture"]["feature_profile"],
        device,
        int(loaded["checkpoint"]["seed"]),
    )
    del student
    outcomes = open_parent_outcomes(descriptors, predictions)
    require(set(outcomes) == set(held), "held outcome parent roster drift")
    folds = leave_one_parent_out(outcomes)
    gate = gate_folds(folds)
    status = (
        "AG_OBSTACLE_EVIDENCE_TRISTATE_CALIBRATION_PASS"
        if gate["pass"]
        else "AG_OBSTACLE_EVIDENCE_TRISTATE_CALIBRATION_FAIL_STOP"
    )
    result = {
        "schema": "blindassist_ag_obstacle_evidence_tristate_calibration_canary_result_v1",
        "status": status,
        "passed": gate["pass"],
        "question": (
            "Can the frozen obstacle logit support parent-disjoint POSITIVE, "
            "VERIFIED_NEGATIVE and UNKNOWN evidence with bounded false-negative and "
            "false-positive rates plus nonzero two-sided coverage?"
        ),
        "protocol": {
            "role": "CONSUMED_CHECKPOINT_HELD_DEVELOPMENT_LEAVE_ONE_PARENT_OUT",
            "sources": ["arkitscenes", "tum_rgbd"],
            "parent_count": len(held),
            "frame_count": len(descriptors),
            "held_parents": list(held),
            "prediction_inputs": "RGB_PLUS_K_ONLY",
            "source_outcomes_opened_after_all_predictions": True,
            "low_threshold_grid": list(LOW_THRESHOLDS),
            "high_threshold_grid": list(HIGH_THRESHOLDS),
            "false_negative_rate_cap": FALSE_NEGATIVE_RATE_CAP,
            "false_positive_rate_cap": FALSE_POSITIVE_RATE_CAP,
            "minimum_side_coverage": MIN_SIDE_COVERAGE,
            "minimum_nonzero_parent_fraction": MIN_NONZERO_PARENT_FRACTION,
            "unknown_rule": "low < probability < high or source-invalid; UNKNOWN is never negative",
            "training_steps": 0,
            "reducer_called": False,
        },
        "inputs": {
            "route_result": {
                "path": str(args.route_result.resolve()),
                "sha256": sha256_file(args.route_result.resolve()),
            },
            "r21_result": {
                "path": str(args.r21_result.resolve()),
                "sha256": EXPECTED_R21_RESULT_SHA256,
            },
            "r21_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": boundary_base.EXPECTED_R21_CHECKPOINT_SHA256,
            },
            "depthart_checkpoint_sha256": sha256_file(
                args.depthart_checkpoint.resolve()
            ),
        },
        "inference": inference,
        "folds": folds,
        "gate": gate,
        "execution": {
            "device": str(torch.cuda.get_device_name(device)),
            "elapsed_seconds": time.perf_counter() - started,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "decision": {
            "obstacle_tristate_calibration_supported": gate["pass"],
            "body_swept_seam_authorized": gate["pass"],
            "support_remains_veto_only": True,
            "current_obstacle_logit_retraining_authorized": False,
            "fresh3_tum_opened": False,
            "default_app_changed": False,
            "next_action_if_pass": (
                "Freeze the six out-of-fold threshold pairs and run one consumed "
                "body-swept tri-state mechanics seam before any independent parent."
            ),
            "next_action_if_fail": (
                "Stop threshold calibration for the current obstacle logit; a new "
                "explicit evidence-validity/abstention head is required before task use."
            ),
        },
        "claim_boundary": (
            "Consumed checkpoint-held factor calibration only. A PASS would authorize "
            "one mechanics seam, not task superiority, independent cross-sensor "
            "generalization, deployment, product, default-App, navigation or assistive safety."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-result", type=Path, default=DEFAULT_ROUTE_RESULT)
    parser.add_argument("--r21-result", type=Path, default=DEFAULT_R21_RESULT)
    parser.add_argument(
        "--r21-checkpoint", type=Path, default=boundary_base.DEFAULT_R21_CHECKPOINT
    )
    parser.add_argument(
        "--depthart-source", type=Path, default=boundary_base.DEFAULT_DEPTHART_SOURCE
    )
    parser.add_argument(
        "--depthart-checkpoint",
        type=Path,
        default=boundary_base.DEFAULT_DEPTHART_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "folds": [
                    {
                        "held_parent": row["held_parent"],
                        "low_threshold": row["low_threshold"],
                        "high_threshold": row["high_threshold"],
                        "held_metrics": row["held_metrics"],
                    }
                    for row in result["folds"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
