#!/usr/bin/env python3
"""Build obstacle evidence from frozen metric depth plus runtime-equivalent pose.

For six consumed checkpoint-held ARKit/TUM parents, DepthART predicts metric
depth from RGB+K.  Source-native camera_to_world is treated only as a stand-in
for product VIO/IMU pose and gravity.  The candidate recovers the lowest
persistent horizontal world-height mode from predicted depth, derives camera
height, and reuses the deterministic factor geometry to produce obstacle
probability.  Source depth/factor truth is opened only after every candidate is
complete.  The reducer and task outcomes are never called.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.assistive_geometry import (
    run_ag_obstacle_evidence_tristate_calibration_canary as tri,
)
from scripts.research.assistive_geometry.build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    FactorLabelPolicy,
    backproject_depth_grid,
    compute_dense_normals,
    compute_geometric_factors,
)
from scripts.research.assistive_geometry.run_ag_st_direct_teacher_to_ag_real_seam import (
    require,
    sha256_file,
    write_json,
)
from scripts.research.assistive_geometry.train_ag_st_masked_student import (
    FrameDescriptor,
    load_depthart_backbone,
    parse_trajectory,
    preprocess_rgb,
    resolve_trajectory_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTE_RESULT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_OBSTACLE_SELECTIVE_INTERACTION_HEAD_RESULT_2026-08-13.json"
)
DEFAULT_PRIOR_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-obstacle-selective-interaction-head-canary-r0/result.json"
)
EXPECTED_PRIOR_RESULT_SHA256 = (
    "0067B9256C1B72519E3F63683DA2BE91D1B32D393FBF3C8C8DFEC4C07B31255F"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-depth-pose-analytic-obstacle-canary-r0"
)


@dataclass(frozen=True)
class HeightModePolicy:
    height_bin_m: float = 0.04
    mode_radius_bins: int = 2
    minimum_persistent_frames: int = 2
    minimum_frame_points: int = 96
    minimum_frame_fraction: float = 0.002
    minimum_total_points: int = 384
    minimum_total_fraction: float = 0.002
    horizontal_tilt_degrees: float = 20.0
    maximum_depth_m: float = 5.0
    sample_stride: int = 2


def persistent_height_modes(
    frame_heights: list[np.ndarray],
    policy: HeightModePolicy = HeightModePolicy(),
) -> list[dict[str, Any]]:
    require(
        len(frame_heights) >= policy.minimum_persistent_frames,
        "too few pose frames",
    )
    require(
        all(values.ndim == 1 and values.size > 0 for values in frame_heights),
        "horizontal height sample empty",
    )
    all_values = np.concatenate(frame_heights).astype(np.float64)
    require(bool(np.isfinite(all_values).all()), "horizontal height non-finite")
    low, high = np.quantile(all_values, (0.002, 0.998))
    first_bin = int(math.floor(float(low) / policy.height_bin_m)) - 1
    last_bin = int(math.ceil(float(high) / policy.height_bin_m)) + 1
    bin_ids = np.arange(first_bin, last_bin + 1, dtype=np.int64)
    frame_counts: list[np.ndarray] = []
    for values in frame_heights:
        ids = np.floor(values / policy.height_bin_m).astype(np.int64)
        counts = np.zeros(len(bin_ids), dtype=np.int64)
        inside = (ids >= first_bin) & (ids <= last_bin)
        np.add.at(counts, ids[inside] - first_bin, 1)
        frame_counts.append(counts)
    stacked = np.stack(frame_counts)
    total = np.sum(stacked, axis=0)
    smooth = np.convolve(
        total.astype(np.float64),
        np.asarray([0.25, 0.50, 0.25], dtype=np.float64),
        mode="same",
    )
    peaks = [
        index
        for index in range(1, len(bin_ids) - 1)
        if smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]
    ]
    peaks.sort(key=lambda index: (-smooth[index], int(bin_ids[index])))
    selected: list[int] = []
    for index in peaks:
        if any(abs(index - prior) <= policy.mode_radius_bins for prior in selected):
            continue
        selected.append(index)
    total_minimum = max(
        policy.minimum_total_points,
        int(math.ceil(policy.minimum_total_fraction * len(all_values))),
    )
    candidates: list[dict[str, Any]] = []
    for index in selected:
        left = max(0, index - policy.mode_radius_bins)
        right = min(len(bin_ids), index + policy.mode_radius_bins + 1)
        per_frame = np.sum(stacked[:, left:right], axis=1)
        frame_minima = np.asarray(
            [
                max(
                    policy.minimum_frame_points,
                    int(math.ceil(policy.minimum_frame_fraction * len(values))),
                )
                for values in frame_heights
            ]
        )
        persistent = per_frame >= frame_minima
        support_count = int(np.sum(per_frame))
        if int(np.sum(persistent)) < policy.minimum_persistent_frames:
            continue
        if support_count < total_minimum:
            continue
        center = (float(bin_ids[index]) + 0.5) * policy.height_bin_m
        near = np.abs(all_values - center) <= (
            (policy.mode_radius_bins + 0.5) * policy.height_bin_m
        )
        candidates.append(
            {
                "world_height_m": float(np.median(all_values[near])),
                "persistent_frame_count": int(np.sum(persistent)),
                "frame_count": len(frame_heights),
                "support_sample_count": support_count,
                "support_fraction": support_count / len(all_values),
                "per_frame_support_sample_count": per_frame.tolist(),
            }
        )
    return sorted(candidates, key=lambda row: float(row["world_height_m"]))


def horizontal_world_heights(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    policy: HeightModePolicy = HeightModePolicy(),
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= 0.05) & (depth <= policy.maximum_depth_m)
    normals_camera, normal_valid = compute_dense_normals(depth, valid, intrinsics)
    points_camera = backproject_depth_grid(depth, intrinsics)
    rotation = np.asarray(camera_to_world[:3, :3], dtype=np.float64)
    translation = np.asarray(camera_to_world[:3, 3], dtype=np.float64)
    normals_world = np.einsum("...j,ij->...i", normals_camera, rotation)
    points_world_z = (
        np.einsum("...j,j->...", points_camera, rotation[2]) + translation[2]
    )
    horizontal = (
        valid
        & normal_valid
        & np.isfinite(points_world_z)
        & (
            np.abs(normals_world[..., 2])
            >= math.cos(math.radians(policy.horizontal_tilt_degrees))
        )
    )
    values = points_world_z[horizontal].astype(np.float64)
    require(values.size > 0, "no predicted horizontal points")
    return values[:: policy.sample_stride]


def extract_depth_and_pose_inputs(
    descriptors: list[FrameDescriptor],
    depthart_source: Path,
    depthart_checkpoint: Path,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    require(device.type == "cuda", "DepthART prediction requires CUDA")
    started = time.perf_counter()
    extractor, scan = load_depthart_backbone(
        depthart_source, depthart_checkpoint, device, seed
    )
    trajectories: dict[str, np.ndarray] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    padded_shapes: set[tuple[int, int]] = set()
    for descriptor in descriptors:
        # K and pose are runtime-equivalent sensor inputs. No source depth or
        # factor outcome is read in this phase.
        with np.load(descriptor.label_path, allow_pickle=False) as payload:
            intrinsics = np.asarray(payload["intrinsics_output"], dtype=np.float64)
            camera_to_world = np.asarray(
                payload["camera_to_world_output"], dtype=np.float64
            )
        require(camera_to_world.shape == (4, 4), "camera pose shape drift")
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
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            cameras = extractor.metric_depthart.cam_embedder(
                intrinsics_tensor[None].to(device),
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
            relative_depth, _, _ = extractor.decode(
                list(features), (padded_height, padded_width)
            )
            scale = extractor.metric_depthart.sfh(features[3], cameras[3])
            depth = (
                relative_depth
                * scale.view(-1, 1, 1, 1)
                * extractor.metric_depthart.max_depth
            )[..., :height, :width].float().clamp(0.05, 20.0)
        predictions[descriptor.frame_stem] = {
            "depth_m": depth[0, 0].cpu().numpy().astype(np.float32),
            "intrinsics": intrinsics,
            "camera_to_world": camera_to_world,
        }
    del extractor
    torch.cuda.empty_cache()
    return predictions, {
        "elapsed_seconds": time.perf_counter() - started,
        "frame_count": len(predictions),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "padded_shapes_hw": [list(value) for value in sorted(padded_shapes)],
        "scan_backend": scan,
    }


def build_analytic_candidates(
    descriptors: list[FrameDescriptor],
    predictions: dict[str, dict[str, np.ndarray]],
    height_policy: HeightModePolicy = HeightModePolicy(),
    factor_policy: FactorLabelPolicy = FactorLabelPolicy(),
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    candidates: dict[str, dict[str, np.ndarray]] = {}
    parent_receipts: list[dict[str, Any]] = []
    for parent in sorted({row.parent_id for row in descriptors}):
        rows = sorted(
            (row for row in descriptors if row.parent_id == parent),
            key=lambda row: row.frame_stem,
        )
        heights: list[np.ndarray] = []
        mode_error: str | None = None
        try:
            for row in rows:
                value = predictions[row.frame_stem]
                heights.append(
                    horizontal_world_heights(
                        value["depth_m"],
                        value["intrinsics"],
                        value["camera_to_world"],
                        height_policy,
                    )
                )
            modes = persistent_height_modes(heights, height_policy)
        except RuntimeError as error:
            modes = []
            mode_error = str(error)
        lowest = float(modes[0]["world_height_m"]) if modes else None
        valid_frames = 0
        frame_receipts: list[dict[str, Any]] = []
        for row in rows:
            value = predictions[row.frame_stem]
            shape = value["depth_m"].shape
            score = np.full(shape, 0.5, dtype=np.float32)
            evidence_valid = np.zeros(shape, dtype=np.bool_)
            camera_height = (
                float(value["camera_to_world"][2, 3]) - lowest
                if lowest is not None
                else None
            )
            height_valid = (
                camera_height is not None
                and math.isfinite(camera_height)
                and factor_policy.plane_height_min_m
                <= camera_height
                <= factor_policy.plane_height_max_m
            )
            if height_valid:
                depth = value["depth_m"]
                metric_valid = (
                    np.isfinite(depth)
                    & (depth >= 0.05)
                    & (depth <= height_policy.maximum_depth_m)
                )
                factors = compute_geometric_factors(
                    depth,
                    metric_valid,
                    value["intrinsics"],
                    value["camera_to_world"],
                    np.where(metric_valid, 0.99, 0.0).astype(np.float32),
                    np.where(metric_valid, TIER_A_SOURCE, 0).astype(np.uint8),
                    np.where(metric_valid, PROVENANCE_SOURCE_NATIVE, 0).astype(
                        np.uint8
                    ),
                    np.where(metric_valid, 0.03 + 0.05 * depth, np.inf).astype(
                        np.float32
                    ),
                    factor_policy,
                    support_camera_height_override_m=float(camera_height),
                    support_plane_residual_override_m=height_policy.height_bin_m,
                )
                evidence_valid = np.asarray(
                    factors["evidence_truth_valid_hw"], dtype=np.bool_
                )
                obstacle = np.asarray(
                    factors["obstacle_evidence_truth_hw"], dtype=np.float32
                )
                score[evidence_valid] = obstacle[evidence_valid]
                valid_frames += 1
            candidates[row.frame_stem] = {
                "score": score,
                "evidence_valid": evidence_valid,
            }
            frame_receipts.append(
                {
                    "frame_id": row.frame_stem,
                    "camera_world_height_m": float(value["camera_to_world"][2, 3]),
                    "camera_height_m": camera_height,
                    "camera_height_valid": height_valid,
                    "candidate_valid_fraction": float(evidence_valid.mean()),
                }
            )
        parent_receipts.append(
            {
                "parent_id": parent,
                "frame_count": len(rows),
                "mode_error": mode_error,
                "horizontal_modes": modes,
                "lowest_persistent_world_height_m": lowest,
                "valid_frame_count": valid_frames,
                "frames": frame_receipts,
            }
        )
    return candidates, parent_receipts


def open_parent_outcomes(
    descriptors: list[FrameDescriptor],
    candidates: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    grouped_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_truth: dict[str, list[np.ndarray]] = defaultdict(list)
    for descriptor in descriptors:
        candidate = candidates[descriptor.frame_stem]
        with np.load(descriptor.label_path, allow_pickle=False) as payload:
            truth = np.asarray(payload["obstacle_evidence_truth_hw"], dtype=np.float32)
            source_valid = np.asarray(
                payload["evidence_truth_valid_hw"], dtype=np.bool_
            )
        require(
            candidate["score"].shape == truth.shape == source_valid.shape,
            "candidate/source outcome shape drift",
        )
        grouped_scores[descriptor.parent_id].append(candidate["score"][source_valid])
        grouped_truth[descriptor.parent_id].append(truth[source_valid] >= 0.5)
    result: dict[str, dict[str, np.ndarray]] = {}
    for parent in sorted(grouped_scores):
        scores = np.concatenate(grouped_scores[parent]).astype(np.float32)
        truth = np.concatenate(grouped_truth[parent]).astype(np.bool_)
        require(scores.size == truth.size > 0, "parent outcome denominator zero")
        require(bool(truth.any()) and bool((~truth).any()), "parent outcome class missing")
        result[parent] = {"scores": scores, "truth": truth}
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(torch.cuda.is_available(), "CUDA required")
    route = json.loads(args.route_result.read_text(encoding="utf-8"))
    require(
        route.get("decision", {}).get("next_successor")
        == "AG_DEPTH_POSE_ANALYTIC_OBSTACLE_CANARY_R0",
        "depth-pose analytic obstacle canary not authorized",
    )
    require(
        sha256_file(args.prior_result.resolve()) == EXPECTED_PRIOR_RESULT_SHA256,
        "interaction terminal SHA drift",
    )
    prior = json.loads(args.prior_result.read_text(encoding="utf-8"))
    require(
        prior.get("status")
        == "AG_OBSTACLE_SELECTIVE_INTERACTION_HEAD_CANARY_FAIL_STOP"
        and prior.get("gate", {}).get("evaluable_parent_count") == 0,
        "interaction terminal status drift",
    )
    require(
        sha256_file(args.r21_result.resolve()) == tri.EXPECTED_R21_RESULT_SHA256,
        "R21 result SHA drift",
    )
    result21 = json.loads(args.r21_result.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.r21_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    held = tri.held_parent_ids(checkpoint)
    descriptors = [
        row for row in tri.load_descriptors(result21) if row.parent_id in set(held)
    ]
    require(
        len(descriptors) == 18
        and {row.parent_id for row in descriptors} == set(held),
        "held descriptor roster drift",
    )
    device = torch.device("cuda")
    predictions, inference = extract_depth_and_pose_inputs(
        descriptors,
        Path(result21["inputs"]["depthart_source"]).resolve(),
        Path(result21["inputs"]["encoder_checkpoint"]).resolve(),
        device,
        int(checkpoint["seed"]),
    )
    candidates, parent_receipts = build_analytic_candidates(
        descriptors, predictions
    )
    parents = open_parent_outcomes(descriptors, candidates)
    folds = tri.leave_one_parent_out(parents)
    factor_gate = tri.gate_folds(folds)
    analytic_parent_count = sum(
        row["valid_frame_count"] == row["frame_count"] for row in parent_receipts
    )
    checks = {
        **factor_gate["checks"],
        "at_least_half_parents_have_complete_pose_analytic_geometry": (
            analytic_parent_count >= int(math.ceil(len(parents) / 2.0))
        ),
        "strict_coverage_gain_over_rgb_factor_family": (
            factor_gate["evaluable_parent_count"]
            > int(prior["gate"]["evaluable_parent_count"])
        ),
    }
    passed = all(checks.values())
    result = {
        "schema": "blindassist_ag_depth_pose_analytic_obstacle_canary_result_v1",
        "status": (
            "AG_DEPTH_POSE_ANALYTIC_OBSTACLE_CANARY_PASS"
            if passed
            else "AG_DEPTH_POSE_ANALYTIC_OBSTACLE_CANARY_FAIL_STOP"
        ),
        "passed": passed,
        "question": (
            "Can frozen metric depth plus runtime-equivalent pose/gravity recover "
            "parent-disjoint two-sided obstacle evidence through deterministic height geometry?"
        ),
        "inputs": {
            "route_result": {
                "path": str(args.route_result.resolve()),
                "sha256": sha256_file(args.route_result.resolve()),
            },
            "interaction_terminal": {
                "path": str(args.prior_result.resolve()),
                "sha256": EXPECTED_PRIOR_RESULT_SHA256,
                "evaluable_parent_count": prior["gate"]["evaluable_parent_count"],
            },
            "r21_result": {
                "path": str(args.r21_result.resolve()),
                "sha256": tri.EXPECTED_R21_RESULT_SHA256,
            },
            "depthart_checkpoint": {
                "path": str(Path(result21["inputs"]["encoder_checkpoint"]).resolve()),
                "sha256": result21["inputs"]["encoder_checkpoint_sha256"],
            },
        },
        "protocol": {
            "role": "CONSUMED_CHECKPOINT_HELD_DEVELOPMENT_PREDICTION_FIRST_PARENT_LOO",
            "sources": ["arkitscenes", "tum_rgbd"],
            "parent_count": len(parents),
            "frame_count": len(descriptors),
            "prediction_inputs": "RGB_PLUS_K_PLUS_CAMERA_TO_WORLD_POSE_GRAVITY",
            "pose_role": "runtime-equivalent VIO/IMU observable; not source depth or factor outcome",
            "source_depth_opened": False,
            "source_factor_outcomes_opened_after_all_candidates": True,
            "task_outcome_read": False,
            "training_steps": 0,
            "lowest_persistent_height_policy": {
                key: getattr(HeightModePolicy(), key)
                for key in HeightModePolicy.__dataclass_fields__
            },
            "factor_policy": "frozen build_ag_st_factor_labels FactorLabelPolicy",
            "candidate_invalid_value": "UNKNOWN encoded as 0.5 between low/high grids",
            "false_negative_rate_cap": tri.FALSE_NEGATIVE_RATE_CAP,
            "false_positive_rate_cap": tri.FALSE_POSITIVE_RATE_CAP,
            "minimum_side_coverage": tri.MIN_SIDE_COVERAGE,
            "reducer_called": False,
        },
        "inference": inference,
        "pose_analytic_geometry": {
            "complete_parent_count": analytic_parent_count,
            "parents": parent_receipts,
        },
        "folds": folds,
        "gate": {
            **factor_gate,
            "checks": checks,
            "pass": passed,
            "rgb_factor_family_evaluable_parent_count": prior["gate"][
                "evaluable_parent_count"
            ],
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "decision": {
            "depth_pose_analytic_obstacle_supported": passed,
            "body_swept_seam_authorized": passed,
            "android_pose_dependency_required": passed,
            "support_can_create_task_validity": False,
            "fresh3_tum_opened": False,
            "default_app_changed": False,
            "next_action_if_pass": (
                "Freeze the parent-LOO thresholds and run one consumed body-swept "
                "mechanics seam with explicit pose availability and support veto-only."
            ),
            "next_action_if_fail": (
                "Stop AG obstacle task landing on the current consumed evidence. New "
                "source-native obstacle supervision or a different representation is required."
            ),
        },
        "claim_boundary": (
            "Consumed six-parent Development evidence for a pose-anchored analytic factor. "
            "A PASS authorizes one mechanics seam only, not independent cross-sensor "
            "generalization, deployment, product, default-App, navigation or assistive safety."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-result", type=Path, default=DEFAULT_ROUTE_RESULT)
    parser.add_argument("--prior-result", type=Path, default=DEFAULT_PRIOR_RESULT)
    parser.add_argument("--r21-result", type=Path, default=tri.DEFAULT_R21_RESULT)
    parser.add_argument(
        "--r21-checkpoint",
        type=Path,
        default=tri.boundary_base.DEFAULT_R21_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "pose_analytic_complete_parent_count": result[
                    "pose_analytic_geometry"
                ]["complete_parent_count"],
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


if __name__ == "__main__":
    main()
