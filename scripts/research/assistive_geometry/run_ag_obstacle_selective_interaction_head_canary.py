#!/usr/bin/env python3
"""Test one tiny factor-interaction obstacle head with nested parent holdout.

The R21 factor checkpoint and DepthART prior remain frozen.  For the six
checkpoint-held ARKit/TUM Development parents, RGB+K observables are computed
before source obstacle outcomes are opened.  Each outer held parent is scored
by a linear interaction head fitted on the other five parents.  Its tri-state
thresholds come only from inner out-of-parent predictions, so no parent is used
to both fit and calibrate its own score.

This is an observability canary.  It does not call the reducer or task outcome.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.assistive_geometry import (
    run_ag_angular_boundary_body_swept_task_canary as boundary_base,
)
from scripts.research.assistive_geometry import (
    run_ag_obstacle_evidence_tristate_calibration_canary as tri,
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
    / "docs/research/assistive-geometry/BLINDASSIST_AG_OBSTACLE_EVIDENCE_TRISTATE_CALIBRATION_RESULT_2026-08-13.json"
)
DEFAULT_PRIOR_RESULT = tri.DEFAULT_OUTPUT_DIR / "result.json"
EXPECTED_PRIOR_RESULT_SHA256 = (
    "9545718261D4BC3D2D4EF734D7D1536D5B5BE26B32B91842B85D0730E6A7852D"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-obstacle-selective-interaction-head-canary-r0"
)
FEATURE_NAMES = (
    "obstacle_logit",
    "support_logit",
    "boundary_logit",
    "log1p_depth_m",
    "log1p_depth_gradient_m",
    "normalized_image_row",
    "obstacle_x_support",
    "obstacle_x_boundary",
    "obstacle_x_depth_gradient",
    "support_x_boundary",
)
MAX_SAMPLES_PER_PARENT_CLASS = 512
L2_WEIGHT = 1e-3
LBFGS_MAX_ITER = 80


def build_observable_tensor(
    output: dict[str, torch.Tensor],
    base_depth: torch.Tensor,
) -> torch.Tensor:
    obstacle = output["obstacle_logits"][0, 0].float().clamp(-12.0, 12.0)
    support = output["support_logits"][0, 0].float().clamp(-12.0, 12.0)
    boundary = output["boundary_logits"][0, 0].float().clamp(-12.0, 12.0)
    depth = base_depth[0, 0].float().clamp(0.05, 20.0)
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[:, 1:] = torch.abs(depth[:, 1:] - depth[:, :-1])
    grad_y[1:, :] = torch.abs(depth[1:, :] - depth[:-1, :])
    depth_gradient = torch.log1p(grad_x + grad_y)
    height = int(depth.shape[0])
    row = torch.linspace(-1.0, 1.0, height, device=depth.device, dtype=depth.dtype)
    row = row[:, None].expand_as(depth)
    values = (
        obstacle,
        support,
        boundary,
        torch.log1p(depth),
        depth_gradient,
        row,
        obstacle * support,
        obstacle * boundary,
        obstacle * depth_gradient,
        support * boundary,
    )
    tensor = torch.stack(values, dim=-1)
    require(tensor.shape[-1] == len(FEATURE_NAMES), "observable width drift")
    require(bool(torch.isfinite(tensor).all()), "observable non-finite")
    return tensor


def extract_observables(
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
    observables: dict[str, np.ndarray] = {}
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
            tensor = build_observable_tensor(output, base_depth)
        observables[descriptor.frame_stem] = tensor.cpu().numpy().astype(np.float32)
    del extractor
    torch.cuda.empty_cache()
    return observables, {
        "elapsed_seconds": time.perf_counter() - started,
        "frame_count": len(observables),
        "feature_names": list(FEATURE_NAMES),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "padded_shapes_hw": [list(value) for value in sorted(padded_shapes)],
        "scan_backend": scan,
    }


def open_parent_outcomes(
    descriptors: list[FrameDescriptor],
    observables: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    grouped_features: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_truth: dict[str, list[np.ndarray]] = defaultdict(list)
    for descriptor in descriptors:
        features = observables[descriptor.frame_stem]
        with np.load(descriptor.label_path, allow_pickle=False) as payload:
            truth = np.asarray(payload["obstacle_evidence_truth_hw"], dtype=np.float32)
            valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
        require(
            features.shape[:2] == truth.shape == valid.shape == descriptor.output_hw,
            "observable/outcome shape drift",
        )
        grouped_features[descriptor.parent_id].append(features[valid])
        grouped_truth[descriptor.parent_id].append(truth[valid] >= 0.5)
    result: dict[str, dict[str, np.ndarray]] = {}
    for parent in sorted(grouped_features):
        features = np.concatenate(grouped_features[parent]).astype(np.float32)
        truth = np.concatenate(grouped_truth[parent]).astype(np.bool_)
        require(features.shape == (truth.size, len(FEATURE_NAMES)), "parent feature shape drift")
        require(bool(truth.any()) and bool((~truth).any()), "parent class missing")
        result[parent] = {"features": features, "truth": truth}
    return result


def deterministic_class_sample(indices: np.ndarray, count: int) -> np.ndarray:
    require(indices.ndim == 1 and indices.size >= count > 0, "class sample invalid")
    if indices.size == count:
        return indices
    positions = np.linspace(0, indices.size - 1, count, dtype=np.int64)
    selected = indices[positions]
    require(np.unique(selected).size == count, "class sample duplicate")
    return selected


def balanced_training_matrix(
    parents: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(len(parents) >= 2, "training parent count too small")
    minimum_class_count = min(
        int(np.sum(value["truth"] == label))
        for value in parents.values()
        for label in (False, True)
    )
    per_parent_class = min(MAX_SAMPLES_PER_PARENT_CLASS, minimum_class_count)
    require(per_parent_class >= 32, "training class sample too small")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    receipt: dict[str, Any] = {}
    for parent, value in sorted(parents.items()):
        parent_counts: dict[str, int] = {}
        for label, name in ((False, "negative"), (True, "positive")):
            indices = np.flatnonzero(value["truth"] == label)
            selected = deterministic_class_sample(indices, per_parent_class)
            features.append(value["features"][selected])
            targets.append(np.full(per_parent_class, float(label), dtype=np.float64))
            parent_counts[name] = per_parent_class
        receipt[parent] = parent_counts
    return (
        np.concatenate(features).astype(np.float64),
        np.concatenate(targets).astype(np.float64),
        {
            "parent_count": len(parents),
            "per_parent_class": per_parent_class,
            "sample_count": len(parents) * per_parent_class * 2,
            "by_parent": receipt,
        },
    )


def fit_linear_head(
    parents: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    x, y, sampling = balanced_training_matrix(parents)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (x - mean) / scale
    x_tensor = torch.from_numpy(standardized)
    y_tensor = torch.from_numpy(y)
    weight = torch.zeros(standardized.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x_tensor @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_tensor)
        loss = loss + L2_WEIGHT * torch.sum(weight.square())
        loss.backward()
        return loss

    final_loss = float(optimizer.step(closure).detach())
    require(math.isfinite(final_loss), "linear head loss non-finite")
    fitted = {
        "mean": mean,
        "scale": scale,
        "weight": weight.detach().numpy().copy(),
        "bias": float(bias.detach()),
    }
    return fitted, {
        "fit_parents": sorted(parents),
        "sampling": sampling,
        "l2_weight": L2_WEIGHT,
        "maximum_iterations": LBFGS_MAX_ITER,
        "optimizer": "torch_lbfgs_strong_wolfe_float64",
        "final_objective": final_loss,
        "standardized_weight": [float(value) for value in fitted["weight"]],
        "bias": fitted["bias"],
    }


def predict_linear_head(
    model: dict[str, np.ndarray | float],
    features: np.ndarray,
) -> np.ndarray:
    standardized = (features.astype(np.float64) - model["mean"]) / model["scale"]
    logits = standardized @ model["weight"] + float(model["bias"])
    logits = np.clip(logits, -30.0, 30.0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    require(bool(np.isfinite(scores).all()), "linear head score non-finite")
    return scores.astype(np.float32)


def nested_leave_one_parent_out(
    parents: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    require(len(parents) == 6, "outer parent roster drift")
    folds: list[dict[str, Any]] = []
    for held_parent in sorted(parents):
        outer_training = {
            parent: value for parent, value in parents.items() if parent != held_parent
        }
        calibration: dict[str, dict[str, np.ndarray]] = {}
        inner_receipts: list[dict[str, Any]] = []
        for calibration_parent in sorted(outer_training):
            inner_training = {
                parent: value
                for parent, value in outer_training.items()
                if parent != calibration_parent
            }
            inner_model, inner_receipt = fit_linear_head(inner_training)
            calibration[calibration_parent] = {
                "scores": predict_linear_head(
                    inner_model,
                    outer_training[calibration_parent]["features"],
                ),
                "truth": outer_training[calibration_parent]["truth"],
            }
            inner_receipts.append(
                {
                    "calibration_parent": calibration_parent,
                    **inner_receipt,
                }
            )
        low, low_receipt = tri.choose_low_threshold(calibration)
        high, high_receipt = tri.choose_high_threshold(calibration)
        valid = low is not None and high is not None and low < high
        outer_model, outer_receipt = fit_linear_head(outer_training)
        held_scores = predict_linear_head(
            outer_model,
            parents[held_parent]["features"],
        )
        held_metrics = (
            tri.threshold_stats(
                held_scores,
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
                "outer_fit_parents": sorted(outer_training),
                "low_threshold": low,
                "high_threshold": high,
                "threshold_pair_valid": valid,
                "held_metrics": held_metrics,
                "inner_out_of_parent_receipts": inner_receipts,
                "low_selection": low_receipt,
                "high_selection": high_receipt,
                "outer_model": outer_receipt,
            }
        )
    return folds


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(torch.cuda.is_available(), "CUDA required")
    route = json.loads(args.route_result.read_text(encoding="utf-8"))
    require(
        route.get("decision", {}).get("next_successor")
        == "AG_OBSTACLE_SELECTIVE_INTERACTION_HEAD_CANARY_R0",
        "selective interaction head canary not authorized",
    )
    require(
        sha256_file(args.prior_result.resolve()) == EXPECTED_PRIOR_RESULT_SHA256,
        "prior tri-state result SHA drift",
    )
    prior = json.loads(args.prior_result.read_text(encoding="utf-8"))
    require(
        prior.get("status") == "AG_OBSTACLE_EVIDENCE_TRISTATE_CALIBRATION_FAIL_STOP"
        and prior.get("gate", {}).get("evaluable_parent_count") == 0,
        "prior scalar-obstacle terminal drift",
    )
    require(
        sha256_file(args.r21_result.resolve()) == tri.EXPECTED_R21_RESULT_SHA256,
        "R21 result SHA drift",
    )
    result21 = json.loads(args.r21_result.read_text(encoding="utf-8"))
    checkpoint_path = args.r21_checkpoint.resolve()
    require(
        sha256_file(checkpoint_path) == boundary_base.EXPECTED_R21_CHECKPOINT_SHA256,
        "R21 checkpoint SHA drift",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    student, metadata = boundary_base.load_student(
        checkpoint_path,
        boundary_base.EXPECTED_R21_CHECKPOINT_SHA256,
        device,
    )
    observables, inference = extract_observables(
        descriptors,
        student,
        Path(result21["inputs"]["depthart_source"]).resolve(),
        Path(result21["inputs"]["encoder_checkpoint"]).resolve(),
        metadata["architecture"]["feature_profile"],
        device,
        int(metadata["checkpoint"]["seed"]),
    )
    del student
    parents = open_parent_outcomes(descriptors, observables)
    folds = nested_leave_one_parent_out(parents)
    gate = tri.gate_folds(folds)
    strict_gain = (
        gate["evaluable_parent_count"]
        > int(prior["gate"]["evaluable_parent_count"])
        and bool(gate["positive_covered_parents"])
        and bool(gate["verified_negative_covered_parents"])
    )
    checks = {
        **gate["checks"],
        "strict_coverage_gain_over_scalar_obstacle": strict_gain,
    }
    passed = all(checks.values())
    result = {
        "schema": "blindassist_ag_obstacle_selective_interaction_head_canary_result_v1",
        "status": (
            "AG_OBSTACLE_SELECTIVE_INTERACTION_HEAD_CANARY_PASS"
            if passed
            else "AG_OBSTACLE_SELECTIVE_INTERACTION_HEAD_CANARY_FAIL_STOP"
        ),
        "passed": passed,
        "question": (
            "Can a tiny obstacle/support/boundary/depth interaction head provide "
            "parent-disjoint two-sided obstacle evidence when its thresholds are "
            "calibrated only from inner out-of-parent predictions?"
        ),
        "inputs": {
            "route_result": {
                "path": str(args.route_result.resolve()),
                "sha256": sha256_file(args.route_result.resolve()),
            },
            "prior_scalar_terminal": {
                "path": str(args.prior_result.resolve()),
                "sha256": EXPECTED_PRIOR_RESULT_SHA256,
                "evaluable_parent_count": prior["gate"]["evaluable_parent_count"],
            },
            "r21_result": {
                "path": str(args.r21_result.resolve()),
                "sha256": tri.EXPECTED_R21_RESULT_SHA256,
            },
            "r21_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": boundary_base.EXPECTED_R21_CHECKPOINT_SHA256,
            },
            "depthart_checkpoint_sha256": result21["inputs"][
                "encoder_checkpoint_sha256"
            ],
        },
        "protocol": {
            "role": "CONSUMED_CHECKPOINT_HELD_DEVELOPMENT_NESTED_PARENT_DISJOINT",
            "sources": ["arkitscenes", "tum_rgbd"],
            "parent_count": len(parents),
            "frame_count": len(descriptors),
            "prediction_inputs": "RGB_PLUS_K_ONLY",
            "source_outcomes_opened_after_all_observables": True,
            "feature_names": list(FEATURE_NAMES),
            "head": "standardized_linear_logistic_interaction",
            "maximum_samples_per_parent_class": MAX_SAMPLES_PER_PARENT_CLASS,
            "inner_calibration": "each calibration parent predicted by a head excluding that parent",
            "outer_evaluation": "each held parent predicted once by a head fitted on the other five parents",
            "false_negative_rate_cap": tri.FALSE_NEGATIVE_RATE_CAP,
            "false_positive_rate_cap": tri.FALSE_POSITIVE_RATE_CAP,
            "minimum_side_coverage": tri.MIN_SIDE_COVERAGE,
            "minimum_nonzero_parent_fraction": tri.MIN_NONZERO_PARENT_FRACTION,
            "unknown_is_negative": False,
            "source_invalid_trains_factor_value": False,
            "reducer_called": False,
            "task_outcome_read": False,
        },
        "inference": inference,
        "folds": folds,
        "gate": {
            **gate,
            "checks": checks,
            "pass": passed,
            "scalar_obstacle_evaluable_parent_count": prior["gate"][
                "evaluable_parent_count"
            ],
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "decision": {
            "selective_interaction_supported": passed,
            "body_swept_seam_authorized": passed,
            "support_remains_veto_only": True,
            "fresh3_tum_opened": False,
            "default_app_changed": False,
            "next_action_if_pass": (
                "Freeze the six outer models and threshold pairs, then run one consumed "
                "body-swept tri-state mechanics seam with support veto-only."
            ),
            "next_action_if_fail": (
                "Stop the current R21 factor-observable family for obstacle task landing; "
                "do not add another selector without new obstacle supervision or representation."
            ),
        },
        "claim_boundary": (
            "Consumed six-parent Development observability only. A PASS authorizes one "
            "mechanics seam, not task superiority, independent cross-sensor generalization, "
            "deployment, product, default-App, navigation or assistive safety."
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
        default=boundary_base.DEFAULT_R21_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
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


if __name__ == "__main__":
    main()
