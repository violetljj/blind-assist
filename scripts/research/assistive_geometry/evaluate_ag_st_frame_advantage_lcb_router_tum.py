#!/usr/bin/env python3
"""Evaluate the frozen AG frame-advantage LCB router on held TUM parents."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_no_regret_selector_bonn import load_selector
from run_ag_factorwise_no_regret_oracle_parent_gate_canary import evaluate_lane_gate
from train_ag_st_masked_student import DEFAULT_DEPTHART_CHECKPOINT, DEFAULT_DEPTHART_SOURCE, write_json_exclusive
from train_ag_st_no_regret_selector import extract_tum_anchor_frames, summarize_selector_observations
from train_ag_st_frame_advantage_lcb_router import (
    FRAME_LCB_SCHEMA,
    FrameAdvantageQuantileRouter,
    collect_frame_observations,
    ensemble_lcb_scores,
    gated_selector_observations,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTER_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-frame-advantage-lcb-router-tum8-r2/frame-advantage-lcb-router.pt"
)
DEFAULT_COHORT_MANIFEST = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_THIRD_TEACHER_COHORT_R2_2026-08-10.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-frame-advantage-lcb-router-tum8-fresh3-evaluation-r0.json"
)


def load_frame_router(
    checkpoint_path: Path, device: torch.device
) -> tuple[
    dict[str, Any],
    list[FrameAdvantageQuantileRouter],
    np.ndarray,
    np.ndarray,
]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "frame LCB checkpoint root invalid")
    require(payload.get("schema") == FRAME_LCB_SCHEMA, "frame LCB schema drift")
    architecture = payload.get("architecture", {})
    state_dicts = payload.get("state_dicts")
    require(isinstance(state_dicts, list) and state_dicts, "frame LCB ensemble missing")
    require(
        len(state_dicts) == int(architecture["ensemble_size"]),
        "frame LCB ensemble size drift",
    )
    models: list[FrameAdvantageQuantileRouter] = []
    for state_dict in state_dicts:
        model = FrameAdvantageQuantileRouter(
            int(architecture["input_features"]), int(architecture["hidden_channels"])
        ).to(device)
        incompatible = model.load_state_dict(state_dict, strict=True)
        require(
            not incompatible.missing_keys and not incompatible.unexpected_keys,
            "frame LCB state drift",
        )
        model.eval().requires_grad_(False)
        models.append(model)
    normalization = payload.get("normalization", {})
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    require(
        mean.shape == std.shape == (int(architecture["input_features"]),),
        "frame LCB normalization drift",
    )
    require(np.all(std > 0.0), "frame LCB normalization scale invalid")
    return payload, models, mean, std


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    checkpoint_path = args.router_checkpoint.resolve()
    cohort_manifest = args.cohort_manifest.resolve()
    output = args.output.resolve()
    require(checkpoint_path.is_file(), "frame LCB checkpoint missing")
    require(not output.exists(), "frame LCB evaluation output collision")
    require(torch.cuda.is_available(), "frame LCB evaluation requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    router_payload, models, mean, std = load_frame_router(checkpoint_path, device)
    pixel_receipt = router_payload.get("pixel_selector", {})
    pixel_checkpoint = Path(str(pixel_receipt["path"])).resolve()
    require(pixel_checkpoint.is_file(), "frozen pixel selector missing")
    require(
        sha256_file(pixel_checkpoint) == pixel_receipt.get("sha256"),
        "frozen pixel selector hash drift",
    )
    (
        pixel_payload,
        pixel_selector,
        checkpoint_pixel_threshold,
        expert_path,
        _expert_payload,
        expert,
    ) = load_selector(pixel_checkpoint, device)
    # Keep older diagnostic checkpoints replayable.  Newer checkpoints split
    # the selector's frozen fallback from an explicitly bound candidate cutoff.
    recorded_checkpoint_threshold = float(
        pixel_receipt["checkpoint_threshold"]
        if "checkpoint_threshold" in pixel_receipt
        else pixel_receipt["threshold"]
    )
    require(
        abs(checkpoint_pixel_threshold - recorded_checkpoint_threshold) <= 1e-12,
        "pixel checkpoint threshold drift",
    )
    pixel_threshold = float(
        pixel_receipt.get("candidate_threshold", recorded_checkpoint_threshold)
    )
    require(0.0 < pixel_threshold <= 1.0, "pixel candidate threshold invalid")
    frame_threshold = router_payload.get("frame_score_threshold")
    require(frame_threshold is not None, "frame LCB checkpoint is full fallback")
    frame_threshold = float(frame_threshold)
    frames, source_provenance = extract_tum_anchor_frames(
        cohort_manifest,
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(pixel_payload["seed"]),
        cohort_role="evaluation",
    )
    evaluation_parents = sorted({frame.descriptor.parent_id for frame in frames})
    split = router_payload.get("split", {})
    consumed_parents = {
        str(parent)
        for key in ("fit_parents", "calibration_parents")
        for parent in split.get(key, [])
    }
    overlap = sorted(consumed_parents & set(evaluation_parents))
    require(not overlap, f"frame LCB evaluation parent overlap: {overlap}")
    parent_domains = {parent: "TUM_RGBD" for parent in evaluation_parents}
    frame_rows = collect_frame_observations(
        frames,
        parent_domains,
        pixel_selector,
        expert,
        pixel_threshold,
        device,
    )
    neural_lower, _neural_scores = ensemble_lcb_scores(
        models,
        frame_rows,
        mean,
        std,
        device,
        disagreement_multiplier=float(
            router_payload["architecture"]["disagreement_multiplier"]
        ),
    )
    knn_support = router_payload.get("knn_support", {})
    neighbors = knn_support.get("neighbors")
    require(neighbors is not None, "frame LCB checkpoint lacks KNN support gate")
    fit_observables = np.asarray(knn_support.get("fit_observables"), dtype=np.float32)
    fit_targets = np.asarray(
        knn_support.get("fit_targets_normalized"), dtype=np.float32
    )
    require(
        fit_observables.ndim == 2
        and fit_targets.shape == (fit_observables.shape[0], 2),
        "frame KNN support bank drift",
    )
    fit_matrix = (fit_observables - mean[None, :]) / std[None, :]
    query_matrix = np.stack(
        [(row.observable - mean) / std for row in frame_rows], axis=0
    ).astype(np.float32)
    knn_lower_rows: list[np.ndarray] = []
    nearest_distances: list[float] = []
    for query in query_matrix:
        distance = np.mean((fit_matrix - query[None, :]) ** 2, axis=1)
        ranked = np.argsort(distance, kind="stable")[: int(neighbors)]
        require(len(ranked) == int(neighbors), "frame KNN evaluation bank too small")
        knn_lower_rows.append(
            np.quantile(
                fit_targets[ranked],
                float(knn_support["lower_quantile"]),
                axis=0,
            ).astype(np.float32)
        )
        nearest_distances.append(float(distance[ranked[0]]))
    knn_lower = np.stack(knn_lower_rows, axis=0)
    lower = np.minimum(neural_lower, knn_lower)
    scores = lower.min(axis=1)
    observations = gated_selector_observations(frame_rows, scores, frame_threshold)
    summary = summarize_selector_observations(observations, pixel_threshold)
    gate = evaluate_lane_gate(summary, "selected")
    status = (
        "AG_FRAME_ADVANTAGE_LCB_FRESH_TUM_PARENT_GATE_PASS"
        if gate["pass"]
        else "AG_FRAME_ADVANTAGE_LCB_FRESH_TUM_PARENT_GATE_FAIL"
    )
    result = {
        "schema": "blindassist_ag_frame_advantage_lcb_tum_evaluation_result_v1",
        "status": status,
        "mode": "PARENT_DISJOINT_DEVELOPMENT_EVALUATION",
        "question": (
            "Does the frozen two-stage pixel-selector plus frame-advantage LCB "
            "retain nonzero correction coverage without parent-level depth regret?"
        ),
        "provenance": {
            "router_checkpoint_path": str(checkpoint_path),
            "router_checkpoint_sha256": sha256_file(checkpoint_path),
            "pixel_selector_checkpoint_path": str(pixel_checkpoint),
            "pixel_selector_checkpoint_sha256": sha256_file(pixel_checkpoint),
            "expert_checkpoint_path": str(expert_path),
            "expert_checkpoint_sha256": sha256_file(expert_path),
            "cohort": source_provenance,
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "parent_firewall": {
            "router_fit_and_calibration_parents": sorted(consumed_parents),
            "evaluation_parents": evaluation_parents,
            "overlap": overlap,
            "evaluation_used_for_training_or_threshold": False,
            "project_global_fresh": False,
            "selector_route_fresh": True,
            "prior_non_selector_teacher_use_disclosed": True,
        },
        "frozen_decision": {
            "pixel_threshold": pixel_threshold,
            "frame_score_threshold": frame_threshold,
            "threshold_selected_in_evaluation": False,
            "training_performed_in_evaluation": False,
        },
        "frame_decisions": [
            {
                "parent_id": row.parent_id,
                "mae_advantage_lcb_normalized": float(bound[0]),
                "bad_rate_advantage_lcb_normalized": float(bound[1]),
                "neural_joint_score": float(neural.min()),
                "knn_joint_score": float(knn.min()),
                "nearest_support_distance": distance,
                "joint_score": float(score),
                "frame_gate_open": float(score) >= frame_threshold,
                "candidate_truth_mae_advantage_m": row.mae_advantage_m,
                "candidate_truth_bad_rate_advantage": row.bad_rate_advantage,
            }
            for row, bound, neural, knn, distance, score in zip(
                frame_rows,
                lower,
                neural_lower,
                knn_lower,
                nearest_distances,
                scores,
                strict=True,
            )
        ],
        "metrics": summary,
        "gate": gate,
        "decision": {
            "parent_gate_pass": gate["pass"],
            "nonzero_correction_coverage": summary["parent_macro"][
                "selected_coverage_fraction"
            ]
            > 0.0,
            "promotion_authorized": False,
            "next_if_pass": (
                "LOCK_NEW_SOURCE_OR_SENSOR_CONFIRMATION_BEFORE_OPENING_OUTCOME"
            ),
            "next_if_fail": (
                "STOP_DEPTH_ROUTER_OR_ADD_NEW_OBSERVABLE_NOT_MORE_THRESHOLD_TUNING"
            ),
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "total_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "development_only": True,
            "source_depth_is_runtime_input": False,
            "boundary_support_unknown_modified": False,
            "reducer_called": False,
            "confirmation_claim_authorized": False,
            "deployment_product_safety_claim_authorized": False,
        },
    }
    write_json_exclusive(output, result)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "evaluation_parents": evaluation_parents,
                "parent_macro": summary["parent_macro"],
                "gate": gate,
                "total_seconds": result["execution"]["total_seconds"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router-checkpoint", type=Path, default=DEFAULT_ROUTER_CHECKPOINT
    )
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
