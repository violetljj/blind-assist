#!/usr/bin/env python3
"""Test runtime-observable self-consistency signals for safe AG correction.

This consumed-Development canary keeps the metric DepthART prior, correction
expert, and pixel candidate frozen.  It tests three frame-veto scores:

* horizontal-flip equivariance gain (runtime RGB+K only),
* temporal reprojection consistency gain (source pose as an idealized VIO
  diagnostic), and
* the conservative minimum of both gains.

Thresholds are selected outside each held parent.  Source-native depth is
opened only after inference to score MAE and >0.10 m error; it is never an
observable.  A gate may only veto correction and fall back to the prior.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ag_st_tum_rgbd import TumSelectedPayload, load_tum_role_payloads
from build_ag_st_factor_labels import projective_depth_residual
from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_no_regret_selector_bonn import load_selector
from evaluate_ag_st_student_bonn_depth import extract_rgb_only_feature_with_intrinsics
from run_ag_factorwise_no_regret_oracle_parent_gate_canary import evaluate_lane_gate
from train_ag_st_frame_advantage_lcb_router import (
    DEFAULT_TUM_MANIFESTS,
    FrameAdvantageObservation,
)
from train_ag_st_masked_student import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    load_depthart_backbone,
    write_json_exclusive,
)
from train_ag_st_no_regret_selector import SelectorObservation


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PIXEL_SELECTOR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-tum8-source-diverse-global-group-dro-r0/no-regret-selector.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-runtime-correction-gain-observability-canary-r0"
)
PIXEL_CANDIDATE_THRESHOLD = 0.05
MINIMUM_PAIR_COVERAGE = 0.01
SCORE_NAMES = (
    "flip_equivariance_gain",
    "temporal_reprojection_gain",
    "conjunctive_self_consistency_gain",
)


@dataclass(frozen=True)
class RuntimeObservation:
    frame: FrameAdvantageObservation
    rgb_row_index: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    base_depth_m: np.ndarray
    candidate_depth_m: np.ndarray
    flip_base_depth_m: np.ndarray
    flip_candidate_depth_m: np.ndarray
    flip_score: float
    flip_receipt: dict[str, float]


@dataclass(frozen=True)
class FrameSufficientStatistics:
    parent_id: str
    domain: str
    base: dict[str, float]
    expert: dict[str, float]
    candidate: dict[str, float]
    oracle: dict[str, float]
    selected_count: float
    oracle_selected_count: float
    selected_beneficial_count: float
    selected_regret_sum: float


def horizontal_flip_intrinsics(intrinsics: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float32).copy()
    require(matrix.shape == (3, 3), "horizontal-flip intrinsics shape drift")
    require(width > 1, "horizontal-flip image width invalid")
    matrix[0, 2] = float(width - 1) - matrix[0, 2]
    return matrix


def _finite_positive_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(first)
        & np.isfinite(second)
        & (first > 0.0)
        & (second > 0.0)
    )


def _error_statistics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    require(finite.size > 0, "self-consistency denominator empty")
    return {
        "median": float(np.quantile(finite, 0.50)),
        "q90": float(np.quantile(finite, 0.90)),
        "mean": float(finite.mean()),
    }


def relative_consistency_gain(
    base_error: np.ndarray,
    candidate_error: np.ndarray,
    *,
    base_floor: float,
) -> tuple[float, dict[str, float]]:
    require(base_floor > 0.0, "self-consistency base floor invalid")
    base = _error_statistics(base_error)
    candidate = _error_statistics(candidate_error)
    median_gain = (base["median"] - candidate["median"]) / max(
        base["median"], base_floor
    )
    q90_gain = (base["q90"] - candidate["q90"]) / max(
        base["q90"], base_floor
    )
    score = min(median_gain, q90_gain)
    return float(score), {
        "base_median": base["median"],
        "base_q90": base["q90"],
        "candidate_median": candidate["median"],
        "candidate_q90": candidate["q90"],
        "median_relative_gain": float(median_gain),
        "q90_relative_gain": float(q90_gain),
        "score": float(score),
    }


def flip_equivariance_gain(
    base_depth_m: np.ndarray,
    flip_base_depth_m: np.ndarray,
    candidate_depth_m: np.ndarray,
    flip_candidate_depth_m: np.ndarray,
) -> tuple[float, dict[str, float]]:
    valid = (
        _finite_positive_pair(base_depth_m, flip_base_depth_m)
        & _finite_positive_pair(candidate_depth_m, flip_candidate_depth_m)
    )
    require(bool(valid.any()), "flip-equivariance denominator empty")
    base_error = np.abs(
        np.log(base_depth_m[valid]) - np.log(flip_base_depth_m[valid])
    )
    candidate_error = np.abs(
        np.log(candidate_depth_m[valid]) - np.log(flip_candidate_depth_m[valid])
    )
    score, receipt = relative_consistency_gain(
        base_error, candidate_error, base_floor=1e-4
    )
    return score, {**receipt, "valid_fraction": float(valid.mean())}


def _projective_error(
    source_depth: np.ndarray,
    source: RuntimeObservation,
    target_depth: np.ndarray,
    target: RuntimeObservation,
) -> tuple[np.ndarray, float]:
    source_valid = np.isfinite(source_depth) & (source_depth > 0.0)
    target_valid = np.isfinite(target_depth) & (target_depth > 0.0)
    residual, valid = projective_depth_residual(
        source_depth,
        source_valid,
        source.intrinsics,
        source.camera_to_world,
        target_depth,
        target_valid,
        target.intrinsics,
        target.camera_to_world,
    )
    if not bool(valid.any()):
        return np.empty(0, dtype=np.float32), 0.0
    normalized = residual[valid] / np.maximum(source_depth[valid], 0.10)
    return normalized.astype(np.float32, copy=False), float(valid.mean())


def temporal_reprojection_gain(
    source: RuntimeObservation,
    siblings: Iterable[RuntimeObservation],
) -> tuple[float, dict[str, Any]]:
    base_errors: list[np.ndarray] = []
    candidate_errors: list[np.ndarray] = []
    pair_receipts: list[dict[str, Any]] = []
    for target in siblings:
        if target.rgb_row_index == source.rgb_row_index:
            continue
        base_error, base_coverage = _projective_error(
            source.base_depth_m, source, target.base_depth_m, target
        )
        candidate_error, candidate_coverage = _projective_error(
            source.candidate_depth_m,
            source,
            target.candidate_depth_m,
            target,
        )
        pair_coverage = min(base_coverage, candidate_coverage)
        pair_receipts.append(
            {
                "target_rgb_row_index": target.rgb_row_index,
                "base_coverage_fraction": base_coverage,
                "candidate_coverage_fraction": candidate_coverage,
                "joint_coverage_fraction": pair_coverage,
            }
        )
        if (
            pair_coverage >= MINIMUM_PAIR_COVERAGE
            and base_error.size
            and candidate_error.size
        ):
            base_errors.append(base_error)
            candidate_errors.append(candidate_error)
    if not base_errors:
        return -math.inf, {
            "score": None,
            "admissible_pair_count": 0,
            "pairs": pair_receipts,
        }
    score, receipt = relative_consistency_gain(
        np.concatenate(base_errors),
        np.concatenate(candidate_errors),
        base_floor=1e-3,
    )
    return score, {
        **receipt,
        "admissible_pair_count": len(base_errors),
        "pairs": pair_receipts,
    }


def score_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    finite = sorted({float(value) for value in values if math.isfinite(float(value))})
    require(finite, "observability score candidates empty")
    return tuple(finite)


def _metric_sums(
    truth: np.ndarray, predicted: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    error = np.abs(predicted[valid] - truth[valid]).astype(np.float64)
    require(error.size > 0, "frame sufficient-statistics denominator empty")
    return {
        "abs_sum": float(error.sum()),
        "bad_sum": float((error > 0.10).sum()),
        "count": float(error.size),
    }


def frame_sufficient_statistics(
    observation: SelectorObservation,
) -> FrameSufficientStatistics:
    selected_mask = observation.selector_probability >= PIXEL_CANDIDATE_THRESHOLD
    candidate_depth = np.where(
        selected_mask, observation.expert_depth_m, observation.base_depth_m
    )
    base_error = np.abs(observation.base_depth_m - observation.truth_depth_m)
    expert_error = np.abs(observation.expert_depth_m - observation.truth_depth_m)
    oracle_mask = observation.valid & (expert_error < base_error)
    oracle_depth = np.where(
        oracle_mask, observation.expert_depth_m, observation.base_depth_m
    )
    active = selected_mask & observation.valid
    return FrameSufficientStatistics(
        parent_id=observation.parent_id,
        domain=observation.domain,
        base=_metric_sums(
            observation.truth_depth_m, observation.base_depth_m, observation.valid
        ),
        expert=_metric_sums(
            observation.truth_depth_m, observation.expert_depth_m, observation.valid
        ),
        candidate=_metric_sums(
            observation.truth_depth_m, candidate_depth, observation.valid
        ),
        oracle=_metric_sums(
            observation.truth_depth_m, oracle_depth, observation.valid
        ),
        selected_count=float(active.sum()),
        oracle_selected_count=float(oracle_mask.sum()),
        selected_beneficial_count=float((active & (expert_error < base_error)).sum()),
        selected_regret_sum=float(
            (expert_error[active] - base_error[active]).sum(dtype=np.float64)
        ),
    )


def summarize_gated_sufficient_statistics(
    rows: list[FrameSufficientStatistics], open_frames: np.ndarray
) -> dict[str, Any]:
    require(rows and len(rows) == len(open_frames), "sufficient-statistics gate drift")
    accumulators: dict[str, dict[str, Any]] = {}
    for row, opened in zip(rows, open_frames, strict=True):
        values = accumulators.setdefault(
            row.parent_id,
            {
                "domain": row.domain,
                "base": defaultdict(float),
                "expert": defaultdict(float),
                "selected": defaultdict(float),
                "oracle": defaultdict(float),
                "selected_count": 0.0,
                "oracle_selected_count": 0.0,
                "selected_beneficial_count": 0.0,
                "selected_regret_sum": 0.0,
            },
        )
        require(values["domain"] == row.domain, "sufficient parent domain drift")
        selected = row.candidate if bool(opened) else row.base
        for name, metric in (
            ("base", row.base),
            ("expert", row.expert),
            ("selected", selected),
            ("oracle", row.oracle),
        ):
            for key, value in metric.items():
                values[name][key] += value
        if bool(opened):
            values["selected_count"] += row.selected_count
            values["selected_beneficial_count"] += row.selected_beneficial_count
            values["selected_regret_sum"] += row.selected_regret_sum
        values["oracle_selected_count"] += row.oracle_selected_count

    parent_rows: list[dict[str, Any]] = []
    for parent_id, values in sorted(accumulators.items()):
        count = values["base"]["count"]
        require(count > 0.0, "sufficient parent denominator empty")

        def finalize(name: str) -> dict[str, float]:
            metric = values[name]
            return {
                "mae_m": metric["abs_sum"] / metric["count"],
                "bad_gt_0_10_m_fraction": metric["bad_sum"] / metric["count"],
            }

        selected_count = values["selected_count"]
        parent_rows.append(
            {
                "parent_id": parent_id,
                "domain": values["domain"],
                "base": finalize("base"),
                "expert": finalize("expert"),
                "selected": finalize("selected"),
                "oracle": finalize("oracle"),
                "selected_coverage_fraction": selected_count / count,
                "oracle_coverage_fraction": values["oracle_selected_count"] / count,
                "selected_beneficial_fraction": (
                    values["selected_beneficial_count"] / selected_count
                    if selected_count > 0.0
                    else None
                ),
                "selected_mean_regret_m": (
                    values["selected_regret_sum"] / selected_count
                    if selected_count > 0.0
                    else 0.0
                ),
            }
        )

    def aggregate(parent_values: list[dict[str, Any]]) -> dict[str, Any]:
        require(parent_values, "sufficient aggregation rows empty")
        result: dict[str, Any] = {
            name: {
                "mae_m": float(np.mean([row[name]["mae_m"] for row in parent_values])),
                "bad_gt_0_10_m_fraction": float(
                    np.mean(
                        [
                            row[name]["bad_gt_0_10_m_fraction"]
                            for row in parent_values
                        ]
                    )
                ),
            }
            for name in ("base", "expert", "selected", "oracle")
        }
        result["selected_coverage_fraction"] = float(
            np.mean([row["selected_coverage_fraction"] for row in parent_values])
        )
        result["oracle_coverage_fraction"] = float(
            np.mean([row["oracle_coverage_fraction"] for row in parent_values])
        )
        result["selected_mae_delta_vs_base_m"] = (
            result["selected"]["mae_m"] - result["base"]["mae_m"]
        )
        result["selected_bad_delta_vs_base"] = (
            result["selected"]["bad_gt_0_10_m_fraction"]
            - result["base"]["bad_gt_0_10_m_fraction"]
        )
        result["oracle_mae_delta_vs_base_m"] = (
            result["oracle"]["mae_m"] - result["base"]["mae_m"]
        )
        result["oracle_bad_delta_vs_base"] = (
            result["oracle"]["bad_gt_0_10_m_fraction"]
            - result["base"]["bad_gt_0_10_m_fraction"]
        )
        result["improved_parent_count"] = sum(
            row["selected"]["mae_m"] < row["base"]["mae_m"]
            for row in parent_values
        )
        return result

    domains = sorted({row["domain"] for row in parent_rows})
    return {
        "threshold": PIXEL_CANDIDATE_THRESHOLD,
        "parent_macro": aggregate(parent_rows),
        "by_domain": {
            domain: aggregate(
                [row for row in parent_rows if row["domain"] == domain]
            )
            for domain in domains
        },
        "per_parent": parent_rows,
    }


def _summary_at_threshold(
    rows: list[FrameSufficientStatistics],
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = summarize_gated_sufficient_statistics(rows, scores >= threshold)
    return summary, evaluate_lane_gate(summary, "selected")


def choose_threshold(
    rows: list[FrameSufficientStatistics], scores: np.ndarray
) -> dict[str, Any]:
    tested: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in score_thresholds(scores):
        summary, gate = _summary_at_threshold(rows, scores, threshold)
        row = {
            "threshold": threshold,
            "gate": gate,
            "parent_macro": summary["parent_macro"],
        }
        tested.append(row)
        if gate["pass"]:
            eligible.append(row)
    if not eligible:
        return {"threshold": None, "tested": tested, "selected": None}
    selected = min(
        eligible,
        key=lambda row: (
            row["parent_macro"]["selected_mae_delta_vs_base_m"],
            row["parent_macro"]["selected_bad_delta_vs_base"],
            -row["parent_macro"]["selected_coverage_fraction"],
            -row["threshold"],
        ),
    )
    return {"threshold": selected["threshold"], "tested": tested, "selected": selected}


def leave_one_parent_out_gate(
    rows: list[FrameSufficientStatistics], scores: np.ndarray
) -> dict[str, Any]:
    require(len(rows) == len(scores), "LOPO score count drift")
    parents = sorted({row.parent_id for row in rows})
    require(len(parents) >= 4, "LOPO parent count too small")
    oof_open = np.zeros(len(rows), dtype=bool)
    folds: list[dict[str, Any]] = []
    for held_parent in parents:
        train_indices = [
            index for index, row in enumerate(rows) if row.parent_id != held_parent
        ]
        held_indices = [
            index for index, row in enumerate(rows) if row.parent_id == held_parent
        ]
        require(train_indices and held_indices, "LOPO fold empty")
        train_rows = [rows[index] for index in train_indices]
        train_scores = scores[train_indices]
        selection = choose_threshold(train_rows, train_scores)
        threshold = selection["threshold"]
        if threshold is not None:
            oof_open[held_indices] = scores[held_indices] >= float(threshold)
        folds.append(
            {
                "held_parent": held_parent,
                "training_parent_count": len(parents) - 1,
                "selected_threshold": threshold,
                "training_gate_pass": bool(selection["selected"] is not None),
            }
        )
    summary = summarize_gated_sufficient_statistics(rows, oof_open)
    gate = evaluate_lane_gate(summary, "selected")
    final = choose_threshold(rows, scores)
    return {
        "folds": folds,
        "summary": summary,
        "gate": gate,
        "final_threshold": final["threshold"],
        "final_calibration": final["selected"],
        "pass": gate["pass"] and final["selected"] is not None,
    }


def _infer_depth_bundle(
    payload: TumSelectedPayload,
    extractor: torch.nn.Module,
    pixel_selector: torch.nn.Module,
    expert: torch.nn.Module,
    feature_profile: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> RuntimeObservation:
    rgb = payload.load_rgb()
    feature, base = extract_rgb_only_feature_with_intrinsics(
        extractor, rgb, payload.intrinsics, feature_profile, device, amp_dtype
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        expert_outputs = expert(feature, base, rgb.shape[:2])
        selector_outputs = pixel_selector(
            feature,
            base,
            expert_outputs["depth_m"],
            expert_outputs["depth_identity_gate"],
            rgb.shape[:2],
        )
    base_np = base[0, 0].float().cpu().numpy()
    expert_np = expert_outputs["depth_m"][0, 0].float().cpu().numpy()
    probability_np = (
        selector_outputs["selector_probability"][0, 0].float().cpu().numpy()
    )
    candidate_np = np.where(
        probability_np >= PIXEL_CANDIDATE_THRESHOLD, expert_np, base_np
    )

    flipped_rgb = np.ascontiguousarray(rgb[:, ::-1])
    flipped_intrinsics = horizontal_flip_intrinsics(payload.intrinsics, rgb.shape[1])
    flip_feature, flip_base = extract_rgb_only_feature_with_intrinsics(
        extractor,
        flipped_rgb,
        flipped_intrinsics,
        feature_profile,
        device,
        amp_dtype,
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        flip_expert_outputs = expert(flip_feature, flip_base, rgb.shape[:2])
        flip_selector_outputs = pixel_selector(
            flip_feature,
            flip_base,
            flip_expert_outputs["depth_m"],
            flip_expert_outputs["depth_identity_gate"],
            rgb.shape[:2],
        )
    flip_base_np = flip_base[0, 0].float().cpu().numpy()[:, ::-1].copy()
    flip_expert_np = (
        flip_expert_outputs["depth_m"][0, 0].float().cpu().numpy()[:, ::-1].copy()
    )
    flip_probability_np = (
        flip_selector_outputs["selector_probability"][0, 0]
        .float()
        .cpu()
        .numpy()[:, ::-1]
        .copy()
    )
    flip_candidate_np = np.where(
        flip_probability_np >= PIXEL_CANDIDATE_THRESHOLD,
        flip_expert_np,
        flip_base_np,
    )
    flip_score, flip_receipt = flip_equivariance_gain(
        base_np, flip_base_np, candidate_np, flip_candidate_np
    )

    truth, truth_valid = payload.load_depth()
    metric_valid = truth_valid & np.isfinite(truth)
    require(bool(metric_valid.any()), "runtime-observability truth denominator empty")
    base_error = np.abs(base_np[metric_valid] - truth[metric_valid])
    candidate_error = np.abs(candidate_np[metric_valid] - truth[metric_valid])
    frame = FrameAdvantageObservation(
        parent_id=payload.parent_id,
        domain="TUM_RGBD",
        observable=np.asarray((flip_score,), dtype=np.float32),
        selector=SelectorObservation(
            parent_id=payload.parent_id,
            domain="TUM_RGBD",
            truth_depth_m=truth,
            valid=metric_valid,
            base_depth_m=base_np,
            expert_depth_m=expert_np,
            selector_probability=probability_np,
        ),
        mae_advantage_m=float(base_error.mean() - candidate_error.mean()),
        bad_rate_advantage=float(
            (base_error > 0.10).mean() - (candidate_error > 0.10).mean()
        ),
    )
    return RuntimeObservation(
        frame=frame,
        rgb_row_index=payload.rgb.row_index,
        intrinsics=np.asarray(payload.intrinsics, dtype=np.float64),
        camera_to_world=np.asarray(payload.camera_to_world, dtype=np.float64),
        base_depth_m=base_np,
        candidate_depth_m=candidate_np,
        flip_base_depth_m=flip_base_np,
        flip_candidate_depth_m=flip_candidate_np,
        flip_score=flip_score,
        flip_receipt=flip_receipt,
    )


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "runtime-observability output collision")
    output_dir.mkdir(parents=True, exist_ok=False)
    require(torch.cuda.is_available(), "runtime-observability canary requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pixel_checkpoint = args.pixel_selector_checkpoint.resolve()
    (
        pixel_payload,
        pixel_selector,
        checkpoint_pixel_threshold,
        expert_path,
        _expert_payload,
        expert,
    ) = load_selector(pixel_checkpoint, device)
    require(
        checkpoint_pixel_threshold > 1.0,
        "source-diverse selector checkpoint is not frozen fallback",
    )
    manifests = [value.resolve() for value in args.tum_cohort_manifest]
    payloads: list[TumSelectedPayload] = []
    source_receipts: list[dict[str, Any]] = []
    for manifest in manifests:
        for role in ("fit", "evaluation"):
            role_payloads, receipt = load_tum_role_payloads(manifest, role)
            payloads.extend(role_payloads)
            source_receipts.append(receipt)
    parents = sorted({payload.parent_id for payload in payloads})
    require(len(parents) == 14, "runtime-observability TUM parent roster drift")
    require(len(payloads) == 3 * len(parents), "runtime-observability frame roster drift")

    extractor, scan = load_depthart_backbone(
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(pixel_payload["seed"]),
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    feature_profile = pixel_payload["expert"]["architecture"]["feature_profile"]
    observations = [
        _infer_depth_bundle(
            payload,
            extractor,
            pixel_selector,
            expert,
            feature_profile,
            device,
            amp_dtype,
        )
        for payload in payloads
    ]
    del extractor
    torch.cuda.empty_cache()

    by_parent: dict[str, list[RuntimeObservation]] = {
        parent: sorted(
            [row for row in observations if row.frame.parent_id == parent],
            key=lambda row: row.rgb_row_index,
        )
        for parent in parents
    }
    temporal_scores: dict[tuple[str, int], float] = {}
    temporal_receipts: dict[tuple[str, int], dict[str, Any]] = {}
    for parent, rows in by_parent.items():
        for row in rows:
            score, receipt = temporal_reprojection_gain(row, rows)
            temporal_scores[(parent, row.rgb_row_index)] = score
            temporal_receipts[(parent, row.rgb_row_index)] = receipt

    frame_rows = [frame_sufficient_statistics(row.frame.selector) for row in observations]
    score_arrays = {
        "flip_equivariance_gain": np.asarray(
            [row.flip_score for row in observations], dtype=np.float64
        ),
        "temporal_reprojection_gain": np.asarray(
            [
                temporal_scores[(row.frame.parent_id, row.rgb_row_index)]
                for row in observations
            ],
            dtype=np.float64,
        ),
    }
    score_arrays["conjunctive_self_consistency_gain"] = np.minimum(
        score_arrays["flip_equivariance_gain"],
        score_arrays["temporal_reprojection_gain"],
    )
    candidate_results = {
        name: leave_one_parent_out_gate(frame_rows, score_arrays[name])
        for name in SCORE_NAMES
    }
    eligible = [name for name in SCORE_NAMES if candidate_results[name]["pass"]]
    if eligible:
        chosen_name = min(
            eligible,
            key=lambda name: (
                candidate_results[name]["gate"]["parent_macro_mae_delta_vs_base_m"],
                candidate_results[name]["gate"][
                    "parent_macro_bad_gt_0_10_m_delta_vs_base"
                ],
                -candidate_results[name]["gate"]["parent_macro_coverage_fraction"],
                SCORE_NAMES.index(name),
            ),
        )
        status = "AG_RUNTIME_CORRECTION_GAIN_OBSERVABILITY_LOPO_PASS"
        successor = "FREEZE_CANDIDATE_AND_EVALUATE_PRELOCKED_FRESH_TUM3"
    else:
        chosen_name = None
        status = "AG_RUNTIME_CORRECTION_GAIN_OBSERVABILITY_LOPO_FAIL_STOP"
        successor = "STOP_SAME_CORRECTION_EXPERT_RETAIN_BOUNDARY_ROUTE"

    result = {
        "schema": "blindassist_ag_runtime_correction_gain_observability_canary_v1",
        "status": status,
        "mode": "PROJECT_CONSUMED_DEVELOPMENT",
        "question": (
            "Can flip-equivariance and pose-assisted temporal self-consistency "
            "observe when the frozen correction candidate is jointly no-regret?"
        ),
        "frozen_candidate": {
            "pixel_selector_checkpoint": str(pixel_checkpoint),
            "pixel_selector_sha256": sha256_file(pixel_checkpoint),
            "checkpoint_fallback_threshold": checkpoint_pixel_threshold,
            "pixel_candidate_threshold": PIXEL_CANDIDATE_THRESHOLD,
            "expert_checkpoint": str(expert_path),
            "expert_sha256": sha256_file(expert_path),
            "modified": False,
        },
        "inputs": {
            "manifests": [
                {"path": str(path), "sha256": sha256_file(path)} for path in manifests
            ],
            "source_receipts": source_receipts,
            "parent_count": len(parents),
            "frame_count": len(observations),
            "source_depth_runtime_input": False,
            "task_outcome_runtime_input": False,
            "flip_candidate_runtime_inputs": "RGB_PLUS_K_ONLY",
            "temporal_candidate_pose_role": (
                "SOURCE_GROUNDTRUTH_POSE_AS_IDEALIZED_VIO_DIAGNOSTIC_NOT_CURRENT_APP_INPUT"
            ),
        },
        "candidate_order": list(SCORE_NAMES),
        "candidates": candidate_results,
        "chosen_candidate": (
            {
                "name": chosen_name,
                "threshold": candidate_results[chosen_name]["final_threshold"],
            }
            if chosen_name is not None
            else None
        ),
        "per_frame": [
            {
                "parent_id": row.frame.parent_id,
                "rgb_row_index_zero_based": row.rgb_row_index,
                "truth_mae_advantage_m": row.frame.mae_advantage_m,
                "truth_bad_rate_advantage": row.frame.bad_rate_advantage,
                "flip_equivariance_gain": row.flip_score,
                "temporal_reprojection_gain": temporal_scores[
                    (row.frame.parent_id, row.rgb_row_index)
                ],
                "conjunctive_self_consistency_gain": min(
                    row.flip_score,
                    temporal_scores[(row.frame.parent_id, row.rgb_row_index)],
                ),
                "flip_receipt": row.flip_receipt,
                "temporal_receipt": temporal_receipts[
                    (row.frame.parent_id, row.rgb_row_index)
                ],
            }
            for row in observations
        ],
        "decision": {
            "chosen_candidate": chosen_name,
            "fresh_tum3_evaluation_authorized": chosen_name is not None,
            "next_successor": successor,
            "full_fallback_is_success": False,
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "depthart_scan": scan,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "total_seconds": time.perf_counter() - started,
        },
        "claim_boundary": (
            "Consumed Development mechanism evidence only. Pose-assisted temporal "
            "scores are an idealized VIO diagnostic, not a current Android capability."
        ),
    }
    write_json_exclusive(output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": status,
                "chosen_candidate": result["chosen_candidate"],
                "candidate_gates": {
                    name: candidate_results[name]["gate"] for name in SCORE_NAMES
                },
                "output": str(output_dir / "result.json"),
                "total_seconds": result["execution"]["total_seconds"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pixel-selector-checkpoint", type=Path, default=DEFAULT_PIXEL_SELECTOR
    )
    parser.add_argument(
        "--tum-cohort-manifest",
        type=Path,
        action="append",
        default=list(DEFAULT_TUM_MANIFESTS),
    )
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
