"""Truth-late grouped evaluation for the frozen DG-SRF F0 candidate."""

from __future__ import annotations

import argparse
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .common import (
    ARMS,
    PROTOCOL_ID,
    SHAPE,
    SINGLE_ARMS,
    decode_packed_mask,
    ensure_artifact_output,
    read_json,
    read_jsonl,
    resolve_repo_path,
    sha256_array,
    sha256_file,
    validate_config,
    verify_file,
    write_json,
    write_jsonl,
)
from .operators import depth_health_and_proximity, structural_scores


class NotEvaluableError(RuntimeError):
    """The frozen evaluation contract cannot be evaluated."""


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def average_precision_tie_group(
    truth: np.ndarray,
    score: np.ndarray,
) -> float:
    y = np.asarray(truth, dtype=np.uint8).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    if y.shape != s.shape or not np.isfinite(s).all():
        raise NotEvaluableError("AP input is invalid")
    positives = int(np.sum(y))
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise NotEvaluableError("AP group has empty positive or negative class")
    order = np.argsort(-s, kind="stable")
    sorted_y = y[order]
    sorted_s = s[order]
    cumulative_tp = np.cumsum(sorted_y, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_y, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_s[1:] != sorted_s[:-1], True]
    )
    tp = cumulative_tp[group_ends].astype(np.float64)
    fp = cumulative_fp[group_ends].astype(np.float64)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1.0)
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def _component_metrics(
    candidate: np.ndarray,
    truth: np.ndarray,
) -> tuple[int, int, int]:
    candidate_u8 = np.asarray(candidate, dtype=np.uint8)
    truth_u8 = np.asarray(truth, dtype=np.uint8)
    predicted_count, predicted_labels = cv2.connectedComponents(
        candidate_u8,
        connectivity=8,
    )
    truth_count, truth_labels = cv2.connectedComponents(
        truth_u8,
        connectivity=8,
    )
    hit_truth = np.unique(truth_labels[candidate_u8.astype(bool)])
    hit_truth_count = int(np.sum(hit_truth > 0))
    false_predicted_count = 0
    for label in range(1, int(predicted_count)):
        if not np.any(truth_u8[predicted_labels == label]):
            false_predicted_count += 1
    return int(truth_count - 1), hit_truth_count, false_predicted_count


def _frame_stat(context: Mapping[str, Any], candidate: np.ndarray) -> dict[str, Any]:
    truth = context["truth_residual"]
    boundary = context["truth_boundary_residual"]
    obstacle = context["truth_obstacle_residual"]
    baseline = context["baseline_residual"]
    candidate_bool = np.asarray(candidate, dtype=bool)
    truth_components, hit_truth_components, false_components = _component_metrics(
        candidate_bool,
        truth,
    )
    return {
        "view_row_id": context["view_row_id"],
        "session_id": context["session_id"],
        "source_role": context["source_role"],
        "candidate_tp": int(np.sum(candidate_bool & truth)),
        "candidate_fp": int(np.sum(candidate_bool & ~context["truth_full"])),
        "baseline_tp": int(np.sum(baseline & truth)),
        "baseline_fp": int(np.sum(baseline & ~context["truth_full"])),
        "candidate_boundary_tp": int(np.sum(candidate_bool & boundary)),
        "baseline_boundary_tp": int(np.sum(baseline & boundary)),
        "candidate_obstacle_tp": int(np.sum(candidate_bool & obstacle)),
        "baseline_obstacle_tp": int(np.sum(baseline & obstacle)),
        "full_truth_pixels": int(np.sum(context["truth_full"])),
        "residual_truth_pixels": int(np.sum(truth)),
        "truth_component_count": truth_components,
        "hit_truth_component_count": hit_truth_components,
        "false_activation_component_count": false_components,
        "total_pixels": int(candidate_bool.size),
    }


def _safe_ratio(numerator: float, denominator: float, name: str) -> float:
    if denominator <= 0:
        raise NotEvaluableError(f"undefined denominator: {name}")
    return float(numerator / denominator)


def utility_values(
    stats: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not stats:
        raise NotEvaluableError("no frame stats")
    sums = {
        field: sum(int(row[field]) for row in stats)
        for field in (
            "candidate_tp",
            "candidate_fp",
            "baseline_tp",
            "baseline_fp",
            "candidate_boundary_tp",
            "baseline_boundary_tp",
            "candidate_obstacle_tp",
            "baseline_obstacle_tp",
            "full_truth_pixels",
            "truth_component_count",
            "hit_truth_component_count",
            "false_activation_component_count",
            "total_pixels",
        )
    }
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in stats:
        by_session[str(row["session_id"])].append(row)
    session_retentions: list[float] = []
    for session, rows in sorted(by_session.items()):
        candidate_tp = sum(int(row["candidate_tp"]) for row in rows)
        baseline_tp = sum(int(row["baseline_tp"]) for row in rows)
        session_retentions.append(
            _safe_ratio(candidate_tp, baseline_tp, f"baseline_tp:{session}")
        )
    return {
        "fp_pixel_reduction_vs_B": 1.0
        - _safe_ratio(
            sums["candidate_fp"],
            sums["baseline_fp"],
            "baseline_fp",
        ),
        "overall_residual_recall_retention_vs_B": _safe_ratio(
            sums["candidate_tp"],
            sums["baseline_tp"],
            "baseline_tp",
        ),
        "minimum_group_residual_recall_retention_vs_B": min(
            session_retentions
        ),
        "boundary_step_curb_recall_retention_vs_B": _safe_ratio(
            sums["candidate_boundary_tp"],
            sums["baseline_boundary_tp"],
            "baseline_boundary_tp",
        ),
        "obstacle_recall_retention_vs_B": _safe_ratio(
            sums["candidate_obstacle_tp"],
            sums["baseline_obstacle_tp"],
            "baseline_obstacle_tp",
        ),
        "delta_recall_C_minus_A": _safe_ratio(
            sums["candidate_tp"],
            sums["full_truth_pixels"],
            "full_truth_pixels",
        ),
        "delta_false_positive_area_fraction_C_minus_A": _safe_ratio(
            sums["candidate_fp"],
            sums["total_pixels"],
            "total_pixels",
        ),
        "residual_truth_component_recall": _safe_ratio(
            sums["hit_truth_component_count"],
            sums["truth_component_count"],
            "truth_component_count",
        ),
        "false_activation_components_per_frame": _safe_ratio(
            sums["false_activation_component_count"],
            len(stats),
            "frame_count",
        ),
    }


GATE_SPECS = (
    ("fp_pixel_reduction_vs_B", "minimum_fp_pixel_reduction_vs_B", "lower"),
    (
        "overall_residual_recall_retention_vs_B",
        "minimum_overall_residual_recall_retention_vs_B",
        "lower",
    ),
    (
        "minimum_group_residual_recall_retention_vs_B",
        "minimum_group_residual_recall_retention_vs_B",
        "lower",
    ),
    (
        "boundary_step_curb_recall_retention_vs_B",
        "minimum_boundary_step_curb_recall_retention_vs_B",
        "lower",
    ),
    (
        "obstacle_recall_retention_vs_B",
        "minimum_obstacle_recall_retention_vs_B",
        "lower",
    ),
    ("delta_recall_C_minus_A", "minimum_delta_recall_C_minus_A", "lower"),
    (
        "delta_false_positive_area_fraction_C_minus_A",
        "maximum_delta_false_positive_area_fraction_C_minus_A",
        "upper",
    ),
    (
        "residual_truth_component_recall",
        "minimum_residual_truth_component_recall",
        "lower",
    ),
    (
        "false_activation_components_per_frame",
        "maximum_false_activation_components_per_frame",
        "upper",
    ),
)


def gate_report(
    values: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    report: dict[str, Any] = {}
    margins: list[float] = []
    thresholds = config["utility_gates"]
    for value_name, threshold_name, direction in GATE_SPECS:
        value = float(values[value_name])
        threshold = float(thresholds[threshold_name])
        if threshold == 0:
            raise ValueError("zero gate threshold is not supported")
        margin = (
            (value - threshold) / abs(threshold)
            if direction == "lower"
            else (threshold - value) / abs(threshold)
        )
        passed = margin >= 0.0
        report[value_name] = {
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "normalized_margin": margin,
            "passed": passed,
        }
        margins.append(margin)
    return report, min(margins)


def _choose_threshold(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not candidates:
        raise NotEvaluableError("threshold search is empty")
    return max(
        candidates,
        key=lambda row: (
            float(row["minimum_normalized_gate_margin"]),
            float(
                row["utility_values"][
                    "minimum_group_residual_recall_retention_vs_B"
                ]
            ),
            float(row["utility_values"]["fp_pixel_reduction_vs_B"]),
            -float(row["threshold"]),
        ),
    )


def _load_contexts(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    prepared_root: Path,
    producer_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepare_receipt = read_json(prepared_root / "prepare_receipt.json")
    inference_manifest_path = prepared_root / "inference_manifest.jsonl"
    if sha256_file(inference_manifest_path) != prepare_receipt["inference_manifest"]["sha256"]:
        raise ValueError("inference manifest SHA mismatch")
    inference_rows = read_jsonl(inference_manifest_path)

    producer_receipt = read_json(producer_root / "producer_receipt.json")
    if producer_receipt["status"] != "COMPLETE" or producer_receipt["mode"] != "full":
        raise NotEvaluableError("full producer did not complete")
    depth_path = producer_root / "depth_maps.npy"
    depth_index_path = producer_root / "depth_index.jsonl"
    if sha256_file(depth_path) != producer_receipt["depth_map"]["sha256"]:
        raise ValueError("depth map SHA mismatch")
    if sha256_file(depth_index_path) != producer_receipt["depth_index"]["sha256"]:
        raise ValueError("depth index SHA mismatch")
    depth_index = read_jsonl(depth_index_path)
    depths = np.load(depth_path, mmap_mode="r")
    if tuple(depths.shape) != tuple(producer_receipt["depth_map"]["shape"]):
        raise ValueError("depth map shape mismatch")
    if len(inference_rows) != len(depth_index) or len(depth_index) != depths.shape[0]:
        raise ValueError("producer membership mismatch")

    source_rows: dict[str, dict[str, Any]] = {}
    for source in config["input_contract"]["frame_sources"]:
        path = resolve_repo_path(repo_root, source["path"])
        verify_file(path, source["sha256"])
        rows = read_jsonl(path)
        for row in rows:
            view_id = row["view_row_id"]
            if view_id in source_rows:
                raise ValueError(f"duplicate source view id: {view_id}")
            source_rows[view_id] = row

    manifest_spec = config["input_contract"]["canonical_manifest"]
    manifest_path = resolve_repo_path(repo_root, manifest_spec["path"])
    verify_file(manifest_path, manifest_spec["sha256"])
    canonical_rows = read_jsonl(manifest_path)
    canonical_index: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in canonical_rows:
        key = (
            row["session_id"],
            int(row["frame_id"]),
            row["image_sha256"],
        )
        canonical_index.setdefault(key, []).append(row)
    canonical_root = resolve_repo_path(
        repo_root,
        config["input_contract"]["canonical_view_root"],
    )
    direction = config["direction_canary"]["frozen_direction"]
    contexts: list[dict[str, Any]] = []
    for index, (inference, depth_row) in enumerate(
        zip(inference_rows, depth_index)
    ):
        if inference["index"] != index or depth_row["index"] != index:
            raise ValueError("row index drift")
        if inference["view_row_id"] != depth_row["view_row_id"]:
            raise ValueError("producer identity drift")
        source = source_rows[inference["view_row_id"]]
        if source["image_sha256"] != inference["image_sha256"]:
            raise ValueError("source image identity drift")
        key = (
            inference["session_id"],
            int(inference["frame_id"]),
            inference["image_sha256"],
        )
        matches = canonical_index.get(key, [])
        if len(matches) != 1:
            raise ValueError("canonical truth mapping drift")
        canonical = matches[0]
        mask_path = canonical_root / canonical["canonical_mask_path"]
        verify_file(mask_path, canonical["canonical_mask_sha256"])
        truth_ids = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if truth_ids is None or truth_ids.shape != SHAPE:
            raise ValueError("canonical mask decode/shape failure")

        packed = source["packed_masks"]
        if tuple(packed["shape"]) != SHAPE:
            raise ValueError("packed mask shape drift")
        a_mask = decode_packed_mask(packed["A"])
        b_mask = decode_packed_mask(packed["B"])
        boundary_candidate = decode_packed_mask(
            packed["candidate_boundary_step_curb"]
        )
        obstacle_candidate = decode_packed_mask(packed["candidate_obstacle"])
        if not np.array_equal(b_mask, boundary_candidate | obstacle_candidate):
            raise ValueError("B/class-union mismatch")
        if np.any(a_mask & b_mask):
            raise ValueError("B is not residual to A")

        raw = np.asarray(depths[index], dtype=np.float32)
        if sha256_array(raw) != depth_row["raw_depth_array_sha256"]:
            raise ValueError("raw depth row SHA mismatch")
        health, proximity = depth_health_and_proximity(
            raw,
            direction=direction,
            config=config,
        )
        if health != depth_row["health"]:
            raise ValueError("depth health recomputation mismatch")
        scores = structural_scores(
            proximity,
            q=int(health["q"]),
            yolo_mask=a_mask,
            config=config,
        )
        truth_full = np.isin(truth_ids, config["hazard_truth_ids"])
        truth_residual = truth_full & ~a_mask
        contexts.append(
            {
                "index": index,
                "view_row_id": inference["view_row_id"],
                "session_id": inference["session_id"],
                "source_role": inference["source_role"],
                "a_mask": a_mask,
                "baseline_residual": b_mask,
                "truth_full": truth_full,
                "truth_residual": truth_residual,
                "truth_boundary_residual": (truth_ids == 1) & ~a_mask,
                "truth_obstacle_residual": (truth_ids == 2) & ~a_mask,
                "q": int(health["q"]),
                "scores": {arm: scores[arm] for arm in ARMS},
            }
        )
    return contexts, producer_receipt


def _group_ap_rows(
    contexts: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for context in contexts:
        by_session[str(context["session_id"])].append(context)
    rows: list[dict[str, Any]] = []
    macro_values: dict[str, list[float]] = {
        arm: [] for arm in ("B", *ARMS)
    }
    for session, group in sorted(by_session.items()):
        outside = np.concatenate(
            [(~context["a_mask"]).reshape(-1) for context in group]
        )
        truth = np.concatenate(
            [context["truth_residual"].reshape(-1) for context in group]
        )[outside]
        scores: dict[str, np.ndarray] = {
            "B": np.concatenate(
                [
                    context["baseline_residual"].astype(np.float32).reshape(-1)
                    for context in group
                ]
            )[outside]
        }
        for arm in ARMS:
            scores[arm] = np.concatenate(
                [context["scores"][arm].reshape(-1) for context in group]
            )[outside]
        values = {
            arm: average_precision_tie_group(truth, arm_score)
            for arm, arm_score in scores.items()
        }
        for arm, value in values.items():
            macro_values[arm].append(value)
        best_single = max(values[arm] for arm in SINGLE_ARMS)
        rows.append(
            {
                "session_id": session,
                "frame_count": len(group),
                "positive_pixel_count": int(np.sum(truth)),
                "negative_pixel_count": int(truth.size - np.sum(truth)),
                "auprc": values,
                "D4_minus_best_single": values["D4"] - best_single,
                "D4_strictly_beats_best_single": (
                    values["D4"] - best_single
                )
                > epsilon,
                "stable_signal_vs_B": {
                    arm: (values[arm] - values["B"]) > epsilon
                    for arm in ("D1", "D2", "D3", "D4")
                },
            }
        )
    macro = {
        arm: float(np.mean(values)) for arm, values in macro_values.items()
    }
    summary = {
        "macro_auprc": macro,
        "D4_positive_advantage_group_count": sum(
            bool(row["D4_strictly_beats_best_single"]) for row in rows
        ),
        "D4_macro_exceeds_each_single_signal": all(
            macro["D4"] - macro[arm] > epsilon for arm in SINGLE_ARMS
        ),
        "stable_signal": {},
    }
    for arm in ("D1", "D2", "D3", "D4"):
        positive_groups = sum(
            bool(row["stable_signal_vs_B"][arm]) for row in rows
        )
        summary["stable_signal"][arm] = {
            "macro_delta_vs_B": macro[arm] - macro["B"],
            "positive_group_count_vs_B": positive_groups,
            "passed": (
                macro[arm] - macro["B"] > epsilon and positive_groups >= 8
            ),
        }
    return rows, summary


def run_evaluation(
    *,
    repo_root: Path,
    config_path: Path,
    prepared_root: Path,
    producer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    prepared_root = prepared_root.resolve()
    producer_root = producer_root.resolve()
    output_root = ensure_artifact_output(repo_root, output_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)
    config = read_json(config_path)
    validate_config(config)
    prepare_receipt = read_json(prepared_root / "prepare_receipt.json")
    if prepare_receipt["config_sha256"] != sha256_file(config_path):
        raise ValueError("prepare/config identity drift")
    contexts, producer_receipt = _load_contexts(
        repo_root=repo_root,
        config=config,
        prepared_root=prepared_root,
        producer_root=producer_root,
    )
    sessions = sorted({str(context["session_id"]) for context in contexts})
    if len(sessions) != int(
        config["input_contract"]["expected_source_session_count"]
    ):
        raise ValueError("source-session count mismatch")

    epsilon = float(
        config["grouped_evaluation"]["strict_advantage_epsilon"]
    )
    ap_rows, ap_summary = _group_ap_rows(contexts, epsilon=epsilon)
    write_jsonl(output_root / "group_auprc.jsonl", ap_rows)

    thresholds = [
        float(value)
        for value in config["grouped_evaluation"]["threshold_grid"]
    ]
    threshold_stats: dict[float, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        threshold_stats[threshold] = [
            _frame_stat(context, context["scores"]["D4"] >= threshold)
            for context in contexts
        ]

    threshold_search_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_by_session: dict[str, float] = {}
    for held_session in sessions:
        training_indices = [
            index
            for index, context in enumerate(contexts)
            if context["session_id"] != held_session
        ]
        candidates: list[dict[str, Any]] = []
        for threshold in thresholds:
            training_stats = [
                threshold_stats[threshold][index] for index in training_indices
            ]
            values = utility_values(training_stats)
            gates, minimum_margin = gate_report(values, config)
            row = {
                "held_out_session_id": held_session,
                "threshold": threshold,
                "utility_values": values,
                "gate_report": gates,
                "minimum_normalized_gate_margin": minimum_margin,
                "all_gates_passed": all(
                    bool(gate["passed"]) for gate in gates.values()
                ),
            }
            threshold_search_rows.append(row)
            candidates.append(row)
        selected = _choose_threshold(candidates)
        selected_threshold = float(selected["threshold"])
        selected_by_session[held_session] = selected_threshold
        fold_rows.append(
            {
                "held_out_session_id": held_session,
                "training_session_ids": [
                    session for session in sessions if session != held_session
                ],
                "selected_threshold": selected_threshold,
                "selection_rule": config["grouped_evaluation"][
                    "threshold_selection"
                ],
                "inner_status": (
                    "ALL_GATE_OPERATING_POINT_AVAILABLE"
                    if any(bool(row["all_gates_passed"]) for row in candidates)
                    else config["grouped_evaluation"]["no_all_gate_marker"]
                ),
                "selected_minimum_normalized_gate_margin": float(
                    selected["minimum_normalized_gate_margin"]
                ),
            }
        )
    write_jsonl(output_root / "threshold_search.jsonl", threshold_search_rows)
    write_jsonl(output_root / "fold_thresholds.jsonl", fold_rows)

    operating_rows: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        threshold = selected_by_session[str(context["session_id"])]
        row = dict(threshold_stats[threshold][index])
        row["selected_threshold"] = threshold
        row["q"] = int(context["q"])
        operating_rows.append(row)
    write_jsonl(output_root / "frame_operating_metrics.jsonl", operating_rows)
    values = utility_values(operating_rows)
    gates, minimum_margin = gate_report(values, config)
    all_utility_gates_passed = all(
        bool(gate["passed"]) for gate in gates.values()
    )

    q_by_session: dict[str, list[int]] = defaultdict(list)
    for context in contexts:
        q_by_session[str(context["session_id"])].append(int(context["q"]))
    group_coverage = {
        session: float(np.mean(values_q))
        for session, values_q in sorted(q_by_session.items())
    }
    overall_coverage = float(
        np.mean([int(context["q"]) for context in contexts])
    )
    health = config["depth_health"]
    coverage_passed = (
        overall_coverage
        >= float(health["overall_evaluable_frame_coverage_minimum"])
        and min(group_coverage.values())
        >= float(health["minimum_group_evaluable_frame_coverage"])
    )
    composite_passed = (
        bool(ap_summary["D4_macro_exceeds_each_single_signal"])
        and int(ap_summary["D4_positive_advantage_group_count"])
        >= int(
            config["grouped_evaluation"][
                "minimum_D4_positive_advantage_group_count"
            ]
        )
    )
    any_stable_signal = any(
        bool(value["passed"])
        for value in ap_summary["stable_signal"].values()
    )
    if not coverage_passed:
        provisional_terminal = "NOT_EVALUABLE"
    elif all_utility_gates_passed and composite_passed:
        provisional_terminal = "STRUCTURAL_SIGNAL_SUPPORTED_FOR_F1_DESIGN"
    elif any_stable_signal:
        provisional_terminal = "SIGNAL_PRESENT_BUT_COMPOSITE_NOT_READY"
    else:
        provisional_terminal = "STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP"

    role_ap: dict[str, Any] = {}
    for role in sorted({str(context["source_role"]) for context in contexts}):
        role_contexts = [
            context for context in contexts if context["source_role"] == role
        ]
        try:
            _, summary = _group_ap_rows(role_contexts, epsilon=epsilon)
            role_ap[role] = summary
        except NotEvaluableError as error:
            role_ap[role] = {"status": "NOT_EVALUABLE", "reason": str(error)}

    result = {
        "schema_version": (
            "blindassist.dg_srf_image_space_structural_"
            "complementarity_f0.evaluation_result.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE_PENDING_INDEPENDENT_VALIDATION",
        "stage": config["stage"],
        "workflow_profile": config["workflow_profile"],
        "git_head": _git_head(repo_root),
        "config_sha256": sha256_file(config_path),
        "prepare_receipt_sha256": sha256_file(
            prepared_root / "prepare_receipt.json"
        ),
        "frame_count": len(contexts),
        "source_session_count": len(sessions),
        "producer_receipt_sha256": sha256_file(
            producer_root / "producer_receipt.json"
        ),
        "depth_map_sha256": producer_receipt["depth_map"]["sha256"],
        "auprc": ap_summary,
        "cross_fitted_operating_point": {
            "selected_thresholds_by_session": selected_by_session,
            "utility_values": values,
            "gate_report": gates,
            "minimum_normalized_gate_margin": minimum_margin,
            "all_utility_gates_passed": all_utility_gates_passed,
        },
        "depth_health_coverage": {
            "overall": overall_coverage,
            "by_session": group_coverage,
            "passed": coverage_passed,
        },
        "composite_advantage_passed": composite_passed,
        "any_stable_signal": any_stable_signal,
        "provisional_scientific_terminal": provisional_terminal,
        "validation_status": "PENDING",
        "detector_identity_stratified_diagnostic_by_confounded_role": role_ap,
        "limitations": [
            "all 10 source-session groups are SANPO-Real",
            "participant route and parent-capture independence are not evaluable",
            "two YOLO detector identities are fully confounded with source role",
            "B is a frozen binary DDRNet residual mask rather than continuous score",
            "no real-time flicker or event effect is evaluated",
            "component hit uses any positive pixel intersection",
        ],
        "claim_ceiling": config["claim_ceiling"],
        "runtime_seconds": time.perf_counter() - started,
        "output_files": {},
    }
    for name in (
        "group_auprc.jsonl",
        "threshold_search.jsonl",
        "fold_thresholds.jsonl",
        "frame_operating_metrics.jsonl",
    ):
        path = output_root / name
        result["output_files"][name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    write_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_evaluation(
        repo_root=args.repo_root,
        config_path=args.config,
        prepared_root=args.prepared_root,
        producer_root=args.producer_root,
        output_root=args.output_root,
    )
    print(
        f"{result['status']} "
        f"terminal={result['provisional_scientific_terminal']}"
    )


if __name__ == "__main__":
    main()
