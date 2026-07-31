"""Run the frozen post-primary R0.1 conditional-gating shadow ablation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    connected_components,
)
from scripts.research.dual_loop_segmentation_conditional_gating.conditional_gating import (
    BASELINE_ID,
    PRIMARY_CANDIDATE_ID,
    SHADOW_CANDIDATE_IDS,
    SHADOW_CLASS_TEMPORAL_ID,
    SHADOW_MULTI_NEGATIVE_ID,
    _aggregate_arm_report,
    _current_git_head,
    _decision_checks,
    _frame_arm_metrics,
    _load_bound_jsonl,
    _validate_membership,
    _verify_output_scope,
    _write_json,
    _write_jsonl,
    apply_frozen_candidate_to_component,
    causal_two_of_three,
    decode_packed_mask,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    upper_head_band,
)


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1"
MODE = "POST_R0_FORWARD_SHADOW_DIAGNOSTIC"
SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_conditional_gating.shadow_result.v1"
)
FIXED_TERMINAL = "POST_TERMINAL_SHADOW_ABLATION_COMPLETE_DIAGNOSTIC_ONLY"
EXPECTED_SHADOW_ORDER = [
    "CLASS_CONDITIONAL_TEMPORAL",
    "MULTI_NEGATIVE",
]
RECALL_GUARD_NAMES = [
    "overall_recall_retention",
    "minimum_session_recall_retention",
    "boundary_step_curb_recall_retention",
    "obstacle_recall_retention",
]
FP_GATE_NAME = "false_positive_reduction"
FIXED_REFERENCE_IDS = [
    "REFERENCE_CAUSAL_2_OF_3_UNION",
    "REFERENCE_CONFIDENCE_GE_0_65",
    "CLASS_CONDITIONED_MULTI_NEGATIVE",
]


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_bound_json(repo_root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(repo_root, str(binding["path"]))
    observed = sha256_file(path)
    if observed != str(binding["sha256"]):
        raise ValueError(f"bound JSON SHA mismatch: {path}")
    return read_json(path)


def _load_bound_jsonl(
    repo_root: Path,
    binding: dict[str, Any],
) -> list[dict[str, Any]]:
    path = _resolve(repo_root, str(binding["path"]))
    observed = sha256_file(path)
    if observed != str(binding["sha256"]):
        raise ValueError(f"bound JSONL SHA mismatch: {path}")
    rows = read_jsonl(path)
    if len(rows) != int(binding["row_count"]):
        raise ValueError(f"bound JSONL row-count mismatch: {path}")
    return rows


def _validate_shadow_config_shape(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected shadow protocol_id")
    if config.get("stage") != "DEVELOPMENT_STANDARD":
        raise ValueError("shadow ablation must remain Development")
    if config.get("mode") != MODE:
        raise ValueError("shadow mode drifted")
    if config.get("analysis_role") != "POST_PRIMARY_TERMINAL_SHADOW_DIAGNOSTIC_ONLY":
        raise ValueError("shadow analysis role drifted")
    if config.get("freeze_status") != "FROZEN_PRE_OUTCOME":
        raise ValueError("shadow configuration is not frozen pre-outcome")
    if config.get("shadow_candidate_order") != EXPECTED_SHADOW_ORDER:
        raise ValueError("shadow candidate order must contain the exact two frozen arms")
    if tuple(config["shadow_candidate_order"]) != tuple(SHADOW_CANDIDATE_IDS):
        raise ValueError("shadow config and imported pure candidate IDs disagree")
    if set(config["shadow_candidate_definitions"]) != set(EXPECTED_SHADOW_ORDER):
        raise ValueError("shadow candidate definition membership drifted")
    material = config["material_rule"]
    if material["fixed_reference_set"] != FIXED_REFERENCE_IDS:
        raise ValueError("fixed shadow reference set drifted")
    if material["recall_guard_names"] != RECALL_GUARD_NAMES:
        raise ValueError("shadow recall guards drifted")
    if (
        material.get("shadow_arms_are_evaluated_independently") is not True
        or material.get("mutual_shadow_selection") is not False
        or material.get("may_rewrite_primary") is not False
    ):
        raise ValueError("shadow material authority drifted")
    heterogeneity = config["heterogeneity_rule"]
    if (
        heterogeneity.get("may_select_shadow") is not False
        or heterogeneity.get("may_rewrite_primary") is not False
    ):
        raise ValueError("shadow heterogeneity authority drifted")
    contract = config["implementation_contract"]
    required_contract = {
        "temporal_unit": "PIXEL",
        "history_source": "RAW_CLASS_MASK",
        "class_isolation": True,
        "component_evidence_source": "RAW_CURRENT_COMPONENT",
        "upper_membership": "ANY_PIXEL_INTERSECTION",
        "missing_confidence": "INSUFFICIENT_NEGATIVE_EVIDENCE_KEEP",
        "post_filter_feature_recomputation": False,
        "shadow_fit_count": 0,
        "shadow_selection": False,
        "primary_is_reference_only": True,
    }
    for field, expected in required_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"shadow implementation contract drifted: {field}")
    output = config["output_contract"]
    expected_output = {
        "frame_metrics": "shadow_frame_metrics.jsonl",
        "component_decisions": "shadow_component_decisions.jsonl",
        "result": "result.json",
        "expected_frame_rows": 520,
        "expected_component_decision_rows": 23514,
        "terminal": FIXED_TERMINAL,
        "selection_status": "NONE_DIAGNOSTIC_ONLY",
        "drives_alerts": False,
    }
    for field, expected in expected_output.items():
        if output.get(field) != expected:
            raise ValueError(f"shadow output contract drifted: {field}")


def _shadow_definition_hash(config: dict[str, Any], base_config: dict[str, Any]) -> str:
    return sha256_json(
        {
            "protocol_id": config["protocol_id"],
            "shadow_candidate_order": config["shadow_candidate_order"],
            "shadow_candidate_definitions": config["shadow_candidate_definitions"],
            "single_variable_contrasts": config["single_variable_contrasts"],
            "implementation_contract": config["implementation_contract"],
            "material_rule": config["material_rule"],
            "heterogeneity_rule": config["heterogeneity_rule"],
            "thresholds": base_config["thresholds"],
        }
    )


def _assert_primary_binding(
    config: dict[str, Any],
    primary_result: dict[str, Any],
    primary_validation: dict[str, Any],
) -> None:
    binding = config["primary_r0"]
    if primary_result.get("protocol_id") != binding["protocol_id"]:
        raise ValueError("bound primary protocol_id drifted")
    if primary_result.get("git_head") != binding["frozen_git_head"]:
        raise ValueError("bound primary Git identity drifted")
    if (
        primary_result.get("candidate_definition_sha256")
        != binding["candidate_definition_sha256"]
    ):
        raise ValueError("bound primary candidate-definition hash drifted")
    if primary_result["candidate_order"] != [binding["candidate_id"]]:
        raise ValueError("bound primary candidate identity drifted")
    if primary_result["decision"]["terminal"] != binding["terminal"]:
        raise ValueError("bound primary terminal drifted")
    if primary_result["decision"]["next_boundary"] != binding["next_boundary"]:
        raise ValueError("bound primary next boundary drifted")
    if primary_validation.get("protocol_id") != binding["protocol_id"]:
        raise ValueError("bound primary validation protocol drifted")
    if primary_validation.get("status") != binding["validation"]["status"]:
        raise ValueError("bound primary validation status drifted")
    if (
        int(primary_validation.get("checks_completed", -1))
        != int(binding["validation"]["checks_completed"])
    ):
        raise ValueError("bound primary validation check count drifted")


def preflight(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    _validate_shadow_config_shape(config)
    base_binding = config["base_r0_config"]
    base_config = _load_bound_json(repo_root, base_binding)
    primary_result = _load_bound_json(repo_root, config["primary_r0"]["result"])
    primary_validation = _load_bound_json(
        repo_root, config["primary_r0"]["validation"]
    )
    _assert_primary_binding(config, primary_result, primary_validation)
    for field in ("frame_metrics", "component_decisions"):
        binding = config["primary_r0"][field]
        path = _resolve(repo_root, binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"bound primary {field} SHA drifted")
    _verify_output_scope(repo_root, output_root)
    primary_root = _resolve(
        repo_root, config["primary_r0"]["result"]["path"]
    ).parent.resolve()
    resolved_output = output_root.resolve()
    if resolved_output == primary_root or primary_root in resolved_output.parents:
        raise ValueError("shadow output must not overlap the primary R0 evidence root")
    if output_root.exists():
        raise FileExistsError(f"shadow output already exists: {output_root}")
    return {
        "schema_version": (
            "blindassist.dual_loop_segmentation_conditional_gating."
            "shadow_preflight.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "READY_TO_RUN_SHADOW_DIAGNOSTIC",
        "shadow_candidate_order": config["shadow_candidate_order"],
        "shadow_definition_sha256": _shadow_definition_hash(config, base_config),
        "primary_result_sha256": config["primary_r0"]["result"]["sha256"],
        "primary_terminal": config["primary_r0"]["terminal"],
        "fixed_shadow_terminal": FIXED_TERMINAL,
        "output_root": str(output_root),
        "outcomes_accessed": False,
    }


def _point_from_decision(decision: dict[str, Any]) -> dict[str, float | None]:
    return {
        "false_positive_reduction": decision["checks"][FP_GATE_NAME]["value"],
        "recall_retention": decision["checks"]["overall_recall_retention"]["value"],
    }


def _strict_pareto_dominates_point(
    candidate: dict[str, float | None],
    reference: dict[str, float | None],
) -> bool:
    candidate_fp = candidate["false_positive_reduction"]
    candidate_recall = candidate["recall_retention"]
    reference_fp = reference["false_positive_reduction"]
    reference_recall = reference["recall_retention"]
    if None in (candidate_fp, candidate_recall, reference_fp, reference_recall):
        return False
    return bool(
        float(candidate_fp) >= float(reference_fp)
        and float(candidate_recall) >= float(reference_recall)
        and (
            float(candidate_fp) > float(reference_fp)
            or float(candidate_recall) > float(reference_recall)
        )
    )


def assess_shadow_material(
    *,
    shadow_decision: dict[str, Any],
    reference_points: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    shadow_checks = shadow_decision["checks"]
    r4 = all(bool(shadow_checks[name]["passed"]) for name in RECALL_GUARD_NAMES)
    f_gate = bool(shadow_checks[FP_GATE_NAME]["passed"])
    shadow_point = _point_from_decision(shadow_decision)
    dominated_references = [
        reference_id
        for reference_id, point in reference_points.items()
        if _strict_pareto_dominates_point(shadow_point, point)
    ]
    frontier = not any(
        _strict_pareto_dominates_point(point, shadow_point)
        for point in reference_points.values()
    )
    n_gate = bool(frontier and dominated_references)
    material = bool(r4 and (f_gate or n_gate))
    h_min = bool(
        not shadow_checks["minimum_session_recall_retention"]["passed"]
        and shadow_checks[FP_GATE_NAME]["passed"]
        and shadow_checks["overall_recall_retention"]["passed"]
        and shadow_checks["boundary_step_curb_recall_retention"]["passed"]
        and shadow_checks["obstacle_recall_retention"]["passed"]
    )
    return {
        "R4_all_recall_guards_pass": r4,
        "F_false_positive_reduction_pass": f_gate,
        "N_frontier_and_dominates_reference": n_gate,
        "frontier_with_fixed_references": frontier,
        "strictly_dominates_reference_ids": dominated_references,
        "MATERIAL": material,
        "H_min": h_min,
        "may_rewrite_primary": False,
    }


def _session_point(
    report: dict[str, Any],
    session_id: str,
) -> dict[str, float | None]:
    comparison = report["by_session_id"][session_id]["comparison_to_baseline"]
    return {
        "false_positive_reduction": comparison["false_positive_reduction"],
        "recall_retention": comparison["recall_retention"],
    }


def assess_cross_session_heterogeneity(
    *,
    reports: dict[str, Any],
) -> dict[str, Any]:
    left = SHADOW_CLASS_TEMPORAL_ID
    right = SHADOW_MULTI_NEGATIVE_ID
    session_ids = sorted(reports[left]["by_session_id"])
    left_dominates: list[str] = []
    right_dominates: list[str] = []
    for session_id in session_ids:
        left_point = _session_point(reports[left], session_id)
        right_point = _session_point(reports[right], session_id)
        if _strict_pareto_dominates_point(left_point, right_point):
            left_dominates.append(session_id)
        if _strict_pareto_dominates_point(right_point, left_point):
            right_dominates.append(session_id)
    return {
        "H_cross": bool(left_dominates and right_dominates),
        "class_conditional_temporal_dominates_sessions": left_dominates,
        "multi_negative_dominates_sessions": right_dominates,
        "used_for_material_rule": False,
        "may_select_shadow": False,
    }


def _parity_fields_match(
    shadow: dict[str, Any],
    primary: dict[str, Any],
) -> bool:
    fields = (
        "predicted_class",
        "raw_area_pixels",
        "raw_top1_confidence_median",
        "confidence_known",
        "low_confidence",
        "small_fragment",
        "intersects_upper_head_band",
        "raw_pixels",
        "causal_supported_pixels",
        "noncausal_pixels",
        "kept_pixels",
        "rejected_pixels",
        "post_fragment_count",
        "action",
    )
    return all(shadow.get(field) == primary.get(field) for field in fields)


def run_shadow_ablation(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    preflight_result = preflight(
        repo_root=repo_root,
        config_path=config_path,
        output_root=output_root,
    )
    config = read_json(config_path)
    base_config = _load_bound_json(repo_root, config["base_r0_config"])
    primary_result = _load_bound_json(repo_root, config["primary_r0"]["result"])
    primary_frames = _load_bound_jsonl(
        repo_root, config["primary_r0"]["frame_metrics"]
    )
    primary_decisions = _load_bound_jsonl(
        repo_root, config["primary_r0"]["component_decisions"]
    )
    frame_rows, frame_provenance = _load_bound_jsonl(
        repo_root, base_config["input_contract"]["frames"]
    )
    component_rows, component_provenance = _load_bound_jsonl(
        repo_root, base_config["input_contract"]["components"]
    )
    _validate_membership(base_config, frame_rows, component_rows)

    shape = tuple(int(value) for value in base_config["analysis_shape"])
    candidate_classes = [str(value) for value in base_config["candidate_classes"]]
    thresholds = base_config["thresholds"]
    upper_mask = upper_head_band(
        shape, float(thresholds["upper_head_y_max_fraction"])
    )
    truth_names = {
        int(key): str(value) for key, value in base_config["truth_classes"].items()
    }
    truth_ids_by_name = {name: class_id for class_id, name in truth_names.items()}
    hazard_ids = np.asarray(base_config["hazard_truth_ids"], dtype=np.uint8)

    manifest_binding = base_config["input_contract"]["canonical_manifest"]
    manifest_path = _resolve(repo_root, manifest_binding["path"])
    if sha256_file(manifest_path) != manifest_binding["sha256"]:
        raise ValueError("canonical manifest SHA mismatch")
    manifest_rows = read_jsonl(manifest_path)
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("canonical manifest contains duplicate id")
    view_root = _resolve(
        repo_root, base_config["input_contract"]["canonical_view_root"]
    )
    ledger_by_id = {str(row["component_id"]): row for row in component_rows}
    if len(ledger_by_id) != len(component_rows):
        raise ValueError("component ledger contains duplicate component_id")

    enriched_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for frame_row in frame_rows:
        manifest = manifest_by_id.get(str(frame_row["view_row_id"]))
        if manifest is None:
            raise ValueError(f"missing canonical manifest row: {frame_row['view_row_id']}")
        for field in (
            "source_id",
            "session_id",
            "frame_id",
            "image_sha256",
            "canonical_mask_sha256",
        ):
            if frame_row[field] != manifest[field]:
                raise ValueError(
                    f"frame/manifest mismatch for {field}: {frame_row['view_row_id']}"
                )
        if str(frame_row["rehearsal_role"]) != str(manifest["role"]):
            raise ValueError("frame/manifest role mismatch")
        enriched_rows.append((frame_row, manifest))
    enriched_rows.sort(
        key=lambda pair: (
            str(pair[1]["session_id"]),
            str(pair[1]["sequence_id"]),
            int(pair[1]["source_capture_timestamp_ns"]),
            int(pair[1]["frame_id"]),
            str(pair[1]["id"]),
        )
    )

    histories: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    output_frames: list[dict[str, Any]] = []
    output_decisions: list[dict[str, Any]] = []
    observed_component_ids: set[str] = set()
    arm_ids = [BASELINE_ID, *config["shadow_candidate_order"]]

    for frame_row, manifest in enriched_rows:
        view_row_id = str(frame_row["view_row_id"])
        session_id = str(frame_row["session_id"])
        sequence_id = str(manifest["sequence_id"])
        truth_path = view_root / str(manifest["canonical_mask_path"])
        if sha256_file(truth_path) != str(frame_row["canonical_mask_sha256"]):
            raise ValueError(f"canonical truth SHA mismatch: {view_row_id}")
        truth = np.asarray(Image.open(truth_path), dtype=np.uint8)
        if truth.shape != shape:
            raise ValueError(f"canonical truth shape mismatch: {view_row_id}")
        a_mask = decode_packed_mask(frame_row["packed_masks"]["A"], shape)
        b_mask = decode_packed_mask(frame_row["packed_masks"]["B"], shape)
        raw_class_masks = {
            class_name: decode_packed_mask(
                frame_row["packed_masks"][f"candidate_{class_name}"], shape
            )
            for class_name in candidate_classes
        }
        if np.count_nonzero(
            raw_class_masks[candidate_classes[0]]
            & raw_class_masks[candidate_classes[1]]
        ):
            raise ValueError(f"candidate class masks overlap: {view_row_id}")
        if not np.array_equal(
            b_mask, np.logical_or.reduce(list(raw_class_masks.values()))
        ):
            raise ValueError(f"candidate class union mismatch: {view_row_id}")
        full_truth_hazard = np.isin(truth, hazard_ids)
        residual_truth = full_truth_hazard & ~a_mask
        class_truth = {
            class_name: (truth == truth_ids_by_name[class_name]) & ~a_mask
            for class_name in candidate_classes
        }
        same_class_causal: dict[str, np.ndarray] = {}
        for class_name, current in raw_class_masks.items():
            history = histories[(session_id, sequence_id, class_name)]
            same_class_causal[class_name] = causal_two_of_three(
                current,
                history[-1] if len(history) >= 1 else None,
                history[-2] if len(history) >= 2 else None,
            )

        arm_class_masks = {
            arm_id: {
                class_name: np.zeros(shape, dtype=bool)
                for class_name in candidate_classes
            }
            for arm_id in arm_ids
        }
        arm_class_masks[BASELINE_ID] = {
            class_name: mask.copy()
            for class_name, mask in raw_class_masks.items()
        }

        for class_name, class_mask in raw_class_masks.items():
            for component in connected_components(
                class_mask, connectivity=int(thresholds["connectivity"])
            ):
                component_id = (
                    f"{frame_row['source_id']}:{int(frame_row['frame_id'])}:"
                    f"{class_name}:{component.index}"
                )
                ledger = ledger_by_id.get(component_id)
                if ledger is None:
                    raise ValueError(f"component missing from ledger: {component_id}")
                if (
                    str(ledger["class_name"]) != class_name
                    or int(ledger["area_pixels"]) != component.area
                    or list(ledger["bbox_xyxy"]) != list(component.bbox)
                ):
                    raise ValueError(f"component ledger geometry mismatch: {component_id}")
                confidence = ledger["top1_confidence_median"]
                if confidence is not None and not math.isfinite(float(confidence)):
                    raise ValueError(f"component confidence is nonfinite: {component_id}")
                observed_component_ids.add(component_id)
                for candidate_id in config["shadow_candidate_order"]:
                    kept, evidence = apply_frozen_candidate_to_component(
                        candidate_id=candidate_id,
                        predicted_class=class_name,
                        component_mask=component.mask,
                        same_class_causal_mask=same_class_causal[class_name],
                        confidence_median=confidence,
                        raw_area_pixels=component.area,
                        upper_band_mask=upper_mask,
                        confidence_minimum=float(thresholds["confidence_minimum"]),
                        small_fragment_max_area_pixels=int(
                            thresholds["small_fragment_max_area_pixels"]
                        ),
                    )
                    arm_class_masks[candidate_id][class_name] |= kept
                    output_decisions.append(
                        {
                            "schema_version": (
                                "blindassist.dual_loop_segmentation_conditional_gating."
                                "shadow_component_decision.v1"
                            ),
                            "protocol_id": PROTOCOL_ID,
                            "analysis_role": config["analysis_role"],
                            "candidate_id": candidate_id,
                            "view_row_id": view_row_id,
                            "source_id": str(frame_row["source_id"]),
                            "session_id": session_id,
                            "sequence_id": sequence_id,
                            "frame_id": int(frame_row["frame_id"]),
                            "component_id": component_id,
                            "component_index": int(component.index),
                            "predicted_class": class_name,
                            "raw_area_pixels": int(component.area),
                            "raw_top1_confidence_median": confidence,
                            "gate_input_fields": [
                                "predicted_class",
                                "raw_component_mask",
                                "same_class_raw_history_masks",
                                "top1_confidence_median",
                                "raw_area_pixels",
                                "upper_head_band_geometry",
                                "frozen_thresholds",
                            ],
                            **evidence,
                        }
                    )
        for arm_id in arm_ids:
            for class_name in candidate_classes:
                if np.count_nonzero(
                    arm_class_masks[arm_id][class_name]
                    & ~raw_class_masks[class_name]
                ):
                    raise ValueError(
                        f"shadow arm created new pixels: "
                        f"{arm_id}/{view_row_id}/{class_name}"
                    )
        output_frames.append(
            {
                "schema_version": (
                    "blindassist.dual_loop_segmentation_conditional_gating."
                    "shadow_frame_metrics.v1"
                ),
                "protocol_id": PROTOCOL_ID,
                "analysis_role": config["analysis_role"],
                "view_row_id": view_row_id,
                "source_id": str(frame_row["source_id"]),
                "session_id": session_id,
                "sequence_id": sequence_id,
                "role": str(frame_row["rehearsal_role"]),
                "scene_bucket": str(manifest["scene_bucket"]),
                "frame_id": int(frame_row["frame_id"]),
                "source_capture_timestamp_ns": int(
                    manifest["source_capture_timestamp_ns"]
                ),
                "arms": {
                    arm_id: _frame_arm_metrics(
                        arm_class_masks[arm_id], residual_truth, class_truth
                    )
                    for arm_id in arm_ids
                },
            }
        )
        for class_name, current in raw_class_masks.items():
            history = histories[(session_id, sequence_id, class_name)]
            history.append(current.copy())
            if len(history) > 2:
                del history[:-2]

    if observed_component_ids != set(ledger_by_id):
        missing = sorted(set(ledger_by_id) - observed_component_ids)
        raise ValueError(f"unmatched component ledger rows: {missing[:3]}")
    expected_decision_rows = int(
        config["output_contract"]["expected_component_decision_rows"]
    )
    if len(output_decisions) != expected_decision_rows:
        raise ValueError(
            f"unexpected shadow component decision count: {len(output_decisions)}"
        )

    reports = {
        arm_id: _aggregate_arm_report(
            frame_rows=output_frames,
            arm_id=arm_id,
            candidate_classes=candidate_classes,
            group_fields=("session_id", "role", "scene_bucket"),
        )
        for arm_id in arm_ids
    }
    if sha256_json(reports[BASELINE_ID]) != sha256_json(
        primary_result["arms"][BASELINE_ID]
    ):
        raise ValueError("shadow baseline does not reproduce the bound primary baseline")

    shadow_decisions = {
        candidate_id: _decision_checks(
            reports[candidate_id],
            reports[BASELINE_ID],
            base_config["decision_rules"],
        )
        for candidate_id in config["shadow_candidate_order"]
    }
    reference_points: dict[str, dict[str, float | None]] = {}
    for reference_id in FIXED_REFERENCE_IDS:
        comparison = primary_result["arms"][reference_id]["overall"][
            "comparison_to_baseline"
        ]
        reference_points[reference_id] = {
            "false_positive_reduction": comparison["false_positive_reduction"],
            "recall_retention": comparison["recall_retention"],
        }
    material_by_candidate = {
        candidate_id: assess_shadow_material(
            shadow_decision=shadow_decisions[candidate_id],
            reference_points=reference_points,
        )
        for candidate_id in config["shadow_candidate_order"]
    }
    cross_heterogeneity = assess_cross_session_heterogeneity(reports=reports)
    any_material = any(
        bool(row["MATERIAL"]) for row in material_by_candidate.values()
    )
    heterogeneous = bool(
        cross_heterogeneity["H_cross"]
        or any(bool(row["H_min"]) for row in material_by_candidate.values())
    )
    family_terminal = (
        "MATERIAL_SHADOW_SIGNAL_PRESENT"
        if any_material
        else config["heterogeneity_rule"][
            "no_material_but_heterogeneous_hypothesis"
        ]
        if heterogeneous
        else "MECHANISM_ONLY"
    )

    primary_by_component = {
        str(row["component_id"]): row for row in primary_decisions
    }
    if len(primary_by_component) != len(primary_decisions):
        raise ValueError("bound primary component decisions contain duplicates")
    parity = {
        SHADOW_CLASS_TEMPORAL_ID: {
            "unchanged_branch": "boundary_step_curb",
            "checked_components": 0,
            "mismatch_count": 0,
        },
        SHADOW_MULTI_NEGATIVE_ID: {
            "unchanged_branch": "obstacle",
            "checked_components": 0,
            "mismatch_count": 0,
        },
    }
    for row in output_decisions:
        candidate_id = str(row["candidate_id"])
        class_name = str(row["predicted_class"])
        unchanged = str(parity[candidate_id]["unchanged_branch"])
        if class_name != unchanged:
            continue
        primary_row = primary_by_component.get(str(row["component_id"]))
        if primary_row is None:
            raise ValueError("shadow parity component missing from bound primary")
        parity[candidate_id]["checked_components"] += 1
        if not _parity_fields_match(row, primary_row):
            parity[candidate_id]["mismatch_count"] += 1
    for candidate_id, row in parity.items():
        if int(row["checked_components"]) <= 0 or int(row["mismatch_count"]) != 0:
            raise ValueError(f"single-variable parity failed: {candidate_id}")

    decision_rows_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_decisions:
        decision_rows_by_candidate[str(row["candidate_id"])].append(row)
    component_outcomes: dict[str, Any] = {}
    for candidate_id in config["shadow_candidate_order"]:
        rows = decision_rows_by_candidate[candidate_id]
        actions = Counter(str(row["action"]) for row in rows)
        component_outcomes[candidate_id] = {
            "raw_component_count": len(rows),
            "fully_retained": actions["KEEP"],
            "partially_retained": actions["PARTIAL"],
            "removed": actions["REJECT"],
            "split_source_components": sum(
                int(row["post_fragment_count"]) > 1 for row in rows
            ),
            "post_fragment_count_from_raw_components": sum(
                int(row["post_fragment_count"]) for row in rows
            ),
        }

    shadow_definition_sha = _shadow_definition_hash(config, base_config)
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": config["stage"],
        "mode": config["mode"],
        "analysis_role": config["analysis_role"],
        "claim_ceiling": config["claim_ceiling"],
        "evidence_instance": config["evidence_instance"],
        "git_head": _current_git_head(repo_root),
        "shadow_definition_sha256": shadow_definition_sha,
        "shadow_candidate_order": config["shadow_candidate_order"],
        "selection_status": "NONE_DIAGNOSTIC_ONLY",
        "drives_alerts": False,
        "confirmation": "NOT_ACTIVATED",
        "preflight": preflight_result,
        "primary_r0": {
            "reference_only": True,
            "result_sha256": config["primary_r0"]["result"]["sha256"],
            "frame_metrics_sha256": config["primary_r0"]["frame_metrics"]["sha256"],
            "component_decisions_sha256": config["primary_r0"][
                "component_decisions"
            ]["sha256"],
            "validation_sha256": config["primary_r0"]["validation"]["sha256"],
            "frozen_git_head": config["primary_r0"]["frozen_git_head"],
            "candidate_id": config["primary_r0"]["candidate_id"],
            "candidate_definition_sha256": config["primary_r0"][
                "candidate_definition_sha256"
            ],
            "terminal": config["primary_r0"]["terminal"],
            "next_boundary": config["primary_r0"]["next_boundary"],
            "terminal_unchanged": True,
        },
        "input": {
            "frame_count": len(output_frames),
            "component_count": len(component_rows),
            "session_count": len(
                base_config["input_contract"]["expected_session_frame_counts"]
            ),
            "frame_files": frame_provenance,
            "component_files": component_provenance,
            "canonical_manifest": manifest_binding,
        },
        "arms": reports,
        "shadow_gate_diagnostics": shadow_decisions,
        "material_assessment": {
            "overall": family_terminal,
            "fixed_reference_points": reference_points,
            "by_candidate": material_by_candidate,
            "shadow_arms_evaluated_independently": True,
            "mutual_shadow_selection": False,
            "any_material": any_material,
            "may_rewrite_primary": False,
        },
        "heterogeneity_assessment": {
            "H_min_by_candidate": {
                candidate_id: bool(material_by_candidate[candidate_id]["H_min"])
                for candidate_id in config["shadow_candidate_order"]
            },
            "cross_session": cross_heterogeneity,
            "heterogeneous": heterogeneous,
            "hypothesis_if_no_material": (
                config["heterogeneity_rule"][
                    "no_material_but_heterogeneous_hypothesis"
                ]
                if heterogeneous and not any_material
                else None
            ),
            "may_rewrite_primary": False,
        },
        "single_variable_parity": parity,
        "component_outcomes": component_outcomes,
        "decision": {
            "terminal": FIXED_TERMINAL,
            "primary_r0_terminal": config["primary_r0"]["terminal"],
            "primary_r0_terminal_unchanged": True,
            "selection": "NONE",
            "shadow_authority": "DIAGNOSTIC_ONLY",
            "gating_route_reopened": False,
            "next_boundary": config["primary_r0"]["next_boundary"],
            "android_or_alert_authority": "NONE",
        },
        "provenance": {
            "config": {
                "path": str(config_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(config_path),
            },
            "base_r0_config": config["base_r0_config"],
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(repo_root)).replace(
                    "\\", "/"
                ),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        frame_name = config["output_contract"]["frame_metrics"]
        decision_name = config["output_contract"]["component_decisions"]
        result_name = config["output_contract"]["result"]
        _write_jsonl(temporary / frame_name, output_frames)
        _write_jsonl(temporary / decision_name, output_decisions)
        result["output_files"] = {
            frame_name: {
                "sha256": sha256_file(temporary / frame_name),
                "row_count": len(output_frames),
            },
            decision_name: {
                "sha256": sha256_file(temporary / decision_name),
                "row_count": len(output_decisions),
            },
        }
        _write_json(temporary / result_name, result)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    config_path = _resolve(repo_root, args.config)
    output_root = _resolve(repo_root, args.output_root)
    if args.preflight_only:
        result = preflight(
            repo_root=repo_root,
            config_path=config_path,
            output_root=output_root,
        )
    else:
        result = run_shadow_ablation(
            repo_root=repo_root,
            config_path=config_path,
            output_root=output_root,
        )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
