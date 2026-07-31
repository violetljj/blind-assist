"""Independently validate the R0.1 post-primary shadow ablation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from scripts.research.dual_loop_segmentation_conditional_gating.conditional_gating import (
    BASELINE_ID,
    PRIMARY_CANDIDATE_ID,
    SHADOW_CLASS_TEMPORAL_ID,
    SHADOW_MULTI_NEGATIVE_ID,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from scripts.research.dual_loop_segmentation_conditional_gating.shadow_ablation import (
    EXPECTED_SHADOW_ORDER,
    FIXED_REFERENCE_IDS,
    FIXED_TERMINAL,
    FP_GATE_NAME,
    MODE,
    PROTOCOL_ID,
    RECALL_GUARD_NAMES,
    _load_bound_json,
    _load_bound_jsonl,
    _resolve,
    _shadow_definition_hash,
    _validate_shadow_config_shape,
)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _assert_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")


def _assert_float_equal(left: Any, right: Any, label: str) -> None:
    if left is None or right is None:
        if left is not right:
            raise ValueError(f"{label} nullability mismatch")
        return
    if abs(float(left) - float(right)) > 1e-12:
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")


def _aggregate_pixel(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    predicted = sum(int(row["predicted_pixels"]) for row in rows)
    truth = sum(int(row["truth_pixels"]) for row in rows)
    empty = tp + fp + fn == 0
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    iou = _ratio(tp, tp + fp + fn)
    if empty:
        precision = recall = iou = 1.0
    f1 = (
        None
        if precision is None or recall is None
        else 0.0
        if precision + recall == 0
        else float(2.0 * precision * recall / (precision + recall))
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": predicted,
        "truth_pixels": truth,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def _aggregate_arm_core(
    rows: Sequence[dict[str, Any]],
    arm_id: str,
    classes: Sequence[str],
) -> dict[str, Any]:
    frame_count = len(rows)
    pixel = _aggregate_pixel([row["arms"][arm_id]["pixel"] for row in rows])
    post = sum(int(row["arms"][arm_id]["post_component_count"]) for row in rows)
    any_false = sum(
        int(row["arms"][arm_id]["any_hazard_false_component_count"])
        for row in rows
    )
    strict_false = sum(
        int(row["arms"][arm_id]["class_strict_false_component_count"])
        for row in rows
    )
    class_reports: dict[str, Any] = {}
    for class_name in classes:
        class_pixel = _aggregate_pixel(
            [
                row["arms"][arm_id]["classes"][class_name]["pixel"]
                for row in rows
            ]
        )
        class_post = sum(
            int(
                row["arms"][arm_id]["classes"][class_name]["component"][
                    "predicted_component_count"
                ]
            )
            for row in rows
        )
        class_false = sum(
            int(
                row["arms"][arm_id]["classes"][class_name]["component"][
                    "false_activation_component_count"
                ]
            )
            for row in rows
        )
        class_reports[class_name] = {
            "pixel": class_pixel,
            "post_component_count": class_post,
            "class_strict_false_component_count": class_false,
            "post_components_per_frame": float(class_post / frame_count),
            "class_strict_false_components_per_frame": float(
                class_false / frame_count
            ),
        }
    return {
        "frame_count": frame_count,
        "pixel": pixel,
        "post_component_count": post,
        "any_hazard_false_component_count": any_false,
        "class_strict_false_component_count": strict_false,
        "post_components_per_frame": float(post / frame_count),
        "any_hazard_false_components_per_frame": float(any_false / frame_count),
        "class_strict_false_components_per_frame": float(
            strict_false / frame_count
        ),
        "classes": class_reports,
    }


def _comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | None]:
    return {
        "false_positive_reduction": _ratio(
            int(baseline["pixel"]["fp"]) - int(candidate["pixel"]["fp"]),
            int(baseline["pixel"]["fp"]),
        ),
        "recall_retention": _ratio(
            int(candidate["pixel"]["tp"]),
            int(baseline["pixel"]["tp"]),
        ),
    }


def _validate_aggregate(
    *,
    rows: Sequence[dict[str, Any]],
    arm_id: str,
    classes: Sequence[str],
    reported: dict[str, Any],
    label: str,
) -> int:
    actual = _aggregate_arm_core(rows, arm_id, classes)
    baseline = _aggregate_arm_core(rows, BASELINE_ID, classes)
    checks = 0
    for key in (
        "frame_count",
        "post_component_count",
        "any_hazard_false_component_count",
        "class_strict_false_component_count",
    ):
        _assert_equal(actual[key], reported[key], f"{label}.{key}")
        checks += 1
    for key in (
        "post_components_per_frame",
        "any_hazard_false_components_per_frame",
        "class_strict_false_components_per_frame",
    ):
        _assert_float_equal(actual[key], reported[key], f"{label}.{key}")
        checks += 1
    for key, value in actual["pixel"].items():
        if isinstance(value, float) or value is None:
            _assert_float_equal(value, reported["pixel"][key], f"{label}.pixel.{key}")
        else:
            _assert_equal(value, reported["pixel"][key], f"{label}.pixel.{key}")
        checks += 1
    expected_comparison = _comparison(actual, baseline)
    for key, value in expected_comparison.items():
        _assert_float_equal(
            value,
            reported["comparison_to_baseline"][key],
            f"{label}.comparison.{key}",
        )
        checks += 1
    for class_name in classes:
        actual_class = actual["classes"][class_name]
        reported_class = reported["classes"][class_name]
        baseline_class = baseline["classes"][class_name]
        for key in (
            "post_component_count",
            "class_strict_false_component_count",
        ):
            _assert_equal(
                actual_class[key],
                reported_class[key],
                f"{label}.{class_name}.{key}",
            )
            checks += 1
        for key, value in actual_class["pixel"].items():
            if isinstance(value, float) or value is None:
                _assert_float_equal(
                    value,
                    reported_class["pixel"][key],
                    f"{label}.{class_name}.pixel.{key}",
                )
            else:
                _assert_equal(
                    value,
                    reported_class["pixel"][key],
                    f"{label}.{class_name}.pixel.{key}",
                )
            checks += 1
        expected_class_comparison = {
            "false_positive_reduction": _ratio(
                int(baseline_class["pixel"]["fp"])
                - int(actual_class["pixel"]["fp"]),
                int(baseline_class["pixel"]["fp"]),
            ),
            "recall_retention": _ratio(
                int(actual_class["pixel"]["tp"]),
                int(baseline_class["pixel"]["tp"]),
            ),
        }
        for key, value in expected_class_comparison.items():
            _assert_float_equal(
                value,
                reported_class["comparison_to_baseline"][key],
                f"{label}.{class_name}.comparison.{key}",
            )
            checks += 1
    return checks


def _expected_rejected(row: dict[str, Any]) -> int:
    candidate_id = str(row["candidate_id"])
    predicted_class = str(row["predicted_class"])
    raw = int(row["raw_pixels"])
    noncausal = int(row["noncausal_pixels"])
    low = bool(row["low_confidence"])
    proxy = bool(row["small_fragment"] or row["intersects_upper_head_band"])
    if candidate_id == SHADOW_CLASS_TEMPORAL_ID:
        if predicted_class == "obstacle":
            return noncausal
        if predicted_class == "boundary_step_curb":
            return raw if low and bool(row["small_fragment"]) else 0
    if candidate_id == SHADOW_MULTI_NEGATIVE_ID:
        return noncausal if low and proxy else 0
    raise ValueError(f"unexpected shadow predicate identity: {candidate_id}")


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


def _validate_component_decisions(
    *,
    rows: Sequence[dict[str, Any]],
    primary_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    base_config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    expected_count = int(config["output_contract"]["expected_component_decision_rows"])
    if len(rows) != expected_count:
        raise ValueError("shadow component decision row count mismatch")
    forbidden = set(base_config["forbidden_candidate_inputs"])
    primary_by_component = {
        str(row["component_id"]): row for row in primary_rows
    }
    if len(primary_by_component) != len(primary_rows):
        raise ValueError("bound primary component decisions contain duplicates")
    seen: set[tuple[str, str]] = set()
    checks = 0
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
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in EXPECTED_SHADOW_ORDER:
            raise ValueError(f"unexpected shadow candidate: {candidate_id}")
        key = (candidate_id, str(row["component_id"]))
        if key in seen:
            raise ValueError("duplicate shadow candidate/component decision")
        seen.add(key)
        if forbidden & set(str(value) for value in row["gate_input_fields"]):
            raise ValueError("shadow predicate leaked a forbidden input")
        raw = int(row["raw_pixels"])
        causal = int(row["causal_supported_pixels"])
        noncausal = int(row["noncausal_pixels"])
        kept = int(row["kept_pixels"])
        rejected = int(row["rejected_pixels"])
        if raw != causal + noncausal or raw != kept + rejected:
            raise ValueError("shadow decision does not partition raw component pixels")
        if rejected != _expected_rejected(row):
            raise ValueError("shadow predicate differs from the frozen independent rule")
        expected_action = "REJECT" if kept == 0 else "KEEP" if kept == raw else "PARTIAL"
        if row["action"] != expected_action:
            raise ValueError("shadow action differs from independent pixel counts")
        unchanged = parity[candidate_id]["unchanged_branch"]
        if row["predicted_class"] == unchanged:
            primary = primary_by_component.get(str(row["component_id"]))
            if primary is None:
                raise ValueError("shadow parity component missing from primary")
            parity[candidate_id]["checked_components"] += 1
            if not _parity_fields_match(row, primary):
                parity[candidate_id]["mismatch_count"] += 1
        checks += 7
    for candidate_id, row in parity.items():
        if row["checked_components"] <= 0 or row["mismatch_count"] != 0:
            raise ValueError(f"shadow single-variable parity failed: {candidate_id}")
        checks += 1
    return checks, parity


def _decision_from_report(
    *,
    report: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    overall = report["overall"]
    comparable = {
        session_id: row["comparison_to_baseline"]["recall_retention"]
        for session_id, row in report["by_session_id"].items()
        if row["comparison_to_baseline"]["recall_retention"] is not None
    }
    minimum_session_id = (
        min(comparable, key=lambda key: (comparable[key], key))
        if comparable
        else None
    )
    values = {
        "false_positive_reduction": overall["comparison_to_baseline"][
            "false_positive_reduction"
        ],
        "overall_recall_retention": overall["comparison_to_baseline"][
            "recall_retention"
        ],
        "minimum_session_recall_retention": (
            comparable[minimum_session_id] if minimum_session_id is not None else None
        ),
        "boundary_step_curb_recall_retention": overall["classes"][
            "boundary_step_curb"
        ]["comparison_to_baseline"]["recall_retention"],
        "obstacle_recall_retention": overall["classes"]["obstacle"][
            "comparison_to_baseline"
        ]["recall_retention"],
    }
    thresholds = {
        "false_positive_reduction": rules["minimum_false_positive_reduction"],
        "overall_recall_retention": rules["minimum_overall_recall_retention"],
        "minimum_session_recall_retention": rules[
            "minimum_session_recall_retention"
        ],
        "boundary_step_curb_recall_retention": rules[
            "minimum_boundary_step_curb_recall_retention"
        ],
        "obstacle_recall_retention": rules["minimum_obstacle_recall_retention"],
    }
    checks: dict[str, Any] = {}
    for name, threshold in thresholds.items():
        checks[name] = {
            "value": values[name],
            "threshold": threshold,
            "passed": values[name] is not None
            and float(values[name]) >= float(threshold),
        }
    checks["minimum_session_recall_retention"]["session_id"] = minimum_session_id
    return {
        "checks": checks,
        "sufficient": all(checks[name]["passed"] for name in thresholds),
    }


def _dominates(
    left: dict[str, float | None],
    right: dict[str, float | None],
) -> bool:
    values = (
        left["false_positive_reduction"],
        left["recall_retention"],
        right["false_positive_reduction"],
        right["recall_retention"],
    )
    if any(value is None for value in values):
        return False
    return bool(
        float(left["false_positive_reduction"])
        >= float(right["false_positive_reduction"])
        and float(left["recall_retention"]) >= float(right["recall_retention"])
        and (
            float(left["false_positive_reduction"])
            > float(right["false_positive_reduction"])
            or float(left["recall_retention"]) > float(right["recall_retention"])
        )
    )


def _point_from_report(report: dict[str, Any]) -> dict[str, float | None]:
    comparison = report["overall"]["comparison_to_baseline"]
    return {
        "false_positive_reduction": comparison["false_positive_reduction"],
        "recall_retention": comparison["recall_retention"],
    }


def _material_assessment(
    *,
    decision: dict[str, Any],
    point: dict[str, float | None],
    references: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    checks = decision["checks"]
    r4 = all(checks[name]["passed"] for name in RECALL_GUARD_NAMES)
    f_gate = bool(checks[FP_GATE_NAME]["passed"])
    dominated = [
        reference_id
        for reference_id, reference_point in references.items()
        if _dominates(point, reference_point)
    ]
    frontier = not any(_dominates(reference_point, point) for reference_point in references.values())
    n_gate = bool(frontier and dominated)
    h_min = bool(
        not checks["minimum_session_recall_retention"]["passed"]
        and checks[FP_GATE_NAME]["passed"]
        and checks["overall_recall_retention"]["passed"]
        and checks["boundary_step_curb_recall_retention"]["passed"]
        and checks["obstacle_recall_retention"]["passed"]
    )
    return {
        "R4_all_recall_guards_pass": r4,
        "F_false_positive_reduction_pass": f_gate,
        "N_frontier_and_dominates_reference": n_gate,
        "frontier_with_fixed_references": frontier,
        "strictly_dominates_reference_ids": dominated,
        "MATERIAL": bool(r4 and (f_gate or n_gate)),
        "H_min": h_min,
        "may_rewrite_primary": False,
    }


def _cross_heterogeneity(reports: dict[str, Any]) -> dict[str, Any]:
    left = SHADOW_CLASS_TEMPORAL_ID
    right = SHADOW_MULTI_NEGATIVE_ID
    left_sessions: list[str] = []
    right_sessions: list[str] = []
    for session_id in sorted(reports[left]["by_session_id"]):
        left_row = reports[left]["by_session_id"][session_id][
            "comparison_to_baseline"
        ]
        right_row = reports[right]["by_session_id"][session_id][
            "comparison_to_baseline"
        ]
        left_point = {
            "false_positive_reduction": left_row["false_positive_reduction"],
            "recall_retention": left_row["recall_retention"],
        }
        right_point = {
            "false_positive_reduction": right_row["false_positive_reduction"],
            "recall_retention": right_row["recall_retention"],
        }
        if _dominates(left_point, right_point):
            left_sessions.append(session_id)
        if _dominates(right_point, left_point):
            right_sessions.append(session_id)
    return {
        "H_cross": bool(left_sessions and right_sessions),
        "class_conditional_temporal_dominates_sessions": left_sessions,
        "multi_negative_dominates_sessions": right_sessions,
        "used_for_material_rule": False,
        "may_select_shadow": False,
    }


def validate(
    *,
    repo_root: Path,
    config_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    checks = 0
    try:
        config = read_json(config_path)
        _validate_shadow_config_shape(config)
        base_config = _load_bound_json(repo_root, config["base_r0_config"])
        primary_result = _load_bound_json(repo_root, config["primary_r0"]["result"])
        primary_validation = _load_bound_json(
            repo_root, config["primary_r0"]["validation"]
        )
        primary_frames = _load_bound_jsonl(
            repo_root, config["primary_r0"]["frame_metrics"]
        )
        primary_components = _load_bound_jsonl(
            repo_root, config["primary_r0"]["component_decisions"]
        )
        result_path = evidence_root / config["output_contract"]["result"]
        frame_path = evidence_root / config["output_contract"]["frame_metrics"]
        component_path = (
            evidence_root / config["output_contract"]["component_decisions"]
        )
        result = read_json(result_path)
        frames = read_jsonl(frame_path)
        component_rows = read_jsonl(component_path)

        if (
            result.get("protocol_id") != PROTOCOL_ID
            or result.get("mode") != MODE
            or result.get("shadow_candidate_order") != EXPECTED_SHADOW_ORDER
        ):
            raise ValueError("shadow result identity drifted")
        if result.get("shadow_definition_sha256") != _shadow_definition_hash(
            config, base_config
        ):
            raise ValueError("shadow definition hash drifted")
        primary_binding = config["primary_r0"]
        if (
            primary_result.get("protocol_id") != primary_binding["protocol_id"]
            or primary_result.get("git_head") != primary_binding["frozen_git_head"]
            or primary_result.get("candidate_definition_sha256")
            != primary_binding["candidate_definition_sha256"]
            or primary_result.get("candidate_order")
            != [primary_binding["candidate_id"]]
            or primary_result["decision"].get("terminal")
            != primary_binding["terminal"]
            or primary_result["decision"].get("next_boundary")
            != primary_binding["next_boundary"]
        ):
            raise ValueError("bound primary result identity drifted")
        if (
            primary_validation.get("protocol_id")
            != primary_binding["protocol_id"]
            or primary_validation.get("status")
            != primary_binding["validation"]["status"]
            or int(primary_validation.get("checks_completed", -1))
            != int(primary_binding["validation"]["checks_completed"])
        ):
            raise ValueError("bound primary validation identity drifted")
        expected_primary_binding = {
            "protocol_id": primary_binding["protocol_id"],
            "result_sha256": primary_binding["result"]["sha256"],
            "frame_metrics_sha256": primary_binding["frame_metrics"]["sha256"],
            "component_decisions_sha256": primary_binding[
                "component_decisions"
            ]["sha256"],
            "validation_sha256": primary_binding["validation"]["sha256"],
            "frozen_git_head": primary_binding["frozen_git_head"],
            "candidate_id": primary_binding["candidate_id"],
            "candidate_definition_sha256": primary_binding[
                "candidate_definition_sha256"
            ],
            "terminal": primary_binding["terminal"],
            "next_boundary": primary_binding["next_boundary"],
        }
        if result.get("primary_r0") != expected_primary_binding:
            raise ValueError("reported primary binding drifted")
        checks += 3
        decision = result["decision"]
        if (
            decision.get("terminal") != FIXED_TERMINAL
            or decision.get("primary_r0_terminal")
            != config["primary_r0"]["terminal"]
            or decision.get("primary_r0_terminal_unchanged") is not True
            or decision.get("selection") != "NONE"
            or decision.get("shadow_authority") != "DIAGNOSTIC_ONLY"
            or decision.get("gating_route_reopened") is not False
            or decision.get("next_boundary") != config["primary_r0"]["next_boundary"]
        ):
            raise ValueError("fixed shadow or primary authority drifted")
        checks += 10
        if result.get("selection_status") != "NONE_DIAGNOSTIC_ONLY":
            raise ValueError("shadow selection status drifted")
        if result.get("drives_alerts") is not False:
            raise ValueError("shadow drives_alerts boundary drifted")
        checks += 2

        if len(frames) != int(config["output_contract"]["expected_frame_rows"]):
            raise ValueError("shadow frame row count mismatch")
        if len({row["view_row_id"] for row in frames}) != len(frames):
            raise ValueError("duplicate shadow output frame")
        primary_view_ids = {str(row["view_row_id"]) for row in primary_frames}
        if {str(row["view_row_id"]) for row in frames} != primary_view_ids:
            raise ValueError("shadow frame identity differs from bound primary")
        checks += 3

        if (
            sha256_file(frame_path)
            != result["output_files"][frame_path.name]["sha256"]
            or sha256_file(component_path)
            != result["output_files"][component_path.name]["sha256"]
        ):
            raise ValueError("shadow output hash mismatch")
        checks += 2

        classes = list(base_config["candidate_classes"])
        arm_ids = [BASELINE_ID, *EXPECTED_SHADOW_ORDER]
        for arm_id in arm_ids:
            checks += _validate_aggregate(
                rows=frames,
                arm_id=arm_id,
                classes=classes,
                reported=result["arms"][arm_id]["overall"],
                label=f"{arm_id}.overall",
            )
            for group_field in ("session_id", "role", "scene_bucket"):
                groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in frames:
                    groups[str(row[group_field])].append(row)
                reported_groups = result["arms"][arm_id][f"by_{group_field}"]
                if set(groups) != set(reported_groups):
                    raise ValueError(f"shadow aggregate group drifted: {group_field}")
                for group_id, group_rows in groups.items():
                    checks += _validate_aggregate(
                        rows=group_rows,
                        arm_id=arm_id,
                        classes=classes,
                        reported=reported_groups[group_id],
                        label=f"{arm_id}.{group_field}.{group_id}",
                    )

        if sha256_json(result["arms"][BASELINE_ID]) != sha256_json(
            primary_result["arms"][BASELINE_ID]
        ):
            raise ValueError("shadow baseline differs from bound primary baseline")
        checks += 1

        component_checks, parity = _validate_component_decisions(
            rows=component_rows,
            primary_rows=primary_components,
            config=config,
            base_config=base_config,
        )
        checks += component_checks
        if sha256_json(parity) != sha256_json(result["single_variable_parity"]):
            raise ValueError("reported single-variable parity drifted")
        checks += 1

        independently_derived_decisions = {
            candidate_id: _decision_from_report(
                report=result["arms"][candidate_id],
                rules=base_config["decision_rules"],
            )
            for candidate_id in EXPECTED_SHADOW_ORDER
        }
        for candidate_id, derived in independently_derived_decisions.items():
            reported = result["shadow_gate_diagnostics"][candidate_id]
            for name in (*RECALL_GUARD_NAMES, FP_GATE_NAME):
                _assert_equal(
                    derived["checks"][name]["passed"],
                    reported["checks"][name]["passed"],
                    f"{candidate_id}.{name}.passed",
                )
                _assert_float_equal(
                    derived["checks"][name]["value"],
                    reported["checks"][name]["value"],
                    f"{candidate_id}.{name}.value",
                )
                checks += 2

        reference_points: dict[str, dict[str, float | None]] = {}
        for reference_id in FIXED_REFERENCE_IDS:
            reference_points[reference_id] = _point_from_report(
                primary_result["arms"][reference_id]
            )
        material = {
            candidate_id: _material_assessment(
                decision=independently_derived_decisions[candidate_id],
                point=_point_from_report(result["arms"][candidate_id]),
                references=reference_points,
            )
            for candidate_id in EXPECTED_SHADOW_ORDER
        }
        if sha256_json(material) != sha256_json(
            result["material_assessment"]["by_candidate"]
        ):
            raise ValueError("reported material assessment drifted")
        if sha256_json(reference_points) != sha256_json(
            result["material_assessment"]["fixed_reference_points"]
        ):
            raise ValueError("fixed-reference points drifted")
        checks += 2

        cross = _cross_heterogeneity(result["arms"])
        if sha256_json(cross) != sha256_json(
            result["heterogeneity_assessment"]["cross_session"]
        ):
            raise ValueError("reported cross-session heterogeneity drifted")
        any_material = any(row["MATERIAL"] for row in material.values())
        heterogeneous = bool(
            cross["H_cross"] or any(row["H_min"] for row in material.values())
        )
        expected_overall = (
            "MATERIAL_SHADOW_SIGNAL_PRESENT"
            if any_material
            else config["heterogeneity_rule"][
                "no_material_but_heterogeneous_hypothesis"
            ]
            if heterogeneous
            else "MECHANISM_ONLY"
        )
        if result["material_assessment"]["overall"] != expected_overall:
            raise ValueError("material family summary drifted")
        if result["material_assessment"]["any_material"] != any_material:
            raise ValueError("material presence flag drifted")
        if result["heterogeneity_assessment"]["heterogeneous"] != heterogeneous:
            raise ValueError("heterogeneity presence flag drifted")
        checks += 4
    except Exception as exc:
        errors.append(str(exc))

    return {
        "schema_version": (
            "blindassist.dual_loop_segmentation_conditional_gating."
            "shadow_validation.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "mode": MODE,
        "status": "VALID" if not errors else "INVALID",
        "checks_completed": checks,
        "errors": errors,
        "terminal_authority": "DIAGNOSTIC_ONLY_PRIMARY_R0_UNCHANGED",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    config_path = _resolve(repo_root, args.config)
    evidence_root = _resolve(repo_root, args.evidence_root)
    output_path = _resolve(repo_root, args.output)
    result = validate(
        repo_root=repo_root,
        config_path=config_path,
        evidence_root=evidence_root,
    )
    _write_json(output_path, result)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
