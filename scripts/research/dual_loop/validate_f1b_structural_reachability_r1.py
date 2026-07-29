from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_dual_loop_f1b_structural_reachability_validation_v2"
ALERTABLE_PAIRS = {"CRITICAL_HIGH", "NEAR_HIGH", "NEAR_MEDIUM"}
REQUIRED_ENDPOINTS = {
    "EARLY_RESPONSE",
    "RISK_DISCRIMINATION",
    "RISK_CONTINUITY",
    "MULTIPLE_INCREMENT",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bound_file(project_root: Path, item: dict[str, Any]) -> None:
    path_value = item.get("path") or item.get("spec_path") or item.get("validation_path")
    expected = (
        item.get("sha256")
        or item.get("spec_sha256")
        or item.get("validation_sha256")
    )
    path = project_root / str(path_value)
    if not path.is_file():
        raise ValueError(f"bound file missing: {path_value}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"bound file hash drift: {path_value} expected={expected} actual={actual}"
        )


def validate_prerequisite(project_root: Path, item: dict[str, Any]) -> None:
    validate_bound_file(project_root, item)
    payload = read_json(project_root / item["path"])
    if payload.get("terminal") != item["required_terminal"]:
        raise ValueError(f"prerequisite terminal mismatch: {item['path']}")
    protocol_status = (
        payload.get("data_protocol_status")
        or payload.get("timing_protocol_status")
        or payload.get("protocol_status")
    )
    if protocol_status != item["required_protocol_status"]:
        raise ValueError(f"prerequisite protocol status mismatch: {item['path']}")


def require_exact_semantics(spec: dict[str, Any]) -> None:
    production = spec["production_semantics"]
    expected_production = {
        "semantic_detection_source": "OBJECT_DETECTOR",
        "detection_source_remains_object_detector": True,
        "profile": "STANDARD",
        "scenario": "GENERAL",
        "enable_approaching_center_person_mid_alert": False,
        "planner_alertable_pairs": [
            "CRITICAL_HIGH",
            "NEAR_HIGH",
            "NEAR_MEDIUM",
        ],
        "near_level_by_direction": {
            "CENTER": "HIGH",
            "LEFT": "MEDIUM",
            "RIGHT": "MEDIUM",
        },
        "critical_level_by_direction": {"CENTER": "HIGH"},
        "mid_base_level": "LOW",
        "far_base_level": "NONE",
        "temporal_promotion_max_steps": 1,
        "temporal_promotion_may_change_proximity": False,
        "high_confirm_frames": 1,
        "medium_confirm_frames": 2,
        "stabilizer_hold_ms": 600,
        "low_confidence_side_person_gate_shared_between_branches": True,
        "object_detector_event_suppression_enabled": False,
        "near_cooldown_ms": 1500,
        "critical_cooldown_ms": 850,
        "fatigue_window_ms": 12000,
        "cooldown_key": "DIRECTION_PROXIMITY",
        "effect_acceptance_rule": "speechAccepted OR vibrationAccepted",
        "state_update_clock": "SOURCE_CAPTURE_TIME",
        "delivery_clock": "RESULT_CONSUME_TIME",
    }
    if production != expected_production:
        raise ValueError("production semantics drifted from the hash-reviewed R1 contract")

    geometry = spec["geometry_information_semantics"]
    expected_geometry = {
        "vector_fields": [
            "success",
            "inlierRatio",
            "validCorridorFraction",
            "corridorResidual",
            "lowerCorridorResidual",
        ],
        "has_target_identity": False,
        "has_left_center_right_region": False,
        "has_approach_direction": False,
        "has_radial_expansion": False,
        "has_ttc": False,
        "only_attributable_region": "CENTER",
        "left_right_attribution_forbidden": True,
    }
    if geometry != expected_geometry:
        raise ValueError("geometry information semantics drifted")

    fusion = spec["admissible_fusion"]
    expected_fusion = {
        "semantic_candidate_required": True,
        "semantic_candidate_must_be_current_and_fresh": True,
        "may_change_detection_source": False,
        "may_change_label": False,
        "may_change_direction": False,
        "may_change_proximity": False,
        "may_change_risk_level": False,
        "may_create_alert_without_semantic_candidate": False,
        "may_hold_missing_semantic_candidate": False,
        "may_suppress_a_alert": False,
        "may_bypass_low_confidence_side_person_gate": False,
        "may_bypass_cooldown": False,
        "may_bypass_fatigue": False,
        "may_change_effect_acceptance": False,
        "only_permitted_action": (
            "SUBSTITUTE_ONE_MEDIUM_CONFIRMATION_IF_GEOMETRY_ATTRIBUTABLE_TO_SAME_REGION"
        ),
        "geometry_attributable_directions": ["CENTER"],
        "abstention_falls_back_to_a": True,
    }
    if fusion != expected_fusion:
        raise ValueError("admissible fusion drifted or became internally contradictory")

    if set(spec["required_endpoint_proofs"]) != REQUIRED_ENDPOINTS:
        raise ValueError("secondary endpoint proof set is incomplete")
    declaration = spec["decision_output_declaration"]
    if declaration["machine_verified"]:
        raise ValueError("decision non-access is a protocol declaration, not machine verification")
    if any(
        declaration[key]
        for key in (
            "yolo_executed_on_decision_sessions",
            "sparse_lk_executed_on_decision_sessions",
            "ab_output_viewed_on_decision_sessions",
        )
    ) or declaration["decision_sessions_consumed"] != 0:
        raise ValueError("decision candidate outputs were declared consumed")


def raw_and_temporal_states() -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = [
        {
            "state_id": "NO_CANDIDATE",
            "candidate_present": False,
            "direction": "NONE",
            "proximity": "FAR",
            "level": "NONE",
            "temporal": False,
        }
    ]
    for direction in ("CENTER", "LEFT", "RIGHT"):
        states.extend(
            [
                {
                    "state_id": f"FAR_NONE_{direction}",
                    "candidate_present": True,
                    "direction": direction,
                    "proximity": "FAR",
                    "level": "NONE",
                    "temporal": False,
                },
                {
                    "state_id": f"FAR_LOW_TEMPORAL_{direction}",
                    "candidate_present": True,
                    "direction": direction,
                    "proximity": "FAR",
                    "level": "LOW",
                    "temporal": True,
                },
                {
                    "state_id": f"MID_LOW_{direction}",
                    "candidate_present": True,
                    "direction": direction,
                    "proximity": "MID",
                    "level": "LOW",
                    "temporal": False,
                },
                {
                    "state_id": f"MID_MEDIUM_TEMPORAL_{direction}",
                    "candidate_present": True,
                    "direction": direction,
                    "proximity": "MID",
                    "level": "MEDIUM",
                    "temporal": True,
                },
            ]
        )
    states.append(
        {
            "state_id": "NEAR_HIGH_CENTER",
            "candidate_present": True,
            "direction": "CENTER",
            "proximity": "NEAR",
            "level": "HIGH",
            "temporal": False,
        }
    )
    for direction in ("LEFT", "RIGHT"):
        states.extend(
            [
                {
                    "state_id": f"NEAR_MEDIUM_{direction}",
                    "candidate_present": True,
                    "direction": direction,
                    "proximity": "NEAR",
                    "level": "MEDIUM",
                    "temporal": False,
                },
                {
                    "state_id": f"NEAR_HIGH_TEMPORAL_{direction}",
                    "candidate_present": True,
                    "direction": direction,
                    "proximity": "NEAR",
                    "level": "HIGH",
                    "temporal": True,
                },
            ]
        )
    states.append(
        {
            "state_id": "CRITICAL_HIGH_CENTER",
            "candidate_present": True,
            "direction": "CENTER",
            "proximity": "CRITICAL",
            "level": "HIGH",
            "temporal": False,
        }
    )
    return states


def derive_fresh_state_table(spec: dict[str, Any]) -> list[dict[str, Any]]:
    production = spec["production_semantics"]
    geometry = spec["geometry_information_semantics"]
    fusion = spec["admissible_fusion"]
    planner_pairs = set(production["planner_alertable_pairs"])
    attributable = set(fusion["geometry_attributable_directions"])
    rows: list[dict[str, Any]] = []
    for state in raw_and_temporal_states():
        pair = f"{state['proximity']}_{state['level']}"
        planner_eligible = state["candidate_present"] and pair in planner_pairs
        confirm_frames = {
            "HIGH": production["high_confirm_frames"],
            "MEDIUM": production["medium_confirm_frames"],
        }.get(state["level"])
        geometry_attributable = (
            state["candidate_present"]
            and state["direction"] in attributable
            and (
                state["direction"] == geometry["only_attributable_region"]
                or not geometry["left_right_attribution_forbidden"]
            )
        )
        b_can_advance = (
            planner_eligible
            and state["level"] == "MEDIUM"
            and confirm_frames is not None
            and confirm_frames > 1
            and geometry_attributable
        )
        rows.append(
            {
                **state,
                "planner_pair": pair,
                "a_deliverable_possible": planner_eligible,
                "a_confirmation_frames": confirm_frames,
                "geometry_attributable": geometry_attributable,
                "b_can_advance_delivery": b_can_advance,
                "b_can_suppress_delivery": (
                    planner_eligible and fusion["may_suppress_a_alert"]
                ),
                "b_can_extend_continuity": fusion["may_hold_missing_semantic_candidate"],
                "lead_upper_bound_frames": 1 if b_can_advance else 0,
            }
        )
    actual_ids = [row["state_id"] for row in rows]
    required_ids = spec["required_fresh_state_ids"]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(required_ids):
        raise ValueError(
            f"fresh-state coverage drifted: expected={required_ids} "
            f"actual={actual_ids}"
        )
    return rows


def derive_endpoint_proofs(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], bool]:
    fusion = spec["admissible_fusion"]
    per_step_equal = not any(
        row["b_can_advance_delivery"]
        or row["b_can_suppress_delivery"]
        or row["b_can_extend_continuity"]
        for row in rows
    )
    same_downstream_transition = all(
        (
            spec["production_semantics"][
                "low_confidence_side_person_gate_shared_between_branches"
            ],
            not fusion["may_bypass_low_confidence_side_person_gate"],
            not fusion["may_bypass_cooldown"],
            not fusion["may_bypass_fatigue"],
            not fusion["may_change_effect_acceptance"],
            not spec["production_semantics"]["object_detector_event_suppression_enabled"],
        )
    )
    history_equivalent = per_step_equal and same_downstream_transition
    proofs = {
        "EARLY_RESPONSE": {
            "reachable": any(row["b_can_advance_delivery"] for row in rows),
            "reason": "No lagged alertable state is both geometry-attributable and mutable.",
        },
        "RISK_DISCRIMINATION": {
            "reachable": any(row["b_can_suppress_delivery"] for row in rows),
            "reason": "The admissible fusion cannot suppress an A alert or alter downstream delivery.",
        },
        "RISK_CONTINUITY": {
            "reachable": any(row["b_can_extend_continuity"] for row in rows),
            "reason": "The admissible fusion cannot hold a missing semantic candidate or alter stabilizer state.",
        },
        "MULTIPLE_INCREMENT": {
            "reachable": False,
            "reason": "No constituent endpoint is reachable.",
        },
    }
    proofs["MULTIPLE_INCREMENT"]["reachable"] = any(
        proof["reachable"]
        for endpoint, proof in proofs.items()
        if endpoint != "MULTIPLE_INCREMENT"
    )
    if set(proofs) != REQUIRED_ENDPOINTS:
        raise ValueError("derived endpoint proof set is incomplete")
    return proofs, history_equivalent


def validate(project_root: Path, spec_path: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    if spec.get("schema") != (
        "blindassist_dual_loop_f1b_structural_reachability_protocol_repair_spec_v1"
    ):
        raise ValueError("unsupported R1 spec schema")
    predecessor = spec["predecessor"]
    validate_bound_file(
        project_root,
        {
            "path": predecessor["spec_path"],
            "sha256": predecessor["spec_sha256"],
        },
    )
    validate_bound_file(
        project_root,
        {
            "path": predecessor["validation_path"],
            "sha256": predecessor["validation_sha256"],
        },
    )
    if predecessor["protocol_status"] != "INVALID":
        raise ValueError("R0 protocol defect must remain explicit")
    validate_bound_file(project_root, spec["contract"])
    for prerequisite in spec["prerequisites"]:
        validate_prerequisite(project_root, prerequisite)
    for identity in spec["implementation_identities"]:
        validate_bound_file(project_root, identity)
    require_exact_semantics(spec)
    rows = derive_fresh_state_table(spec)
    proofs, history_equivalent = derive_endpoint_proofs(spec, rows)
    any_reachable = any(proof["reachable"] for proof in proofs.values())
    terminal = "NO_INCREMENT" if not any_reachable and history_equivalent else "REQUIRES_EMPIRICAL_AB"
    terminal_rule = spec["terminal_rule"]
    if terminal == "NO_INCREMENT" and (
        terminal_rule["f1c_authorized_on_no_increment"]
        or not terminal_rule["paper_claim_stops_on_no_increment"]
    ):
        raise ValueError("NO_INCREMENT terminal semantics drifted")
    return {
        "schema": SCHEMA,
        "protocol_id": spec["protocol_id"],
        "spec_sha256": sha256_file(spec_path),
        "predecessor_protocol_status": "INVALID",
        "checked_implementation_count": len(spec["implementation_identities"]),
        "prerequisite_count": len(spec["prerequisites"]),
        "decision_output_status": "DECLARED_NOT_ACCESSED_NOT_MACHINE_VERIFIED",
        "decision_sessions_consumed": 0,
        "fresh_state_table": rows,
        "fresh_state_count": len(rows),
        "per_step_transition_equal": not any(
            row["b_can_advance_delivery"]
            or row["b_can_suppress_delivery"]
            or row["b_can_extend_continuity"]
            for row in rows
        ),
        "history_equivalent_by_induction": history_equivalent,
        "endpoint_proofs": proofs,
        "lead_upper_bound_frames": max(
            row["lead_upper_bound_frames"] for row in rows
        ),
        "terminal": terminal,
        "science_protocol_status": "VALID" if terminal == "NO_INCREMENT" else "DESIGN_FROZEN",
        "reason": "NO_ADMISSIBLE_FRESH_OR_HISTORY_TRANSITION_CAN_CHANGE_DELIVERED_ALERTS",
        "f1c_authorized": False,
        "paper_claim_stops": terminal == "NO_INCREMENT",
        "claim_ceiling": "DEVELOPMENT_ROUTE_REJECTION_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    spec_path = (
        args.spec.resolve()
        if args.spec.is_absolute()
        else (project_root / args.spec).resolve()
    )
    output = (
        args.output.resolve()
        if args.output.is_absolute()
        else (project_root / args.output).resolve()
    )
    if output.exists():
        raise SystemExit("refusing to overwrite output")
    payload = validate(project_root, spec_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "terminal": payload["terminal"],
                "science_protocol_status": payload["science_protocol_status"],
                "fresh_state_count": payload["fresh_state_count"],
                "history_equivalent_by_induction": payload[
                    "history_equivalent_by_induction"
                ],
                "lead_upper_bound_frames": payload["lead_upper_bound_frames"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
