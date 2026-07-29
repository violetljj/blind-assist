from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_dual_loop_f1b_structural_reachability_validation_v3"
REQUIRED_ENDPOINTS = {
    "EARLY_RESPONSE",
    "RISK_DISCRIMINATION",
    "RISK_CONTINUITY",
    "MULTIPLE_INCREMENT",
}
EXPECTED_IMPLEMENTATION_PATHS = {
    "device-benchmark/src/main/java/com/linnan/blindassist/benchmark/SparseLkGeometryProbe.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/RiskAnalyzer.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/ConservativeRiskFusionPolicy.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/TemporalRiskTracker.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/RiskStabilizer.kt",
    "core/assist/src/main/java/com/linnan/blindassist/feedback/FeedbackPlanner.kt",
    "core/assist/src/main/java/com/linnan/blindassist/alert/AlertProfile.kt",
    "core/assist/src/main/java/com/linnan/blindassist/session/AssistEngine.kt",
    "core/assist/src/main/java/com/linnan/blindassist/session/AssistDecisionKernel.kt",
    "core/assist/src/main/java/com/linnan/blindassist/session/AssistSessionCoordinator.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/RiskEventTracker.kt",
    "core/device/src/main/java/com/linnan/blindassist/feedback/FeedbackController.kt",
    "core/device/src/main/java/com/linnan/blindassist/feedback/FeedbackFatigueController.kt",
}
EXPECTED_PREREQUISITE_PATHS = {
    "artifacts.local/evidence/dual-loop/f1a-negative-category-supplement-r1/validation.json",
    "artifacts.local/evidence/dual-loop/f1b0-timing-baseline-r0/validation.json",
}


def load_r1_module() -> Any:
    path = Path(__file__).with_name("validate_f1b_structural_reachability_r1.py")
    module_spec = importlib.util.spec_from_file_location(
        "validate_f1b_structural_reachability_r1_for_r2", path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load R1 validator helpers")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


R1 = load_r1_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_inherited_identity_sets(r1_spec: dict[str, Any]) -> None:
    implementation_paths = {
        identity["path"] for identity in r1_spec["implementation_identities"]
    }
    if implementation_paths != EXPECTED_IMPLEMENTATION_PATHS:
        raise ValueError(
            "inherited implementation identity set drifted: "
            f"missing={sorted(EXPECTED_IMPLEMENTATION_PATHS - implementation_paths)} "
            f"extra={sorted(implementation_paths - EXPECTED_IMPLEMENTATION_PATHS)}"
        )
    prerequisite_paths = {
        prerequisite["path"] for prerequisite in r1_spec["prerequisites"]
    }
    if prerequisite_paths != EXPECTED_PREREQUISITE_PATHS:
        raise ValueError("inherited prerequisite set drifted")


def validate_predecessor(project_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    predecessor = spec["predecessor"]
    R1.validate_bound_file(
        project_root,
        {
            "path": predecessor["spec_path"],
            "sha256": predecessor["spec_sha256"],
        },
    )
    R1.validate_bound_file(
        project_root,
        {
            "path": predecessor["validation_path"],
            "sha256": predecessor["validation_sha256"],
        },
    )
    if predecessor["protocol_status"] != "INVALID":
        raise ValueError("R1 protocol defect must remain explicit")
    r1_spec = read_json(project_root / predecessor["spec_path"])
    require_inherited_identity_sets(r1_spec)
    R1.validate_bound_file(project_root, r1_spec["contract"])
    for prerequisite in r1_spec["prerequisites"]:
        R1.validate_prerequisite(project_root, prerequisite)
    for identity in r1_spec["implementation_identities"]:
        R1.validate_bound_file(project_root, identity)
    R1.require_exact_semantics(r1_spec)
    return r1_spec


def require_exact_r2_repair(spec: dict[str, Any]) -> None:
    if spec["inherited_bindings"] != {
        "contract_prerequisites_and_13_implementation_identities": (
            "EXACTLY_INHERITED_FROM_HASH_BOUND_R1_SPEC"
        ),
        "decision_output_declaration": "EXACTLY_INHERITED_FROM_HASH_BOUND_R1_SPEC",
        "geometry_information_semantics": "EXACTLY_INHERITED_FROM_HASH_BOUND_R1_SPEC",
        "production_semantics_except_repaired_side_temporal_state": (
            "EXACTLY_INHERITED_FROM_HASH_BOUND_R1_SPEC"
        ),
        "admissible_fusion_except_repaired_action_precondition": (
            "EXACTLY_INHERITED_FROM_HASH_BOUND_R1_SPEC"
        ),
    }:
        raise ValueError("R2 inherited binding contract drifted")
    expected = {
        "side_near_temporal_level": "MEDIUM",
        "confirmation_substitution_requires_planner_eligible_pair": True,
        "fusion_action_reachability_definition": [
            "candidate_present",
            "level_is_MEDIUM",
            "confirmation_frames_gt_1",
            "geometry_attributable_to_same_region",
            "planner_pair_is_alertable",
        ],
        "per_step_transition_equal_definition": (
            "No fusion action, alert suppression, missing-candidate hold, "
            "source/label/direction/proximity/level change, downstream-gate bypass, "
            "cooldown/fatigue bypass, or effect-acceptance change is reachable in any fresh state."
        ),
        "history_induction_definition": (
            "Identical initial state plus identical internal and delivered transition at every "
            "step implies identical temporal, stabilizer, side-person, event, hold, cooldown, "
            "fatigue and effect-delivery history."
        ),
    }
    if spec["repairs"] != expected:
        raise ValueError("R2 repair semantics drifted")
    if set(spec["required_endpoint_proofs"]) != REQUIRED_ENDPOINTS:
        raise ValueError("R2 endpoint proof set is incomplete")
    if spec["terminal_rule"] != {
        "no_internal_or_delivered_transition_and_history_equivalent": "NO_INCREMENT",
        "any_reachable_internal_or_delivered_difference": "REQUIRES_EMPIRICAL_AB",
        "identity_or_semantic_drift": "INVALID",
        "f1c_authorized_on_no_increment": False,
        "paper_claim_stops_on_no_increment": True,
    }:
        raise ValueError("R2 terminal rule drifted")


def effective_semantics(
    r1_spec: dict[str, Any],
    r2_spec: dict[str, Any],
) -> dict[str, Any]:
    effective = copy.deepcopy(r1_spec)
    effective["required_fresh_state_ids"] = r2_spec["required_fresh_state_ids"]
    effective["required_endpoint_proofs"] = r2_spec["required_endpoint_proofs"]
    effective["production_semantics"]["side_near_temporal_level"] = r2_spec["repairs"][
        "side_near_temporal_level"
    ]
    effective["admissible_fusion"][
        "confirmation_substitution_requires_planner_eligible_pair"
    ] = r2_spec["repairs"][
        "confirmation_substitution_requires_planner_eligible_pair"
    ]
    return effective


def fresh_states() -> list[dict[str, Any]]:
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
                    "state_id": f"NEAR_MEDIUM_TEMPORAL_{direction}",
                    "candidate_present": True,
                    "direction": direction,
                    "proximity": "NEAR",
                    "level": "MEDIUM",
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


def derive_state_table(effective: dict[str, Any]) -> list[dict[str, Any]]:
    production = effective["production_semantics"]
    geometry = effective["geometry_information_semantics"]
    fusion = effective["admissible_fusion"]
    planner_pairs = set(production["planner_alertable_pairs"])
    attributable = set(fusion["geometry_attributable_directions"])
    requires_planner = fusion[
        "confirmation_substitution_requires_planner_eligible_pair"
    ]
    rows: list[dict[str, Any]] = []
    for state in fresh_states():
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
        fusion_action_reachable = (
            state["candidate_present"]
            and state["level"] == "MEDIUM"
            and confirm_frames is not None
            and confirm_frames > 1
            and geometry_attributable
            and (planner_eligible or not requires_planner)
        )
        rows.append(
            {
                **state,
                "planner_pair": pair,
                "a_deliverable_possible": planner_eligible,
                "a_confirmation_frames": confirm_frames,
                "geometry_attributable": geometry_attributable,
                "fusion_action_reachable": fusion_action_reachable,
                "b_can_advance_delivery": (
                    fusion_action_reachable and planner_eligible
                ),
                "b_can_suppress_delivery": (
                    planner_eligible and fusion["may_suppress_a_alert"]
                ),
                "b_can_extend_continuity": fusion[
                    "may_hold_missing_semantic_candidate"
                ],
                "lead_upper_bound_frames": (
                    1 if fusion_action_reachable and planner_eligible else 0
                ),
            }
        )
    actual_ids = [row["state_id"] for row in rows]
    required_ids = effective["required_fresh_state_ids"]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(required_ids):
        raise ValueError(
            f"R2 fresh-state coverage drifted: expected={required_ids} actual={actual_ids}"
        )
    return rows


def derive_proofs(
    effective: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], bool, bool]:
    fusion = effective["admissible_fusion"]
    no_internal_or_delivery_transition = not any(
        row["fusion_action_reachable"]
        or row["b_can_suppress_delivery"]
        or row["b_can_extend_continuity"]
        for row in rows
    )
    same_downstream_transition = all(
        (
            effective["production_semantics"][
                "low_confidence_side_person_gate_shared_between_branches"
            ],
            not fusion["may_change_detection_source"],
            not fusion["may_change_label"],
            not fusion["may_change_direction"],
            not fusion["may_change_proximity"],
            not fusion["may_change_risk_level"],
            not fusion["may_bypass_low_confidence_side_person_gate"],
            not fusion["may_bypass_cooldown"],
            not fusion["may_bypass_fatigue"],
            not fusion["may_change_effect_acceptance"],
        )
    )
    history_equivalent = no_internal_or_delivery_transition and same_downstream_transition
    proofs = {
        "EARLY_RESPONSE": {
            "reachable": any(row["b_can_advance_delivery"] for row in rows),
            "reason": "No planner-eligible lagged state is geometry-attributable.",
        },
        "RISK_DISCRIMINATION": {
            "reachable": any(row["b_can_suppress_delivery"] for row in rows),
            "reason": "Fusion cannot suppress A or alter downstream delivery.",
        },
        "RISK_CONTINUITY": {
            "reachable": any(row["b_can_extend_continuity"] for row in rows),
            "reason": "Fusion cannot hold missing semantic candidates.",
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
    return proofs, no_internal_or_delivery_transition, history_equivalent


def validate(project_root: Path, spec_path: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    if spec.get("schema") != (
        "blindassist_dual_loop_f1b_structural_reachability_protocol_repair_spec_v2"
    ):
        raise ValueError("unsupported R2 spec schema")
    r1_spec = validate_predecessor(project_root, spec)
    require_exact_r2_repair(spec)
    effective = effective_semantics(r1_spec, spec)
    rows = derive_state_table(effective)
    proofs, transition_equal, history_equal = derive_proofs(effective, rows)
    any_endpoint_reachable = any(proof["reachable"] for proof in proofs.values())
    terminal = (
        "NO_INCREMENT"
        if transition_equal and history_equal and not any_endpoint_reachable
        else "REQUIRES_EMPIRICAL_AB"
    )
    return {
        "schema": SCHEMA,
        "protocol_id": spec["protocol_id"],
        "spec_sha256": sha256_file(spec_path),
        "predecessor_protocol_status": "INVALID",
        "checked_implementation_count": len(r1_spec["implementation_identities"]),
        "prerequisite_count": len(r1_spec["prerequisites"]),
        "decision_output_status": "DECLARED_NOT_ACCESSED_NOT_MACHINE_VERIFIED",
        "decision_sessions_consumed": 0,
        "fresh_state_table": rows,
        "fresh_state_count": len(rows),
        "fusion_action_reachable_count": sum(
            row["fusion_action_reachable"] for row in rows
        ),
        "per_step_internal_and_delivery_transition_equal": transition_equal,
        "history_equivalent_by_induction": history_equal,
        "endpoint_proofs": proofs,
        "lead_upper_bound_frames": max(
            row["lead_upper_bound_frames"] for row in rows
        ),
        "terminal": terminal,
        "science_protocol_status": (
            "VALID" if terminal == "NO_INCREMENT" else "DESIGN_FROZEN"
        ),
        "reason": "NO_ADMISSIBLE_INTERNAL_OR_DELIVERED_TRANSITION_EXISTS",
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
                "fusion_action_reachable_count": payload[
                    "fusion_action_reachable_count"
                ],
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
