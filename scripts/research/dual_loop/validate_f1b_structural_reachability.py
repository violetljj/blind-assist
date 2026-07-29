from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_dual_loop_f1b_structural_reachability_validation_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bound_file(project_root: Path, item: dict[str, Any]) -> None:
    path = project_root / item["path"]
    if not path.is_file():
        raise ValueError(f"bound file missing: {item['path']}")
    actual = sha256_file(path)
    if actual != item["sha256"]:
        raise ValueError(
            f"bound file hash drift: {item['path']} expected={item['sha256']} actual={actual}"
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


def analyze_reachability(spec: dict[str, Any]) -> dict[str, Any]:
    decision = spec["decision_output_access"]
    if any(
        decision[key]
        for key in (
            "yolo_executed_on_decision_sessions",
            "sparse_lk_executed_on_decision_sessions",
            "ab_output_viewed_on_decision_sessions",
        )
    ):
        raise ValueError("decision outputs must remain unconsumed for the structural gate")

    semantics = spec["frozen_semantics"]
    fusion = spec["maximally_thin_admissible_fusion"]
    if any(
        semantics[key]
        for key in (
            "geometry_has_target_identity",
            "geometry_has_left_center_right_region",
            "geometry_has_approach_direction",
            "geometry_has_radial_expansion",
            "geometry_has_ttc",
        )
    ):
        raise ValueError("spec overstates the existing Sparse-LK vector")
    if semantics["only_defensible_geometry_region"] != "CENTER_CORRIDOR_GLOBAL":
        raise ValueError("unexpected geometry region semantics")
    if semantics["center_near_level"] != "HIGH" or semantics["center_critical_level"] != "HIGH":
        raise ValueError("center near/critical must retain immediate HIGH semantics")
    if semantics["side_near_level"] != "MEDIUM" or semantics["medium_confirm_frames"] != 2:
        raise ValueError("side-near confirmation semantics drifted")
    if semantics["high_confirm_frames"] != 1:
        raise ValueError("HIGH must remain immediate")

    forbidden_changes = (
        "may_change_label",
        "may_change_direction",
        "may_change_proximity",
        "may_change_risk_level",
        "may_create_alert_from_no_semantic_candidate",
        "may_hold_missing_semantic_candidate",
        "may_bypass_production_cooldown_or_fatigue",
    )
    if any(fusion[key] for key in forbidden_changes):
        raise ValueError("fusion exceeds the frozen thin boundary")
    if fusion["region_attribution_rule"] != (
        "CENTER only; LEFT and RIGHT abstain because the existing vector is global "
        "center-corridor evidence"
    ):
        raise ValueError("region attribution rule drifted")
    if not fusion["abstention_falls_back_to_a"]:
        raise ValueError("abstention must fall back to A")

    rows = spec["structural_truth_table"]
    required_states = {
        "NO_CANDIDATE",
        "FAR_NONE",
        "MID_LOW",
        "NEAR_HIGH",
        "CRITICAL_HIGH",
        "NEAR_MEDIUM",
    }
    actual_states = {row["semantic_state"] for row in rows}
    if actual_states != required_states:
        raise ValueError(
            f"truth-table state mismatch: expected={sorted(required_states)} "
            f"actual={sorted(actual_states)}"
        )
    reachable = [row for row in rows if row["b_permitted_difference"]]
    lead_upper_bound_frames = 1 if reachable else 0
    stop = spec["dominance_stop_rule"]
    if lead_upper_bound_frames != stop["lead_upper_bound_frames"]:
        raise ValueError("declared structural upper bound does not match the truth table")
    terminal = "NO_INCREMENT" if lead_upper_bound_frames == 0 else "REQUIRES_EMPIRICAL_AB"
    if terminal != stop["terminal"]:
        raise ValueError("declared terminal does not match structural reachability")
    if terminal == "NO_INCREMENT" and (
        stop["decision_execution_required"]
        or stop["f1c_authorized"]
        or not stop["paper_claim_stops"]
    ):
        raise ValueError("NO_INCREMENT stop semantics drifted")

    return {
        "reachable_b_advance_rows": reachable,
        "lead_upper_bound_frames": lead_upper_bound_frames,
        "terminal": terminal,
        "science_protocol_status": stop["science_protocol_status"],
    }


def validate(project_root: Path, spec_path: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    if spec.get("schema") != "blindassist_dual_loop_f1b_structural_reachability_spec_v1":
        raise ValueError("unsupported spec schema")
    validate_bound_file(project_root, spec["contract"])
    for prerequisite in spec["prerequisites"]:
        validate_prerequisite(project_root, prerequisite)
    for identity in spec["implementation_identities"]:
        validate_bound_file(project_root, identity)
    analysis = analyze_reachability(spec)
    return {
        "schema": SCHEMA,
        "protocol_id": spec["protocol_id"],
        "spec_sha256": sha256_file(spec_path),
        "checked_implementation_count": len(spec["implementation_identities"]),
        "prerequisite_count": len(spec["prerequisites"]),
        "decision_outputs_accessed": False,
        "decision_sessions_consumed": 0,
        "primary_endpoint": spec["fixed_primary_endpoint"]["name"],
        "reachable_b_advance_rows": analysis["reachable_b_advance_rows"],
        "lead_upper_bound_frames": analysis["lead_upper_bound_frames"],
        "terminal": analysis["terminal"],
        "science_protocol_status": analysis["science_protocol_status"],
        "reason": "EXISTING_GEOMETRY_INTERFACE_HAS_NO_ADMISSIBLE_DELIVERABLE_INCREMENT",
        "f1c_authorized": False,
        "paper_claim_stops": True,
        "claim_ceiling": "DEVELOPMENT_ROUTE_SCREEN_ONLY",
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
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
