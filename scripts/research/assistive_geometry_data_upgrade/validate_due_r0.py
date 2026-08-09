"""Validate the AG-DUE R0 lock and evaluate metadata-only source manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELATIVE = Path(
    "docs/research/assistive-geometry-data-upgrade/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_GAP_DRIVEN_SOURCE_ADMISSION_PROTOCOL_2026-08-10.json"
)

DIRECT_PROVENANCE = {
    "SOURCE_NATIVE_ANNOTATION",
    "SOURCE_NATIVE_SENSOR",
    "DETERMINISTIC_DERIVED",
}
UPGRADEABLE_PATHS = {"DETERMINISTIC_DERIVATION", "MULTI_TEACHER_CONSENSUS"}
VALID_PROVENANCE = DIRECT_PROVENANCE | {
    "TEACHER_CONSENSUS",
    "SINGLE_TEACHER_OR_HEURISTIC_PROPOSAL",
    "UNKNOWN",
}
VALID_QUALITY = {"VALIDATED_FOR_CLAIM", "CHARACTERIZED_NOT_VALIDATED", "UNKNOWN"}
VALID_UPGRADE_PATHS = UPGRADEABLE_PATHS | {
    "NONE",
    "SINGLE_TEACHER",
    "HEURISTIC",
    "VLM_PROPOSAL",
}
DIRECT_EVIDENCE_BASIS = {
    "TRACKED_PROJECT_MANIFEST",
    "PUBLISHED_SOURCE_MANIFEST",
    "SOURCE_SPECIFIC_INTEGRITY_AUDIT",
}
VALID_EVIDENCE_BASIS = DIRECT_EVIDENCE_BASIS | {
    "TEACHER_OR_HEURISTIC_PROPOSAL",
    "UNKNOWN",
}
SOURCE_REQUIRED = {
    "schema",
    "manifest_id",
    "status",
    "source",
    "license",
    "ethics_privacy_access",
    "ancestry",
    "independence",
    "identity_roster",
    "access_receipt",
    "capabilities",
}
OBSERVATION_REQUIRED = {
    "total_frames",
    "orientation_frame_counts",
    "parent_frame_counts",
    "provenance_kind",
    "quality_status",
    "upgrade_path",
    "evidence_basis",
    "parent_identity_namespace",
    "orientation_and_camera_basis",
    "derivation_receipt",
    "teacher_receipts",
    "unknown_treated_as_negative",
}
EVIDENCE_REQUIRED = {
    "kind",
    "receipt",
    "receipt_sha256",
    "claim_id",
    "claim_definition",
    "count_basis",
    "source_object_sha256",
    "source_field_mapping",
    "alignment_registration_units_coordinate_receipt_sha256",
    "source_specific_verifier_sha256",
}
VALID_ORIENTATION_BASES = {
    "DISPLAY_UPRIGHT_WITH_BOUND_CAMERA_K",
    "SOURCE_NATIVE_WITH_EXPLICIT_UPRIGHT_K_MAPPING",
    "NOT_APPLICABLE",
    "UNKNOWN",
}


class ContractError(ValueError):
    """Raised when a frozen DUE contract or source manifest drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789ABCDEF" for char in value)
    )


def _validate_receipt(receipt: dict[str, Any]) -> None:
    required = {
        "payload_opened",
        "rgb_visual_access",
        "geometry_payload_access",
        "model_outcome_access",
        "confirmation_outcome_access",
        "selection_or_tuning_influence",
    }
    require(set(receipt) == required, "access receipt field set drift")
    require(all(receipt[key] is False for key in required), "metadata-only access boundary violated")


def validate_source_schema_contract(schema: dict[str, Any]) -> None:
    """Keep the declared JSON Schema and the fail-closed Python checker aligned."""

    require(schema.get("$id") == "blindassist.assistive_geometry_due.r0_source_manifest.v1", "declared source schema id drift")
    require(set(schema.get("required", [])) == SOURCE_REQUIRED, "declared source required fields drift")
    observation_schema = schema["properties"]["capabilities"]["additionalProperties"]
    require(set(observation_schema.get("required", [])) == OBSERVATION_REQUIRED, "declared capability required fields drift")
    evidence_schema = observation_schema["properties"]["evidence_basis"]
    require(set(evidence_schema.get("required", [])) == EVIDENCE_REQUIRED, "declared evidence basis required fields drift")


def validate_gap_semantics(
    gap_contract: dict[str, Any],
    dca_requirements: dict[str, Any],
    dca_protocol: dict[str, Any],
    f1_protocol: dict[str, Any],
) -> None:
    """Prove DUE did not silently weaken DCA or F1 source requirements."""

    dca_mapping = {
        "CAPABILITY_GAP_RIGHT_CENSOR": "AG_QSF_H1_REOPEN",
        "CAPABILITY_GAP_CORRIDOR": "AG_CBF_R0_STYLE_GRID",
        "CAPABILITY_GAP_R2_FACTOR_INTERVENTION": "AG_FCI_R0_FOR_R2_DECISION",
    }
    for gap_id, hypothesis_id in dca_mapping.items():
        due_gap = gap_contract["gaps"][gap_id]
        dca_gap = dca_requirements["hypotheses"][hypothesis_id]
        require(due_gap["capabilities"] == dca_gap["capabilities"], f"DCA capability threshold drift: {gap_id}")
        require(due_gap["joint_parent_gate"] == dca_gap["joint_parent_gate"], f"DCA joint-parent gate drift: {gap_id}")

    firewall = gap_contract["protected_identity_firewall"]
    require(
        firewall["forbidden_roster_contract_sha256"] == gap_contract["source_dca"]["protocol_sha256"],
        "forbidden roster binding drift",
    )
    require(firewall["forbidden_parent_ids"] == dca_protocol["protected_parent_ids"], "protected parent roster drift")
    require(firewall["forbidden_session_ids"] == dca_protocol["protected_visit_ids"], "protected session roster drift")

    f1_gap = gap_contract["gaps"]["CAPABILITY_GAP_R2_F1_SUPERVISION"]
    require(
        set(f1_gap["capabilities"])
        == {"oracle_depth_factor", "oracle_support_factor", "r2_obstacle_boundary_truth_materialized"},
        "F1 source capability set drift",
    )
    require(f1_gap["joint_parent_gate"] == {
        "minimum_joint_parents": 12,
        "minimum_fit_parents": 8,
        "minimum_eval_parents": 4,
    }, "F1 joint role feasibility drift")
    require(
        f1_protocol["unique_successor"]["id"]
        == "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK",
        "F1 successor drift",
    )
    require(
        "teacher, model consensus" in f1_protocol["supervision_admission_contract"]["provenance_policy"],
        "F1 teacher provenance firewall drift",
    )
    for gap_id, gap in gap_contract["gaps"].items():
        require(
            gap["permitted_upgrade_paths"] == ["DETERMINISTIC_DERIVATION"],
            f"teacher or heuristic path leaked into truth gate: {gap_id}",
        )


def validate_source_manifest(source: dict[str, Any]) -> list[str]:
    """Validate manifest shape and return hard governance rejection reasons."""

    require(set(source) == SOURCE_REQUIRED, "source manifest field set drift")
    require(source["schema"] == "blindassist.assistive_geometry_due.r0_source_manifest.v1", "source schema drift")
    require(source["status"] == "SOURCE_DISCOVERY_METADATA_ONLY", "source status drift")
    require(isinstance(source["manifest_id"], str) and len(source["manifest_id"]) >= 8, "manifest id invalid")

    identity = source["source"]
    require(
        set(identity) == {"source_id", "source_family", "source_version", "identity_basis", "payload_presence"},
        "source identity field set drift",
    )
    for key in ("source_id", "source_family", "source_version", "identity_basis"):
        require(isinstance(identity[key], str) and identity[key].strip(), f"source identity missing: {key}")
    require(identity["payload_presence"] in {"ABSENT", "LOCAL_EXISTING_NOT_OPENED", "UNKNOWN"}, "payload presence drift")

    license_info = source["license"]
    require(set(license_info) == {"internal_research_status", "redistribution_status", "receipt"}, "license field set drift")
    require(
        license_info["internal_research_status"] in {"VERIFIED_FOR_INTERNAL_RESEARCH", "UNKNOWN", "REJECTED"},
        "license status drift",
    )
    require(license_info["redistribution_status"] in {"VERIFIED", "NOT_AUTHORIZED", "UNKNOWN"}, "redistribution status drift")
    require(isinstance(license_info["receipt"], str) and license_info["receipt"].strip(), "license receipt missing")

    ethics = source["ethics_privacy_access"]
    require(
        set(ethics)
        == {
            "human_subject_presence",
            "privacy_review_status",
            "access_terms_status",
            "sensitive_content_handling",
            "receipt",
            "receipt_sha256",
        },
        "ethics/privacy/access field set drift",
    )
    require(ethics["human_subject_presence"] in {"NONE", "POSSIBLE", "PRESENT", "UNKNOWN"}, "human-subject status drift")
    require(
        ethics["privacy_review_status"] in {"VERIFIED_FOR_INTERNAL_RESEARCH", "REQUIRED_NOT_COMPLETE", "UNKNOWN"},
        "privacy review status drift",
    )
    require(ethics["access_terms_status"] in {"VERIFIED", "UNKNOWN", "REJECTED"}, "access terms status drift")
    require(isinstance(ethics["sensitive_content_handling"], str) and ethics["sensitive_content_handling"].strip(), "sensitive-content handling missing")
    require(isinstance(ethics["receipt"], str) and ethics["receipt"].strip(), "ethics/privacy/access receipt missing")
    require(_is_sha256(ethics["receipt_sha256"]), "ethics/privacy/access receipt SHA invalid")

    ancestry = source["ancestry"]
    require(set(ancestry) == {"status", "root_identity", "derivative_chain"}, "ancestry field set drift")
    require(ancestry["status"] in {"VERIFIED", "UNKNOWN", "CONFLICT"}, "ancestry status drift")
    require(isinstance(ancestry["root_identity"], str) and ancestry["root_identity"].strip(), "root identity missing")
    require(isinstance(ancestry["derivative_chain"], list), "derivative chain invalid")

    independence = source["independence"]
    require(
        set(independence) == {"status", "parent_identity_type", "independence_group_basis"},
        "independence field set drift",
    )
    require(independence["status"] in {"VERIFIED", "UNKNOWN", "CONFLICT"}, "independence status drift")
    require(isinstance(independence["parent_identity_type"], str) and independence["parent_identity_type"].strip(), "parent identity missing")
    require(isinstance(independence["independence_group_basis"], str) and independence["independence_group_basis"].strip(), "independence basis missing")
    roster = source["identity_roster"]
    require(
        set(roster)
        == {
            "parent_ids",
            "session_ids",
            "ancestry_group_ids",
            "history_roles",
            "claimed_freshness",
            "forbidden_roster_contract_sha256",
        },
        "identity roster field set drift",
    )
    for key in ("parent_ids", "session_ids", "ancestry_group_ids", "history_roles"):
        require(isinstance(roster[key], list), f"identity roster list invalid: {key}")
        require(len(roster[key]) == len(set(roster[key])), f"identity roster duplicates: {key}")
    require(roster["history_roles"], "history roles missing")
    valid_history_roles = {
        "SOURCE_DISCOVERY",
        "TRAIN",
        "CANARY",
        "SYNTHETIC",
        "PROJECT_CONSUMED_DEVELOPMENT",
        "DEVELOPMENT_SELECTION",
        "DEVELOPMENT_CALIBRATION",
        "CALIBRATION",
        "CONFIRMATION",
        "SEALED_UNSEEN",
    }
    require(set(roster["history_roles"]) <= valid_history_roles, "history role drift")
    require(
        roster["claimed_freshness"]
        in {"FRESH_SOURCE_DISCOVERY", "DISCLOSED_EXISTING_TRAIN", "CONSUMED_OR_PROTECTED", "UNKNOWN"},
        "claimed freshness drift",
    )
    require(_is_sha256(roster["forbidden_roster_contract_sha256"]), "forbidden roster contract SHA invalid")
    _validate_receipt(source["access_receipt"])

    capabilities = source["capabilities"]
    require(isinstance(capabilities, dict) and capabilities, "capabilities must be a non-empty object")
    for name, observation in capabilities.items():
        require(isinstance(name, str) and name, "capability name invalid")
        require(set(observation) == OBSERVATION_REQUIRED, f"capability field set drift: {name}")
        total = observation["total_frames"]
        require(isinstance(total, int) and not isinstance(total, bool) and total >= 0, f"total frames invalid: {name}")
        orientations = observation["orientation_frame_counts"]
        require(set(orientations) == {"portrait", "landscape"}, f"orientation field set drift: {name}")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in orientations.values()), f"orientation count invalid: {name}")
        require(sum(orientations.values()) == total, f"orientation total mismatch: {name}")
        parent_counts = observation["parent_frame_counts"]
        require(isinstance(parent_counts, dict) and parent_counts, f"parent counts missing: {name}")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in parent_counts.values()), f"parent count invalid: {name}")
        require(sum(parent_counts.values()) == total, f"parent total mismatch: {name}")
        namespace = observation["parent_identity_namespace"]
        require(isinstance(namespace, str) and namespace.strip(), f"parent namespace missing: {name}")
        require(
            all(parent_id.startswith(f"{namespace}:") for parent_id in parent_counts),
            f"parent id escaped declared namespace: {name}",
        )
        require(set(parent_counts) <= set(roster["parent_ids"]), f"capability parent outside identity roster: {name}")
        require(
            observation["orientation_and_camera_basis"] in VALID_ORIENTATION_BASES,
            f"orientation and camera basis drift: {name}",
        )
        require(observation["provenance_kind"] in VALID_PROVENANCE, f"provenance drift: {name}")
        require(observation["quality_status"] in VALID_QUALITY, f"quality drift: {name}")
        require(observation["upgrade_path"] in VALID_UPGRADE_PATHS, f"upgrade path drift: {name}")
        require(observation["unknown_treated_as_negative"] is False, f"UNKNOWN used as negative: {name}")
        evidence = observation["evidence_basis"]
        require(
            set(evidence) == EVIDENCE_REQUIRED,
            f"evidence basis field set drift: {name}",
        )
        require(evidence["kind"] in VALID_EVIDENCE_BASIS, f"evidence basis kind drift: {name}")
        require(isinstance(evidence["receipt"], str) and evidence["receipt"].strip(), f"evidence receipt missing: {name}")
        require(
            _is_sha256(evidence["receipt_sha256"]),
            f"evidence receipt SHA invalid: {name}",
        )
        require(
            isinstance(evidence["claim_id"], str) and len(evidence["claim_id"].strip()) >= 3,
            f"claim id missing: {name}",
        )
        require(
            isinstance(evidence["claim_definition"], str) and len(evidence["claim_definition"].strip()) >= 8,
            f"claim definition missing: {name}",
        )
        require(
            isinstance(evidence["count_basis"], str) and len(evidence["count_basis"].strip()) >= 8,
            f"count basis missing: {name}",
        )
        require(_is_sha256(evidence["source_object_sha256"]), f"source object SHA invalid: {name}")
        require(
            _is_sha256(evidence["alignment_registration_units_coordinate_receipt_sha256"]),
            f"alignment/registration/units/coordinate receipt SHA invalid: {name}",
        )
        require(
            isinstance(evidence["source_field_mapping"], list)
            and evidence["source_field_mapping"]
            and all(isinstance(item, str) and item.strip() for item in evidence["source_field_mapping"]),
            f"source field mapping missing: {name}",
        )
        require(
            _is_sha256(evidence["source_specific_verifier_sha256"]),
            f"source verifier SHA invalid: {name}",
        )
        require(observation["derivation_receipt"] is None or isinstance(observation["derivation_receipt"], str), f"derivation receipt invalid: {name}")
        require(isinstance(observation["teacher_receipts"], list), f"teacher receipts invalid: {name}")
        if observation["provenance_kind"] == "DETERMINISTIC_DERIVED":
            require(bool(observation["derivation_receipt"]), f"deterministic derivation receipt missing: {name}")
        if observation["provenance_kind"] == "TEACHER_CONSENSUS":
            require(len(observation["teacher_receipts"]) >= 2, f"teacher consensus needs at least two receipts: {name}")

    hard_reasons: list[str] = []
    if license_info["internal_research_status"] != "VERIFIED_FOR_INTERNAL_RESEARCH":
        hard_reasons.append("LICENSE_NOT_VERIFIED_FOR_INTERNAL_RESEARCH")
    if ethics["privacy_review_status"] != "VERIFIED_FOR_INTERNAL_RESEARCH":
        hard_reasons.append("PRIVACY_REVIEW_NOT_VERIFIED")
    if ethics["access_terms_status"] != "VERIFIED":
        hard_reasons.append("ACCESS_TERMS_NOT_VERIFIED")
    if ancestry["status"] != "VERIFIED":
        hard_reasons.append("ANCESTRY_NOT_VERIFIED")
    if independence["status"] != "VERIFIED":
        hard_reasons.append("INDEPENDENCE_NOT_VERIFIED")
    return hard_reasons


def _evaluate_capability(observation: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    direct = (
        observation["provenance_kind"] in DIRECT_PROVENANCE
        and observation["quality_status"] == "VALIDATED_FOR_CLAIM"
        and observation["evidence_basis"]["kind"] in DIRECT_EVIDENCE_BASIS
        and observation["orientation_and_camera_basis"] != "UNKNOWN"
    )
    minimum_per_parent = int(gate["minimum_frames_per_parent"])
    eligible_parents = sorted(
        parent_id
        for parent_id, count in observation["parent_frame_counts"].items()
        if int(count) >= minimum_per_parent
    )
    orientation_minimum = int(gate["minimum_frames_per_orientation"])
    count_pass = (
        int(observation["total_frames"]) >= int(gate["minimum_total_frames"])
        and len(eligible_parents) >= int(gate["minimum_parents"])
        and int(observation["orientation_frame_counts"]["portrait"]) >= orientation_minimum
        and int(observation["orientation_frame_counts"]["landscape"]) >= orientation_minimum
    )
    return {
        "direct_evidence": direct,
        "count_pass": count_pass,
        "screening_match": direct and count_pass,
        "eligible_parents": eligible_parents,
        "upgradeable": observation["upgrade_path"] in UPGRADEABLE_PATHS,
        "provenance_kind": observation["provenance_kind"],
        "quality_status": observation["quality_status"],
        "evidence_basis_kind": observation["evidence_basis"]["kind"],
    }


def evaluate_source_manifest(source: dict[str, Any], gap_contract: dict[str, Any]) -> dict[str, Any]:
    hard_reasons = validate_source_manifest(source)
    firewall = gap_contract["protected_identity_firewall"]
    roster = source["identity_roster"]
    if roster["forbidden_roster_contract_sha256"] != firewall["forbidden_roster_contract_sha256"]:
        hard_reasons.append("FORBIDDEN_ROSTER_CONTRACT_DRIFT")
    forbidden_role_intersection = sorted(set(roster["history_roles"]) & set(firewall["forbidden_history_roles"]))
    if forbidden_role_intersection:
        hard_reasons.append("CONSUMED_OR_PROTECTED_HISTORY_ROLE")
    forbidden_parent_intersection: list[str] = []
    forbidden_session_intersection: list[str] = []
    if source["source"]["source_family"] == firewall["source_family"]:
        raw_parent_ids = {value.split(":", 1)[-1] for value in roster["parent_ids"]}
        raw_session_ids = {value.split(":", 1)[-1] for value in roster["session_ids"]}
        forbidden_parent_intersection = sorted(raw_parent_ids & set(firewall["forbidden_parent_ids"]))
        forbidden_session_intersection = sorted(raw_session_ids & set(firewall["forbidden_session_ids"]))
        if forbidden_parent_intersection:
            hard_reasons.append("PROTECTED_PARENT_IDENTITY_INTERSECTION")
        if forbidden_session_intersection:
            hard_reasons.append("PROTECTED_SESSION_IDENTITY_INTERSECTION")
    allowed_capabilities = {
        capability_name
        for gap in gap_contract["gaps"].values()
        for capability_name in gap["capabilities"]
    }
    unknown_capabilities = sorted(set(source["capabilities"]) - allowed_capabilities)
    require(not unknown_capabilities, f"capability outside frozen gap allowlist: {unknown_capabilities}")
    gap_results: dict[str, Any] = {}
    any_relevant = False
    any_partial = False
    any_screening_match = False

    for gap_id, gap in gap_contract["gaps"].items():
        capability_results: dict[str, Any] = {}
        eligible_sets: list[set[str]] = []
        all_pass = True
        gap_relevant = False
        for capability_name, gate in gap["capabilities"].items():
            observation = source["capabilities"].get(capability_name)
            if observation is None:
                capability_results[capability_name] = {
                    "screening_match": False,
                    "direct_evidence": False,
                    "count_pass": False,
                    "eligible_parents": [],
                    "upgradeable": False,
                    "missing": True,
                }
                eligible_sets.append(set())
                all_pass = False
                continue
            gap_relevant = True
            result = _evaluate_capability(observation, gate)
            capability_results[capability_name] = result
            eligible_sets.append(set(result["eligible_parents"]))
            all_pass = all_pass and bool(result["screening_match"])

        joint = set.intersection(*eligible_sets) if eligible_sets else set()
        joint_gate = gap["joint_parent_gate"]
        joint_pass = len(joint) >= int(joint_gate["minimum_joint_parents"])
        partition_feasible = len(joint) >= (
            int(joint_gate["minimum_fit_parents"]) + int(joint_gate["minimum_eval_parents"])
        )
        freshness_match = roster["claimed_freshness"] in set(gap["allowed_freshness"])
        screening_match = all_pass and joint_pass and partition_feasible and freshness_match and not hard_reasons
        permitted_upgrade_paths = set(gap["permitted_upgrade_paths"])
        upgradeable = gap_relevant and any(
            source["capabilities"].get(capability_name, {}).get("upgrade_path") in permitted_upgrade_paths
            for capability_name in gap["capabilities"]
        )
        partial = gap_relevant and not screening_match
        gap_results[gap_id] = {
            "screening_match": screening_match,
            "partial": partial,
            "upgradeable": upgradeable,
            "joint_eligible_parents": sorted(joint),
            "joint_parent_gate_pass": joint_pass and partition_feasible,
            "freshness_match": freshness_match,
            "capability_results": capability_results,
            "claim_ceiling": gap["claim_ceiling"],
        }
        any_relevant = any_relevant or gap_relevant
        any_partial = any_partial or partial or upgradeable
        any_screening_match = any_screening_match or screening_match

    if hard_reasons:
        decision = "REJECT"
    elif any_screening_match:
        decision = "PRESCREEN_ADMIT"
    elif any_relevant or any_partial:
        decision = "PARTIAL"
    else:
        decision = "REJECT"
        hard_reasons.append("NO_RELEVANT_CAPABILITY_SIGNAL")

    return {
        "schema": "blindassist.assistive_geometry_due.r0_source_prescreen_result.v1",
        "manifest_id": source["manifest_id"],
        "source_id": source["source"]["source_id"],
        "decision": decision,
        "hard_rejection_reasons": hard_reasons,
        "forbidden_identity_intersections": {
            "parent_ids": forbidden_parent_intersection,
            "session_ids": forbidden_session_intersection,
            "history_roles": forbidden_role_intersection,
        },
        "gap_results": gap_results,
        "source_data_support_established": False,
        "supported_for_protocol_lock": False,
        "next_action": (
            "LOCK_SOURCE_SPECIFIC_INTEGRITY_AND_PAYLOAD_AUDIT"
            if decision == "PRESCREEN_ADMIT"
            else "RESOLVE_GOVERNANCE_OR_CAPABILITY_GAPS_BEFORE_PAYLOAD_WORK"
        ),
        "execution_authorized": False,
        "claim_ceiling": gap_contract["claim_ceiling"],
    }


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    require(protocol["schema"] == "blindassist.assistive_geometry_due.r0_protocol.v1", "protocol schema drift")
    require(protocol["status"] == "GAP_DRIVEN_SOURCE_ADMISSION_PROTOCOL_LOCKED", "protocol status drift")
    require(protocol["research_mode"] == "REVERSIBLE_EXPLORATION", "research mode drift")
    require(protocol["research_style"] == "WILD_LAB", "research style drift")
    require(protocol["execution_authority"] == {
        "source_manifest_lock": False,
        "real_source_prescreen": False,
        "payload_download_or_open": False,
        "teacher_or_pseudo_label_generation": False,
        "data_materialization": False,
        "model_or_training": False,
        "development_or_confirmation": False,
        "android_or_default_app": False,
    }, "execution authority drift")

    for name in (
        "gap_contract",
        "source_manifest_schema",
        "dca_protocol",
        "dca_requirements",
        "dca_result",
        "f1_protocol",
        "f1_result",
    ):
        receipt = protocol[name]
        path = REPO_ROOT / receipt["path"]
        require(path.is_file(), f"protocol input missing: {name}")
        require(sha256_file(path) == receipt["sha256"], f"protocol input SHA drift: {name}")

    gap_contract = load_json(REPO_ROOT / protocol["gap_contract"]["path"])
    source_schema = load_json(REPO_ROOT / protocol["source_manifest_schema"]["path"])
    dca_requirements = load_json(REPO_ROOT / protocol["dca_requirements"]["path"])
    dca_protocol = load_json(REPO_ROOT / protocol["dca_protocol"]["path"])
    f1_protocol = load_json(REPO_ROOT / protocol["f1_protocol"]["path"])
    require(gap_contract["status"] == "FROZEN_PRE_SOURCE_OUTCOME", "gap contract status drift")
    require(gap_contract["evidence_policy"]["source_native_is_automatic_ground_truth"] is False, "source-native upgraded to GT")
    require(gap_contract["evidence_policy"]["unknown_is_negative"] is False, "UNKNOWN policy drift")
    require(gap_contract["source_dca"]["protocol_sha256"] == protocol["dca_protocol"]["sha256"], "DCA protocol binding drift")
    require(gap_contract["source_dca"]["requirements_sha256"] == protocol["dca_requirements"]["sha256"], "DCA requirements binding drift")
    require(gap_contract["source_dca"]["result_sha256"] == protocol["dca_result"]["sha256"], "DCA result binding drift")
    require(gap_contract["source_f1"]["protocol_sha256"] == protocol["f1_protocol"]["sha256"], "F1 protocol binding drift")
    require(gap_contract["source_f1"]["result_sha256"] == protocol["f1_result"]["sha256"], "F1 result binding drift")
    validate_source_schema_contract(source_schema)
    validate_gap_semantics(gap_contract, dca_requirements, dca_protocol, f1_protocol)

    expected_implementation = {
        "scripts/research/assistive_geometry_data_upgrade/validate_due_r0.py",
        "scripts/research/assistive_geometry_data_upgrade/test_validate_due_r0.py",
    }
    require(set(protocol["implementation"]) == expected_implementation, "implementation path set drift")
    for logical_path, expected_hash in protocol["implementation"].items():
        require(sha256_file(REPO_ROOT / logical_path) == expected_hash, f"implementation SHA drift: {logical_path}")
    return gap_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(REPO_ROOT / PROTOCOL_RELATIVE))
    parser.add_argument("--source-manifest")
    parser.add_argument("--output")
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    require(protocol_path == (REPO_ROOT / PROTOCOL_RELATIVE).resolve(), "custom protocol is not authorized")
    protocol = load_json(protocol_path)
    gap_contract = validate_protocol(protocol)
    if not args.source_manifest:
        print(json.dumps({"status": "VALID", "protocol_id": protocol["protocol_id"]}, sort_keys=True))
        return 0

    result = evaluate_source_manifest(load_json(Path(args.source_manifest)), gap_contract)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
