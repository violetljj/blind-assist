"""Fail-closed checks for a P0-S0 visual candidate generator admission record."""

from __future__ import annotations

from typing import Any


ALLOWED_VERDICTS = {
    "P0_S0_VISUAL_CANDIDATE_GENERATOR_ADMITTED",
    "P0_S0_VISUAL_CANDIDATE_GENERATOR_ADMITTED_WITH_CONSTRAINTS",
    "P0_S0_VISUAL_CANDIDATE_GENERATOR_NOT_ADMITTED",
}
FORBIDDEN_TRUTH_FIELDS = {
    "entrance_truth",
    "entrance_of_truth",
    "target_building_truth",
    "silver_quality_class",
    "evaluator_truth",
}
ADMISSION_BLOCKERS = {
    "UNAVAILABLE",
    "UNKNOWN",
    "INSUFFICIENT_FOR_ADMISSION",
}


def validate_admission(record: dict[str, Any]) -> list[str]:
    """Return stable error codes; an empty list means internally valid."""
    errors: list[str] = []
    verdict = record.get("verdict")
    if verdict not in ALLOWED_VERDICTS:
        errors.append("INVALID_VERDICT")
    if record.get("authority") != "VISUAL_PROPOSAL_ONLY":
        errors.append("INVALID_AUTHORITY")

    schema = record.get("required_candidate_provenance_schema_for_any_successor", {})
    forbidden = set(schema.get("forbidden_truth_fields", []))
    if forbidden != FORBIDDEN_TRUTH_FIELDS:
        errors.append("TRUTH_FIREWALL_INCOMPLETE")

    lineage = record.get("lineage_independence", {})
    denied_authorities = (
        "candidate_confidence_is_truth",
        "candidate_can_set_map_or_geometry_truth",
        "candidate_can_set_multiview_truth",
        "candidate_can_promote_silver_a_primary",
        "candidate_can_be_final_evaluator_truth",
    )
    if any(lineage.get(key) is not False for key in denied_authorities):
        errors.append("PROPOSAL_AUTHORITY_ESCALATION")

    audited = record.get("audited_generator", {})
    behavior = record.get("audited_upstream_behavior", {})
    admission_requested = verdict in {
        "P0_S0_VISUAL_CANDIDATE_GENERATOR_ADMITTED",
        "P0_S0_VISUAL_CANDIDATE_GENERATOR_ADMITTED_WITH_CONSTRAINTS",
    }
    if admission_requested:
        required_identity = (
            "checkpoint_repository_revision",
            "checkpoint_sha256_from_lfs_oid",
            "declared_checkpoint_repository_license",
        )
        if any(not audited.get(key) for key in required_identity):
            errors.append("CHECKPOINT_IDENTITY_OR_LICENSE_MISSING")
        if audited.get("training_data_provenance") in ADMISSION_BLOCKERS:
            errors.append("TRAINING_PROVENANCE_NOT_ADMISSIBLE")
        if behavior.get("unbound_replay_parameters"):
            errors.append("REPLAY_CONFIG_INCOMPLETE")
        if record.get("blocking_reasons"):
            errors.append("UNRESOLVED_BLOCKERS")
        if record.get("execution_authorized") is not True:
            errors.append("ADMISSION_NOT_EXECUTION_AUTHORIZED")
    else:
        if record.get("execution_authorized") is not False:
            errors.append("REJECTED_GENERATOR_EXECUTION_ENABLED")
        if not record.get("blocking_reasons"):
            errors.append("REJECTION_WITHOUT_REASON")

    return errors
