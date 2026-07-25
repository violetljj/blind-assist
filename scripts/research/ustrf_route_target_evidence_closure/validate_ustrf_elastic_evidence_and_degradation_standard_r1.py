#!/usr/bin/env python3
"""Validate the reusable USTRF elastic evidence/degradation contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "blindassist_ustrf_elastic_evidence_and_degradation_standard_r1"
VALIDATION_SCHEMA = f"{SCHEMA}_validation"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    classes = config["defect_classes"]
    anti = config["anti_gaming"]
    required_fields = {
        "claim_id",
        "required_roles",
        "optional_roles",
        "unit_of_analysis",
        "defect_class",
        "missingness_mechanism",
        "union_denominator",
        "eligible_denominator",
        "abstained_denominator",
        "coverage",
        "cluster_distribution",
        "propagation_evidence",
        "bias_risk",
        "disposition",
        "maximum_claim_scope",
        "authority_granted",
        "authority_closed",
    }
    checks = {
        "identity": config["schema"] == SCHEMA and config["version"] == "R1" and config["status"] == "current",
        "granularity_order": config["impact_granularity"]
        == ["field", "observation_or_object", "frame", "window", "sequence", "source", "program"],
        "normal_missingness_is_local": (
            classes["normal_observation_missingness"]["default_action"]
            == "abstain_affected_unit_and_report_union_coverage"
            and classes["normal_observation_missingness"]["global_failure_allowed_by_default"] is False
        ),
        "structural_escalation_needs_evidence": classes["structural_integrity"][
            "global_escalation_requires_propagation_evidence"
        ]
        is True,
        "support_shortfall_not_algorithm_failure": classes["support_shortfall"][
            "algorithm_failure_allowed"
        ]
        is False,
        "performance_not_source_rejection": classes["performance_failure"][
            "source_unusable_inference_allowed"
        ]
        is False,
        "authority_gap_preserves_lower_evidence": classes["authority_gap"][
            "erase_lower_level_evidence_allowed"
        ]
        is False,
        "coverage_bands_descriptive": (
            config["coverage_bands"]["HIGH_COVERAGE"]["minimum"] == 0.95
            and config["coverage_bands"]["MODERATE_COVERAGE"]["minimum"] == 0.8
            and config["coverage_bands"]["bands_are_automatic_pass_fail_gates"] is False
        ),
        "mandatory_decision_fields_complete": required_fields.issubset(config["mandatory_decision_fields"]),
        "anti_gaming_closed": (
            anti["historical_terminal_rewrite"] is False
            and anti["silent_missing_unit_drop"] is False
            and anti["union_denominator_reduction"] is False
            and anti["missing_as_zero_success_or_failure"] is False
            and anti["post_outcome_eligibility_rescue"] is False
            and anti["pooled_coverage_hides_zero_cluster"] is False
            and anti["relaxed_data_handling_grants_higher_authority"] is False
            and anti["validator_rebuilds_from_immutable_inputs"] is True
        ),
        "required_dispositions": {
            "ADMIT_COMPLETE",
            "ADMIT_WITH_ABSTENTION",
            "DIAGNOSTIC_ONLY",
            "NOT_EVALUABLE_FOR_CLAIM",
            "REJECT_METHOD_OR_CANDIDATE",
            "FAIL_CLOSED_CORRUPT_EVIDENCE",
            "AUTHORITY_HOLD",
        }.issubset(config["dispositions"]),
        "three_axis_reporting": (
            config["report_axes"]["artifact_integrity"] == ["VALID", "INVALID"]
            and "AVAILABLE_WITH_DEGRADATION" in config["report_axes"]["claim_status"]
            and config["report_axes"]["authority_ceiling"][0] == "DIAGNOSTIC"
            and "VALID_WITH_PARTIAL_OR_DEGRADED_CLAIMS" in config["overall_statuses"]
        ),
        "denominator_conservation": (
            config["denominator_invariant"]
            == "expected_denominator_equals_eligible_plus_abstained_plus_invalid"
            and config["source_native_expected_denominator_required"] is True
            and config["interpolated_or_imputed_counts_as_direct_observation"] is False
        ),
        "localized_unknown_missingness": config["unknown_missingness_default"].startswith(
            "abstain_affected_unit"
        ),
        "defect_records_scoped": {
            "defect_id",
            "defect_class",
            "scope_type",
            "scope_ids",
            "affected_modalities",
            "affected_claims",
            "reason_code",
            "localized",
            "denominator_impact",
            "evidence_refs",
        }.issubset(config["mandatory_defect_fields"]),
    }
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "VALID" if all(checks.values()) else "INVALID",
        "config_sha256": sha256_file(config_path),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = validate(config_path)
    output = repo / config["outputs"]["validation"]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(result)
    if output.exists() and output.read_bytes() != payload:
        raise RuntimeError(f"immutable_validation_drift:{output}")
    output.write_bytes(payload)
    print(json.dumps({"output": output.as_posix(), "status": result["status"]}, ensure_ascii=False))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
