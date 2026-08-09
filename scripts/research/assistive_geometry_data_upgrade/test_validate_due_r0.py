from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry_data_upgrade.validate_due_r0 import (
    PROTOCOL_RELATIVE,
    REPO_ROOT,
    ContractError,
    evaluate_source_manifest,
    load_json,
    validate_gap_semantics,
    validate_protocol,
    validate_source_schema_contract,
)


def _observation(
    parent_counts: dict[str, int],
    *,
    portrait: int,
    landscape: int,
    provenance: str = "SOURCE_NATIVE_SENSOR",
    quality: str = "VALIDATED_FOR_CLAIM",
    upgrade_path: str = "NONE",
) -> dict:
    return {
        "total_frames": sum(parent_counts.values()),
        "orientation_frame_counts": {"portrait": portrait, "landscape": landscape},
        "parent_frame_counts": parent_counts,
        "provenance_kind": provenance,
        "quality_status": quality,
        "upgrade_path": upgrade_path,
        "evidence_basis": {
            "kind": "TRACKED_PROJECT_MANIFEST",
            "receipt": "synthetic claim-bound source manifest",
            "receipt_sha256": "A" * 64,
            "claim_id": "SYNTHETIC_CAPABILITY_CLAIM",
            "claim_definition": "Synthetic mechanics-only capability claim",
            "count_basis": "Explicit synthetic frame by parent and orientation enumeration",
            "source_object_sha256": "C" * 64,
            "source_field_mapping": ["synthetic_field -> capability"],
            "alignment_registration_units_coordinate_receipt_sha256": "D" * 64,
            "source_specific_verifier_sha256": "B" * 64,
        },
        "parent_identity_namespace": "synthetic",
        "orientation_and_camera_basis": "DISPLAY_UPRIGHT_WITH_BOUND_CAMERA_K",
        "derivation_receipt": None,
        "teacher_receipts": [],
        "unknown_treated_as_negative": False,
    }


def _source(capabilities: dict) -> dict:
    parent_ids = sorted(
        {
            parent_id
            for observation in capabilities.values()
            for parent_id in observation["parent_frame_counts"]
        }
    )
    return {
        "schema": "blindassist.assistive_geometry_due.r0_source_manifest.v1",
        "manifest_id": "SYNTHETIC_SOURCE_MANIFEST_R0",
        "status": "SOURCE_DISCOVERY_METADATA_ONLY",
        "source": {
            "source_id": "SYNTHETIC_SOURCE",
            "source_family": "SYNTHETIC_TEST_ONLY",
            "source_version": "r0",
            "identity_basis": "synthetic unit-test identity",
            "payload_presence": "ABSENT",
        },
        "license": {
            "internal_research_status": "VERIFIED_FOR_INTERNAL_RESEARCH",
            "redistribution_status": "NOT_AUTHORIZED",
            "receipt": "synthetic test fixture",
        },
        "ethics_privacy_access": {
            "human_subject_presence": "NONE",
            "privacy_review_status": "VERIFIED_FOR_INTERNAL_RESEARCH",
            "access_terms_status": "VERIFIED",
            "sensitive_content_handling": "synthetic fixture contains no people",
            "receipt": "synthetic ethics and access fixture",
            "receipt_sha256": "E" * 64,
        },
        "ancestry": {
            "status": "VERIFIED",
            "root_identity": "SYNTHETIC_ROOT",
            "derivative_chain": [],
        },
        "independence": {
            "status": "VERIFIED",
            "parent_identity_type": "synthetic_parent",
            "independence_group_basis": "explicit fixture parent ids",
        },
        "identity_roster": {
            "parent_ids": parent_ids,
            "session_ids": ["synthetic:session-0"],
            "ancestry_group_ids": ["synthetic:ancestry-0"],
            "history_roles": ["SYNTHETIC"],
            "claimed_freshness": "FRESH_SOURCE_DISCOVERY",
            "forbidden_roster_contract_sha256": "3E5DB5410FBF7AA9B64DBFC57C5E4557E2094222E131B4513024440099C4F860",
        },
        "access_receipt": {
            "payload_opened": False,
            "rgb_visual_access": False,
            "geometry_payload_access": False,
            "model_outcome_access": False,
            "confirmation_outcome_access": False,
            "selection_or_tuning_influence": False,
        },
        "capabilities": capabilities,
    }


class DueR0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_json(REPO_ROOT / PROTOCOL_RELATIVE)
        cls.gaps = load_json(REPO_ROOT / cls.protocol["gap_contract"]["path"])

    def test_frozen_protocol_validates(self) -> None:
        validate_protocol(self.protocol)

    def test_claim_bound_metadata_can_only_admit_for_integrity_audit(self) -> None:
        parents = {f"synthetic:parent-{index}": 2 for index in range(4)}
        source = _source(
            {
                "finite_clearance_event": _observation(parents, portrait=4, landscape=4),
                "right_censor": _observation(parents, portrait=4, landscape=4),
            }
        )
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("PRESCREEN_ADMIT", result["decision"])
        self.assertTrue(result["gap_results"]["CAPABILITY_GAP_RIGHT_CENSOR"]["screening_match"])
        self.assertFalse(result["source_data_support_established"])
        self.assertFalse(result["supported_for_protocol_lock"])
        self.assertEqual("LOCK_SOURCE_SPECIFIC_INTEGRITY_AND_PAYLOAD_AUDIT", result["next_action"])

    def test_source_native_without_claim_validation_is_not_direct_truth(self) -> None:
        parents = {f"synthetic:parent-{index}": 2 for index in range(4)}
        source = _source(
            {
                "finite_clearance_event": _observation(parents, portrait=4, landscape=4),
                "right_censor": _observation(
                    parents,
                    portrait=4,
                    landscape=4,
                    quality="CHARACTERIZED_NOT_VALIDATED",
                    upgrade_path="DETERMINISTIC_DERIVATION",
                ),
            }
        )
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("PARTIAL", result["decision"])
        self.assertFalse(result["gap_results"]["CAPABILITY_GAP_RIGHT_CENSOR"]["screening_match"])

    def test_teacher_consensus_is_candidate_only_for_frozen_truth_gaps(self) -> None:
        parents = {f"synthetic:parent-{index}": 2 for index in range(4)}
        teacher = _observation(
            parents,
            portrait=4,
            landscape=4,
            provenance="TEACHER_CONSENSUS",
            quality="CHARACTERIZED_NOT_VALIDATED",
            upgrade_path="MULTI_TEACHER_CONSENSUS",
        )
        teacher["teacher_receipts"] = ["teacher-a", "teacher-b"]
        source = _source({"right_censor": teacher})
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("PARTIAL", result["decision"])
        self.assertFalse(result["gap_results"]["CAPABILITY_GAP_RIGHT_CENSOR"]["upgradeable"])

    def test_source_native_without_claim_bound_receipt_cannot_screen_match(self) -> None:
        parents = {f"synthetic:parent-{index}": 2 for index in range(4)}
        source = _source(
            {
                "finite_clearance_event": _observation(parents, portrait=4, landscape=4),
                "right_censor": _observation(parents, portrait=4, landscape=4),
            }
        )
        source["capabilities"]["right_censor"]["evidence_basis"]["kind"] = "UNKNOWN"
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("PARTIAL", result["decision"])
        self.assertFalse(result["gap_results"]["CAPABILITY_GAP_RIGHT_CENSOR"]["screening_match"])

    def test_unknown_license_hard_rejects(self) -> None:
        source = _source(
            {"explicit_timestamp_materialized": _observation({"synthetic:parent-a": 1}, portrait=1, landscape=0)}
        )
        source["license"]["internal_research_status"] = "UNKNOWN"
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("REJECT", result["decision"])
        self.assertIn("LICENSE_NOT_VERIFIED_FOR_INTERNAL_RESEARCH", result["hard_rejection_reasons"])

    def test_unresolved_privacy_review_hard_rejects(self) -> None:
        source = _source(
            {"explicit_timestamp_materialized": _observation({"synthetic:parent-a": 1}, portrait=1, landscape=0)}
        )
        source["ethics_privacy_access"]["privacy_review_status"] = "REQUIRED_NOT_COMPLETE"
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("REJECT", result["decision"])
        self.assertIn("PRIVACY_REVIEW_NOT_VERIFIED", result["hard_rejection_reasons"])

    def test_consumed_development_history_role_hard_rejects(self) -> None:
        source = _source(
            {"explicit_timestamp_materialized": _observation({"synthetic:parent-a": 1}, portrait=1, landscape=0)}
        )
        source["identity_roster"]["history_roles"] = ["PROJECT_CONSUMED_DEVELOPMENT"]
        source["identity_roster"]["claimed_freshness"] = "CONSUMED_OR_PROTECTED"
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("REJECT", result["decision"])
        self.assertIn("CONSUMED_OR_PROTECTED_HISTORY_ROLE", result["hard_rejection_reasons"])

    def test_unknown_cannot_be_encoded_as_negative(self) -> None:
        source = _source(
            {"explicit_timestamp_materialized": _observation({"synthetic:parent-a": 1}, portrait=1, landscape=0)}
        )
        source["capabilities"]["explicit_timestamp_materialized"]["unknown_treated_as_negative"] = True
        with self.assertRaisesRegex(ContractError, "UNKNOWN used as negative"):
            evaluate_source_manifest(source, self.gaps)

    def test_joint_gate_rejects_disjoint_capability_parents(self) -> None:
        source = _source(
            {
                "finite_clearance_event": _observation(
                    {f"synthetic:event-{index}": 1 for index in range(4)}, portrait=2, landscape=2
                ),
                "right_censor": _observation(
                    {f"synthetic:censor-{index}": 1 for index in range(4)}, portrait=2, landscape=2
                ),
            }
        )
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("PARTIAL", result["decision"])
        self.assertEqual([], result["gap_results"]["CAPABILITY_GAP_RIGHT_CENSOR"]["joint_eligible_parents"])

    def test_capability_outside_frozen_gap_allowlist_is_rejected(self) -> None:
        source = _source({"invented_capability": _observation({"synthetic:parent-a": 1}, portrait=1, landscape=0)})
        with self.assertRaisesRegex(ContractError, "outside frozen gap allowlist"):
            evaluate_source_manifest(source, self.gaps)

    def test_declared_schema_and_checker_required_fields_cannot_drift(self) -> None:
        schema = load_json(REPO_ROOT / self.protocol["source_manifest_schema"]["path"])
        schema["properties"]["capabilities"]["additionalProperties"]["required"].remove("teacher_receipts")
        with self.assertRaisesRegex(ContractError, "declared capability required fields drift"):
            validate_source_schema_contract(schema)

    def test_dca_threshold_mutation_is_rejected(self) -> None:
        gaps = copy.deepcopy(self.gaps)
        gaps["gaps"]["CAPABILITY_GAP_CORRIDOR"]["capabilities"]["full_2_5d_grid"]["minimum_total_frames"] = 639
        dca_requirements = load_json(REPO_ROOT / self.protocol["dca_requirements"]["path"])
        dca_protocol = load_json(REPO_ROOT / self.protocol["dca_protocol"]["path"])
        f1_protocol = load_json(REPO_ROOT / self.protocol["f1_protocol"]["path"])
        with self.assertRaisesRegex(ContractError, "DCA capability threshold drift"):
            validate_gap_semantics(gaps, dca_requirements, dca_protocol, f1_protocol)

    def test_protected_arkitscenes_parent_intersection_rejects(self) -> None:
        source = _source(
            {"explicit_timestamp_materialized": _observation({"arkit:41127065": 1}, portrait=1, landscape=0)}
        )
        source["source"]["source_family"] = "ARKITSCENES"
        source["capabilities"]["explicit_timestamp_materialized"]["parent_identity_namespace"] = "arkit"
        result = evaluate_source_manifest(source, self.gaps)
        self.assertEqual("REJECT", result["decision"])
        self.assertIn("PROTECTED_PARENT_IDENTITY_INTERSECTION", result["hard_rejection_reasons"])

    def test_existing_train_cannot_match_fresh_fci_gate(self) -> None:
        parents_32 = {f"synthetic:parent-{index}": 32 for index in range(8)}
        parents_8 = {f"synthetic:parent-{index}": 8 for index in range(8)}
        source = _source(
            {
                "r2_complete_factor_schema_truth": _observation(parents_32, portrait=128, landscape=128),
                "fci_factor_truth_bundle": _observation(parents_32, portrait=128, landscape=128),
                "fci_truth_clear_bundle": _observation(parents_8, portrait=32, landscape=32),
                "fci_truth_occupied_bundle": _observation(parents_8, portrait=32, landscape=32),
            }
        )
        source["identity_roster"]["history_roles"] = ["TRAIN"]
        source["identity_roster"]["claimed_freshness"] = "DISCLOSED_EXISTING_TRAIN"
        result = evaluate_source_manifest(source, self.gaps)
        fci = result["gap_results"]["CAPABILITY_GAP_R2_FACTOR_INTERVENTION"]
        self.assertEqual("PARTIAL", result["decision"])
        self.assertFalse(fci["freshness_match"])
        self.assertFalse(fci["screening_match"])


if __name__ == "__main__":
    unittest.main()
