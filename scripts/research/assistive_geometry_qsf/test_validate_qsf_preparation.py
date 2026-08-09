from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import Mock

from scripts.research.assistive_geometry_qsf.validate_qsf_preparation import (
    PROTOCOL_RELATIVE,
    ROUTE_ID,
    ValidationError,
    _is_b1_protected_artifact_path,
    _is_linklike,
    validate_planned_outputs,
    validate_protocol,
    validate_resource_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PINNED_PARENT = "f3b6261177c4ecb9e1b2122db7a19c9a62a2a0ce"


def _protocol() -> dict:
    return json.loads((REPO_ROOT / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))


def _file_sha(relative: str) -> str:
    return hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()


def _resource(**overrides: object) -> dict:
    value = {
        "resource_id": "B0_TASK_CONTRACT",
        "kind": "PROTOCOL",
        "logical_path": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.json",
        "producer_route": "BLINDASSIST_ASSISTIVE_GEOMETRY",
        "access": "READ_ONLY",
        "immutable": True,
        "identity": {"basis": "GIT_COMMIT", "value": PINNED_PARENT},
        "data_role": "NOT_APPLICABLE",
        "provenance": "tracked B0 task contract",
        "license_scope": "NOT_APPLICABLE",
        "outcome_access": "NONE",
        "selection_influence": "NONE",
        "claim_use": "SCHEMA_ONLY",
    }
    value.update(overrides)
    return value


def _manifest(*resources: dict) -> dict:
    return {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "manifest_id": "qsf-h1-preparation-test",
        "resources": list(resources),
    }


class QsfPreparationValidationTest(unittest.TestCase):
    def test_symlink_and_windows_junction_are_linklike(self) -> None:
        symlink = Mock()
        symlink.is_symlink.return_value = True
        symlink.is_junction.return_value = False
        self.assertTrue(_is_linklike(symlink))

        junction = Mock()
        junction.is_symlink.return_value = False
        junction.is_junction.return_value = True
        self.assertTrue(_is_linklike(junction))

        regular = Mock()
        regular.is_symlink.return_value = False
        regular.is_junction.return_value = False
        self.assertFalse(_is_linklike(regular))

    def test_tracked_protocol_and_owned_output_are_valid(self) -> None:
        protocol = _protocol()
        report = validate_protocol(protocol, REPO_ROOT)
        self.assertEqual(3, len(report["owned_output_roots"]))
        self.assertEqual(
            ["artifacts.local/evidence/assistive-geometry-qsf/run-001"],
            validate_planned_outputs(
                ["artifacts.local/evidence/assistive-geometry-qsf/run-001"], protocol
            ),
        )

    def test_read_only_versioned_directory_manifest_is_shareable(self) -> None:
        manifest_path = (
            "scripts/research/assistive_geometry_qsf/fixtures/"
            "directory_identity/directory.manifest.json"
        )
        resource = _resource(
            resource_id="QSF_TEST_TOOL_DIRECTORY",
            kind="TEST_TOOL",
            logical_path=(
                "scripts/research/assistive_geometry_qsf/fixtures/directory_identity"
            ),
            producer_route=ROUTE_ID,
            identity={
                "basis": "MANIFEST_SHA256",
                "manifest_path": manifest_path,
                "value": _file_sha(manifest_path),
            },
            data_role="NOT_APPLICABLE",
            provenance="tracked QSF preparation test tool directory",
            license_scope="PROJECT_INTERNAL",
            claim_use="TEST_ONLY",
        )
        report = validate_resource_manifest(_manifest(resource), _protocol(), REPO_ROOT)
        self.assertEqual("READ_ONLY", report["access"])

    def test_directory_manifest_must_bind_every_member(self) -> None:
        incomplete_manifest = (
            "scripts/research/assistive_geometry_qsf/shared_resource_manifest.template.json"
        )
        resource = _resource(
            resource_id="INCOMPLETE_DIRECTORY_IDENTITY",
            kind="TEST_TOOL",
            logical_path="scripts/research/assistive_geometry_qsf",
            producer_route=ROUTE_ID,
            identity={
                "basis": "MANIFEST_SHA256",
                "manifest_path": incomplete_manifest,
                "value": _file_sha(incomplete_manifest),
            },
            provenance="deliberately incomplete directory identity",
            license_scope="PROJECT_INTERNAL",
            claim_use="TEST_ONLY",
        )
        with self.assertRaisesRegex(ValidationError, "complete schema v1"):
            validate_resource_manifest(_manifest(resource), _protocol(), REPO_ROOT)

    def test_active_checkpoint_and_foreign_write_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "not shareable"):
            validate_resource_manifest(
                _manifest(_resource(kind="ACTIVE_CHECKPOINT")), _protocol(), REPO_ROOT
            )
        with self.assertRaisesRegex(ValidationError, "immutable and READ_ONLY"):
            validate_resource_manifest(
                _manifest(_resource(access="READ_WRITE")), _protocol(), REPO_ROOT
            )

    def test_development_confirmation_and_owned_output_reuse_are_rejected(self) -> None:
        for role in ("DEVELOPMENT_SELECTION", "CONFIRMATION", "SEALED_UNSEEN"):
            with self.subTest(role=role), self.assertRaisesRegex(ValidationError, "data role"):
                validate_resource_manifest(
                    _manifest(_resource(data_role=role)), _protocol(), REPO_ROOT
                )
        with self.assertRaisesRegex(ValidationError, "owned output"):
            validate_resource_manifest(
                _manifest(
                    _resource(
                        logical_path="Artifacts.local/Evidence/Assistive-Geometry-QSF/run-001"
                    )
                ),
                _protocol(),
                REPO_ROOT,
            )

    def test_b1_consumed_development_protocol_cannot_use_generic_allowed_fields(self) -> None:
        logical = (
            "docs/research/assistive-geometry/"
            "BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_PROTOCOL_2026-08-09.json"
        )
        disguised = _resource(
            resource_id="B1_DEVELOPMENT_DISGUISED",
            kind="TRAIN_DIAGNOSTIC",
            logical_path=logical,
            identity={"basis": "SHA256", "value": _file_sha(logical)},
            data_role="PROJECT_CONSUMED_DEVELOPMENT",
            outcome_access="OUTPUT_INSPECTED_DIAGNOSTIC_ONLY",
            claim_use="DIAGNOSTIC_ONLY",
        )
        with self.assertRaisesRegex(ValidationError, "Development/Confirmation"):
            validate_resource_manifest(_manifest(disguised), _protocol(), REPO_ROOT)

    def test_b1_development_protocol_is_schema_only(self) -> None:
        logical = (
            "docs/research/assistive-geometry/"
            "BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_PROTOCOL_2026-08-09.json"
        )
        schema = _resource(
            resource_id="B1_DEVELOPMENT_SCHEMA",
            kind="PROTOCOL",
            logical_path=logical,
            identity={"basis": "SHA256", "value": _file_sha(logical)},
            data_role="NOT_APPLICABLE",
        )
        report = validate_resource_manifest(_manifest(schema), _protocol(), REPO_ROOT)
        self.assertEqual(1, report["resource_count"])

        disguised = copy.deepcopy(schema)
        disguised["resource_id"] = "B1_DEVELOPMENT_PRODUCER_SPOOF"
        disguised["producer_route"] = "OTHER_ROUTE"
        disguised["kind"] = "TRAIN_DIAGNOSTIC"
        disguised["outcome_access"] = "OUTPUT_INSPECTED_DIAGNOSTIC_ONLY"
        disguised["claim_use"] = "DIAGNOSTIC_ONLY"
        with self.assertRaisesRegex(ValidationError, "schema-only"):
            validate_resource_manifest(_manifest(disguised), _protocol(), REPO_ROOT)

    def test_non_b1_consumed_development_diagnostic_keeps_generic_policy(self) -> None:
        logical = "scripts/research/assistive_geometry_qsf/fixtures/directory_identity/payload.txt"
        resource = _resource(
            resource_id="OTHER_ROUTE_DEVELOPMENT_DIAGNOSTIC",
            kind="TRAIN_DIAGNOSTIC",
            logical_path=logical,
            producer_route="OTHER_ROUTE",
            identity={"basis": "SHA256", "value": _file_sha(logical)},
            data_role="PROJECT_CONSUMED_DEVELOPMENT",
            outcome_access="OUTPUT_INSPECTED_DIAGNOSTIC_ONLY",
            claim_use="DIAGNOSTIC_ONLY",
        )
        report = validate_resource_manifest(_manifest(resource), _protocol(), REPO_ROOT)
        self.assertEqual(1, report["diagnostic_only_count"])

    def test_b1_protected_artifact_path_mutations(self) -> None:
        protected = (
            "artifacts.local/datasets/assistive-geometry-b1-development-selection-targets-r0",
            "artifacts.local/evidence/hftf/assistive-geometry-b1-a0-development-r0",
            "artifacts.local/models/assistive-geometry-b1-confirmation-r0",
        )
        for value in protected:
            with self.subTest(value=value):
                self.assertTrue(_is_b1_protected_artifact_path(PurePosixPath(value)))
        self.assertFalse(
            _is_b1_protected_artifact_path(
                PurePosixPath("artifacts.local/models/assistive-geometry-qsf/h1-r0")
            )
        )
        self.assertFalse(
            _is_b1_protected_artifact_path(
                PurePosixPath(
                    "docs/research/assistive-geometry/"
                    "BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FORMAL_TRAIN_EXECUTION_PROTOCOL.md"
                )
            )
        )

    def test_foreign_active_run_cannot_be_disguised_as_a_protocol(self) -> None:
        with self.assertRaisesRegex(ValidationError, "active-run path"):
            validate_resource_manifest(
                _manifest(
                    _resource(
                        kind="PROTOCOL",
                        logical_path=(
                            "artifacts.local/evidence/hftf/"
                            "assistive-geometry-b1-a0-formal-train-20260809-r2/seed-29/progress.json"
                        ),
                    )
                ),
                _protocol(),
                REPO_ROOT,
            )

    def test_inspected_output_is_diagnostic_only(self) -> None:
        diagnostic = _resource(
            resource_id="SEED29_OOM_LESSON",
            kind="OPERATIONAL_LESSON",
            logical_path="docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FORMAL_TRAIN_EXECUTION_PROTOCOL_2026-08-09_ATTEMPT_02.md",
            outcome_access="OUTPUT_INSPECTED_DIAGNOSTIC_ONLY",
            claim_use="DIAGNOSTIC_ONLY",
            selection_influence="NONE",
        )
        report = validate_resource_manifest(_manifest(diagnostic), _protocol(), REPO_ROOT)
        self.assertEqual(1, report["diagnostic_only_count"])
        invalid = copy.deepcopy(diagnostic)
        invalid["claim_use"] = "SCHEMA_ONLY"
        with self.assertRaisesRegex(ValidationError, "DIAGNOSTIC_ONLY"):
            validate_resource_manifest(_manifest(invalid), _protocol(), REPO_ROOT)

    def test_selection_influence_and_claim_use_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValidationError, "selection influence"):
            validate_resource_manifest(
                _manifest(_resource(selection_influence="B1_THRESHOLD_SELECTION")),
                _protocol(),
                REPO_ROOT,
            )
        with self.assertRaisesRegex(ValidationError, "claim use"):
            validate_resource_manifest(
                _manifest(_resource(claim_use="CONFIRMATION")),
                _protocol(),
                REPO_ROOT,
            )

    def test_missing_resource_and_forged_sha_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "does not exist"):
            validate_resource_manifest(
                _manifest(
                    _resource(
                        logical_path="artifacts.local/does-not-exist/fake.pt",
                        identity={"basis": "SHA256", "value": "b" * 64},
                    )
                ),
                _protocol(),
                REPO_ROOT,
            )
        with self.assertRaisesRegex(ValidationError, "SHA256 mismatch"):
            validate_resource_manifest(
                _manifest(
                    _resource(identity={"basis": "SHA256", "value": "b" * 64})
                ),
                _protocol(),
                REPO_ROOT,
            )

    def test_custom_protocol_cannot_expand_share_or_write_authority(self) -> None:
        kind_downgrade = copy.deepcopy(_protocol())
        policy = kind_downgrade["shared_resource_policy"]
        policy["forbidden_kinds"].remove("ACTIVE_CHECKPOINT")
        policy["allowed_kinds"].append("ACTIVE_CHECKPOINT")
        with self.assertRaisesRegex(ValidationError, "kind policy"):
            validate_protocol(kind_downgrade, REPO_ROOT)

        role_downgrade = copy.deepcopy(_protocol())
        policy = role_downgrade["shared_resource_policy"]
        policy["forbidden_data_roles"].remove("CONFIRMATION")
        policy["allowed_data_roles"].append("CONFIRMATION")
        with self.assertRaisesRegex(ValidationError, "data-role policy"):
            validate_protocol(role_downgrade, REPO_ROOT)

        root_downgrade = copy.deepcopy(_protocol())
        root_downgrade["owned_output_roots"][0] = "artifacts.local/evidence/hftf"
        with self.assertRaisesRegex(ValidationError, "owned_output_roots drifted"):
            validate_protocol(root_downgrade, REPO_ROOT)

    def test_output_path_cannot_escape_qsf_namespace(self) -> None:
        with self.assertRaisesRegex(ValidationError, "outside QSF-owned"):
            validate_planned_outputs(
                ["artifacts.local/evidence/hftf/assistive-geometry-b1-a0-formal-train"],
                _protocol(),
            )
        with self.assertRaisesRegex(ValidationError, "logical namespace"):
            validate_planned_outputs(
                ["artifacts.local/evidence/assistive-geometry-qsf/../foreign"],
                _protocol(),
            )
        with self.assertRaisesRegex(ValidationError, "POSIX-style relative"):
            validate_planned_outputs(
                ["C:/foreign/assistive-geometry-qsf"],
                _protocol(),
            )


if __name__ == "__main__":
    unittest.main()
