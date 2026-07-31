from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .freeze_d0a1_pilot import (
    PilotFreezeError,
    ordered_asset_identity,
    write_bundle,
)
from .freeze_input_universe import canonical_bytes, sha256_file
from .validate_d0a1_pilot import PilotValidationError, validate
from .validate_d0a1_primary_review import (
    PrimaryReviewValidationError,
    validate_primary_review,
)


class D0A1PilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        (self.repo / "configs").mkdir()
        (self.repo / "docs").mkdir()
        (self.repo / "inputs").mkdir()
        protocol = {
            "protocol_id": "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A",
            "phase_plan": [
                {
                    "phase": "D0-A1",
                    "candidate_output_access": False,
                }
            ],
        }
        workflow = {
            "workflows": {
                "central_obstruction_agent_label_readiness_d0a_v1": {
                    "candidate_output_hidden_from_reviewers": True,
                }
            }
        }
        self._write_json(self.repo / "docs" / "protocol.json", protocol)
        self._write_json(self.repo / "configs" / "workflow.json", workflow)
        self._write_json(self.repo / "d0a0-manifest.json", {"status": "VALID"})
        (self.repo / "prompt.md").write_text("fixture prompt\n", encoding="utf-8")
        role_rows = []
        self.source_rows = []
        for source_number in range(3):
            source_id = f"calibration-{source_number}"
            source_root = self.repo / "inputs" / source_id
            source_root.mkdir()
            for frame_index in range(3):
                image = np.full(
                    (120, 160, 3),
                    30 + source_number * 50 + frame_index * 5,
                    dtype=np.uint8,
                )
                self.assertTrue(cv2.imwrite(str(source_root / f"{frame_index:03d}.png"), image))
            role_rows.append(
                {
                    "source_id": source_id,
                    "session_id": f"session-{source_number}",
                    "admission_disposition": "ADMIT_D0_A_CALIBRATION_ONLY",
                }
            )
            self.source_rows.append(
                {
                    "source_id": source_id,
                    "session_id": f"session-{source_number}",
                    "kind": "image_sequence",
                    "path": f"inputs/{source_id}",
                    "content_identity": ordered_asset_identity(self.repo, source_root),
                }
            )
        role_rows.append(
            {
                "source_id": "production-source",
                "session_id": "production-session",
                "admission_disposition": "ADMIT_D0_A_PRODUCTION_LABELING",
            }
        )
        self.role_path = self.repo / "roles.jsonl"
        self.role_path.write_bytes(b"".join(canonical_bytes(row) for row in role_rows))
        self.lock_path = self.repo / "lock.json"
        self.output_root = self.repo / "output"
        self.lock = self._lock()
        self._write_json(self.lock_path, self.lock)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_bytes(canonical_bytes(value))

    def _binding(self, relative: str) -> dict[str, str]:
        path = self.repo / relative
        return {"path": relative, "sha256": sha256_file(path)}

    def _lock(self) -> dict:
        return {
            "schema_version": "blindassist.central_obstruction_d0a1_lock.v1",
            "protocol_id": "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A",
            "phase": "D0-A1",
            "evidence_instance": "FIXTURE_D0_A1_R0",
            "status": "LOCKED_INPUTS_NOT_YET_REVIEWED",
            "output_root": "output",
            "candidate_output_access": False,
            "bindings": {
                "protocol": self._binding("docs/protocol.json"),
                "ai_workflow": self._binding("configs/workflow.json"),
                "d0a0_manifest": self._binding("d0a0-manifest.json"),
                "d0a0_reuse_role_ledger": self._binding("roles.jsonl"),
                "review_prompt": {"path": "prompt.md"},
            },
            "roi": {
                "name": "CENTRAL_IMAGE_ATTENTION_REGION",
                "normalized_xyxy": [0.25, 0.15, 0.75, 0.95],
                "minimum_native_roi_width_px": 32,
                "minimum_native_roi_height_px": 32,
                "semantics": "image only",
            },
            "observation_contract": {
                "labels": [
                    "VISIBLE_CENTRAL_OBSTRUCTION_PRESENT",
                    "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE",
                    "NOT_EVALUABLE",
                ],
                "quality_states": [
                    "STABLE",
                    "TURNING",
                    "BLURRED",
                    "DARK",
                    "OCCLUDED",
                    "OTHER_NOT_EVALUABLE",
                ],
            },
            "parent_event_rule": {
                "unit": "FROZEN_CLIP_PARENT_EVENT",
                "definition": "maximal run",
                "always_close_on": ["label_change"],
                "bridge_rule": "NO_BRIDGING",
            },
            "ambiguity_and_risk_strata": {
                "claim_critical": ["positive", "not evaluable"],
                "low_risk": ["stable negative"],
                "unresolved_rule": "NOT_EVALUABLE",
            },
            "audit_rule": {
                "claim_critical_review_passes": 2,
                "low_risk_primary_passes": 1,
                "low_risk_independent_audit_fraction": 0.2,
                "low_risk_minimum_audit_per_source": 1,
                "fresh_context_per_independent_pass": True,
                "other_review_hidden_before_submission": True,
                "material_disagreement_adjudicator_passes": 1,
                "human_queue_required": False,
            },
            "readiness_thresholds": {
                "minimum_calibration_sources": 3,
                "minimum_pilot_clips": 3,
                "minimum_pilot_observations": 9,
                "required_observed_labels": [
                    "VISIBLE_CENTRAL_OBSTRUCTION_PRESENT",
                    "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE",
                    "NOT_EVALUABLE",
                ],
                "minimum_observed_quality_states": 3,
                "overall_observation_label_agreement": 0.8,
                "claim_critical_observation_label_agreement": 0.8,
                "parent_event_match_rate": 0.75,
                "boundary_delta_p95_max_observations": 1,
                "maximum_unresolved_fraction": 0.1,
                "maximum_not_evaluable_fraction": 0.4,
                "decision_rule": "All pass; missing isolated-review evidence is NOT_READY.",
            },
            "calibration_sources": self.source_rows,
            "pilot_clips": [
                {
                    "clip_id": f"clip-{source_number}",
                    "source_id": f"calibration-{source_number}",
                    "frame_indices": [0, 1, 2],
                }
                for source_number in range(3)
            ],
        }

    def test_freeze_and_independent_validation(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        result = validate(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
        )
        self.assertEqual("VALID", result["status"])
        self.assertEqual(9, result["pilot_observation_count"])
        self.assertEqual(0, result["isolated_review_pass_count"])
        self.assertFalse(result["d0a2_production_labeling_authorized"])

    def test_production_source_is_rejected(self) -> None:
        self.lock["calibration_sources"][0]["source_id"] = "production-source"
        self.lock["calibration_sources"][0]["session_id"] = "production-session"
        self.lock["pilot_clips"][0]["source_id"] = "production-source"
        self._write_json(self.lock_path, self.lock)
        with self.assertRaises(PilotFreezeError):
            write_bundle(
                repo_root=self.repo,
                lock_path=self.lock_path,
                output_root=self.output_root,
                frozen_at_utc="2026-07-31T00:00:00Z",
            )

    def test_source_tamper_fails_validation(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        image_path = self.repo / "inputs" / "calibration-0" / "000.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image[0, 0] = 255
        self.assertTrue(cv2.imwrite(str(image_path), image))
        with self.assertRaises(PilotValidationError):
            validate(
                repo_root=self.repo,
                lock_path=self.lock_path,
                output_root=self.output_root,
            )

    def test_write_once_refuses_overwrite(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        with self.assertRaises(PilotFreezeError):
            write_bundle(
                repo_root=self.repo,
                lock_path=self.lock_path,
                output_root=self.output_root,
                frozen_at_utc="2026-07-31T00:00:01Z",
            )

    def _write_primary_fixture(
        self,
        *,
        isolated_context: bool = False,
        submitted_at_utc: str = "2026-07-31T00:00:02Z",
    ) -> None:
        input_validation = validate(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
        )
        self._write_json(self.output_root / "pilot-input-validation.json", input_validation)
        labels = [
            "VISIBLE_CENTRAL_OBSTRUCTION_PRESENT",
            "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE",
            "NOT_EVALUABLE",
        ]
        qualities = ["OCCLUDED", "STABLE", "TURNING"]
        review = {
            "schema_version": "blindassist.central_obstruction_d0a1_primary_review.v1",
            "protocol_id": self.lock["protocol_id"],
            "phase": "D0-A1",
            "evidence_instance": self.lock["evidence_instance"],
            "review_id": "fixture-primary",
            "reviewer_id": "fixture-reviewer",
            "reviewer_type": "ai_model",
            "review_context": "PRIMARY_CURRENT_TASK_NON_ISOLATED_SOURCE_ONLY",
            "isolated_context": isolated_context,
            "source_only_view": True,
            "candidate_output_visible": False,
            "prior_review_visible": False,
            "other_review_visible_before_submission": False,
            "labels_generated_before_r1_lock": False,
            "pilot_input_manifest_sha256": sha256_file(
                self.output_root / "pilot-input-manifest.json"
            ),
            "prompt_sha256": sha256_file(self.repo / "prompt.md"),
            "submitted_at_utc": submitted_at_utc,
            "clip_reviews": [
                {
                    "clip_id": f"clip-{source_number}",
                    "source_frame_indices": [0, 1, 2],
                    "labels": [labels[source_number]] * 3,
                    "quality_states": [qualities[source_number]] * 3,
                    "rationale_codes": [f"FIXTURE_{source_number}"] * 3,
                }
                for source_number in range(3)
            ],
            "claim_ceiling": "fixture",
        }
        self._write_json(self.output_root / "primary-review.json", review)

    def test_primary_review_is_complete_but_not_ready(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture()
        result, events = validate_primary_review(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
        )
        self.assertEqual("VALID", result["status"])
        self.assertTrue(result["coverage_preconditions_pass"])
        self.assertFalse(result["readiness_evaluated"])
        self.assertFalse(result["d0a2_production_labeling_authorized"])
        self.assertEqual(3, len(events))

    def test_primary_review_context_overclaim_is_rejected(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture(isolated_context=True)
        with self.assertRaises(PrimaryReviewValidationError):
            validate_primary_review(
                repo_root=self.repo,
                lock_path=self.lock_path,
                output_root=self.output_root,
            )

    def test_future_primary_review_submission_is_rejected(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture(submitted_at_utc="2999-01-01T00:00:00Z")
        with self.assertRaises(PrimaryReviewValidationError):
            validate_primary_review(
                repo_root=self.repo,
                lock_path=self.lock_path,
                output_root=self.output_root,
            )


if __name__ == "__main__":
    unittest.main()
