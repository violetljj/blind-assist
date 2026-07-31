from __future__ import annotations

import json
import datetime as dt
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
from .validate_d0a1_isolated_review import (
    IsolatedReviewValidationError,
    validate_isolated_review,
)
from .finalize_d0a1_adjudication import (
    AdjudicationValidationError,
    finalize_adjudication,
    write_finalization,
)
from .validate_d0a1_final_readiness import validate_final


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
                "reviewer_event_match": {
                    "same_source_and_clip": True,
                    "same_label": True,
                    "minimum_temporal_iou": 0.5,
                    "maximum_start_or_end_delta_observations": 1,
                },
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

    def _write_isolated_fixture(
        self,
        *,
        disagreement: bool = False,
        isolated_context: bool = True,
    ) -> Path:
        primary_result, _ = validate_primary_review(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
        )
        self._write_json(self.output_root / "primary-review-validation.json", primary_result)
        labels = [
            "VISIBLE_CENTRAL_OBSTRUCTION_PRESENT",
            "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE",
            "NOT_EVALUABLE",
        ]
        qualities = ["OCCLUDED", "STABLE", "TURNING"]
        clip_reviews = []
        for source_number in range(3):
            clip_labels = [labels[source_number]] * 3
            if disagreement and source_number == 0:
                clip_labels[0] = "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE"
            clip_reviews.append(
                {
                    "clip_id": f"clip-{source_number}",
                    "observations": [
                        {
                            "clip_observation_ordinal": ordinal,
                            "source_frame_index": ordinal,
                            "label": clip_labels[ordinal],
                            "quality_state": qualities[source_number],
                            "rationale": f"fixture {source_number} {ordinal}",
                        }
                        for ordinal in range(3)
                    ],
                }
            )
        review = {
            "schema_version": "blindassist.central_obstruction_d0a1_isolated_review.v1",
            "protocol_id": self.lock["protocol_id"],
            "phase": "D0-A1",
            "evidence_instance": self.lock["evidence_instance"],
            "review_id": "0d91ed10-6ea3-4e0b-aac9-b625e47c28c1",
            "reviewer_id": "fixture-isolated-reviewer",
            "reviewer_type": "CODEX_AGENT",
            "review_context": "FRESH_ISOLATED_SECOND_PASS",
            "isolated_context": isolated_context,
            "source_only_view": True,
            "candidate_output_visible": False,
            "prior_review_visible": False,
            "other_review_visible_before_submission": False,
            "pilot_input_manifest_sha256": sha256_file(
                self.output_root / "pilot-input-manifest.json"
            ),
            "prompt_sha256": sha256_file(self.repo / "prompt.md"),
            "submitted_at_utc": "2026-07-31T00:00:03Z",
            "clip_reviews": clip_reviews,
            "attestation": {"prohibited_content_read": False, "coverage": "9/9"},
        }
        review_path = self.repo / "isolated-review.json"
        self._write_json(review_path, review)
        return review_path

    def test_perfect_isolated_review_authorizes_d0a2_design(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture()
        review_path = self._write_isolated_fixture()
        result, events, packet = validate_isolated_review(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            review_path=review_path,
        )
        self.assertEqual("READY_FOR_D0_A2_PRIMARY_AGENT_LABELING", result["decision"])
        self.assertTrue(result["d0a2_production_labeling_authorized"])
        self.assertEqual(1.0, result["overall_observation_label_agreement"])
        self.assertEqual(3, len(events))
        self.assertIsNone(packet)

    def test_material_disagreement_freezes_adjudication_packet(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture()
        review_path = self._write_isolated_fixture(disagreement=True)
        result, _, packet = validate_isolated_review(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            review_path=review_path,
        )
        self.assertEqual("MATERIAL_DISAGREEMENT_ADJUDICATION_REQUIRED", result["decision"])
        self.assertFalse(result["d0a2_production_labeling_authorized"])
        self.assertEqual(1, result["material_disagreement_count"])
        self.assertIsNotNone(packet)
        self.assertEqual(1, packet["material_disagreement_count"])

    def test_isolated_context_overclaim_is_rejected(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture()
        review_path = self._write_isolated_fixture(isolated_context=False)
        with self.assertRaises(IsolatedReviewValidationError):
            validate_isolated_review(
                repo_root=self.repo,
                lock_path=self.lock_path,
                output_root=self.output_root,
                review_path=review_path,
            )

    def test_isolated_review_prompt_hash_alias_is_accepted(self) -> None:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture()
        review_path = self._write_isolated_fixture()
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["review_prompt_sha256"] = review.pop("prompt_sha256")
        self._write_json(review_path, review)
        result, _, _ = validate_isolated_review(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            review_path=review_path,
        )
        self.assertEqual("VALID", result["status"])

    def _prepare_adjudication_fixture(self) -> Path:
        write_bundle(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            frozen_at_utc="2026-07-31T00:00:00Z",
        )
        self._write_primary_fixture()
        isolated_source = self._write_isolated_fixture(disagreement=True)
        isolated_review = json.loads(isolated_source.read_text(encoding="utf-8"))
        isolated_review["clip_reviews"][0]["observations"][1][
            "label"
        ] = "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE"
        self._write_json(isolated_source, isolated_review)
        agreement, events, packet = validate_isolated_review(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            review_path=isolated_source,
        )
        self.assertIsNotNone(packet)
        (self.output_root / "isolated-second-review.json").write_bytes(
            isolated_source.read_bytes()
        )
        (self.output_root / "isolated-second-parent-events.jsonl").write_bytes(
            b"".join(canonical_bytes(row) for row in events)
        )
        self._write_json(self.output_root / "d0a1-initial-agreement.json", agreement)
        self._write_json(self.output_root / "d0a1-adjudication-packet.json", packet)
        review = {
            "schema_version": "blindassist.central_obstruction_d0a1_adjudication_review.v1",
            "protocol_id": self.lock["protocol_id"],
            "phase": "D0-A1",
            "evidence_instance": self.lock["evidence_instance"],
            "review_id": "92db7361-bd9f-493f-8532-0c769ca0d89c",
            "reviewer_id": "fixture-third-adjudicator",
            "reviewer_type": "CODEX_AGENT",
            "review_context": "FRESH_THIRD_AGENT_MATERIAL_DISAGREEMENT_ADJUDICATION",
            "source_only_view": True,
            "candidate_output_visible": False,
            "pair_labels_visible": True,
            "aggregate_metrics_visible": False,
            "adjudication_packet_sha256": sha256_file(
                self.output_root / "d0a1-adjudication-packet.json"
            ),
            "prompt_sha256": sha256_file(self.repo / "prompt.md"),
            "submitted_at_utc": dt.datetime.now(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "items": [
                {
                    "item_id": item["item_id"],
                    "clip_id": item["clip_id"],
                    "clip_observation_ordinal": item["clip_observation_ordinal"],
                    "source_frame_index": item["source_frame_index"],
                    "final_label": item["primary_label"],
                    "final_quality_state": item["primary_quality_state"],
                    "disposition": "ADJUDICATED_LABEL",
                    "rationale": "fixture adjudication",
                }
                for item in packet["items"]
            ],
            "attestation": {"coverage": "complete", "prohibited_content_read": False},
        }
        review_path = self.repo / "adjudication-review.json"
        self._write_json(review_path, review)
        return review_path

    def test_adjudication_preserves_raw_failed_threshold(self) -> None:
        review_path = self._prepare_adjudication_fixture()
        result, labels, events = finalize_adjudication(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            review_path=review_path,
        )
        self.assertEqual("AGENT_LABEL_PROTOCOL_NOT_RELIABLE", result["terminal"])
        self.assertFalse(result["d0a2_production_labeling_authorized"])
        self.assertEqual(9, len(labels))
        self.assertGreater(len(events), 0)
        self.assertTrue(result["threshold_checks"]["adjudication_complete"])

    def test_incomplete_adjudication_is_rejected(self) -> None:
        review_path = self._prepare_adjudication_fixture()
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["items"] = []
        self._write_json(review_path, review)
        with self.assertRaises(AdjudicationValidationError):
            finalize_adjudication(
                repo_root=self.repo,
                lock_path=self.lock_path,
                output_root=self.output_root,
                review_path=review_path,
            )

    def test_adjudication_packet_hash_alias_is_accepted(self) -> None:
        review_path = self._prepare_adjudication_fixture()
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["packet_sha256"] = review.pop("adjudication_packet_sha256")
        self._write_json(review_path, review)
        result, _, _ = finalize_adjudication(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            review_path=review_path,
        )
        self.assertEqual("VALID", result["status"])

    def test_stored_final_readiness_recomputes_exactly(self) -> None:
        review_path = self._prepare_adjudication_fixture()
        write_finalization(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
            review_path=review_path,
        )
        result = validate_final(
            repo_root=self.repo,
            lock_path=self.lock_path,
            output_root=self.output_root,
        )
        self.assertEqual("VALID", result["status"])
        self.assertEqual("AGENT_LABEL_PROTOCOL_NOT_RELIABLE", result["terminal"])


if __name__ == "__main__":
    unittest.main()
