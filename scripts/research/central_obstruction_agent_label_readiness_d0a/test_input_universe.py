from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.central_obstruction_agent_label_readiness_d0a.freeze_input_universe import (
    FreezeError,
    sha256_file,
    write_freeze,
)
from scripts.research.central_obstruction_agent_label_readiness_d0a.validate_input_universe import (
    ValidationError,
    validate,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class InputUniverseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        payload = self.root / "inputs" / "session" / "frames"
        payload.mkdir(parents=True)
        rows = []
        for index, data in enumerate((b"frame-a", b"frame-b")):
            image = payload / f"frame_{index:06d}.jpg"
            image.write_bytes(data)
            rows.append(
                {
                    "frame_id": index,
                    "height": 480,
                    "image_path": f"frames/{image.name}",
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "source_capture_timestamp_ns": index * 100_000_000,
                    "source_id": "source",
                    "width": 640,
                }
            )
        ledger = self.root / "inputs" / "session" / "manifest.jsonl"
        ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        write_json(self.root / "inputs" / "source.json", {"source": "fixture"})
        protocol = {
            "protocol_id": "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A",
            "execution_authorized": True,
            "status": "AUTHORIZED_NOT_RUN",
            "phase_plan": [
                {
                    "phase": "D0-A0",
                    "candidate_output_access": False,
                    "output": "reuse-first role ledger",
                }
            ],
            "artifact_contract": {"required_before_d0a2": ["reuse-role-ledger.jsonl"]},
            "reuse_first_admission_policy": {
                "required_session_manifest_fields": [
                    "source_id",
                    "session_id",
                    "dataset_name",
                    "content_identity",
                    "independence_group",
                    "ancestry",
                    "current_task_fitness",
                    "missing_current_task_requirements",
                    "prior_content_access",
                    "prior_algorithm_output_access",
                    "claim_relevant_outcome_overlap",
                    "selection_or_tuning_influence",
                    "assigned_current_role",
                    "admission_disposition",
                    "exclusion_reason",
                    "reuse_candidates",
                ]
            },
        }
        write_json(self.root / "protocol.json", protocol)
        (self.root / "fitness_prompt.md").write_text("fitness only\n", encoding="utf-8")
        workflow = {
            "workflows": {
                "central_obstruction_agent_label_readiness_d0a_v1": {
                    "candidate_output_hidden_from_reviewers": True
                }
            }
        }
        write_json(self.root / "workflow.json", workflow)
        spec = {
            "protocol_id": "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A",
            "phase": "D0-A0",
            "evidence_instance": "TEST",
            "candidate_output_access": False,
            "protocol_path": "protocol.json",
            "workflow_path": "workflow.json",
            "output_root": "output",
            "eligibility_rule": {"id": "TEST", "include": ["all"], "exclude": ["none"]},
            "fitness_review": {
                "prompt_path": "fitness_prompt.md",
                "reviewer_id": "fixture-reviewer",
                "reviewer_type": "ai_model",
                "reviewer_role": "codex_multimodal_source_fitness_reviewer",
                "provider": "openai",
                "model": "codex",
                "model_version": "fixture",
                "review_run_id": "fixture-run",
                "isolated_context": True,
                "candidate_output_visible": False,
                "truth_or_review_label_visible": False,
                "sample_rule": "8 evenly spaced frames including both endpoints per admitted session",
                "confidence": 0.9,
                "abstained": False,
                "abstain_reasons": [],
            },
            "sessions": [
                {
                    "source_id": "source",
                    "session_id": "session",
                    "source_ancestry_group": "group",
                    "source_kind": "fixture",
                    "source_uri": "fixture://source",
                    "license_or_rights_metadata": "fixture",
                    "ledger_adapter": "PUBLIC_VIDEO_REPLAY_RGB_V1",
                    "ledger_path": "inputs/session/manifest.jsonl",
                    "ledger_sha256": sha256_file(ledger),
                    "payload_root": "inputs/session/frames",
                    "expected_frame_count": 2,
                    "expected_width": 640,
                    "expected_height": 480,
                    "timestamp_semantics": "DERIVED_FIXED_10HZ_FROM_VIDEO_START_NS",
                    "expected_frame_step_ns": 100_000_000,
                    "materialization": "fixture",
                    "prior_access_state": "CONTENT_INSPECTED_DEVELOPMENT",
                    "prior_content_access": True,
                    "prior_candidate_output_access": False,
                    "dataset_name": "fixture",
                    "independence_group": "group",
                    "ancestry": ["fixture"],
                    "current_task_fitness": "PASS",
                    "missing_current_task_requirements": ["D0-A1 lock"],
                    "prior_algorithm_output_access": False,
                    "claim_relevant_outcome_overlap": "NONE",
                    "selection_or_tuning_influence": "NONE",
                    "assigned_current_role": "CANARY",
                    "admission_disposition": "ADMIT_D0_A_PRODUCTION_LABELING",
                    "exclusion_reason": None,
                    "reuse_candidates": ["fixture"],
                    "ancestry_receipt_paths": ["inputs/source.json"],
                }
            ],
        }
        self.spec_path = self.root / "spec.json"
        write_json(self.spec_path, spec)
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def freeze(self) -> None:
        write_freeze(
            repo_root=self.root,
            spec_path=self.spec_path,
            output_root=self.output,
            frozen_at_utc="2026-07-31T08:00:00Z",
        )

    def test_freeze_and_independent_validation(self) -> None:
        self.freeze()
        result = validate(
            repo_root=self.root,
            spec_path=self.spec_path,
            output_root=self.output,
            validated_at_utc="2026-07-31T08:01:00Z",
        )
        self.assertEqual("VALID", result["status"])
        self.assertEqual(2, result["frame_count"])
        self.assertFalse(result["candidate_output_access"])
        self.assertEqual(1, result["reuse_role_row_count"])

    def test_write_once_refuses_overwrite(self) -> None:
        self.freeze()
        with self.assertRaises(FreezeError):
            self.freeze()

    def test_payload_hash_drift_fails_freeze(self) -> None:
        (self.root / "inputs/session/frames/frame_000001.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(FreezeError, "payload SHA-256 mismatch"):
            self.freeze()

    def test_manifest_tamper_fails_validation(self) -> None:
        self.freeze()
        path = self.output / "input-universe-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["frames"][0]["frame_index"] = 99
        write_json(path, manifest)
        with self.assertRaisesRegex(ValidationError, "manifest receipt hash mismatch"):
            validate(
                repo_root=self.root,
                spec_path=self.spec_path,
                output_root=self.output,
                validated_at_utc="2026-07-31T08:01:00Z",
            )

    def test_missing_reuse_role_ledger_fails_validation(self) -> None:
        self.freeze()
        (self.output / "reuse-role-ledger.jsonl").unlink()
        with self.assertRaises((ValidationError, FileNotFoundError)):
            validate(
                repo_root=self.root,
                spec_path=self.spec_path,
                output_root=self.output,
                validated_at_utc="2026-07-31T08:01:00Z",
            )

    def test_candidate_firewall_must_be_false(self) -> None:
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        spec["candidate_output_access"] = True
        write_json(self.spec_path, spec)
        with self.assertRaisesRegex(FreezeError, "candidate-output firewall"):
            self.freeze()

    def test_declared_parent_escape_is_rejected(self) -> None:
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        spec["sessions"][0]["ledger_path"] = "../outside.jsonl"
        write_json(self.spec_path, spec)
        with self.assertRaisesRegex(FreezeError, "path escapes repository"):
            self.freeze()


if __name__ == "__main__":
    unittest.main()
