from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("counterfactual", SCRIPTS / "validate_sanpo_counterfactual_episodes.py")
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RouteConditionedEventTruthTest(unittest.TestCase):
    def test_complete_pair_requires_and_reports_route_bound_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.manifest(root)
            report = subject.validate(self.config(), manifest, root=root, require_complete=True)
            self.assertTrue(report["route_conditioned_truth_eligible"])
            self.assertEqual(2, report["route_bound_episode_count"])
            self.assertFalse(report["production_model_replacement_authorized"])

    def test_future_route_or_cross_pair_route_intent_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.manifest(root)
            route_path = root / manifest["episodes"][0]["route_intent_path"]
            route = json.loads(route_path.read_text(encoding="utf-8"))
            route["provider"]["type"] = "train_only_future_video_oracle"
            route_path.write_text(json.dumps(route), encoding="utf-8")
            manifest["episodes"][0]["route_intent_sha256"] = sha(route_path)
            with self.assertRaisesRegex(subject.ContractError, "not allowed at runtime"):
                subject.validate(self.config(), manifest, root=root, require_complete=True)

            manifest = self.manifest(root)
            route_path = root / manifest["episodes"][1]["route_intent_path"]
            route = json.loads(route_path.read_text(encoding="utf-8"))
            route["route_intent_id"] = "different-route"
            route_path.write_text(json.dumps(route), encoding="utf-8")
            manifest["episodes"][1]["route_intent_sha256"] = sha(route_path)
            with self.assertRaisesRegex(subject.ContractError, "crosses route_intent_id"):
                subject.validate(self.config(), manifest, root=root, require_complete=True)

    def test_sparse_route_and_manifest_adjudication_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.manifest(root)
            route_path = root / manifest["episodes"][0]["route_intent_path"]
            route = json.loads(route_path.read_text(encoding="utf-8"))
            route["samples"] = route["samples"][::4]
            route_path.write_text(json.dumps(route), encoding="utf-8")
            manifest["episodes"][0]["route_intent_sha256"] = sha(route_path)
            with self.assertRaisesRegex(subject.ContractError, "gap above policy"):
                subject.validate(self.config(), manifest, root=root, require_complete=True)

            manifest = self.manifest(root)
            annotation_path = root / manifest["episodes"][0]["annotation_evidence_path"]
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotation["adjudication"]["alertable_start_ms"] = 300
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            manifest["episodes"][0]["annotation_evidence_sha256"] = sha(annotation_path)
            with self.assertRaisesRegex(subject.ContractError, "manifest alertable_start_ms differs"):
                subject.validate(self.config(), manifest, root=root, require_complete=True)

    @staticmethod
    def config() -> dict:
        fields = [
            "episode_id", "session_id", "scene_id", "matched_pair_id", "pair_role",
            "risk_event_id", "expected_should_alert", "expected_critical", "criticality_reason",
            "video_path", "video_sha256", "source_receipt_id", "capture_clock_receipt_path",
            "capture_clock_receipt_sha256", "route_intent_path", "route_intent_sha256",
            "annotation_reviewer_ids", "annotation_evidence_path", "annotation_evidence_sha256",
            "duration_ms", "risk_profile", "lifecycle_intervals_ms",
        ]
        return {
            "schema": "blindassist_sanpo_counterfactual_episode_collection_v1",
            "design": {"session_count": 1, "scene_count": 1, "matched_pairs_per_session_scene": 1},
            "sessions": [{"session_id": "s1"}],
            "scenes": [{"scene_id": "route_obstacle"}],
            "source_receipt_schema": {
                "allowed_license_status": ["owned_and_consented"],
                "required_privacy_review_status": "green",
                "hash_license_and_privacy_evidence": True,
            },
            "episode_duration_policy": {"minimum_duration_ms": 10000, "maximum_duration_ms": 20000},
            "episode_record_schema": {"required_fields": fields, "pair_role_allowed": ["positive", "matched_negative"]},
            "annotation_evidence_schema": {
                "schema": "blindassist_sanpo_counterfactual_annotation_evidence_v1",
                "minimum_independent_reviewers_per_episode": 2,
                "reviewer_id_must_match_episode": True,
                "positive_anchor_agreement_tolerance_ms": 500,
            },
            "matrix_contract": {
                "matched_pair_members_must_share_capture_context": [
                    "location", "lighting", "device", "camera_configuration", "camera_frame",
                ]
            },
            "route_conditioning_policy": {
                "required": True,
                "minimum_confidence": 0.7,
                "minimum_valid_sample_fraction": 0.95,
                "maximum_valid_sample_gap_ms": 500,
                "endpoint_coverage_tolerance_ms": 500,
                "capture_clock_receipt_schema": "blindassist_capture_clock_receipt_v1",
                "manifest_must_match_hashed_adjudication": True,
            },
        }

    @classmethod
    def manifest(cls, root: Path) -> dict:
        raw = root / "raw.mp4"
        raw.write_bytes(b"controlled route event")
        license_file = root / "license.json"
        privacy_file = root / "privacy.json"
        inventory = root / "inventory.json"
        license_file.write_text("{}", encoding="utf-8")
        privacy_file.write_text("{}", encoding="utf-8")
        inventory.write_text("{}", encoding="utf-8")
        receipt = {
            "source_receipt_id": "source-1",
            "source_owner_or_dataset": "controlled-owned",
            "collection_date": "2026-07-20",
            "license_status": "owned_and_consented",
            "license_evidence_path": license_file.name,
            "license_evidence_sha256": sha(license_file),
            "privacy_review_status": "green",
            "privacy_evidence_path": privacy_file.name,
            "privacy_evidence_sha256": sha(privacy_file),
            "reviewer_id": "privacy-reviewer",
            "raw_video_path": raw.name,
            "raw_video_sha256": sha(raw),
            "episode_manifest_path": inventory.name,
            "episode_manifest_sha256": sha(inventory),
        }
        context = {
            "location": "controlled-corner",
            "lighting": "day",
            "device": "test-device",
            "camera_configuration": "rear-wide",
            "camera_frame": "camera-v1",
        }
        episodes = [
            cls.episode(root, raw, context, "pos", "positive", True),
            cls.episode(root, raw, context, "neg", "matched_negative", False),
        ]
        return {
            "schema": "blindassist_sanpo_counterfactual_episode_manifest_v1",
            "collection_status": "complete",
            "source_receipts": [receipt],
            "episodes": episodes,
        }

    @classmethod
    def episode(cls, root: Path, raw: Path, context: dict, episode_id: str, role: str, positive: bool) -> dict:
        clock_path = root / f"{episode_id}-clock.json"
        clock_path.write_text(json.dumps({
            "schema": "blindassist_capture_clock_receipt_v1",
            "episode_id": episode_id,
            "camera_frame": "camera-v1",
            "timestamp_unit": "nanoseconds",
            "timestamps_strictly_monotonic": True,
            "frame_count": 300,
        }), encoding="utf-8")
        route_path = root / f"{episode_id}-route.json"
        samples = [{
            "timestamp_ms": timestamp,
            "valid_until_timestamp_ms": timestamp + 500,
            "confidence": 0.95,
            "route_valid": True,
            "horizon_waypoints": [
                {"horizon_ms": 1000, "xy_norm": [0.5, 0.9]},
                {"horizon_ms": 2000, "xy_norm": [0.5, 0.7]},
                {"horizon_ms": 3000, "xy_norm": [0.5, 0.5]},
            ],
        } for timestamp in range(0, 10001, 500)]
        route_path.write_text(json.dumps({
            "schema": "blindassist_explicit_route_intent_episode_v1",
            "episode_id": episode_id,
            "route_intent_id": "shared-straight-route",
            "parent_source_id": "source-1",
            "provider": {
                "type": "navigation",
                "provider_id": "nav-1",
                "inferred_by_risk_model": False,
                "input_space": "current_camera_frame",
            },
            "coordinate_contract": {
                "space": "normalized_current_camera_frame_xy",
                "projection_receipt_id": f"projection-{episode_id}",
                "device_to_world_alignment_receipt_id": None,
            },
            "samples": samples,
            "fallback": {
                "missing_stale_or_low_confidence_route": "context_attention_only",
                "directional_instruction_allowed": False,
                "intervention_upgrade_allowed": False,
            },
            "training_isolation": {"future_video_teacher_allowed_in_eval_or_runtime": False},
        }), encoding="utf-8")
        annotation_path = root / f"{episode_id}-annotation.json"
        reviews = [{
            "reviewer_id": reviewer,
            "reviewer_type": "human",
            "should_alert": positive,
            "critical": positive,
            "first_visible_ms": 0 if positive else None,
            "alertable_start_ms": 200 if positive else None,
            "passed_or_cleared_ms": 9000 if positive else None,
        } for reviewer in ("reviewer-1", "reviewer-2")]
        adjudication = {
            "method": "reviewer_consensus",
            "should_alert": positive,
            "critical": positive,
            "first_visible_ms": 0 if positive else None,
            "alertable_start_ms": 200 if positive else None,
            "passed_or_cleared_ms": 9000 if positive else None,
        }
        annotation_path.write_text(json.dumps({
            "schema": "blindassist_sanpo_counterfactual_annotation_evidence_v1",
            "episode_id": episode_id,
            "reviews": reviews,
            "adjudication": adjudication,
        }), encoding="utf-8")
        row = {
            "episode_id": episode_id,
            "session_id": "s1",
            "scene_id": "route_obstacle",
            "matched_pair_id": "pair-1",
            "pair_role": role,
            "risk_event_id": f"event-{episode_id}",
            "expected_should_alert": positive,
            "expected_critical": positive,
            "criticality_reason": "controlled contact risk" if positive else "route remains clear",
            "video_path": raw.name,
            "video_sha256": sha(raw),
            "source_receipt_id": "source-1",
            "capture_clock_receipt_path": clock_path.name,
            "capture_clock_receipt_sha256": sha(clock_path),
            "route_intent_path": route_path.name,
            "route_intent_sha256": sha(route_path),
            "annotation_reviewer_ids": ["reviewer-1", "reviewer-2"],
            "annotation_evidence_path": annotation_path.name,
            "annotation_evidence_sha256": sha(annotation_path),
            "duration_ms": 10000,
            "capture_context": dict(context),
            "risk_profile": {
                "primary_hazard_type": "route_obstacle",
                "corridor_relation": "enters_or_blocks" if positive else "outside_or_nonblocking",
                "lifecycle": "approach_alertable_clear" if positive else "no_alert",
            },
            "lifecycle_intervals_ms": {
                "approach": [0, 200],
                "alertable": [200, 9000],
                "post_event": [9000, 10000],
            } if positive else {"non_alert": [0, 10000]},
            "first_visible_ms": 0 if positive else None,
            "alertable_start_ms": 200 if positive else None,
            "passed_or_cleared_ms": 9000 if positive else None,
        }
        if not positive:
            row["negative_reason"] = "matched obstacle remains outside selected route"
        return row


if __name__ == "__main__":
    unittest.main()
