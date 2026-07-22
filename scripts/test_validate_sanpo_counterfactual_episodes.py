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
counterfactual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(counterfactual)
BUILDER_SPEC = importlib.util.spec_from_file_location("risk_lifecycle_builder", SCRIPTS / "build_sanpo_risk_lifecycle_targets.py")
assert BUILDER_SPEC and BUILDER_SPEC.loader
risk_lifecycle_builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(risk_lifecycle_builder)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CounterfactualEpisodeValidatorTest(unittest.TestCase):
    def test_complete_matched_pair_is_training_eligible_but_never_production_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            manifest = self._manifest(root)
            report = counterfactual.validate(config, manifest, root=root, require_complete=True)
            self.assertTrue(report["ok"])
            self.assertTrue(report["training_eligible"])
            self.assertFalse(report["production_model_replacement_authorized"])

    def test_rejects_negative_with_alert_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            manifest["episodes"][1]["alertable_start_ms"] = 1
            with self.assertRaisesRegex(counterfactual.ContractError, "must be null"):
                counterfactual.validate(self._config(root), manifest, root=root, require_complete=True)

    def test_rejects_pair_context_mismatch_and_incomplete_training_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            manifest["episodes"][1]["capture_context"]["lighting"] = "night"
            with self.assertRaisesRegex(counterfactual.ContractError, "unmatched capture_context"):
                counterfactual.validate(self._config(root), manifest, root=root, require_complete=True)
            manifest = self._manifest(root)
            manifest["collection_status"] = "in_review"
            with self.assertRaisesRegex(counterfactual.ContractError, "collection_status=complete"):
                counterfactual.validate(self._config(root), manifest, root=root, require_complete=True)

    def test_rejects_short_episode_and_lifecycle_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            manifest["episodes"][0]["duration_ms"] = 9999
            with self.assertRaisesRegex(counterfactual.ContractError, "within 10000..20000"):
                counterfactual.validate(self._config(root), manifest, root=root, require_complete=True)
            manifest = self._manifest(root)
            manifest["episodes"][0]["lifecycle_intervals_ms"]["alertable"] = [101, 9000]
            with self.assertRaisesRegex(counterfactual.ContractError, "lifecycle_intervals_ms.alertable"):
                counterfactual.validate(self._config(root), manifest, root=root, require_complete=True)

    def test_rejects_missing_independent_review_or_unstable_positive_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            evidence = root / "pos-review.json"
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["reviews"] = payload["reviews"][:1]
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            manifest["episodes"][0]["annotation_evidence_sha256"] = sha(evidence)
            with self.assertRaisesRegex(counterfactual.ContractError, "independent reviewers"):
                counterfactual.validate(self._config(root), manifest, root=root, require_complete=True)
            manifest = self._manifest(root)
            evidence = root / "pos-review.json"
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["reviews"][1]["alertable_start_ms"] = 700
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            manifest["episodes"][0]["annotation_evidence_sha256"] = sha(evidence)
            with self.assertRaisesRegex(counterfactual.ContractError, "alertable_start_ms exceeds"):
                counterfactual.validate(self._config(root), manifest, root=root, require_complete=True)

    def test_builds_risk_lifecycle_targets_and_authorizes_research_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            targets, report = risk_lifecycle_builder.build(self._config(root), self._manifest(root), root=root)
            self.assertEqual(2, len(targets))
            self.assertEqual("auxiliary_only", targets[0]["pixel_supervision_role"])
            self.assertTrue(report["training_execution_authorized"])
            self.assertEqual("hash_bound_model_consensus", report["supervision_tier"])
            self.assertFalse(report["production_model_replacement_authorized"])

    @staticmethod
    def _config(root: Path) -> dict:
        return {
            "schema": "blindassist_sanpo_counterfactual_episode_collection_v1",
            "design": {"session_count": 1, "scene_count": 1, "matched_pairs_per_session_scene": 1},
            "sessions": [{"session_id": "s1"}],
            "scenes": [{"scene_id": "step_curb"}],
            "source_receipt_schema": {"allowed_license_status": ["public_download_unknown_recorded"], "allowed_privacy_review_status": ["unknown_recorded"], "hash_license_and_privacy_evidence": False},
            "episode_duration_policy": {"minimum_duration_ms": 10000, "maximum_duration_ms": 20000},
            "episode_record_schema": {"required_fields": ["episode_id", "session_id", "scene_id", "matched_pair_id", "pair_role", "risk_event_id", "expected_should_alert", "video_path", "video_sha256", "source_receipt_id", "annotation_reviewer_ids", "annotation_evidence_path", "annotation_evidence_sha256", "duration_ms", "risk_profile", "lifecycle_intervals_ms"], "pair_role_allowed": ["positive", "matched_negative"]},
            "annotation_evidence_schema": {"schema": "blindassist_sanpo_counterfactual_annotation_evidence_v1", "minimum_independent_reviewers_per_episode": 2, "reviewer_id_must_match_episode": True, "positive_anchor_agreement_tolerance_ms": 500},
            "matrix_contract": {"matched_pair_members_must_share_capture_context": ["location", "lighting", "device", "camera_configuration", "object_category"]},
        }

    @staticmethod
    def _manifest(root: Path) -> dict:
        raw = root / "raw.mp4"
        evidence = root / "receipt-evidence.json"
        raw.write_bytes(b"controlled route video")
        evidence.write_text("{}", encoding="utf-8")
        receipt = {"source_receipt_id": "r1", "source_owner_or_dataset": "SANPO", "collection_date": "2026-07-13", "source_url": "https://example.test/public-video", "retrieved_at": "2026-07-13T00:00:00Z", "license_status": "public_download_unknown_recorded", "privacy_review_status": "unknown_recorded", "reviewer_id": "codex-provenance-review", "raw_video_path": "raw.mp4", "raw_video_sha256": sha(raw), "episode_manifest_path": "receipt-evidence.json", "episode_manifest_sha256": sha(evidence)}
        context = {"location": "corner", "lighting": "day", "device": "chest", "camera_configuration": "left", "object_category": "curb"}
        pos_review = root / "pos-review.json"
        neg_review = root / "neg-review.json"
        pos_review.write_text(json.dumps({"schema": "blindassist_sanpo_counterfactual_annotation_evidence_v1", "episode_id": "pos", "reviews": [{"reviewer_id": "r1", "reviewer_type": "ai_model", "should_alert": True, "first_visible_ms": 0, "alertable_start_ms": 100, "passed_or_cleared_ms": 9000}, {"reviewer_id": "r2", "reviewer_type": "ai_model", "should_alert": True, "first_visible_ms": 50, "alertable_start_ms": 200, "passed_or_cleared_ms": 9100}]}), encoding="utf-8")
        neg_review.write_text(json.dumps({"schema": "blindassist_sanpo_counterfactual_annotation_evidence_v1", "episode_id": "neg", "reviews": [{"reviewer_id": "r1", "reviewer_type": "ai_model", "should_alert": False}, {"reviewer_id": "r2", "reviewer_type": "ai_model", "should_alert": False}]}), encoding="utf-8")
        common = {"session_id": "s1", "scene_id": "step_curb", "matched_pair_id": "p1", "video_path": "raw.mp4", "video_sha256": sha(raw), "source_receipt_id": "r1", "annotation_reviewer_ids": ["r1", "r2"]}
        return {"schema": "blindassist_sanpo_counterfactual_episode_manifest_v1", "collection_status": "complete", "source_receipts": [receipt], "episodes": [{**common, "capture_context": dict(context), "episode_id": "pos", "pair_role": "positive", "risk_event_id": "event-1", "expected_should_alert": True, "annotation_evidence_path": "pos-review.json", "annotation_evidence_sha256": sha(pos_review), "duration_ms": 10000, "risk_profile": {"primary_hazard_type": "step_curb", "corridor_relation": "enters_or_blocks", "lifecycle": "approach_alertable_clear"}, "lifecycle_intervals_ms": {"approach": [0, 100], "alertable": [100, 9000], "post_event": [9000, 10000]}, "first_visible_ms": 0, "alertable_start_ms": 100, "passed_or_cleared_ms": 9000}, {**common, "capture_context": dict(context), "episode_id": "neg", "pair_role": "matched_negative", "risk_event_id": "event-1-neg", "expected_should_alert": False, "annotation_evidence_path": "neg-review.json", "annotation_evidence_sha256": sha(neg_review), "duration_ms": 10000, "risk_profile": {"primary_hazard_type": "step_curb", "corridor_relation": "outside_or_nonblocking", "lifecycle": "no_alert"}, "lifecycle_intervals_ms": {"non_alert": [0, 10000]}, "first_visible_ms": None, "alertable_start_ms": None, "passed_or_cleared_ms": None, "negative_reason": "already passed"}]}


if __name__ == "__main__":
    unittest.main()
