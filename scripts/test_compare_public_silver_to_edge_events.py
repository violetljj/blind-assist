#!/usr/bin/env python3
"""Unit tests for model-silver versus edge-event comparison."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from compare_public_silver_to_edge_events import ComparisonError, compare


class SilverEdgeComparisonTest(unittest.TestCase):
    def _inputs(self, directory: Path) -> tuple[Path, Path, dict, dict]:
        source_path = directory / "source.json"
        source_path.write_text(json.dumps({"frames": [{"sha256": "a" * 64}, {"sha256": "b" * 64}]}), encoding="utf-8")
        silver_path = directory / "silver.json"
        silver = {
            "schema": "blindassist_public_video_silver_labels_v1",
            "source": {"source_id": "public", "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "human_event_truth_present": False, "privacy_audit_required": True},
            "labeler": {"provider": "openai", "model": "gpt-5", "prompt_id": "p", "prompt_sha256": "c" * 64, "review_mode": "multiframe_temporal"},
            "episodes": [
                {"episode_id": "negative", "silver_should_alert": "candidate_no_alert", "confidence": 0.8, "evidence_frame_sha256": ["a" * 64, "b" * 64], "uncertainty_reasons": [], "risk_profile": {"corridor_relation": "outside_or_nonblocking"}, "negative_decision_quality": {"corridor_heading_stability": 0.8, "near_field_visibility": 0.8, "corridor_clearance": 0.8, "near_field_lateral_intrusion_absent": 0.8}},
                {"episode_id": "unknown", "silver_should_alert": "abstain", "confidence": 0.3, "evidence_frame_sha256": ["a" * 64, "b" * 64], "uncertainty_reasons": ["occluded"]},
            ],
            "training_execution_authorized": False, "calibration_authorized": False, "blind_evaluation_authorized": False, "production_model_replacement_authorized": False,
        }
        silver_path.write_text(json.dumps(silver), encoding="utf-8")
        edge = {"schema": "blindassist_public_video_edge_event_report_v1", "silver_manifest_sha256": hashlib.sha256(silver_path.read_bytes()).hexdigest(), "risk_config": "current", "human_event_truth_present": False, "production_model_replacement_authorized": False, "episodes": [{"episode_id": "negative", "edge_should_alert": False, "edge_event_ids": []}, {"episode_id": "unknown", "edge_should_alert": True, "edge_event_ids": ["x"]}], "frames": [{"episode_id": "negative", "edge_should_alert": False, "risk_event_id": None}, {"episode_id": "unknown", "edge_should_alert": True, "risk_event_id": "x"}]}
        return silver_path, source_path, silver, edge

    def test_abstentions_are_not_in_candidate_agreement_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            silver_path, source_path, silver, edge = self._inputs(Path(temporary))
            result = compare(silver, edge, silver_path=silver_path, source_manifest_path=source_path)
        self.assertEqual(result["comparable_episode_count"], 1)
        self.assertEqual(result["silver_abstain_count"], 1)
        self.assertEqual(result["candidate_agreement_rate"], 1.0)
        self.assertEqual(result["risk_config"], "current")
        self.assertEqual(result["comparison_rows"][0]["edge_duplicate_event_trigger_count"], 0)

    def test_rejects_report_for_a_different_silver_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            silver_path, source_path, silver, edge = self._inputs(Path(temporary))
            edge["silver_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ComparisonError, "not bound"):
                compare(silver, edge, silver_path=silver_path, source_manifest_path=source_path)

    def test_rejects_edge_report_without_risk_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            silver_path, source_path, silver, edge = self._inputs(Path(temporary))
            del edge["risk_config"]
            with self.assertRaisesRegex(ComparisonError, "risk_config"):
                compare(silver, edge, silver_path=silver_path, source_manifest_path=source_path)

    def test_preserves_untracked_trigger_as_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            silver_path, source_path, silver, edge = self._inputs(Path(temporary))
            edge["frames"][1]["risk_event_id"] = None
            result = compare(silver, edge, silver_path=silver_path, source_manifest_path=source_path)
            self.assertEqual(result["comparison_rows"][1]["edge_untracked_trigger_count"], 1)


if __name__ == "__main__":
    unittest.main()
