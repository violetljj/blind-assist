#!/usr/bin/env python3
"""Unit tests for the public-video GPT/VLM silver-label quarantine gate."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_public_video_silver_labels import SilverLabelError, validate


class SilverLabelContractTest(unittest.TestCase):
    def _source(self, directory: Path) -> tuple[Path, str, str]:
        first, second = "a" * 64, "b" * 64
        path = directory / "source.json"
        path.write_text(json.dumps({"frames": [{"sha256": first}, {"sha256": second}]}), encoding="utf-8")
        return path, first, second

    def _manifest(self, source_path: Path, first: str, second: str) -> dict:
        return {
            "schema": "blindassist_public_video_silver_labels_v1",
            "source": {"source_id": "public", "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "human_event_truth_present": False, "privacy_audit_required": True},
            "labeler": {"provider": "openai", "model": "gpt-5", "prompt_id": "blindassist-risk-silver-v1", "prompt_sha256": "c" * 64, "review_mode": "multiframe_temporal"},
            "episodes": [{"episode_id": "clip-1", "silver_should_alert": "candidate_no_alert", "confidence": 0.71, "evidence_frame_sha256": [first, second], "uncertainty_reasons": ["sparse sampling"], "risk_profile": {"corridor_relation": "outside_or_nonblocking"}, "negative_decision_quality": {"corridor_heading_stability": 0.8, "near_field_visibility": 0.8, "corridor_clearance": 0.8, "near_field_lateral_intrusion_absent": 0.8}}],
            "training_execution_authorized": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "production_model_replacement_authorized": False,
        }

    def test_accepts_hash_bound_model_only_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, first, second = self._source(Path(temporary))
            result = validate(self._manifest(source, first, second), source_manifest_path=source)
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidate_no_alert_count"], 1)
        self.assertFalse(result["training_execution_authorized"])

    def test_rejects_unknown_evidence_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, first, second = self._source(Path(temporary))
            manifest = self._manifest(source, first, second)
            manifest["episodes"][0]["evidence_frame_sha256"][1] = "z" * 64
            with self.assertRaisesRegex(SilverLabelError, "known frame hashes"):
                validate(manifest, source_manifest_path=source)

    def test_rejects_training_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, first, second = self._source(Path(temporary))
            manifest = self._manifest(source, first, second)
            manifest["training_execution_authorized"] = True
            with self.assertRaisesRegex(SilverLabelError, "must be false"):
                validate(manifest, source_manifest_path=source)

    def test_accepts_v2_provisional_training_with_authorized_ccby_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = "a" * 64, "b" * 64
            source = directory / "source.json"
            source.write_text(json.dumps({
                "frames": [{"sha256": first}, {"sha256": second}],
                "source": {"license": "CC-BY-4.0"},
                "provisional_training_authorized": True,
            }), encoding="utf-8")
            manifest = self._manifest(source, first, second)
            manifest.update({
                "schema": "blindassist_public_video_silver_labels_v2",
                "training_execution_authorized": True,
                "training_mode": "provisional_model_supervision",
            })
            result = validate(manifest, source_manifest_path=source)
        self.assertTrue(result["training_execution_authorized"])
        self.assertTrue(result["provisional_model_supervision"])

    def test_accepts_v2_training_without_license_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, first, second = self._source(Path(temporary))
            manifest = self._manifest(source, first, second)
            manifest.update({
                "schema": "blindassist_public_video_silver_labels_v2",
                "training_execution_authorized": True,
                "training_mode": "provisional_model_supervision",
            })
            result = validate(manifest, source_manifest_path=source)
        self.assertTrue(result["training_execution_authorized"])

    def test_accepts_reviewed_wikimedia_ccby3_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = "a" * 64, "b" * 64
            source = directory / "source.json"
            source.write_text(json.dumps({
                "frames": [{"sha256": first}, {"sha256": second}],
                "source": {"license": "CC-BY-3.0", "author": "POPtravel"},
                "license_review": {
                    "status": "license_confirmed_by_youtube_review_bot",
                    "reviewed_at": "2021-03-30",
                    "file_page_url": "https://commons.wikimedia.org/wiki/File:Walking.webm",
                    "original_source_url": "https://www.youtube.com/watch?v=example",
                    "license_url": "https://creativecommons.org/licenses/by/3.0/",
                    "author": "POPtravel",
                },
                "provisional_training_authorized": True,
            }), encoding="utf-8")
            manifest = self._manifest(source, first, second)
            manifest.update({
                "schema": "blindassist_public_video_silver_labels_v2",
                "training_execution_authorized": True,
                "training_mode": "provisional_model_supervision",
            })
            result = validate(manifest, source_manifest_path=source)
        self.assertTrue(result["training_execution_authorized"])

    def test_accepts_unreviewed_ccby3_source_for_isolated_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = "a" * 64, "b" * 64
            source = directory / "source.json"
            source.write_text(json.dumps({
                "frames": [{"sha256": first}, {"sha256": second}],
                "source": {"license": "CC-BY-3.0", "author": "POPtravel"},
                "provisional_training_authorized": True,
            }), encoding="utf-8")
            manifest = self._manifest(source, first, second)
            manifest.update({
                "schema": "blindassist_public_video_silver_labels_v2",
                "training_execution_authorized": True,
                "training_mode": "provisional_model_supervision",
            })
            result = validate(manifest, source_manifest_path=source)
        self.assertTrue(result["training_execution_authorized"])

    def test_rejects_ambiguous_candidate_no_alert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, first, second = self._source(Path(temporary))
            manifest = self._manifest(source, first, second)
            manifest["episodes"][0]["negative_decision_quality"]["corridor_heading_stability"] = 0.69
            with self.assertRaisesRegex(SilverLabelError, "otherwise abstain"):
                validate(manifest, source_manifest_path=source)

    def test_accepts_v3_context_alert_with_causal_basis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = "a" * 64, "b" * 64
            source = directory / "source.json"
            source.write_text(json.dumps({
                "frames": [{"sha256": first}, {"sha256": second}],
                "source": {"license": "CC-BY-4.0"},
                "provisional_training_authorized": True,
            }), encoding="utf-8")
            manifest = self._manifest(source, first, second)
            manifest.update({
                "schema": "blindassist_public_video_silver_labels_v3",
                "training_execution_authorized": True,
                "training_mode": "provisional_model_supervision",
            })
            episode = manifest["episodes"][0]
            episode.update({
                "silver_should_alert": "candidate_alert",
                "silver_actionability": "context_only",
                "causal_evidence_basis": "past_or_current_only",
                "eventual_outcome": "safe_pass",
            })
            result = validate(manifest, source_manifest_path=source)
        self.assertTrue(result["causal_actionability_supervision"])

    def test_v3_rejects_eventual_safe_pass_as_no_alert_after_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = "a" * 64, "b" * 64
            source = directory / "source.json"
            source.write_text(json.dumps({
                "frames": [{"sha256": first}, {"sha256": second}],
                "source": {"license": "CC0-1.0"},
                "provisional_training_authorized": True,
            }), encoding="utf-8")
            manifest = self._manifest(source, first, second)
            manifest.update({
                "schema": "blindassist_public_video_silver_labels_v3",
                "training_execution_authorized": True,
                "training_mode": "provisional_model_supervision",
            })
            manifest["episodes"][0].update({
                "silver_actionability": "intervention_then_route_clear",
                "causal_evidence_basis": "past_or_current_only",
                "eventual_outcome": "safe_pass",
            })
            with self.assertRaisesRegex(SilverLabelError, "eventual outcome cannot redefine"):
                validate(manifest, source_manifest_path=source)

    def test_v3_rejects_future_evidence_basis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = "a" * 64, "b" * 64
            source = directory / "source.json"
            source.write_text(json.dumps({
                "frames": [{"sha256": first}, {"sha256": second}],
                "source": {"license": "CC0-1.0"},
                "provisional_training_authorized": True,
            }), encoding="utf-8")
            manifest = self._manifest(source, first, second)
            manifest.update({
                "schema": "blindassist_public_video_silver_labels_v3",
                "training_execution_authorized": True,
                "training_mode": "provisional_model_supervision",
            })
            manifest["episodes"][0].update({
                "silver_actionability": "no_attention",
                "causal_evidence_basis": "uses_future_route",
                "eventual_outcome": "safe_pass",
            })
            with self.assertRaisesRegex(SilverLabelError, "past_or_current_only"):
                validate(manifest, source_manifest_path=source)


if __name__ == "__main__":
    unittest.main()
