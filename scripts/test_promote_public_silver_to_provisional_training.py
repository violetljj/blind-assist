#!/usr/bin/env python3
"""Tests for immutable v1 to provisional v2 public-silver promotion."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from promote_public_silver_to_provisional_training import PromotionError, promote


class PublicSilverPromotionTest(unittest.TestCase):
    def _inputs(self, root: Path, *, license: str = "CC-BY-4.0") -> tuple[Path, Path]:
        images = root / "images"
        images.mkdir()
        (images / "first.png").write_bytes(b"first")
        (images / "second.png").write_bytes(b"second")
        first = hashlib.sha256(b"first").hexdigest()
        second = hashlib.sha256(b"second").hexdigest()
        source = root / "source.json"
        source.write_text(json.dumps({
            "format": "blindassist_public_rgb_timeline_source_manifest_v1",
            "source": {"license": license},
            "frames": [{"file_name": "first.png", "sha256": first}, {"file_name": "second.png", "sha256": second}],
            "human_event_truth_present": False,
            "privacy_audit_required": True,
            "training_execution_authorized": False,
        }), encoding="utf-8")
        silver = root / "silver.json"
        silver.write_text(json.dumps({
            "schema": "blindassist_public_video_silver_labels_v1",
            "source": {"source_id": "public", "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "human_event_truth_present": False, "privacy_audit_required": True},
            "labeler": {"provider": "openai", "model": "gpt-5", "prompt_id": "prompt", "prompt_sha256": "c" * 64, "review_mode": "multiframe_temporal"},
            "episodes": [{"episode_id": "episode", "silver_should_alert": "candidate_alert", "confidence": 0.8, "evidence_frame_sha256": [first, second], "uncertainty_reasons": ["sparse samples"], "risk_profile": {"corridor_relation": "center_or_blocking"}}],
            "training_execution_authorized": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "production_model_replacement_authorized": False,
        }), encoding="utf-8")
        return silver, source

    def test_creates_separate_validated_v2_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silver, source = self._inputs(root)
            receipt = promote(legacy_silver_path=silver, legacy_source_path=source, output_root=root / "v2")
            promoted = json.loads((root / "v2" / "silver_labels_v2.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["validation"]["training_execution_authorized"])
        self.assertEqual("blindassist_public_video_silver_labels_v2", promoted["schema"])
        self.assertTrue(promoted["training_execution_authorized"])
        self.assertEqual(str((root / "images").resolve()), receipt["image_root"])

    def test_rejects_source_without_bound_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silver, source = self._inputs(root)
            (root / "images").rename(root / "moved-images")
            with self.assertRaisesRegex(PromotionError, "image directory is missing"):
                promote(legacy_silver_path=silver, legacy_source_path=source, output_root=root / "v2")

    def test_rejects_non_ccby_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silver, source = self._inputs(root, license="CC-BY-NC-4.0")
            with self.assertRaisesRegex(PromotionError, "CC-BY-4.0 or attested CC0-1.0"):
                promote(legacy_silver_path=silver, legacy_source_path=source, output_root=root / "v2")

    def test_accepts_cc0_source_with_bound_candidate_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset" / "machine_redacted_rgb"
            root.mkdir(parents=True)
            silver, source = self._inputs(root, license="legacy-no-inline-license")
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            source_payload.pop("source")
            source_payload["source_id"] = "cc0-source"
            source.write_text(json.dumps(source_payload), encoding="utf-8")
            silver_payload = json.loads(silver.read_text(encoding="utf-8"))
            silver_payload["source"]["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            silver.write_text(json.dumps(silver_payload), encoding="utf-8")
            (root.parent / "public_candidate_receipt.json").write_text(json.dumps({"source_id": "cc0-source", "expected_license": "CC0 1.0"}), encoding="utf-8")
            receipt = promote(legacy_silver_path=silver, legacy_source_path=source, output_root=root / "v2")
            promoted_source = json.loads((root / "v2" / "source_manifest_v2.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["validation"]["training_execution_authorized"])
        self.assertEqual("CC0-1.0", promoted_source["source"]["license"])


if __name__ == "__main__":
    unittest.main()
