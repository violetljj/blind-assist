#!/usr/bin/env python3
"""Tests for the inference-only public-video device-set builder."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_public_video_edge_inference_set import InferenceSetError, build


class InferenceSetBuilderTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path, dict, dict]:
        images = root / "source-images"
        images.mkdir()
        values = []
        for index, contents in enumerate((b"one", b"two")):
            path = images / f"frame_{index:04d}.png"
            path.write_bytes(contents)
            values.append({"frame_index": index, "source_frame_index": 100 + index, "file_name": path.name, "sha256": hashlib.sha256(contents).hexdigest()})
        source_path = root / "source.json"
        source = {"frames": values}
        source_path.write_text(json.dumps(source), encoding="utf-8")
        silver_path = root / "silver.json"
        silver = {
            "schema": "blindassist_public_video_silver_labels_v1",
            "source": {"source_id": "public", "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "human_event_truth_present": False, "privacy_audit_required": True},
            "labeler": {"provider": "openai", "model": "gpt-5", "prompt_id": "p", "prompt_sha256": "c" * 64, "review_mode": "multiframe_temporal"},
            "episodes": [{"episode_id": "episode", "silver_should_alert": "candidate_no_alert", "confidence": 0.7, "evidence_frame_sha256": [values[0]["sha256"], values[1]["sha256"]], "uncertainty_reasons": [], "risk_profile": {"corridor_relation": "outside_or_nonblocking"}, "negative_decision_quality": {"corridor_heading_stability": 0.8, "near_field_visibility": 0.8, "corridor_clearance": 0.8, "near_field_lateral_intrusion_absent": 0.8}}],
            "training_execution_authorized": False, "calibration_authorized": False, "blind_evaluation_authorized": False, "production_model_replacement_authorized": False,
        }
        silver_path.write_text(json.dumps(silver), encoding="utf-8")
        return silver_path, source_path, images, silver, source

    def test_materializes_only_hash_attested_evidence_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silver_path, source_path, images, silver, source = self._inputs(root)
            output = root / "prepared"
            result = build(silver, source, silver_path=silver_path, source_path=source_path, source_images_dir=images, output_root=output)
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(result["ok"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode_id"], "episode")
        self.assertEqual(rows[0]["timeline_frame_index"], 0)
        self.assertEqual(rows[0]["source_frame_index"], 100)

    def test_rejects_source_image_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silver_path, source_path, images, silver, source = self._inputs(root)
            (images / "frame_0000.png").write_bytes(b"changed")
            with self.assertRaisesRegex(InferenceSetError, "does not match"):
                build(silver, source, silver_path=silver_path, source_path=source_path, source_images_dir=images, output_root=root / "prepared")


if __name__ == "__main__":
    unittest.main()
