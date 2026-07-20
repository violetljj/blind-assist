#!/usr/bin/env python3
"""Tests for RGB-only public timeline source materialization."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from materialize_public_rgb_timeline_source_manifest import TimelineSourceError, materialize


class TimelineSourceMaterializationTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[list[dict], dict]:
        images = root / "images"
        images.mkdir()
        timeline: list[dict] = []
        for timeline_index, contents in enumerate((b"first-frame", b"second-frame")):
            image = images / f"frame_{timeline_index:04d}.png"
            image.write_bytes(contents)
            timeline.append({
                "timeline_index": timeline_index,
                "source_frame_index": 120 + timeline_index * 15,
                "image_path": str(image.relative_to(root)).replace("\\", "/"),
                "image_sha256": hashlib.sha256(contents).hexdigest(),
            })
        candidate_spec = {
            "format": "blindassist_sanpo_rgb_timeline_candidate_v1",
            "human_event_truth_present": False,
            "training_execution_authorized": False,
            "source": {"session_id": "session", "license": "CC-BY-4.0"},
        }
        return timeline, candidate_spec

    def test_writes_hash_attested_non_training_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeline, candidate_spec = self._inputs(root)
            result = materialize(root, timeline, candidate_spec, root / "output" / "source.json")
        self.assertEqual(result["source_id"], "sanpo_real_session")
        self.assertEqual(result["frame_count"], 2)
        self.assertFalse(result["training_execution_authorized"])
        self.assertEqual(result["frames"][1]["source_frame_index"], 135)

    def test_rejects_tampered_timeline_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeline, candidate_spec = self._inputs(root)
            (root / timeline[0]["image_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(TimelineSourceError, "does not match"):
                materialize(root, timeline, candidate_spec, root / "source.json")

    def test_writes_v2_provisional_training_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeline, candidate_spec = self._inputs(root)
            candidate_spec = {**candidate_spec, "format": "blindassist_sanpo_rgb_timeline_candidate_v2", "provisional_training_authorized": True}
            result = materialize(root, timeline, candidate_spec, root / "output" / "source.json")
        self.assertTrue(result["provisional_training_authorized"])
        self.assertTrue(result["training_execution_authorized"])
        self.assertEqual("blindassist_public_rgb_timeline_source_manifest_v2", result["format"])


if __name__ == "__main__":
    unittest.main()
