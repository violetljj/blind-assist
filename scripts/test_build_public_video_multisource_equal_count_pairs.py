#!/usr/bin/env python3
"""Tests for multisource equal-count path-relation generation."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

import build_public_video_multisource_equal_count_pairs as subject
import build_public_video_path_intrusion_counterfactuals as base


class MultisourceEqualCountPairsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "parents").mkdir()
        Image.new("RGB", (640, 360), (120, 130, 140)).save(self.root / "parents" / "parent.png")
        asset = Image.new("RGBA", (20, 40), (0, 0, 0, 0))
        ImageDraw.Draw(asset).polygon([(10, 0), (2, 35), (18, 35)], fill=(255, 90, 0, 255))
        asset.save(self.root / "asset.png")
        self.spec = {
            "isolation": {
                "train_only": True,
                "real_evaluation_credit": False,
                "human_truth_claimed": False,
                "training_authorized": False,
                "android_runtime_change_authorized": False,
            },
            "pairs": [{
                "pair_id": "sample",
                "parent_filename": "parent.png",
                "parent_sha256": base.sha256_file(self.root / "parents" / "parent.png"),
                "parent_source_id": "source-a",
                "parent_timestamp_ms": 1000,
                "clear_placements": [[500, 320, 60]],
                "risk_placements": [[320, 320, 60]],
            }],
        }
        self.spec_path = self.root / "dataset_spec.json"
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self, output: str = "out") -> argparse.Namespace:
        return argparse.Namespace(spec=self.spec_path, asset=self.root / "asset.png", output=self.root / output)

    def test_builds_source_bound_pixel_invariant_pair(self) -> None:
        report = subject.run(self.args())
        self.assertEqual(report["summary"]["pair_count"], 1)
        self.assertEqual(report["summary"]["source_count"], 1)
        self.assertTrue(report["summary"]["all_parent_pixel_invariants_passed"])
        rows = [json.loads(line) for line in (self.root / "out" / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual({row["source"]["parent_source_id"] for row in rows}, {"source-a"})
        self.assertEqual({row["split"] for row in rows}, {"train"})

    def test_rejects_scale_or_depth_change(self) -> None:
        self.spec["pairs"][0]["risk_placements"] = [[320, 310, 50]]
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "horizontal translation only"):
            subject.run(self.args())

    def test_rejects_parent_hash_drift(self) -> None:
        self.spec["pairs"][0]["parent_sha256"] = "0" * 64
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parent hash differs"):
            subject.run(self.args())

    def test_rejects_authorization_escalation(self) -> None:
        self.spec["isolation"]["training_authorized"] = True
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            subject.run(self.args())

    def test_rejects_independent_direction_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "independent direction"):
            subject.run(self.args("secondary-corridor-causal/out"))


if __name__ == "__main__":
    unittest.main()
