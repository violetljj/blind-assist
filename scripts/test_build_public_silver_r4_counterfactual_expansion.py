#!/usr/bin/env python3
"""Pure tests for the public-video r4 counterfactual builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_public_silver_r4_counterfactual_expansion as builder


class PublicSilverR4CounterfactualExpansionTest(unittest.TestCase):
    def test_expansion_creates_balanced_same_source_pair(self) -> None:
        parent = {
            "schema": "blindassist_public_video_silver_labels_v2",
            "source": {"source_id": "sanpo_real_gie8"},
            "labeler": {"provider": "old"},
            "episodes": [{"episode_id": "old"}],
        }
        result = builder.expanded_gie8_manifest(parent)
        self.assertEqual(2, len(result["episodes"]))
        self.assertEqual(
            {"candidate_no_alert", "candidate_alert"},
            {row["silver_should_alert"] for row in result["episodes"]},
        )
        self.assertEqual(
            {builder.PAIR_ID},
            {row["counterfactual_pair_id"] for row in result["episodes"]},
        )
        negative = next(row for row in result["episodes"] if row["silver_should_alert"] == "candidate_no_alert")
        self.assertGreaterEqual(min(negative["negative_decision_quality"].values()), 0.7)
        self.assertEqual("old", parent["episodes"][0]["episode_id"])

    def test_independent_direction_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secondary-corridor-causal" / "run"
            with self.assertRaisesRegex(ValueError, "independent model direction"):
                builder.reject_independent_direction(path)

    def test_prompt_hash_is_deterministic(self) -> None:
        self.assertEqual(64, len(builder.prompt_sha256()))
        self.assertEqual(builder.prompt_sha256(), builder.prompt_sha256())


if __name__ == "__main__":
    unittest.main()
