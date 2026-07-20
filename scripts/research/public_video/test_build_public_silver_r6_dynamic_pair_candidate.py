#!/usr/bin/env python3
"""Focused tests for the r6 JtMY exploratory dynamic-pair builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_public_silver_r6_dynamic_pair_candidate as builder


class PublicSilverR6DynamicPairCandidateTest(unittest.TestCase):
    def test_silver_retains_low_alert_confidence(self) -> None:
        frames = [
            {"source_frame_index": index, "sha256": f"{index:064x}"}
            for index in (226, 262, 300, 339, 384, 429)
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "source_manifest_v2.json"
            manifest.write_text("{}\n", encoding="utf-8")
            silver = builder.build_silver(manifest, frames)
        self.assertEqual(2, len(silver["episodes"]))
        self.assertEqual(0.63, silver["episodes"][1]["confidence"])
        self.assertEqual(builder.PAIR_ID, silver["episodes"][0]["counterfactual_pair_id"])
        self.assertEqual(
            "dynamic_agent_approach",
            silver["episodes"][1]["risk_profile"]["risk_mechanism"],
        )

    def test_independent_direction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            builder.reject_independent_direction(Path("artifacts.local/experiments/secondary-corridor-causal/r6"))


if __name__ == "__main__":
    unittest.main()
