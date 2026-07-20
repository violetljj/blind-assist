#!/usr/bin/env python3
"""Focused tests for the isolated Wikimedia r7 builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_public_silver_r7_wikimedia_counterfactuals as builder


class PublicSilverR7WikimediaCounterfactualsTest(unittest.TestCase):
    def test_silver_contains_two_confidence_qualified_mechanism_pairs(self) -> None:
        frames = [
            {
                "source_timestamp_ms": int(round(timestamp * 1000.0)),
                "sha256": f"{index + 1:064x}",
            }
            for index, (_, timestamp) in enumerate(builder.FRAME_SPECS)
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "source_manifest_v2.json"
            manifest.write_text("{}\n", encoding="utf-8")
            silver = builder.build_silver(manifest, frames)
        self.assertEqual(4, len(silver["episodes"]))
        pairs = {
            episode["counterfactual_pair_id"]
            for episode in silver["episodes"]
        }
        mechanisms = {
            episode["risk_profile"]["risk_mechanism"]
            for episode in silver["episodes"]
        }
        self.assertEqual({builder.DYNAMIC_PAIR_ID, builder.STATIC_PAIR_ID}, pairs)
        self.assertEqual(
            {"dynamic_agent_approach", "static_corridor_narrowing"},
            mechanisms,
        )
        self.assertGreaterEqual(
            min(episode["confidence"] for episode in silver["episodes"]),
            0.65,
        )

    def test_independent_direction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            builder.reject_independent_direction(
                Path("artifacts.local/experiments/secondary-corridor-causal/r7")
            )


if __name__ == "__main__":
    unittest.main()
