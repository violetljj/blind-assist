#!/usr/bin/env python3
"""Pure tests for the public-video r5 static-mechanism expansion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_public_silver_r5_static_mechanism_expansion as builder


class PublicSilverR5StaticMechanismExpansionTest(unittest.TestCase):
    def test_silver_has_one_balanced_static_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            source.write_text("{}\n", encoding="utf-8")
            silver = builder.build_silver(source)
        self.assertEqual(
            {"candidate_no_alert", "candidate_alert"},
            {row["silver_should_alert"] for row in silver["episodes"]},
        )
        self.assertEqual(
            {"static_corridor_narrowing"},
            {row["risk_profile"]["risk_mechanism"] for row in silver["episodes"]},
        )
        self.assertEqual({builder.PAIR_ID}, {row["counterfactual_pair_id"] for row in silver["episodes"]})
        negative = next(row for row in silver["episodes"] if row["silver_should_alert"] == "candidate_no_alert")
        self.assertGreaterEqual(min(negative["negative_decision_quality"].values()), 0.7)

    def test_independent_direction_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secondary-corridor-causal" / "r5"
            with self.assertRaisesRegex(ValueError, "independent model direction"):
                builder.reject_independent_direction(path)


if __name__ == "__main__":
    unittest.main()
