#!/usr/bin/env python3
"""Unit tests for public-video silver campaign aggregation."""

from __future__ import annotations

import unittest
from pathlib import Path

from summarize_public_video_silver_campaign import CampaignSummaryError, summarize


def comparison(*, comparable: bool, agreement: bool | None, risk_config: str = "current") -> dict:
    return {
        "schema": "blindassist_public_video_silver_edge_comparison_v1",
        "silver_manifest_sha256": "a" * 64,
        "risk_config": risk_config,
        "comparison_rows": [{"episode_id": "episode", "comparable": comparable, "agreement": agreement, "edge_trigger_count": 1, "edge_unique_event_count": 1, "edge_duplicate_event_trigger_count": 0, "edge_untracked_trigger_count": 0}],
        "comparable_episode_count": int(comparable),
        "candidate_agreement_count": int(agreement is True),
        "silver_abstain_count": int(not comparable),
        "production_model_replacement_authorized": False,
    }


class CampaignSummaryTest(unittest.TestCase):
    def test_aggregates_candidates_and_abstentions_without_reinterpreting_them(self) -> None:
        result = summarize([
            (Path("positive.json"), comparison(comparable=True, agreement=True)),
            (Path("abstain.json"), comparison(comparable=False, agreement=None)),
        ])
        self.assertEqual(result["candidate_episode_count"], 1)
        self.assertEqual(result["silver_abstain_count"], 1)
        self.assertEqual(result["candidate_agreement_rate"], 1.0)
        self.assertEqual(result["risk_config"], "current")
        self.assertEqual(result["event_diagnostics"]["tracked_unique_event_count"], 2)
        self.assertFalse(result["production_model_replacement_authorized"])

    def test_rejects_inconsistent_aggregate_counts(self) -> None:
        bad = comparison(comparable=True, agreement=True)
        bad["candidate_agreement_count"] = 0
        with self.assertRaisesRegex(CampaignSummaryError, "aggregate counts"):
                summarize([(Path("bad.json"), bad)])

    def test_rejects_mixed_risk_configs(self) -> None:
        with self.assertRaisesRegex(CampaignSummaryError, "cannot mix risk_config"):
            summarize([
                (Path("default.json"), comparison(comparable=True, agreement=True)),
                (Path("experiment.json"), comparison(comparable=True, agreement=True, risk_config="animal_aware_candidate")),
            ])

    def test_rejects_inconsistent_event_diagnostics(self) -> None:
        bad = comparison(comparable=True, agreement=True)
        bad["comparison_rows"][0]["edge_unique_event_count"] = 2
        with self.assertRaisesRegex(CampaignSummaryError, "exceed trigger count"):
            summarize([(Path("bad-events.json"), bad)])


if __name__ == "__main__":
    unittest.main()
