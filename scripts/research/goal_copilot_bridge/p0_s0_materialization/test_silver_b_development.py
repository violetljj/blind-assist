from __future__ import annotations

import copy
import unittest

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import silver_b_development as silver_b
from scripts.research.goal_copilot_bridge.p0_s0_materialization.test_materializer import admitted_record, bundle


def metadata() -> dict:
    return {
        "images": [
            {
                "id": "f1", "path": "C:/ignored/f1.jpg", "image_sha256": "1" * 64,
                "captured_at": 1, "sequence_id": "s1", "width": 100, "height": 100,
            },
            {
                "id": "f2", "path": "C:/ignored/f2.jpg", "image_sha256": "2" * 64,
                "captured_at": 2, "sequence_id": "s2", "width": 100, "height": 100,
            },
        ]
    }


def development_bundle(record: dict | None = None) -> dict:
    value = bundle(record)
    for rank, candidate in enumerate(value["records"][0]["candidates"], start=1):
        candidate.update({
            "proposal_rank": rank,
            "proposal_label": "entrance",
            "proposal_score": 0.5,
            "proposal_score_semantics": "MODEL_PROPOSAL_RANKING_SCORE_NOT_TRUTH",
        })
    return value


class SilverBDevelopmentTest(unittest.TestCase):
    def test_primary_parent_is_reused_only_at_lower_b_authority(self) -> None:
        source = development_bundle()
        parent = materializer.materialize_bundle(source)
        report = silver_b.export_development_cohort(source, metadata(), parent, data_role="CONSUMED_DEVELOPMENT")
        self.assertEqual(2, report["episode_count"])
        self.assertTrue(all(item["development_quality_class"] == "SILVER_B_MAP_GEOMETRY" for item in report["episodes"]))
        self.assertTrue(all(item["parent_quality_class"] == "SILVER_A_PRIMARY" for item in report["episodes"]))
        self.assertTrue(all(item["goal_reference_truth"]["resolution"] == "AMBIGUOUS" for item in report["episodes"]))
        self.assertTrue(all(item["goal_reference_truth"]["valid_target_instance_ids"] == [] for item in report["episodes"]))
        self.assertTrue(all(
            candidate["same_physical_entrance_identity"] == "NOT_ESTABLISHED"
            for episode in report["episodes"] for candidate in episode["weak_positive_candidates"]
        ))
        self.assertIn("GROUNDING_DINO_RECALL_OR_PRECISION", report["forbidden_claims"])

    def test_existing_secondary_parent_exports_and_rejection_does_not(self) -> None:
        secondary_record = admitted_record()
        secondary_record["frames"][1]["camera_position"] = copy.deepcopy(secondary_record["frames"][0]["camera_position"])
        secondary_record["candidates"][1]["frame_id"] = "f1"
        secondary_source = development_bundle(secondary_record)
        secondary_parent = materializer.materialize_bundle(secondary_source)
        report = silver_b.export_development_cohort(secondary_source, metadata(), secondary_parent, data_role="DEVELOPMENT")
        self.assertEqual(1, report["episode_count"])
        self.assertEqual(2, report["candidate_count"])

        rejected_record = admitted_record()
        rejected_record["conflicts"] = ["conflict"]
        rejected_source = development_bundle(rejected_record)
        rejected_parent = materializer.materialize_bundle(rejected_source)
        rejected = silver_b.export_development_cohort(rejected_source, metadata(), rejected_parent, data_role="DEVELOPMENT")
        self.assertEqual(0, rejected["episode_count"])

    def test_parent_hash_mismatch_fails_closed(self) -> None:
        source = development_bundle()
        parent = materializer.materialize_bundle(source)
        parent["input_sha256"] = "0" * 64
        with self.assertRaises(silver_b.SilverBDevelopmentError):
            silver_b.export_development_cohort(source, metadata(), parent, data_role="DEVELOPMENT")


if __name__ == "__main__":
    unittest.main()
