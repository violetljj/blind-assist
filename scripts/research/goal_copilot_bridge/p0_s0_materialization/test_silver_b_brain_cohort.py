from __future__ import annotations

import copy
import unittest

from scripts.research.goal_copilot_bridge.p0_s0_materialization import silver_b_brain_cohort as brain_cohort


def source_inputs() -> tuple[dict, dict, dict]:
    episodes = []
    images = []
    receipt_images = []
    for index in range(3):
        frame_id = f"f{index + 1}"
        episode_id = f"silver-b-{frame_id}"
        episodes.append({
            "episode_id": episode_id,
            "frame_id": frame_id,
            "captured_at": index * 1000,
            "image_path": f"C:/ignored/{frame_id}.jpg",
            "image_sha256": str(index + 1) * 64,
        })
        images.append({
            "id": frame_id,
            "width": 100,
            "height": 100,
            "image_sha256": str(index + 1) * 64,
        })
        receipt_images.append({
            "image_id": frame_id,
            "image_sha256": str(index + 1) * 64,
            "proposals": [
                {"bbox_xyxy": [10, 10, 40, 80], "label": "entrance", "score": 0.8},
                {"bbox_xyxy": [60, 10, 90, 80], "label": "entrance", "score": 0.7},
            ],
        })
    return (
        {"report_sha256": "a" * 64, "development_quality_class": "SILVER_B_MAP_GEOMETRY", "episodes": episodes},
        {"images": images},
        {"receipt_sha256": "b" * 64, "images": receipt_images},
    )


def annotation(episode_id: str, resolution: str) -> dict:
    source_episode_id = episode_id
    frame_id = source_episode_id.removeprefix("silver-b-")
    valid_targets = []
    if resolution in {"UNIQUE", "SET_VALUED"}:
        valid_targets.append({
            "target_instance_id": f"{frame_id}-east",
            "candidate_id": f"gdino-{frame_id}-001",
            "region_normalized_xyxy": [0.12, 0.12, 0.38, 0.78],
        })
    if resolution == "SET_VALUED":
        valid_targets.append({
            "target_instance_id": f"{frame_id}-west",
            "candidate_id": f"gdino-{frame_id}-002",
            "region_normalized_xyxy": [0.6, 0.1, 0.9, 0.8],
        })
    return {
        "episode_id": episode_id,
        "source_episode_id": source_episode_id,
        "reviewer_id": "independent-reviewer-1",
        "reviewed_at": "2026-08-21T00:00:00Z",
        "evidence_note": "Full-frame visual review; proposal scores were not used as truth.",
        "target_name": "Example Building",
        "goal_text": "Find the entrance to Example Building.",
        "resolution": resolution,
        "valid_targets": valid_targets,
        "distractors": [],
        "target_min_side_px": 30,
        "visibility_fraction": 0.9,
        "text_support": "NONE",
        "scene_condition": {
            "target_size": "MEDIUM" if resolution != "AMBIGUOUS" else "ABSENT",
            "visibility": "FULL" if resolution != "AMBIGUOUS" else "ABSENT",
            "entrance_count": "MULTIPLE",
            "same_class_distractor": True,
            "illumination": "DAYLIGHT",
            "view_angle": "FRONTAL",
        },
    }


class SilverBBrainCohortTest(unittest.TestCase):
    def test_three_referent_semantics_validate_against_frozen_evaluator(self) -> None:
        silver, metadata, receipt = source_inputs()
        annotations = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "episodes": [
                annotation("silver-b-f1", "UNIQUE"),
                annotation("silver-b-f2", "SET_VALUED"),
                annotation("silver-b-f3", "AMBIGUOUS"),
            ],
        }
        report = brain_cohort.build_brain_cohort(silver, metadata, receipt, annotations)
        self.assertEqual(3, report["episode_count"])
        self.assertEqual(3, report["unique_source_frame_count"])
        self.assertEqual({"AMBIGUOUS": 1, "SET_VALUED": 1, "UNIQUE": 1}, report["resolution_counts"])
        self.assertTrue(all(len(item["candidates"]) == 2 for item in report["episodes"]))
        self.assertEqual(
            0.12,
            report["episodes"][0]["evaluator_episode"]["acceptable_spatial_regions"][0]["x_min"],
        )
        ambiguous = report["episodes"][2]["evaluator_episode"]
        self.assertFalse(ambiguous["target_visible"])
        self.assertEqual([], ambiguous["valid_target_instances"])

    def test_multiple_goals_share_one_source_frame_without_inflating_frame_count(self) -> None:
        silver, metadata, receipt = source_inputs()
        generic = annotation("silver-b-f1", "SET_VALUED")
        generic["episode_id"] = "goal-f1-any-entrance"
        specific = annotation("silver-b-f1", "UNIQUE")
        specific["episode_id"] = "goal-f1-left-entrance"
        specific["goal_text"] = "Find the left entrance to Example Building."
        annotations = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "episodes": [generic, specific],
        }
        report = brain_cohort.build_brain_cohort(silver, metadata, receipt, annotations)
        self.assertEqual(2, report["episode_count"])
        self.assertEqual(1, report["unique_source_frame_count"])
        self.assertEqual("silver-b-f1", report["episodes"][0]["source_episode_id"])

    def test_truth_is_never_inferred_and_unknown_candidate_fails_closed(self) -> None:
        silver, metadata, receipt = source_inputs()
        reviewed = annotation("silver-b-f1", "UNIQUE")
        reviewed["valid_targets"][0]["candidate_id"] = "missing"
        annotations = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "episodes": [reviewed],
        }
        with self.assertRaises(brain_cohort.BrainCohortError):
            brain_cohort.build_brain_cohort(silver, metadata, receipt, annotations)

    def test_aggregate_preserves_semantics_and_counts_unique_frames(self) -> None:
        silver, metadata, receipt = source_inputs()
        first_annotations = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "episodes": [annotation("silver-b-f1", "UNIQUE")],
        }
        second_annotations = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "episodes": [annotation("silver-b-f2", "AMBIGUOUS")],
        }
        parts = [
            brain_cohort.build_brain_cohort(silver, metadata, receipt, first_annotations),
            brain_cohort.build_brain_cohort(silver, metadata, receipt, second_annotations),
        ]
        report = brain_cohort.aggregate_brain_cohorts(parts)
        self.assertEqual(2, report["episode_count"])
        self.assertEqual(2, report["unique_source_frame_count"])
        self.assertEqual({"AMBIGUOUS": 1, "SET_VALUED": 0, "UNIQUE": 1}, report["resolution_counts"])

    def test_proposal_hash_drift_fails_closed(self) -> None:
        silver, metadata, receipt = source_inputs()
        drifted = copy.deepcopy(receipt)
        drifted["images"][0]["image_sha256"] = "0" * 64
        annotations = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "episodes": [annotation("silver-b-f1", "UNIQUE")],
        }
        with self.assertRaises(brain_cohort.BrainCohortError):
            brain_cohort.build_brain_cohort(silver, metadata, drifted, annotations)


if __name__ == "__main__":
    unittest.main()
