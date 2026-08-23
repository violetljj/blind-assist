from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import visible_identity_probe as sut


class VisibleIdentityProbeTest(unittest.TestCase):
    def test_provider_prompt_precedes_variadic_image_flags(self) -> None:
        command = sut._provider_command(Path("codex.exe"), Path("case"), Path("schema.json"), "PROMPT")
        self.assertLess(command.index("PROMPT"), command.index("--image"))

    def test_native_instances_selects_largest_polygon_per_object(self) -> None:
        annotation = {
            "objects": [{"name": "chair: cub"}, {"name": "chair: cub 2"}],
            "frames": [
                {
                    "polygon": [
                        {"object": 0, "x": [10, 20, 20], "y": [10, 10, 20]},
                        {"object": 0, "x": [10, 110, 110], "y": [10, 10, 110]},
                        {"object": 1, "x": [300, 400, 400], "y": [100, 100, 200]},
                    ]
                }
            ],
        }
        instances = sut.native_instances_for_frame(annotation, "sequence", 0)
        self.assertEqual(2, len(instances))
        self.assertEqual([10 / 640, 10 / 480, 110 / 640, 110 / 480], instances[0]["bbox_xyxy_normalized"])
        self.assertEqual("chair: cub", instances[1]["normalized_label"])

    def test_parse_prediction_enforces_decision_region_consistency(self) -> None:
        found = sut.parse_prediction({"decision": "FOUND", "region_xyxy_norm_1000": [100, 200, 300, 400]})
        self.assertEqual([0.1, 0.2, 0.3, 0.4], found["region_xyxy_normalized"])
        self.assertEqual("ABSTAIN", sut.parse_prediction({"decision": "ABSTAIN", "region_xyxy_norm_1000": None})["decision"])
        with self.assertRaises(sut.ProbeError):
            sut.parse_prediction({"decision": "FOUND", "region_xyxy_norm_1000": [300, 200, 100, 400]})

    def test_assignment_is_threshold_free_center_containment(self) -> None:
        private_input = {
            "target_physical_instance_id": "target",
            "target_normalized_label": "chair",
            "native_instances": [
                {
                    "native_object_id": 1,
                    "physical_instance_id": "target",
                    "normalized_label": "chair",
                    "bbox_xyxy_normalized": [0.1, 0.1, 0.3, 0.3],
                    "bbox_area_fraction": 0.04,
                },
                {
                    "native_object_id": 2,
                    "physical_instance_id": "distractor",
                    "normalized_label": "chair",
                    "bbox_xyxy_normalized": [0.5, 0.1, 0.7, 0.3],
                    "bbox_area_fraction": 0.04,
                },
                {
                    "native_object_id": 3,
                    "physical_instance_id": "table",
                    "normalized_label": "table",
                    "bbox_xyxy_normalized": [0.1, 0.5, 0.4, 0.8],
                    "bbox_area_fraction": 0.09,
                },
            ],
        }
        self.assertEqual("SAME_INSTANCE", sut.assign_region([0.12, 0.12, 0.28, 0.28], private_input)["identity_outcome"])
        self.assertEqual("SAME_CLASS_DISTRACTOR", sut.assign_region([0.52, 0.12, 0.68, 0.28], private_input)["identity_outcome"])
        self.assertEqual("UNRELATED_OBJECT", sut.assign_region([0.12, 0.52, 0.38, 0.78], private_input)["identity_outcome"])
        self.assertEqual("BACKGROUND", sut.assign_region([0.8, 0.8, 0.9, 0.9], private_input)["identity_outcome"])

    def test_summary_reports_counts_and_three_view_stability(self) -> None:
        rows = [
            {"case_id": "a", "observation_id": f"a-{index}", "identity_outcome": "SAME_INSTANCE"}
            for index in range(3)
        ] + [
            {"case_id": "b", "observation_id": "b-1", "identity_outcome": "SAME_CLASS_DISTRACTOR"},
            {"case_id": "b", "observation_id": "b-2", "identity_outcome": "ABSTAIN"},
            {"case_id": "b", "observation_id": "b-3", "identity_outcome": "BACKGROUND"},
        ]
        summary = sut.summarize_rows(rows)
        self.assertEqual(5, summary["found_count"])
        self.assertEqual(3, summary["same_instance_correct_count"])
        self.assertEqual(1, summary["same_class_distractor_count"])
        self.assertEqual(1, summary["abstain_count"])
        self.assertEqual(1, summary["three_view_stable_same_instance_episode_count"])


if __name__ == "__main__":
    unittest.main()
