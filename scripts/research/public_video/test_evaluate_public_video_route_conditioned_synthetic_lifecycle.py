#!/usr/bin/env python3

import unittest

import evaluate_public_video_route_conditioned_synthetic_lifecycle as subject


class RouteConditionedSyntheticLifecycleTest(unittest.TestCase):
    def test_build_sequences_keeps_distance_order(self) -> None:
        generation = [{"id": f"i{index}", "attributes": {"distance_index": index}} for index in range(3)]
        examples = [{"example_id": f"e{index}", "image_id": f"i{index}", "parent_source_id": "s",
                     "asset_name": "a", "obstacle_direction": "LEFT", "route_choice": "LEFT",
                     "route_blocked": value} for index, value in enumerate((False, True, True))]
        rows = subject.build_sequences(examples, generation, [0, 1, 1], consecutive=2)
        self.assertEqual(1, len(rows))
        self.assertTrue(rows[0]["expected_intervention_open"])
        self.assertTrue(rows[0]["predicted_intervention_open"])

    def test_nonconsecutive_predictions_do_not_open(self) -> None:
        generation = [{"id": f"i{index}", "attributes": {"distance_index": index}} for index in range(3)]
        examples = [{"example_id": f"e{index}", "image_id": f"i{index}", "parent_source_id": "s",
                     "asset_name": "a", "obstacle_direction": "LEFT", "route_choice": "LEFT",
                     "route_blocked": True} for index in range(3)]
        rows = subject.build_sequences(examples, generation, [1, 0, 1], consecutive=2)
        self.assertFalse(rows[0]["predicted_intervention_open"])


if __name__ == "__main__":
    unittest.main()
