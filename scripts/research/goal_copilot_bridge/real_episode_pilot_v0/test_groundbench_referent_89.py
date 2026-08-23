import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.freeze_groundbench_referent_89 import (
    coco_transport_url,
    expression_from_question,
    polygon_bbox,
    select_rows,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_referent_89 import (
    build_episode,
    expression_prompt,
)


class GroundBenchReferent89Test(unittest.TestCase):
    def test_expression_and_prompt_do_not_include_target(self) -> None:
        expression = expression_from_question(
            "<image>Please provide the 64 points polygon coordinate of the region this sentence describes: Red car left"
        )
        self.assertEqual(expression, "Red car left")
        self.assertEqual(expression_prompt(expression), "red car left .")
        self.assertEqual(
            coco_transport_url("https://images.cocodataset.org/train2014/example.jpg"),
            "http://images.cocodataset.org/train2014/example.jpg",
        )

    def test_polygon_bbox(self) -> None:
        polygon = []
        for index in range(64):
            polygon.extend([10 if index % 2 == 0 else 30, 20 if index % 2 == 0 else 40])
        self.assertEqual(polygon_bbox(polygon, 100, 100), [10.0, 20.0, 30.0, 40.0])

    def test_selection_is_deterministic_and_filters_distractors(self) -> None:
        def row(index: int, group: str, distractors: int) -> dict:
            return {
                "image": f"train2014/{index}.jpg",
                "annotations": {
                    "dataset": "refcoco", "image_id": index, "ann_id": 1000 + index,
                    "category_group": group, "same_class_distractors": distractors,
                },
            }
        rows = [row(index, "vehicle", 1) for index in range(100)] + [row(100, "person", 2), row(101, "vehicle", 0)]
        first, count = select_rows(rows)
        second, _ = select_rows(list(reversed(rows)))
        self.assertEqual(count, 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 89)

    def test_episode_contains_public_goal_and_candidates_only(self) -> None:
        item = {
            "observation_id": "groundbench-ref-001", "goal_text": "the red car on the left",
            "absolute_rgb_path": "C:/image.jpg", "image_width": 100, "image_height": 50,
        }
        episode = build_episode(item, [{
            "bbox_xyxy": [10, 5, 50, 25], "score": 0.8, "label": "red car",
        }], 1)
        self.assertEqual(episode["goal_text"], item["goal_text"])
        self.assertEqual(episode["candidates"][0]["region"]["x_min"], 0.1)
        self.assertNotIn("native_mask_bbox_xyxy", episode)


if __name__ == "__main__":
    unittest.main()
