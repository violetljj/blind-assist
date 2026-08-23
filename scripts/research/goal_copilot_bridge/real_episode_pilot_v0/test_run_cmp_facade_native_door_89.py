import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    build_episode,
    iou,
    truth_box,
    xml_truth_box,
)


class CmpFacadeNativeDoorRunTest(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertEqual(iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(iou([0, 0, 0.25, 0.25], [0.75, 0.75, 1, 1]), 0.0)

    def test_episode_uses_only_public_inputs(self) -> None:
        item = {
            "rgb_path": "base/base/example.jpg",
            "absolute_rgb_path": "C:/example.jpg",
            "image_width": 100,
            "image_height": 200,
            "native_mask_bbox_xyxy": [30, 40, 80, 80],
            "native_xml_door": {"x": [0.2, 0.4], "y": [0.3, 0.8]},
        }
        episode = build_episode(
            item,
            [{"bbox_xyxy": [20, 60, 40, 160], "score": 0.9, "label": "door"}],
            1,
        )
        self.assertEqual(episode["goal_text"], "the door")
        self.assertEqual(episode["candidates"][0]["region"]["x_min"], 0.2)
        self.assertNotIn("native_xml_door", episode)
        self.assertEqual(truth_box(item), [0.3, 0.2, 0.8, 0.4])
        self.assertEqual(xml_truth_box(item), [0.3, 0.2, 0.8, 0.4])


if __name__ == "__main__":
    unittest.main()
