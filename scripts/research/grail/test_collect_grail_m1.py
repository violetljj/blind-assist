import unittest

import numpy as np

from collect_grail_m1 import expanded_crop_array, local_pose


class GrailM1CollectionTests(unittest.TestCase):
    def test_world_pose_is_transformed_to_query_camera_frame(self) -> None:
        camera = {"position": {"x": 1.0, "z": 2.0}, "rotation": {"y": 90.0}}
        transformed = local_pose({"x": 3.0, "z": 2.0, "rotation": 180.0}, camera)
        self.assertAlmostEqual(transformed["x"], 0.0, places=6)
        self.assertAlmostEqual(transformed["z"], 2.0, places=6)
        self.assertAlmostEqual(transformed["yaw"], 90.0, places=6)

    def test_expanded_crop_matches_frozen_padding_rule(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        self.assertEqual(expanded_crop_array(frame, [40, 30, 50, 50]).shape, (82, 74, 3))
        self.assertEqual(expanded_crop_array(frame, [0, 0, 10, 10]).shape, (42, 42, 3))


if __name__ == "__main__":
    unittest.main()
