import unittest

from collect_grail_m1 import local_pose


class GrailM1CollectionTests(unittest.TestCase):
    def test_world_pose_is_transformed_to_query_camera_frame(self) -> None:
        camera = {"position": {"x": 1.0, "z": 2.0}, "rotation": {"y": 90.0}}
        transformed = local_pose({"x": 3.0, "z": 2.0, "rotation": 180.0}, camera)
        self.assertAlmostEqual(transformed["x"], 0.0, places=6)
        self.assertAlmostEqual(transformed["z"], 2.0, places=6)
        self.assertAlmostEqual(transformed["yaw"], 90.0, places=6)


if __name__ == "__main__":
    unittest.main()
