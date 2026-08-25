import math
import unittest

from grail_canonical_coordinates_r1c import canonicalize_scene, owner_local_coordinate


class CanonicalCoordinatesR1CTest(unittest.TestCase):
    def test_owner_yaw_canonicalizes_world_rotation(self) -> None:
        owner = {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"y": 90}}
        part = {"position": {"x": 0, "y": 1, "z": -2}}
        right, up, front = owner_local_coordinate(part, owner)
        self.assertAlmostEqual(right, 2.0)
        self.assertAlmostEqual(up, 1.0)
        self.assertAlmostEqual(front, 0.0, places=6)

    def test_native_component_owner_produces_stable_slots(self) -> None:
        objects = [
            {"objectId": "Desk|0", "objectType": "Desk", "position": {"x": 1, "y": 0, "z": 2},
             "rotation": {"y": 180}},
            {"objectId": "Desk|0___0", "objectType": "Drawer", "position": {"x": 2, "y": .2, "z": 2},
             "rotation": {"y": 180}},
            {"objectId": "Desk|0___1", "objectType": "Drawer", "position": {"x": 1, "y": .8, "z": 2},
             "rotation": {"y": 180}},
            {"objectId": "Desk|0___2", "objectType": "Drawer", "position": {"x": 0, "y": 1.4, "z": 2},
             "rotation": {"y": 180}},
        ]
        result = canonicalize_scene(objects)
        self.assertEqual(result["Desk|0___0"]["horizontal"], "LEFT")
        self.assertEqual(result["Desk|0___1"]["horizontal"], "CENTER")
        self.assertEqual(result["Desk|0___2"]["horizontal"], "RIGHT")
        self.assertEqual(result["Desk|0___0"]["vertical"], "BOTTOM")
        self.assertEqual(result["Desk|0___2"]["vertical"], "TOP")
        self.assertTrue(all(result[key]["owner_id"] == "Desk|0" for key in result if "___" in key))

    def test_missing_frame_is_not_evaluable(self) -> None:
        result = canonicalize_scene([{"objectId": "Broken|0", "objectType": "Broken"}])
        self.assertFalse(result["Broken|0"]["evaluable"])


if __name__ == "__main__":
    unittest.main()

