"""Targeted tests for missed crossings, rounded corners and proxy class contrast."""
import math
import unittest

import street_scenarios as scenarios


class SweptTruthTests(unittest.TestCase):
    def disc(self, x, y, height=1.75, base=0):
        return {"id": "other", "shape": "disc", "x_m": x, "y_m": y,
                "radius_m": 0.28, "height_m": height, "base_m": base}

    def test_complete_crossing_between_disjoint_endpoints(self):
        ego = {"x_m": 0, "y_m": 0}
        hit = scenarios.contact_between(ego, ego, self.disc(-2, 0), self.disc(2, 0))
        self.assertEqual(hit["contact_type"], "BODY_COLLISION_PROXY")
        self.assertAlmostEqual(hit["first_fraction"], (2 - 0.56) / 4)

    def test_height_and_scene_surface(self):
        ego = {"x_m": 0, "y_m": 0}
        for height in (0.004, 0.006):
            actor = self.disc(0, 0, height, base=0.27)
            self.assertIsNone(scenarios.contact_between(ego, ego, actor, actor, 0.27))
        low = self.disc(0, 0, 0.12, base=0.27)
        self.assertEqual(scenarios.contact_between(ego, ego, low, low, 0.27)["contact_type"],
                         "FOOT_TRIP_PROXY")
        overhead = self.disc(0, 0, 0.2, base=2.1)
        self.assertIsNone(scenarios.contact_between(ego, ego, overhead, overhead, 0.27))

    def test_rounded_box_corner_is_not_expanded_square(self):
        box = {"id": "box", "shape": "box", "half_extents_m": [1, 1],
               "x_m": 0, "y_m": 0, "base_m": 0, "height_m": 2, "yaw_deg": 0}
        miss = {"x_m": 1.25, "y_m": 1.25}
        self.assertIsNone(scenarios.contact_between(miss, miss, box, box))
        tangent = {"x_m": 1 + 0.28 / math.sqrt(2), "y_m": 1 + 0.28 / math.sqrt(2)}
        self.assertIsNotNone(scenarios.contact_between(tangent, tangent, box, box))

    def test_rotated_box_sweep(self):
        box = {"id": "box", "shape": "box", "half_extents_m": [1, 0.1],
               "x_m": 0, "y_m": 0, "height_m": 2, "yaw_deg": 90}
        start, end = {"x_m": -2, "y_m": 0}, {"x_m": 2, "y_m": 0}
        hit = scenarios.contact_between(start, end, box, box)
        self.assertAlmostEqual(hit["first_fraction"], (2 - 0.38) / 4)

    def test_script_breakpoint_keeps_out_and_back_contact(self):
        actor = self.disc(-2, 0)
        actor["waypoints"] = [[0, -2, 0], [1, 2, 0], [2, -2, 0]]
        spec = {"actors": [actor]}
        ego = {"x_m": 0, "y_m": 0}
        contacts = scenarios.contacts_for_step(spec, 0, 2, ego, ego)
        self.assertEqual(len(contacts), 1)
        self.assertAlmostEqual(contacts[0]["time_s"], (2 - 0.56) / 4)

    def test_all_predeclared_contact_nearmiss_controls(self):
        result = scenarios.validate_catalog()
        self.assertEqual(len(result["scenarios"]), 8)
        self.assertTrue(result["passed"], result)
        for spec in scenarios.scenario_catalog():
            # Measured ground translation cannot alter contact truth.
            ego = spec["ego_start"]
            end = dict(ego, x_m=8)
            shifted = scenarios.contacts_for_step(spec, 0, 8, ego, end, 0.27)
            self.assertEqual(bool(shifted), spec["expected_open_loop_contact"])


if __name__ == "__main__":
    unittest.main()
