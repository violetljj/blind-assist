from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.acquire_pa3_public_observations import select_frame


class SelectFrameTest(unittest.TestCase):
    def test_selects_by_geometry_without_pixel_fields(self) -> None:
        anchor = {"lat": 0.0, "lon": 0.0}
        policy = {
            "minimum_distance_m": 8.0,
            "maximum_distance_m": 45.0,
            "maximum_absolute_bearing_error_deg": 45.0,
            "panoramas_allowed": False,
        }
        raw = [
            {"id": "worse", "computed_geometry": {"coordinates": [-0.00018, 0.0]}, "compass_angle": 80.0, "captured_at": 1, "sequence": "a", "is_pano": False},
            {"id": "best", "computed_geometry": {"coordinates": [-0.00018, 0.0]}, "compass_angle": 90.0, "captured_at": 2, "sequence": "b", "is_pano": False},
            {"id": "pano", "computed_geometry": {"coordinates": [-0.00018, 0.0]}, "compass_angle": 90.0, "captured_at": 0, "sequence": "c", "is_pano": True},
        ]
        selected, eligible = select_frame(raw, anchor, policy)
        self.assertEqual("best", selected["image_id"])
        self.assertEqual(2, eligible)

    def test_returns_none_when_no_geometry_candidate(self) -> None:
        selected, eligible = select_frame([], {"lat": 0.0, "lon": 0.0}, {
            "minimum_distance_m": 8.0, "maximum_distance_m": 45.0,
            "maximum_absolute_bearing_error_deg": 45.0, "panoramas_allowed": False,
        })
        self.assertIsNone(selected)
        self.assertEqual(0, eligible)


if __name__ == "__main__":
    unittest.main()
