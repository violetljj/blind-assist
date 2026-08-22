from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.acquire_pa3_public_observations import select_frame, select_frames, select_frames_for_anchors


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

    def test_multiview_selection_is_bounded_and_spatially_distinct(self) -> None:
        anchor = {"lat": 0.0, "lon": 0.0}
        policy = {
            "minimum_distance_m": 8.0,
            "maximum_distance_m": 45.0,
            "maximum_absolute_bearing_error_deg": 45.0,
            "panoramas_allowed": False,
            "selected_per_episode": 2,
            "minimum_viewpoint_separation_m": 5.0,
        }
        raw = [
            {"id": "best", "computed_geometry": {"coordinates": [-0.00018, 0.0]}, "compass_angle": 90.0, "captured_at": 1, "sequence": "a"},
            {"id": "too-close", "computed_geometry": {"coordinates": [-0.000181, 0.0]}, "compass_angle": 90.0, "captured_at": 2, "sequence": "a"},
            {"id": "distinct", "computed_geometry": {"coordinates": [0.0, -0.00018]}, "compass_angle": 0.0, "captured_at": 3, "sequence": "b"},
        ]
        selected, eligible = select_frames(raw, anchor, policy)
        self.assertEqual(["best", "distinct"], [row["image_id"] for row in selected])
        self.assertEqual(3, eligible)

    def test_multiview_rejects_more_than_three_frozen_views(self) -> None:
        with self.assertRaises(ValueError):
            select_frames([], {"lat": 0.0, "lon": 0.0}, {
                "minimum_distance_m": 8.0,
                "maximum_distance_m": 45.0,
                "maximum_absolute_bearing_error_deg": 45.0,
                "panoramas_allowed": False,
                "selected_per_episode": 4,
                "minimum_viewpoint_separation_m": 5.0,
            })

    def test_candidate_set_deduplicates_images_and_preserves_public_candidate_identity(self) -> None:
        policy = {
            "minimum_distance_m": 8.0,
            "maximum_distance_m": 45.0,
            "maximum_absolute_bearing_error_deg": 45.0,
            "panoramas_allowed": False,
            "selected_per_episode": 2,
            "minimum_viewpoint_separation_m": 0.0,
        }
        shared = {
            "id": "shared", "computed_geometry": {"coordinates": [0.0, -0.00015]},
            "computed_compass_angle": 0.0, "captured_at": 1, "sequence": "s",
        }
        second = {
            "id": "second", "computed_geometry": {"coordinates": [0.00015, 0.0]},
            "computed_compass_angle": 270.0, "captured_at": 2, "sequence": "s",
        }
        selected, eligible = select_frames_for_anchors([
            ({"candidate_id": "entrance-a", "lat": 0.0, "lon": 0.0}, [shared, second]),
            ({"candidate_id": "frontage-b", "lat": 0.0, "lon": 0.0}, [shared]),
        ], policy)
        self.assertEqual(2, eligible)
        self.assertEqual(["shared", "second"], [row["image_id"] for row in selected])
        self.assertEqual("entrance-a", selected[0]["public_spatial_candidate_id"])


if __name__ == "__main__":
    unittest.main()
