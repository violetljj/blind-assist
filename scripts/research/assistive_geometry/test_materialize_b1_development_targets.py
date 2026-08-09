from __future__ import annotations

import unittest

from scripts.research.assistive_geometry.materialize_b1_development_targets import (
    select_frozen_videos,
)


class DevelopmentTargetMaterializerTests(unittest.TestCase):
    def test_selects_exact_frozen_order_without_cross_role_access(self) -> None:
        source = {
            "videos": [
                {"role": "TRAIN", "visit_id": "t", "video_id": "train"},
                *[
                    {"role": "DEVELOPMENT", "visit_id": str(index), "video_id": f"d{index}"}
                    for index in range(8)
                ],
                {"role": "CONFIRMATION", "visit_id": "c", "video_id": "confirm"},
            ]
        }
        expected = [
            {"visit_id": "6", "video_id": "d6"},
            {"visit_id": "1", "video_id": "d1"},
            {"visit_id": "7", "video_id": "d7"},
            {"visit_id": "3", "video_id": "d3"},
        ]
        selected = select_frozen_videos(source, expected)
        self.assertEqual([row["video_id"] for row in selected], ["d6", "d1", "d7", "d3"])
        self.assertTrue(all(row["role"] == "DEVELOPMENT" for row in selected))

    def test_rejects_visit_identity_drift(self) -> None:
        source = {
            "videos": [
                {"role": "DEVELOPMENT", "visit_id": str(index), "video_id": f"d{index}"}
                for index in range(8)
            ]
        }
        expected = [
            {"visit_id": "wrong", "video_id": "d0"},
            {"visit_id": "1", "video_id": "d1"},
            {"visit_id": "2", "video_id": "d2"},
            {"visit_id": "3", "video_id": "d3"},
        ]
        with self.assertRaises(ValueError):
            select_frozen_videos(source, expected)


if __name__ == "__main__":
    unittest.main()
