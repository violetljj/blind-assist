from __future__ import annotations

import unittest

from bbox_route_attribution_probe import (
    bbox_only_score,
    bbox_uniform_route_score,
    person_box_rows,
)


class BboxRouteAttributionProbeTest(unittest.TestCase):
    def test_all_arms_reuse_unchanged_person_confidence(self) -> None:
        detections = [
            {
                "label": "person",
                "confidence": 0.8,
                "box": [0.0, 0.0, 50.0, 100.0],
            },
            {
                "label": "chair",
                "confidence": 0.99,
                "box": [0.0, 0.0, 100.0, 100.0],
            },
        ]
        self.assertEqual(person_box_rows(detections)[0][0], 0.8)
        self.assertEqual(bbox_only_score(detections), 0.8)
        self.assertAlmostEqual(
            bbox_uniform_route_score(
                detections,
                source_width=100,
                source_height=100,
            ),
            0.4,
        )

    def test_uniform_route_clips_bbox_to_source_frame(self) -> None:
        score = bbox_uniform_route_score(
            [
                {
                    "label": "person",
                    "confidence": 0.75,
                    "box": [-50.0, -50.0, 50.0, 50.0],
                }
            ],
            source_width=100,
            source_height=100,
        )
        self.assertAlmostEqual(score, 0.1875)

    def test_empty_person_field_is_zero(self) -> None:
        detections = [
            {
                "label": "chair",
                "confidence": 0.9,
                "box": [0.0, 0.0, 100.0, 100.0],
            }
        ]
        self.assertEqual(bbox_only_score(detections), 0.0)
        self.assertEqual(
            bbox_uniform_route_score(
                detections,
                source_width=100,
                source_height=100,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
