from __future__ import annotations

import unittest

import numpy as np

from .postprocess import filter_candidate_by_class


CLASS_TO_ID = {
    "walkable": 0,
    "boundary_step_curb": 1,
    "obstacle": 2,
    "unknown_nonwalkable": 3,
}


class PostprocessTests(unittest.TestCase):
    def _config(self) -> dict[str, object]:
        return {
            "hazard_classes": ["boundary_step_curb", "obstacle"],
            "minimum_component_area_pixels": 4,
            "minimum_component_confidence_median": 0.6,
            "minimum_component_margin_median": 0.1,
            "minimum_component_bottom_fraction": 0.5,
        }

    def test_filters_area_confidence_margin_space_and_yolo_overlap(self) -> None:
        ids = np.zeros((256, 256), dtype=np.uint8)
        confidence = np.ones((256, 256), dtype=np.float32)
        margin = np.ones((256, 256), dtype=np.float32)
        yolo = np.zeros((256, 256), dtype=bool)
        ids[200:203, 10:13] = 2  # retained
        ids[200:201, 30:31] = 2  # too small
        ids[200:203, 50:53] = 2
        confidence[200:203, 50:53] = 0.5  # low confidence
        ids[200:203, 70:73] = 2
        margin[200:203, 70:73] = 0.05  # low margin
        ids[20:23, 90:93] = 2  # too high in image
        ids[200:203, 110:113] = 2
        yolo[200:203, 110:113] = True  # removed before components

        output = filter_candidate_by_class(
            ids=ids,
            confidence=confidence,
            margin=margin,
            detector_mask=yolo,
            class_to_id=CLASS_TO_ID,
            config=self._config(),
        )

        self.assertEqual(int(np.count_nonzero(output["obstacle"])), 9)
        self.assertTrue(output["obstacle"][201, 11])
        self.assertFalse(output["obstacle"][201, 51])
        self.assertFalse(output["obstacle"][201, 111])


if __name__ == "__main__":
    unittest.main()
