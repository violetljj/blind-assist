"""Unit tests for the Development-only visual sidecar."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from scripts.research.dual_loop_segmentation_failure_atlas.visual_sidecar import (
    _heatmap,
    _reason_for_probe,
    _tint,
)


class VisualSidecarTest(unittest.TestCase):
    def test_tint_changes_only_selected_pixels(self) -> None:
        source = Image.new("RGB", (2, 2), (100, 100, 100))
        mask = np.array([[True, False], [False, False]])

        rendered = np.asarray(_tint(source, [(mask, (255, 0, 0), 0.5)]))

        self.assertFalse(np.array_equal(rendered[0, 0], np.array([100, 100, 100])))
        np.testing.assert_array_equal(rendered[1, 1], np.array([100, 100, 100]))

    def test_heatmap_ignores_walkable_pixels(self) -> None:
        source = Image.new("RGB", (2, 2), (80, 80, 80))
        ids = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        confidence = np.ones((2, 2), dtype=np.float32)

        rendered = np.asarray(_heatmap(source, ids, confidence))

        np.testing.assert_array_equal(rendered[0, 0], np.array([80, 80, 80]))
        np.testing.assert_array_equal(rendered[1, 1], np.array([80, 80, 80]))
        self.assertGreater(rendered[0, 1, 0], 80)
        self.assertGreater(rendered[1, 0, 0], 80)

    def test_probe_reasons_are_visual_only_descriptions(self) -> None:
        reason = _reason_for_probe("TEMPORAL:CAUSAL_2_OF_3")
        self.assertIn("prior observation", reason)
        self.assertNotIn("safe", reason.lower())
        self.assertNotIn("danger", reason.lower())


if __name__ == "__main__":
    unittest.main()
