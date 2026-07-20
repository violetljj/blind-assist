#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from machine_redact_public_rgb_candidate import expanded_box, redact_regions, valid_lpd_region, valid_privacy_object_region


class MachineRedactPublicRgbCandidateTests(unittest.TestCase):
    def test_expands_and_clips_redaction_box(self) -> None:
        self.assertEqual(expanded_box((0, 0, 10, 10), width=20, height=20), (0, 0, 12, 12))
        self.assertEqual(expanded_box((15, 15, 10, 10), width=20, height=20), (13, 13, 20, 20))

    def test_redaction_changes_target_region_without_changing_outside_pixel(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        checkerboard = (np.indices((10, 10)).sum(axis=0) % 2 * 255).astype(np.uint8)
        image[5:15, 5:15] = checkerboard[:, :, None]
        redacted = redact_regions(image, [(5, 5, 10, 10)])
        self.assertFalse(np.array_equal(redacted[5:15, 5:15], image[5:15, 5:15]))
        self.assertTrue(np.array_equal(redacted[0, 0], image[0, 0]))

    def test_lpd_gate_rejects_invalid_or_excessive_regions(self) -> None:
        self.assertTrue(valid_lpd_region((10, 10, 80, 20), width=1280, height=720))
        self.assertFalse(valid_lpd_region((10, 10, 6, 6), width=1280, height=720))
        self.assertFalse(valid_lpd_region((10, 10, 600, 120), width=1280, height=720))

    def test_whole_object_gate_accepts_vehicle_but_rejects_near_full_frame(self) -> None:
        self.assertTrue(valid_privacy_object_region((0, 2, 184, 197), width=1280, height=720))
        self.assertFalse(valid_privacy_object_region((0, 0, 1200, 700), width=1280, height=720))


if __name__ == "__main__":
    unittest.main()
