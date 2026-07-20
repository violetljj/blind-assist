#!/usr/bin/env python3

import unittest

import numpy as np

import audit_public_video_route_conditioned_synthetic_dataset as subject


class RouteConditionedSyntheticAuditTest(unittest.TestCase):
    def test_mask_bbox_is_exact(self) -> None:
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 3:9] = 255
        self.assertEqual([3, 2, 9, 6], subject.mask_bbox(mask))

    def test_outside_bbox_equality_ignores_only_bbox(self) -> None:
        clear = np.zeros((6, 8, 3), dtype=np.uint8)
        composite = clear.copy()
        composite[2:5, 3:7] = 255
        self.assertTrue(subject.outside_bbox_equal(clear, composite, [3, 2, 7, 5]))
        composite[0, 0] = 1
        self.assertFalse(subject.outside_bbox_equal(clear, composite, [3, 2, 7, 5]))

    def test_contact_sheet_preserves_all_cells(self) -> None:
        images = [np.full((10, 20, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
        sheet = subject.contact_sheet(images, columns=3, cell_width=20)
        self.assertEqual((20, 60, 3), sheet.shape)


if __name__ == "__main__":
    unittest.main()
