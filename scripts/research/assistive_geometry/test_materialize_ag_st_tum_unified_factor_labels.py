#!/usr/bin/env python3

import unittest

from materialize_ag_st_tum_unified_factor_labels import (
    base_validation_frame_id,
    select_tum_boundary_rows,
    validation_sha_by_base_stem,
)


class TumUnifiedFactorLabelsTest(unittest.TestCase):
    def test_base_validation_frame_id_removes_only_role(self) -> None:
        self.assertEqual(
            base_validation_frame_id("evaluation__rgbd_dataset_freiburg1_room__rgb000314"),
            "rgbd_dataset_freiburg1_room__rgb000314",
        )

    def test_validation_sha_map_requires_all_21_receipts(self) -> None:
        receipts = [
            {"frame_id": f"parent__rgb{index:06d}", "output_sha256": "A" * 64}
            for index in range(21)
        ]
        mapping = validation_sha_by_base_stem({"output_receipts": receipts})
        self.assertEqual(len(mapping), 21)

    def test_select_boundary_rows_is_source_specific(self) -> None:
        rows = [
            {
                "source": "tum_rgbd",
                "frame_id": f"fit__parent__rgb{index:06d}",
            }
            for index in range(21)
        ]
        rows.append({"source": "arkitscenes", "frame_id": "not_tum"})
        selected = select_tum_boundary_rows({"frames": rows})
        self.assertEqual(len(selected), 21)
        self.assertNotIn("not_tum", selected)


if __name__ == "__main__":
    unittest.main()
