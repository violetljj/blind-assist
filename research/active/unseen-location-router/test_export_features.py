from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("export_features.py")
SPEC = importlib.util.spec_from_file_location("ulr_export_features", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FeatureSelectionTest(unittest.TestCase):
    def test_selects_one_frame_per_capture_group_and_never_test(self):
        rows = []
        for split in ("train", "development", "test"):
            for role in ("gallery", "query"):
                for group in range(5):
                    for frame in range(3):
                        rows.append({
                            "split": split,
                            "role": role,
                            "location_id": f"N-{split}-1",
                            "capture_group": f"IMG_{group}",
                            "source_kind": "field_capture",
                            "image_id": f"{split}-{role}-{group}-{frame}",
                        })
        selected = MODULE.select_rows(rows, gallery_limit=2, query_limit=3, salt="fixed")
        self.assertFalse(any(row["split"] == "test" for row in selected))
        self.assertEqual(10, len(selected))
        keys = {(row["split"], row["role"], row["capture_group"]) for row in selected}
        self.assertEqual(len(selected), len(keys))

    def test_query_selection_prefers_gps_capable_field_captures(self):
        rows = []
        for index in range(6):
            rows.append({
                "split": "development", "role": "query", "location_id": "N-1-1",
                "capture_group": f"group-{index}", "image_id": f"image-{index}",
                "source_kind": "field_capture" if index >= 3 else "social_media",
            })
        selected = MODULE.select_rows(rows, gallery_limit=1, query_limit=2, salt="fixed")
        self.assertEqual(2, len(selected))
        self.assertTrue(all(row["source_kind"] == "field_capture" for row in selected))


if __name__ == "__main__":
    unittest.main()
