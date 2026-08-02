from __future__ import annotations

import unittest

from inspect_thor_magni_archive import _member_kind, _select_archive


class InspectThorMagniArchiveTest(unittest.TestCase):
    def test_member_kind_is_metadata_only(self) -> None:
        self.assertEqual(_member_kind("capture/scene.mp4"), "VIDEO")
        self.assertEqual(_member_kind("motion/pose.csv"), "TABULAR_OR_JSON")
        self.assertEqual(_member_kind("lidar/cloud.pcd"), "POINT_CLOUD")

    def test_archive_selection_requires_single_named_zip(self) -> None:
        selected = _select_archive({
            "files": [{
                "key": "THOR_MAGNI.zip",
                "size": 10,
                "checksum": "md5:abc",
                "links": {"self": "https://example.invalid/archive"},
            }]
        })
        self.assertEqual(selected["key"], "THOR_MAGNI.zip")
        with self.assertRaises(ValueError):
            _select_archive({"files": []})


if __name__ == "__main__":
    unittest.main()
