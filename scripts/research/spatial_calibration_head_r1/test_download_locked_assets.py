#!/usr/bin/env python3

import tempfile
import unittest
import zipfile
from pathlib import Path

from download_locked_assets import (
    extract_named_members,
    nearest_pincam_member_names,
    pincam_members,
    png_members_by_stem,
    remove_empty_archive_tree,
    safe_delete_archive,
)


class SelectiveArchiveTest(unittest.TestCase):
    def test_only_named_members_are_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "frames.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("lowres_wide/1_0.000.png", b"first")
                bundle.writestr("lowres_wide/1_0.017.png", b"second")
            members = png_members_by_stem(archive)
            output = root / "selected"
            rows = extract_named_members(archive, [members["1_0.017"]], output)
            self.assertEqual(len(rows), 1)
            self.assertEqual((output / "1_0.017.png").read_bytes(), b"second")
            self.assertFalse((output / "1_0.000.png").exists())

    def test_pincam_members_are_timestamp_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "intrinsics.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("intrinsics/1_2.001.pincam", b"256 192 1 1 1 1")
                bundle.writestr("intrinsics/1_1.999.pincam", b"256 192 1 1 1 1")
            self.assertEqual([row[0] for row in pincam_members(archive)], [1.999, 2.001])

    def test_nearest_pincam_keeps_zip_member_separator_on_windows(self) -> None:
        candidates = [
            (1.999, "lowres_wide_intrinsics/1_1.999.pincam"),
            (2.001, "lowres_wide_intrinsics/1_2.001.pincam"),
        ]
        selected = nearest_pincam_member_names(candidates, ["1_2.0002"])
        self.assertEqual(selected, ["lowres_wide_intrinsics/1_2.001.pincam"])
        self.assertNotIn("\\", selected[0])

    def test_archive_cleanup_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_root = root / "archives"
            archive_root.mkdir()
            inside = archive_root / "one.zip"
            inside.touch()
            safe_delete_archive(inside, archive_root)
            self.assertFalse(inside.exists())
            outside = root / "outside.zip"
            outside.touch()
            with self.assertRaises(ValueError):
                safe_delete_archive(outside, archive_root)

    def test_empty_archive_tree_cleanup_allows_only_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "archives"
            (root / "video").mkdir(parents=True)
            remove_empty_archive_tree(root)
            self.assertFalse(root.exists())

            root.mkdir()
            leftover = root / "leftover.zip"
            leftover.touch()
            with self.assertRaises(ValueError):
                remove_empty_archive_tree(root)
            self.assertTrue(leftover.exists())


if __name__ == "__main__":
    unittest.main()
