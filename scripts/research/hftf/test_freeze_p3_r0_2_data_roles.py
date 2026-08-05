#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("freeze_p3_r0_2_data_roles.py")
SPEC = importlib.util.spec_from_file_location("freeze_roles", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FreezeRolesUnitTest(unittest.TestCase):
    def test_selects_nonoverlapping_four_frame_clips(self) -> None:
        rows = [{"timestamp_ns": 1_000_000_000 + index * 100_000_000, "frame_id": str(index)} for index in range(9)]
        clips = MODULE.select_nonoverlap_clips(rows, None)
        self.assertEqual(2, len(clips))
        self.assertEqual(["0", "1", "2", "3"], [row["frame_id"] for row in clips[0]])
        self.assertEqual(["4", "5", "6", "7"], [row["frame_id"] for row in clips[1]])

    def test_over_gap_window_is_not_admitted(self) -> None:
        rows = [{"timestamp_ns": value, "frame_id": str(index)} for index, value in enumerate((0, 100_000_000, 700_000_000, 800_000_000, 900_000_000, 1_000_000_000))]
        clips = MODULE.select_nonoverlap_clips(rows, None)
        self.assertEqual(1, len(clips))
        self.assertEqual(["2", "3", "4", "5"], [row["frame_id"] for row in clips[0]])

    def test_missing_depth_reference_is_not_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sequence = Path(temporary) / "sequence"
            (sequence / "rgb").mkdir(parents=True)
            (sequence / "depth").mkdir()
            rgb_rows, depth_rows = [], []
            for index in range(5):
                timestamp = 1.0 + index * 0.1
                rgb = f"rgb/{index}.png"
                depth = f"depth/{index}.png"
                (sequence / rgb).write_bytes(f"rgb{index}".encode())
                if index != 2:
                    (sequence / depth).write_bytes(f"depth{index}".encode())
                rgb_rows.append(f"{timestamp:.1f} {rgb}")
                depth_rows.append(f"{timestamp:.1f} {depth}")
            (sequence / "rgb.txt").write_text("\n".join(rgb_rows), encoding="utf-8")
            (sequence / "depth.txt").write_text("\n".join(depth_rows), encoding="utf-8")
            paired = MODULE.paired_rgb_rows(sequence)
            self.assertEqual(4, len(paired))
            self.assertNotIn("sequence:1.20000", {row["frame_id"] for row in paired})

    def test_output_directory_overwrite_is_refused_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "output directory already exists"):
                MODULE.require(not output.exists(), f"output directory already exists: {output}")

    def test_missing_frozen_parent_fails_closed(self) -> None:
        manifests = {
            "train": {"clips": [{"parent_id": "train-a"}]},
            "validation": {"clips": []},
            "public_holdout": {"clips": [{"parent_id": "holdout-a"}]},
        }
        roles = {"train": ["train-a"], "validation": ["validation-a"], "public_holdout": ["holdout-a"]}
        with self.assertRaisesRegex(ValueError, "validation clip parent coverage mismatch"):
            MODULE.require_parent_coverage(manifests, roles)


if __name__ == "__main__":
    unittest.main()
