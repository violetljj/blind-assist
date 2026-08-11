#!/usr/bin/env python3
"""Focused local-source tests for the TARO R6 confirmation I/O boundary."""

from __future__ import annotations

import unittest
import zipfile
from collections import Counter
from pathlib import Path

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation as r6
from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io


REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY = REPO_ROOT / "artifacts.local/evidence/taro/o0r-r6-untouched-inventory-r0/exact-frame-plan.json"


@unittest.skipUnless(INVENTORY.is_file(), "sealed R6 local inventory is unavailable")
class R6ConfirmationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frames = r6io.load_exact_cohort(INVENTORY, REPO_ROOT)

    def test_inventory_loads_exact_frozen_120_frame_sequence(self) -> None:
        self.assertEqual(r6.EXPECTED_FRAME_COUNT, len(self.frames))
        self.assertEqual(r6.ROSTER, tuple(dict.fromkeys((row.parent_id, row.video_id) for row in self.frames)))
        self.assertEqual(r6.expected_parent_frame_counts(), Counter(row.parent_id for row in self.frames))

    def test_phase_a_reader_never_opens_faro_member(self) -> None:
        frame = self.frames[0]
        reads: Counter[str] = Counter()
        with zipfile.ZipFile(frame.upsampling_archive) as up_bundle, zipfile.ZipFile(frame.intrinsics_archive) as intr_bundle:
            loaded = r6io.read_phase_a_frame(frame, up_bundle, intr_bundle, observer=lambda role, _: reads.update([role]))
        self.assertEqual({"color", "lowres_depth", "confidence", "intrinsics", "trajectory"}, set(reads))
        self.assertNotIn("highres_depth", reads)
        self.assertEqual(frame.physical_frame_id, loaded["source_receipt"]["physical_frame_id"])

    def test_phase_b_reader_opens_only_exact_faro_member(self) -> None:
        frame = self.frames[0]
        reads: Counter[str] = Counter()
        with zipfile.ZipFile(frame.upsampling_archive) as up_bundle:
            faro, binding = r6io.read_faro_frame(frame, up_bundle, observer=lambda role, _: reads.update([role]))
        self.assertEqual({"highres_depth": 1}, dict(reads))
        self.assertEqual(frame.container_bindings["upsampling"]["sha256"], binding["container_sha256"])
        self.assertEqual((1440, 1920), faro.shape)


if __name__ == "__main__":
    unittest.main()
