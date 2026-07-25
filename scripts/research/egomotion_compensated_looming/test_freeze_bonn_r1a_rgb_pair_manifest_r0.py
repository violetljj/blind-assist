#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_bonn_r1a_rgb_pair_manifest_r0 as subject  # noqa: E402


class BonnR1ARgbPairManifestTest(unittest.TestCase):
    def test_rows_require_strict_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            subject.rows(b"1.0 a\n1.0 b\n", 2)

    def test_window_is_frozen_to_ten_seconds(self) -> None:
        self.assertEqual(subject.WINDOW_SECONDS, 10.0)


if __name__ == "__main__":
    unittest.main()
