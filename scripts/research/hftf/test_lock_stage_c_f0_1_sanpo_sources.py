from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lock_stage_c_f0_1_sanpo_sources import (
    _expected_frames,
    _receipt_ready,
)


class StageCF01SanpoSourceLockTest(unittest.TestCase):
    def test_expected_frames_preserve_physical_timeline(self) -> None:
        self.assertEqual(list(range(25)), _expected_frames(5.0))
        self.assertEqual(list(range(0, 50, 2)), _expected_frames(20.0))
        with self.assertRaisesRegex(ValueError, "5 or 20"):
            _expected_frames(10.0)

    def test_receipt_requires_all_source_identity_fields(self) -> None:
        receipt = {
            "name": "object",
            "generation": "7",
            "size": 10,
            "md5_base64": "md5",
            "crc32c_base64": "crc",
        }
        self.assertTrue(_receipt_ready(receipt))
        for key in tuple(receipt):
            damaged = dict(receipt)
            damaged.pop(key)
            self.assertFalse(_receipt_ready(damaged), key)


if __name__ == "__main__":
    unittest.main()
