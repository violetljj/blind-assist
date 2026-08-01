from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lock_stage_c_e0_2_fixed_batch as subject


def _entry(name: str, date: str, size: int) -> dict:
    return {
        "trajectory": name,
        "recording_date": date,
        "total_bytes": size,
        "metadata_healthy": True,
    }


class E02FixedBatchLockTest(unittest.TestCase):
    def test_excludes_consumed_trajectories_and_dates(self) -> None:
        ledger = [
            _entry("burned-id", "x", 1),
            _entry("burned-date", "burned", 2),
            *[
                _entry(chr(ord("a") + index), f"d{index}", index + 3)
                for index in range(6)
            ],
        ]
        selected = subject._select(
            ledger, {"burned-id"}, {"burned"}
        )
        self.assertEqual(
            [item["trajectory"] for item in selected],
            ["a", "b", "c", "d", "e", "f"],
        )

    def test_fixed_batch_requires_six_unique_dates(self) -> None:
        ledger = [_entry(str(index), "same", index) for index in range(7)]
        with self.assertRaisesRegex(ValueError, "six-source"):
            subject._select(ledger, set(), set())


if __name__ == "__main__":
    unittest.main()
