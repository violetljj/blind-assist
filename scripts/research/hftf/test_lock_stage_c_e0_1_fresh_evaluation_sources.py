from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lock_stage_c_e0_1_fresh_evaluation_sources as subject


def _entry(name: str, date: str, size: int, healthy: bool = True) -> dict:
    return {
        "trajectory": name,
        "recording_date": date,
        "total_bytes": size,
        "metadata_healthy": healthy,
    }


class E01FreshEvaluationLockTest(unittest.TestCase):
    def test_selects_two_smallest_unique_dates(self) -> None:
        ledger = [
            _entry("burned", "d0", 1),
            _entry("bad", "d1", 2, False),
            _entry("a", "d2", 3),
            _entry("same", "d2", 4),
            _entry("b", "d3", 5),
        ]
        selected = subject._select(ledger, {"burned"})
        self.assertEqual(
            [item["trajectory"] for item in selected], ["a", "b"]
        )

    def test_fails_when_unique_dates_are_insufficient(self) -> None:
        ledger = [
            _entry("a", "d", 1),
            _entry("b", "d", 2),
        ]
        with self.assertRaisesRegex(ValueError, "unique-date"):
            subject._select(ledger, set())

    def test_canonical_preserves_exact_file_bindings(self) -> None:
        item = {
            **_entry("a", "d", 6),
            "rows": 10,
            "camera_height_m": 1.2,
            "repo_paths": {
                "pose": "p",
                "rgb": "r",
                "depth": "d",
            },
            "files": {
                "pose": {"size_bytes": 1, "sha256": "a"},
                "rgb": {"size_bytes": 2, "sha256": "b"},
                "depth": {"size_bytes": 3, "sha256": "c"},
            },
        }
        result = subject._canonical(item, "dev")
        self.assertEqual(result["role"], "dev")
        self.assertEqual(result["files"]["depth"]["path"], "d")
        self.assertEqual(result["files"]["depth"]["sha256"], "c")


if __name__ == "__main__":
    unittest.main()
