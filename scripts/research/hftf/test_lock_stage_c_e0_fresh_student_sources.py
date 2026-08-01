from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lock_stage_c_e0_fresh_student_sources as subject


def _entry(
    trajectory: str,
    date: str,
    total_bytes: int,
    healthy: bool = True,
) -> dict:
    return {
        "trajectory": trajectory,
        "recording_date": date,
        "metadata_healthy": healthy,
        "rows": 100,
        "camera_height_m": 1.2,
        "total_bytes": total_bytes,
        "repo_paths": {
            "pose": f"data/{trajectory}.parquet",
            "rgb": f"video/rgb/{trajectory}__rgb.mp4",
            "depth": f"video/depth/{trajectory}__depth.mkv",
        },
        "files": {
            "pose": {"size_bytes": 1, "sha256": "a" * 64},
            "rgb": {"size_bytes": 2, "sha256": "b" * 64},
            "depth": {"size_bytes": 3, "sha256": "c" * 64},
        },
    }


class FreshSourceLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = [
            _entry("burned", "2024_01_01", 1),
            _entry("bad", "2024_01_02", 2, healthy=False),
            _entry("a", "2024_01_03", 10),
            _entry("same-date-larger", "2024_01_03", 11),
            _entry("b", "2024_01_04", 12),
            _entry("c", "2024_01_05", 13),
            _entry("d", "2024_01_06", 14),
            _entry("e", "2024_01_07", 15),
            _entry("f", "2024_01_08", 16),
        ]
        selected = subject._select_sources(
            self.ledger, {"burned"}
        )
        roles = ["train", "train", "train", "train", "dev", "heldout"]
        self.protocol = {
            "dataset_binding": {
                "dataset_repo": "repo",
                "dataset_revision": "rev",
                "consumed_trajectory_exclusions": ["burned"],
            },
            "frozen_sources": [
                subject._canonical_inventory_source(item, role)
                for item, role in zip(selected, roles)
            ],
        }
        self.inventory = {
            "terminal": "C0_EGOWALK_METADATA_COHORT_LOCKED",
            "dataset_repo": "repo",
            "dataset_revision": "rev",
            "metadata_healthy_count": 95,
            "rgb_or_depth_media_content_read": False,
            "inventory_ledger": self.ledger,
        }

    def test_selects_smallest_six_unique_dates_after_exclusions(self) -> None:
        selected = subject._select_sources(self.ledger, {"burned"})
        self.assertEqual(
            [item["trajectory"] for item in selected],
            ["a", "b", "c", "d", "e", "f"],
        )

    def test_exact_frozen_selection_validates(self) -> None:
        selected = subject._validate_selection(
            self.protocol, self.inventory
        )
        self.assertEqual(
            [item["role"] for item in selected],
            ["train", "train", "train", "train", "dev", "heldout"],
        )

    def test_source_mutation_fails_closed(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["frozen_sources"][0]["total_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "does not recompute"):
            subject._validate_selection(protocol, self.inventory)

    def test_inventory_media_read_fails_closed(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        inventory["rgb_or_depth_media_content_read"] = True
        with self.assertRaisesRegex(ValueError, "unexpectedly read"):
            subject._validate_selection(self.protocol, inventory)


if __name__ == "__main__":
    unittest.main()
