from __future__ import annotations

import unittest

from scripts.research.vi_task_geometry_g0.capture_contract import (
    EPISODE_TYPES,
    SURFACE_STRATA,
    validate_camera_slots,
    validate_imu_samples,
    validate_roster,
)


class VitgG0CaptureContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = "esp32_boot_monotonic:test-boot"
        strata = sorted(SURFACE_STRATA)
        self.parents = [f"parent-{index:02d}" for index in range(8)]
        self.episodes = [f"episode-{index:02d}" for index in range(7)]
        self.manifest = {
            "frozen_parent_records": [
                {
                    "parent_id": parent,
                    "site_id": f"site-{index % 2}",
                    "surface_stratum": strata[index % 4],
                }
                for index, parent in enumerate(self.parents)
            ],
            "frozen_episode_records": [
                {"episode_id": episode, "episode_type": kind, "duration_seconds": 20.0}
                for episode, kind in zip(self.episodes, sorted(EPISODE_TYPES))
            ],
        }
        self.expected_pairs = {(parent, episode) for parent in self.parents for episode in self.episodes}
        self.one_pair = {(self.parents[0], self.episodes[0])}

    def test_roster_requires_two_sites_and_balanced_surface_strata(self) -> None:
        validate_roster(self.manifest)
        self.manifest["frozen_parent_records"][0]["site_id"] = "site-1"
        self.manifest["frozen_parent_records"][2]["site_id"] = "site-1"
        self.manifest["frozen_parent_records"][4]["site_id"] = "site-1"
        self.manifest["frozen_parent_records"][6]["site_id"] = "site-1"
        with self.assertRaisesRegex(ValueError, "at least two physical sites"):
            validate_roster(self.manifest)

    def camera_rows(self) -> list[dict]:
        rows = []
        for parent, episode in sorted(self.one_pair):
            for index in range(301):
                rows.append(
                    {
                        "parent_id": parent,
                        "episode_id": episode,
                        "frame_id": f"{parent}-{episode}-{index}",
                        "frame_sequence": index,
                        "capture_timestamp_us": index * 66_667,
                        "status": "VALID_JPEG",
                        "clock_domain": self.clock,
                    }
                )
        return rows

    def test_camera_gap_must_be_materialized(self) -> None:
        rows = self.camera_rows()
        validate_camera_slots(rows, self.one_pair, self.clock)
        del rows[100]
        with self.assertRaisesRegex(ValueError, "deleted instead of materialized"):
            validate_camera_slots(rows, self.one_pair, self.clock)

    def test_forbidden_tof_field_fails_closed(self) -> None:
        rows = self.camera_rows()
        rows[0]["tof_range_mm"] = 800
        with self.assertRaisesRegex(ValueError, "forbidden ToF"):
            validate_camera_slots(rows, self.one_pair, self.clock)

    def test_imu_must_bracket_every_valid_camera_frame(self) -> None:
        camera = validate_camera_slots(self.camera_rows(), self.one_pair, self.clock)
        rows = []
        for parent, episode in sorted(self.one_pair):
            for index in range(4002):
                rows.append(
                    {
                        "parent_id": parent,
                        "episode_id": episode,
                        "timestamp_us": index * 5_000,
                        "accelerometer_mps2": [0.0, 0.0, 9.81],
                        "gyroscope_rad_s": [0.0, 0.0, 0.0],
                        "clock_domain": self.clock,
                    }
                )
        validate_imu_samples(rows, self.one_pair, self.clock, camera)
        rows = [row for row in rows if not (
            row["parent_id"] == self.parents[0]
            and row["episode_id"] == self.episodes[0]
            and row["timestamp_us"] in {5_000, 10_000, 15_000, 20_000, 25_000}
        )]
        with self.assertRaisesRegex(ValueError, "gap above 20 ms"):
            validate_imu_samples(rows, self.one_pair, self.clock, camera)


if __name__ == "__main__":
    unittest.main()
