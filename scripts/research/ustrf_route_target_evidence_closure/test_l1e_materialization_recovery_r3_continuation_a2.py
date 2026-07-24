#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import l1e_materialization_recovery_r3_continuation_a2 as a2
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


REPO = Path(__file__).resolve().parents[3]
CONFIG = (
    REPO
    / "configs/ustrf_route_target_l1e_materialization_recovery_r3_continuation_a2.json"
)


class ContinuationA2ContractTest(unittest.TestCase):
    def test_config_and_a1_terminal_validate(self) -> None:
        config = a2.verify_config(REPO, CONFIG)
        materializer = frozen_materializer.verify_overlay(
            REPO, a2.materializer_config_path(REPO, config)
        )
        summary, _ = a2.coverage(REPO, materializer)
        self.assertGreaterEqual(summary["verified_ledgers"], 12)
        self.assertGreaterEqual(summary["verified_frames"], 20_844)
        self.assertEqual(summary["discontinuity_resets"], 15)

    def test_short_control_path(self) -> None:
        root = Path(r"E:\x")
        path = a2._a2_attempt_root(
            root,
            "crowdbot_0410_shared_control",
            "defaced_2021-04-10-10-25-43-008_filtered_lidar_odom",
        ) / "a002" / "r.json"
        self.assertLess(len(str(path)), 80)

    def test_user_authorized_memory_guard_is_exactly_four_gib(self) -> None:
        self.assertEqual(a2.PARENT_MEMORY_GUARD_BYTES, 6 * 1024**3)
        self.assertEqual(a2.A2_MEMORY_GUARD_BYTES, 4 * 1024**3)

    def test_only_durable_a1_receipt_counts(self) -> None:
        materializer_config = frozen_materializer.verify_overlay(
            REPO,
            REPO
            / "configs/ustrf_route_target_l1e_materialization_recovery_r3_one_shard.json",
        )
        root = a2.output_root(REPO, materializer_config)
        self.assertEqual(
            a2.cumulative_attempt_count(
                root,
                "crowdbot_0410_shared_control",
                "defaced_2021-04-10-10-25-43-008_filtered_lidar_odom",
            ),
            1,
        )

    def test_authority_cannot_be_opened(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["authority"]["candidate_execution"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(a2.ContinuationA2Error):
                a2.verify_config(REPO, path)


if __name__ == "__main__":
    unittest.main()
