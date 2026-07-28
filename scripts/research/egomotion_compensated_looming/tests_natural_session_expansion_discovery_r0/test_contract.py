from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from scripts.research.egomotion_compensated_looming.natural_session_expansion_discovery_r0 import (
    prepare_sources,
    runner,
)


REPO = Path(__file__).resolve().parents[4]
CONTRACT = (
    REPO
    / "docs/research/rcle/"
    "RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0_CONTRACT_2026-07-28.json"
)


class ContractTest(unittest.TestCase):
    def test_metadata_freeze_has_four_discovery_and_one_sealed(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        sessions = contract["sessions"]
        self.assertEqual(len(sessions), 5)
        self.assertEqual(
            sum(item["algorithm_execution_authorized"] for item in sessions),
            4,
        )
        sealed = [
            item for item in sessions
            if item["pre_run_access_state"] == "SEALED_UNSEEN"
        ]
        self.assertEqual([item["session_id"] for item in sealed], [
            "ADVIO_OFFICE04_SEQUENCE16_IPHONE"
        ])
        self.assertFalse(sealed[0]["algorithm_execution_authorized"])

    def test_algorithm_and_analysis_lock(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        lock = contract["algorithm_lock"]
        self.assertEqual(lock["threshold_per_s"], 0.01)
        self.assertEqual(lock["required_consecutive_pairs"], 3)
        self.assertFalse(lock["algorithm_adjustment_authorized"])
        forbidden = contract["reporting_lock"]["forbidden"]
        self.assertIn("AUROC", forbidden)
        self.assertIn("F1", forbidden)
        self.assertIn("pair pooling as sample size", forbidden)

    def test_sealed_session_is_fail_closed_before_path_access(self) -> None:
        with mock.patch.object(
            prepare_sources, "download", side_effect=AssertionError
        ):
            with self.assertRaisesRegex(
                PermissionError, "SEALED_UNSEEN"
            ):
                prepare_sources.extract(
                    16, Path("not-opened.zip"), Path("not-created")
                )
        with self.assertRaisesRegex(PermissionError, "SEALED_UNSEEN"):
            runner.run(
                16,
                Path("not-resolved"),
                Path("not-read"),
                Path("not-written"),
            )

    def test_fixed_segment_is_single_continuous_601_pair_run(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        segment = contract["segment_rule"]
        self.assertEqual(segment["frame_start_zero_based"], 0)
        self.assertEqual(segment["candidate_pair_count"], 601)
        self.assertEqual(segment["frame_end_exclusive_zero_based"], 602)
        self.assertIn("uninterrupted", segment["continuity"])


if __name__ == "__main__":
    unittest.main()
