from __future__ import annotations

import unittest
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
from run_r2_l1x_l2p_transport_amendment_a1 import (
    SAFE_REMOTE,
    verify_amendment,
)


class R2TransportAmendmentA1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config = (
            cls.repo
            / "configs/ustrf_route_target_r2_l1x_l2p_transport_amendment_a1.json"
        )

    def test_failed_r2_parent_is_hash_bound_and_preserved(self) -> None:
        config = verify_amendment(self.repo, self.config)["overlay"]
        self.assertFalse(
            config["transport_amendment"][
                "parent_attempts_count_toward_amendment"
            ]
        )

    def test_only_exact_bounded_attempt_leaf_is_cleanup_eligible(self) -> None:
        self.assertIsNotNone(
            SAFE_REMOTE.fullmatch(
                "r2l1e-r2/source_sequence__0123456789ab/attempt-001"
            )
        )
        self.assertIsNone(SAFE_REMOTE.fullmatch("r2l1e-r2"))
        self.assertIsNone(
            SAFE_REMOTE.fullmatch("r2l1e-r2/source/attempt-004")
        )
        self.assertIsNone(
            SAFE_REMOTE.fullmatch("r2l1e-r2/source/../attempt-001")
        )

    def test_r1_and_failed_r2_terminal_receipts_remain_distinct(self) -> None:
        verified = verify_amendment(self.repo, self.config)
        config = verified["parent"]
        overlay = verified["overlay"]
        r1_path = (
            self.repo
            / config["immutable_r1_parent"]["bindings"]["terminal_receipt"]["path"]
        )
        failed_r2_path = (
            self.repo
            / overlay["transport_amendment"]["failed_r2_parent_bindings"][
                "terminal_receipt"
            ]["path"]
        )
        self.assertNotEqual(r1_path.resolve(), failed_r2_path.resolve())
        self.assertEqual(r1.sha256_file(r1_path), "2be31cca0b64f195e648293a2c0ef4a85e1e57dd5d9730099c58bbaf73df7e6d")
        self.assertEqual(r1.sha256_file(failed_r2_path), "66396d7a55dcbc3d69e49f942ae13fd014056c1d2d56455157a08259ed811fb1")


if __name__ == "__main__":
    unittest.main()
