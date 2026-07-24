#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import l1e_materialization_recovery_r3_remaining as continuation
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


REPO = Path(__file__).resolve().parents[3]
CONFIG = (
    REPO
    / "configs/ustrf_route_target_l1e_materialization_recovery_r3_continuation_a1.json"
)


class RemainingR3ContractTest(unittest.TestCase):
    def test_frozen_config_and_current_coverage_validate(self) -> None:
        config = continuation.verify_config(REPO, CONFIG)
        materializer = frozen_materializer.verify_overlay(
            REPO, continuation.materializer_config_path(REPO, config)
        )
        summary, _ = continuation.coverage(REPO, materializer)
        self.assertEqual(summary["expected_ledgers"], 41)
        self.assertEqual(summary["expected_frames"], 62_229)
        self.assertEqual(summary["discontinuity_resets"], 15)
        self.assertGreaterEqual(summary["verified_ledgers"], 3)
        self.assertGreaterEqual(summary["verified_frames"], 6049)

    def test_authority_cannot_be_opened(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["authority"]["candidate_execution"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(continuation.ContinuationError):
                continuation.verify_config(REPO, path)

    def test_process_and_shard_limits_cannot_expand(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["execution"]["maximum_total_remaining_crowdbot_shards"] = 39
        payload["execution"]["one_host_process_per_shard"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(continuation.ContinuationError):
                continuation.verify_config(REPO, path)

    def test_full_gate_constants_are_frozen(self) -> None:
        self.assertEqual(continuation.EXPECTED_LEDGERS, 41)
        self.assertEqual(continuation.EXPECTED_FRAMES, 62_229)
        self.assertEqual(continuation.EXPECTED_RESETS, 15)
        self.assertEqual(continuation.EXPECTED_REMAINING_CROWDBOT_SHARDS, 38)


if __name__ == "__main__":
    unittest.main()
