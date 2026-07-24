#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import l1e_materialization_recovery_r3_continuation_a3 as a3
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


REPO = Path(__file__).resolve().parents[3]
CONFIG = (
    REPO
    / "configs/ustrf_route_target_l1e_materialization_recovery_r3_continuation_a3.json"
)


class ContinuationA3ContractTest(unittest.TestCase):
    def test_config_and_current_coverage_validate(self) -> None:
        config = a3.verify_config(REPO, CONFIG)
        overlay = frozen_materializer.verify_overlay(
            REPO, a3.materializer_config_path(REPO, config)
        )
        summary, _ = a3.coverage(REPO, overlay)
        self.assertGreaterEqual(summary["verified_ledgers"], 13)
        self.assertGreaterEqual(summary["verified_frames"], 22_699)
        self.assertEqual(summary["discontinuity_resets"], 15)

    def test_extended_path_adapter(self) -> None:
        sample = Path(r"E:\linnan\x")
        if __import__("os").name == "nt":
            self.assertTrue(str(a3.extended_windows_path(sample)).startswith("\\\\?\\"))

    def test_four_gib_guard_is_preserved(self) -> None:
        self.assertEqual(a3.A3_MEMORY_GUARD_BYTES, 4 * 1024**3)

    def test_authority_cannot_be_opened(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["authority"]["candidate_execution"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(a3.ContinuationA3Error):
                a3.verify_config(REPO, path)


if __name__ == "__main__":
    unittest.main()
