#!/usr/bin/env python3
"""Focused contract tests for R3 transport and one-shard recovery."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import run_l1e_materialization_recovery_r3_canary as canary
import run_l1e_materialization_recovery_r3_one_shard as one_shard


REPO = Path(__file__).resolve().parents[3]
CANARY_CONFIG = (
    REPO / "configs/ustrf_route_target_l1e_materialization_recovery_r3_canary.json"
)
ONE_SHARD_CONFIG = (
    REPO
    / "configs/ustrf_route_target_l1e_materialization_recovery_r3_one_shard.json"
)


class RecoveryR3ContractTest(unittest.TestCase):
    def assert_overlay_rejected(self, payload: dict) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises((one_shard.MaterializationError, canary.CanaryError)):
                one_shard.verify_overlay(REPO, path)

    def test_frozen_configs_validate(self) -> None:
        self.assertEqual(canary.verify_config(REPO, CANARY_CONFIG)["stage"], canary.STAGE)
        self.assertEqual(
            one_shard.verify_overlay(REPO, ONE_SHARD_CONFIG)["execution"][
                "maximum_crowdbot_shards"
            ],
            1,
        )

    def test_authority_cannot_be_opened(self) -> None:
        payload = copy.deepcopy(canary.load_json(ONE_SHARD_CONFIG))
        payload["authority"]["selection"] = True
        self.assert_overlay_rejected(payload)

    def test_memory_floor_cannot_be_lowered(self) -> None:
        payload = copy.deepcopy(canary.load_json(ONE_SHARD_CONFIG))
        payload["execution"]["minimum_system_available_physical_memory_bytes"] -= 1
        self.assert_overlay_rejected(payload)

    def test_shard_limit_cannot_expand(self) -> None:
        payload = copy.deepcopy(canary.load_json(ONE_SHARD_CONFIG))
        payload["execution"]["maximum_crowdbot_shards"] = 2
        self.assert_overlay_rejected(payload)

    def test_implementation_hash_drift_is_rejected(self) -> None:
        payload = copy.deepcopy(canary.load_json(ONE_SHARD_CONFIG))
        payload["implementation_bindings"]["device_exporter"]["sha256"] = "0" * 64
        self.assert_overlay_rejected(payload)

    def test_canary_receipt_drift_is_rejected(self) -> None:
        payload = copy.deepcopy(canary.load_json(ONE_SHARD_CONFIG))
        payload["canary_receipt"]["sha256"] = "0" * 64
        self.assert_overlay_rejected(payload)


if __name__ == "__main__":
    unittest.main()
