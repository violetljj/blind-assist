#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from freeze_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0 import (
    build_freeze,
    metadata_manifest,
    rank_hash,
)
from run_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0 import (
    contrast,
    delta,
    direction_status,
    support_fraction,
)
from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import canonical_bytes


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.json"
FREEZE = REPO / "artifacts.local/evidence/jrdb-person-3d-trajectory-sensor-support-and-bias-cross-sequence-replication-r0/sequence-freeze.json"


def row(classification: str, range_band: str, presence: str, residual: float | None = None) -> dict:
    return {
        "classification": classification,
        "range_band": range_band,
        "cross_modal_presence": presence,
        "centroid_residual_3d_m": residual,
    }


class CrossSequenceReplicationTest(unittest.TestCase):
    def test_metadata_freeze_exact_rebuild(self) -> None:
        actual = json.loads(FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(build_freeze(CONFIG), actual)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(len(metadata_manifest(config)["selected"]), 3)

    def test_frozen_rank_hash(self) -> None:
        self.assertEqual(
            rank_hash("gates-basement-elevators-2019-01-17_1"),
            "012bd0ea814510fe8b5725ee2fa7bfee5cfb489171902af7913b1cb3911919c0",
        )

    def test_support_fraction_and_delta(self) -> None:
        rows = [row("sensor-supported", "0-10", "3d-and-2d"), row("abstained", "40-plus", "3d-only")]
        self.assertEqual(support_fraction(rows), 0.5)
        self.assertAlmostEqual(delta(0.5, 0.4), 0.1)
        self.assertIsNone(delta(None, 0.4))

    def test_far_and_3d_only_contrasts(self) -> None:
        rows = [
            row("sensor-supported", "0-10", "3d-and-2d", 0.1),
            row("sensor-supported", "10-20", "3d-and-2d", 0.2),
            row("annotation-only", "40-plus", "3d-only"),
        ]
        self.assertTrue(contrast(rows, "far")["adverse_direction"])
        self.assertTrue(contrast(rows, "three_d_only")["adverse_direction"])

    def test_direction_requires_two_evaluable_sequences(self) -> None:
        adverse = {"status": "EVALUABLE", "adverse_direction": True}
        opposite = {"status": "EVALUABLE", "adverse_direction": False}
        missing = {"status": "NOT_EVALUABLE"}
        self.assertEqual(direction_status([adverse, missing]), "NOT_EVALUABLE")
        self.assertEqual(direction_status([adverse, adverse]), "DIRECTION_REPLICATED")
        self.assertEqual(direction_status([adverse, opposite]), "MIXED_OR_CONTRADICTED")


if __name__ == "__main__":
    unittest.main()
