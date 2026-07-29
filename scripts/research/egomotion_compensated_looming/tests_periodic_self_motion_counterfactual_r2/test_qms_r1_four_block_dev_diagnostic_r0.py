from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_qms_r1_four_block_dev_diagnostic_independent_r0 as validator,
)


ROOT = Path(__file__).resolve().parents[4]
CONTRACT = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_FOUR_BLOCK_DEV_DIAGNOSTIC_R0_CONTRACT_2026-07-29.json"
)
LOCK = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_FOUR_BLOCK_DEV_DIAGNOSTIC_R0_IDENTITY_LOCK_2026-07-29.json"
)


def _identity() -> dict:
    return validator.load_json(LOCK)["identities"][0]


def _rows(response: float = 0.0) -> list[dict]:
    identity = _identity()
    return [
        {
            "sequence_id": identity["sequence_id"],
            "cluster_id": identity["cluster_id"],
            "block": identity["block"],
            "ordinal": identity["ordinal"],
            "role": identity["role"],
            "arm": identity["arm"],
            "pair_index": index,
            "evaluable": True,
            "compensated_expansion_median_per_s": response,
            "compensated_three_pair_trigger": (
                response > 0.01 and index >= 2
            ),
            "detected_feature_count": 100,
            "forward_backward_consistent_count": 90,
            "forward_backward_consistent_fraction": 0.9,
            "occupied_3x3_cells": 9,
            "median_forward_backward_error_px": 0.1,
        }
        for index in range(validator.PAIR_COUNT)
    ]


class QmsR1FourBlockDevDiagnosticTests(unittest.TestCase):
    def test_contract_and_identity_grid_are_independently_valid(self) -> None:
        validator.validate_contract(ROOT, CONTRACT)
        lock, expected, overlaps = validator.validate_identity_lock(
            ROOT, LOCK, CONTRACT
        )
        self.assertEqual(len(lock["seeds"]), 8)
        self.assertEqual(len(expected), 48)
        self.assertEqual(set(overlaps.values()), {0})
        self.assertFalse(lock["formal_execution_authorized"])
        self.assertEqual(lock["formal_sequences_run"], 0)

    def test_seed_derivation_mutation_is_rejected(self) -> None:
        value = copy.deepcopy(validator.load_json(LOCK))
        value["seeds"][0]["numeric_seed_uint64"] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                validator.InvalidIndependentDev, "SEED_GRID"
            ):
                validator.validate_identity_lock(
                    ROOT, path, CONTRACT
                )

    def test_identity_overlap_mutation_is_rejected(self) -> None:
        value = copy.deepcopy(validator.load_json(LOCK))
        old = validator.load_json(ROOT / validator.EXCLUSION_FILES[0][1])
        reused = old["identities"][0]["numeric_seed_uint64"]
        value["seeds"][0]["numeric_seed_uint64"] = reused
        value["identities"][0]["numeric_seed_uint64"] = reused
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(validator.InvalidIndependentDev):
                validator.validate_identity_lock(
                    ROOT, path, CONTRACT
                )

    def test_threshold_equality_is_not_a_trigger(self) -> None:
        summary = validator.reduce_rows(_rows(0.01), _identity())
        self.assertEqual(summary["trigger_count"], 0)

    def test_abstention_resets_three_pair_streak(self) -> None:
        rows = _rows(0.02)
        rows[2]["evaluable"] = False
        rows[2]["compensated_expansion_median_per_s"] = None
        rows[2]["compensated_three_pair_trigger"] = False
        rows[3]["compensated_three_pair_trigger"] = False
        rows[4]["compensated_three_pair_trigger"] = False
        summary = validator.reduce_rows(rows, _identity())
        self.assertEqual(summary["trigger_count"], validator.PAIR_COUNT - 5)

    def test_forged_trigger_is_rejected(self) -> None:
        rows = _rows(0.0)
        rows[0]["compensated_three_pair_trigger"] = True
        with self.assertRaisesRegex(
            validator.InvalidIndependentDev, "FORGED_TRIGGER"
        ):
            validator.reduce_rows(rows, _identity())


if __name__ == "__main__":
    unittest.main()
