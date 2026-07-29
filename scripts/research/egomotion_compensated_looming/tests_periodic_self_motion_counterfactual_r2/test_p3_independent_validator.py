from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from ..periodic_self_motion_counterfactual_r2 import validate_p3_independent_r0 as validator


class P3IndependentValidatorTests(unittest.TestCase):
    def test_independent_identity_reconstruction(self) -> None:
        value = validator.expected_identity_lock()
        self.assertEqual(value["identity_count"], 8)
        self.assertEqual(
            value["seeds"][0]["numeric_seed_uint64"],
            1727242067111453576,
        )
        self.assertEqual(
            value["seeds"][1]["numeric_seed_uint64"],
            18409799703140433944,
        )
        self.assertEqual(value["worker_profiles"], [4, 8])
        self.assertEqual(value["prohibited_worker_profiles"], [12, 16])

    def test_identity_mutation_is_rejected(self) -> None:
        value = validator.expected_identity_lock()
        value["identities"][0]["pair_count"] = 600
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "identity.json"
            path.write_bytes(validator.canonical_bytes(value))
            with self.assertRaisesRegex(validator.InvalidP3, "DRIFT"):
                validator.validate_identity_lock(path)

    def test_projection_includes_separate_retry_reserve(self) -> None:
        receipts = []
        for kind, arms in (
            ("FACTORIAL", validator.FACTORIAL_ARMS),
            ("GUARDRAIL", validator.GUARDRAIL_ARMS),
        ):
            for arm in arms:
                receipts.append(
                    {
                        "cluster_kind": kind,
                        "arm": arm,
                        "timing": {
                            "render_seconds": 4.0,
                            "r3_seconds": 3.0,
                            "validation_and_hash_seconds": 1.0,
                        },
                    }
                )
        profile = {
            "workers": 8,
            "sequence_receipts": receipts,
        }
        projection = validator.profile_projection(profile)
        self.assertAlmostEqual(
            projection["retry_reserve_seconds"],
            8.0 * 496.0 / 8.0 * 0.10,
        )
        self.assertAlmostEqual(
            projection["total_seconds"],
            8.0 * 496.0 / 8.0 * 1.10,
        )
        self.assertEqual(projection["formal_factorial_sequences"], 480)
        self.assertEqual(projection["formal_guardrail_sequences"], 16)

    def test_projection_does_not_overweight_guardrails(self) -> None:
        receipts = [
            {
                "cluster_kind": "FACTORIAL",
                "timing": {
                    "render_seconds": 1.0,
                    "r3_seconds": 1.0,
                    "validation_and_hash_seconds": 0.0,
                },
            }
            for _ in validator.FACTORIAL_ARMS
        ]
        receipts.extend(
            {
                "cluster_kind": "GUARDRAIL",
                "timing": {
                    "render_seconds": 10.0,
                    "r3_seconds": 10.0,
                    "validation_and_hash_seconds": 0.0,
                },
            }
            for _ in validator.GUARDRAIL_ARMS
        )
        projection = validator.profile_projection(
            {"workers": 8, "sequence_receipts": receipts}
        )
        expected_component = (6 * 80 + 2 * 8 * 10) / 8
        self.assertAlmostEqual(projection["render_seconds"], expected_component)
        self.assertAlmostEqual(projection["r3_seconds"], expected_component)

    def test_formal_and_authority_constants_remain_false(self) -> None:
        expected = validator.expected_identity_lock()
        self.assertFalse(expected["formal_seed_access"])
        self.assertFalse(expected["formal_execution_authorized"])
        self.assertFalse(expected["p4_activated"])


if __name__ == "__main__":
    unittest.main()
