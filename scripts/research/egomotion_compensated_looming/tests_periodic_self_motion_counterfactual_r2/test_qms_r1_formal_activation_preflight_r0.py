from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    qms_r1_formal_activation_preflight_r0 as producer,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_qms_r1_formal_activation_preflight_independent_r0 as validator,
)


ROOT = Path(__file__).resolve().parents[4]
OPERATOR = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_OPERATOR_LOCK_R0_2026-07-29.json"
)
FORMAL = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json"
)
PREFLIGHT = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_FORMAL_ACTIVATION_PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json"
)


class QmsR1ActivationPreflightTests(unittest.TestCase):
    def test_operator_source_and_identity_are_frozen(self) -> None:
        result = validator.validate_operator(ROOT, OPERATOR)
        self.assertEqual(
            result["operator_source_sha256"],
            validator.EXPECTED_OPERATOR_SOURCE_SHA256,
        )

    def test_formal_and_preflight_locks_are_disjoint(self) -> None:
        operator = validator.validate_operator(ROOT, OPERATOR)
        formal, formal_overlap = validator.validate_formal_lock(
            ROOT, FORMAL, operator["operator_lock_sha256"]
        )
        preflight, preflight_overlap = validator.validate_preflight_lock(
            ROOT,
            PREFLIGHT,
            formal,
            FORMAL,
            operator["operator_lock_sha256"],
        )
        self.assertEqual(len(formal["identities"]), 496)
        self.assertEqual(len(preflight["identities"]), 8)
        self.assertEqual(set(formal_overlap.values()), {0})
        self.assertEqual(set(preflight_overlap.values()), {0})

    def test_seed_derivation_mutation_is_rejected(self) -> None:
        value = copy.deepcopy(producer.load_json(FORMAL))
        value["seeds"][0]["numeric_seed_uint64"] ^= 1
        operator = validator.validate_operator(ROOT, OPERATOR)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                validator.InvalidIndependentPreflight,
                "FORMAL_SEED_DERIVATION",
            ):
                validator.validate_formal_lock(
                    ROOT, path, operator["operator_lock_sha256"]
                )

    def test_overlap_mutation_is_rejected(self) -> None:
        formal = producer.load_json(FORMAL)
        old = producer.load_json(ROOT / producer.OLD_FORMAL)
        candidates = copy.deepcopy(formal["identities"])
        candidates[0]["numeric_seed_uint64"] = old["identities"][0][
            "numeric_seed_uint64"
        ]
        excluded = {
            field: validator._collect_values(old, field)
            for field in (
                "numeric_seed_uint64",
                "token",
                "token_sha256",
                "cluster_id",
                "sequence_id",
                "scene_geometry_sha256",
            )
        }
        with self.assertRaisesRegex(
            validator.InvalidIndependentPreflight,
            "IDENTITY_OVERLAP:numeric_seed_uint64",
        ):
            validator._assert_zero_overlap(candidates, excluded)

    def test_preflight_exact_eight_is_not_formal_authority(self) -> None:
        value = producer.load_json(PREFLIGHT)
        self.assertEqual(value["identity_count"], 8)
        self.assertFalse(value["formal_execution_authorized"])
        self.assertFalse(value["formal_seed_execution"])
        self.assertEqual(
            {item["cluster_kind"] for item in value["identities"]},
            {"FACTORIAL", "GUARDRAIL"},
        )


if __name__ == "__main__":
    unittest.main()
