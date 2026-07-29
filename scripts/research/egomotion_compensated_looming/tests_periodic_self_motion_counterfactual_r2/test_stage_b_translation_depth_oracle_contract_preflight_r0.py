from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_stage_b_translation_depth_oracle_contract_preflight_r0 as preflight,
)


ROOT = Path(__file__).resolve().parents[4]


class StageBTranslationDepthOracleContractPreflightTests(unittest.TestCase):
    def test_valid_contract_closes_without_execution_activation(self) -> None:
        receipt, decision = preflight.validate_and_build(ROOT)
        self.assertEqual(receipt["gate_pass_count"], 12)
        self.assertEqual(receipt["gate_fail_count"], 0)
        self.assertEqual(
            receipt["terminal"],
            "CONTRACT_PREFLIGHT_PASS / VALID / EXECUTION_NOT_ACTIVATED",
        )
        self.assertEqual(
            decision["decision"],
            "HOLD_STAGE_B_EXECUTION_PENDING_SEPARATE_ACTIVATION",
        )
        self.assertFalse(decision["stage_b_response_access_authorized"])
        self.assertFalse(decision["stage_b_execution_authorized"])
        self.assertFalse(decision["formal_authority_consumed"])

    def test_numeric_seed_mutation_fails_closed(self) -> None:
        identity = preflight.load_json(ROOT / preflight.IDENTITY_RELATIVE)
        forged = copy.deepcopy(identity)
        forged["clusters"][0]["numeric_seed_uint64"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged_identity.json"
            path.write_text(
                json.dumps(forged, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                preflight.InvalidStageBContractPreflight,
                "CLUSTER_NUMERIC_SEED",
            ):
                preflight.validate_and_build(ROOT, identity_path=path)

    def test_rotation_boundary_cannot_be_relaxed(self) -> None:
        contract = preflight.load_json(ROOT / preflight.CONTRACT_RELATIVE)
        forged = copy.deepcopy(contract)
        forged["future_execution_gates"]["rotation_boundary"][
            "required_clusters"
        ] = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged_contract.json"
            path.write_text(
                json.dumps(forged, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                preflight.InvalidStageBContractPreflight,
                "ROTATION_REQUIRED_CLUSTERS",
            ):
                preflight.validate_and_build(ROOT, contract_path=path)

    def test_response_authority_mutation_fails_closed(self) -> None:
        contract = preflight.load_json(ROOT / preflight.CONTRACT_RELATIVE)
        forged = copy.deepcopy(contract)
        forged["response_access_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged_contract.json"
            path.write_text(
                json.dumps(forged, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                preflight.InvalidStageBContractPreflight,
                "CONTRACT_RESPONSE_ACCESS",
            ):
                preflight.validate_and_build(ROOT, contract_path=path)

    def test_zero_translation_is_zero_after_rotation_alignment(self) -> None:
        contract = preflight.load_json(ROOT / preflight.CONTRACT_RELATIVE)
        intrinsic = np.asarray(
            contract["coordinate_and_unit_contract"]["image"]["intrinsic"],
            dtype=np.float64,
        )
        pixels = np.asarray(((80.0, 120.0), (250.0, 500.0)))
        depths = np.asarray((3.0, 7.0))
        displacement = preflight.oracle_displacement(
            pixels,
            depths,
            intrinsic,
            np.eye(3),
            np.zeros(3),
            preflight._rotation_y(0.2),
            np.zeros(3),
        )
        np.testing.assert_allclose(displacement, 0.0, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
