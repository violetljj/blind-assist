#!/usr/bin/env python3
"""Mutation tests for the non-execution FactorTensorAdapter protocol lock."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.validate_geometry_r2_f1_adapter_protocol import validate_all


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_2026-08-10.json"
FIXTURE_PATH = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_FIXTURE_2026-08-10.json"
FACTOR_SCHEMA_PATH = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_SCHEMA_2026-08-10.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class GeometryR2F1AdapterProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load(PROTOCOL_PATH)
        cls.fixture = load(FIXTURE_PATH)
        cls.factor_schema = load(FACTOR_SCHEMA_PATH)

    def errors(self, protocol: dict | None = None, fixture: dict | None = None, *, bindings: bool = False) -> list[str]:
        return validate_all(
            protocol or copy.deepcopy(self.protocol),
            fixture or copy.deepcopy(self.fixture),
            copy.deepcopy(self.factor_schema),
            REPO_ROOT,
            check_bindings=bindings,
            check_absence=False,
        )

    def test_frozen_contract_and_bindings_are_valid(self) -> None:
        self.assertEqual(self.errors(bindings=True), [])

    def test_missing_operation_fails_closed(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["operations"] = [item for item in protocol["operations"] if item["id"] != "scale_sigma_m_from_calibrated_scale_distribution"]
        errors = self.errors(protocol)
        self.assertIn("OPERATION_SET", errors)
        self.assertIn("GAP_AUDIT_NOT_CLOSED", errors)

    def test_prediction_field_may_not_be_dropped(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        del protocol["prediction_field_consumers"]["depth_log_sigma_hw"]
        self.assertIn("PREDICTION_FIELD_COVERAGE", self.errors(protocol))

    def test_reducer_field_may_not_be_dropped(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        del protocol["reducer_field_producers"]["support.normal_sigma_rad"]
        self.assertIn("REDUCER_FIELD_COVERAGE", self.errors(protocol))

    def test_execution_or_trainable_authority_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["authority"]["trainable_parameters"] = 1
        protocol["execution_authority"]["adapter_implementation"] = True
        errors = self.errors(protocol)
        self.assertIn("ADAPTER_AUTHORITY", errors)
        self.assertIn("EXECUTION_AUTHORITY", errors)

    def test_final_task_shortcut_in_fixture_input_is_rejected(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][0]["patch"]["final_state"] = "OCCUPIED_OBSERVED"
        errors = self.errors(fixture=fixture)
        self.assertTrue(any(item.startswith("FORBIDDEN_INPUT_KEY:") for item in errors))

    def test_local_missing_depth_cannot_become_positive(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][2]["expected"]["forbidden_state"] = "CLEAR_OBSERVED"
        self.assertIn("FIXTURE_LOCAL_MISSING_FAIL_CLOSED", self.errors(fixture=fixture))

    def test_support_invalid_must_be_all_unknown(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][4]["expected"]["all_unknown"] = False
        self.assertIn("FIXTURE_SUPPORT_FAIL_CLOSED", self.errors(fixture=fixture))

    def test_receipt_identity_mismatch_must_be_all_unknown(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][5]["expected"]["all_unknown"] = False
        self.assertIn("FIXTURE_RECEIPT_FAIL_CLOSED", self.errors(fixture=fixture))

    def test_uncertainty_may_not_strengthen_occupancy(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][3]["expected"]["occupancy_strength_relation_to_nominal"] = "STRONGER"
        self.assertIn("FIXTURE_UNCERTAINTY_MONOTONICITY", self.errors(fixture=fixture))

    def test_successor_cannot_gain_execution_authority(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["unique_successor"]["execution_authority"] = True
        self.assertIn("UNIQUE_SUCCESSOR", self.errors(protocol))

    def test_binding_hash_mutation_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["bindings"][0]["sha256"] = "0" * 64
        self.assertIn("BINDING_SHA:FROZEN_F1_FACTOR_SCHEMA", self.errors(protocol, bindings=True))

    def test_fixture_case_identity_and_cardinality_are_frozen(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"].append(copy.deepcopy(fixture["cases"][0]))
        errors = self.errors(fixture=fixture)
        self.assertIn("FIXTURE_CASE_IDS", errors)
        self.assertIn("FIXTURE_CASE_ID_DUPLICATE", errors)


if __name__ == "__main__":
    unittest.main()
