#!/usr/bin/env python3
"""Static tests for the F1-schema to F0-reducer adapter gap audit."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.audit_geometry_r2_f1_adapter_gap import (
    BLOCKED_TERMINAL,
    READY_FOR_CANARY_TERMINAL,
    REQUIRED_ADAPTER_OPERATIONS,
    audit_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FACTOR_SCHEMA = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_SCHEMA_2026-08-10.json"


def schema() -> dict:
    return json.loads(FACTOR_SCHEMA.read_text(encoding="utf-8"))


def complete_static_contract() -> dict:
    return {
        "operations": [{"id": operation} for operation in sorted(REQUIRED_ADAPTER_OPERATIONS)],
        "authority": {
            "outside_learned_graph": True,
            "deterministic": True,
            "trainable_parameters": 0,
            "final_task_shortcut_allowed": False,
            "execution_authority": False,
        },
    }


class GeometryR2F1AdapterGapAuditTests(unittest.TestCase):
    def test_current_missing_adapter_is_hard_blocker(self) -> None:
        result = audit_contract(schema())
        self.assertEqual(result["terminal"], BLOCKED_TERMINAL)
        self.assertFalse(result["adapter_contract_present"])
        self.assertFalse(result["current_execution_authority"])

    def test_complete_static_contract_only_reaches_canary_not_execution(self) -> None:
        result = audit_contract(schema(), complete_static_contract())
        self.assertEqual(result["terminal"], READY_FOR_CANARY_TERMINAL)
        self.assertTrue(result["adapter_static_contract_complete"])
        self.assertFalse(result["current_execution_authority"])

    def test_scalar_scale_uncertainty_operation_is_required(self) -> None:
        contract = complete_static_contract()
        contract["operations"] = [item for item in contract["operations"] if item["id"] != "scale_sigma_m_from_calibrated_scale_distribution"]
        result = audit_contract(schema(), contract)
        self.assertIn("scale_sigma_m_from_calibrated_scale_distribution", result["missing_adapter_operations"])
        self.assertEqual(result["terminal"], BLOCKED_TERMINAL)

    def test_support_normal_and_height_uncertainty_are_required(self) -> None:
        contract = complete_static_contract()
        contract["operations"] = [item for item in contract["operations"] if item["id"] not in {"support_normal_sigma_rad", "support_height_sigma_m"}]
        result = audit_contract(schema(), contract)
        self.assertIn("support_normal_sigma_rad", result["missing_adapter_operations"])
        self.assertIn("support_height_sigma_m", result["missing_adapter_operations"])

    def test_dense_to_obstacle_component_mapping_is_required(self) -> None:
        contract = complete_static_contract()
        contract["operations"] = [item for item in contract["operations"] if item["id"] != "dense_evidence_to_obstacle_components"]
        result = audit_contract(schema(), contract)
        self.assertIn("dense_evidence_to_obstacle_components", result["missing_adapter_operations"])
        self.assertIn("obstacle.evidence_sigma", result["gap_groups"]["dense_to_obstacle_list"]["reducer_fields"])

    def test_adapter_must_stay_outside_learned_graph(self) -> None:
        contract = complete_static_contract()
        contract["authority"]["outside_learned_graph"] = False
        result = audit_contract(schema(), contract)
        self.assertFalse(result["adapter_authority_valid"])
        self.assertEqual(result["terminal"], BLOCKED_TERMINAL)

    def test_adapter_contract_never_grants_execution_authority(self) -> None:
        contract = complete_static_contract()
        contract["authority"]["execution_authority"] = True
        result = audit_contract(schema(), contract)
        self.assertFalse(result["adapter_authority_valid"])
        self.assertFalse(result["current_execution_authority"])


if __name__ == "__main__":
    unittest.main()
