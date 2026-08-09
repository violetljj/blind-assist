#!/usr/bin/env python3
"""Mutation tests for the static R2 F1-P protocol validator."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.validate_geometry_r2_f1_protocol import (
    validate_static_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_TRAIN_ONLY_FACTOR_LEARNABILITY_PROTOCOL_LOCK_2026-08-10.json"
SCHEMA = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_SCHEMA_2026-08-10.json"
DCA = REPO_ROOT / "docs/research/assistive-geometry-data-capability/BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.json"
F0 = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_RESULT_2026-08-10.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class GeometryR2F1ProtocolValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load(PROTOCOL)
        cls.schema = load(SCHEMA)
        cls.dca = load(DCA)
        cls.f0 = load(F0)

    def validate(self, *, protocol: dict | None = None, schema: dict | None = None, dca: dict | None = None, f0: dict | None = None) -> list[str]:
        return validate_static_contract(
            copy.deepcopy(self.protocol if protocol is None else protocol),
            copy.deepcopy(self.schema if schema is None else schema),
            copy.deepcopy(self.dca if dca is None else dca),
            copy.deepcopy(self.f0 if f0 is None else f0),
        )

    def test_frozen_contract_is_valid(self) -> None:
        self.assertEqual(self.validate(), [])

    def test_execution_authority_drift_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["execution_authority"]["optimizer_step"] = True
        self.assertIn("EXECUTION_AUTHORITY_EXCEEDED", self.validate(protocol=protocol))

    def test_final_task_field_is_rejected(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["prediction_fields"].append({"group": "depth_scale", "name": "final_state"})
        errors = self.validate(schema=schema)
        self.assertIn("PREDICTION_FIELD_SET", errors)
        self.assertIn("TASK_SHORTCUT_IN_FACTOR_FIELDS", errors)

    def test_unknown_as_negative_is_rejected(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["unknown_and_mask_contract"]["unknown_is_negative"] = True
        self.assertIn("UNKNOWN_AS_NEGATIVE", self.validate(schema=schema))

    def test_capability_count_drift_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        target = next(item for item in protocol["factor_supervision_availability_matrix"] if item["id"] == "continuous_boundary_truth")
        target["frames"] = 1
        self.assertIn("AVAILABILITY_DRIFT:continuous_boundary_truth:frames", self.validate(protocol=protocol))

    def test_aggregate_loss_cannot_select_checkpoint(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["loss_contract"]["aggregate_loss_is_checkpoint_metric"] = True
        self.assertIn("AGGREGATE_CHECKPOINT_METRIC", self.validate(protocol=protocol))

    def test_downstream_reducer_cannot_rescue_factor_failure(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["f1_success_rule"]["reducer_task_metric_may_rescue"] = True
        self.assertIn("DOWNSTREAM_RESCUE", self.validate(protocol=protocol))

    def test_checkpoint_role_must_remain_parent_disjoint(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["train_internal_role_contract"]["parent_disjoint"] = False
        self.assertIn("TRAIN_ROLE_DISJOINTNESS", self.validate(protocol=protocol))

    def test_f0_pass_is_required_but_does_not_grant_f1(self) -> None:
        f0 = copy.deepcopy(self.f0)
        f0["passed"] = False
        self.assertIn("F0_PREDECESSOR", self.validate(f0=f0))
        self.assertFalse(self.protocol["execution_authority"]["f1_execution"])


if __name__ == "__main__":
    unittest.main()
