from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry as r0_generator,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry_r1 as r1_generator,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_geometry_independent as r0_validator,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_geometry_independent_r1 as r1_validator,
)


class GeometrySpecRepairR1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.r0_records = r0_validator.load_jsonl(
            r1_validator.R0_EVIDENCE / "all_seed_geometry_manifest.jsonl"
        )
        cls.r1_records = r0_validator.load_jsonl(
            r1_validator.DEFAULT_EVIDENCE
            / "all_seed_geometry_manifest.jsonl"
        )
        cls.trajectories = r0_validator.load_json(
            r1_validator.DEFAULT_EVIDENCE / "trajectory_manifest.json"
        )
        cls.r1_guards = [
            item
            for item in cls.r1_records
            if item["record_type"] == "guardrail_cluster"
        ]

    def test_r0_terminal_is_immutable(self) -> None:
        receipt_path = (
            r1_validator.R0_EVIDENCE
            / "independent_geometry_validation_receipt.json"
        )
        receipt = r0_validator.load_json(receipt_path)
        self.assertEqual(
            r1_validator.EXPECTED_R0_RECEIPT_SHA256,
            r0_validator.sha256_file(receipt_path),
        )
        self.assertEqual("INTERVENTION_NOT_EVALUABLE", receipt["terminal"])
        self.assertEqual("HOLD_P1", receipt["state"])

    def test_radial_gate_is_derived_from_unchanged_20_percent(self) -> None:
        amendment = r0_validator.load_json(r1_validator.AMENDMENT_PATH)
        gate = amendment["g13_replacement"]["common_gates"][
            "integrated_endpoint_log_radial_expansion"
        ]
        self.assertIn("ln(1.20)=0.1823215567939546", gate)
        self.assertAlmostEqual(math.log(1.20), r1_validator.LOG_1P20)

    def test_all_main_records_are_byte_identical(self) -> None:
        r0_main = [
            item for item in self.r0_records if item["record_type"] == "main_cluster"
        ]
        r1_main = [
            item for item in self.r1_records if item["record_type"] == "main_cluster"
        ]
        self.assertEqual(80, len(r0_main))
        self.assertEqual(
            [r0_validator.canonical_bytes(item) for item in r0_main],
            [r0_validator.canonical_bytes(item) for item in r1_main],
        )

    def test_guard_seeds_and_trajectories_are_unchanged(self) -> None:
        r0_guard = {
            item["cluster_id"]: item
            for item in self.r0_records
            if item["record_type"] == "guardrail_cluster"
        }
        for repaired in self.r1_guards:
            original = r0_guard[repaired["cluster_id"]]
            self.assertEqual(
                original["numeric_seed_uint64"], repaired["numeric_seed_uint64"]
            )
            for old_arm, new_arm in zip(original["arms"], repaired["arms"]):
                self.assertEqual(
                    old_arm["trajectory_sha256"], new_arm["trajectory_sha256"]
                )
                self.assertEqual(old_arm["trajectory"], new_arm["trajectory"])

    def test_repaired_scene_is_deterministic_and_target_is_on_mesh(self) -> None:
        first = r1_generator.build_guard_scene_r1("ADVIO_15", 0)
        second = r1_generator.build_guard_scene_r1("ADVIO_15", 0)
        self.assertEqual(first, second)
        target = first["designated_target"]
        obj = next(
            item
            for item in first["world"]["objects"]
            if item["object_id"] == target["object_id"]
        )
        x, y, z = target["world_point_m"]
        x0, x1, y0, y1 = obj["bounds_xy_m"]
        self.assertEqual(z, obj["plane_z_m"])
        self.assertTrue(x0 <= x <= x1 and y0 <= y <= y1)
        self.assertEqual(12, len(first["world"]["objects"]))

    def test_consumed_r1_g13_fails_closed_on_rotation_angle_tolerance(self) -> None:
        gate = r1_validator.gate_g13_r1(self.r1_guards, self.trajectories)
        self.assertEqual("FAIL", gate["status"])
        self.assertEqual(16, gate["sequence_count"])
        self.assertEqual(8, len(gate["failures"]))
        self.assertTrue(
            all(
                item.endswith(":MONOTONIC_APPROACH_PLUS_PERIODIC")
                for item in gate["failures"]
            )
        )
        self.assertTrue(
            all(
                item["persistent_visible_frame_count"] == 602
                for item in gate["summaries"]
            )
        )

    def test_target_or_trajectory_mutation_fails_g13(self) -> None:
        mutated = copy.deepcopy(self.r1_guards)
        mutated[0]["scene"]["designated_target"]["world_point_m"] = [
            10.0,
            10.0,
            4.0,
        ]
        mutated[1]["arms"][1]["trajectory"][100]["translation_m"][2] += 0.001
        gate = r1_validator.gate_g13_r1(mutated, self.trajectories)
        self.assertEqual("FAIL", gate["status"])
        self.assertTrue(gate["failures"])

    def test_declared_target_drift_fails_consumed_r1_g13(self) -> None:
        mutated = copy.deepcopy(self.r1_guards)
        mutated[0]["scene"]["designated_target"]["object_id"] = 8
        gate = r1_validator.gate_g13_r1(mutated, self.trajectories)
        self.assertEqual("FAIL", gate["status"])
        self.assertTrue(gate["failures"])

    def test_r1_validator_imports_no_generator_or_rcle(self) -> None:
        source = Path(r1_validator.__file__).read_text(encoding="utf-8")
        imported = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(
            any(
                "generator_geometry" in name
                or "ecological_response" in name
                or "rgb_algorithm" in name
                for name in imported
            )
        )

    def test_r1_lock_keeps_all_future_authority_closed(self) -> None:
        lock = r0_validator.load_json(r1_validator.DEFAULT_LOCK)
        self.assertIs(False, lock["formal_execution_authorized"])
        self.assertIs(False, lock["quality_calibration_authorized"])
        self.assertIs(False, lock["automatic_p2_authority"])
        self.assertIs(False, lock["firewall"]["rcle_output_read_or_run"])

    def test_consumed_r1_receipt_and_source_identities_are_immutable(self) -> None:
        receipt_path = (
            r1_validator.DEFAULT_EVIDENCE
            / "independent_geometry_validation_receipt.json"
        )
        receipt = r0_validator.load_json(receipt_path)
        self.assertEqual(
            "af00df05c115036ea31bb3d05addbebfcebad73122d2b354f7e52170c2277e9a",
            r0_validator.sha256_file(receipt_path),
        )
        self.assertEqual("VALID_FAIL_CLOSED", receipt["status"])
        self.assertEqual("INTERVENTION_NOT_EVALUABLE", receipt["terminal"])
        self.assertEqual("HOLD_P1", receipt["state"])
        self.assertEqual(["G13_MONOTONIC_APPROACH_TRUTH"], receipt["failed_gates"])
        self.assertEqual(
            "5be754efdcd04e4fcaa3fafc64a6b39ce92a5e36594e4b3fd2e141a62c5b9d8b",
            r0_validator.sha256_file(Path(r1_validator.__file__)),
        )
        self.assertEqual(
            "fd80e5b2d12f30fe7ba02c37e8311b9af37f52148362fc3b53cc8580a0166539",
            receipt["validator_source_sha256"],
        )
        self.assertNotEqual(
            r0_validator.sha256_file(Path(r1_validator.__file__)),
            receipt["validator_source_sha256"],
        )
        self.assertEqual(
            "521fd5fe523e9970c437c82e0dd5f3091a283e57de78e953db48b5d0cb0bfe48",
            r0_validator.sha256_file(r1_validator.AMENDMENT_PATH),
        )
        self.assertEqual(
            "b49efb5ef2d267dbcb50a3ff85f1890b4026272d77188a914dca2e9a91cc624d",
            r0_validator.sha256_file(r1_validator.DEFAULT_LOCK),
        )
        self.assertEqual(
            r0_validator.sha256_file(r1_validator.DEFAULT_LOCK),
            receipt["implementation_lock_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
