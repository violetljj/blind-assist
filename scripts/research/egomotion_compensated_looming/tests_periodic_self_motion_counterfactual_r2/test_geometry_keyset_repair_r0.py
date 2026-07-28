from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry as r0_generator,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry_r1 as r1_generator,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry_r2_keyset_repair_r0 as keyset_generator,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_geometry_independent as r0_validator,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_geometry_independent_r2_keyset_repair_r0 as keyset_validator,
)


class GeometryKeysetRepairR0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.r0_records = r0_validator.load_jsonl(
            keyset_validator.R0_EVIDENCE / "all_seed_geometry_manifest.jsonl"
        )
        cls.r2_records = r0_validator.load_jsonl(
            keyset_validator.DEFAULT_EVIDENCE
            / "all_seed_geometry_manifest.jsonl"
        )
        cls.trajectories = r0_validator.load_json(
            keyset_validator.DEFAULT_EVIDENCE / "trajectory_manifest.json"
        )
        cls.r2_guards = [
            item
            for item in cls.r2_records
            if item["record_type"] == "guardrail_cluster"
        ]

    def test_r0_terminal_is_immutable(self) -> None:
        receipt_path = (
            keyset_validator.R0_EVIDENCE
            / "independent_geometry_validation_receipt.json"
        )
        receipt = r0_validator.load_json(receipt_path)
        self.assertEqual(
            keyset_validator.EXPECTED_R0_RECEIPT_SHA256,
            r0_validator.sha256_file(receipt_path),
        )
        self.assertEqual("INTERVENTION_NOT_EVALUABLE", receipt["terminal"])
        self.assertEqual("HOLD_P1", receipt["state"])

    def test_r0_receipt_keyset_is_exact_and_wrong_alias_is_rejected(self) -> None:
        receipt = r0_validator.load_json(
            keyset_validator.R0_EVIDENCE
            / "independent_geometry_validation_receipt.json"
        )
        self.assertTrue(
            keyset_validator._has_exact_r0_evidence_keyset(receipt)
        )
        mutated = copy.deepcopy(receipt)
        hashes = mutated["evidence_sha256"]
        hashes["generator_receipt.json"] = hashes.pop("producer_receipt.json")
        self.assertFalse(
            keyset_validator._has_exact_r0_evidence_keyset(mutated)
        )

    def test_radial_gate_is_derived_from_unchanged_20_percent(self) -> None:
        amendment = r0_validator.load_json(keyset_validator.AMENDMENT_PATH)
        self.assertEqual(
            math.log(1.20),
            amendment["machine_gate_lock"][
                "integrated_endpoint_log_radial_expansion_gte"
            ],
        )
        self.assertAlmostEqual(math.log(1.20), keyset_validator.LOG_1P20)

    def test_all_main_records_are_byte_identical(self) -> None:
        r0_main = [
            item for item in self.r0_records if item["record_type"] == "main_cluster"
        ]
        r2_main = [
            item for item in self.r2_records if item["record_type"] == "main_cluster"
        ]
        self.assertEqual(80, len(r0_main))
        self.assertEqual(
            [r0_validator.canonical_bytes(item) for item in r0_main],
            [r0_validator.canonical_bytes(item) for item in r2_main],
        )

    def test_guard_seeds_and_trajectories_are_unchanged(self) -> None:
        r0_guard = {
            item["cluster_id"]: item
            for item in self.r0_records
            if item["record_type"] == "guardrail_cluster"
        }
        for repaired in self.r2_guards:
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

    def test_repaired_g13_passes_all_sixteen_arms(self) -> None:
        gate = keyset_validator.gate_g13_r2(
            self.r2_guards, self.trajectories
        )
        self.assertEqual("PASS", gate["status"])
        self.assertEqual(16, gate["sequence_count"])
        self.assertEqual([], gate["failures"])
        self.assertTrue(
            all(
                item["persistent_visible_frame_count"] == 602
                for item in gate["summaries"]
            )
        )

    def test_target_or_trajectory_mutation_fails_g13(self) -> None:
        mutated = copy.deepcopy(self.r2_guards)
        mutated[0]["scene"]["designated_target"]["world_point_m"] = [
            10.0,
            10.0,
            4.0,
        ]
        mutated[1]["arms"][1]["trajectory"][100]["translation_m"][2] += 0.001
        gate = keyset_validator.gate_g13_r2(mutated, self.trajectories)
        self.assertEqual("FAIL", gate["status"])
        self.assertTrue(gate["failures"])

    def test_guard_layout_or_declared_target_drift_is_invalid(self) -> None:
        mutated = copy.deepcopy(self.r2_guards)
        mutated[0]["scene"]["designated_target"]["object_id"] = 8
        mutated[1]["scene"]["world"]["objects"][0]["bounds_xy_m"][0] += 0.01
        errors: list[str] = []
        keyset_validator._validate_guard_scene_contract(mutated, errors)
        self.assertTrue(
            any(item.startswith("R1_GUARD_TARGET_IDENTITY:") for item in errors)
        )
        self.assertTrue(
            any(item.startswith("R1_GUARD_NEAR_BOUNDS:") for item in errors)
        )

    def test_r2_validator_imports_no_generator_or_rcle(self) -> None:
        source = Path(keyset_validator.__file__).read_text(encoding="utf-8")
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

    def test_r2_lock_keeps_all_future_authority_closed(self) -> None:
        lock = r0_validator.load_json(keyset_validator.DEFAULT_LOCK)
        self.assertIs(False, lock["formal_execution_authorized"])
        self.assertIs(False, lock["quality_calibration_authorized"])
        self.assertIs(False, lock["automatic_p2_authority"])
        self.assertIs(False, lock["firewall"]["rcle_output_read_or_run"])

    def test_r1_failed_receipt_is_immutable(self) -> None:
        path = (
            keyset_validator.R1_EVIDENCE
            / "independent_geometry_validation_receipt.json"
        )
        receipt = r0_validator.load_json(path)
        self.assertEqual(
            keyset_validator.EXPECTED_R1_FAILED_RECEIPT_SHA256,
            r0_validator.sha256_file(path),
        )
        self.assertEqual("INTERVENTION_NOT_EVALUABLE", receipt["terminal"])
        self.assertEqual("HOLD_P1", receipt["state"])
        self.assertEqual(["G13_MONOTONIC_APPROACH_TRUTH"], receipt["failed_gates"])

    def test_r2_failed_receipt_is_immutable(self) -> None:
        path = (
            keyset_validator.R2_EVIDENCE
            / "independent_geometry_validation_receipt.json"
        )
        receipt = r0_validator.load_json(path)
        self.assertEqual(
            keyset_validator.EXPECTED_R2_FAILED_RECEIPT_SHA256,
            r0_validator.sha256_file(path),
        )
        self.assertEqual("INVALID", receipt["status"])
        self.assertEqual("INTERVENTION_NOT_EVALUABLE", receipt["terminal"])
        self.assertEqual("HOLD_P1", receipt["state"])
        self.assertEqual(14, receipt["gate_pass_count"])
        self.assertEqual(
            ["R0_RECEIPT_EVIDENCE_HASH_KEYSET"], receipt["errors"]
        )

    def test_all_records_are_byte_identical_to_r2(self) -> None:
        r1_path = (
            keyset_validator.R2_EVIDENCE
            / "all_seed_geometry_manifest.jsonl"
        )
        r2_path = (
            keyset_validator.DEFAULT_EVIDENCE
            / "all_seed_geometry_manifest.jsonl"
        )
        self.assertEqual(r1_path.read_bytes(), r2_path.read_bytes())

    def test_formal_receipt_is_exclusive_create(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            receipt = {"status": "VALID"}
            keyset_validator._write_receipt_exclusive(path, receipt)
            self.assertEqual(
                r0_validator.canonical_bytes(receipt), path.read_bytes()
            )
            with self.assertRaises(FileExistsError):
                keyset_validator._write_receipt_exclusive(path, receipt)

    def test_generator_output_directory_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new_evidence"
            keyset_generator.produce(output)
            self.assertTrue(output.is_dir())
            with self.assertRaises(FileExistsError):
                keyset_generator.produce(output)

    def test_guard_replay_mutation_fails_g14(self) -> None:
        replay = r0_validator.load_json(
            keyset_validator.DEFAULT_EVIDENCE
            / "deterministic_replay_ledger.json"
        )
        guard_replay = r0_validator.load_json(
            keyset_validator.DEFAULT_EVIDENCE
            / "guard_scene_replay_ledger.json"
        )
        fixtures = r0_validator.load_json(
            keyset_validator.DEFAULT_EVIDENCE
            / "analytic_fixture_ledger.json"
        )
        mutated = copy.deepcopy(guard_replay)
        mutated["items"][0]["second_scene_sha256"] = "0" * 64
        mutated["items"][0]["match"] = True
        mutated["mismatch_count"] = 0
        gate = keyset_validator.gate_g14_r2(
            replay, mutated, fixtures, self.r2_guards
        )
        self.assertEqual("FAIL", gate["status"])
        self.assertEqual(1, gate["guard_recomputed_mismatch_count"])

    def test_pairing_and_geometry_hash_mutations_fail(self) -> None:
        main = [
            item
            for item in self.r2_records
            if item["record_type"] == "main_cluster"
        ]
        pairing_mutation = copy.deepcopy(main)
        pairing_mutation[0]["arms"][1]["quality_operator_status"] = "DRIFT"
        self.assertEqual(
            "FAIL", keyset_validator.gate_g11_r2(pairing_mutation)["status"]
        )
        hash_mutation = copy.deepcopy(main)
        hash_mutation[0]["arms"][0]["depth_sha256"] = "0" * 64
        self.assertEqual(
            "FAIL",
            keyset_validator.gate_g12_r2(
                hash_mutation, self.trajectories
            )["status"],
        )

    def test_reference_hash_mutation_fails_g01(self) -> None:
        mutated = copy.deepcopy(self.r2_guards[:1])
        mutated[0]["reference_metrics"]["reference_depth_sha256"] = "0" * 64
        g01, _ = keyset_validator.gate_g01_g02_r2([], mutated)
        self.assertEqual("FAIL", g01["status"])

    def test_r2_producer_replay_is_exact(self) -> None:
        records = r0_validator.load_jsonl(
            keyset_validator.DEFAULT_EVIDENCE
            / "all_seed_geometry_manifest.jsonl"
        )
        replay = keyset_generator._build_guard_replay(records)
        self.assertEqual(0, replay["mismatch_count"])
        self.assertEqual(8, len(replay["items"]))


if __name__ == "__main__":
    unittest.main()
