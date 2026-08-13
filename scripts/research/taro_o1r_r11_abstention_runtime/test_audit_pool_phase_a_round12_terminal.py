from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.research.taro_o1r_r11_abstention_runtime import (
    audit_pool_phase_a_round12_terminal as auditor,
)
from scripts.research.taro_o1r_r11_abstention_runtime import (
    validate_pool_phase_a as original,
)


class PhaseARound12RepairAuditTests(unittest.TestCase):
    def test_exact_round12_representation_closes_only_the_observed_mismatch(self) -> None:
        reconstructed = -0.4032351806954706
        stored = -0.403235180695
        self.assertNotEqual(stored, reconstructed)
        self.assertTrue(auditor.canonical_representation_matches(stored, reconstructed))
        self.assertFalse(auditor.canonical_representation_matches(stored + 1e-9, reconstructed))

    def test_negative_zero_and_nested_matrices_follow_canonical_json_representation(self) -> None:
        self.assertEqual(auditor.canonicalize_round12_numeric(-1e-15), 0.0)
        self.assertEqual(
            auditor.canonicalize_round12_numeric([[1.00000000000049, -2.00000000000051]]),
            [[1.0, -2.000000000001]],
        )

    def test_only_two_trajectory_numeric_fields_are_changed(self) -> None:
        members = {
            ("visit", "video", "token", "trajectory"): {
                "binding": {"sha256": "A" * 64},
                "camera_to_world_4x4": [[-0.4032351806954706]],
                "gravity_up_camera_xyz": [-0.07748568160183153],
                "sensor_timestamp_ns": 1,
                "max_source_timestamp_ns": 2,
            },
            ("visit", "video", "token", "color"): {
                "sha256": "B" * 64,
                "decoded_sha256": "C" * 64,
            },
        }
        before = copy.deepcopy(members)
        original_frame_count = original.FRAME_COUNT
        try:
            original.FRAME_COUNT = 1
            count = auditor._canonicalize_trajectory_member_bindings(members)
        finally:
            original.FRAME_COUNT = original_frame_count
        self.assertEqual(count, 1)
        self.assertEqual(members[("visit", "video", "token", "trajectory")]["binding"], before[("visit", "video", "token", "trajectory")]["binding"])
        self.assertEqual(members[("visit", "video", "token", "trajectory")]["sensor_timestamp_ns"], 1)
        self.assertEqual(members[("visit", "video", "token", "trajectory")]["max_source_timestamp_ns"], 2)
        self.assertEqual(members[("visit", "video", "token", "color")], before[("visit", "video", "token", "color")])
        self.assertEqual(members[("visit", "video", "token", "trajectory")]["camera_to_world_4x4"], [[-0.403235180695]])
        self.assertEqual(members[("visit", "video", "token", "trajectory")]["gravity_up_camera_xyz"], [-0.077485681602])

    def test_patch_context_restores_original_function_after_failure(self) -> None:
        stats: dict[str, int] = {}
        with self.assertRaisesRegex(RuntimeError, "fixture"), auditor._round12_inventory_binding_repair(stats):
            self.assertIsNot(original._inventory_member_bindings, auditor._ORIGINAL_INVENTORY_MEMBER_BINDINGS)
            raise RuntimeError("fixture")
        self.assertIs(original._inventory_member_bindings, auditor._ORIGINAL_INVENTORY_MEMBER_BINDINGS)

    def test_source_contains_no_tolerance_or_producer_import(self) -> None:
        source = Path(auditor.__file__).read_text(encoding="utf-8")
        self.assertNotIn("run_pool_phase_a", source)
        executable = source.replace('"""Compare exactly after round-12 serialization; no epsilon or tolerance is used."""', "")
        executable = executable.replace('"tolerance": None', "").replace('"epsilon": None', "")
        self.assertNotIn("isclose", executable)
        self.assertNotIn("allclose", executable)
        self.assertNotIn("atol", executable)
        self.assertNotIn("rtol", executable)
        self.assertNotIn("abs(", executable)
        self.assertEqual(auditor.REPAIRED_FIELDS, ("camera_to_world_4x4", "gravity_up_camera_xyz"))

    def test_nonfinite_and_unrelated_types_fail_closed(self) -> None:
        with self.assertRaises(auditor.RepairAuditError):
            auditor.canonicalize_round12_numeric(float("nan"))
        with self.assertRaises(auditor.RepairAuditError):
            auditor.canonicalize_round12_numeric({"not": "a numeric structure"})

    def test_resealed_numeric_contract_mutation_is_rejected(self) -> None:
        record = auditor._load_json(auditor._repo_path(auditor.REPAIR_RECEIPT_RELATIVE))
        self.assertEqual(auditor._validate_repair_receipt(record)["status"], auditor.REPAIR_STATUS)
        mutated = copy.deepcopy(record)
        mutated["numeric_contract"]["repaired_fields"].append("sensor_timestamp_ns")
        mutated.pop("content_sha256")
        mutated["content_sha256"] = original.adapter.canonical_sha256(mutated)
        with self.assertRaises(auditor.RepairAuditError):
            auditor._validate_repair_receipt(mutated)

    def test_output_root_collision_fails_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collision = Path(directory).resolve()
            real_repo_path = auditor._repo_path

            def redirected_repo_path(relative: str) -> Path:
                if relative == auditor.OUTPUT_ROOT_RELATIVE:
                    return collision
                return real_repo_path(relative)

            with mock.patch.object(auditor, "_repo_path", side_effect=redirected_repo_path), self.assertRaisesRegex(
                auditor.RepairAuditError,
                "fresh repair output root already exists",
            ):
                auditor.audit_same_sealed_root(
                    real_repo_path(auditor.REPAIR_RECEIPT_RELATIVE),
                    collision,
                )

    def test_single_file_result_root_is_published_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "formal-root"
            value = {"schema": auditor.RESULT_SCHEMA, "status": "FIXTURE"}
            output_bytes, output_sha256 = auditor._publish_single_file_root_atomic(target, value)
            output = target / "post-result-audit.json"
            self.assertEqual([path.name for path in target.iterdir()], [output.name])
            self.assertEqual(output.stat().st_size, output_bytes)
            self.assertEqual(auditor._sha256_file(output), output_sha256)
            self.assertEqual(auditor._validate_seal(auditor._load_json(output), auditor.RESULT_SCHEMA)["status"], "FIXTURE")

    def test_fsync_failure_never_publishes_formal_root_or_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "formal-root"
            value = {"schema": auditor.RESULT_SCHEMA, "status": "FIXTURE"}
            with mock.patch.object(auditor.os, "fsync", side_effect=OSError("fixture fsync")), self.assertRaisesRegex(
                OSError,
                "fixture fsync",
            ):
                auditor._publish_single_file_root_atomic(target, value)
            self.assertFalse(target.exists())
            self.assertEqual(list(parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
