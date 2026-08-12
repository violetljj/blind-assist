from __future__ import annotations

import copy
import importlib
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest import mock

from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared_r7
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a as r10_phase_a
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_phase_a as runner


def _member(role: str) -> runner.PhaseAMemberRef:
    return runner.PhaseAMemberRef(role=role, path=f"{role}/member", bytes=1, crc32="00000000")


def _frame(*, color_role: str = "color") -> runner.PhaseAFrameRef:
    return runner.PhaseAFrameRef(
        parent_id="p",
        video_id="v",
        timestamp_token="1.000",
        physical_frame_id="v:1.000",
        upsampling_archive=Path("up.zip"),
        intrinsics_archive=Path("intr.zip"),
        trajectory_path=Path("wide.traj"),
        container_bindings={},
        trajectory_rows=(),
        payload_members={
            "color": _member(color_role),
            "intrinsics": _member("intrinsics"),
            "lowres_depth": _member("lowres_depth"),
            "confidence": _member("confidence"),
        },
    )


class _UntouchedBundle:
    def __init__(self) -> None:
        self.touched = False

    def getinfo(self, _path: str):
        self.touched = True
        raise AssertionError("ZIP lookup must not occur")


class PoolPhaseATests(unittest.TestCase):
    def test_exact_r11_constants_and_file_formula(self) -> None:
        self.assertEqual(runner.PARENT_COUNT, 48)
        self.assertEqual(runner.FRAME_COUNT, 1043)
        self.assertEqual(runner.QUERY_COUNT, 9387)
        self.assertEqual(sum(runner.FROZEN_FRAME_COUNTS), runner.FRAME_COUNT)
        self.assertEqual(runner.PRE_MANIFEST_FILE_COUNT, 5219)
        self.assertEqual(runner.EXPECTED_AUTHORITY["highres_depth_member_payload_read"], False)
        self.assertEqual(runner.EXPECTED_AUTHORITY["source_only_parent_scoring"], False)
        self.assertEqual(runner.EXPECTED_AUTHORITY["top24_selection"], False)

    def test_inventory_evidence_is_exact_without_source_frame_read(self) -> None:
        with mock.patch.object(runner, "_load_frames", side_effect=AssertionError("source access forbidden")):
            inventory = runner._verify_inventory_evidence()
        self.assertEqual(inventory["parent_count"], 48)
        self.assertEqual(inventory["exact_pose_bounded_frame_count"], 1043)
        self.assertEqual(len(runner._inventory_frame_keys(inventory)), 1043)

    def test_seal_schema_and_mutation_fail_closed(self) -> None:
        schema = "blindassist.taro.o1r.r11_test.v1"
        record = runner._seal({"schema": schema, "value": 1})
        self.assertEqual(runner._validate_seal(record, schema), record)
        mutated = copy.deepcopy(record)
        mutated["value"] = 2
        with self.assertRaises(runner.FreshPhaseAError):
            runner._validate_seal(mutated, schema)
        with self.assertRaises(runner.FreshPhaseAError):
            runner._validate_seal(record, "blindassist.taro.o1r.r10_test.v1")

    def test_phase_a_frame_cannot_retain_highres_capability(self) -> None:
        members = _frame().payload_members
        members["highres_depth"] = _member("highres_depth")
        with self.assertRaisesRegex(runner.FreshPhaseAError, "forbidden or missing"):
            runner.PhaseAFrameRef(
                "p", "v", "1.000", "v:1.000", Path("u"), Path("i"), Path("t"), {}, (), members
            )

    def test_forbidden_roles_fail_before_zip_lookup_and_record_attempt(self) -> None:
        for phase, role in (
            ("CANDIDATE", "highres_depth"),
            ("CANDIDATE", "lowres_depth"),
            ("SOURCE_FEATURE", "color"),
            ("SOURCE_FEATURE", "intrinsics"),
        ):
            with self.subTest(phase=phase, role=role):
                bundle = _UntouchedBundle()
                ledger = runner.PayloadReadLedger()
                with self.assertRaises(runner.FreshPhaseAError):
                    runner._read_allowed_member(bundle, _frame(), role, phase, ledger)
                self.assertFalse(bundle.touched)
                self.assertEqual(ledger.attempts_by_role[role], 1)
                self.assertEqual(ledger.completed_by_role[role], 0)

    def test_role_path_mismatch_fails_before_zip_lookup(self) -> None:
        bundle = _UntouchedBundle()
        ledger = runner.PayloadReadLedger()
        with self.assertRaisesRegex(runner.FreshPhaseAError, "role/path"):
            runner._read_allowed_member(bundle, _frame(color_role="confidence"), "color", "CANDIDATE", ledger)
        self.assertFalse(bundle.touched)

    def test_allowed_reader_validates_payload_and_completed_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("color/member", b"x")
            with zipfile.ZipFile(archive) as source:
                info = source.getinfo("color/member")
                frame = _frame()
                frame.payload_members["color"] = runner.PhaseAMemberRef(
                    role="color",
                    path="color/member",
                    bytes=1,
                    crc32=f"{info.CRC:08X}",
                )
                ledger = runner.PayloadReadLedger()
                payload, binding = runner._read_allowed_member(source, frame, "color", "CANDIDATE", ledger)
            self.assertEqual(payload, b"x")
            self.assertEqual(binding["bytes"], 1)
            self.assertEqual(ledger.attempts_by_role, Counter({"color": 1}))
            self.assertEqual(ledger.completed_by_role, Counter({"color": 1}))
            self.assertEqual(ledger.bytes_by_role, Counter({"color": 1}))

    def test_factor_pair_enforces_candidate_subset_and_abstention_identity(self) -> None:
        def bundle(states: list[str], *, candidate: bool = False) -> dict:
            rows = [
                {"query_id": f"q{i}", "grid_index": i, "state": state}
                for i, state in enumerate(states)
            ]
            value = {
                "physical_frame_id": "v:1.000",
                "source_frame_record_sha256": "A" * 64,
                "query_results": rows,
            }
            if candidate:
                base_count = sum(state == "OCCUPIED_OBSERVED" for state in states)
                value.update(
                    {
                        "base_positive_count": base_count + 1,
                        "candidate_positive_count": base_count,
                        "abstained_base_positive_count": 1,
                    }
                )
            return value

        base_states = ["OCCUPIED_OBSERVED", "OCCUPIED_OBSERVED"] + ["UNKNOWN"] * 7
        candidate_states = ["OCCUPIED_OBSERVED"] + ["UNKNOWN"] * 8
        base = bundle(base_states)
        candidate = bundle(candidate_states, candidate=True)
        with mock.patch.object(runner.r7_positive, "validate_positive_occupancy_factor", return_value=base), mock.patch.object(
            runner.abstention_candidate, "validate_abstention_bundle", return_value=candidate
        ):
            base_counts, candidate_counts, abstained = runner._validate_factor_pair(base, candidate)
        self.assertEqual(base_counts["OCCUPIED_OBSERVED"], 2)
        self.assertEqual(candidate_counts["OCCUPIED_OBSERVED"], 1)
        self.assertEqual(abstained, 1)

        invalid = bundle(["UNKNOWN", "UNKNOWN", "OCCUPIED_OBSERVED"] + ["UNKNOWN"] * 6, candidate=True)
        invalid["base_positive_count"] = 2
        with mock.patch.object(runner.r7_positive, "validate_positive_occupancy_factor", return_value=base), mock.patch.object(
            runner.abstention_candidate, "validate_abstention_bundle", return_value=invalid
        ):
            with self.assertRaisesRegex(runner.FreshPhaseAError, "not a subset"):
                runner._validate_factor_pair(base, invalid)

    def test_runtime_bindings_freeze_candidate_source_r7_r11_and_next_selector(self) -> None:
        required = {
            "DEPTHART_RUNTIME",
            "CANDIDATE_INPUT_RUNTIME",
            "PROSPECTIVE_RUNTIME",
            "R6_REDUCER_RUNTIME",
            "LOCKED_UNCERTAINTY_ARTIFACT",
            "LOCKED_UNCERTAINTY_RECEIPT",
            "R7_SOURCE_FEATURE_RUNTIME",
            "R7_POSITIVE_FACTOR_RUNTIME",
            "R11_ABSTENTION_RUNTIME",
            "R9_SELECTOR_RUNTIME",
            "R9_SELECTOR_ARTIFACT",
            "EVIDENCE_WRITER",
        }
        self.assertTrue(required.issubset(runner.EXPECTED_BINDINGS))
        self.assertEqual(runner.EXPECTED_NEXT_STAGE_SELECTOR["rule_id"], "02CE016D6B0011F0")
        self.assertFalse(runner.EXPECTED_NEXT_STAGE_SELECTOR["phase_a_scoring_performed"])

    def test_import_order_does_not_mutate_r7_or_r10_globals(self) -> None:
        before = (shared_r7.PARENT_COUNT, shared_r7.FRAME_COUNT, r10_phase_a.PARENT_COUNT, r10_phase_a.FRAME_COUNT)
        importlib.reload(runner)
        after = (shared_r7.PARENT_COUNT, shared_r7.FRAME_COUNT, r10_phase_a.PARENT_COUNT, r10_phase_a.FRAME_COUNT)
        self.assertEqual(before, after)
        self.assertEqual(after, (8, 170, 32, 710))

    def test_runner_has_no_r6_faro_reader_or_r10_orchestration_import(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("read_faro_frame", source)
        self.assertNotIn("r6_confirmation_io", source)
        self.assertNotIn("taro_o1r_r10_clear_runtime import run_pool_phase_a", source)


if __name__ == "__main__":
    unittest.main()
