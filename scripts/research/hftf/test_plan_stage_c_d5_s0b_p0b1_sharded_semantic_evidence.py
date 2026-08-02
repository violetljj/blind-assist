from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0b_p0b1_sharded_semantic_evidence import (
    AGGREGATE_SHARD_MAXIMUM_BYTES,
    CANONICAL_ROOT,
    CAP_MANIFEST,
    CAP_MANIFEST_SHA256,
    CONTRACT_RELATIVE_PATH,
    CONTRACT_SCHEMA,
    CONTRACT_STATUS,
    CONTROL_ARTIFACT_MAXIMUM_BYTES,
    DraftNotExecutable,
    canonical_json_bytes,
    failure_allowed_set,
    load_contract_fail_closed,
    locked_closed_set,
    not_evaluable_closed_set,
    repo_root,
    shard_filename,
    shard_filenames,
    validate_frozen_capacity_constants,
)


class P0B1DraftSkeletonTest(unittest.TestCase):
    def test_cap_manifest_is_exact_and_self_consistent(self) -> None:
        validate_frozen_capacity_constants()
        self.assertEqual(18, len(CAP_MANIFEST))
        self.assertEqual(
            AGGREGATE_SHARD_MAXIMUM_BYTES,
            sum(int(row["maximum_shard_bytes"]) for row in CAP_MANIFEST),
        )
        self.assertEqual(
            CAP_MANIFEST_SHA256,
            hashlib.sha256(canonical_json_bytes(CAP_MANIFEST)).hexdigest(),
        )
        for index, row in enumerate(CAP_MANIFEST):
            self.assertEqual(index, row["manifest_index"])
            self.assertEqual(
                max(
                    CONTROL_ARTIFACT_MAXIMUM_BYTES,
                    512 * int(row["p0a_blob_bytes"]),
                ),
                row["maximum_shard_bytes"],
            )

    def test_shard_names_are_fixed_and_bounded(self) -> None:
        self.assertEqual("shard_000.json", shard_filename(0))
        self.assertEqual("shard_017.json", shard_filename(17))
        self.assertEqual(18, len(shard_filenames()))
        self.assertEqual(18, len(set(shard_filenames())))
        for invalid in (-1, 18, True):
            with self.assertRaises((TypeError, ValueError)):
                shard_filename(invalid)

    def test_terminal_closed_sets_are_disjoint_and_fail_closed(self) -> None:
        locked = locked_closed_set()
        not_evaluable = not_evaluable_closed_set()
        failure = failure_allowed_set()
        self.assertIn("index.json", locked)
        self.assertNotIn("not-evaluable.json", locked)
        self.assertIn("not-evaluable.json", not_evaluable)
        self.assertNotIn("index.json", not_evaluable)
        self.assertIn("failure.json", failure)
        self.assertTrue(set(shard_filenames()) <= failure)
        self.assertIn("result.json", failure)
        self.assertIn("index.json", failure)
        self.assertIn("not-evaluable.json", failure)

    def test_real_draft_contract_is_rejected_without_root_access(self) -> None:
        root = repo_root() / CANONICAL_ROOT
        self.assertFalse(root.exists())
        with self.assertRaisesRegex(DraftNotExecutable, "UNBOUND"):
            load_contract_fail_closed(repo_root() / CONTRACT_RELATIVE_PATH)
        self.assertFalse(root.exists())

    def test_executable_or_bound_receipt_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "contract.json"
            for value in (
                {
                    "schema": CONTRACT_SCHEMA,
                    "status": CONTRACT_STATUS,
                    "executable": True,
                    "implementation_receipts": {"status": "UNBOUND_TODO"},
                    "test_receipts": {"status": "UNBOUND_TODO"},
                },
                {
                    "schema": CONTRACT_SCHEMA,
                    "status": CONTRACT_STATUS,
                    "executable": False,
                    "implementation_receipts": {"status": "BOUND"},
                    "test_receipts": {"status": "UNBOUND_TODO"},
                },
            ):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(DraftNotExecutable):
                    load_contract_fail_closed(path)

    def test_missing_or_wrong_contract_never_becomes_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.json"
            with self.assertRaises(DraftNotExecutable):
                load_contract_fail_closed(missing)
            wrong = Path(temp) / "wrong.json"
            wrong.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(DraftNotExecutable, "schema"):
                load_contract_fail_closed(wrong)

    def test_top_level_and_nested_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            for text in (
                '{"schema":"a","schema":"b"}',
                '{"schema":"a","nested":{"status":1,"status":2}}',
            ):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(
                    DraftNotExecutable, "duplicate JSON object key"
                ):
                    load_contract_fail_closed(path)


if __name__ == "__main__":
    unittest.main()
