from __future__ import annotations

import ast
import hashlib
import json
import os
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
    EVIDENCE_INVALID,
    EVIDENCE_LOCKED,
    EVIDENCE_NOT_EVALUABLE,
    EXECUTABLE_CONTRACT_STATUS,
    FAILURE_FILENAME,
    INDEX_FILENAME,
    NOT_EVALUABLE_FILENAME,
    RESULT_FILENAME,
    TEST_ONLY_TOOLKIT_COMMIT,
    TEST_ONLY_TOOLKIT_REPOSITORY,
    _FORMAL_EXECUTION_GATE,
    _TEST_EXECUTION_GATE,
    _extract_sharded_evidence,
    canonical_json_bytes,
    detect_source_encoding,
    _execute_with_validated_context,
    failure_allowed_set,
    load_contract_fail_closed,
    locked_closed_set,
    not_evaluable_closed_set,
    repo_root,
    shard_filename,
    shard_filenames,
    validate_executable_contract,
    validate_frozen_capacity_constants,
)
from plan_stage_c_d5_s0a_tartanground_catalog import (
    write_bytes_exclusive_fsync,
)
from test_plan_stage_c_d5_s0b_p0b_provider_semantic_evidence import (
    FakeGit,
    make_closure,
)
from validate_stage_c_d5_s0b_p0b1_sharded_semantic_evidence import (
    TerminalValidationError,
    validate_terminal,
)


def exact_sized_source(size: int, *, invalid: bool = False) -> bytes:
    prefix = (
        b"def broken(:\n"
        if invalid
        else b"x='u'\ny=f(x)\nimport os\n"
    )
    remaining = size - len(prefix)
    if remaining == 0:
        return prefix
    if remaining == 1:
        return prefix + b"\n"
    return prefix + b"#" + (b"a" * (remaining - 2)) + b"\n"


def make_exact_sources(
    *, invalid_index: int | None = None
) -> dict[str, bytes]:
    return {
        row["path"]: exact_sized_source(
            int(row["p0a_blob_bytes"]),
            invalid=index == invalid_index,
        )
        for index, row in enumerate(CAP_MANIFEST)
    }


class InterruptingWriter:
    def __init__(self, target: str) -> None:
        self.target = target
        self.interrupted = False

    def __call__(self, path: Path, payload: bytes) -> None:
        if path.name == self.target and not self.interrupted:
            self.interrupted = True
            with path.open("xb") as handle:
                handle.write(payload[: max(1, len(payload) // 2)])
                handle.flush()
                os.fsync(handle.fileno())
            raise OSError(f"injected write interruption: {self.target}")
        write_bytes_exclusive_fsync(path, payload)


class P0B1DraftSkeletonTest(unittest.TestCase):
    def make_exact_fixture(
        self,
        base: Path,
        *,
        invalid_index: int | None = None,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        Path,
        FakeGit,
    ]:
        sources = make_exact_sources(invalid_index=invalid_index)
        closure = make_closure(sources)
        contract = json.loads(
            (repo_root() / CONTRACT_RELATIVE_PATH).read_text(
                encoding="utf-8"
            )
        )
        contract["source_authority"][
            "toolkit_repository"
        ] = TEST_ONLY_TOOLKIT_REPOSITORY
        contract["source_authority"][
            "toolkit_commit"
        ] = TEST_ONLY_TOOLKIT_COMMIT
        toolkit = base / "toolkit"
        toolkit.mkdir()
        return (
            contract,
            closure,
            toolkit,
            FakeGit(closure, sources),
        )

    def build_fake_shards(
        self,
    ) -> tuple[dict[str, object], list[dict[str, object]], FakeGit]:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            contract = json.loads(
                (repo_root() / CONTRACT_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            contract["source_authority"][
                "toolkit_repository"
            ] = TEST_ONLY_TOOLKIT_REPOSITORY
            contract["source_authority"][
                "toolkit_commit"
            ] = TEST_ONLY_TOOLKIT_COMMIT
            contract["source_authority"]["exact_ordered_blob_count"] = len(
                closure["observation"]["closure_rows"]
            )
            fake = FakeGit(closure)
            terminal, result = _extract_sharded_evidence(
                contract,
                closure,
                base,
                git_runner=fake,
                source_gate=_TEST_EXECUTION_GATE,
            )
        self.assertEqual(EVIDENCE_LOCKED, terminal)
        self.assertIsInstance(result, list)
        return contract, result, fake

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

    def test_real_contract_gate_never_opens_canonical_root(self) -> None:
        root = repo_root() / CANONICAL_ROOT
        self.assertFalse(root.exists())
        contract_path = repo_root() / CONTRACT_RELATIVE_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract["status"] == EXECUTABLE_CONTRACT_STATUS:
            context = validate_executable_contract(
                contract_path, verify_git=False
            )
            self.assertTrue(context["contract"]["executable"])
        else:
            with self.assertRaisesRegex(DraftNotExecutable, "UNBOUND"):
                load_contract_fail_closed(contract_path)
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

    def test_fake_source_builds_exact_deduplicated_shards(self) -> None:
        contract, shards, _ = self.build_fake_shards()
        self.assertEqual(2, len(shards))
        schemas = contract["exact_artifact_schemas"]
        for shard in shards:
            self.assertEqual(
                set(schemas["shard_json_exact_keys"]), set(shard)
            )
            for row in shard["node_receipts"]:
                self.assertEqual(
                    set(schemas["node_receipt_exact_keys"]), set(row)
                )
                self.assertIsInstance(row["canonical_ast_object"], dict)
            for row in shard["expression_records"]:
                self.assertEqual(
                    set(schemas["expression_record_exact_keys"]), set(row)
                )
                self.assertNotIn("source_segment", row)
                self.assertNotIn("canonical_ast_dump", row)
            for key, schema_key in (
                ("call_site_records", "call_site_record_exact_keys"),
                (
                    "assignment_records",
                    "assignment_record_exact_keys",
                ),
                (
                    "string_literal_records",
                    "string_literal_record_exact_keys",
                ),
            ):
                for row in shard[key]:
                    self.assertEqual(set(schemas[schema_key]), set(row))

    def test_fake_extraction_reads_each_blob_once_without_network(self) -> None:
        _, shards, fake = self.build_fake_shards()
        reads = [
            call for call in fake.calls if call[:2] == ("cat-file", "blob")
        ]
        self.assertEqual(len(shards), len(reads))
        self.assertFalse(
            any("fetch" in call or "http" in " ".join(call) for call in fake.calls)
        )

    def test_fake_shards_respect_precomputed_caps(self) -> None:
        _, shards, _ = self.build_fake_shards()
        aggregate = 0
        for shard in shards:
            size = len(canonical_json_bytes(shard))
            aggregate += size
            self.assertLessEqual(
                size,
                CAP_MANIFEST[shard["manifest_index"]][
                    "maximum_shard_bytes"
                ],
            )
        self.assertLessEqual(aggregate, AGGREGATE_SHARD_MAXIMUM_BYTES)

    def test_exact_18_phase_barrier_is_receipts_then_encoding_then_parse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract, closure, toolkit, fake = (
                self.make_exact_fixture(base)
            )
            events: list[str] = []

            def runner(args: list[str], cwd: Path) -> bytes:
                if args[:2] == ["cat-file", "blob"]:
                    events.append("blob")
                return fake(args, cwd)

            def detector(blob: bytes) -> str:
                events.append("encoding")
                return detect_source_encoding(blob)

            def parser(source: str, filename: str) -> ast.AST:
                events.append("parse")
                return ast.parse(source, filename=filename)

            terminal, shards = _extract_sharded_evidence(
                contract,
                closure,
                toolkit,
                git_runner=runner,
                encoding_detector=detector,
                parser=parser,
                source_gate=_TEST_EXECUTION_GATE,
            )
        self.assertEqual(EVIDENCE_LOCKED, terminal)
        self.assertEqual(18, len(shards))
        self.assertEqual(18, events.count("blob"))
        self.assertEqual(18, events.count("encoding"))
        self.assertEqual(18, events.count("parse"))
        self.assertLess(
            max(index for index, item in enumerate(events) if item == "blob"),
            min(
                index
                for index, item in enumerate(events)
                if item == "encoding"
            ),
        )
        self.assertLess(
            max(
                index
                for index, item in enumerate(events)
                if item == "encoding"
            ),
            min(index for index, item in enumerate(events) if item == "parse"),
        )

    def test_dynamic_not_evaluable_reads_zero_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract, closure, toolkit, fake = (
                self.make_exact_fixture(base)
            )
            closure["observation"]["dynamic_import_call_count"] = 1
            terminal, payload = _extract_sharded_evidence(
                contract,
                closure,
                toolkit,
                git_runner=fake,
                source_gate=_TEST_EXECUTION_GATE,
            )
        self.assertEqual(EVIDENCE_NOT_EVALUABLE, terminal)
        self.assertEqual(0, payload["source_blob_read_count"])
        self.assertEqual([], fake.calls)
        self.assertIsNone(payload["failed_manifest_index"])

    def test_syntax_not_evaluable_binds_failed_row_and_full_receipts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract, closure, toolkit, fake = (
                self.make_exact_fixture(base, invalid_index=7)
            )
            terminal, payload = _extract_sharded_evidence(
                contract,
                closure,
                toolkit,
                git_runner=fake,
                source_gate=_TEST_EXECUTION_GATE,
            )
        self.assertEqual(EVIDENCE_NOT_EVALUABLE, terminal)
        self.assertEqual(18, payload["source_blob_read_count"])
        self.assertEqual(18, payload["object_receipt_count"])
        self.assertEqual(7, payload["failed_manifest_index"])
        self.assertEqual(
            CAP_MANIFEST[7]["path"], payload["failed_path"]
        )
        self.assertEqual(7, payload["parse_receipt_count"])
        self.assertIsNotNone(payload["detected_source_encoding"])

    def test_locked_execution_reopens_and_validates_durable_terminal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract, closure, toolkit, fake = (
                self.make_exact_fixture(base)
            )
            root = base / "locked-root"
            result = _execute_with_validated_context(
                contract,
                closure,
                toolkit,
                root,
                git_runner=fake,
                execution_gate=_TEST_EXECUTION_GATE,
            )
            validated = validate_terminal(
                root,
                contract,
                repo_root() / CONTRACT_RELATIVE_PATH,
                closure,
            )
        self.assertEqual(EVIDENCE_LOCKED, result["terminal"])
        self.assertEqual(result, validated)
        self.assertEqual(
            "CONSUMED_SOURCE_RECOVERY_NOT_FRESH_VALIDATION",
            result["source_evidence_role"],
        )
        self.assertTrue(
            all(value is False for value in result["claim_ceiling"].values())
        )

    def test_not_evaluable_execution_binds_exact_failure_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract, closure, toolkit, fake = (
                self.make_exact_fixture(base, invalid_index=4)
            )
            root = base / "ne-root"
            result = _execute_with_validated_context(
                contract,
                closure,
                toolkit,
                root,
                git_runner=fake,
                execution_gate=_TEST_EXECUTION_GATE,
            )
            evidence = json.loads(
                (root / NOT_EVALUABLE_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            validated = validate_terminal(
                root,
                contract,
                repo_root() / CONTRACT_RELATIVE_PATH,
                closure,
            )
        self.assertEqual(EVIDENCE_NOT_EVALUABLE, result["terminal"])
        self.assertEqual(result, validated)
        self.assertEqual(4, evidence["manifest_index"])
        self.assertEqual(CAP_MANIFEST[4]["path"], evidence["source_path"])
        self.assertEqual("AST_PARSE_FAILED", evidence["parse_status"])

    def test_raw_execution_core_rejects_missing_gate_and_canonical_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract, closure, toolkit, fake = (
                self.make_exact_fixture(base)
            )
            with self.assertRaisesRegex(
                DraftNotExecutable, "validated executable authority"
            ):
                _execute_with_validated_context(
                    contract,
                    closure,
                    toolkit,
                    base / "ungated",
                    git_runner=fake,
                )
            with self.assertRaisesRegex(
                DraftNotExecutable, "temporary paths"
            ):
                _execute_with_validated_context(
                    contract,
                    closure,
                    toolkit,
                    repo_root() / CANONICAL_ROOT,
                    git_runner=fake,
                    execution_gate=_TEST_EXECUTION_GATE,
                )
            formal_root = base / "forged-formal"
            with self.assertRaises(
                (DraftNotExecutable, ValueError)
            ):
                _execute_with_validated_context(
                    contract,
                    closure,
                    toolkit,
                    formal_root,
                    git_runner=fake,
                    execution_gate=_FORMAL_EXECUTION_GATE,
                    execution_commit="0" * 40,
                )
            self.assertFalse(formal_root.exists())
            self.assertEqual([], fake.calls)

    def test_required_failure_injections_validate_raw_partial_state(
        self,
    ) -> None:
        cases = (
            (0, NOT_EVALUABLE_FILENAME),
            (0, RESULT_FILENAME),
            (None, INDEX_FILENAME),
            (None, RESULT_FILENAME),
        )
        for invalid_index, target in cases:
            with self.subTest(target=target, invalid=invalid_index):
                with tempfile.TemporaryDirectory() as temp:
                    base = Path(temp)
                    (
                        contract,
                        closure,
                        toolkit,
                        fake,
                    ) = self.make_exact_fixture(
                        base, invalid_index=invalid_index
                    )
                    root = base / "failure-root"
                    writer = InterruptingWriter(target)
                    with self.assertRaisesRegex(
                        OSError, "injected write interruption"
                    ):
                        _execute_with_validated_context(
                            contract,
                            closure,
                            toolkit,
                            root,
                            git_runner=fake,
                            writer=writer,
                            execution_gate=_TEST_EXECUTION_GATE,
                        )
                    validated = validate_terminal(
                        root,
                        contract,
                        repo_root() / CONTRACT_RELATIVE_PATH,
                        closure,
                    )
                    failure = json.loads(
                        (root / FAILURE_FILENAME).read_text(
                            encoding="utf-8"
                        )
                    )
                self.assertEqual(
                    EVIDENCE_INVALID, validated["terminal"]
                )
                self.assertFalse(
                    failure["resume_or_rerun_authorized"]
                )
                self.assertFalse(failure["source_reread_authorized"])
                self.assertIn(
                    target,
                    {
                        row["name"]
                        for row in failure["observed_artifacts"]
                    },
                )

    def test_semantic_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract, closure, toolkit, fake = (
                self.make_exact_fixture(base)
            )
            root = base / "tamper-root"
            _execute_with_validated_context(
                contract,
                closure,
                toolkit,
                root,
                git_runner=fake,
                execution_gate=_TEST_EXECUTION_GATE,
            )
            shard_path = root / shard_filename(0)
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            shard["node_receipts"][0]["depth"] = 1
            shard_path.write_bytes(canonical_json_bytes(shard))
            with self.assertRaises(TerminalValidationError):
                validate_terminal(
                    root,
                    contract,
                    repo_root() / CONTRACT_RELATIVE_PATH,
                    closure,
                )


if __name__ == "__main__":
    unittest.main()
