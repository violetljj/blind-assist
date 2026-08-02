from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0b_p0b_provider_semantic_evidence import (
    CANONICAL_ROOT,
    CONTRACT_RELATIVE_PATH,
    EVIDENCE_INVALID,
    EVIDENCE_LOCKED,
    EVIDENCE_NOT_EVALUABLE,
    EXPECTED_ALGORITHM_LOCK,
    artifact_state,
    canonical_dump_parts,
    canonical_json_bytes,
    canonical_preorder_node_ids,
    execute_with_failure_closure,
    extract_evidence,
    freeze_existing_partial,
    repo_root,
    require_canonical_root,
    runtime_receipt,
    validate_contract,
    validate_existing_terminal,
)


COMMIT = "1" * 40
SOURCES = {
    "tartanair/__init__.py": (
        b'"""module docs https://docs.invalid/example"""\n'
        b"from .downloader import download\n"
    ),
    "tartanair/downloader.py": (
        b"import requests as rq\n"
        b"from urllib.parse import urljoin\n"
        b"BASE = 'https://provider.invalid/data/'\n"
        b"RAW = b'\\x00\\xff'\n"
        b"TABLE = {'base': BASE}\n"
        b"class Client:\n"
        b"    \"\"\"class docs\"\"\"\n"
        b"    def download(self, name='default.zip'):\n"
        b"        \"\"\"function docs\"\"\"\n"
        b"        url = f'{BASE}{name}'\n"
        b"        alt = '%s.zip' % name\n"
        b"        other = '{}'.format(name)\n"
        b"        joined = urljoin(TABLE['base'], name)\n"
        b"        alt += other\n"
        b"        return rq.get(url, timeout=10)\n"
        b"def typed(arg: 'arg_annotation') -> 'return_annotation':\n"
        b"    value: 'variable_annotation'\n"
        b"    assigned: 'assigned_annotation' = 'assigned_value'\n"
        b"    return arg\n"
    ),
}


def make_rows(sources: dict[str, bytes]) -> list[dict[str, object]]:
    rows = []
    for index, (path, blob) in enumerate(sorted(sources.items()), 1):
        rows.append(
            {
                "path": path,
                "git_blob_oid": f"{index:040x}",
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "local_import_targets": [],
                "unresolved_local_imports": [],
                "dynamic_import_call_count": 0,
                "indirect_dynamic_import_or_exec_count": 0,
            }
        )
    return rows


def make_closure(
    sources: dict[str, bytes] | None = None,
) -> dict[str, object]:
    current = sources or SOURCES
    rows = make_rows(current)
    return {
        "schema": "p0a-closure",
        "observation": {
            "closure_blob_count": len(rows),
            "closure_total_source_bytes": sum(
                int(row["bytes"]) for row in rows
            ),
            "closure_rows": rows,
            "dynamic_import_call_count": 0,
            "indirect_dynamic_import_or_exec_count": 0,
            "unresolved_local_imports": [],
            "python_tree_path_count": len(rows),
        },
    }


def make_contract(base: Path, closure: dict[str, object]) -> tuple[
    Path, dict[str, object], dict[str, object]
]:
    closure_path = base / "closure.json"
    closure_path.write_text(json.dumps(closure))
    design = base / "design.json"
    result = base / "result.json"
    planner = base / "planner.py"
    helper = base / "helper.py"
    mirror = base / "mirror.md"
    test = base / "test.py"
    design.write_text('{"status":"design"}')
    result.write_text('{"terminal":"locked"}')
    planner.write_text("planner")
    helper.write_text("helper")
    mirror.write_text("mirror")
    test.write_text(
        "\n".join(f"    def test_{index}(self): pass" for index in range(12))
    )
    rows = closure["observation"]["closure_rows"]
    contract: dict[str, object] = {
        "schema": (
            "blindassist_hftf_stage_c_d5_s0b_p0b_provider_semantic_"
            "evidence_execution_contract"
        ),
        "status": (
            "FROZEN_AFTER_P0B_DESIGN_BEFORE_FIRST_P0B_SOURCE_BLOB_"
            "SEMANTIC_READ"
        ),
        "parents": {
            "design": {
                "path": str(design),
                "sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
                "required_status": "design",
            },
            "result": {
                "path": str(result),
                "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                "required_terminal": "locked",
            },
        },
        "implementations": {
            "planner": {
                "path": str(planner),
                "sha256": hashlib.sha256(planner.read_bytes()).hexdigest(),
            },
            "helper": {
                "path": str(helper),
                "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
            },
            "mirror": {
                "path": str(mirror),
                "sha256": hashlib.sha256(mirror.read_bytes()).hexdigest(),
            },
        },
        "implementation_tests": {
            "planner_test": {
                "path": str(test),
                "sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
            },
            "test_count": 12,
        },
        "runtime_lock": runtime_receipt(),
        "algorithm_lock": EXPECTED_ALGORITHM_LOCK,
        "source_boundary": {
            "toolkit_commit": COMMIT,
            "exact_blob_count": len(rows),
            "exact_total_source_bytes": sum(
                int(row["bytes"]) for row in rows
            ),
            "p0a_unresolved_local_import_count": 0,
            "p0a_python_tree_path_count": len(rows),
            "p0a_closure_artifact": {
                "path": str(closure_path),
                "bytes": closure_path.stat().st_size,
                "sha256": hashlib.sha256(
                    closure_path.read_bytes()
                ).hexdigest(),
            },
        },
        "frozen_extraction": {
            "maximum_string_literal_records": 4096,
            "maximum_call_site_records": 8192,
            "maximum_assignment_records": 4096,
            "maximum_function_records": 2048,
            "maximum_import_alias_records": 4096,
            "maximum_expression_records": 32768,
            "maximum_ast_nodes": 65536,
            "maximum_ast_depth": 256,
            "maximum_single_string_utf8_bytes": 16384,
            "maximum_single_source_segment_utf8_bytes": 16384,
            "maximum_total_evidence_json_bytes": 8388608,
            "string_literal_role_classes": [
                "module_docstring",
                "class_docstring",
                "function_docstring",
                "ordinary_expression",
                "assignment_value",
                "call_positional_argument",
                "call_keyword_argument",
                "default_or_annotation",
                "other_lexical_context",
            ],
        },
    }
    contract_path = base / "contract.json"
    contract_path.write_text(json.dumps(contract))
    context = {
        "contract": contract,
        "contract_path": contract_path,
        "closure": closure,
        "closure_path": closure_path,
    }
    return contract_path, contract, context


class FakeGit:
    def __init__(
        self,
        closure: dict[str, object],
        sources: dict[str, bytes] | None = None,
    ) -> None:
        self.sources = sources or SOURCES
        self.rows = closure["observation"]["closure_rows"]
        self.by_oid = {
            row["git_blob_oid"]: self.sources[row["path"]]
            for row in self.rows
        }
        self.path_to_oid = {
            row["path"]: row["git_blob_oid"] for row in self.rows
        }
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: list[str], cwd: Path) -> bytes:
        self.calls.append(tuple(args))
        if args[0] == "rev-parse" and ":" in args[1]:
            path = args[1].split(":", 1)[1]
            return (self.path_to_oid[path] + "\n").encode()
        if args[:2] == ["cat-file", "-t"]:
            return b"blob\n"
        if args[:2] == ["cat-file", "-s"]:
            return f"{len(self.by_oid[args[2]])}\n".encode()
        if args[:2] == ["cat-file", "blob"]:
            return self.by_oid[args[2]]
        raise AssertionError(f"Unexpected Git call: {args}")


class P0BProviderSemanticEvidenceTest(unittest.TestCase):
    def test_extracts_full_syntax_evidence_without_provider_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            contract_path, contract, context = make_contract(base, closure)
            terminal, evidence = extract_evidence(
                contract,
                closure,
                base,
                git_runner=FakeGit(closure),
            )
        self.assertEqual(EVIDENCE_LOCKED, terminal)
        self.assertEqual(2, evidence["object_receipt_count"])
        self.assertEqual(2, evidence["parse_receipt_count"])
        self.assertGreater(evidence["expression_count"], 0)
        self.assertGreater(evidence["import_alias_count"], 0)
        self.assertFalse(evidence["official_provider_url_template_established"])
        roles = {
            row["lexical_role"] for row in evidence["string_literals"]
        }
        self.assertIn("module_docstring", roles)
        self.assertIn("class_docstring", roles)
        self.assertIn("function_docstring", roles)
        role_by_value = {
            row["value"]: row["lexical_role"]
            for row in evidence["string_literals"]
        }
        for value in (
            "arg_annotation",
            "return_annotation",
            "variable_annotation",
            "assigned_annotation",
        ):
            self.assertEqual("default_or_annotation", role_by_value[value])
        self.assertEqual(
            "assignment_value", role_by_value["assigned_value"]
        )
        node_types = {
            row["node_type"] for row in evidence["expressions"]
        }
        self.assertTrue(
            {"JoinedStr", "BinOp", "Subscript", "Call"} <= node_types
        )
        self.assertTrue(
            any(
                "bytes" in row["canonical_ast_dump"]
                for row in evidence["expressions"]
            )
        )

    def test_repeated_ast_singletons_receive_distinct_occurrence_edges(self) -> None:
        sources = {
            "tartanair/__init__.py": (
                b"x = a + b\n"
                b"y = c + d\n"
            ),
            "tartanair/downloader.py": b"z = x + y\n",
        }
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure(sources)
            _, contract, _ = make_contract(base, closure)
            terminal, evidence = extract_evidence(
                contract,
                closure,
                base,
                git_runner=FakeGit(closure, sources),
            )
        self.assertEqual(EVIDENCE_LOCKED, terminal)
        node_map = {
            row["node_id"]: row for row in evidence["ast_node_receipts"]
        }
        add_ids: list[str] = []
        context_ids: list[str] = []
        for row in evidence["ast_node_receipts"]:
            dump = json.loads(row["canonical_ast_dump"])
            fields = dump["fields"]
            if row["node_type"] == "BinOp":
                add_ids.append(fields["op"]["node_id"])
            if row["node_type"] == "Name":
                context_ids.append(fields["ctx"]["node_id"])
        self.assertEqual(len(add_ids), len(set(add_ids)))
        self.assertEqual(len(context_ids), len(set(context_ids)))
        for parent_id, child_id in [
            *[
                (
                    row["node_id"],
                    json.loads(row["canonical_ast_dump"])["fields"]["op"][
                        "node_id"
                    ],
                )
                for row in evidence["ast_node_receipts"]
                if row["node_type"] == "BinOp"
            ],
            *[
                (
                    row["node_id"],
                    json.loads(row["canonical_ast_dump"])["fields"]["ctx"][
                        "node_id"
                    ],
                )
                for row in evidence["ast_node_receipts"]
                if row["node_type"] == "Name"
            ],
        ]:
            self.assertEqual(parent_id, node_map[child_id]["parent_node_id"])

    def test_canonical_preorder_uses_field_and_list_sibling_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            _, contract, _ = make_contract(base, closure)
            _, evidence = extract_evidence(
                contract,
                closure,
                base,
                git_runner=FakeGit(closure),
            )
        node_map = {
            row["node_id"]: row for row in evidence["ast_node_receipts"]
        }
        dump_map = {
            node_id: canonical_dump_parts(row["canonical_ast_dump"])
            for node_id, row in node_map.items()
        }
        path = "tartanair/downloader.py"
        path_rows = [
            row
            for row in evidence["ast_node_receipts"]
            if row["source_path"] == path
        ]
        expected = [row["node_id"] for row in path_rows]
        self.assertEqual(
            expected,
            canonical_preorder_node_ids(
                path_rows[0]["node_id"], node_map, dump_map
            ),
        )

        binop = next(
            row for row in path_rows if row["node_type"] == "BinOp"
        )
        swapped_fields = copy.deepcopy(dump_map)
        binop_parts = swapped_fields[binop["node_id"]]
        binop_fields = dict(binop_parts[1])
        binop_fields["left"], binop_fields["right"] = (
            binop_fields["right"],
            binop_fields["left"],
        )
        swapped_fields[binop["node_id"]] = (
            binop_parts[0],
            binop_fields,
            binop_parts[2],
        )
        self.assertNotEqual(
            expected,
            canonical_preorder_node_ids(
                path_rows[0]["node_id"], node_map, swapped_fields
            ),
        )

        list_parent = next(
            row
            for row in path_rows
            if any(
                isinstance(value, list)
                and sum(
                    isinstance(item, dict) and set(item) == {"node_id"}
                    for item in value
                )
                >= 2
                for value in dump_map[row["node_id"]][1].values()
            )
        )
        swapped_list = copy.deepcopy(dump_map)
        list_parts = swapped_list[list_parent["node_id"]]
        list_fields = copy.deepcopy(list_parts[1])
        list_field = next(
            field
            for field, value in list_fields.items()
            if isinstance(value, list)
            and sum(
                isinstance(item, dict) and set(item) == {"node_id"}
                for item in value
            )
            >= 2
        )
        list_fields[list_field][0], list_fields[list_field][1] = (
            list_fields[list_field][1],
            list_fields[list_field][0],
        )
        swapped_list[list_parent["node_id"]] = (
            list_parts[0],
            list_fields,
            list_parts[2],
        )
        self.assertNotEqual(
            expected,
            canonical_preorder_node_ids(
                path_rows[0]["node_id"], node_map, swapped_list
            ),
        )

    def test_object_receipts_complete_before_first_ast_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            _, contract, _ = make_contract(base, closure)
            fake = FakeGit(closure)
            real_parse = __import__("ast").parse

            def checking_parse(*args: object, **kwargs: object) -> object:
                reads = [
                    call
                    for call in fake.calls
                    if call[:2] == ("cat-file", "blob")
                ]
                self.assertEqual(2, len(reads))
                return real_parse(*args, **kwargs)

            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "ast.parse",
                side_effect=checking_parse,
            ):
                extract_evidence(
                    contract, closure, base, git_runner=fake
                )

    def test_each_blob_content_is_read_exactly_once_and_no_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            _, contract, _ = make_contract(base, closure)
            fake = FakeGit(closure)
            extract_evidence(contract, closure, base, git_runner=fake)
        reads = [
            call for call in fake.calls if call[:2] == ("cat-file", "blob")
        ]
        self.assertEqual(2, len(reads))
        self.assertFalse(
            any("fetch" in call or "http" in " ".join(call) for call in fake.calls)
        )

    def test_dynamic_evidence_is_not_evaluable_without_blob_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            closure["observation"]["dynamic_import_call_count"] = 1
            _, contract, _ = make_contract(base, closure)
            fake = FakeGit(closure)
            terminal, evidence = extract_evidence(
                contract, closure, base, git_runner=fake
            )
        self.assertEqual(EVIDENCE_NOT_EVALUABLE, terminal)
        self.assertEqual(0, evidence["source_blob_read_count"])
        self.assertEqual([], fake.calls)

    def test_zero_url_literals_remains_neutral_locked_evidence(self) -> None:
        sources = {
            "tartanair/__init__.py": b"VALUE = 'plain text'\n",
            "tartanair/downloader.py": b"def f(): return 'relative/path'\n",
        }
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure(sources)
            _, contract, _ = make_contract(base, closure)
            terminal, evidence = extract_evidence(
                contract,
                closure,
                base,
                git_runner=FakeGit(closure, sources),
            )
        self.assertEqual(EVIDENCE_LOCKED, terminal)
        self.assertEqual(0, evidence["url_literal_count"])
        self.assertFalse(evidence["official_provider_url_template_established"])

    def test_syntax_incompatibility_is_not_evaluable_after_receipts(self) -> None:
        sources = {
            "tartanair/__init__.py": b"VALUE = 1\n",
            "tartanair/downloader.py": b"OTHER = 2\n",
            "tartanair/zbroken.py": b"def broken(:\n",
        }
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure(sources)
            contract_path, contract, context = make_contract(base, closure)
            fake = FakeGit(closure, sources)
            terminal, evidence = extract_evidence(
                contract, closure, base, git_runner=fake
            )
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "validate_contract",
                return_value=context,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(closure, sources),
                    verify_git=False,
                )
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                self.assertTrue(
                    validate_existing_terminal(root, artifact_state(root))
                )
                evidence_path = root / "evidence.json"
                result_path = root / "result.json"
                original_evidence = evidence_path.read_bytes()
                original_result = result_path.read_bytes()

                def mutate_and_rebind(
                    mutate: Callable[[dict[str, object]], None],
                ) -> bool:
                    changed = json.loads(original_evidence)
                    mutate(changed)
                    evidence_path.write_text(json.dumps(changed))
                    changed_result = json.loads(original_result)
                    changed_result["bindings"]["evidence_sha256"] = (
                        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                    )
                    result_path.write_text(json.dumps(changed_result))
                    return validate_existing_terminal(
                        root, artifact_state(root)
                    )

                self.assertFalse(
                    mutate_and_rebind(
                        lambda value: value["observation"].__setitem__(
                            "official_provider_url_template_established", True
                        )
                    )
                )
                self.assertFalse(
                    mutate_and_rebind(
                        lambda value: value["observation"].__setitem__(
                            "parse_receipt_count", 1
                        )
                    )
                )

                def overflow_prefix_ast(
                    value: dict[str, object],
                ) -> None:
                    receipts = value["observation"]["parse_receipts"]
                    for row in receipts:
                        row["ast_node_count"] = 65536
                    value["observation"][
                        "parsed_prefix_ast_node_count"
                    ] = 131072

                self.assertFalse(mutate_and_rebind(overflow_prefix_ast))
                evidence_path.write_bytes(original_evidence)
                result_path.write_bytes(original_result)
        self.assertEqual(EVIDENCE_NOT_EVALUABLE, terminal)
        self.assertEqual(EVIDENCE_NOT_EVALUABLE, result["terminal"])
        self.assertEqual(3, evidence["object_receipt_count"])
        reads = [
            call for call in fake.calls if call[:2] == ("cat-file", "blob")
        ]
        self.assertEqual(3, len(reads))

    def test_hash_mismatch_is_invalid_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            closure["observation"]["closure_rows"][0]["sha256"] = "0" * 64
            _, contract, _ = make_contract(base, closure)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                extract_evidence(
                    contract,
                    closure,
                    base,
                    git_runner=FakeGit(closure),
                )
    def test_record_cap_overflow_is_invalid_not_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            contract_path, contract, context = make_contract(base, closure)
            contract["frozen_extraction"][
                "maximum_string_literal_records"
            ] = 0
            with self.assertRaisesRegex(ValueError, "record cap"):
                extract_evidence(
                    contract,
                    closure,
                    base,
                    git_runner=FakeGit(closure),
                )
            contract["frozen_extraction"][
                "maximum_string_literal_records"
            ] = 4096
            _, observation = extract_evidence(
                contract,
                closure,
                base,
                git_runner=FakeGit(closure),
            )
            contract["frozen_extraction"][
                "maximum_total_evidence_json_bytes"
            ] = len(canonical_json_bytes(observation)) + 1
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "validate_contract",
                return_value=context,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                with self.assertRaisesRegex(
                    ValueError, "complete evidence artifact"
                ):
                    execute_with_failure_closure(
                        contract_path,
                        base / "run",
                        git_runner=FakeGit(closure),
                        verify_git=False,
                    )

    def test_prefix_record_cap_overflow_precedes_later_syntax_error(self) -> None:
        sources = {
            "tartanair/__init__.py": b"VALUE = 'already over cap'\n",
            "tartanair/downloader.py": b"def broken(:\n",
        }
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure(sources)
            _, contract, _ = make_contract(base, closure)
            contract["frozen_extraction"][
                "maximum_string_literal_records"
            ] = 0
            with self.assertRaisesRegex(ValueError, "record cap"):
                extract_evidence(
                    contract,
                    closure,
                    base,
                    git_runner=FakeGit(closure, sources),
                )

    def test_execute_attempt_and_preflight_precede_blob_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            contract_path, _, context = make_contract(base, closure)
            root = base / "run"
            fake = FakeGit(closure)

            def checking(args: list[str], cwd: Path) -> bytes:
                if args[:2] == ["cat-file", "blob"]:
                    self.assertTrue((root / "attempt.json").is_file())
                    self.assertTrue((root / "preflight.json").is_file())
                return fake(args, cwd)

            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "validate_contract",
                return_value=context,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=checking,
                    verify_git=False,
                )
        self.assertEqual(EVIDENCE_LOCKED, result["terminal"])

    def test_execute_failure_writes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            closure["observation"]["closure_rows"][0]["sha256"] = "0" * 64
            contract_path, _, context = make_contract(base, closure)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "validate_contract",
                return_value=context,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                with self.assertRaises(ValueError):
                    execute_with_failure_closure(
                        contract_path,
                        root,
                        git_runner=FakeGit(closure),
                        verify_git=False,
                    )
            failure = json.loads((root / "failure.json").read_text())
            names = artifact_state(root)
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                self.assertTrue(validate_existing_terminal(root, names))
                preflight_path = root / "preflight.json"
                preflight = json.loads(preflight_path.read_text())
                preflight["status"] = "tampered"
                preflight_path.write_text(json.dumps(preflight))
                self.assertFalse(validate_existing_terminal(root, names))
        self.assertEqual(EVIDENCE_INVALID, failure["terminal"])

    def test_valid_terminal_and_semantic_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            closure = make_closure()
            contract_path, _, context = make_contract(base, closure)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "validate_contract",
                return_value=context,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(closure),
                    verify_git=False,
                )
            names = artifact_state(root)
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                self.assertTrue(validate_existing_terminal(root, names))
                evidence_path = root / "evidence.json"
                result_path = root / "result.json"
                original_evidence = evidence_path.read_bytes()
                original_result = result_path.read_bytes()

                def write_evidence_and_rebind(
                    evidence: dict[str, object],
                ) -> None:
                    evidence_path.write_text(json.dumps(evidence))
                    result = json.loads(original_result)
                    result["bindings"]["evidence_sha256"] = hashlib.sha256(
                        evidence_path.read_bytes()
                    ).hexdigest()
                    result_path.write_text(json.dumps(result))

                evidence = json.loads(original_evidence)
                evidence["observation"][
                    "provider_control_flow_or_url_derivation_interpreted"
                ] = True
                write_evidence_and_rebind(evidence)
                self.assertFalse(validate_existing_terminal(root, names))

                evidence_path.write_bytes(original_evidence)
                result_path.write_bytes(original_result)
                evidence = json.loads(original_evidence)
                evidence["observation"][
                    "p0a_unresolved_local_import_count"
                ] = 1
                write_evidence_and_rebind(evidence)
                self.assertFalse(validate_existing_terminal(root, names))

                evidence_path.write_bytes(original_evidence)
                result_path.write_bytes(original_result)
                evidence = json.loads(original_evidence)
                evidence["observation"]["call_sites"][0][
                    "callee_syntax"
                ] = "tampered.provider"
                write_evidence_and_rebind(evidence)
                self.assertFalse(validate_existing_terminal(root, names))

                def reject_locked_mutation(
                    mutate: Callable[[dict[str, object]], None],
                ) -> None:
                    evidence_path.write_bytes(original_evidence)
                    result_path.write_bytes(original_result)
                    changed = json.loads(original_evidence)
                    mutate(changed)
                    write_evidence_and_rebind(changed)
                    self.assertFalse(
                        validate_existing_terminal(root, names)
                    )

                def drop_url_literal(value: dict[str, object]) -> None:
                    observation = value["observation"]
                    index = next(
                        index
                        for index, row in enumerate(
                            observation["string_literals"]
                        )
                        if row["url_scheme_class"] != "NONE"
                    )
                    row = observation["string_literals"].pop(index)
                    observation["string_literal_count"] -= 1
                    observation["url_literal_count"] -= 1
                    if row["archive_suffix_class"] == "ZIP":
                        observation["zip_string_literal_count"] -= 1

                def drop_call(value: dict[str, object]) -> None:
                    observation = value["observation"]
                    observation["call_sites"].pop()
                    observation["call_site_count"] -= 1

                def drop_assignment(value: dict[str, object]) -> None:
                    observation = value["observation"]
                    observation["assignments"].pop()
                    observation["assignment_count"] -= 1

                def duplicate_string(value: dict[str, object]) -> None:
                    observation = value["observation"]
                    row = dict(observation["string_literals"][0])
                    observation["string_literals"].append(row)
                    observation["string_literals"].sort(
                        key=lambda item: (
                            item["source_path"],
                            item["lineno"],
                            item["col_offset"],
                            json.dumps(item, sort_keys=True),
                        )
                    )
                    observation["string_literal_count"] += 1
                    if row["url_scheme_class"] != "NONE":
                        observation["url_literal_count"] += 1
                    if row["archive_suffix_class"] == "ZIP":
                        observation["zip_string_literal_count"] += 1

                def drop_free_expression(value: dict[str, object]) -> None:
                    observation = value["observation"]
                    referenced = {
                        row["node_id"]
                        for row in observation["string_literals"]
                    }
                    for row in observation["call_sites"]:
                        referenced.add(row["callee_node_id"])
                        referenced.update(
                            row["positional_argument_node_ids"]
                        )
                        referenced.update(
                            item["node_id"]
                            for item in row["keyword_argument_node_ids"]
                        )
                    for row in observation["assignments"]:
                        referenced.add(row["value_node_id"])
                        referenced.update(row["target_node_ids"])
                    index = next(
                        index
                        for index, row in enumerate(
                            observation["expressions"]
                        )
                        if row["node_id"] not in referenced
                    )
                    observation["expressions"].pop(index)
                    observation["expression_count"] -= 1

                reject_locked_mutation(drop_url_literal)
                reject_locked_mutation(drop_call)
                reject_locked_mutation(drop_assignment)
                reject_locked_mutation(duplicate_string)
                reject_locked_mutation(drop_free_expression)
                reject_locked_mutation(
                    lambda value: value["observation"][
                        "ast_node_receipts"
                    ][1].__setitem__(
                        "depth",
                        value["observation"]["ast_node_receipts"][1][
                            "depth"
                        ]
                        + 1,
                    )
                )
                reject_locked_mutation(
                    lambda value: next(
                        row
                        for row in value["observation"]["string_literals"]
                        if row["is_docstring"]
                    ).update(
                        {
                            "is_docstring": False,
                            "lexical_role": "assignment_value",
                        }
                    )
                )
                reject_locked_mutation(
                    lambda value: value["observation"]["import_aliases"][
                        0
                    ].__setitem__("module", "tampered.module")
                )
                reject_locked_mutation(
                    lambda value: value["observation"]["functions"][
                        0
                    ]["argument_names"].append("tampered_arg")
                )
                reject_locked_mutation(
                    lambda value: value["observation"][
                        "ast_node_receipts"
                    ][0].__setitem__("node_id", "0" * 64)
                )
                reject_locked_mutation(
                    lambda value: value["observation"][
                        "ast_node_receipts"
                    ][0].__setitem__(
                        "canonical_ast_dump",
                        value["observation"]["ast_node_receipts"][0][
                            "canonical_ast_dump"
                        ].replace('"Module"', '"Expression"', 1),
                    )
                )
                reject_locked_mutation(
                    lambda value: value["observation"][
                        "ast_node_receipts"
                    ][0].__setitem__(
                        "source_path",
                        value["observation"]["ast_node_receipts"][-1][
                            "source_path"
                        ],
                    )
                )

                for reason in (
                    "p0a_dynamic_import_evidence_nonzero",
                    (
                        "verified_source_incompatible_with_frozen_"
                        "encoding_or_ast_grammar"
                    ),
                ):
                    evidence_path.write_bytes(original_evidence)
                    result_path.write_bytes(original_result)
                    evidence = json.loads(original_evidence)
                    evidence["terminal"] = EVIDENCE_NOT_EVALUABLE
                    evidence["status"] = (
                        "EXACT_SOURCE_AST_EVIDENCE_NOT_EVALUABLE"
                    )
                    evidence["observation"] = {"reason": reason}
                    evidence_path.write_text(json.dumps(evidence))
                    result = json.loads(original_result)
                    result["terminal"] = EVIDENCE_NOT_EVALUABLE
                    result["next_authority"] = (
                        "stop_provider_resolution_as_source_evidence_"
                        "not_evaluable"
                    )
                    result["bindings"]["evidence_sha256"] = hashlib.sha256(
                        evidence_path.read_bytes()
                    ).hexdigest()
                    result_path.write_text(json.dumps(result))
                    self.assertFalse(
                        validate_existing_terminal(root, names)
                    )

    def test_existing_partial_freezes_invalid_without_blob_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            (root / "unknown.json").write_text("{}")
            names = artifact_state(root)
            with mock.patch(
                "plan_stage_c_d5_s0b_p0b_provider_semantic_evidence."
                "p0a_fetch_head",
                return_value=COMMIT,
            ):
                self.assertEqual(2, freeze_existing_partial(root, names))
            failure = json.loads((root / "failure.json").read_text())
        self.assertEqual(EVIDENCE_INVALID, failure["terminal"])

    def test_canonical_root_and_real_contract(self) -> None:
        expected = (repo_root() / CANONICAL_ROOT).resolve()
        self.assertEqual(expected, require_canonical_root(expected))
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            require_canonical_root(repo_root() / "wrong")
        context = validate_contract(
            repo_root() / CONTRACT_RELATIVE_PATH,
            verify_git=False,
        )
        self.assertEqual(
            "158a6844d782942110967325ca3082f50ab2bfc7",
            context["contract"]["source_boundary"]["toolkit_commit"],
        )


if __name__ == "__main__":
    unittest.main()
