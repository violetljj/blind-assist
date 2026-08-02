from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0b_p0a_toolkit_source_closure import (
    CANONICAL_ROOT,
    CLOSURE_INVALID,
    CLOSURE_LOCKED,
    CLOSURE_NOT_EVALUABLE,
    CONTRACT_RELATIVE_PATH,
    build_source_closure,
    execute_with_failure_closure,
    local_import_targets,
    parse_tree_paths,
    repo_root,
    require_canonical_root,
    resolve_module,
    resolve_module_chain,
    validate_contract,
    validate_existing_terminal,
    freeze_existing_partial,
)


COMMIT = "1" * 40
TREE_PATHS = [
    "tartanair/__init__.py",
    "tartanair/downloader.py",
    "tartanair/unrelated.py",
    "tartanair/utils.py",
]
SOURCES = {
    "tartanair/__init__.py": b"from .downloader import download\n",
    "tartanair/downloader.py": (
        b"from tartanair.utils import helper\n"
        b"import os\n"
        b"def download(): return helper()\n"
    ),
    "tartanair/utils.py": b"def helper(): return 1\n",
    "tartanair/unrelated.py": b"PROVIDER = 'must-not-be-read'\n",
}


def tree_bytes(paths: list[str] | None = None) -> bytes:
    return b"\x00".join(
        path.encode() for path in (paths or TREE_PATHS)
    ) + b"\x00"


class FakeGit:
    def __init__(self, sources: dict[str, bytes] | None = None) -> None:
        self.sources = sources or SOURCES
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: list[str], cwd: Path) -> bytes:
        self.calls.append(tuple(args))
        if args[:2] == ["init", "."]:
            (cwd / ".git").mkdir()
            return b""
        if args[:3] == ["remote", "add", "origin"] or "fetch" in args:
            return b""
        if args == ["rev-parse", "FETCH_HEAD"]:
            return (COMMIT + "\n").encode()
        if args[:5] == [
            "ls-tree", "-r", "-z", "--name-only", COMMIT
        ]:
            return tree_bytes(sorted(self.sources))
        if args[:2] == ["cat-file", "-s"]:
            path = args[2].split(":", 1)[1]
            return f"{len(self.sources[path])}\n".encode()
        if args[:2] == ["cat-file", "blob"]:
            path = args[2].split(":", 1)[1]
            return self.sources[path]
        if args[0] == "rev-parse" and ":" in args[1]:
            path = args[1].split(":", 1)[1]
            return (hashlib.sha1(path.encode()).hexdigest() + "\n").encode()
        raise AssertionError(f"Unexpected Git call: {args}")


def make_context(base: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
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
        "\n".join(f"    def test_{index}(self): pass" for index in range(16))
    )
    contract_path = base / "contract.json"
    contract: dict[str, object] = {
        "schema": (
            "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_source_"
            "closure_execution_contract"
        ),
        "status": (
            "FROZEN_AFTER_S0B_DESIGN_BEFORE_P0A_EXACT_COMMIT_FETCH_"
            "OR_SOURCE_READ"
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
            "test_count": 16,
        },
        "source_lock": {
            "toolkit_repository": "https://github.com/castacks/tartanairpy.git",
            "toolkit_commit": COMMIT,
        },
        "source_closure": {
            "seed_path": "tartanair/__init__.py",
            "maximum_python_blobs": 128,
            "maximum_total_source_bytes": 4194304,
            (
                "source_byte_budget_enforced_before_content_read_using_"
                "git_object_size"
            ): True,
            (
                "direct_and_simple_alias_dynamic_import_calls_recorded_"
                "not_executed"
            ): True,
            (
                "indirect_dynamic_import_or_exec_signals_recorded_not_"
                "executed"
            ): True,
            "p0b_must_not_evaluable_if_dynamic_evidence_nonzero": True,
            "zero_dynamic_evidence_is_not_runtime_completeness_proof": True,
        },
        "authorization": {
            "dataset_host_request_authorized": False,
            "provider_semantics_or_url_mapping_authorized": False,
        },
    }
    contract_path.write_text(json.dumps(contract))
    context = {
        "contract": contract,
        "contract_path": contract_path,
        "parent_paths": [(design, "design"), (result, "result")],
        "implementation_paths": [
            (planner, "planner"),
            (helper, "helper"),
            (mirror, "mirror"),
        ],
        "test_path": test,
    }
    return contract_path, contract, context


class P0AToolkitSourceClosureTest(unittest.TestCase):
    def test_tree_parser_accepts_sorted_nul_paths(self) -> None:
        self.assertEqual(TREE_PATHS, parse_tree_paths(tree_bytes()))

    def test_tree_parser_rejects_non_nul_terminated(self) -> None:
        with self.assertRaisesRegex(ValueError, "NUL"):
            parse_tree_paths(b"tartanair/x.py")

    def test_tree_parser_rejects_duplicate_unsorted_and_unsafe(self) -> None:
        for value in (
            tree_bytes(["tartanair/x.py", "tartanair/x.py"]),
            tree_bytes(["tartanair/z.py", "tartanair/a.py"]),
            tree_bytes(["tartanair/../outside.py"]),
        ):
            with self.assertRaises(ValueError):
                parse_tree_paths(value)

    def test_module_resolution_prefers_exact_single_path(self) -> None:
        paths = set(TREE_PATHS)
        self.assertEqual(
            "tartanair/downloader.py",
            resolve_module("tartanair.downloader", paths),
        )
        self.assertIsNone(resolve_module("tartanair.missing", paths))

    def test_nested_import_includes_package_init_chain(self) -> None:
        paths = set(TREE_PATHS) | {
            "tartanair/sub/__init__.py",
            "tartanair/sub/mod.py",
        }
        self.assertEqual(
            [
                "tartanair/__init__.py",
                "tartanair/sub/__init__.py",
                "tartanair/sub/mod.py",
            ],
            resolve_module_chain("tartanair.sub.mod", paths),
        )
        imports, _, _, _ = local_import_targets(
            "tartanair/downloader.py",
            "import tartanair.sub.mod\n",
            paths,
        )
        self.assertEqual(
            [
                "tartanair/__init__.py",
                "tartanair/sub/__init__.py",
                "tartanair/sub/mod.py",
            ],
            imports,
        )

    def test_relative_and_absolute_import_targets(self) -> None:
        imports, unresolved, calls, indirect = local_import_targets(
            "tartanair/__init__.py",
            "from .downloader import download\n"
            "from tartanair.utils import helper\n",
            set(TREE_PATHS),
        )
        self.assertEqual(
            [
                "tartanair/__init__.py",
                "tartanair/downloader.py",
                "tartanair/utils.py",
            ],
            imports,
        )
        self.assertEqual([], unresolved)
        self.assertEqual(0, calls)
        self.assertEqual(0, indirect)

    def test_relative_import_levels_stop_at_package_root(self) -> None:
        paths = set(TREE_PATHS) | {
            "tartanair/sub/__init__.py",
            "tartanair/sub/mod.py",
        }
        _, valid_one, _, _ = local_import_targets(
            "tartanair/sub/mod.py",
            "from . import mod\n",
            paths,
        )
        imports_two, valid_two, _, _ = local_import_targets(
            "tartanair/sub/mod.py",
            "from .. import utils\n",
            paths,
        )
        imports_three, invalid_three, _, _ = local_import_targets(
            "tartanair/sub/mod.py",
            "from ... import utils\n",
            paths,
        )
        self.assertEqual([], valid_one)
        self.assertEqual([], valid_two)
        self.assertIn("tartanair/utils.py", imports_two)
        self.assertEqual([], imports_three)
        self.assertEqual(["relative-level-3:"], invalid_three)

    def test_dynamic_import_is_counted_not_executed(self) -> None:
        _, _, calls, indirect = local_import_targets(
            "tartanair/downloader.py",
            "import importlib\nimportlib.import_module(name)\n",
            set(TREE_PATHS),
        )
        self.assertEqual(1, calls)
        self.assertEqual(0, indirect)

    def test_dynamic_import_aliases_and_indirect_constructs_are_recorded(
        self,
    ) -> None:
        _, _, calls, indirect = local_import_targets(
            "tartanair/downloader.py",
            "from importlib import import_module as load\n"
            "from builtins import __import__ as builtin_load\n"
            "import importlib as il\n"
            "import importlib.util\n"
            "import builtins as b\n"
            "alias = il.import_module\n"
            "alias2 = alias\n"
            "il2 = il\n"
            "b2 = b\n"
            "box = [il.import_module]\n"
            "find = getattr\n"
            "run = exec\n"
            "load(name)\n"
            "builtin_load(name)\n"
            "alias2(name)\n"
            "importlib.import_module(name)\n"
            "il2.import_module(name)\n"
            "b2.__import__(name)\n"
            "b.__dict__['__import__'](name)\n"
            "find(il, name)(name)\n"
            "run(code)\n",
            set(TREE_PATHS),
        )
        self.assertEqual(6, calls)
        self.assertEqual(4, indirect)

    def test_closure_reads_only_import_reachable_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            _, contract, _ = make_context(base)
            fake = FakeGit()
            closure = build_source_closure(
                contract,
                base,
                TREE_PATHS,
                git_runner=fake,
            )
        paths = [row["path"] for row in closure["closure_rows"]]
        self.assertEqual(
            [
                "tartanair/__init__.py",
                "tartanair/downloader.py",
                "tartanair/utils.py",
            ],
            paths,
        )
        self.assertNotIn("tartanair/unrelated.py", paths)

    def test_closure_missing_seed_is_not_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            _, contract, _ = make_context(base)
            closure = build_source_closure(
                contract,
                base,
                ["tartanair/downloader.py"],
                git_runner=FakeGit(),
            )
        self.assertEqual(CLOSURE_NOT_EVALUABLE, closure["terminal"])

    def test_closure_budget_exhaustion_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            _, contract, _ = make_context(base)
            contract["source_closure"]["maximum_python_blobs"] = 1
            with self.assertRaisesRegex(ValueError, "budget"):
                build_source_closure(
                    contract,
                    base,
                    TREE_PATHS,
                    git_runner=FakeGit(),
                )

    def test_blob_cap_stops_before_excess_blob_content_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            _, contract, _ = make_context(base)
            contract["source_closure"]["maximum_python_blobs"] = 1
            fake = FakeGit()
            with self.assertRaisesRegex(ValueError, "blob budget"):
                build_source_closure(
                    contract,
                    base,
                    TREE_PATHS,
                    git_runner=fake,
                )
        content_reads = [
            call for call in fake.calls if call[:2] == ("cat-file", "blob")
        ]
        self.assertEqual(1, len(content_reads))

    def test_byte_cap_stops_before_oversize_blob_content_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            _, contract, _ = make_context(base)
            contract["source_closure"][
                "maximum_total_source_bytes"
            ] = len(SOURCES["tartanair/__init__.py"]) - 1
            fake = FakeGit()
            with self.assertRaisesRegex(ValueError, "byte budget"):
                build_source_closure(
                    contract,
                    base,
                    TREE_PATHS,
                    git_runner=fake,
                )
        content_reads = [
            call for call in fake.calls if call[:2] == ("cat-file", "blob")
        ]
        self.assertEqual([], content_reads)

    def test_execute_attempt_preflight_before_exact_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            fake = FakeGit()

            def checking(args: list[str], cwd: Path) -> bytes:
                if "fetch" in args:
                    self.assertTrue((cwd.parent / "attempt.json").is_file())
                    self.assertTrue((cwd.parent / "preflight.json").is_file())
                return fake(args, cwd)

            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    base / "run",
                    git_runner=checking,
                    verify_git=False,
                )
        self.assertEqual(CLOSURE_LOCKED, result["terminal"])
        fetches = [call for call in fake.calls if "fetch" in call]
        self.assertEqual(
            [(
                "-c", "protocol.version=2", "fetch", "--no-tags",
                "--depth=1", "--recurse-submodules=no", "origin", COMMIT,
            )],
            fetches,
        )

    def test_execute_does_not_interpret_provider_or_call_host(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    base / "run",
                    git_runner=FakeGit(),
                    verify_git=False,
                )
                closure = json.loads((base / "run/closure.json").read_text())
        self.assertFalse(result["provider_semantics_interpreted"])
        self.assertFalse(result["dataset_host_request_made"])
        self.assertFalse(closure["observation"]["url_literal_or_template_extracted"])

    def test_missing_seed_execution_has_exact_not_evaluable_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            fake = FakeGit(
                {"tartanair/downloader.py": b"def download(): pass\n"}
            )
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=fake,
                    verify_git=False,
                )
            closure_path = root / "closure.json"
            closure = json.loads(closure_path.read_text())
            self.assertEqual(CLOSURE_NOT_EVALUABLE, result["terminal"])
            self.assertEqual(
                "EXACT_SEED_PATH_MISSING_SOURCE_CLOSURE_NOT_EVALUABLE",
                closure["status"],
            )
            names = {path.name for path in root.iterdir()}
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "local_fetch_head_or_none",
                return_value=COMMIT,
            ):
                self.assertTrue(validate_existing_terminal(root, names))
                closure["status"] = (
                    "EXACT_COMMIT_PYTHON_IMPORT_CLOSURE_LOCKED"
                )
                closure_path.write_text(json.dumps(closure))
                result_path = root / "result.json"
                result["bindings"]["closure_sha256"] = hashlib.sha256(
                    closure_path.read_bytes()
                ).hexdigest()
                result_path.write_text(json.dumps(result))
                self.assertFalse(validate_existing_terminal(root, names))

    def test_transport_failure_writes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)

            def failing(args: list[str], cwd: Path) -> bytes:
                if args[:2] == ["init", "."]:
                    (cwd / ".git").mkdir()
                    return b""
                if "fetch" in args:
                    raise OSError("transport")
                return b""

            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                with self.assertRaisesRegex(OSError, "transport"):
                    execute_with_failure_closure(
                        contract_path,
                        base / "run",
                        git_runner=failing,
                        verify_git=False,
                    )
            failure = json.loads((base / "run/failure.json").read_text())
        self.assertEqual(CLOSURE_INVALID, failure["terminal"])

    def test_valid_terminal_hash_chain_and_fetch_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(),
                    verify_git=False,
                )
            canonical = repo_root() / CONTRACT_RELATIVE_PATH
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "local_fetch_head_or_none",
                return_value=COMMIT,
            ):
                self.assertTrue(
                    validate_existing_terminal(
                        root, {path.name for path in root.iterdir()}
                    )
                )
                (root / "tree.json").write_text("{}")
                self.assertFalse(
                    validate_existing_terminal(
                        root, {path.name for path in root.iterdir()}
                    )
                )
            self.assertTrue(canonical.is_absolute())

    def test_terminal_rejects_semantic_claim_and_closure_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(),
                    verify_git=False,
                )
            names = {path.name for path in root.iterdir()}
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "local_fetch_head_or_none",
                return_value=COMMIT,
            ):
                result_path = root / "result.json"
                original_result = result_path.read_bytes()
                result = json.loads(original_result)
                result["provider_semantics_interpreted"] = True
                result_path.write_text(json.dumps(result))
                self.assertFalse(validate_existing_terminal(root, names))
                result_path.write_bytes(original_result)

                closure_path = root / "closure.json"
                closure = json.loads(closure_path.read_text())
                closure["observation"]["closure_blob_count"] += 1
                closure_path.write_text(json.dumps(closure))
                result = json.loads(result_path.read_text())
                result["bindings"]["closure_sha256"] = hashlib.sha256(
                    closure_path.read_bytes()
                ).hexdigest()
                result_path.write_text(json.dumps(result))
                self.assertFalse(validate_existing_terminal(root, names))

    def test_terminal_rejects_unreachable_row_and_budget_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract, context = make_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(),
                    verify_git=False,
                )
            names = {path.name for path in root.iterdir()}
            closure_path = root / "closure.json"
            result_path = root / "result.json"
            original_closure = closure_path.read_bytes()
            original_result = result_path.read_bytes()

            def write_closure_and_rebind(closure: dict[str, object]) -> None:
                closure_path.write_text(json.dumps(closure))
                result = json.loads(original_result)
                result["bindings"]["closure_sha256"] = hashlib.sha256(
                    closure_path.read_bytes()
                ).hexdigest()
                result_path.write_text(json.dumps(result))

            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "local_fetch_head_or_none",
                return_value=COMMIT,
            ):
                closure = json.loads(original_closure)
                rows = closure["observation"]["closure_rows"]
                unrelated = {
                    "path": "tartanair/unrelated.py",
                    "git_blob_oid": hashlib.sha1(
                        b"tartanair/unrelated.py"
                    ).hexdigest(),
                    "bytes": len(SOURCES["tartanair/unrelated.py"]),
                    "sha256": hashlib.sha256(
                        SOURCES["tartanair/unrelated.py"]
                    ).hexdigest(),
                    "local_import_targets": [],
                    "unresolved_local_imports": [],
                    "dynamic_import_call_count": 0,
                    "indirect_dynamic_import_or_exec_count": 0,
                }
                rows.append(unrelated)
                rows.sort(key=lambda row: row["path"])
                closure["observation"]["closure_blob_count"] += 1
                closure["observation"]["closure_total_source_bytes"] += (
                    unrelated["bytes"]
                )
                write_closure_and_rebind(closure)
                self.assertFalse(validate_existing_terminal(root, names))

                closure_path.write_bytes(original_closure)
                result_path.write_bytes(original_result)
                closure = json.loads(original_closure)
                closure["observation"]["closure_total_source_bytes"] = (
                    int(
                        contract["source_closure"][
                            "maximum_total_source_bytes"
                        ]
                    )
                    + 1
                )
                write_closure_and_rebind(closure)
                self.assertFalse(validate_existing_terminal(root, names))

                closure_path.write_bytes(original_closure)
                result_path.write_bytes(original_result)
                closure = json.loads(original_closure)
                closure["observation"]["closure_rows"][0][
                    "dynamic_import_call_count"
                ] = -1
                closure["observation"]["dynamic_import_call_count"] = -1
                write_closure_and_rebind(closure)
                self.assertFalse(validate_existing_terminal(root, names))

                closure_path.write_bytes(original_closure)
                result_path.write_bytes(original_result)
                closure = json.loads(original_closure)
                closure["observation"]["closure_rows"][0][
                    "dynamic_import_call_count"
                ] = 1
                closure["observation"]["dynamic_import_call_count"] = 1
                closure["observation"][
                    "no_dynamic_import_evidence_under_frozen_detector"
                ] = True
                write_closure_and_rebind(closure)
                self.assertFalse(validate_existing_terminal(root, names))

    def test_terminal_rejects_coordinated_tree_status_and_binding_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(),
                    verify_git=False,
                )
            names = {path.name for path in root.iterdir()}
            tree_path = root / "tree.json"
            closure_path = root / "closure.json"
            result_path = root / "result.json"
            original_tree = tree_path.read_bytes()
            original_closure = closure_path.read_bytes()
            original_result = result_path.read_bytes()

            def coordinate_tree_tamper(tree: dict[str, object]) -> None:
                tree_path.write_text(json.dumps(tree))
                closure = json.loads(original_closure)
                closure["bindings"]["tree_sha256"] = hashlib.sha256(
                    tree_path.read_bytes()
                ).hexdigest()
                closure_path.write_text(json.dumps(closure))
                result = json.loads(original_result)
                result["bindings"]["tree_sha256"] = hashlib.sha256(
                    tree_path.read_bytes()
                ).hexdigest()
                result["bindings"]["closure_sha256"] = hashlib.sha256(
                    closure_path.read_bytes()
                ).hexdigest()
                result_path.write_text(json.dumps(result))

            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "local_fetch_head_or_none",
                return_value=COMMIT,
            ):
                tree = json.loads(original_tree)
                tree["status"] = "tampered"
                coordinate_tree_tamper(tree)
                self.assertFalse(validate_existing_terminal(root, names))

                tree_path.write_bytes(original_tree)
                closure_path.write_bytes(original_closure)
                result_path.write_bytes(original_result)
                tree = json.loads(original_tree)
                tree["bindings"]["attempt_sha256"] = "0" * 64
                coordinate_tree_tamper(tree)
                self.assertFalse(validate_existing_terminal(root, names))

    def test_existing_attempt_preflight_partial_freezes_validly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            complete = base / "complete"
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path,
                    complete,
                    git_runner=FakeGit(),
                    verify_git=False,
                )
            root = base / "partial"
            root.mkdir()
            for name in ("attempt.json", "preflight.json"):
                (root / name).write_bytes((complete / name).read_bytes())
            original_names = {path.name for path in root.iterdir()}
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "local_fetch_head_or_none",
                return_value=None,
            ):
                self.assertEqual(
                    2, freeze_existing_partial(root, original_names)
                )
            names = {path.name for path in root.iterdir()}
            failure = json.loads((root / "failure.json").read_text())
            self.assertEqual(
                hashlib.sha256(
                    (root / "preflight.json").read_bytes()
                ).hexdigest(),
                failure["preflight_sha256"],
            )
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ), mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "local_fetch_head_or_none",
                return_value=None,
            ):
                self.assertTrue(validate_existing_terminal(root, names))

    def test_failure_terminal_validates_preflight_and_fetch_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"

            def failing(args: list[str], cwd: Path) -> bytes:
                if args[:2] == ["init", "."]:
                    (cwd / ".git").mkdir()
                    return b""
                if "fetch" in args:
                    raise OSError("transport")
                return b""

            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "validate_contract",
                return_value=context,
            ):
                with self.assertRaises(OSError):
                    execute_with_failure_closure(
                        contract_path,
                        root,
                        git_runner=failing,
                        verify_git=False,
                    )
            names = {path.name for path in root.iterdir()}
            with mock.patch(
                "plan_stage_c_d5_s0b_p0a_toolkit_source_closure."
                "CONTRACT_RELATIVE_PATH",
                contract_path,
            ):
                self.assertTrue(validate_existing_terminal(root, names))
                preflight_path = root / "preflight.json"
                original = preflight_path.read_bytes()
                preflight = json.loads(original)
                preflight["status"] = "tampered"
                preflight_path.write_text(json.dumps(preflight))
                self.assertFalse(validate_existing_terminal(root, names))
                preflight_path.write_bytes(original)
                failure_path = root / "failure.json"
                failure = json.loads(failure_path.read_text())
                failure["fetched_commit"] = COMMIT
                failure_path.write_text(json.dumps(failure))
                self.assertFalse(validate_existing_terminal(root, names))

    def test_canonical_root_rejects_alternate(self) -> None:
        expected = (repo_root() / CANONICAL_ROOT).resolve()
        self.assertEqual(expected, require_canonical_root(expected))
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            require_canonical_root(repo_root() / "wrong")

    def test_contract_rejects_provider_authority_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract, _ = make_context(base)
            contract["authorization"][
                "provider_semantics_or_url_mapping_authorized"
            ] = True
            contract_path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "interpretation"):
                validate_contract(contract_path, verify_git=False)

    def test_real_contract_validates_without_git_gate(self) -> None:
        context = validate_contract(
            repo_root() / CONTRACT_RELATIVE_PATH,
            verify_git=False,
        )
        self.assertEqual(
            "158a6844d782942110967325ca3082f50ab2bfc7",
            context["contract"]["source_lock"]["toolkit_commit"],
        )


if __name__ == "__main__":
    unittest.main()
