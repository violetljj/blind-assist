#!/usr/bin/env python3
"""Lock bounded AST evidence for later exact provider resolution."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0a_tartanground_catalog import (  # noqa: E402
    git_local,
    load_json,
    require_tracked_clean,
    resolve_bound,
    sha256,
    test_definition_count,
    write_json_exclusive_fsync,
)
from plan_stage_c_d5_s0b_p0a_toolkit_source_closure import (  # noqa: E402
    artifact_state as p0a_artifact_state,
    validate_existing_terminal as validate_p0a_terminal,
)


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b_provider_semantic_evidence_"
    "execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_P0B_DESIGN_BEFORE_FIRST_P0B_SOURCE_BLOB_SEMANTIC_READ"
)
ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b_provider_semantic_evidence_attempt"
)
PREFLIGHT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b_provider_semantic_evidence_preflight"
)
EVIDENCE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b_provider_semantic_evidence"
)
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b_provider_semantic_evidence_result"
)
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b_provider_semantic_evidence_failure"
)

EVIDENCE_LOCKED = (
    "D5_S0B_P0B_PROVIDER_SEMANTIC_EVIDENCE_LOCKED_REQUIRES_"
    "P0C_PROVIDER_RESOLUTION"
)
EVIDENCE_NOT_EVALUABLE = (
    "D5_S0B_P0B_PROVIDER_SEMANTIC_EVIDENCE_NOT_EVALUABLE"
)
EVIDENCE_INVALID = (
    "D5_S0B_P0B_PROVIDER_SEMANTIC_EVIDENCE_INVALID_STOP"
)

CANONICAL_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0b-provider-semantic-evidence-20260802"
)
P0A_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0a-toolkit-source-closure-20260802"
)
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/hftf/"
    "HFTF_STAGE_C_D5_S0B_P0B_PROVIDER_SEMANTIC_EVIDENCE_"
    "EXECUTION_CONTRACT_2026-08-02.json"
)
FILENAMES = {
    "attempt": "attempt.json",
    "preflight": "preflight.json",
    "evidence": "evidence.json",
    "result": "result.json",
    "failure": "failure.json",
}
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
GitRunner = Callable[[Sequence[str], Path], bytes]
EXPECTED_ALGORITHM_LOCK = {
    "canonical_json": (
        "utf8_ensure_ascii_false_sort_keys_true_separators_comma_colon_"
        "single_lf"
    ),
    "node_id": (
        "sha256_utf8_nul_join_path_preorder_node_type_start_end_span_"
        "canonical_shallow_ast_dump_with_direct_child_occurrence_ids_"
        "preorder_ast_iter_fields_depth_first_children_hashed_first"
    ),
    "canonical_ast_dump": (
        "shallow_node_type_and_ast_iter_fields_scalars_with_direct_"
        "children_replaced_by_node_id_json_scalars_direct_bytes_hex_"
        "complex_repr_ellipsis_tag_using_canonical_json_without_lf"
    ),
    "source_segment": "ast_get_source_segment_on_detected_decoded_source",
    "url_scheme_class": (
        "case_insensitive_prefix_https_then_http_else_none"
    ),
    "archive_suffix_class": "case_insensitive_endswith_dot_zip_else_none",
    "lexical_scope": "class_and_function_name_stack_dot_joined",
    "literal_role_precedence": (
        "docstring_then_call_arg_then_assignment_then_default_annotation_"
        "then_ordinary_expr_then_other"
    ),
    "cap_scope": (
        "record_counts_ast_nodes_and_json_bytes_global_ast_depth_per_blob_"
        "single_string_and_segment_per_record"
    ),
}
RECORD_FIELDS = {
    "string_literals": {
        "source_path", "node_id", "lineno", "col_offset", "end_lineno",
        "end_col_offset", "value", "value_sha256", "url_scheme_class",
        "archive_suffix_class", "enclosing_lexical_scope", "lexical_role",
        "is_docstring",
    },
    "call_sites": {
        "source_path", "node_id", "lineno", "col_offset", "end_lineno",
        "end_col_offset", "callee_node_id", "callee_syntax",
        "positional_argument_node_ids", "keyword_argument_node_ids",
        "source_segment", "source_segment_sha256", "enclosing_function",
    },
    "assignments": {
        "source_path", "node_id", "lineno", "col_offset", "end_lineno",
        "end_col_offset", "target_node_ids", "value_node_id",
        "target_syntax", "value_syntax", "source_segment",
        "source_segment_sha256", "enclosing_function",
    },
    "functions": {
        "source_path", "node_id", "qualified_name", "lineno", "col_offset",
        "end_lineno", "end_col_offset", "argument_names",
    },
    "import_aliases": {
        "source_path", "node_id", "lineno", "end_lineno", "import_kind",
        "module", "name", "asname", "enclosing_lexical_scope",
    },
    "expressions": {
        "source_path", "node_id", "parent_node_id", "parent_field",
        "node_type", "lineno", "col_offset", "end_lineno",
        "end_col_offset", "source_segment", "source_segment_sha256",
        "canonical_ast_dump", "enclosing_lexical_scope", "lexical_role",
    },
}
NODE_RECEIPT_FIELDS = {
    "source_path", "node_id", "parent_node_id", "parent_field",
    "preorder_index", "node_type", "lineno", "col_offset", "end_lineno",
    "end_col_offset", "depth", "canonical_ast_dump",
}
PARSE_RECEIPT_FIELDS = {
    "path", "git_blob_oid", "detected_source_encoding", "ast_parse_status",
    "ast_node_count", "maximum_ast_depth",
}
SYNTAX_NOT_EVALUABLE_FIELDS = {
    "reason", "failed_path", "error_type", "object_receipt_count",
    "object_receipt_set_sha256",
    "object_receipts_completed_before_ast_extraction", "object_receipts",
    "source_total_bytes", "parse_receipt_count", "parse_receipts",
    "parsed_prefix_ast_node_count",
    "parsed_prefix_maximum_ast_depth_observed",
    "parsed_prefix_string_literal_count",
    "parsed_prefix_call_site_count",
    "parsed_prefix_assignment_count",
    "parsed_prefix_function_count",
    "parsed_prefix_import_alias_count",
    "parsed_prefix_expression_count",
}
FAILURE_FIELDS = {
    "schema", "terminal", "reason", "observed_top_level_names",
    "execution_contract_sha256", "attempt_sha256", "preflight_sha256",
    "evidence_sha256", "p0a_fetch_head", "resume_or_rerun_authorized",
    "dataset_host_request_made",
}
OBJECT_RECEIPT_FIELDS = {
    "path", "expected_git_blob_oid", "actual_commit_path_oid",
    "git_object_type", "expected_bytes", "actual_object_size_bytes",
    "actual_content_bytes", "expected_sha256", "actual_content_sha256",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def require_canonical_root(path: Path) -> Path:
    expected = (repo_root() / CANONICAL_ROOT).resolve()
    actual = path.resolve()
    if actual != expected:
        raise ValueError(f"Noncanonical P0B output root: {actual}")
    return actual


def artifact_state(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir()}


def subprocess_git_runner(args: Sequence[str], cwd: Path) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def p0a_fetch_head() -> str | None:
    toolkit = repo_root() / P0A_ROOT / "toolkit"
    if not toolkit.is_dir():
        return None
    try:
        value = git_local("rev-parse", "FETCH_HEAD", cwd=toolkit)
    except (OSError, subprocess.SubprocessError):
        return None
    return value if HEX40_RE.fullmatch(value) else None


def runtime_receipt() -> dict[str, Any]:
    ast_path = Path(ast.__file__).resolve()
    tokenize_path = Path(tokenize.__file__).resolve()
    launcher_path = Path(sys.executable).resolve()
    base_executable_path = Path(sys._base_executable).resolve()
    parser_runtime_path = (
        Path(sys.base_prefix).resolve()
        / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    )
    stable_abi_path = Path(sys.base_prefix).resolve() / "python3.dll"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_launcher_path": str(launcher_path),
        "python_launcher_sha256": sha256(launcher_path),
        "python_base_executable_path": str(base_executable_path),
        "python_base_executable_sha256": sha256(base_executable_path),
        "python_parser_runtime_path": str(parser_runtime_path),
        "python_parser_runtime_sha256": sha256(parser_runtime_path),
        "python_stable_abi_path": str(stable_abi_path),
        "python_stable_abi_sha256": sha256(stable_abi_path),
        "ast_module_path": str(ast_path),
        "ast_module_sha256": sha256(ast_path),
        "tokenize_module_path": str(tokenize_path),
        "tokenize_module_sha256": sha256(tokenize_path),
    }


def validate_contract(
    contract_path: Path,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unexpected P0B contract schema")
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("Unexpected P0B contract status")

    parent_paths: list[tuple[Path, str]] = []
    for label, binding in contract["parents"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"P0B parent hash drift: {label}")
        value = load_json(path)
        if binding.get("required_status") and value.get("status") != binding[
            "required_status"
        ]:
            raise ValueError(f"P0B parent status drift: {label}")
        if binding.get("required_terminal") and value.get(
            "terminal"
        ) != binding["required_terminal"]:
            raise ValueError(f"P0B parent terminal drift: {label}")
        parent_paths.append((path, label))

    implementation_paths: list[tuple[Path, str]] = []
    for label, binding in contract["implementations"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"P0B implementation hash drift: {label}")
        implementation_paths.append((path, label))
    test_binding = contract["implementation_tests"]["planner_test"]
    test_path = resolve_bound(str(test_binding["path"]))
    if sha256(test_path) != str(test_binding["sha256"]):
        raise ValueError("P0B test hash drift")
    if test_definition_count(test_path) != int(
        contract["implementation_tests"]["test_count"]
    ):
        raise ValueError("P0B test count drift")

    source = contract["source_boundary"]
    closure_path = resolve_bound(str(source["p0a_closure_artifact"]["path"]))
    if closure_path.stat().st_size != int(
        source["p0a_closure_artifact"]["bytes"]
    ):
        raise ValueError("P0B P0A closure byte drift")
    if sha256(closure_path) != str(
        source["p0a_closure_artifact"]["sha256"]
    ):
        raise ValueError("P0B P0A closure hash drift")
    closure = load_json(closure_path)
    observation = closure["observation"]
    for label, binding in source["p0a_artifacts"].items():
        path = resolve_bound(str(binding["path"]))
        if (
            path.stat().st_size != int(binding["bytes"])
            or sha256(path) != str(binding["sha256"])
        ):
            raise ValueError(f"P0B P0A artifact drift: {label}")
    if (
        observation["closure_blob_count"] != source["exact_blob_count"]
        or observation["closure_total_source_bytes"]
        != source["exact_total_source_bytes"]
    ):
        raise ValueError("P0B P0A closure aggregate drift")
    row_manifest = [
        {
            key: row[key]
            for key in ("path", "git_blob_oid", "bytes", "sha256")
        }
        for row in observation["closure_rows"]
    ]
    if hashlib.sha256(canonical_json_bytes(row_manifest)).hexdigest() != str(
        source["exact_ordered_row_manifest_sha256"]
    ):
        raise ValueError("P0B exact ordered row manifest drift")
    p0a_root = (repo_root() / P0A_ROOT).resolve()
    if not validate_p0a_terminal(
        p0a_root, p0a_artifact_state(p0a_root)
    ):
        raise ValueError("P0B P0A terminal does not validate")
    if p0a_fetch_head() != source["toolkit_commit"]:
        raise ValueError("P0B P0A FETCH_HEAD drift")
    if contract["runtime_lock"] != runtime_receipt():
        raise ValueError("P0B Python AST runtime drift")
    if contract["algorithm_lock"] != EXPECTED_ALGORITHM_LOCK:
        raise ValueError("P0B extraction algorithm drift")
    for key in (
        "unresolved_or_unreachable_source_read",
        "external_txt_or_config_read",
        "new_git_fetch_checkout_or_network",
    ):
        if source[key] is not False:
            raise ValueError(f"P0B forbidden source authority drift: {key}")
    if (
        contract["frozen_extraction"]["no_truncation_drop_sampling_or_early_success"]
        is not True
        or contract["frozen_extraction"][
            "all_object_receipts_complete_before_ast"
        ]
        is not True
    ):
        raise ValueError("P0B extraction completeness drift")
    expected_firewall = {
        "ast_parse_pycf_only_ast_only": True,
        "code_object_or_bytecode_compiled": False,
        "source_code_import_or_execution": False,
        "provider_control_flow_or_url_derivation_interpreted": False,
        "official_provider_url_template_established": False,
        "exact_198_parent_archive_url_mapping_established": False,
        "dataset_host_request_made": False,
        "dataset_zip_or_payload_read": False,
    }
    if contract["firewall"] != expected_firewall:
        raise ValueError("P0B firewall drift")
    authorization = contract["authorization"]
    for key in (
        "new_git_fetch_checkout_or_network_authorized",
        "dataset_host_request_authorized",
        "p0c_execution_authorized_automatically",
        "p1_s0b_payload_or_effect_authorized",
        "research_mainline_or_default_app_changed",
        "production_or_safety_claim_authorized",
    ):
        if authorization[key] is not False:
            raise ValueError(f"P0B authorization drift: {key}")
    if (
        authorization["commit_and_push_contract_implementation_and_tests"]
        is not True
        or authorization[
            "execute_once_only_after_push_git_gate_and_double_audit"
        ]
        is not True
        or authorization[
            "p0b_exact_local_source_read_authorized_after_all_gates"
        ]
        is not True
    ):
        raise ValueError("P0B positive authorization drift")

    if verify_git:
        if git_local("rev-parse", "HEAD") != git_local(
            "rev-parse", "origin/master"
        ):
            raise ValueError("HEAD differs from origin/master")
        for path, label in [
            (contract_path, "P0B execution contract"),
            *parent_paths,
            *implementation_paths,
            (test_path, "P0B planner test"),
        ]:
            require_tracked_clean(path, label)
    return {
        "contract": contract,
        "contract_path": contract_path,
        "closure": closure,
        "closure_path": closure_path,
    }


def source_segment(source: str, node: ast.AST, cap: int) -> str:
    value = ast.get_source_segment(source, node) or ""
    if len(value.encode("utf-8")) > cap:
        raise ValueError("P0B source segment exceeds cap")
    return value


def index_ast(
    path: str,
    tree: ast.AST,
    maximum_depth: int,
) -> tuple[
    dict[ast.AST, str],
    dict[ast.AST, tuple[ast.AST, str]],
    list[dict[str, Any]],
    dict[ast.AST, dict[str, str]],
    int,
    int,
]:
    node_ids: dict[ast.AST, str] = {}
    parents: dict[ast.AST, tuple[ast.AST, str]] = {}
    occurrences: list[dict[str, Any]] = []
    maximum_observed_depth = 0

    def visit(
        node: ast.AST,
        depth: int,
        parent_index: int | None = None,
        parent_field: str | None = None,
    ) -> None:
        nonlocal maximum_observed_depth
        if depth > maximum_depth:
            raise ValueError("P0B AST depth cap exceeded")
        maximum_observed_depth = max(maximum_observed_depth, depth)
        occurrence_index = len(occurrences)
        occurrences.append(
            {
                "node": node,
                "parent_index": parent_index,
                "parent_field": parent_field,
                "depth": depth,
            }
        )
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                parents[value] = (node, field)
                visit(value, depth + 1, occurrence_index, field)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        child_field = f"{field}[{index}]"
                        parents[child] = (node, child_field)
                        visit(
                            child,
                            depth + 1,
                            occurrence_index,
                            child_field,
                        )

    visit(tree, 0)
    occurrence_ids: dict[int, str] = {}
    occurrence_dumps: dict[int, str] = {}
    edge_node_ids_by_node: dict[ast.AST, dict[str, str]] = {}
    children_by_parent: dict[int, list[int]] = {}
    for index, occurrence in enumerate(occurrences):
        parent_index = occurrence["parent_index"]
        if parent_index is not None:
            children_by_parent.setdefault(parent_index, []).append(index)
    for preorder in range(len(occurrences) - 1, -1, -1):
        occurrence = occurrences[preorder]
        node = occurrence["node"]
        edge_ids = {
            occurrences[index]["parent_field"]: occurrence_ids[index]
            for index in children_by_parent.get(preorder, [])
        }
        dump = canonical_shallow_ast_dump(node, edge_ids)
        occurrence_dumps[preorder] = dump
        identity = "\0".join(
            [
                path,
                str(preorder),
                type(node).__name__,
                str(getattr(node, "lineno", "")),
                str(getattr(node, "col_offset", "")),
                str(getattr(node, "end_lineno", "")),
                str(getattr(node, "end_col_offset", "")),
                dump,
            ]
        )
        node_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        occurrence_ids[preorder] = node_id
        node_ids[node] = node_id
        edge_node_ids_by_node[node] = edge_ids
    node_receipts = []
    for preorder, occurrence in enumerate(occurrences):
        node = occurrence["node"]
        parent_index = occurrence["parent_index"]
        node_receipts.append(
            {
                "source_path": path,
                "node_id": occurrence_ids[preorder],
                "parent_node_id": (
                    occurrence_ids[parent_index]
                    if parent_index is not None
                    else None
                ),
                "parent_field": occurrence["parent_field"],
                "preorder_index": preorder,
                "node_type": type(node).__name__,
                "lineno": getattr(node, "lineno", None),
                "col_offset": getattr(node, "col_offset", None),
                "end_lineno": getattr(node, "end_lineno", None),
                "end_col_offset": getattr(node, "end_col_offset", None),
                "depth": occurrence["depth"],
                "canonical_ast_dump": occurrence_dumps[preorder],
            }
        )
    return (
        node_ids,
        parents,
        node_receipts,
        edge_node_ids_by_node,
        len(occurrences),
        maximum_observed_depth,
    )


def canonical_shallow_ast_dump(
    node: ast.AST,
    edge_node_ids: dict[str, str],
) -> str:
    def scalar(value: Any) -> Any:
        if value is Ellipsis:
            return {"scalar_type": "ellipsis"}
        if isinstance(value, bytes):
            return {"scalar_type": "bytes", "hex": value.hex()}
        if isinstance(value, complex):
            return {
                "scalar_type": "complex",
                "real_repr": repr(value.real),
                "imag_repr": repr(value.imag),
            }
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ValueError(
            f"Unsupported P0B AST scalar type: {type(value).__name__}"
        )

    fields: dict[str, Any] = {}
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            fields[field] = {"node_id": edge_node_ids[field]}
        elif isinstance(value, list):
            fields[field] = [
                {"node_id": edge_node_ids[f"{field}[{index}]"]}
                if isinstance(item, ast.AST)
                else scalar(item)
                for index, item in enumerate(value)
            ]
        else:
            fields[field] = scalar(value)
    return canonical_json_bytes(
        {"node_type": type(node).__name__, "fields": fields}
    ).decode("utf-8").rstrip("\n")


class EvidenceVisitor(ast.NodeVisitor):
    def __init__(
        self,
        path: str,
        source: str,
        rules: dict[str, Any],
        node_ids: dict[ast.AST, str],
        parents: dict[ast.AST, tuple[ast.AST, str]],
        edge_node_ids_by_node: dict[ast.AST, dict[str, str]],
    ) -> None:
        self.path = path
        self.source = source
        self.rules = rules
        self.node_ids = node_ids
        self.parents = parents
        self.edge_node_ids_by_node = edge_node_ids_by_node
        self.scope_stack: list[str] = []
        self.strings: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.assignments: list[dict[str, Any]] = []
        self.functions: list[dict[str, Any]] = []
        self.imports: list[dict[str, Any]] = []
        self.expressions: list[dict[str, Any]] = []

    def enclosing(self) -> str | None:
        return ".".join(self.scope_stack) if self.scope_stack else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        qualified = ".".join([*self.scope_stack, node.name])
        args = [
            arg.arg
            for arg in [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
        ]
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)
        self.functions.append(
            {
                "source_path": self.path,
                "node_id": self.node_ids[node],
                "qualified_name": qualified,
                "lineno": node.lineno,
                "col_offset": node.col_offset,
                "end_lineno": node.end_lineno,
                "end_col_offset": node.end_col_offset,
                "argument_names": args,
            }
        )
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def literal_role(self, node: ast.Constant) -> tuple[str, bool]:
        parent_info = self.parents.get(node)
        if parent_info:
            parent, _ = parent_info
            if isinstance(parent, ast.Expr):
                grand = self.parents.get(parent)
                if grand:
                    owner, field = grand
                    body = getattr(owner, field.split("[", 1)[0], None)
                    if (
                        isinstance(
                            owner,
                            (
                                ast.Module,
                                ast.ClassDef,
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                            ),
                        )
                        and isinstance(body, list)
                        and body
                        and body[0] is parent
                    ):
                        label = (
                            "module_docstring"
                            if isinstance(owner, ast.Module)
                            else "class_docstring"
                            if isinstance(owner, ast.ClassDef)
                            else "function_docstring"
                        )
                        return label, True
        current: ast.AST = node
        while current in self.parents:
            parent, field = self.parents[current]
            if isinstance(parent, ast.AnnAssign):
                return (
                    "default_or_annotation"
                    if field == "annotation"
                    else "assignment_value"
                    if field == "value"
                    else "other_lexical_context",
                    False,
                )
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if field == "returns":
                    return "default_or_annotation", False
            if isinstance(parent, ast.Call):
                return (
                    "call_keyword_argument"
                    if field.startswith("keywords")
                    else "call_positional_argument",
                    False,
                )
            if isinstance(parent, (ast.Assign, ast.AugAssign, ast.NamedExpr)):
                return "assignment_value", False
            if isinstance(parent, (ast.arguments, ast.arg)):
                return "default_or_annotation", False
            if isinstance(parent, ast.Expr):
                return "ordinary_expression", False
            current = parent
        return "other_lexical_context", False

    def expression_role(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.literal_role(node)[0]
        current = node
        while current in self.parents:
            parent, field = self.parents[current]
            if isinstance(parent, ast.AnnAssign):
                return (
                    "default_or_annotation"
                    if field == "annotation"
                    else "assignment_value"
                    if field == "value"
                    else "other_lexical_context"
                )
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if field == "returns":
                    return "default_or_annotation"
            if isinstance(parent, ast.Call):
                return (
                    "call_keyword_argument"
                    if field.startswith("keywords")
                    else "call_positional_argument"
                )
            if isinstance(parent, (ast.Assign, ast.AugAssign, ast.NamedExpr)):
                return "assignment_value"
            if isinstance(parent, (ast.arguments, ast.arg)):
                return "default_or_annotation"
            current = parent
        return "other_lexical_context"

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.expr):
            cap = int(
                self.rules["maximum_single_source_segment_utf8_bytes"]
            )
            segment = source_segment(self.source, node, cap)
            parent, field = self.parents.get(node, (None, ""))
            self.expressions.append(
                {
                    "source_path": self.path,
                    "node_id": self.node_ids[node],
                    "parent_node_id": (
                        self.node_ids[parent] if parent is not None else None
                    ),
                    "parent_field": field or None,
                    "node_type": type(node).__name__,
                    "lineno": getattr(node, "lineno", None),
                    "col_offset": getattr(node, "col_offset", None),
                    "end_lineno": getattr(node, "end_lineno", None),
                    "end_col_offset": getattr(node, "end_col_offset", None),
                    "source_segment": segment,
                    "source_segment_sha256": hashlib.sha256(
                        segment.encode("utf-8")
                    ).hexdigest(),
                    "canonical_ast_dump": canonical_shallow_ast_dump(
                        node, self.edge_node_ids_by_node[node]
                    ),
                    "enclosing_lexical_scope": self.enclosing(),
                    "lexical_role": self.expression_role(node),
                }
            )
        super().generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            value = node.value
            if len(value.encode("utf-8")) > int(
                self.rules["maximum_single_string_utf8_bytes"]
            ):
                raise ValueError("P0B string literal exceeds cap")
            lower = value.lower()
            scheme = (
                "HTTPS"
                if lower.startswith("https://")
                else "HTTP"
                if lower.startswith("http://")
                else "NONE"
            )
            role, is_docstring = self.literal_role(node)
            self.strings.append(
                {
                    "source_path": self.path,
                    "node_id": self.node_ids[node],
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "end_lineno": node.end_lineno,
                    "end_col_offset": node.end_col_offset,
                    "value": value,
                    "value_sha256": hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest(),
                    "url_scheme_class": scheme,
                    "archive_suffix_class": (
                        "ZIP" if lower.endswith(".zip") else "NONE"
                    ),
                    "enclosing_lexical_scope": self.enclosing(),
                    "lexical_role": role,
                    "is_docstring": is_docstring,
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        cap = int(self.rules["maximum_single_source_segment_utf8_bytes"])
        segment = source_segment(self.source, node, cap)
        self.calls.append(
            {
                "source_path": self.path,
                "node_id": self.node_ids[node],
                "lineno": node.lineno,
                "col_offset": node.col_offset,
                "end_lineno": node.end_lineno,
                "end_col_offset": node.end_col_offset,
                "callee_node_id": self.node_ids[node.func],
                "callee_syntax": source_segment(
                    self.source, node.func, cap
                ),
                "positional_argument_node_ids": [
                    self.node_ids[arg] for arg in node.args
                ],
                "keyword_argument_node_ids": [
                    {
                        "name": keyword.arg,
                        "node_id": self.node_ids[keyword.value],
                    }
                    for keyword in node.keywords
                ],
                "source_segment": segment,
                "source_segment_sha256": hashlib.sha256(
                    segment.encode("utf-8")
                ).hexdigest(),
                "enclosing_function": self.enclosing(),
            }
        )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_assignment(node, node.targets, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_assignment(node, [node.target], node.value)
        else:
            self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._record_assignment(node, [node.target], node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_assignment(node, [node.target], node.value)

    def _record_assignment(
        self,
        node: ast.AST,
        targets: list[ast.expr],
        value: ast.expr,
    ) -> None:
        cap = int(self.rules["maximum_single_source_segment_utf8_bytes"])
        segment = source_segment(self.source, node, cap)
        self.assignments.append(
            {
                "source_path": self.path,
                "node_id": self.node_ids[node],
                "lineno": node.lineno,
                "col_offset": node.col_offset,
                "end_lineno": getattr(node, "end_lineno", None),
                "end_col_offset": getattr(node, "end_col_offset", None),
                "target_node_ids": [
                    self.node_ids[target] for target in targets
                ],
                "value_node_id": self.node_ids[value],
                "target_syntax": ",".join(
                    source_segment(self.source, target, cap)
                    for target in targets
                ),
                "value_syntax": source_segment(self.source, value, cap),
                "source_segment": segment,
                "source_segment_sha256": hashlib.sha256(
                    segment.encode("utf-8")
                ).hexdigest(),
                "enclosing_function": self.enclosing(),
            }
        )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                {
                    "source_path": self.path,
                    "node_id": self.node_ids[node],
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "import_kind": "import",
                    "module": alias.name,
                    "name": None,
                    "asname": alias.asname,
                    "enclosing_lexical_scope": self.enclosing(),
                }
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imports.append(
                {
                    "source_path": self.path,
                    "node_id": self.node_ids[node],
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "import_kind": "from",
                    "module": "." * node.level + (node.module or ""),
                    "name": alias.name,
                    "asname": alias.asname,
                    "enclosing_lexical_scope": self.enclosing(),
                }
            )
        self.generic_visit(node)


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("source_path", "")),
        int(record.get("lineno", -1)),
        int(record.get("col_offset", -1)),
        canonical_json_bytes(record),
    )


def extract_evidence(
    contract: dict[str, Any],
    closure: dict[str, Any],
    toolkit: Path,
    *,
    git_runner: GitRunner,
) -> tuple[str, dict[str, Any]]:
    observation = closure["observation"]
    if (
        observation["dynamic_import_call_count"] != 0
        or observation["indirect_dynamic_import_or_exec_count"] != 0
    ):
        return EVIDENCE_NOT_EVALUABLE, {
            "reason": "p0a_dynamic_import_evidence_nonzero",
            "source_blob_read_count": 0,
        }
    rules = contract["frozen_extraction"]
    object_receipts: list[dict[str, Any]] = []
    verified_sources: list[tuple[dict[str, Any], bytes]] = []
    parse_receipts: list[dict[str, Any]] = []
    strings: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    expressions: list[dict[str, Any]] = []
    node_receipts: list[dict[str, Any]] = []
    total_ast_nodes = 0
    maximum_ast_depth_observed = 0
    rows = observation["closure_rows"]
    if [row["path"] for row in rows] != sorted(row["path"] for row in rows):
        raise ValueError("P0B P0A closure row order drift")
    for row in rows:
        oid = str(row["git_blob_oid"])
        if not HEX40_RE.fullmatch(oid):
            raise ValueError("P0B invalid blob OID")
        commit_path = (
            f"{contract['source_boundary']['toolkit_commit']}:{row['path']}"
        )
        actual_oid = git_runner(
            ["rev-parse", commit_path], toolkit
        ).decode("ascii").strip()
        object_type = git_runner(
            ["cat-file", "-t", oid], toolkit
        ).decode("ascii").strip()
        size_text = git_runner(
            ["cat-file", "-s", oid], toolkit
        ).decode("ascii").strip()
        if not re.fullmatch(r"0|[1-9][0-9]*", size_text):
            raise ValueError(f"P0B invalid object size: {row['path']}")
        actual_size = int(size_text)
        if (
            actual_oid != oid
            or object_type != "blob"
            or actual_size != int(row["bytes"])
        ):
            raise ValueError(f"P0B object receipt mismatch: {row['path']}")
        blob = git_runner(["cat-file", "blob", oid], toolkit)
        actual_sha = hashlib.sha256(blob).hexdigest()
        if len(blob) != actual_size or actual_sha != row["sha256"]:
            raise ValueError(f"P0B blob hash mismatch: {row['path']}")
        object_receipts.append(
            {
                "path": row["path"],
                "expected_git_blob_oid": oid,
                "actual_commit_path_oid": actual_oid,
                "git_object_type": object_type,
                "expected_bytes": int(row["bytes"]),
                "actual_object_size_bytes": actual_size,
                "actual_content_bytes": len(blob),
                "expected_sha256": row["sha256"],
                "actual_content_sha256": actual_sha,
            }
        )
        verified_sources.append((row, blob))

    object_receipt_set_sha256 = hashlib.sha256(
        canonical_json_bytes(object_receipts)
    ).hexdigest()
    for row, blob in verified_sources:
        try:
            encoding, _ = tokenize.detect_encoding(
                io.BytesIO(blob).readline
            )
            source = blob.decode(encoding)
            tree = ast.parse(source, filename=str(row["path"]))
        except (UnicodeError, SyntaxError) as error:
            return EVIDENCE_NOT_EVALUABLE, {
                "reason": (
                    "verified_source_incompatible_with_frozen_encoding_"
                    "or_ast_grammar"
                ),
                "failed_path": row["path"],
                "error_type": type(error).__name__,
                "object_receipt_count": len(object_receipts),
                "object_receipt_set_sha256": object_receipt_set_sha256,
                "object_receipts_completed_before_ast_extraction": True,
                "object_receipts": object_receipts,
                "source_total_bytes": sum(
                    row["actual_content_bytes"]
                    for row in object_receipts
                ),
                "parse_receipt_count": len(parse_receipts),
                "parse_receipts": parse_receipts,
                "parsed_prefix_ast_node_count": total_ast_nodes,
                "parsed_prefix_maximum_ast_depth_observed": (
                    maximum_ast_depth_observed
                ),
                "parsed_prefix_string_literal_count": len(strings),
                "parsed_prefix_call_site_count": len(calls),
                "parsed_prefix_assignment_count": len(assignments),
                "parsed_prefix_function_count": len(functions),
                "parsed_prefix_import_alias_count": len(imports),
                "parsed_prefix_expression_count": len(expressions),
            }
        (
            node_ids,
            parents,
            blob_node_receipts,
            edge_node_ids_by_node,
            node_count,
            depth,
        ) = index_ast(
            str(row["path"]),
            tree,
            int(rules["maximum_ast_depth"]),
        )
        total_ast_nodes += node_count
        maximum_ast_depth_observed = max(
            maximum_ast_depth_observed, depth
        )
        if total_ast_nodes > int(rules["maximum_ast_nodes"]):
            raise ValueError("P0B global AST node cap exceeded")
        visitor = EvidenceVisitor(
            str(row["path"]),
            source,
            rules,
            node_ids,
            parents,
            edge_node_ids_by_node,
        )
        visitor.visit(tree)
        for current_count, added_count, cap_name in (
            (
                len(strings),
                len(visitor.strings),
                "maximum_string_literal_records",
            ),
            (
                len(calls),
                len(visitor.calls),
                "maximum_call_site_records",
            ),
            (
                len(assignments),
                len(visitor.assignments),
                "maximum_assignment_records",
            ),
            (
                len(functions),
                len(visitor.functions),
                "maximum_function_records",
            ),
            (
                len(imports),
                len(visitor.imports),
                "maximum_import_alias_records",
            ),
            (
                len(expressions),
                len(visitor.expressions),
                "maximum_expression_records",
            ),
        ):
            if current_count + added_count > int(rules[cap_name]):
                raise ValueError(
                    f"P0B evidence record cap exceeded: {cap_name}"
                )
        node_receipts.extend(blob_node_receipts)
        parse_receipts.append(
            {
                "path": row["path"],
                "git_blob_oid": row["git_blob_oid"],
                "detected_source_encoding": encoding,
                "ast_parse_status": "PARSED",
                "ast_node_count": node_count,
                "maximum_ast_depth": depth,
            }
        )
        strings.extend(visitor.strings)
        calls.extend(visitor.calls)
        assignments.extend(visitor.assignments)
        functions.extend(visitor.functions)
        imports.extend(visitor.imports)
        expressions.extend(visitor.expressions)

    for records, cap_name in (
        (strings, "maximum_string_literal_records"),
        (calls, "maximum_call_site_records"),
        (assignments, "maximum_assignment_records"),
        (functions, "maximum_function_records"),
        (imports, "maximum_import_alias_records"),
        (expressions, "maximum_expression_records"),
    ):
        if len(records) > int(rules[cap_name]):
            raise ValueError(f"P0B evidence record cap exceeded: {cap_name}")
        records.sort(key=record_key)
    evidence = {
        "object_receipt_count": len(object_receipts),
        "object_receipt_set_sha256": object_receipt_set_sha256,
        "object_receipts_completed_before_ast_extraction": True,
        "object_receipts": object_receipts,
        "parse_receipt_count": len(parse_receipts),
        "parse_receipts": parse_receipts,
        "source_total_bytes": sum(
            row["actual_content_bytes"] for row in object_receipts
        ),
        "ast_node_count": total_ast_nodes,
        "maximum_ast_depth_observed": maximum_ast_depth_observed,
        "ast_node_receipts": node_receipts,
        "string_literal_count": len(strings),
        "url_literal_count": sum(
            row["url_scheme_class"] != "NONE" for row in strings
        ),
        "zip_string_literal_count": sum(
            row["archive_suffix_class"] == "ZIP" for row in strings
        ),
        "call_site_count": len(calls),
        "assignment_count": len(assignments),
        "function_count": len(functions),
        "import_alias_count": len(imports),
        "expression_count": len(expressions),
        "string_literals": strings,
        "call_sites": calls,
        "assignments": assignments,
        "functions": functions,
        "import_aliases": imports,
        "expressions": expressions,
        "p0a_unresolved_local_import_count": len(
            observation["unresolved_local_imports"]
        ),
        "p0a_python_tree_path_count": observation[
            "python_tree_path_count"
        ],
        "p0a_static_import_closure_blob_count": observation[
            "closure_blob_count"
        ],
        "unresolved_or_unreachable_blob_followup_made": False,
        "external_txt_or_config_read": False,
        "ast_parse_pycf_only_ast_used": True,
        "code_object_or_bytecode_compiled": False,
        "source_code_or_dynamic_import_executed": False,
        "provider_control_flow_or_url_derivation_interpreted": False,
        "official_provider_url_template_established": False,
        "exact_198_parent_archive_url_mapping_established": False,
        "dataset_host_request_made": False,
        "dataset_zip_or_payload_read": False,
    }
    if len(canonical_json_bytes(evidence)) > int(
        rules["maximum_total_evidence_json_bytes"]
    ):
        raise ValueError("P0B total evidence JSON byte cap exceeded")
    return EVIDENCE_LOCKED, evidence


def execute(
    contract_path: Path,
    root: Path,
    *,
    git_runner: GitRunner = subprocess_git_runner,
    verify_git: bool = True,
) -> dict[str, Any]:
    context = validate_contract(contract_path, verify_git=verify_git)
    contract = context["contract"]
    root.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "ATTEMPT_FSYNCED_BEFORE_FIRST_P0B_SOURCE_BLOB_READ",
        "execution_contract_sha256": sha256(contract_path),
        "p0a_closure_sha256": sha256(context["closure_path"]),
        "toolkit_commit": contract["source_boundary"]["toolkit_commit"],
        "new_git_fetch_or_network_authorized": False,
        "dataset_host_request_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["attempt"], attempt)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_P0B_SOURCE_BLOB_READ",
        "attempt_sha256": sha256(root / FILENAMES["attempt"]),
        "p0a_terminal_validated": True,
        "p0a_fetch_head": p0a_fetch_head(),
        "exact_blob_count": contract["source_boundary"]["exact_blob_count"],
        "exact_total_source_bytes": contract["source_boundary"][
            "exact_total_source_bytes"
        ],
        "runtime_lock": runtime_receipt(),
        "new_git_fetch_or_network_made": False,
        "dataset_host_request_made": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["preflight"], preflight)
    terminal, observation = extract_evidence(
        contract,
        context["closure"],
        repo_root() / P0A_ROOT / "toolkit",
        git_runner=git_runner,
    )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "status": (
            "EXACT_SOURCE_AST_EVIDENCE_LOCKED"
            if terminal == EVIDENCE_LOCKED
            else "EXACT_SOURCE_AST_EVIDENCE_NOT_EVALUABLE"
        ),
        "terminal": terminal,
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "p0a_closure_sha256": sha256(context["closure_path"]),
        },
        "toolkit_commit": contract["source_boundary"]["toolkit_commit"],
        "observation": observation,
    }
    if len(canonical_json_bytes(evidence)) > int(
        contract["frozen_extraction"]["maximum_total_evidence_json_bytes"]
    ):
        raise ValueError("P0B complete evidence artifact byte cap exceeded")
    write_json_exclusive_fsync(root / FILENAMES["evidence"], evidence)
    result = {
        "schema": RESULT_SCHEMA,
        "terminal": terminal,
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "evidence_sha256": sha256(root / FILENAMES["evidence"]),
        },
        "provider_resolution_established": False,
        "dataset_host_request_made": False,
        "dataset_zip_or_payload_read": False,
        "source_code_or_dynamic_import_executed": False,
        "p0c_execution_authorized_automatically": False,
        "next_authority": (
            "freeze_hash_bound_p0c_provider_resolution_contract"
            if terminal == EVIDENCE_LOCKED
            else "stop_provider_resolution_as_source_evidence_not_evaluable"
        ),
    }
    write_json_exclusive_fsync(root / FILENAMES["result"], result)
    return result


def object_receipts_match(
    observation: dict[str, Any],
    expected_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> bool:
    receipts = observation.get("object_receipts")
    if not isinstance(receipts, list):
        return False
    if (
        observation.get("object_receipt_count") != len(receipts)
        or len(receipts) != len(expected_rows)
        or len(receipts)
        != int(contract["source_boundary"]["exact_blob_count"])
        or observation.get("object_receipts_completed_before_ast_extraction")
        is not True
        or observation.get("object_receipt_set_sha256")
        != hashlib.sha256(canonical_json_bytes(receipts)).hexdigest()
        or observation.get("source_total_bytes")
        != int(contract["source_boundary"]["exact_total_source_bytes"])
    ):
        return False
    for row, expected in zip(receipts, expected_rows, strict=True):
        if (
            set(row) != OBJECT_RECEIPT_FIELDS
            or row.get("path") != expected["path"]
            or row.get("expected_git_blob_oid") != expected["git_blob_oid"]
            or row.get("actual_commit_path_oid") != expected["git_blob_oid"]
            or row.get("git_object_type") != "blob"
            or row.get("expected_bytes") != expected["bytes"]
            or row.get("actual_object_size_bytes") != expected["bytes"]
            or row.get("actual_content_bytes") != expected["bytes"]
            or row.get("expected_sha256") != expected["sha256"]
            or row.get("actual_content_sha256") != expected["sha256"]
        ):
            return False
    return True


def parse_receipt_matches(
    row: Any,
    expected: dict[str, Any],
    maximum_nodes: int,
    maximum_depth: int,
) -> bool:
    return (
        isinstance(row, dict)
        and set(row) == PARSE_RECEIPT_FIELDS
        and row.get("path") == expected["path"]
        and row.get("git_blob_oid") == expected["git_blob_oid"]
        and isinstance(row.get("detected_source_encoding"), str)
        and bool(row["detected_source_encoding"])
        and row.get("ast_parse_status") == "PARSED"
        and isinstance(row.get("ast_node_count"), int)
        and not isinstance(row.get("ast_node_count"), bool)
        and 1 <= row["ast_node_count"] <= maximum_nodes
        and isinstance(row.get("maximum_ast_depth"), int)
        and not isinstance(row.get("maximum_ast_depth"), bool)
        and 0 <= row["maximum_ast_depth"] <= maximum_depth
    )


def canonical_dump_parts(
    text: Any,
) -> tuple[str, dict[str, Any], dict[str, str]] | None:
    if not isinstance(text, str):
        return None
    try:
        dump = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(dump, dict)
        or set(dump) != {"node_type", "fields"}
        or not isinstance(dump.get("node_type"), str)
        or not isinstance(dump.get("fields"), dict)
        or canonical_json_bytes(dump).decode("utf-8").rstrip("\n") != text
    ):
        return None

    def scalar_valid(value: Any) -> bool:
        if value is None or isinstance(value, (bool, int, float, str)):
            return True
        if not isinstance(value, dict):
            return False
        if value == {"scalar_type": "ellipsis"}:
            return True
        if (
            set(value) == {"scalar_type", "hex"}
            and value.get("scalar_type") == "bytes"
            and isinstance(value.get("hex"), str)
            and re.fullmatch(r"(?:[0-9a-f]{2})*", value["hex"])
        ):
            return True
        return (
            set(value) == {"scalar_type", "real_repr", "imag_repr"}
            and value.get("scalar_type") == "complex"
            and isinstance(value.get("real_repr"), str)
            and isinstance(value.get("imag_repr"), str)
        )

    edges: dict[str, str] = {}
    for field, value in dump["fields"].items():
        if not isinstance(field, str):
            return None
        if isinstance(value, dict) and set(value) == {"node_id"}:
            if not HEX64_RE.fullmatch(str(value["node_id"])):
                return None
            edges[field] = value["node_id"]
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict) and set(item) == {"node_id"}:
                    if not HEX64_RE.fullmatch(str(item["node_id"])):
                        return None
                    edges[f"{field}[{index}]"] = item["node_id"]
                elif not scalar_valid(item):
                    return None
        elif not scalar_valid(value):
            return None
    return dump["node_type"], dump["fields"], edges


def node_receipt_id(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        "\0".join(
            [
                str(row["source_path"]),
                str(row["preorder_index"]),
                str(row["node_type"]),
                str(row["lineno"] if row["lineno"] is not None else ""),
                str(
                    row["col_offset"]
                    if row["col_offset"] is not None
                    else ""
                ),
                str(
                    row["end_lineno"]
                    if row["end_lineno"] is not None
                    else ""
                ),
                str(
                    row["end_col_offset"]
                    if row["end_col_offset"] is not None
                    else ""
                ),
                str(row["canonical_ast_dump"]),
            ]
        ).encode("utf-8")
    ).hexdigest()


def canonical_preorder_node_ids(
    root_id: str,
    node_map: dict[str, dict[str, Any]],
    node_dump_map: dict[
        str, tuple[str, dict[str, Any], dict[str, str]]
    ],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id not in node_map:
            raise ValueError("P0B canonical AST graph cycle or duplicate child")
        seen.add(node_id)
        ordered.append(node_id)
        node_type, fields, _ = node_dump_map[node_id]
        node_class = getattr(ast, node_type, None)
        if not isinstance(node_class, type) or not issubclass(
            node_class, ast.AST
        ):
            raise ValueError("P0B canonical AST node type invalid")
        children: list[str] = []
        for field in node_class._fields:
            value = fields[field]
            if isinstance(value, dict) and set(value) == {"node_id"}:
                children.append(str(value["node_id"]))
            elif isinstance(value, list):
                children.extend(
                    str(item["node_id"])
                    for item in value
                    if isinstance(item, dict)
                    and set(item) == {"node_id"}
                )
        stack.extend(reversed(children))
    return ordered


def validate_existing_terminal(root: Path, names: set[str]) -> bool:
    locked = {
        FILENAMES["attempt"],
        FILENAMES["preflight"],
        FILENAMES["evidence"],
        FILENAMES["result"],
    }
    contract_path = (repo_root() / CONTRACT_RELATIVE_PATH).resolve()
    if names == locked:
        try:
            contract = load_json(contract_path)
            attempt = load_json(root / FILENAMES["attempt"])
            preflight = load_json(root / FILENAMES["preflight"])
            evidence = load_json(root / FILENAMES["evidence"])
            result = load_json(root / FILENAMES["result"])
            terminal = evidence.get("terminal")
            if (
                contract.get("schema") != CONTRACT_SCHEMA
                or contract.get("status") != CONTRACT_STATUS
                or attempt.get("schema") != ATTEMPT_SCHEMA
                or attempt.get("status")
                != "ATTEMPT_FSYNCED_BEFORE_FIRST_P0B_SOURCE_BLOB_READ"
                or preflight.get("schema") != PREFLIGHT_SCHEMA
                or preflight.get("status")
                != (
                    "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_P0B_"
                    "SOURCE_BLOB_READ"
                )
                or evidence.get("schema") != EVIDENCE_SCHEMA
                or result.get("schema") != RESULT_SCHEMA
                or terminal not in {EVIDENCE_LOCKED, EVIDENCE_NOT_EVALUABLE}
                or result.get("terminal") != terminal
                or attempt.get("execution_contract_sha256")
                != sha256(contract_path)
                or attempt.get("p0a_closure_sha256")
                != contract["source_boundary"]["p0a_closure_artifact"][
                    "sha256"
                ]
                or attempt.get("new_git_fetch_or_network_authorized")
                is not False
                or attempt.get("dataset_host_request_authorized") is not False
                or attempt.get("toolkit_commit")
                != contract["source_boundary"]["toolkit_commit"]
                or preflight.get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
                or preflight.get("p0a_fetch_head")
                != contract["source_boundary"]["toolkit_commit"]
                or preflight.get("runtime_lock") != contract["runtime_lock"]
                or preflight.get("p0a_terminal_validated") is not True
                or preflight.get("exact_blob_count")
                != contract["source_boundary"]["exact_blob_count"]
                or preflight.get("exact_total_source_bytes")
                != contract["source_boundary"]["exact_total_source_bytes"]
                or preflight.get("new_git_fetch_or_network_made")
                is not False
                or preflight.get("dataset_host_request_made") is not False
            ):
                return False
            if (
                evidence.get("bindings", {}).get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
                or evidence.get("bindings", {}).get("preflight_sha256")
                != sha256(root / FILENAMES["preflight"])
                or evidence.get("bindings", {}).get("p0a_closure_sha256")
                != contract["source_boundary"]["p0a_closure_artifact"][
                    "sha256"
                ]
                or evidence.get("toolkit_commit")
                != contract["source_boundary"]["toolkit_commit"]
                or result.get("bindings", {}).get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
                or result.get("bindings", {}).get("preflight_sha256")
                != sha256(root / FILENAMES["preflight"])
                or result.get("bindings", {}).get("evidence_sha256")
                != sha256(root / FILENAMES["evidence"])
                or result.get("provider_resolution_established") is not False
                or result.get("dataset_host_request_made") is not False
                or result.get("dataset_zip_or_payload_read") is not False
                or result.get("source_code_or_dynamic_import_executed")
                is not False
                or result.get("p0c_execution_authorized_automatically")
                is not False
            ):
                return False
            observation = evidence.get("observation")
            if not isinstance(observation, dict):
                return False
            closure = load_json(
                resolve_bound(
                    contract["source_boundary"]["p0a_closure_artifact"][
                        "path"
                    ]
                )
            )
            expected_rows = closure["observation"]["closure_rows"]
            maximum_nodes = int(
                contract["frozen_extraction"]["maximum_ast_nodes"]
            )
            maximum_depth = int(
                contract["frozen_extraction"]["maximum_ast_depth"]
            )
            if len(canonical_json_bytes(evidence)) > int(
                contract["frozen_extraction"][
                    "maximum_total_evidence_json_bytes"
                ]
            ):
                return False
            if terminal == EVIDENCE_NOT_EVALUABLE:
                if (
                    evidence.get("status")
                    != "EXACT_SOURCE_AST_EVIDENCE_NOT_EVALUABLE"
                    or result.get("next_authority")
                    != (
                        "stop_provider_resolution_as_source_evidence_"
                        "not_evaluable"
                    )
                ):
                    return False
                reason = observation.get("reason")
                if reason == "p0a_dynamic_import_evidence_nonzero":
                    return (
                        observation
                        == {
                            "reason": reason,
                            "source_blob_read_count": 0,
                        }
                        and (
                            closure["observation"][
                                "dynamic_import_call_count"
                            ]
                            != 0
                            or closure["observation"][
                                "indirect_dynamic_import_or_exec_count"
                            ]
                            != 0
                        )
                    )
                if reason != (
                    "verified_source_incompatible_with_frozen_encoding_"
                    "or_ast_grammar"
                ) or set(observation) != SYNTAX_NOT_EVALUABLE_FIELDS or (
                    not object_receipts_match(
                        observation, expected_rows, contract
                    )
                ) or (
                    closure["observation"]["dynamic_import_call_count"] != 0
                    or closure["observation"][
                        "indirect_dynamic_import_or_exec_count"
                    ]
                    != 0
                ):
                    return False
                parse_receipts = observation.get("parse_receipts")
                if (
                    not isinstance(parse_receipts, list)
                    or observation.get("parse_receipt_count")
                    != len(parse_receipts)
                    or len(parse_receipts) >= len(expected_rows)
                ):
                    return False
                if observation.get("failed_path") != expected_rows[
                    len(parse_receipts)
                ]["path"]:
                    return False
                for parsed, expected in zip(
                    parse_receipts,
                    expected_rows[: len(parse_receipts)],
                    strict=True,
                ):
                    if not parse_receipt_matches(
                        parsed, expected, maximum_nodes, maximum_depth
                    ):
                        return False
                prefix_node_count = sum(
                    row["ast_node_count"] for row in parse_receipts
                )
                prefix_maximum_depth = (
                    max(
                        row["maximum_ast_depth"]
                        for row in parse_receipts
                    )
                    if parse_receipts
                    else 0
                )
                if (
                    prefix_node_count > maximum_nodes
                    or observation.get("parsed_prefix_ast_node_count")
                    != prefix_node_count
                    or observation.get(
                        "parsed_prefix_maximum_ast_depth_observed"
                    )
                    != prefix_maximum_depth
                ):
                    return False
                for count_key, cap_key in (
                    (
                        "parsed_prefix_string_literal_count",
                        "maximum_string_literal_records",
                    ),
                    (
                        "parsed_prefix_call_site_count",
                        "maximum_call_site_records",
                    ),
                    (
                        "parsed_prefix_assignment_count",
                        "maximum_assignment_records",
                    ),
                    (
                        "parsed_prefix_function_count",
                        "maximum_function_records",
                    ),
                    (
                        "parsed_prefix_import_alias_count",
                        "maximum_import_alias_records",
                    ),
                    (
                        "parsed_prefix_expression_count",
                        "maximum_expression_records",
                    ),
                ):
                    count = observation.get(count_key)
                    if (
                        not isinstance(count, int)
                        or isinstance(count, bool)
                        or count < 0
                        or count
                        > int(contract["frozen_extraction"][cap_key])
                    ):
                        return False
                return (
                    observation.get("error_type")
                    in {"SyntaxError", "UnicodeDecodeError"}
                    and isinstance(observation.get("failed_path"), str)
                )
            if (
                evidence.get("status") != "EXACT_SOURCE_AST_EVIDENCE_LOCKED"
                or result.get("next_authority")
                != "freeze_hash_bound_p0c_provider_resolution_contract"
                or closure["observation"]["dynamic_import_call_count"] != 0
                or closure["observation"][
                    "indirect_dynamic_import_or_exec_count"
                ]
                != 0
            ):
                return False
            receipts = observation.get("object_receipts")
            parse_receipts = observation.get("parse_receipts")
            if not isinstance(receipts, list) or not isinstance(
                parse_receipts, list
            ):
                return False
            if (
                [row.get("path") for row in receipts]
                != [row["path"] for row in expected_rows]
                or observation.get("object_receipt_count") != len(receipts)
                or observation.get("parse_receipt_count")
                != len(parse_receipts)
                or len(parse_receipts) != len(receipts)
                or observation.get("source_total_bytes")
                != sum(
                    int(row.get("actual_content_bytes", -1))
                    for row in receipts
                )
                or len(receipts)
                != int(contract["source_boundary"]["exact_blob_count"])
                or observation.get("source_total_bytes")
                != int(
                    contract["source_boundary"]["exact_total_source_bytes"]
                )
                or observation.get(
                    "object_receipts_completed_before_ast_extraction"
                )
                is not True
                or observation.get("object_receipt_set_sha256")
                != hashlib.sha256(
                    canonical_json_bytes(receipts)
                ).hexdigest()
                or not object_receipts_match(
                    observation, expected_rows, contract
                )
            ):
                return False
            for row, expected, parsed in zip(
                receipts, expected_rows, parse_receipts, strict=True
            ):
                if (
                    row.get("path") != expected["path"]
                    or row.get("expected_git_blob_oid")
                    != expected["git_blob_oid"]
                    or row.get("actual_commit_path_oid")
                    != expected["git_blob_oid"]
                    or row.get("git_object_type") != "blob"
                    or row.get("expected_bytes") != expected["bytes"]
                    or row.get("actual_object_size_bytes")
                    != expected["bytes"]
                    or row.get("actual_content_bytes") != expected["bytes"]
                    or row.get("expected_sha256") != expected["sha256"]
                    or row.get("actual_content_sha256")
                    != expected["sha256"]
                    or not parse_receipt_matches(
                        parsed, expected, maximum_nodes, maximum_depth
                    )
                ):
                    return False
            for key, count_key, cap_key in (
                (
                    "string_literals",
                    "string_literal_count",
                    "maximum_string_literal_records",
                ),
                ("call_sites", "call_site_count", "maximum_call_site_records"),
                (
                    "assignments",
                    "assignment_count",
                    "maximum_assignment_records",
                ),
                ("functions", "function_count", "maximum_function_records"),
                (
                    "import_aliases",
                    "import_alias_count",
                    "maximum_import_alias_records",
                ),
                (
                    "expressions",
                    "expression_count",
                    "maximum_expression_records",
                ),
            ):
                records = observation.get(key)
                if (
                    not isinstance(records, list)
                    or records != sorted(records, key=record_key)
                    or observation.get(count_key) != len(records)
                    or len(records)
                    > int(contract["frozen_extraction"][cap_key])
                ):
                    return False
            allowed_paths = {row["path"] for row in expected_rows}
            nodes = observation.get("ast_node_receipts")
            if (
                not isinstance(nodes, list)
                or len(nodes) != observation.get("ast_node_count")
            ):
                return False
            node_map: dict[str, dict[str, Any]] = {}
            node_dump_map: dict[
                str, tuple[str, dict[str, Any], dict[str, str]]
            ] = {}
            path_nodes: dict[str, list[dict[str, Any]]] = {
                row["path"]: [] for row in expected_rows
            }
            for row in nodes:
                node_id = str(row.get("node_id", ""))
                path = row.get("source_path")
                dump_parts = canonical_dump_parts(
                    row.get("canonical_ast_dump")
                )
                node_class = getattr(
                    ast, str(row.get("node_type", "")), None
                )
                if (
                    set(row) != NODE_RECEIPT_FIELDS
                    or path not in allowed_paths
                    or not HEX64_RE.fullmatch(node_id)
                    or node_id in node_map
                    or not isinstance(row.get("node_type"), str)
                    or not isinstance(row.get("preorder_index"), int)
                    or isinstance(row.get("preorder_index"), bool)
                    or not isinstance(row.get("depth"), int)
                    or isinstance(row.get("depth"), bool)
                    or row["depth"] < 0
                    or row["depth"] > maximum_depth
                    or any(
                        value is not None
                        and (
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value < 0
                        )
                        for value in (
                            row.get("lineno"),
                            row.get("col_offset"),
                            row.get("end_lineno"),
                            row.get("end_col_offset"),
                        )
                    )
                    or dump_parts is None
                    or dump_parts[0] != row["node_type"]
                    or not isinstance(node_class, type)
                    or not issubclass(node_class, ast.AST)
                    or set(dump_parts[1]) != set(node_class._fields)
                    or node_receipt_id(row) != node_id
                ):
                    return False
                node_map[node_id] = row
                node_dump_map[node_id] = dump_parts
                path_nodes[str(path)].append(row)
            if [row["source_path"] for row in nodes] != [
                expected["path"]
                for expected, parsed in zip(
                    expected_rows, parse_receipts, strict=True
                )
                for _ in range(parsed["ast_node_count"])
            ]:
                return False
            for expected, parsed in zip(
                expected_rows, parse_receipts, strict=True
            ):
                path = expected["path"]
                rows_for_path = path_nodes[path]
                if (
                    len(rows_for_path) != parsed["ast_node_count"]
                    or [row["preorder_index"] for row in rows_for_path]
                    != list(range(len(rows_for_path)))
                    or max(row["depth"] for row in rows_for_path)
                    != parsed["maximum_ast_depth"]
                ):
                    return False
                for row in rows_for_path:
                    parent_id = row["parent_node_id"]
                    if row["preorder_index"] == 0:
                        if (
                            parent_id is not None
                            or row["parent_field"] is not None
                            or row["depth"] != 0
                            or row["node_type"] != "Module"
                        ):
                            return False
                    else:
                        parent = node_map.get(str(parent_id))
                        if (
                            parent is None
                            or parent["source_path"] != path
                            or not isinstance(row["parent_field"], str)
                            or parent["preorder_index"]
                            >= row["preorder_index"]
                            or parent["depth"] != row["depth"] - 1
                        ):
                            return False
            for node_id, row in node_map.items():
                edges = node_dump_map[node_id][2]
                if len(set(edges.values())) != len(edges):
                    return False
                for field, child_id in edges.items():
                    child = node_map.get(child_id)
                    if (
                        child is None
                        or child["source_path"] != row["source_path"]
                        or child["parent_node_id"] != node_id
                        or child["parent_field"] != field
                    ):
                        return False
                if row["parent_node_id"] is not None:
                    parent_edges = node_dump_map[
                        row["parent_node_id"]
                    ][2]
                    if (
                        parent_edges.get(str(row["parent_field"]))
                        != node_id
                    ):
                        return False
            for expected in expected_rows:
                rows_for_path = path_nodes[expected["path"]]
                if canonical_preorder_node_ids(
                    rows_for_path[0]["node_id"],
                    node_map,
                    node_dump_map,
                ) != [row["node_id"] for row in rows_for_path]:
                    return False
            for key, records in (
                (key, observation[key]) for key in RECORD_FIELDS
            ):
                for row in records:
                    node_id = row.get("node_id")
                    node = node_map.get(str(node_id))
                    if (
                        set(row) != RECORD_FIELDS[key]
                        or row.get("source_path") not in allowed_paths
                        or node is None
                        or node["source_path"] != row["source_path"]
                    ):
                        return False
                    for field in (
                        "lineno",
                        "col_offset",
                        "end_lineno",
                        "end_col_offset",
                    ):
                        if field in row and row[field] != node[field]:
                            return False
            expression_ids = {
                row["node_id"] for row in observation["expressions"]
            }
            expression_map = {
                row["node_id"]: row for row in observation["expressions"]
            }
            expected_expression_ids = {
                node_id
                for node_id, node in node_map.items()
                if issubclass(getattr(ast, node["node_type"]), ast.expr)
            }
            expected_string_ids = {
                node_id
                for node_id, node in node_map.items()
                if node["node_type"] == "Constant"
                and isinstance(node_dump_map[node_id][1].get("value"), str)
            }
            expected_call_ids = {
                node_id
                for node_id, node in node_map.items()
                if node["node_type"] == "Call"
            }
            expected_assignment_ids = {
                node_id
                for node_id, node in node_map.items()
                if node["node_type"] in {"Assign", "AugAssign", "NamedExpr"}
                or (
                    node["node_type"] == "AnnAssign"
                    and isinstance(
                        node_dump_map[node_id][1].get("value"), dict
                    )
                )
            }

            def exact_record_coverage(
                key: str, expected_ids: set[str]
            ) -> bool:
                ids = [row["node_id"] for row in observation[key]]
                return len(ids) == len(set(ids)) and set(ids) == expected_ids

            if (
                len(expression_ids) != len(observation["expressions"])
                or expression_ids != expected_expression_ids
                or not exact_record_coverage(
                    "string_literals", expected_string_ids
                )
                or not exact_record_coverage("call_sites", expected_call_ids)
                or not exact_record_coverage(
                    "assignments", expected_assignment_ids
                )
                or observation.get("ast_node_count")
                != sum(
                    int(row["ast_node_count"]) for row in parse_receipts
                )
                or observation.get("ast_node_count")
                > maximum_nodes
                or observation.get("maximum_ast_depth_observed")
                != max(
                    int(row["maximum_ast_depth"])
                    for row in parse_receipts
                )
                or observation.get("maximum_ast_depth_observed")
                > maximum_depth
            ):
                return False

            for row in observation["expressions"]:
                node = node_map[row["node_id"]]
                if (
                    row["node_type"] != node["node_type"]
                    or row["parent_node_id"] != node["parent_node_id"]
                    or row["parent_field"] != node["parent_field"]
                    or row["canonical_ast_dump"]
                    != node["canonical_ast_dump"]
                    or not issubclass(
                        getattr(ast, row["node_type"]), ast.expr
                    )
                ):
                    return False

            def node_fields(node_id: str) -> dict[str, Any]:
                return node_dump_map[node_id][1]

            def child_ref(value: Any) -> str | None:
                if isinstance(value, dict) and set(value) == {"node_id"}:
                    return str(value["node_id"])
                return None

            def child_ref_list(value: Any) -> list[str] | None:
                if not isinstance(value, list):
                    return None
                refs = [child_ref(item) for item in value]
                return (
                    [str(item) for item in refs]
                    if all(item is not None for item in refs)
                    else None
                )

            def lexical_scope(node_id: str, include_self: bool = False) -> str | None:
                names: list[str] = []
                current_id: str | None = (
                    node_id
                    if include_self
                    else node_map[node_id]["parent_node_id"]
                )
                while current_id is not None:
                    current = node_map[current_id]
                    if current["node_type"] in {
                        "ClassDef", "FunctionDef", "AsyncFunctionDef"
                    }:
                        name = node_fields(current_id).get("name")
                        if not isinstance(name, str):
                            return None
                        names.append(name)
                    current_id = current["parent_node_id"]
                return ".".join(reversed(names)) if names else None

            def lexical_role(node_id: str) -> tuple[str, bool]:
                current_id = node_id
                parent_id = node_map[current_id]["parent_node_id"]
                parent_field = node_map[current_id]["parent_field"]
                if (
                    parent_id is not None
                    and node_map[parent_id]["node_type"] == "Expr"
                ):
                    owner_id = node_map[parent_id]["parent_node_id"]
                    owner_field = node_map[parent_id]["parent_field"]
                    if (
                        owner_id is not None
                        and owner_field == "body[0]"
                        and node_map[owner_id]["node_type"]
                        in {
                            "Module",
                            "ClassDef",
                            "FunctionDef",
                            "AsyncFunctionDef",
                        }
                    ):
                        owner_type = node_map[owner_id]["node_type"]
                        return (
                            "module_docstring"
                            if owner_type == "Module"
                            else "class_docstring"
                            if owner_type == "ClassDef"
                            else "function_docstring",
                            True,
                        )
                while parent_id is not None:
                    parent_type = node_map[parent_id]["node_type"]
                    if parent_type == "AnnAssign":
                        return (
                            "default_or_annotation"
                            if parent_field == "annotation"
                            else "assignment_value"
                            if parent_field == "value"
                            else "other_lexical_context",
                            False,
                        )
                    if (
                        parent_type in {"FunctionDef", "AsyncFunctionDef"}
                        and parent_field == "returns"
                    ):
                        return "default_or_annotation", False
                    if parent_type == "Call":
                        return (
                            "call_keyword_argument"
                            if str(parent_field).startswith("keywords")
                            else "call_positional_argument",
                            False,
                        )
                    if parent_type in {
                        "Assign", "AugAssign", "NamedExpr"
                    }:
                        return "assignment_value", False
                    if parent_type in {"arguments", "arg"}:
                        return "default_or_annotation", False
                    if parent_type == "Expr":
                        return "ordinary_expression", False
                    current_id = parent_id
                    parent_field = node_map[current_id]["parent_field"]
                    parent_id = node_map[current_id]["parent_node_id"]
                return "other_lexical_context", False

            for row in observation["expressions"]:
                if (
                    row["enclosing_lexical_scope"]
                    != lexical_scope(row["node_id"])
                    or row["lexical_role"]
                    != lexical_role(row["node_id"])[0]
                ):
                    return False

            allowed_roles = set(
                contract["frozen_extraction"][
                    "string_literal_role_classes"
                ]
            )
            for row in observation["string_literals"]:
                value = row.get("value")
                expected_role, expected_docstring = lexical_role(
                    row["node_id"]
                )
                if (
                    not isinstance(value, str)
                    or node_map[row["node_id"]]["node_type"] != "Constant"
                    or node_fields(row["node_id"]).get("value") != value
                    or hashlib.sha256(value.encode("utf-8")).hexdigest()
                    != row.get("value_sha256")
                    or row.get("node_id") not in expression_ids
                    or row.get("lexical_role") not in allowed_roles
                    or row.get("lexical_role") != expected_role
                    or row.get("is_docstring") is not expected_docstring
                    or row.get("enclosing_lexical_scope")
                    != lexical_scope(row["node_id"])
                    or len(value.encode("utf-8"))
                    > int(
                        contract["frozen_extraction"][
                            "maximum_single_string_utf8_bytes"
                        ]
                    )
                    or row.get("url_scheme_class")
                    != (
                        "HTTPS"
                        if value.lower().startswith("https://")
                        else "HTTP"
                        if value.lower().startswith("http://")
                        else "NONE"
                    )
                    or row.get("archive_suffix_class")
                    != (
                        "ZIP"
                        if value.lower().endswith(".zip")
                        else "NONE"
                    )
                ):
                    return False
            for key in ("call_sites", "assignments", "expressions"):
                for row in observation[key]:
                    segment = row.get("source_segment")
                    if (
                        not isinstance(segment, str)
                        or hashlib.sha256(
                            segment.encode("utf-8")
                        ).hexdigest()
                        != row.get("source_segment_sha256")
                        or len(segment.encode("utf-8"))
                        > int(
                            contract["frozen_extraction"][
                                "maximum_single_source_segment_utf8_bytes"
                            ]
                        )
                    ):
                        return False
            for row in observation["call_sites"]:
                fields = node_fields(row["node_id"])
                keyword_node_ids = child_ref_list(fields.get("keywords"))
                expected_keywords: list[dict[str, Any]] = []
                if keyword_node_ids is not None:
                    for keyword_id in keyword_node_ids:
                        if node_map[keyword_id]["node_type"] != "keyword":
                            return False
                        keyword_fields = node_fields(keyword_id)
                        value_id = child_ref(keyword_fields.get("value"))
                        if value_id is None:
                            return False
                        expected_keywords.append(
                            {
                                "name": keyword_fields.get("arg"),
                                "node_id": value_id,
                            }
                        )
                if (
                    node_map[row["node_id"]]["node_type"] != "Call"
                    or row["callee_node_id"]
                    != child_ref(fields.get("func"))
                    or row["positional_argument_node_ids"]
                    != child_ref_list(fields.get("args"))
                    or keyword_node_ids is None
                    or row["keyword_argument_node_ids"]
                    != expected_keywords
                    or row["enclosing_function"]
                    != lexical_scope(row["node_id"])
                    or row["callee_node_id"] not in expression_map
                    or row["callee_syntax"]
                    != expression_map[row["callee_node_id"]][
                        "source_segment"
                    ]
                    or any(
                        node_id not in expression_map
                        for node_id in row["positional_argument_node_ids"]
                    )
                    or any(
                        not isinstance(item, dict)
                        or set(item) != {"name", "node_id"}
                        or item["node_id"] not in expression_map
                        for item in row["keyword_argument_node_ids"]
                    )
                ):
                    return False
            for row in observation["assignments"]:
                node_type = node_map[row["node_id"]]["node_type"]
                fields = node_fields(row["node_id"])
                expected_targets = (
                    child_ref_list(fields.get("targets"))
                    if node_type == "Assign"
                    else [child_ref(fields.get("target"))]
                )
                if (
                    node_type
                    not in {"Assign", "AnnAssign", "AugAssign", "NamedExpr"}
                    or expected_targets is None
                    or any(item is None for item in expected_targets)
                    or row["target_node_ids"] != expected_targets
                    or row["value_node_id"]
                    != child_ref(fields.get("value"))
                    or row["enclosing_function"]
                    != lexical_scope(row["node_id"])
                    or row["value_node_id"] not in expression_map
                    or row["value_syntax"]
                    != expression_map[row["value_node_id"]][
                        "source_segment"
                    ]
                    or any(
                        node_id not in expression_map
                        for node_id in row["target_node_ids"]
                    )
                    or row["target_syntax"]
                    != ",".join(
                        expression_map[node_id]["source_segment"]
                        for node_id in row["target_node_ids"]
                    )
                ):
                    return False
            expected_functions: list[dict[str, Any]] = []
            for node_id, node in node_map.items():
                if node["node_type"] not in {
                    "FunctionDef", "AsyncFunctionDef"
                }:
                    continue
                fields = node_fields(node_id)
                arguments_id = child_ref(fields.get("args"))
                if (
                    not isinstance(fields.get("name"), str)
                    or arguments_id is None
                    or node_map[arguments_id]["node_type"] != "arguments"
                ):
                    return False
                argument_fields = node_fields(arguments_id)
                argument_names: list[str] = []
                for field in ("posonlyargs", "args", "kwonlyargs"):
                    argument_ids = child_ref_list(argument_fields.get(field))
                    if argument_ids is None:
                        return False
                    for argument_id in argument_ids:
                        if node_map[argument_id]["node_type"] != "arg":
                            return False
                        argument_name = node_fields(argument_id).get("arg")
                        if not isinstance(argument_name, str):
                            return False
                        argument_names.append(argument_name)
                for field, prefix in (("vararg", "*"), ("kwarg", "**")):
                    argument_id = child_ref(argument_fields.get(field))
                    if argument_id is not None:
                        if node_map[argument_id]["node_type"] != "arg":
                            return False
                        argument_name = node_fields(argument_id).get("arg")
                        if not isinstance(argument_name, str):
                            return False
                        argument_names.append(prefix + argument_name)
                expected_functions.append(
                    {
                        "source_path": node["source_path"],
                        "node_id": node_id,
                        "qualified_name": lexical_scope(
                            node_id, include_self=True
                        ),
                        "lineno": node["lineno"],
                        "col_offset": node["col_offset"],
                        "end_lineno": node["end_lineno"],
                        "end_col_offset": node["end_col_offset"],
                        "argument_names": argument_names,
                    }
                )
            expected_functions.sort(key=record_key)
            if observation["functions"] != expected_functions:
                return False

            expected_imports: list[dict[str, Any]] = []
            for node_id, node in node_map.items():
                if node["node_type"] not in {"Import", "ImportFrom"}:
                    continue
                fields = node_fields(node_id)
                alias_ids = child_ref_list(fields.get("names"))
                if alias_ids is None:
                    return False
                for alias_id in alias_ids:
                    if node_map[alias_id]["node_type"] != "alias":
                        return False
                    alias_fields = node_fields(alias_id)
                    alias_name = alias_fields.get("name")
                    alias_asname = alias_fields.get("asname")
                    if (
                        not isinstance(alias_name, str)
                        or (
                            alias_asname is not None
                            and not isinstance(alias_asname, str)
                        )
                    ):
                        return False
                    is_from = node["node_type"] == "ImportFrom"
                    module = (
                        "." * int(fields.get("level", 0))
                        + str(fields.get("module") or "")
                        if is_from
                        else alias_name
                    )
                    expected_imports.append(
                        {
                            "source_path": node["source_path"],
                            "node_id": node_id,
                            "lineno": node["lineno"],
                            "end_lineno": node["end_lineno"],
                            "import_kind": "from" if is_from else "import",
                            "module": module,
                            "name": alias_name if is_from else None,
                            "asname": alias_asname,
                            "enclosing_lexical_scope": lexical_scope(node_id),
                        }
                    )
            expected_imports.sort(key=record_key)
            if observation["import_aliases"] != expected_imports:
                return False
            return (
                observation.get("url_literal_count")
                == sum(
                    row.get("url_scheme_class") != "NONE"
                    for row in observation["string_literals"]
                )
                and observation.get("zip_string_literal_count")
                == sum(
                    row.get("archive_suffix_class") == "ZIP"
                    for row in observation["string_literals"]
                )
                and observation.get("source_code_or_dynamic_import_executed")
                is False
                and observation.get(
                    "provider_control_flow_or_url_derivation_interpreted"
                )
                is False
                and observation.get("official_provider_url_template_established")
                is False
                and observation.get(
                    "exact_198_parent_archive_url_mapping_established"
                )
                is False
                and observation.get("dataset_host_request_made") is False
                and observation.get(
                    "unresolved_or_unreachable_blob_followup_made"
                )
                is False
                and observation.get("external_txt_or_config_read") is False
                and observation.get("p0a_unresolved_local_import_count")
                == len(closure["observation"]["unresolved_local_imports"])
                == int(
                    contract["source_boundary"][
                        "p0a_unresolved_local_import_count"
                    ]
                )
                and observation.get("p0a_python_tree_path_count")
                == closure["observation"]["python_tree_path_count"]
                == int(
                    contract["source_boundary"][
                        "p0a_python_tree_path_count"
                    ]
                )
                and observation.get(
                    "p0a_static_import_closure_blob_count"
                )
                == closure["observation"]["closure_blob_count"]
                == int(contract["source_boundary"]["exact_blob_count"])
                and observation.get("ast_parse_pycf_only_ast_used") is True
                and observation.get("code_object_or_bytecode_compiled")
                is False
                and observation.get("dataset_zip_or_payload_read") is False
                and len(canonical_json_bytes(evidence))
                <= int(
                    contract["frozen_extraction"][
                        "maximum_total_evidence_json_bytes"
                    ]
                )
                and p0a_fetch_head()
                == contract["source_boundary"]["toolkit_commit"]
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            subprocess.SubprocessError,
        ):
            return False
    if FILENAMES["failure"] in names and FILENAMES["result"] not in names:
        try:
            contract = load_json(contract_path)
            failure = load_json(root / FILENAMES["failure"])
            if (
                contract.get("schema") != CONTRACT_SCHEMA
                or set(failure) != FAILURE_FIELDS
                or failure.get("schema") != FAILURE_SCHEMA
                or failure.get("terminal") != EVIDENCE_INVALID
                or failure.get("resume_or_rerun_authorized") is not False
                or failure.get("dataset_host_request_made") is not False
                or failure.get("execution_contract_sha256")
                != sha256(contract_path)
                or set(failure.get("observed_top_level_names", []))
                != names - {FILENAMES["failure"]}
            ):
                return False
            for key in ("attempt", "preflight", "evidence"):
                path = root / FILENAMES[key]
                recorded = failure.get(f"{key}_sha256")
                if path.is_file() and recorded != sha256(path):
                    return False
                if not path.exists() and recorded is not None:
                    return False
            attempt_path = root / FILENAMES["attempt"]
            if attempt_path.is_file():
                attempt = load_json(attempt_path)
                if (
                    attempt.get("schema") != ATTEMPT_SCHEMA
                    or attempt.get("status")
                    != "ATTEMPT_FSYNCED_BEFORE_FIRST_P0B_SOURCE_BLOB_READ"
                    or attempt.get("execution_contract_sha256")
                    != sha256(contract_path)
                    or attempt.get("p0a_closure_sha256")
                    != contract["source_boundary"]["p0a_closure_artifact"][
                        "sha256"
                    ]
                    or attempt.get("toolkit_commit")
                    != contract["source_boundary"]["toolkit_commit"]
                    or attempt.get("new_git_fetch_or_network_authorized")
                    is not False
                    or attempt.get("dataset_host_request_authorized")
                    is not False
                ):
                    return False
            preflight_path = root / FILENAMES["preflight"]
            if preflight_path.is_file():
                preflight = load_json(preflight_path)
                if (
                    not attempt_path.is_file()
                    or preflight.get("schema") != PREFLIGHT_SCHEMA
                    or preflight.get("status")
                    != (
                        "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_P0B_"
                        "SOURCE_BLOB_READ"
                    )
                    or preflight.get("attempt_sha256")
                    != sha256(attempt_path)
                    or preflight.get("p0a_fetch_head")
                    != contract["source_boundary"]["toolkit_commit"]
                    or preflight.get("runtime_lock")
                    != contract["runtime_lock"]
                    or preflight.get("p0a_terminal_validated") is not True
                    or preflight.get("exact_blob_count")
                    != contract["source_boundary"]["exact_blob_count"]
                    or preflight.get("exact_total_source_bytes")
                    != contract["source_boundary"][
                        "exact_total_source_bytes"
                    ]
                    or preflight.get("new_git_fetch_or_network_made")
                    is not False
                    or preflight.get("dataset_host_request_made") is not False
                ):
                    return False
            return (
                failure.get("p0a_fetch_head") == p0a_fetch_head()
                == contract["source_boundary"]["toolkit_commit"]
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            subprocess.SubprocessError,
        ):
            return False
    return False


def freeze_existing_partial(root: Path, names: set[str]) -> int:
    if FILENAMES["failure"] in names:
        raise ValueError("Existing P0B failure is corrupt or ambiguous")
    write_json_exclusive_fsync(
        root / FILENAMES["failure"],
        {
            "schema": FAILURE_SCHEMA,
            "terminal": EVIDENCE_INVALID,
            "reason": "existing_partial_or_unknown_p0b_root",
            "observed_top_level_names": sorted(names),
            "execution_contract_sha256": sha256(
                repo_root() / CONTRACT_RELATIVE_PATH
            ),
            "attempt_sha256": (
                sha256(root / FILENAMES["attempt"])
                if (root / FILENAMES["attempt"]).is_file()
                else None
            ),
            "preflight_sha256": (
                sha256(root / FILENAMES["preflight"])
                if (root / FILENAMES["preflight"]).is_file()
                else None
            ),
            "evidence_sha256": (
                sha256(root / FILENAMES["evidence"])
                if (root / FILENAMES["evidence"]).is_file()
                else None
            ),
            "p0a_fetch_head": p0a_fetch_head(),
            "resume_or_rerun_authorized": False,
            "dataset_host_request_made": False,
        },
    )
    return 2


def execute_with_failure_closure(
    contract_path: Path,
    root: Path,
    *,
    git_runner: GitRunner = subprocess_git_runner,
    verify_git: bool = True,
) -> dict[str, Any]:
    try:
        return execute(
            contract_path,
            root,
            git_runner=git_runner,
            verify_git=verify_git,
        )
    except BaseException as error:
        if root.exists() and FILENAMES["failure"] not in artifact_state(root):
            try:
                write_json_exclusive_fsync(
                    root / FILENAMES["failure"],
                    {
                        "schema": FAILURE_SCHEMA,
                        "terminal": EVIDENCE_INVALID,
                        "reason": f"{type(error).__name__}: {error}",
                        "observed_top_level_names": sorted(
                            artifact_state(root)
                        ),
                        "execution_contract_sha256": sha256(contract_path),
                        "attempt_sha256": (
                            sha256(root / FILENAMES["attempt"])
                            if (root / FILENAMES["attempt"]).is_file()
                            else None
                        ),
                        "preflight_sha256": (
                            sha256(root / FILENAMES["preflight"])
                            if (root / FILENAMES["preflight"]).is_file()
                            else None
                        ),
                        "evidence_sha256": (
                            sha256(root / FILENAMES["evidence"])
                            if (root / FILENAMES["evidence"]).is_file()
                            else None
                        ),
                        "p0a_fetch_head": p0a_fetch_head(),
                        "resume_or_rerun_authorized": False,
                        "dataset_host_request_made": False,
                    },
                )
            except BaseException:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-contract",
        type=Path,
        default=repo_root() / CONTRACT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / CANONICAL_ROOT,
    )
    args = parser.parse_args()
    root = require_canonical_root(args.output_root)
    names = artifact_state(root)
    if names:
        if validate_existing_terminal(root, names):
            raise ValueError("P0B validated terminal already exists")
        return freeze_existing_partial(root, names)
    result = execute_with_failure_closure(
        args.execution_contract.resolve(), root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
