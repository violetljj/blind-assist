#!/usr/bin/env python3
"""Fail-closed validators for HFTF D5-S0B P0B.1 durable terminals."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any

from plan_stage_c_d5_s0b_p0b1_sharded_semantic_evidence import (
    AGGREGATE_SHARD_MAXIMUM_BYTES,
    ATTEMPT_FILENAME,
    ATTEMPT_SCHEMA,
    CAP_MANIFEST,
    CAP_MANIFEST_SHA256,
    CONTROL_ARTIFACT_MAXIMUM_BYTES,
    DESIGN_SHA256,
    EVIDENCE_INVALID,
    EVIDENCE_LOCKED,
    EVIDENCE_NOT_EVALUABLE,
    FAILURE_FILENAME,
    FAILURE_SCHEMA,
    INDEX_FILENAME,
    INDEX_SCHEMA,
    NOT_EVALUABLE_FILENAME,
    NOT_EVALUABLE_SCHEMA,
    PREFLIGHT_FILENAME,
    PREFLIGHT_SCHEMA,
    RESULT_FILENAME,
    RESULT_SCHEMA,
    SHARD_COUNT,
    SHARD_SCHEMA,
    canonical_json_bytes,
    canonical_object_sha256,
    failure_allowed_set,
    locked_closed_set,
    not_evaluable_closed_set,
    reject_duplicate_object_pairs,
    shard_filename,
    shard_filenames,
)
from plan_stage_c_d5_s0b_p0b_provider_semantic_evidence import (
    canonical_preorder_node_ids,
    node_receipt_id,
)


HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class TerminalValidationError(ValueError):
    """A P0B.1 artifact root does not validate as its claimed terminal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalValidationError(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_keys(
    value: Any,
    expected: list[str] | set[str],
    label: str,
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} is not an object")
    require(set(value) == set(expected), f"{label} exact-key mismatch")
    return value


def load_json_strict(path: Path, maximum_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    require(size <= maximum_bytes, f"{path.name} exceeds byte cap")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_object_pairs,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise TerminalValidationError(
            f"{path.name} is not strict JSON: {error}"
        ) from error
    require(
        raw == canonical_json_bytes(value),
        f"{path.name} is not canonical JSON",
    )
    require(isinstance(value, dict), f"{path.name} must be an object")
    return value


def canonical_ast_parts(
    value: Any,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    exact_keys(value, {"node_type", "fields"}, "canonical_ast_object")
    node_type = value["node_type"]
    fields = value["fields"]
    require(isinstance(node_type, str), "AST node type is not text")
    require(isinstance(fields, dict), "AST fields is not an object")
    node_class = getattr(ast, node_type, None)
    require(
        isinstance(node_class, type)
        and issubclass(node_class, ast.AST),
        "AST node type is invalid",
    )
    require(
        set(fields) == set(node_class._fields),
        "AST runtime field set mismatch",
    )

    def scalar_valid(item: Any) -> bool:
        if item is None or isinstance(item, (bool, int, float, str)):
            return True
        if not isinstance(item, dict):
            return False
        if item == {"scalar_type": "ellipsis"}:
            return True
        if (
            set(item) == {"scalar_type", "hex"}
            and item.get("scalar_type") == "bytes"
            and isinstance(item.get("hex"), str)
            and re.fullmatch(r"(?:[0-9a-f]{2})*", item["hex"])
        ):
            return True
        return (
            set(item) == {
                "scalar_type",
                "real_repr",
                "imag_repr",
            }
            and item.get("scalar_type") == "complex"
            and isinstance(item.get("real_repr"), str)
            and isinstance(item.get("imag_repr"), str)
        )

    edges: dict[str, str] = {}
    for field, item in fields.items():
        if isinstance(item, dict) and set(item) == {"node_id"}:
            require(
                bool(HEX64_RE.fullmatch(str(item["node_id"]))),
                "AST child node id invalid",
            )
            edges[field] = item["node_id"]
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if (
                    isinstance(child, dict)
                    and set(child) == {"node_id"}
                ):
                    require(
                        bool(
                            HEX64_RE.fullmatch(
                                str(child["node_id"])
                            )
                        ),
                        "AST list child node id invalid",
                    )
                    edges[f"{field}[{index}]"] = child["node_id"]
                else:
                    require(
                        scalar_valid(child),
                        "AST list scalar invalid",
                    )
        else:
            require(scalar_valid(item), "AST scalar invalid")
    return node_type, fields, edges


def validate_object_receipt(
    receipt: Any,
    row: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    receipt = exact_keys(
        receipt,
        schema["object_receipt_exact_keys"],
        "object_receipt",
    )
    require(
        receipt["path"] == row["path"]
        and receipt["expected_git_blob_oid"] == row["git_blob_oid"]
        and receipt["actual_commit_path_oid"] == row["git_blob_oid"]
        and receipt["git_object_type"] == "blob"
        and receipt["expected_bytes"] == row["bytes"]
        and receipt["actual_object_size_bytes"] == row["bytes"]
        and receipt["actual_content_bytes"] == row["bytes"]
        and receipt["expected_sha256"] == row["sha256"]
        and receipt["actual_content_sha256"] == row["sha256"],
        "object receipt does not bind the P0A row",
    )


def node_kind(node_type: str, base: type[ast.AST]) -> bool:
    node_class = getattr(ast, node_type, None)
    return (
        isinstance(node_class, type)
        and issubclass(node_class, base)
    )


def validate_record_order(
    records: list[dict[str, Any]],
    preorder: dict[str, int],
    label: str,
) -> None:
    require(
        records
        == sorted(
            records,
            key=lambda row: (
                preorder[row["node_id"]],
                canonical_json_bytes(row),
            ),
        ),
        f"{label} record order mismatch",
    )


def validate_shard(
    shard: dict[str, Any],
    contract: dict[str, Any],
    closure_row: dict[str, Any],
    manifest_index: int,
) -> dict[str, Any]:
    schema = contract["exact_artifact_schemas"]
    limits = contract["unchanged_extraction_limits"]
    exact_keys(shard, schema["shard_json_exact_keys"], "shard")
    require(
        shard["schema"] == SHARD_SCHEMA
        and shard["status"] == "AST_SEMANTIC_SHARD_LOCKED"
        and shard["manifest_index"] == manifest_index
        and shard["source_path"] == closure_row["path"]
        and shard["toolkit_commit"]
        == contract["source_authority"]["toolkit_commit"]
        and shard["git_blob_oid"] == closure_row["git_blob_oid"]
        and shard["source_blob_bytes"] == closure_row["bytes"]
        and shard["source_blob_sha256"] == closure_row["sha256"]
        and isinstance(shard["detected_source_encoding"], str)
        and bool(shard["detected_source_encoding"])
        and shard["parse_status"] == "PARSED"
        and shard["runtime_lock_sha256"]
        == canonical_object_sha256(contract["runtime_lock"])
        and shard["algorithm_lock_sha256"]
        == canonical_object_sha256(contract["algorithm_lock"]),
        "shard identity mismatch",
    )
    require(
        CAP_MANIFEST[manifest_index]["path"] == closure_row["path"]
        and CAP_MANIFEST[manifest_index]["p0a_blob_bytes"]
        == closure_row["bytes"],
        "cap manifest source binding mismatch",
    )
    validate_object_receipt(
        shard["object_receipt"], closure_row, schema
    )

    nodes = shard["node_receipts"]
    require(isinstance(nodes, list) and bool(nodes), "node list invalid")
    require(
        len(nodes) <= limits["maximum_ast_nodes_global"],
        "shard node count exceeds global cap",
    )
    node_map: dict[str, dict[str, Any]] = {}
    dump_map: dict[
        str, tuple[str, dict[str, Any], dict[str, str]]
    ] = {}
    for index, node in enumerate(nodes):
        exact_keys(node, schema["node_receipt_exact_keys"], "node")
        require(
            node["source_path"] == closure_row["path"]
            and node["preorder_index"] == index
            and is_int(node["depth"])
            and 0 <= node["depth"]
            <= limits["maximum_ast_depth_per_blob"],
            "node identity/order/depth mismatch",
        )
        for key in (
            "lineno",
            "col_offset",
            "end_lineno",
            "end_col_offset",
        ):
            require(
                node[key] is None or is_int(node[key]),
                "node source span type invalid",
            )
        node_id = node["node_id"]
        require(
            isinstance(node_id, str)
            and bool(HEX64_RE.fullmatch(node_id))
            and node_id not in node_map,
            "node id invalid or duplicate",
        )
        parts = canonical_ast_parts(node["canonical_ast_object"])
        adapted = dict(node)
        adapted["canonical_ast_dump"] = (
            canonical_json_bytes(node["canonical_ast_object"])
            .decode("utf-8")
            .rstrip("\n")
        )
        require(
            node_receipt_id(adapted) == node_id,
            "node id recomputation mismatch",
        )
        node_map[node_id] = node
        dump_map[node_id] = parts

    root = nodes[0]
    require(
        root["node_type"] == "Module"
        and root["parent_node_id"] is None
        and root["parent_field"] is None
        and root["depth"] == 0,
        "module root invariant failed",
    )
    for node in nodes[1:]:
        parent_id = node["parent_node_id"]
        parent_field = node["parent_field"]
        require(
            parent_id in node_map
            and isinstance(parent_field, str)
            and node_map[parent_id]["preorder_index"]
            < node["preorder_index"]
            and node_map[parent_id]["depth"] + 1 == node["depth"]
            and dump_map[parent_id][2].get(parent_field)
            == node["node_id"],
            "node parent/edge invariant failed",
        )
    for parent_id, (_, _, edges) in dump_map.items():
        for field, child_id in edges.items():
            require(
                child_id in node_map
                and node_map[child_id]["parent_node_id"] == parent_id
                and node_map[child_id]["parent_field"] == field,
                "AST edge/parent bidirectional invariant failed",
            )
    try:
        reconstructed = canonical_preorder_node_ids(
            root["node_id"], node_map, dump_map
        )
    except (KeyError, ValueError) as error:
        raise TerminalValidationError(
            f"canonical AST traversal invalid: {error}"
        ) from error
    require(
        reconstructed == [node["node_id"] for node in nodes],
        "canonical preorder mismatch",
    )
    require(
        shard["ast_node_count"] == len(nodes)
        and shard["maximum_ast_depth"]
        == max(node["depth"] for node in nodes),
        "shard node count/depth mismatch",
    )
    preorder = {
        node["node_id"]: node["preorder_index"] for node in nodes
    }

    record_specs = (
        (
            "expression_records",
            "expression_record_exact_keys",
            "expressions",
            limits["maximum_expression_records_global"],
        ),
        (
            "string_literal_records",
            "string_literal_record_exact_keys",
            "strings",
            limits["maximum_string_literal_records_global"],
        ),
        (
            "call_site_records",
            "call_site_record_exact_keys",
            "calls",
            limits["maximum_call_site_records_global"],
        ),
        (
            "assignment_records",
            "assignment_record_exact_keys",
            "assignments",
            limits["maximum_assignment_records_global"],
        ),
        (
            "function_records",
            "function_record_exact_keys",
            "functions",
            limits["maximum_function_records_global"],
        ),
        (
            "import_alias_records",
            "import_alias_record_exact_keys",
            "imports",
            limits["maximum_import_alias_records_global"],
        ),
    )
    for array_key, schema_key, _, cap in record_specs:
        records = shard[array_key]
        require(
            isinstance(records, list) and len(records) <= cap,
            f"{array_key} invalid or over cap",
        )
        for record in records:
            exact_keys(record, schema[schema_key], array_key)
            require(
                record["source_path"] == closure_row["path"]
                and record["node_id"] in node_map,
                f"{array_key} node reference invalid",
            )
        validate_record_order(records, preorder, array_key)

    expressions = shard["expression_records"]
    expected_expression_ids = {
        node_id
        for node_id, node in node_map.items()
        if node_kind(node["node_type"], ast.expr)
    }
    require(
        len(expressions) == len(expected_expression_ids)
        and {row["node_id"] for row in expressions}
        == expected_expression_ids,
        "expression one-to-one coverage mismatch",
    )
    for record in expressions:
        node = node_map[record["node_id"]]
        require(
            record["node_type"] == node["node_type"]
            and record["parent_node_id"] == node["parent_node_id"]
            and record["parent_field"] == node["parent_field"]
            and all(
                record[key] == node[key]
                for key in (
                    "lineno",
                    "col_offset",
                    "end_lineno",
                    "end_col_offset",
                )
            )
            and bool(
                HEX64_RE.fullmatch(record["source_segment_sha256"])
            )
            and is_int(record["source_segment_utf8_bytes"])
            and 0 <= record["source_segment_utf8_bytes"]
            <= limits["maximum_single_source_segment_utf8_bytes"]
            and record["source_encoding"]
            == shard["detected_source_encoding"]
            and record["lexical_role"]
            in limits["string_literal_role_classes"],
            "expression semantic receipt mismatch",
        )

    strings = shard["string_literal_records"]
    expected_strings = {
        node_id: dump_map[node_id][1]["value"]
        for node_id, node in node_map.items()
        if node["node_type"] == "Constant"
        and isinstance(dump_map[node_id][1].get("value"), str)
    }
    require(
        len(strings) == len(expected_strings)
        and {row["node_id"] for row in strings}
        == set(expected_strings),
        "string one-to-one coverage mismatch",
    )
    for record in strings:
        value = record["value"]
        require(
            isinstance(value, str)
            and value == expected_strings[record["node_id"]]
            and record["value_sha256"]
            == hashlib.sha256(value.encode("utf-8")).hexdigest()
            and record["value_utf8_bytes"] == len(value.encode("utf-8"))
            and record["value_utf8_bytes"]
            <= limits["maximum_single_string_utf8_bytes"]
            and record["url_scheme_class"]
            == (
                "HTTPS"
                if value.lower().startswith("https://")
                else "HTTP"
                if value.lower().startswith("http://")
                else "NONE"
            )
            and record["archive_suffix_class"]
            == ("ZIP" if value.lower().endswith(".zip") else "NONE")
            and record["lexical_role"]
            in limits["string_literal_role_classes"]
            and isinstance(record["is_docstring"], bool),
            "string literal receipt mismatch",
        )

    calls = shard["call_site_records"]
    expected_call_ids = {
        node_id
        for node_id, node in node_map.items()
        if node["node_type"] == "Call"
    }
    require(
        len(calls) == len(expected_call_ids)
        and {row["node_id"] for row in calls} == expected_call_ids,
        "call one-to-one coverage mismatch",
    )
    for record in calls:
        fields = dump_map[record["node_id"]][1]
        callee_id = fields["func"]["node_id"]
        positional = [
            item["node_id"] for item in fields["args"]
        ]
        keyword_values = []
        for keyword_edge in fields["keywords"]:
            keyword_id = keyword_edge["node_id"]
            keyword = dump_map[keyword_id][1]
            keyword_values.append(
                {
                    "name": keyword["arg"],
                    "node_id": keyword["value"]["node_id"],
                }
            )
        for item in record["keyword_argument_node_ids"]:
            exact_keys(
                item,
                schema["keyword_argument_item_exact_keys"],
                "keyword argument",
            )
        require(
            record["callee_node_id"] == callee_id
            and record["positional_argument_node_ids"] == positional
            and record["keyword_argument_node_ids"] == keyword_values
            and all(
                node_id in node_map
                for node_id in [
                    record["callee_node_id"],
                    *record["positional_argument_node_ids"],
                    *[
                        item["node_id"]
                        for item in record[
                            "keyword_argument_node_ids"
                        ]
                    ],
                ]
            ),
            "call node references mismatch",
        )
        for text_key in ("callee_syntax", "source_segment"):
            value = record[text_key]
            require(
                isinstance(value, str)
                and record[f"{text_key}_sha256"]
                == hashlib.sha256(value.encode("utf-8")).hexdigest()
                and record[f"{text_key}_utf8_bytes"]
                == len(value.encode("utf-8"))
                and record[f"{text_key}_utf8_bytes"]
                <= limits[
                    "maximum_single_source_segment_utf8_bytes"
                ],
                f"call {text_key} hash/length mismatch",
            )

    assignments = shard["assignment_records"]
    expected_assignments: dict[str, tuple[list[str], str]] = {}
    for node_id, node in node_map.items():
        node_type = node["node_type"]
        fields = dump_map[node_id][1]
        if node_type in {"Assign", "AugAssign", "NamedExpr"} or (
            node_type == "AnnAssign" and fields.get("value") is not None
        ):
            targets = (
                [item["node_id"] for item in fields["targets"]]
                if node_type == "Assign"
                else [fields["target"]["node_id"]]
            )
            expected_assignments[node_id] = (
                targets,
                fields["value"]["node_id"],
            )
    require(
        len(assignments) == len(expected_assignments)
        and {row["node_id"] for row in assignments}
        == set(expected_assignments),
        "assignment one-to-one coverage mismatch",
    )
    for record in assignments:
        targets, value_id = expected_assignments[record["node_id"]]
        require(
            record["target_node_ids"] == targets
            and record["value_node_id"] == value_id
            and all(node_id in node_map for node_id in targets)
            and value_id in node_map,
            "assignment node references mismatch",
        )
        for text_key in (
            "target_syntax",
            "value_syntax",
            "source_segment",
        ):
            value = record[text_key]
            require(
                isinstance(value, str)
                and record[f"{text_key}_sha256"]
                == hashlib.sha256(value.encode("utf-8")).hexdigest()
                and record[f"{text_key}_utf8_bytes"]
                == len(value.encode("utf-8"))
                and record[f"{text_key}_utf8_bytes"]
                <= limits[
                    "maximum_single_source_segment_utf8_bytes"
                ],
                f"assignment {text_key} hash/length mismatch",
            )

    functions = shard["function_records"]
    expected_function_ids = {
        node_id
        for node_id, node in node_map.items()
        if node["node_type"] in {"FunctionDef", "AsyncFunctionDef"}
    }
    require(
        len(functions) == len(expected_function_ids)
        and {row["node_id"] for row in functions}
        == expected_function_ids,
        "function one-to-one coverage mismatch",
    )

    imports = shard["import_alias_records"]
    expected_import_counts = {
        node_id: len(dump_map[node_id][1]["names"])
        for node_id, node in node_map.items()
        if node["node_type"] in {"Import", "ImportFrom"}
    }
    actual_import_counts: dict[str, int] = {}
    for record in imports:
        actual_import_counts[record["node_id"]] = (
            actual_import_counts.get(record["node_id"], 0) + 1
        )
    require(
        actual_import_counts == expected_import_counts,
        "import-alias occurrence coverage mismatch",
    )

    counts = exact_keys(
        shard["record_counts"],
        schema["record_counts_exact_keys"],
        "record_counts",
    )
    expected_counts = {
        "ast_nodes": len(nodes),
        "expressions": len(expressions),
        "strings": len(strings),
        "calls": len(calls),
        "assignments": len(assignments),
        "functions": len(functions),
        "imports": len(imports),
    }
    require(counts == expected_counts, "shard record counts mismatch")
    return expected_counts


def validate_attempt_preflight(
    root: Path,
    contract: dict[str, Any],
    contract_path: Path,
    closure: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = contract["exact_artifact_schemas"]
    attempt = load_json_strict(
        root / ATTEMPT_FILENAME, CONTROL_ARTIFACT_MAXIMUM_BYTES
    )
    preflight = load_json_strict(
        root / PREFLIGHT_FILENAME, CONTROL_ARTIFACT_MAXIMUM_BYTES
    )
    exact_keys(
        attempt, schema["attempt_json_exact_keys"], "attempt"
    )
    exact_keys(
        preflight, schema["preflight_json_exact_keys"], "preflight"
    )
    rows = closure["observation"]["closure_rows"]
    require(
        attempt["schema"] == ATTEMPT_SCHEMA
        and attempt["status"]
        == "ATTEMPT_FSYNCED_BEFORE_FIRST_SOURCE_BLOB_READ"
        and attempt["execution_contract_sha256"]
        == sha256_path(contract_path)
        and attempt["p0b1_design_sha256"] == DESIGN_SHA256
        and attempt["p0b_invalid_result_sha256"]
        == contract["parents"]["p0b_invalid_result"]["sha256"]
        and attempt["p0a_closure_sha256"]
        == contract["source_authority"]["p0a_closure_artifact"][
            "sha256"
        ]
        and attempt["toolkit_commit"]
        == contract["source_authority"]["toolkit_commit"]
        and attempt["new_git_fetch_or_network_authorized"] is False
        and attempt["dataset_host_request_authorized"] is False
        and attempt["old_p0b_root_reopen_authorized"] is False,
        "attempt binding mismatch",
    )
    require(
        preflight["schema"] == PREFLIGHT_SCHEMA
        and preflight["status"]
        == "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_SOURCE_BLOB_READ"
        and preflight["attempt_sha256"]
        == sha256_path(root / ATTEMPT_FILENAME)
        and isinstance(preflight["execution_commit"], str)
        and bool(preflight["execution_commit"])
        and preflight["head_equal_origin_master"] is True
        and preflight["tracked_clean"] is True
        and preflight["new_canonical_root_absent_before_attempt"] is True
        and preflight["p0a_terminal_validated"] is True
        and preflight["p0b_invalid_terminal_validated"] is True
        and preflight["exact_blob_count"] == len(rows)
        and preflight["exact_total_source_bytes"]
        == sum(row["bytes"] for row in rows)
        and preflight["exact_ordered_row_manifest_sha256"]
        == contract["source_authority"][
            "exact_ordered_row_manifest_sha256"
        ]
        and preflight["runtime_lock"] == contract["runtime_lock"]
        and preflight["runtime_lock_sha256"]
        == canonical_object_sha256(contract["runtime_lock"])
        and preflight["algorithm_lock_sha256"]
        == canonical_object_sha256(contract["algorithm_lock"])
        and preflight["new_git_fetch_or_network_made"] is False
        and preflight["dataset_host_request_made"] is False,
        "preflight binding mismatch",
    )
    return attempt, preflight


def claim_ceiling_is_false(value: Any, contract: dict[str, Any]) -> bool:
    keys = contract["exact_artifact_schemas"][
        "claim_ceiling_exact_keys"
    ]
    return (
        isinstance(value, dict)
        and set(value) == set(keys)
        and all(value[key] is False for key in keys)
    )


def validate_locked_terminal(
    root: Path,
    contract: dict[str, Any],
    contract_path: Path,
    closure: dict[str, Any],
) -> dict[str, Any]:
    require(
        root_names(root) == locked_closed_set(),
        "locked closed set mismatch",
    )
    validate_attempt_preflight(root, contract, contract_path, closure)
    rows = closure["observation"]["closure_rows"]
    schema = contract["exact_artifact_schemas"]
    shards: list[dict[str, Any]] = []
    counts = {
        key: 0
        for key in (
            "ast_nodes",
            "expressions",
            "strings",
            "calls",
            "assignments",
            "functions",
            "imports",
        )
    }
    aggregate = 0
    for index, row in enumerate(rows):
        path = root / shard_filename(index)
        maximum = CAP_MANIFEST[index]["maximum_shard_bytes"]
        shard = load_json_strict(path, maximum)
        validate_shard(shard, contract, row, index)
        shards.append(shard)
        aggregate += path.stat().st_size
        for key, value in shard["record_counts"].items():
            counts[key] += value
    require(
        aggregate <= AGGREGATE_SHARD_MAXIMUM_BYTES,
        "aggregate shard cap exceeded",
    )
    for key, limit_key in (
        ("ast_nodes", "maximum_ast_nodes_global"),
        ("expressions", "maximum_expression_records_global"),
        ("strings", "maximum_string_literal_records_global"),
        ("calls", "maximum_call_site_records_global"),
        ("assignments", "maximum_assignment_records_global"),
        ("functions", "maximum_function_records_global"),
        ("imports", "maximum_import_alias_records_global"),
    ):
        require(
            counts[key]
            <= contract["unchanged_extraction_limits"][limit_key],
            f"global {key} cap exceeded",
        )
    index = load_json_strict(
        root / INDEX_FILENAME, CONTROL_ARTIFACT_MAXIMUM_BYTES
    )
    exact_keys(index, schema["index_json_exact_keys"], "index")
    expected_index_rows = []
    for shard in shards:
        path = root / shard_filename(shard["manifest_index"])
        expected_index_rows.append(
            {
                "manifest_index": shard["manifest_index"],
                "source_path": shard["source_path"],
                "shard_filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "record_counts": shard["record_counts"],
                "maximum_ast_depth": shard["maximum_ast_depth"],
                "maximum_shard_bytes": CAP_MANIFEST[
                    shard["manifest_index"]
                ]["maximum_shard_bytes"],
            }
        )
    for row in index["shards"]:
        exact_keys(row, schema["index_row_exact_keys"], "index row")
    require(
        index["schema"] == INDEX_SCHEMA
        and index["status"] == "SHARDED_AST_SEMANTIC_EVIDENCE_LOCKED"
        and index["attempt_sha256"]
        == sha256_path(root / ATTEMPT_FILENAME)
        and index["preflight_sha256"]
        == sha256_path(root / PREFLIGHT_FILENAME)
        and index["toolkit_commit"]
        == contract["source_authority"]["toolkit_commit"]
        and index["p0a_manifest_sha256"]
        == contract["source_authority"][
            "exact_ordered_row_manifest_sha256"
        ]
        and index["object_receipt_set_sha256"]
        == hashlib.sha256(
            canonical_json_bytes(
                [shard["object_receipt"] for shard in shards]
            )
        ).hexdigest()
        and index["runtime_lock_sha256"]
        == canonical_object_sha256(contract["runtime_lock"])
        and index["algorithm_lock_sha256"]
        == canonical_object_sha256(contract["algorithm_lock"])
        and index["cap_manifest_sha256"] == CAP_MANIFEST_SHA256
        and index["shards"] == expected_index_rows
        and index["aggregate_shard_bytes"] == aggregate
        and index["global_record_counts"] == counts
        and index["global_maximum_ast_depth"]
        == max(shard["maximum_ast_depth"] for shard in shards),
        "index recomputation mismatch",
    )
    result = load_json_strict(
        root / RESULT_FILENAME, CONTROL_ARTIFACT_MAXIMUM_BYTES
    )
    exact_keys(result, schema["result_json_exact_keys"], "result")
    require(
        result["schema"] == RESULT_SCHEMA
        and result["terminal"] == EVIDENCE_LOCKED
        and result["attempt_sha256"]
        == sha256_path(root / ATTEMPT_FILENAME)
        and result["preflight_sha256"]
        == sha256_path(root / PREFLIGHT_FILENAME)
        and result["index_sha256"] == sha256_path(root / INDEX_FILENAME)
        and result["not_evaluable_sha256"] is None
        and result["toolkit_commit"]
        == contract["source_authority"]["toolkit_commit"]
        and result["exact_blob_count"] == SHARD_COUNT
        and result["aggregate_shard_bytes"] == aggregate
        and result["global_record_counts"] == counts
        and result["provider_resolution_performed"] is False
        and result["dataset_host_request_made"] is False
        and result["source_evidence_role"]
        == "CONSUMED_SOURCE_RECOVERY_NOT_FRESH_VALIDATION"
        and result["next_authority"]
        == (
            "FREEZE_SEPARATE_HASH_BOUND_P0C_PROVIDER_"
            "RESOLUTION_CONTRACT"
        )
        and result["p0c_execution_authorized_automatically"] is False
        and claim_ceiling_is_false(result["claim_ceiling"], contract),
        "locked result binding mismatch",
    )
    return result


def validate_parse_receipt(
    receipt: Any,
    closure_row: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    exact_keys(
        receipt,
        contract["exact_artifact_schemas"][
            "parse_receipt_exact_keys"
        ],
        "parse receipt",
    )
    limits = contract["unchanged_extraction_limits"]
    require(
        receipt["path"] == closure_row["path"]
        and receipt["git_blob_oid"] == closure_row["git_blob_oid"]
        and isinstance(receipt["detected_source_encoding"], str)
        and bool(receipt["detected_source_encoding"])
        and receipt["ast_parse_status"] == "PARSED"
        and is_int(receipt["ast_node_count"])
        and 1 <= receipt["ast_node_count"]
        <= limits["maximum_ast_nodes_global"]
        and is_int(receipt["maximum_ast_depth"])
        and 0 <= receipt["maximum_ast_depth"]
        <= limits["maximum_ast_depth_per_blob"],
        "parse receipt mismatch",
    )


def validate_not_evaluable_terminal(
    root: Path,
    contract: dict[str, Any],
    contract_path: Path,
    closure: dict[str, Any],
) -> dict[str, Any]:
    require(
        root_names(root) == not_evaluable_closed_set(),
        "not-evaluable closed set mismatch",
    )
    validate_attempt_preflight(root, contract, contract_path, closure)
    schema = contract["exact_artifact_schemas"]
    evidence = load_json_strict(
        root / NOT_EVALUABLE_FILENAME,
        CONTROL_ARTIFACT_MAXIMUM_BYTES,
    )
    exact_keys(
        evidence,
        schema["not_evaluable_json_exact_keys"],
        "not-evaluable",
    )
    require(
        evidence["schema"] == NOT_EVALUABLE_SCHEMA
        and evidence["terminal"] == EVIDENCE_NOT_EVALUABLE
        and evidence["attempt_sha256"]
        == sha256_path(root / ATTEMPT_FILENAME)
        and evidence["preflight_sha256"]
        == sha256_path(root / PREFLIGHT_FILENAME)
        and evidence["toolkit_commit"]
        == contract["source_authority"]["toolkit_commit"]
        and evidence["new_git_fetch_or_network_made"] is False
        and evidence["dataset_host_request_made"] is False,
        "not-evaluable common binding mismatch",
    )
    rows = closure["observation"]["closure_rows"]
    reason = evidence["reason_class"]
    prefix_keys = (
        "prefix_ast_node_count",
        "prefix_maximum_ast_depth",
        "prefix_string_count",
        "prefix_call_count",
        "prefix_assignment_count",
        "prefix_function_count",
        "prefix_import_count",
        "prefix_expression_count",
    )
    if reason == "p0a_dynamic_import_evidence_nonzero":
        require(
            closure["observation"]["dynamic_import_call_count"] != 0
            or closure["observation"][
                "indirect_dynamic_import_or_exec_count"
            ]
            != 0,
            "dynamic reason lacks dynamic evidence",
        )
        require(
            evidence["source_blob_read_count"] == 0
            and evidence["object_receipt_count"] == 0
            and evidence["object_receipts"] == []
            and evidence["object_receipt_set_sha256"] is None
            and evidence["source_total_bytes"] == 0
            and evidence["manifest_index"] is None
            and evidence["source_path"] is None
            and evidence["detected_source_encoding"] is None
            and evidence["parse_status"] is None
            and evidence["error_type"] is None
            and evidence["parse_receipt_count"] == 0
            and evidence["parse_receipts"] == []
            and all(evidence[key] == 0 for key in prefix_keys),
            "dynamic not-evaluable zero invariant failed",
        )
    else:
        require(
            reason
            == (
                "verified_source_incompatible_with_frozen_encoding_"
                "or_ast_grammar"
            ),
            "not-evaluable reason invalid",
        )
        receipts = evidence["object_receipts"]
        require(
            evidence["source_blob_read_count"] == len(rows)
            and evidence["object_receipt_count"] == len(rows)
            and isinstance(receipts, list)
            and len(receipts) == len(rows)
            and evidence["source_total_bytes"]
            == sum(row["bytes"] for row in rows)
            and evidence["object_receipt_set_sha256"]
            == hashlib.sha256(
                canonical_json_bytes(receipts)
            ).hexdigest(),
            "syntax not-evaluable object receipt invariant failed",
        )
        for receipt, row in zip(receipts, rows, strict=True):
            validate_object_receipt(receipt, row, schema)
        manifest_index = evidence["manifest_index"]
        require(
            is_int(manifest_index)
            and 0 <= manifest_index < len(rows)
            and evidence["source_path"] == rows[manifest_index]["path"]
            and (
                evidence["detected_source_encoding"] is None
                or (
                    isinstance(
                        evidence["detected_source_encoding"], str
                    )
                    and bool(evidence["detected_source_encoding"])
                )
            )
            and evidence["error_type"]
            in {"SyntaxError", "UnicodeDecodeError", "UnicodeError"},
            "syntax not-evaluable failed identity invalid",
        )
        expected_parse_status = (
            "ENCODING_DETECTION_FAILED"
            if evidence["detected_source_encoding"] is None
            else "SOURCE_DECODE_FAILED"
            if evidence["error_type"]
            in {"UnicodeDecodeError", "UnicodeError"}
            else "AST_PARSE_FAILED"
        )
        require(
            evidence["parse_status"] == expected_parse_status,
            "syntax not-evaluable parse status invalid",
        )
        parse_receipts = evidence["parse_receipts"]
        require(
            isinstance(parse_receipts, list)
            and evidence["parse_receipt_count"] == len(parse_receipts)
            and len(parse_receipts) == manifest_index,
            "syntax not-evaluable parse prefix invalid",
        )
        for receipt, row in zip(
            parse_receipts, rows[:manifest_index], strict=True
        ):
            validate_parse_receipt(receipt, row, contract)
        require(
            evidence["prefix_ast_node_count"]
            == sum(row["ast_node_count"] for row in parse_receipts)
            and evidence["prefix_maximum_ast_depth"]
            == (
                max(
                    row["maximum_ast_depth"]
                    for row in parse_receipts
                )
                if parse_receipts
                else 0
            ),
            "syntax not-evaluable parse-prefix node metrics invalid",
        )
        for key, limit_key in (
            (
                "prefix_ast_node_count",
                "maximum_ast_nodes_global",
            ),
            (
                "prefix_string_count",
                "maximum_string_literal_records_global",
            ),
            (
                "prefix_call_count",
                "maximum_call_site_records_global",
            ),
            (
                "prefix_assignment_count",
                "maximum_assignment_records_global",
            ),
            (
                "prefix_function_count",
                "maximum_function_records_global",
            ),
            (
                "prefix_import_count",
                "maximum_import_alias_records_global",
            ),
            (
                "prefix_expression_count",
                "maximum_expression_records_global",
            ),
        ):
            require(
                is_int(evidence[key])
                and 0
                <= evidence[key]
                <= contract["unchanged_extraction_limits"][limit_key],
                f"{key} invalid",
            )
    result = load_json_strict(
        root / RESULT_FILENAME, CONTROL_ARTIFACT_MAXIMUM_BYTES
    )
    exact_keys(result, schema["result_json_exact_keys"], "result")
    require(
        result["schema"] == RESULT_SCHEMA
        and result["terminal"] == EVIDENCE_NOT_EVALUABLE
        and result["attempt_sha256"]
        == sha256_path(root / ATTEMPT_FILENAME)
        and result["preflight_sha256"]
        == sha256_path(root / PREFLIGHT_FILENAME)
        and result["index_sha256"] is None
        and result["not_evaluable_sha256"]
        == sha256_path(root / NOT_EVALUABLE_FILENAME)
        and result["toolkit_commit"]
        == contract["source_authority"]["toolkit_commit"]
        and result["exact_blob_count"] == len(rows)
        and result["aggregate_shard_bytes"] == 0
        and result["global_record_counts"] is None
        and result["provider_resolution_performed"] is False
        and result["dataset_host_request_made"] is False
        and result["source_evidence_role"]
        == "CONSUMED_SOURCE_RECOVERY_NOT_FRESH_VALIDATION"
        and result["next_authority"]
        == (
            "STOP_PROVIDER_RESOLUTION_SOURCE_EVIDENCE_"
            "NOT_EVALUABLE"
        )
        and result["p0c_execution_authorized_automatically"] is False
        and claim_ceiling_is_false(result["claim_ceiling"], contract),
        "not-evaluable result binding mismatch",
    )
    return result


def root_names(root: Path) -> frozenset[str]:
    require(root.is_dir(), "artifact root is not a directory")
    names: set[str] = set()
    for path in root.iterdir():
        info = path.lstat()
        attributes = getattr(info, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        require(
            stat.S_ISREG(info.st_mode)
            and not (attributes & reparse),
            "top-level artifact is nonregular or a reparse point",
        )
        require(path.name not in names, "duplicate artifact name")
        names.add(path.name)
    return frozenset(names)


def validate_failure_terminal(
    root: Path,
    contract: dict[str, Any],
    contract_path: Path,
    closure: dict[str, Any],
) -> dict[str, Any]:
    del closure
    names = root_names(root)
    require(
        FAILURE_FILENAME in names
        and names <= failure_allowed_set(),
        "failure closed set contains unknown artifacts",
    )
    present_shards = [
        name for name in shard_filenames() if name in names
    ]
    require(
        present_shards
        == list(shard_filenames()[: len(present_shards)]),
        "failure shard set is not an exact prefix",
    )
    require(
        not (
            INDEX_FILENAME in names
            and len(present_shards) != SHARD_COUNT
        ),
        "index exists without all shard names",
    )
    require(
        not (
            NOT_EVALUABLE_FILENAME in names
            and present_shards
        )
        and not (
            NOT_EVALUABLE_FILENAME in names
            and INDEX_FILENAME in names
        ),
        "failure mixes mutually exclusive terminal paths",
    )
    failure = load_json_strict(
        root / FAILURE_FILENAME, CONTROL_ARTIFACT_MAXIMUM_BYTES
    )
    exact_keys(
        failure,
        contract["exact_artifact_schemas"]["failure_json_exact_keys"],
        "failure",
    )
    observed = []
    for name in sorted(names - {FAILURE_FILENAME}):
        path = root / name
        observed.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    for row in failure["observed_artifacts"]:
        exact_keys(
            row,
            contract["exact_artifact_schemas"][
                "failure_observed_artifact_exact_keys"
            ],
            "failure observed artifact",
        )
    require(
        failure["schema"] == FAILURE_SCHEMA
        and failure["terminal"] == EVIDENCE_INVALID
        and isinstance(failure["reason_class"], str)
        and bool(failure["reason_class"])
        and (
            failure["attempt_sha256"] is None
            or bool(
                HEX64_RE.fullmatch(str(failure["attempt_sha256"]))
            )
        )
        and failure["execution_contract_sha256"]
        == sha256_path(contract_path)
        and failure["observed_artifacts"] == observed
        and failure["resume_or_rerun_authorized"] is False
        and failure["source_reread_authorized"] is False
        and failure["new_git_fetch_or_network_made"] is False
        and failure["dataset_host_request_made"] is False,
        "failure binding mismatch",
    )
    if ATTEMPT_FILENAME in names:
        require(
            failure["attempt_sha256"]
            == sha256_path(root / ATTEMPT_FILENAME),
            "failure attempt hash mismatch",
        )
    else:
        require(
            failure["attempt_sha256"] is None,
            "failure attempt hash present without attempt",
        )
    return failure


def validate_terminal(
    root: Path,
    contract: dict[str, Any],
    contract_path: Path,
    closure: dict[str, Any],
) -> dict[str, Any]:
    """Validate in frozen order: LOCKED, NOT_EVALUABLE, then INVALID."""
    normal_errors: list[str] = []
    for validator in (
        validate_locked_terminal,
        validate_not_evaluable_terminal,
    ):
        try:
            return validator(root, contract, contract_path, closure)
        except (OSError, TerminalValidationError) as error:
            normal_errors.append(str(error))
    try:
        return validate_failure_terminal(
            root, contract, contract_path, closure
        )
    except (OSError, TerminalValidationError) as error:
        raise TerminalValidationError(
            "root validates as no terminal; "
            f"locked={normal_errors[0]}; "
            f"not_evaluable={normal_errors[1]}; "
            f"failure={error}"
        ) from error
