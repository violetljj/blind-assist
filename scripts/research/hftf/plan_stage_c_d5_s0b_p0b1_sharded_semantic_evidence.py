#!/usr/bin/env python3
"""Fail-closed draft skeleton for HFTF D5-S0B P0B.1 sharded evidence.

This module freezes names, schemas, terminals, capacity constants, and closed
artifact sets only.  It is deliberately not executable: implementation and
test receipts are not yet bound, and no source blob or canonical artifact root
may be opened by this draft.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Any, Callable, Sequence

sys_path_parent = str(Path(__file__).resolve().parent)
import sys

if sys_path_parent not in sys.path:
    sys.path.insert(0, sys_path_parent)

from plan_stage_c_d5_s0b_p0b_provider_semantic_evidence import (
    EvidenceVisitor,
    index_ast,
)
from plan_stage_c_d5_s0a_tartanground_catalog import (
    sha256,
    write_bytes_exclusive_fsync,
)


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_sharded_semantic_evidence_"
    "execution_contract"
)
CONTRACT_STATUS = "DRAFT_NOT_EXECUTABLE"
DESIGN_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_sharded_semantic_evidence_"
    "repair_design"
)
DESIGN_STATUS = (
    "FROZEN_AFTER_P0B_EVIDENCE_CAP_INVALID_BEFORE_P0B1_"
    "EXECUTION_CONTRACT_OR_SOURCE_REREAD"
)

ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_sharded_semantic_evidence_attempt"
)
PREFLIGHT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_sharded_semantic_evidence_preflight"
)
SHARD_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_semantic_evidence_shard"
)
INDEX_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_semantic_evidence_shard_index"
)
NOT_EVALUABLE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_sharded_semantic_evidence_"
    "not_evaluable"
)
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_sharded_semantic_evidence_result"
)
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0b1_sharded_semantic_evidence_failure"
)

EVIDENCE_LOCKED = (
    "D5_S0B_P0B1_SHARDED_SEMANTIC_EVIDENCE_LOCKED_REQUIRES_"
    "P0C_PROVIDER_RESOLUTION"
)
EVIDENCE_NOT_EVALUABLE = (
    "D5_S0B_P0B1_SHARDED_SEMANTIC_EVIDENCE_NOT_EVALUABLE"
)
EVIDENCE_INVALID = (
    "D5_S0B_P0B1_SHARDED_SEMANTIC_EVIDENCE_INVALID_STOP"
)

CANONICAL_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0b1-sharded-semantic-evidence-20260802"
)
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/hftf/"
    "HFTF_STAGE_C_D5_S0B_P0B1_SHARDED_SEMANTIC_EVIDENCE_"
    "EXECUTION_CONTRACT_2026-08-02.json"
)
DESIGN_RELATIVE_PATH = Path(
    "docs/research/hftf/"
    "HFTF_STAGE_C_D5_S0B_P0B1_SHARDED_SEMANTIC_EVIDENCE_"
    "REPAIR_DESIGN_2026-08-02.json"
)
DESIGN_SHA256 = (
    "6b2523091a967b2a64e2062c9314d1cc4d6eaf37b99de204f4fd9ccf953f5d9d"
)

ATTEMPT_FILENAME = "attempt.json"
PREFLIGHT_FILENAME = "preflight.json"
INDEX_FILENAME = "index.json"
NOT_EVALUABLE_FILENAME = "not-evaluable.json"
RESULT_FILENAME = "result.json"
FAILURE_FILENAME = "failure.json"
SHARD_COUNT = 18
CONTROL_ARTIFACT_MAXIMUM_BYTES = 1_048_576
AGGREGATE_SHARD_MAXIMUM_BYTES = 129_690_624
CAP_MANIFEST_SHA256 = (
    "a7e3203057f17467dfe50e5671ab51fa578b832d439305764895a7c845f0a9f8"
)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
GitRunner = Callable[[Sequence[str], Path], bytes]
BytesWriter = Callable[[Path, bytes], None]
EncodingDetector = Callable[[bytes], str]
AstParser = Callable[[str, str], ast.AST]
_FORMAL_EXECUTION_GATE = object()
_TEST_EXECUTION_GATE = object()
CAP_MANIFEST: tuple[dict[str, Any], ...] = (
    {"manifest_index": 0, "path": "tartanair/__init__.py",
     "p0a_blob_bytes": 49, "maximum_shard_bytes": 1048576},
    {"manifest_index": 1, "path": "tartanair/customizer.py",
     "p0a_blob_bytes": 32995, "maximum_shard_bytes": 16893440},
    {"manifest_index": 2, "path": "tartanair/dataloader.py",
     "p0a_blob_bytes": 12963, "maximum_shard_bytes": 6637056},
    {"manifest_index": 3, "path": "tartanair/dataset.py",
     "p0a_blob_bytes": 40478, "maximum_shard_bytes": 20724736},
    {"manifest_index": 4, "path": "tartanair/downloader.py",
     "p0a_blob_bytes": 32876, "maximum_shard_bytes": 16832512},
    {"manifest_index": 5,
     "path": "tartanair/eval_utils/trajectory_evaluator_ate.py",
     "p0a_blob_bytes": 4551, "maximum_shard_bytes": 2330112},
    {"manifest_index": 6,
     "path": "tartanair/eval_utils/trajectory_evaluator_base.py",
     "p0a_blob_bytes": 9774, "maximum_shard_bytes": 5004288},
    {"manifest_index": 7,
     "path": "tartanair/eval_utils/trajectory_evaluator_rpe.py",
     "p0a_blob_bytes": 4972, "maximum_shard_bytes": 2545664},
    {"manifest_index": 8, "path": "tartanair/evaluator.py",
     "p0a_blob_bytes": 4644, "maximum_shard_bytes": 2377728},
    {"manifest_index": 9, "path": "tartanair/flow_calculation.py",
     "p0a_blob_bytes": 18471, "maximum_shard_bytes": 9457152},
    {"manifest_index": 10, "path": "tartanair/flow_utils.py",
     "p0a_blob_bytes": 3252, "maximum_shard_bytes": 1665024},
    {"manifest_index": 11, "path": "tartanair/iterator.py",
     "p0a_blob_bytes": 16432, "maximum_shard_bytes": 8413184},
    {"manifest_index": 12, "path": "tartanair/lister.py",
     "p0a_blob_bytes": 1314, "maximum_shard_bytes": 1048576},
    {"manifest_index": 13, "path": "tartanair/reader.py",
     "p0a_blob_bytes": 8297, "maximum_shard_bytes": 4248064},
    {"manifest_index": 14, "path": "tartanair/tartanair.py",
     "p0a_blob_bytes": 32905, "maximum_shard_bytes": 16847360},
    {"manifest_index": 15, "path": "tartanair/tartanair_module.py",
     "p0a_blob_bytes": 15206, "maximum_shard_bytes": 7785472},
    {"manifest_index": 16, "path": "tartanair/unzipper.py",
     "p0a_blob_bytes": 3004, "maximum_shard_bytes": 1538048},
    {"manifest_index": 17, "path": "tartanair/visualizer.py",
     "p0a_blob_bytes": 8386, "maximum_shard_bytes": 4293632},
)

SHARD_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "manifest_index",
        "source_path",
        "toolkit_commit",
        "git_blob_oid",
        "source_blob_bytes",
        "source_blob_sha256",
        "detected_source_encoding",
        "parse_status",
        "ast_node_count",
        "maximum_ast_depth",
        "runtime_lock_sha256",
        "algorithm_lock_sha256",
    }
)
NODE_RECEIPT_FIELDS = frozenset(
    {
        "source_path",
        "node_id",
        "parent_node_id",
        "parent_field",
        "preorder_index",
        "node_type",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
        "depth",
        "canonical_ast_object",
    }
)
EXPRESSION_FIELDS = frozenset(
    {
        "source_path",
        "node_id",
        "parent_node_id",
        "parent_field",
        "node_type",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
        "source_segment_sha256",
        "source_segment_utf8_bytes",
        "source_encoding",
        "enclosing_lexical_scope",
        "lexical_role",
    }
)


class DraftNotExecutable(RuntimeError):
    """The P0B.1 draft lacks authority to open source or artifact roots."""


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


def reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DraftNotExecutable(
                f"P0B.1 duplicate JSON object key: {key}"
            )
        value[key] = item
    return value


def shard_filename(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("P0B.1 shard index must be an integer")
    if index < 0 or index >= SHARD_COUNT:
        raise ValueError("P0B.1 shard index outside 000..017")
    return f"shard_{index:03d}.json"


def shard_filenames() -> tuple[str, ...]:
    return tuple(shard_filename(index) for index in range(SHARD_COUNT))


def locked_closed_set() -> frozenset[str]:
    return frozenset(
        {
            ATTEMPT_FILENAME,
            PREFLIGHT_FILENAME,
            *shard_filenames(),
            INDEX_FILENAME,
            RESULT_FILENAME,
        }
    )


def not_evaluable_closed_set() -> frozenset[str]:
    return frozenset(
        {
            ATTEMPT_FILENAME,
            PREFLIGHT_FILENAME,
            NOT_EVALUABLE_FILENAME,
            RESULT_FILENAME,
        }
    )


def failure_allowed_set() -> frozenset[str]:
    return frozenset(
        {
            ATTEMPT_FILENAME,
            PREFLIGHT_FILENAME,
            *shard_filenames(),
            INDEX_FILENAME,
            NOT_EVALUABLE_FILENAME,
            RESULT_FILENAME,
            FAILURE_FILENAME,
        }
    )


def validate_frozen_capacity_constants() -> None:
    rendered = canonical_json_bytes(list(CAP_MANIFEST))
    if hashlib.sha256(rendered).hexdigest() != CAP_MANIFEST_SHA256:
        raise DraftNotExecutable("P0B.1 cap manifest hash drift")
    if sum(
        int(row["maximum_shard_bytes"]) for row in CAP_MANIFEST
    ) != AGGREGATE_SHARD_MAXIMUM_BYTES:
        raise DraftNotExecutable("P0B.1 aggregate shard cap drift")
    if [row["manifest_index"] for row in CAP_MANIFEST] != list(
        range(SHARD_COUNT)
    ):
        raise DraftNotExecutable("P0B.1 cap manifest order drift")
    for row in CAP_MANIFEST:
        expected = max(
            CONTROL_ARTIFACT_MAXIMUM_BYTES,
            512 * int(row["p0a_blob_bytes"]),
        )
        if int(row["maximum_shard_bytes"]) != expected:
            raise DraftNotExecutable("P0B.1 per-shard cap formula drift")


def canonical_object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def false_claim_ceiling(contract: dict[str, Any]) -> dict[str, bool]:
    return {
        key: False
        for key in contract["exact_artifact_schemas"][
            "claim_ceiling_exact_keys"
        ]
    }


def serialize_control_artifact(
    name: str,
    value: dict[str, Any],
) -> bytes:
    payload = canonical_json_bytes(value)
    if len(payload) > CONTROL_ARTIFACT_MAXIMUM_BYTES:
        raise ValueError(f"P0B.1 control artifact cap exceeded: {name}")
    return payload


def detect_source_encoding(blob: bytes) -> str:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(blob).readline)
    return encoding


def parse_source_ast(source: str, filename: str) -> ast.AST:
    return ast.parse(source, filename=filename)


def records_for_path(
    observation: dict[str, Any],
    key: str,
    path: str,
) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(row)
        for row in observation[key]
        if row["source_path"] == path
    ]


def build_shards_from_p0b_observation(
    contract: dict[str, Any],
    closure: dict[str, Any],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert complete in-memory P0B evidence into deduplicated path shards."""
    rows = closure["observation"]["closure_rows"]
    if len(rows) != int(
        contract["source_authority"]["exact_ordered_blob_count"]
    ):
        raise ValueError("P0B.1 closure blob count drift")
    object_by_path = {
        row["path"]: row for row in observation["object_receipts"]
    }
    parse_by_path = {
        row["path"]: row for row in observation["parse_receipts"]
    }
    runtime_sha = canonical_object_sha256(contract["runtime_lock"])
    algorithm_sha = canonical_object_sha256(contract["algorithm_lock"])
    shards: list[dict[str, Any]] = []
    aggregate_bytes = 0
    for manifest_index, closure_row in enumerate(rows):
        path = closure_row["path"]
        if len(rows) == SHARD_COUNT and (
            CAP_MANIFEST[manifest_index]["path"] != path
            or CAP_MANIFEST[manifest_index]["p0a_blob_bytes"]
            != int(closure_row["bytes"])
        ):
            raise ValueError("P0B.1 cap manifest source binding drift")
        object_receipt = copy.deepcopy(object_by_path[path])
        parse_receipt = parse_by_path[path]
        nodes = records_for_path(
            observation, "ast_node_receipts", path
        )
        node_map = {row["node_id"]: row for row in nodes}
        for node in nodes:
            node["canonical_ast_object"] = json.loads(
                node.pop("canonical_ast_dump")
            )
        expressions = records_for_path(
            observation, "expressions", path
        )
        for expression in expressions:
            segment = expression.pop("source_segment")
            expression.pop("canonical_ast_dump")
            expression["source_segment_utf8_bytes"] = len(
                segment.encode("utf-8")
            )
            expression["source_encoding"] = parse_receipt[
                "detected_source_encoding"
            ]
        strings = records_for_path(
            observation, "string_literals", path
        )
        for string in strings:
            string["value_utf8_bytes"] = len(
                string["value"].encode("utf-8")
            )
        calls = records_for_path(observation, "call_sites", path)
        for call in calls:
            call["callee_syntax_sha256"] = hashlib.sha256(
                call["callee_syntax"].encode("utf-8")
            ).hexdigest()
            call["callee_syntax_utf8_bytes"] = len(
                call["callee_syntax"].encode("utf-8")
            )
            call["source_segment_utf8_bytes"] = len(
                call["source_segment"].encode("utf-8")
            )
            for field in ("callee_syntax", "source_segment"):
                if call[f"{field}_utf8_bytes"] > int(
                    contract["unchanged_extraction_limits"][
                        "maximum_single_source_segment_utf8_bytes"
                    ]
                ):
                    raise ValueError(
                        f"P0B.1 retained call text cap exceeded: {field}"
                    )
        assignments = records_for_path(
            observation, "assignments", path
        )
        for assignment in assignments:
            for field in ("target_syntax", "value_syntax"):
                assignment[f"{field}_sha256"] = hashlib.sha256(
                    assignment[field].encode("utf-8")
                ).hexdigest()
                assignment[f"{field}_utf8_bytes"] = len(
                    assignment[field].encode("utf-8")
                )
            assignment["source_segment_utf8_bytes"] = len(
                assignment["source_segment"].encode("utf-8")
            )
            for field in (
                "target_syntax",
                "value_syntax",
                "source_segment",
            ):
                if assignment[f"{field}_utf8_bytes"] > int(
                    contract["unchanged_extraction_limits"][
                        "maximum_single_source_segment_utf8_bytes"
                    ]
                ):
                    raise ValueError(
                        "P0B.1 retained assignment text cap exceeded: "
                        f"{field}"
                    )
        functions = records_for_path(
            observation, "functions", path
        )
        imports = records_for_path(
            observation, "import_aliases", path
        )
        for import_record in imports:
            node = node_map[import_record["node_id"]]
            import_record["col_offset"] = node["col_offset"]
            import_record["end_col_offset"] = node["end_col_offset"]
        counts = {
            "ast_nodes": len(nodes),
            "expressions": len(expressions),
            "strings": len(strings),
            "calls": len(calls),
            "assignments": len(assignments),
            "functions": len(functions),
            "imports": len(imports),
        }
        shard = {
            "schema": SHARD_SCHEMA,
            "status": "AST_SEMANTIC_SHARD_LOCKED",
            "manifest_index": manifest_index,
            "source_path": path,
            "toolkit_commit": contract["source_authority"][
                "toolkit_commit"
            ],
            "git_blob_oid": closure_row["git_blob_oid"],
            "source_blob_bytes": closure_row["bytes"],
            "source_blob_sha256": closure_row["sha256"],
            "detected_source_encoding": parse_receipt[
                "detected_source_encoding"
            ],
            "parse_status": parse_receipt["ast_parse_status"],
            "ast_node_count": parse_receipt["ast_node_count"],
            "maximum_ast_depth": parse_receipt["maximum_ast_depth"],
            "runtime_lock_sha256": runtime_sha,
            "algorithm_lock_sha256": algorithm_sha,
            "object_receipt": object_receipt,
            "node_receipts": nodes,
            "expression_records": expressions,
            "string_literal_records": strings,
            "call_site_records": calls,
            "assignment_records": assignments,
            "function_records": functions,
            "import_alias_records": imports,
            "record_counts": counts,
        }
        shard_bytes = canonical_json_bytes(shard)
        maximum = int(
            CAP_MANIFEST[manifest_index]["maximum_shard_bytes"]
        )
        if len(shard_bytes) > maximum:
            raise ValueError(
                f"P0B.1 shard cap exceeded: {manifest_index}"
            )
        aggregate_bytes += len(shard_bytes)
        shards.append(shard)
    if aggregate_bytes > AGGREGATE_SHARD_MAXIMUM_BYTES:
        raise ValueError("P0B.1 aggregate shard cap exceeded")
    return shards


def extract_sharded_evidence(
    contract: dict[str, Any],
    p0b_contract: dict[str, Any],
    closure: dict[str, Any],
    toolkit: Path,
    *,
    git_runner: GitRunner,
    encoding_detector: EncodingDetector = detect_source_encoding,
    parser: AstParser = parse_source_ast,
) -> tuple[str, dict[str, Any] | list[dict[str, Any]]]:
    """Read each bound blob once and build all proposed shards in memory."""
    closure_observation = closure["observation"]
    if (
        closure_observation["dynamic_import_call_count"] != 0
        or closure_observation[
            "indirect_dynamic_import_or_exec_count"
        ] != 0
    ):
        return EVIDENCE_NOT_EVALUABLE, {
            "reason": "p0a_dynamic_import_evidence_nonzero",
            "source_blob_read_count": 0,
            "object_receipt_count": 0,
            "object_receipts": [],
            "object_receipt_set_sha256": None,
            "source_total_bytes": 0,
            "failed_manifest_index": None,
            "failed_path": None,
            "detected_source_encoding": None,
            "error_type": None,
            "parse_receipt_count": 0,
            "parse_receipts": [],
            "parsed_prefix_ast_node_count": 0,
            "parsed_prefix_maximum_ast_depth_observed": 0,
            "parsed_prefix_string_literal_count": 0,
            "parsed_prefix_call_site_count": 0,
            "parsed_prefix_assignment_count": 0,
            "parsed_prefix_function_count": 0,
            "parsed_prefix_import_alias_count": 0,
            "parsed_prefix_expression_count": 0,
        }

    rows = closure_observation["closure_rows"]
    if [row["path"] for row in rows] != sorted(
        row["path"] for row in rows
    ):
        raise ValueError("P0B.1 P0A closure row order drift")
    if len(rows) != int(
        contract["source_authority"]["exact_ordered_blob_count"]
    ):
        raise ValueError("P0B.1 P0A closure blob count drift")

    p0b_rules = p0b_contract["frozen_extraction"]
    p0b1_limits = contract["unchanged_extraction_limits"]
    limit_bindings = {
        "maximum_string_literal_records": (
            "maximum_string_literal_records_global"
        ),
        "maximum_call_site_records": (
            "maximum_call_site_records_global"
        ),
        "maximum_assignment_records": (
            "maximum_assignment_records_global"
        ),
        "maximum_function_records": "maximum_function_records_global",
        "maximum_import_alias_records": (
            "maximum_import_alias_records_global"
        ),
        "maximum_expression_records": (
            "maximum_expression_records_global"
        ),
        "maximum_ast_nodes": "maximum_ast_nodes_global",
        "maximum_ast_depth": "maximum_ast_depth_per_blob",
        "maximum_single_string_utf8_bytes": (
            "maximum_single_string_utf8_bytes"
        ),
        "maximum_single_source_segment_utf8_bytes": (
            "maximum_single_source_segment_utf8_bytes"
        ),
        "string_literal_role_classes": "string_literal_role_classes",
    }
    for p0b_key, p0b1_key in limit_bindings.items():
        if p0b_rules[p0b_key] != p0b1_limits[p0b1_key]:
            raise ValueError(
                f"P0B.1 extraction-limit binding drift: {p0b1_key}"
            )

    object_receipts: list[dict[str, Any]] = []
    verified_sources: list[tuple[dict[str, Any], bytes]] = []
    commit = contract["source_authority"]["toolkit_commit"]
    if commit != p0b_contract["source_boundary"]["toolkit_commit"]:
        raise ValueError("P0B.1 toolkit commit binding drift")
    for row in rows:
        oid = str(row["git_blob_oid"])
        if not re.fullmatch(r"[0-9a-f]{40}", oid):
            raise ValueError("P0B.1 invalid blob OID")
        actual_oid = git_runner(
            ["rev-parse", f"{commit}:{row['path']}"], toolkit
        ).decode("ascii").strip()
        object_type = git_runner(
            ["cat-file", "-t", oid], toolkit
        ).decode("ascii").strip()
        size_text = git_runner(
            ["cat-file", "-s", oid], toolkit
        ).decode("ascii").strip()
        if not re.fullmatch(r"0|[1-9][0-9]*", size_text):
            raise ValueError(
                f"P0B.1 invalid object size: {row['path']}"
            )
        actual_size = int(size_text)
        if (
            actual_oid != oid
            or object_type != "blob"
            or actual_size != int(row["bytes"])
        ):
            raise ValueError(
                f"P0B.1 object receipt mismatch: {row['path']}"
            )
        blob = git_runner(["cat-file", "blob", oid], toolkit)
        actual_sha = hashlib.sha256(blob).hexdigest()
        if len(blob) != actual_size or actual_sha != row["sha256"]:
            raise ValueError(f"P0B.1 blob hash mismatch: {row['path']}")
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
    detected_sources: list[
        tuple[dict[str, Any], bytes, str | None, BaseException | None]
    ] = []
    for row, blob in verified_sources:
        try:
            encoding = encoding_detector(blob)
            detected_sources.append((row, blob, encoding, None))
        except (UnicodeError, SyntaxError) as error:
            detected_sources.append((row, blob, None, error))

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

    def not_evaluable_payload(
        manifest_index: int,
        row: dict[str, Any],
        encoding: str | None,
        error: BaseException,
    ) -> dict[str, Any]:
        return {
            "reason": (
                "verified_source_incompatible_with_frozen_encoding_"
                "or_ast_grammar"
            ),
            "source_blob_read_count": len(verified_sources),
            "object_receipt_count": len(object_receipts),
            "object_receipts": object_receipts,
            "object_receipt_set_sha256": object_receipt_set_sha256,
            "source_total_bytes": sum(
                receipt["actual_content_bytes"]
                for receipt in object_receipts
            ),
            "failed_manifest_index": manifest_index,
            "failed_path": row["path"],
            "detected_source_encoding": encoding,
            "error_type": type(error).__name__,
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

    for manifest_index, (
        row,
        blob,
        encoding,
        detection_error,
    ) in enumerate(detected_sources):
        if detection_error is not None:
            return (
                EVIDENCE_NOT_EVALUABLE,
                not_evaluable_payload(
                    manifest_index,
                    row,
                    encoding,
                    detection_error,
                ),
            )
        assert encoding is not None
        try:
            source = blob.decode(encoding)
            tree = parser(source, str(row["path"]))
        except (UnicodeError, SyntaxError) as error:
            return (
                EVIDENCE_NOT_EVALUABLE,
                not_evaluable_payload(
                    manifest_index, row, encoding, error
                ),
            )
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
            int(p0b_rules["maximum_ast_depth"]),
        )
        if (
            total_ast_nodes + node_count
            > int(p0b_rules["maximum_ast_nodes"])
        ):
            raise ValueError("P0B.1 global AST node cap exceeded")
        visitor = EvidenceVisitor(
            str(row["path"]),
            source,
            p0b_rules,
            node_ids,
            parents,
            edge_node_ids_by_node,
        )
        visitor.visit(tree)
        additions = (
            (
                strings,
                visitor.strings,
                "maximum_string_literal_records",
            ),
            (calls, visitor.calls, "maximum_call_site_records"),
            (
                assignments,
                visitor.assignments,
                "maximum_assignment_records",
            ),
            (
                functions,
                visitor.functions,
                "maximum_function_records",
            ),
            (
                imports,
                visitor.imports,
                "maximum_import_alias_records",
            ),
            (
                expressions,
                visitor.expressions,
                "maximum_expression_records",
            ),
        )
        for existing, added, cap_name in additions:
            if len(existing) + len(added) > int(
                p0b_rules[cap_name]
            ):
                raise ValueError(
                    f"P0B.1 evidence record cap exceeded: {cap_name}"
                )
        total_ast_nodes += node_count
        maximum_ast_depth_observed = max(
            maximum_ast_depth_observed, depth
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
        for existing, added, _ in additions:
            existing.extend(added)

    preorder_by_node_id = {
        row["node_id"]: int(row["preorder_index"])
        for row in node_receipts
    }
    for records in (
        strings,
        calls,
        assignments,
        functions,
        imports,
        expressions,
    ):
        records.sort(
            key=lambda row: (
                preorder_by_node_id[row["node_id"]],
                canonical_json_bytes(row),
            )
        )
    observation = {
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
        "string_literals": strings,
        "call_sites": calls,
        "assignments": assignments,
        "functions": functions,
        "import_aliases": imports,
        "expressions": expressions,
    }
    return (
        EVIDENCE_LOCKED,
        build_shards_from_p0b_observation(
            contract, closure, observation
        ),
    )


def artifact_state(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir()}


def build_index(
    contract: dict[str, Any],
    closure: dict[str, Any],
    shards: list[dict[str, Any]],
    *,
    attempt_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    index_rows: list[dict[str, Any]] = []
    global_counts = {
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
    for shard in shards:
        content = canonical_json_bytes(shard)
        aggregate += len(content)
        for key, count in shard["record_counts"].items():
            global_counts[key] += int(count)
        index_rows.append(
            {
                "manifest_index": shard["manifest_index"],
                "source_path": shard["source_path"],
                "shard_filename": shard_filename(
                    shard["manifest_index"]
                ),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "record_counts": shard["record_counts"],
                "maximum_ast_depth": shard["maximum_ast_depth"],
                "maximum_shard_bytes": CAP_MANIFEST[
                    shard["manifest_index"]
                ]["maximum_shard_bytes"],
            }
        )
    return {
        "schema": INDEX_SCHEMA,
        "status": "SHARDED_AST_SEMANTIC_EVIDENCE_LOCKED",
        "attempt_sha256": attempt_sha256,
        "preflight_sha256": preflight_sha256,
        "toolkit_commit": contract["source_authority"]["toolkit_commit"],
        "p0a_manifest_sha256": contract["source_authority"][
            "exact_ordered_row_manifest_sha256"
        ],
        "object_receipt_set_sha256": hashlib.sha256(
            canonical_json_bytes(
                [shard["object_receipt"] for shard in shards]
            )
        ).hexdigest(),
        "runtime_lock_sha256": canonical_object_sha256(
            contract["runtime_lock"]
        ),
        "algorithm_lock_sha256": canonical_object_sha256(
            contract["algorithm_lock"]
        ),
        "cap_manifest_sha256": CAP_MANIFEST_SHA256,
        "shards": index_rows,
        "aggregate_shard_bytes": aggregate,
        "global_record_counts": global_counts,
        "global_maximum_ast_depth": max(
            shard["maximum_ast_depth"] for shard in shards
        ),
    }


def observed_artifacts(root: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == FAILURE_FILENAME:
            continue
        if not path.is_file():
            raise ValueError("P0B.1 nonregular top-level artifact")
        rows.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def execute_with_context(
    contract: dict[str, Any],
    p0b_contract: dict[str, Any],
    closure: dict[str, Any],
    toolkit: Path,
    root: Path,
    *,
    git_runner: GitRunner,
    writer: BytesWriter = write_bytes_exclusive_fsync,
    execution_gate: object | None = None,
) -> dict[str, Any]:
    """Gated execution core; raw draft dictionaries are never sufficient."""
    if execution_gate is _TEST_EXECUTION_GATE:
        resolved_repo = repo_root().resolve()
        resolved_root = root.resolve()
        resolved_toolkit = toolkit.resolve()
        if (
            resolved_root == (resolved_repo / CANONICAL_ROOT).resolve()
            or resolved_repo in resolved_root.parents
            or resolved_repo in resolved_toolkit.parents
        ):
            raise DraftNotExecutable(
                "P0B.1 test gate accepts only non-repository temporary paths"
            )
    elif execution_gate is not _FORMAL_EXECUTION_GATE:
        raise DraftNotExecutable(
            "P0B.1 execution core requires validated executable authority"
        )
    root.mkdir(parents=True, exist_ok=False)
    contract_path = repo_root() / CONTRACT_RELATIVE_PATH
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "ATTEMPT_FSYNCED_BEFORE_FIRST_SOURCE_BLOB_READ",
        "execution_contract_sha256": (
            sha256(contract_path) if contract_path.is_file() else "0" * 64
        ),
        "p0b1_design_sha256": DESIGN_SHA256,
        "p0b_invalid_result_sha256": contract["parents"][
            "p0b_invalid_result"
        ]["sha256"],
        "p0a_closure_sha256": contract["source_authority"][
            "p0a_closure_artifact"
        ]["sha256"],
        "toolkit_commit": contract["source_authority"]["toolkit_commit"],
        "new_git_fetch_or_network_authorized": False,
        "dataset_host_request_authorized": False,
        "old_p0b_root_reopen_authorized": False,
    }
    try:
        writer(
            root / ATTEMPT_FILENAME,
            serialize_control_artifact(ATTEMPT_FILENAME, attempt),
        )
        preflight = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_SOURCE_BLOB_READ",
            "attempt_sha256": sha256(root / ATTEMPT_FILENAME),
            "execution_commit": contract.get(
                "draft_execution_commit", "TEST_CONTEXT"
            ),
            "head_equal_origin_master": True,
            "tracked_clean": True,
            "new_canonical_root_absent_before_attempt": True,
            "p0a_terminal_validated": True,
            "p0b_invalid_terminal_validated": True,
            "exact_blob_count": len(
                closure["observation"]["closure_rows"]
            ),
            "exact_total_source_bytes": sum(
                int(row["bytes"])
                for row in closure["observation"]["closure_rows"]
            ),
            "exact_ordered_row_manifest_sha256": contract[
                "source_authority"
            ]["exact_ordered_row_manifest_sha256"],
            "runtime_lock": contract["runtime_lock"],
            "runtime_lock_sha256": canonical_object_sha256(
                contract["runtime_lock"]
            ),
            "algorithm_lock_sha256": canonical_object_sha256(
                contract["algorithm_lock"]
            ),
            "new_git_fetch_or_network_made": False,
            "dataset_host_request_made": False,
        }
        writer(
            root / PREFLIGHT_FILENAME,
            serialize_control_artifact(PREFLIGHT_FILENAME, preflight),
        )
        terminal, payload = extract_sharded_evidence(
            contract,
            p0b_contract,
            closure,
            toolkit,
            git_runner=git_runner,
        )
        if terminal == EVIDENCE_NOT_EVALUABLE:
            not_evaluable = {
                "schema": NOT_EVALUABLE_SCHEMA,
                "terminal": terminal,
                "attempt_sha256": sha256(root / ATTEMPT_FILENAME),
                "preflight_sha256": sha256(root / PREFLIGHT_FILENAME),
                "toolkit_commit": contract["source_authority"][
                    "toolkit_commit"
                ],
                "reason_class": payload["reason"],
                "source_blob_read_count": payload.get(
                    "source_blob_read_count",
                    payload.get("object_receipt_count", 0),
                ),
                "object_receipt_count": payload.get(
                    "object_receipt_count", 0
                ),
                "object_receipts": payload.get("object_receipts", []),
                "object_receipt_set_sha256": payload.get(
                    "object_receipt_set_sha256"
                ),
                "source_total_bytes": payload.get("source_total_bytes", 0),
                "manifest_index": payload.get("failed_manifest_index"),
                "source_path": payload.get("failed_path"),
                "detected_source_encoding": payload.get(
                    "detected_source_encoding"
                ),
                "parse_status": (
                    None
                    if payload["reason"]
                    == "p0a_dynamic_import_evidence_nonzero"
                    else "ENCODING_DETECTION_FAILED"
                    if payload.get("detected_source_encoding") is None
                    else "SOURCE_DECODE_FAILED"
                    if payload.get("error_type")
                    in {"UnicodeDecodeError", "UnicodeError"}
                    else "AST_PARSE_FAILED"
                ),
                "error_type": payload.get("error_type"),
                "parse_receipt_count": payload.get(
                    "parse_receipt_count", 0
                ),
                "parse_receipts": payload.get("parse_receipts", []),
                "prefix_ast_node_count": payload.get(
                    "parsed_prefix_ast_node_count", 0
                ),
                "prefix_maximum_ast_depth": payload.get(
                    "parsed_prefix_maximum_ast_depth_observed", 0
                ),
                "prefix_string_count": payload.get(
                    "parsed_prefix_string_literal_count", 0
                ),
                "prefix_call_count": payload.get(
                    "parsed_prefix_call_site_count", 0
                ),
                "prefix_assignment_count": payload.get(
                    "parsed_prefix_assignment_count", 0
                ),
                "prefix_function_count": payload.get(
                    "parsed_prefix_function_count", 0
                ),
                "prefix_import_count": payload.get(
                    "parsed_prefix_import_alias_count", 0
                ),
                "prefix_expression_count": payload.get(
                    "parsed_prefix_expression_count", 0
                ),
                "new_git_fetch_or_network_made": False,
                "dataset_host_request_made": False,
            }
            if set(not_evaluable) != set(
                contract["exact_artifact_schemas"][
                    "not_evaluable_json_exact_keys"
                ]
            ):
                raise ValueError(
                    "P0B.1 not-evaluable exact schema mismatch"
                )
            writer(
                root / NOT_EVALUABLE_FILENAME,
                serialize_control_artifact(
                    NOT_EVALUABLE_FILENAME, not_evaluable
                ),
            )
            result = {
                "schema": RESULT_SCHEMA,
                "terminal": terminal,
                "attempt_sha256": sha256(root / ATTEMPT_FILENAME),
                "preflight_sha256": sha256(root / PREFLIGHT_FILENAME),
                "index_sha256": None,
                "not_evaluable_sha256": sha256(
                    root / NOT_EVALUABLE_FILENAME
                ),
                "toolkit_commit": contract["source_authority"][
                    "toolkit_commit"
                ],
                "exact_blob_count": len(
                    closure["observation"]["closure_rows"]
                ),
                "aggregate_shard_bytes": 0,
                "global_record_counts": None,
                "provider_resolution_performed": False,
                "dataset_host_request_made": False,
                "source_evidence_role": (
                    "CONSUMED_SOURCE_RECOVERY_NOT_FRESH_VALIDATION"
                ),
                "next_authority": (
                    "STOP_PROVIDER_RESOLUTION_SOURCE_EVIDENCE_"
                    "NOT_EVALUABLE"
                ),
                "p0c_execution_authorized_automatically": False,
                "claim_ceiling": false_claim_ceiling(contract),
            }
            writer(
                root / RESULT_FILENAME,
                serialize_control_artifact(RESULT_FILENAME, result),
            )
            from validate_stage_c_d5_s0b_p0b1_sharded_semantic_evidence import (
                validate_not_evaluable_terminal,
            )

            return validate_not_evaluable_terminal(
                root, contract, contract_path, closure
            )
        shards = payload
        from validate_stage_c_d5_s0b_p0b1_sharded_semantic_evidence import (
            load_json_strict,
            validate_locked_terminal,
            validate_shard,
        )

        serialized_shards: list[bytes] = []
        for manifest_index, shard in enumerate(shards):
            validate_shard(
                shard,
                contract,
                closure["observation"]["closure_rows"][manifest_index],
                manifest_index,
            )
            content = canonical_json_bytes(shard)
            if len(content) > int(
                CAP_MANIFEST[manifest_index]["maximum_shard_bytes"]
            ):
                raise ValueError(
                    f"P0B.1 shard cap exceeded: {manifest_index}"
                )
            serialized_shards.append(content)
        if (
            sum(len(content) for content in serialized_shards)
            > AGGREGATE_SHARD_MAXIMUM_BYTES
        ):
            raise ValueError("P0B.1 aggregate shard cap exceeded")
        for shard, content in zip(
            shards, serialized_shards, strict=True
        ):
            writer(
                root / shard_filename(shard["manifest_index"]),
                content,
            )
        durable_shards = []
        for manifest_index in range(len(shards)):
            path = root / shard_filename(manifest_index)
            durable = load_json_strict(
                path,
                int(
                    CAP_MANIFEST[manifest_index][
                        "maximum_shard_bytes"
                    ]
                ),
            )
            validate_shard(
                durable,
                contract,
                closure["observation"]["closure_rows"][manifest_index],
                manifest_index,
            )
            durable_shards.append(durable)
        index = build_index(
            contract,
            closure,
            durable_shards,
            attempt_sha256=sha256(root / ATTEMPT_FILENAME),
            preflight_sha256=sha256(root / PREFLIGHT_FILENAME),
        )
        writer(
            root / INDEX_FILENAME,
            serialize_control_artifact(INDEX_FILENAME, index),
        )
        durable_index = load_json_strict(
            root / INDEX_FILENAME, CONTROL_ARTIFACT_MAXIMUM_BYTES
        )
        recomputed_index = build_index(
            contract,
            closure,
            durable_shards,
            attempt_sha256=sha256(root / ATTEMPT_FILENAME),
            preflight_sha256=sha256(root / PREFLIGHT_FILENAME),
        )
        if durable_index != recomputed_index:
            raise ValueError("P0B.1 durable index validation failed")
        result = {
            "schema": RESULT_SCHEMA,
            "terminal": EVIDENCE_LOCKED,
            "attempt_sha256": sha256(root / ATTEMPT_FILENAME),
            "preflight_sha256": sha256(root / PREFLIGHT_FILENAME),
            "index_sha256": sha256(root / INDEX_FILENAME),
            "not_evaluable_sha256": None,
            "toolkit_commit": contract["source_authority"][
                "toolkit_commit"
            ],
            "exact_blob_count": len(shards),
            "aggregate_shard_bytes": durable_index[
                "aggregate_shard_bytes"
            ],
            "global_record_counts": durable_index[
                "global_record_counts"
            ],
            "provider_resolution_performed": False,
            "dataset_host_request_made": False,
            "source_evidence_role": (
                "CONSUMED_SOURCE_RECOVERY_NOT_FRESH_VALIDATION"
            ),
            "next_authority": (
                "FREEZE_SEPARATE_HASH_BOUND_P0C_PROVIDER_"
                "RESOLUTION_CONTRACT"
            ),
            "p0c_execution_authorized_automatically": False,
            "claim_ceiling": false_claim_ceiling(contract),
        }
        writer(
            root / RESULT_FILENAME,
            serialize_control_artifact(RESULT_FILENAME, result),
        )
        return validate_locked_terminal(
            root, contract, contract_path, closure
        )
    except BaseException as error:
        from validate_stage_c_d5_s0b_p0b1_sharded_semantic_evidence import (
            TerminalValidationError,
            validate_attempt_preflight,
            validate_locked_terminal,
            validate_not_evaluable_terminal,
        )

        for normal_validator in (
            validate_locked_terminal,
            validate_not_evaluable_terminal,
        ):
            try:
                return normal_validator(
                    root, contract, contract_path, closure
                )
            except (OSError, TerminalValidationError):
                pass
        if root.exists() and not (root / FAILURE_FILENAME).exists():
            try:
                validate_attempt_preflight(
                    root, contract, contract_path, closure
                )
                reason = f"{type(error).__name__}: {error}"
                failure = {
                    "schema": FAILURE_SCHEMA,
                    "terminal": EVIDENCE_INVALID,
                    "reason_class": reason[:1024],
                    "attempt_sha256": sha256(
                        root / ATTEMPT_FILENAME
                    ),
                    "execution_contract_sha256": sha256(contract_path),
                    "observed_artifacts": observed_artifacts(root),
                    "resume_or_rerun_authorized": False,
                    "source_reread_authorized": False,
                    "new_git_fetch_or_network_made": False,
                    "dataset_host_request_made": False,
                }
                writer(
                    root / FAILURE_FILENAME,
                    serialize_control_artifact(
                        FAILURE_FILENAME, failure
                    ),
                )
            except BaseException:
                pass
        raise


def load_contract_fail_closed(contract_path: Path) -> dict[str, Any]:
    """Load identity only, then refuse because receipts are deliberately unbound."""
    validate_frozen_capacity_constants()
    path = contract_path.resolve()
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_object_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DraftNotExecutable(
            f"P0B.1 execution contract is absent or unreadable: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DraftNotExecutable("P0B.1 execution contract must be an object")
    if value.get("schema") != CONTRACT_SCHEMA:
        raise DraftNotExecutable("P0B.1 execution contract schema mismatch")
    if value.get("status") != CONTRACT_STATUS:
        raise DraftNotExecutable(
            "P0B.1 skeleton accepts only the non-executable draft status"
        )
    if value.get("executable") is not False:
        raise DraftNotExecutable(
            "P0B.1 draft unexpectedly claims executable status"
        )
    implementations = value.get("implementation_receipts")
    tests = value.get("test_receipts")
    expected_implementations = {
        "status": "UNBOUND_TODO",
        "planner": {"path": "UNBOUND_TODO", "sha256": "UNBOUND_TODO"},
        "durability_helper": {
            "path": "UNBOUND_TODO",
            "sha256": "UNBOUND_TODO",
        },
        "terminal_validator": {
            "path": "UNBOUND_TODO",
            "sha256": "UNBOUND_TODO",
        },
    }
    expected_tests = {
        "status": "UNBOUND_TODO",
        "focused_test_path": "UNBOUND_TODO",
        "focused_test_sha256": "UNBOUND_TODO",
        "focused_test_count": "UNBOUND_TODO",
        "focused_tests_passed": "UNBOUND_TODO",
        "full_hftf_test_count": "UNBOUND_TODO",
        "full_hftf_tests_passed": "UNBOUND_TODO",
        "failure_injection_tests_passed": "UNBOUND_TODO",
        "independent_scientific_audit": "UNBOUND_TODO",
        "independent_engineering_audit": "UNBOUND_TODO",
    }
    if (
        implementations != expected_implementations
        or tests != expected_tests
    ):
        raise DraftNotExecutable(
            "P0B.1 draft unexpectedly contains executable receipts"
        )
    parent = value.get("parents", {}).get("p0b1_repair_design", {})
    if (
        parent.get("path") != DESIGN_RELATIVE_PATH.as_posix()
        or parent.get("sha256") != DESIGN_SHA256
        or parent.get("required_status") != DESIGN_STATUS
    ):
        raise DraftNotExecutable("P0B.1 repair-design binding drift")
    authorization = value.get("authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("p0b1_execution") is not False
        or authorization.get("p0b1_source_blob_reread") is not False
        or authorization.get("new_git_fetch_checkout_or_network") is not False
        or authorization.get("dataset_host_or_zip_request") is not False
    ):
        raise DraftNotExecutable("P0B.1 draft authorization drift")
    capacity = value.get("capacity_contract")
    if (
        not isinstance(capacity, dict)
        or capacity.get("exact_ordered_per_shard_cap_manifest")
        != list(CAP_MANIFEST)
        or capacity.get("exact_ordered_per_shard_cap_manifest_sha256")
        != CAP_MANIFEST_SHA256
        or capacity.get("aggregate_shard_maximum_bytes")
        != AGGREGATE_SHARD_MAXIMUM_BYTES
    ):
        raise DraftNotExecutable("P0B.1 draft capacity binding drift")
    schemas = value.get("exact_artifact_schemas")
    if (
        not isinstance(schemas, dict)
        or set(schemas.get("node_receipt_exact_keys", []))
        != NODE_RECEIPT_FIELDS
        or set(schemas.get("expression_record_exact_keys", []))
        != EXPRESSION_FIELDS
    ):
        raise DraftNotExecutable("P0B.1 draft receipt schema drift")
    closed_sets = value.get("terminal_closed_sets", {})
    if (
        closed_sets.get("locked_exact_names_in_write_order")
        != [
            ATTEMPT_FILENAME,
            PREFLIGHT_FILENAME,
            *shard_filenames(),
            INDEX_FILENAME,
            RESULT_FILENAME,
        ]
        or closed_sets.get("not_evaluable_exact_names_in_write_order")
        != [
            ATTEMPT_FILENAME,
            PREFLIGHT_FILENAME,
            NOT_EVALUABLE_FILENAME,
            RESULT_FILENAME,
        ]
    ):
        raise DraftNotExecutable("P0B.1 draft terminal closed-set drift")
    raise DraftNotExecutable(
        "P0B.1 contract is UNBOUND: planner/test/runtime receipts, "
        "terminal validators, extraction, and failure closure remain TODO; "
        "source and canonical artifact root access is forbidden"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-contract",
        type=Path,
        default=repo_root() / CONTRACT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / CANONICAL_ROOT,
        help="Identity only; this draft never resolves, reads, or creates it.",
    )
    args = parser.parse_args()
    del args.output_root
    try:
        load_contract_fail_closed(args.execution_contract)
    except DraftNotExecutable as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": CONTRACT_STATUS,
                    "error": str(error),
                    "source_blob_read_count": 0,
                    "canonical_root_opened_or_created": False,
                    "network_request_made": False,
                },
                ensure_ascii=False,
            )
        )
        return 2
    raise AssertionError("P0B.1 draft contract unexpectedly became executable")


if __name__ == "__main__":
    raise SystemExit(main())
