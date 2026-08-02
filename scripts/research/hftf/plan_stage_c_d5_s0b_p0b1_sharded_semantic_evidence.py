#!/usr/bin/env python3
"""Fail-closed draft skeleton for HFTF D5-S0B P0B.1 sharded evidence.

This module freezes names, schemas, terminals, capacity constants, and closed
artifact sets only.  It is deliberately not executable: implementation and
test receipts are not yet bound, and no source blob or canonical artifact root
may be opened by this draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


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
