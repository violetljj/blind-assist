#!/usr/bin/env python3
"""Validate the RGB algorithm canary R0 design without reading algorithm outcomes."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "blindassist.research_protocol.v1"
MANIFEST_SCHEMA = "rcle.rgb_algorithm_canary.data_role_manifest.v1"
PROTOCOL_ID = "RCLE-PHASE-B-RGB-ALGORITHM-CANARY-R0"
FIXTURE_SCHEMA = "rcle.rgb_algorithm_canary.synthetic_validator_fixture.v1"
HEX64 = set("0123456789abcdef")

EXPECTED_ROLE_SETS = {
    "tum-fr2-rpy-window-0": {
        "CANARY",
        "BURNED",
        "GEOMETRY_SELECTED",
        "ROTATION_STRESS",
    },
    "tum-fr2-rpy-window-3": {
        "CANARY",
        "BURNED",
        "GEOMETRY_SELECTED",
        "ROTATION_STRESS",
    },
    "tum-fr2-rpy-window-6": {
        "CANARY",
        "BURNED",
        "GEOMETRY_SELECTED",
        "ROTATION_STRESS",
    },
    "tum-fr2-rpy-window-4": {
        "ABSTENTION_OR_INTERFACE_STRESS_ONLY",
        "BURNED",
        "GEOMETRY_SELECTED",
    },
    "bonn-frozen-cohort": {"REGRESSION_OR_COUNTEREXAMPLE_ONLY"},
    "phase-a-synthetic-r1": {"CALIBRATION_OR_FIXTURE_ONLY"},
    "icl-nuim-candidates": {"NOT_ADMITTED", "RESERVED"},
    "eth3d-candidates": {"NOT_ADMITTED", "RESERVED"},
}
EXPECTED_PARTITION_ORDER = list(EXPECTED_ROLE_SETS)
PAIR_IDENTITY_SCHEMA = (
    "Canonical JSON array sorted by pair_index; object keys are "
    "current_depth_timestamp,current_rgb_timestamp,dt_s,pair_index,"
    "previous_depth_timestamp,previous_rgb_timestamp,window_index; UTF-8, "
    "sort_keys=true, separators=(',',':')."
)
GEOMETRY_PAIR_LEDGER_SHA256 = (
    "fa68672ae208c57ec02dca3e7c80fc006c0090e63d011ed22dafe879a2f8d0b1"
)
TUM_IDENTITY_BASIS = (
    "Official TGZ SHA-256 plus the integer-Unix-second, non-overlapping 10 s "
    "window rule bound by the source-native geometry audit."
)
EXPECTED_PARTITION_POLICY = {
    "tum-fr2-rpy-window-0": {
        "content_identity": "sha256:3a35b7999af8631b6421ed9087a5325ffea1564c23d52ba640f0c239af62b51f#fixed-window-0",
        "independence_group": "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799",
        "access_level": "GEOMETRY_ONLY",
        "content_identity_status": "BOUND",
        "identity_basis": TUM_IDENTITY_BASIS,
        "future_execution_access": "FULL_ONLY_AFTER_SEPARATE_IMPLEMENTATION_AUTHORIZATION",
        "reuse_policy": "NEVER_CONFIRMATION_FOR_THIS_CLAIM; MAY_REMAIN_CANARY_OR_REGRESSION",
        "window_start_unix_s": "1311867719",
        "window_end_unix_s": "1311867729",
        "pair_identity_sha256": "b5d9d099a10197afa8f62f4f461ee23b44de64feeb0a9490160e78f52b2b2f64",
        "pair_identity_schema": PAIR_IDENTITY_SCHEMA,
        "pair_ledger_sha256": GEOMETRY_PAIR_LEDGER_SHA256,
    },
    "tum-fr2-rpy-window-3": {
        "content_identity": "sha256:3a35b7999af8631b6421ed9087a5325ffea1564c23d52ba640f0c239af62b51f#fixed-window-3",
        "independence_group": "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799",
        "access_level": "GEOMETRY_ONLY",
        "content_identity_status": "BOUND",
        "identity_basis": TUM_IDENTITY_BASIS,
        "future_execution_access": "FULL_ONLY_AFTER_SEPARATE_IMPLEMENTATION_AUTHORIZATION",
        "reuse_policy": "NEVER_CONFIRMATION_FOR_THIS_CLAIM; MAY_REMAIN_CANARY_OR_REGRESSION",
        "window_start_unix_s": "1311867749",
        "window_end_unix_s": "1311867759",
        "pair_identity_sha256": "85dbd89dc84c287809983807ecfb859a5dd8279b86e9f6dde9f2c0383331e9d2",
        "pair_identity_schema": PAIR_IDENTITY_SCHEMA,
        "pair_ledger_sha256": GEOMETRY_PAIR_LEDGER_SHA256,
    },
    "tum-fr2-rpy-window-6": {
        "content_identity": "sha256:3a35b7999af8631b6421ed9087a5325ffea1564c23d52ba640f0c239af62b51f#fixed-window-6",
        "independence_group": "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799",
        "access_level": "GEOMETRY_ONLY",
        "content_identity_status": "BOUND",
        "identity_basis": TUM_IDENTITY_BASIS,
        "future_execution_access": "FULL_ONLY_AFTER_SEPARATE_IMPLEMENTATION_AUTHORIZATION",
        "reuse_policy": "NEVER_CONFIRMATION_FOR_THIS_CLAIM; MAY_REMAIN_CANARY_OR_REGRESSION",
        "window_start_unix_s": "1311867779",
        "window_end_unix_s": "1311867789",
        "pair_identity_sha256": "2237c54db4fd0abac2db69565be8503e1fa2d26c47fc6e46580c954c39da3c8f",
        "pair_identity_schema": PAIR_IDENTITY_SCHEMA,
        "pair_ledger_sha256": GEOMETRY_PAIR_LEDGER_SHA256,
    },
    "tum-fr2-rpy-window-4": {
        "content_identity": "sha256:3a35b7999af8631b6421ed9087a5325ffea1564c23d52ba640f0c239af62b51f#fixed-window-4",
        "independence_group": "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799",
        "access_level": "GEOMETRY_ONLY",
        "content_identity_status": "BOUND",
        "identity_basis": TUM_IDENTITY_BASIS,
        "future_execution_access": "NO_SCIENTIFIC_SCORE; CONTRACT_ONLY_AFTER_SEPARATE_IMPLEMENTATION_AUTHORIZATION",
        "reuse_policy": "NEVER_SCIENTIFIC_PASS; NEVER_CONFIRMATION; MAY_TEST_ABSTENTION_OR_INTERFACE_ONLY",
        "window_start_unix_s": "1311867759",
        "window_end_unix_s": "1311867769",
        "pair_identity_sha256": "713e9b3cfe6ff632e2adcf35b57ba1600740e4dffa7b2df9d4e1ec6ff381383d",
        "pair_identity_schema": PAIR_IDENTITY_SCHEMA,
        "pair_ledger_sha256": GEOMETRY_PAIR_LEDGER_SHA256,
    },
    "bonn-frozen-cohort": {
        "content_identity": "sha256:513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e",
        "independence_group": "BONN_FROZEN_COHORT_513b770d",
        "access_level": "FULL",
        "content_identity_status": "BOUND",
        "identity_basis": (
            "Previously frozen six-sequence cohort identity and archive hashes; "
            "prior geometry outcome access is permanent."
        ),
        "future_execution_access": "NO_CANARY_SCIENTIFIC_ROLE",
        "reuse_policy": "REGRESSION_OR_COUNTEREXAMPLE_ONLY; NEVER_CONFIRMATION",
    },
    "phase-a-synthetic-r1": {
        "content_identity": "sha256:d5edb9528abfa6d79b973bddfed5f4234795262fb303258c9e1a9e2628ca2b15",
        "independence_group": "RCLE_PHASE_A_SYNTHETIC_GENERATOR_FAMILY",
        "access_level": "FULL",
        "content_identity_status": "BOUND",
        "identity_basis": (
            "Hash-bound Phase A R1 formal receipt; synthetic outcome is already accessed."
        ),
        "future_execution_access": "NO_REAL_DATA_SCIENTIFIC_UNIT",
        "reuse_policy": "CALIBRATION_OR_FIXTURE_ONLY; NEVER_REAL_DATA_CONFIRMATION",
    },
    "icl-nuim-candidates": {
        "content_identity": "UNBOUND_NOT_ADMITTED",
        "independence_group": "RESERVED_UNASSIGNED",
        "access_level": "NONE",
        "content_identity_status": "NOT_BOUND",
        "identity_basis": "No content has been admitted or frozen for this protocol.",
        "future_execution_access": "FORBIDDEN_IN_R0",
        "reuse_policy": "REQUIRES_SEPARATE_SOURCE_ADMISSION_AND_NEW_VERSION",
    },
    "eth3d-candidates": {
        "content_identity": "UNBOUND_NOT_ADMITTED",
        "independence_group": "RESERVED_UNASSIGNED",
        "access_level": "NONE",
        "content_identity_status": "NOT_BOUND",
        "identity_basis": "No content has been admitted or frozen for this protocol.",
        "future_execution_access": "FORBIDDEN_IN_R0",
        "reuse_policy": "REQUIRES_SEPARATE_SOURCE_ADMISSION_AND_NEW_VERSION",
    },
}

EXPECTED_SOURCE_BINDINGS = {
    "tum-source-audit-receipt": {
        "path": "artifacts.local/evidence/rcle_tum_fr2_rpy_geometry_audit_r0/receipt.json",
        "sha256": "1476cab6a6226ef3964eb07b1cbb5564b5848c08c0ea2a47317b30f09ab6396a",
        "required_json_fields": {
            "archive_sha256": "3a35b7999af8631b6421ed9087a5325ffea1564c23d52ba640f0c239af62b51f",
            "groundtruth_sha256": "e62810c806e3513ffea573780c1bd47c2c0d1d6169918d8f30f7b1ec818e755d",
            "result_sha256": "ae388f8e2f7decec8fadbd0a461ae50e2fe77da3d30bff72e3fa3d26f6c578b1",
        },
    },
    "geometry-canary-receipt": {
        "path": "artifacts.local/evidence/rcle_phase_b_real_data_geometry_canary_r0/formal_run_r0/receipt.json",
        "sha256": "b55417cebe7188cdbee40db02c36b06e58294acd8a9906edd6aba2ad00f211cd",
        "required_json_fields": {
            "archive_sha256": "3a35b7999af8631b6421ed9087a5325ffea1564c23d52ba640f0c239af62b51f",
            "pair_ledger_sha256": GEOMETRY_PAIR_LEDGER_SHA256,
            "pair_record_count": 1196,
        },
    },
    "geometry-canary-validation": {
        "path": "artifacts.local/evidence/rcle_phase_b_real_data_geometry_canary_r0/formal_run_r0/validation.json",
        "sha256": "c28c65accb9d19ee3cb409b3a391245cb29cf082ce6b3d46a0c8d39cce557154",
        "required_json_fields": {
            "producer_pair_record_count": 1196,
            "validator_pair_record_count": 1196,
            "terminal": "VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY",
        },
    },
    "bonn-burned-cohort-receipt": {
        "path": "artifacts.local/evidence/rcle_phase_b_bonn_b1/b1a_geometry_admission/receipt.json",
        "sha256": "f06ef1069f5f1182ee47477851a15c07c898e949c8c0cd609fc97598c8c0c7c1",
        "required_json_fields": {
            "cohort_identity_sha256": "513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e"
        },
    },
    "phase-a-r1-receipt": {
        "path": "artifacts.local/evidence/rcle_minimal_r1/formal_run_r1/receipt.json",
        "sha256": "d5edb9528abfa6d79b973bddfed5f4234795262fb303258c9e1a9e2628ca2b15",
        "required_json_fields": {},
    },
}

REQUIRED_PROGRESS_FIELDS = {
    "phase",
    "completed",
    "total",
    "throughput_per_s",
    "eta_s",
    "last_progress_at",
    "pid",
    "input_sha256",
    "implementation_sha256",
    "status",
}

FORBIDDEN_OUTCOME_KEYS = {
    "algorithm_outcome",
    "real_algorithm_outcome",
    "observed_algorithm_result",
    "approach_pass",
    "closing_retention_pass",
}
ALLOWED_OUTCOME_KEYS = {
    "outcome_access",
    "outcome_access_started",
    "algorithm_outcome_access_started",
    "algorithm_outcome_firewall",
    "forbidden_outcome_access",
    "scientific_outcome",
    "invalid_execution_scientific_outcome",
}
PASS_LIKE_KEY_RE = re.compile(
    r"^(observed|measured|actual)_|"
    r"(algorithm|approach|closing|mechanism).*(outcome|result|pass|score)$",
    re.IGNORECASE,
)
ARTIFACT_NAME_RE = re.compile(
    r"(claim|output|failure|receipt|activation|implementation[_-]?lock)",
    re.IGNORECASE,
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= HEX64
    )


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def _repo_path(repo_root: Path, relative: str) -> Path | None:
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return candidate


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_walk_keys(nested))
    return keys


def _outcome_field_violations(value: Any, prefix: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_lower in FORBIDDEN_OUTCOME_KEYS:
                violations.append(path)
            elif key_lower not in ALLOWED_OUTCOME_KEYS and PASS_LIKE_KEY_RE.search(
                key_text
            ):
                violations.append(path)
            if key_lower == "scientific_outcome" and nested != "NOT_RUN":
                violations.append(path)
            violations.extend(_outcome_field_violations(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            violations.extend(_outcome_field_violations(nested, f"{prefix}[{index}]"))
    return violations


def _inventory_firewall_violations(repo_root: Path) -> list[str]:
    """Inspect path names only; never open a suspected algorithm artifact."""

    violations: list[str] = []
    evidence_root = (
        repo_root / "artifacts.local/evidence/rcle_phase_b_rgb_algorithm_canary_r0"
    )
    if evidence_root.exists():
        violations.append(str(evidence_root.relative_to(repo_root)))
        for path in evidence_root.rglob("*"):
            violations.append(str(path.relative_to(repo_root)))
    evidence_parent = repo_root / "artifacts.local/evidence"
    if evidence_parent.is_dir():
        for path in evidence_parent.rglob("*"):
            relative_text = str(path.relative_to(evidence_parent)).lower()
            if (
                all(token in relative_text for token in ("rcle", "rgb", "algorithm"))
                and ARTIFACT_NAME_RE.search(path.name)
            ):
                violations.append(str(path.relative_to(repo_root)))

    implementation_root = (
        repo_root
        / "scripts/research/egomotion_compensated_looming/rgb_algorithm_canary_r0"
    )
    if implementation_root.is_dir():
        for path in implementation_root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if ARTIFACT_NAME_RE.search(name) and name not in {
                "validate_design_package.py",
            }:
                violations.append(str(path.relative_to(repo_root)))
            if re.search(r"(^run_|producer|formal[_-]?validator)", name, re.IGNORECASE):
                violations.append(str(path.relative_to(repo_root)))
    return sorted(set(violations))


def _parse_rfc3339_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.tzinfo == timezone.utc else None


def _canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _geometry_pair_identity_hashes(
    pair_ledger_path: Path,
) -> dict[int, tuple[int, int, int, str]]:
    """Recompute identity only from the already authorized geometry ledger."""

    fields = [
        "window_index",
        "pair_index",
        "previous_rgb_timestamp",
        "current_rgb_timestamp",
        "dt_s",
        "previous_depth_timestamp",
        "current_depth_timestamp",
    ]
    rows_by_window: dict[int, list[dict[str, Any]]] = {}
    with pair_ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            identity = {field: record[field] for field in fields}
            rows_by_window.setdefault(int(record["window_index"]), []).append(identity)
    result: dict[int, tuple[int, int, int, str]] = {}
    for window, rows in rows_by_window.items():
        rows.sort(key=lambda item: int(item["pair_index"]))
        encoded = json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        result[window] = (
            len(rows),
            int(rows[0]["pair_index"]),
            int(rows[-1]["pair_index"]),
            _sha256_bytes(encoded),
        )
    return result


def validate_design_objects(
    repo_root: Path,
    contract: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    """Validate static design bindings and fail closed without opening outcomes."""

    errors: list[str] = []
    if contract.get("schema_version") != SCHEMA:
        errors.append("CONTRACT_SCHEMA")
    if contract.get("protocol_id") != PROTOCOL_ID:
        errors.append("CONTRACT_PROTOCOL")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("MANIFEST_SCHEMA")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        errors.append("MANIFEST_PROTOCOL")

    for binding in contract.get("upstream_hash_bindings", []):
        if not isinstance(binding, dict):
            errors.append("UPSTREAM_BINDING_OBJECT")
            continue
        path = _repo_path(repo_root, str(binding.get("path", "")))
        expected = binding.get("sha256")
        if path is None or not path.is_file():
            errors.append(f"UPSTREAM_MISSING:{binding.get('id')}")
        elif not _is_hex64(expected) or _sha256_file(path) != expected:
            errors.append(f"UPSTREAM_HASH:{binding.get('id')}")

    for binding in contract.get("bound_design_documents", []):
        if not isinstance(binding, dict):
            errors.append("DESIGN_BINDING_OBJECT")
            continue
        path = _repo_path(repo_root, str(binding.get("path", "")))
        expected = binding.get("sha256")
        if path is None or not path.is_file():
            errors.append(f"DESIGN_MISSING:{binding.get('id')}")
        elif not _is_hex64(expected) or _sha256_file(path) != expected:
            errors.append(f"DESIGN_HASH:{binding.get('id')}")

    freeze = contract.get("freeze", {})
    result_model = contract.get("result_model", {})
    current_terminal = contract.get("current_design_terminal", {})
    if freeze.get("level") != "F1" or freeze.get("outcome_access_started") is not False:
        errors.append("OUTCOME_FREEZE")
    if result_model.get("execution_validity") != "NOT_RUN":
        errors.append("EXECUTION_PREWRITTEN")
    if result_model.get("scientific_outcome") != "NOT_RUN":
        errors.append("SCIENTIFIC_OUTCOME_PREWRITTEN")
    if current_terminal.get("maximum_authority") != "EXECUTION_NOT_AUTHORIZED":
        errors.append("AUTHORITY_ESCALATION")
    if (
        current_terminal.get("execution_readiness")
        != "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE"
    ):
        errors.append("APPROACH_HOLD_MISSING")

    if manifest.get("algorithm_outcome_access_started") is not False:
        errors.append("MANIFEST_OUTCOME_ACCESS")
    if manifest.get("confirmation_partition_reserved") is not True:
        errors.append("CONFIRMATION_NOT_RESERVED")
    if manifest.get("identity_closure_status") != "CLOSED_FOR_ALL_ADMITTED_CONTENT":
        errors.append("IDENTITY_NOT_CLOSED")

    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, list) or not source_bindings:
        errors.append("SOURCE_BINDINGS")
        source_bindings = []
    binding_inventory = {
        binding.get("id"): {
            "path": binding.get("path"),
            "sha256": binding.get("sha256"),
            "required_json_fields": binding.get("required_json_fields"),
        }
        for binding in source_bindings
        if isinstance(binding, dict) and isinstance(binding.get("id"), str)
    }
    if binding_inventory != EXPECTED_SOURCE_BINDINGS:
        errors.append("SOURCE_BINDING_INVENTORY")
    seen_binding_ids: set[str] = set()
    for binding in source_bindings:
        if not isinstance(binding, dict):
            errors.append("SOURCE_BINDING_OBJECT")
            continue
        binding_id = binding.get("id")
        if not isinstance(binding_id, str) or binding_id in seen_binding_ids:
            errors.append(f"SOURCE_BINDING_ID:{binding_id}")
            continue
        seen_binding_ids.add(binding_id)
        path = _repo_path(repo_root, str(binding.get("path", "")))
        if path is None or not path.is_file():
            errors.append(f"SOURCE_BINDING_MISSING:{binding_id}")
            continue
        if not _is_hex64(binding.get("sha256")):
            errors.append(f"SOURCE_BINDING_SHA_FORMAT:{binding_id}")
            continue
        if _sha256_file(path) != binding["sha256"]:
            errors.append(f"SOURCE_BINDING_SHA:{binding_id}")
            continue
        required_json_fields = binding.get("required_json_fields")
        if not isinstance(required_json_fields, dict):
            errors.append(f"SOURCE_BINDING_FIELDS:{binding_id}")
            continue
        if required_json_fields:
            try:
                payload = _load_object(path)
            except (OSError, ValueError, json.JSONDecodeError):
                errors.append(f"SOURCE_BINDING_JSON:{binding_id}")
                continue
            for field, expected in required_json_fields.items():
                if payload.get(field) != expected:
                    errors.append(f"SOURCE_BINDING_VALUE:{binding_id}:{field}")

    partitions = manifest.get("partitions")
    if not isinstance(partitions, list):
        errors.append("PARTITIONS")
        partitions = []
    partition_order = [
        item.get("id") if isinstance(item, dict) else None for item in partitions
    ]
    if partition_order != EXPECTED_PARTITION_ORDER:
        errors.append("PARTITION_ORDER")
    seen_ids: set[str] = set()
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for partition in partitions:
        if not isinstance(partition, dict):
            errors.append("PARTITION_OBJECT")
            continue
        partition_id = partition.get("id")
        if not isinstance(partition_id, str) or partition_id in seen_ids:
            errors.append(f"PARTITION_ID:{partition_id}")
            continue
        seen_ids.add(partition_id)
        manifest_by_id[partition_id] = partition
        required = {
            "source_identity",
            "content_identity",
            "content_identity_status",
            "identity_basis",
            "independence_group",
            "ancestry",
            "roles",
            "access_level",
            "future_execution_access",
            "reuse_policy",
        }
        if partition_id.startswith("tum-fr2-rpy-window-"):
            required |= {
                "window_start_unix_s",
                "window_end_unix_s",
                "pair_identity_schema",
                "pair_identity_sha256",
                "pair_ledger_sha256",
                "candidate_pair_count",
                "first_pair_index",
                "last_pair_index",
            }
        missing = sorted(required - set(partition))
        if missing:
            errors.append(f"PARTITION_FIELDS:{partition_id}:{','.join(missing)}")
        if not isinstance(partition.get("ancestry"), list):
            errors.append(f"ANCESTRY:{partition_id}")
        roles = partition.get("roles")
        if not isinstance(roles, list):
            errors.append(f"ROLES:{partition_id}")
            roles = []
        role_set = set(roles)
        if partition_id in EXPECTED_ROLE_SETS and role_set != EXPECTED_ROLE_SETS[partition_id]:
            errors.append(f"ROLE_SET:{partition_id}")
        if any("CONFIRMATION" in str(role).upper() for role in roles):
            errors.append(f"CONFIRMATION_ROLE_OVERLAP:{partition_id}")
        expected_policy = EXPECTED_PARTITION_POLICY.get(partition_id)
        if expected_policy:
            for field, expected in expected_policy.items():
                if partition.get(field) != expected:
                    errors.append(f"PARTITION_POLICY:{partition_id}:{field}")
        expected_ancestry = {
            "tum-fr2-rpy-window-0": [
                "TUM_RGBD_FREIBURG2_RPY_OFFICIAL",
                "TUM_FR2_RPY_ARCHIVE_3a35b799",
                "TUM_FR2_RPY_FIXED_WINDOWS_R0",
                "REAL_DATA_GEOMETRY_CANARY_R0",
            ],
            "tum-fr2-rpy-window-3": [
                "TUM_RGBD_FREIBURG2_RPY_OFFICIAL",
                "TUM_FR2_RPY_ARCHIVE_3a35b799",
                "TUM_FR2_RPY_FIXED_WINDOWS_R0",
                "REAL_DATA_GEOMETRY_CANARY_R0",
            ],
            "tum-fr2-rpy-window-6": [
                "TUM_RGBD_FREIBURG2_RPY_OFFICIAL",
                "TUM_FR2_RPY_ARCHIVE_3a35b799",
                "TUM_FR2_RPY_FIXED_WINDOWS_R0",
                "REAL_DATA_GEOMETRY_CANARY_R0",
            ],
            "tum-fr2-rpy-window-4": [
                "TUM_RGBD_FREIBURG2_RPY_OFFICIAL",
                "TUM_FR2_RPY_ARCHIVE_3a35b799",
                "TUM_FR2_RPY_FIXED_WINDOWS_R0",
                "REAL_DATA_GEOMETRY_CANARY_R0",
            ],
            "bonn-frozen-cohort": [
                "RCLE_PHASE_B_BONN_B0_R1",
                "RCLE_PHASE_B_BONN_B1A_CONSUMED_CLOSED",
            ],
            "phase-a-synthetic-r1": [
                "RCLE_MINIMAL_PHASE_A_R0",
                "RCLE_MINIMAL_PHASE_A_COVERAGE_REVISION_R1",
            ],
            "icl-nuim-candidates": [],
            "eth3d-candidates": [],
        }.get(partition_id)
        if expected_ancestry is not None and partition.get("ancestry") != expected_ancestry:
            errors.append(f"PARTITION_ANCESTRY:{partition_id}")
        if (
            partition.get("content_identity_status") == "BOUND"
            and not str(partition.get("content_identity", "")).startswith("sha256:")
        ):
            errors.append(f"BOUND_IDENTITY:{partition_id}")
        if (
            partition.get("content_identity_status") == "BOUND"
            and not partition.get("ancestry")
        ):
            errors.append(f"BOUND_ANCESTRY:{partition_id}")

    if seen_ids != set(EXPECTED_ROLE_SETS):
        errors.append("PARTITION_INVENTORY")

    contract_partitions = contract.get("data_partitions")
    if not isinstance(contract_partitions, list):
        errors.append("CONTRACT_DATA_PARTITIONS")
        contract_partitions = []
    contract_id_map = {
        "tum-fr2-rpy-window-4-interface-stress": "tum-fr2-rpy-window-4"
    }
    for contract_partition in contract_partitions:
        if not isinstance(contract_partition, dict):
            errors.append("CONTRACT_PARTITION_OBJECT")
            continue
        contract_id = str(contract_partition.get("id", ""))
        manifest_id = contract_id_map.get(contract_id, contract_id)
        manifest_partition = manifest_by_id.get(manifest_id)
        if manifest_partition is None:
            errors.append(f"CONTRACT_MANIFEST_PARTITION:{contract_id}")
            continue
        cross_fields = {
            "source_identity": "source_identity",
            "content_identity": "content_identity",
            "independence_group": "independence_group",
            "ancestry": "ancestry",
            "outcome_access": "access_level",
            "reuse_policy": "reuse_policy",
            "window_start_unix_s": "window_start_unix_s",
            "window_end_unix_s": "window_end_unix_s",
            "pair_identity_sha256": "pair_identity_sha256",
            "pair_ledger_sha256": "pair_ledger_sha256",
            "candidate_pair_count": "candidate_pair_count",
        }
        for contract_field, manifest_field in cross_fields.items():
            if contract_partition.get(contract_field) != manifest_partition.get(
                manifest_field
            ):
                errors.append(
                    f"CONTRACT_MANIFEST_FIELD:{contract_id}:{contract_field}"
                )

    geometry_ledger = _repo_path(
        repo_root,
        "artifacts.local/evidence/rcle_phase_b_real_data_geometry_canary_r0/"
        "formal_run_r0/pair_ledger.jsonl",
    )
    if geometry_ledger is None or not geometry_ledger.is_file():
        errors.append("GEOMETRY_PAIR_LEDGER_MISSING")
    elif (
        _sha256_file(geometry_ledger)
        != GEOMETRY_PAIR_LEDGER_SHA256
    ):
        errors.append("GEOMETRY_PAIR_LEDGER_HASH")
    else:
        try:
            recomputed = _geometry_pair_identity_hashes(geometry_ledger)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            errors.append("GEOMETRY_PAIR_IDENTITY_RECOMPUTE")
            recomputed = {}
        for window in (0, 3, 4, 6):
            partition = manifest_by_id.get(f"tum-fr2-rpy-window-{window}")
            actual = recomputed.get(window)
            if partition is None or actual is None:
                errors.append(f"GEOMETRY_PAIR_IDENTITY_MISSING:{window}")
                continue
            expected = (
                partition.get("candidate_pair_count"),
                partition.get("first_pair_index"),
                partition.get("last_pair_index"),
                partition.get("pair_identity_sha256"),
            )
            if actual != expected:
                errors.append(f"GEOMETRY_PAIR_IDENTITY_MISMATCH:{window}")

    reservation = manifest.get("confirmation_reservation", {})
    if reservation.get("assigned_content_identities") != []:
        errors.append("CONFIRMATION_CONTENT_ASSIGNED")
    forbidden_groups = set(reservation.get("forbidden_independence_groups", []))
    required_forbidden = {
        "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799",
        "BONN_FROZEN_COHORT_513b770d",
        "RCLE_PHASE_A_SYNTHETIC_GENERATOR_FAMILY",
    }
    if not required_forbidden.issubset(forbidden_groups):
        errors.append("CONFIRMATION_ANCESTRY_FIREWALL")

    missing_roles = manifest.get("missing_roles", [])
    if not any(
        isinstance(item, dict)
        and item.get("role") == "REAL_POSITIVE_APPROACH"
        and item.get("status") == "ABSENT"
        and item.get("terminal_if_unresolved")
        == "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE"
        for item in missing_roles
    ):
        errors.append("REAL_APPROACH_GAP_NOT_DECLARED")

    numeric_fields = {
        "unit",
        "rationale",
        "calibration_source",
        "sensitivity_plan",
        "revision_policy",
    }
    for constraint in contract.get("constraints", []):
        if not isinstance(constraint, dict):
            errors.append("CONSTRAINT_OBJECT")
            continue
        if isinstance(constraint.get("threshold"), (int, float)):
            missing = sorted(
                field
                for field in numeric_fields
                if not isinstance(constraint.get(field), str)
                or not constraint.get(field).strip()
            )
            if missing:
                errors.append(
                    f"NUMERIC_JUSTIFICATION:{constraint.get('id')}:{','.join(missing)}"
                )

    algorithm_spec = contract.get("algorithm_specification")
    if not isinstance(algorithm_spec, dict):
        errors.append("ALGORITHM_SPECIFICATION")
        algorithm_spec = {}
    if (
        algorithm_spec.get("implementation_choice_after_preregistration")
        != "FORBIDDEN"
    ):
        errors.append("ALGORITHM_CHOICE_NOT_FROZEN")
    normative_bindings = algorithm_spec.get("normative_source_bindings")
    if not isinstance(normative_bindings, list) or len(normative_bindings) < 6:
        errors.append("ALGORITHM_NORMATIVE_BINDINGS")
        normative_bindings = []
    for index, binding in enumerate(normative_bindings):
        if not isinstance(binding, dict):
            errors.append(f"ALGORITHM_BINDING_OBJECT:{index}")
            continue
        path = _repo_path(repo_root, str(binding.get("path", "")))
        expected = binding.get("sha256")
        if path is None or not path.is_file():
            errors.append(f"ALGORITHM_BINDING_MISSING:{index}")
        elif not _is_hex64(expected) or _sha256_file(path) != expected:
            errors.append(f"ALGORITHM_BINDING_HASH:{index}")

    validator_contract = contract.get("independent_validator_contract", {})
    if validator_contract.get("producer_import_forbidden") is not True:
        errors.append("VALIDATOR_MAY_IMPORT_PRODUCER")
    if validator_contract.get("producer_summary_trusted") is not False:
        errors.append("VALIDATOR_TRUSTS_SUMMARY")
    required_mutations = {
        "data role overlap",
        "algorithm outcome leakage",
        "missing field",
        "record order change",
        "numeric drift",
        "forged summary",
        "cache member tamper",
        "missing progress contract",
        "invalid progress timestamp, phase, status, PID, ETA, or freshness",
    }
    if not required_mutations.issubset(
        set(validator_contract.get("mutation_tests_required", []))
    ):
        errors.append("MUTATION_CONTRACT_INCOMPLETE")

    performance = contract.get("performance_qualification", {})
    if performance.get("status") != "NOT_RUN":
        errors.append("PERFORMANCE_PREWRITTEN")
    if performance.get("formal_claim_before_qualification") is not False:
        errors.append("CLAIM_BEFORE_PERFORMANCE")
    if performance.get("minimum_real_progress_samples", 0) < 2:
        errors.append("PROGRESS_SAMPLE_REQUIREMENT")
    if not REQUIRED_PROGRESS_FIELDS.issubset(set(performance.get("progress_fields", []))):
        errors.append("PROGRESS_FIELDS")

    firewall = contract.get("algorithm_outcome_firewall", {})
    for relative in firewall.get("current_expected_absent_paths", []):
        path = _repo_path(repo_root, str(relative))
        if path is None:
            errors.append(f"FIREWALL_PATH_SCOPE:{relative}")
        elif path.exists():
            # Deliberately do not open or inspect a suspected outcome artifact.
            errors.append(f"FIREWALL_PATH_EXISTS:{relative}")

    for relative in _inventory_firewall_violations(repo_root):
        errors.append(f"FIREWALL_FILENAME:{relative}")
    for path in _outcome_field_violations(contract, "contract"):
        errors.append(f"FORBIDDEN_OUTCOME_FIELD:{path}")
    for path in _outcome_field_violations(manifest, "manifest"):
        errors.append(f"FORBIDDEN_OUTCOME_FIELD:{path}")

    return sorted(set(errors))


def validate_synthetic_fixture(fixture: dict[str, Any]) -> list[str]:
    """Exercise future validator invariants using synthetic bytes only."""

    errors: list[str] = []
    if fixture.get("schema_version") != FIXTURE_SCHEMA:
        errors.append("FIXTURE_SCHEMA")
    if fixture.get("synthetic_only") is not True:
        errors.append("FIXTURE_NOT_SYNTHETIC")
    for path in _outcome_field_violations(fixture, "fixture"):
        errors.append(f"FIXTURE_OUTCOME_LEAKAGE:{path}")

    cache = fixture.get("cache_manifest")
    if not isinstance(cache, dict):
        errors.append("CACHE_MANIFEST")
        cache = {}
    members = cache.get("members")
    if not isinstance(members, list):
        errors.append("CACHE_MEMBERS")
        members = []
    paths: list[str] = []
    ordinals: list[int] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            errors.append(f"CACHE_MEMBER_OBJECT:{index}")
            continue
        try:
            content = base64.b64decode(member["synthetic_content_b64"], validate=True)
        except (KeyError, ValueError):
            errors.append(f"CACHE_MEMBER_CONTENT:{index}")
            continue
        paths.append(str(member.get("member_path", "")))
        ordinal = member.get("archive_ordinal")
        if not isinstance(ordinal, int):
            errors.append(f"CACHE_MEMBER_ORDINAL:{index}")
        else:
            ordinals.append(ordinal)
        if member.get("size_bytes") != len(content):
            errors.append(f"CACHE_MEMBER_SIZE:{index}")
        if member.get("sha256") != _sha256_bytes(content):
            errors.append(f"CACHE_MEMBER_HASH:{index}")
    if ordinals != list(range(len(members))):
        errors.append("CACHE_ARCHIVE_ORDER")
    if len(paths) != len(set(paths)) or any(
        not path or path.startswith(("/", "\\")) or ".." in Path(path).parts
        for path in paths
    ):
        errors.append("CACHE_MEMBER_PATH")
    if cache.get("member_count") != len(members):
        errors.append("CACHE_MEMBER_COUNT")
    if cache.get("manifest_sha256") != _canonical_manifest_hash(cache):
        errors.append("CACHE_MANIFEST_HASH")

    records = fixture.get("pair_records")
    if not isinstance(records, list):
        errors.append("PAIR_RECORDS")
        records = []
    identities: list[tuple[int, int]] = []
    scores: list[float] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"PAIR_OBJECT:{index}")
            continue
        required = {
            "window_index",
            "pair_index",
            "raw_rotation_leakage_s_inverse",
            "compensated_rotation_leakage_s_inverse",
            "paired_leakage_reduction_s_inverse",
            "raw_hex",
            "compensated_hex",
            "score_hex",
            "evaluable",
            "abstention_reason",
        }
        missing = sorted(required - set(record))
        if missing:
            errors.append(f"PAIR_FIELDS:{index}:{','.join(missing)}")
            continue
        identities.append((record["window_index"], record["pair_index"]))
        if record["evaluable"] is True:
            raw = record["raw_rotation_leakage_s_inverse"]
            compensated = record["compensated_rotation_leakage_s_inverse"]
            score = record["paired_leakage_reduction_s_inverse"]
            if not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (raw, compensated, score)
            ):
                errors.append(f"PAIR_NUMERIC:{index}")
                continue
            expected = float(raw) - float(compensated)
            if float(score).hex() != expected.hex():
                errors.append(f"PAIR_NUMERIC_DRIFT:{index}")
            if record["raw_hex"] != float(raw).hex():
                errors.append(f"PAIR_RAW_HEX:{index}")
            if record["compensated_hex"] != float(compensated).hex():
                errors.append(f"PAIR_COMP_HEX:{index}")
            if record["score_hex"] != float(score).hex():
                errors.append(f"PAIR_SCORE_HEX:{index}")
            scores.append(float(score))
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        errors.append("PAIR_ORDER_OR_DUPLICATE")

    summary = fixture.get("summary")
    if not isinstance(summary, dict):
        errors.append("SUMMARY")
    else:
        expected_count = len(scores)
        expected_sum = sum(scores)
        if summary.get("evaluable_pair_count") != expected_count:
            errors.append("SUMMARY_COUNT_FORGED")
        if not isinstance(summary.get("score_sum"), (int, float)) or float(
            summary["score_sum"]
        ).hex() != float(expected_sum).hex():
            errors.append("SUMMARY_NUMERIC_FORGED")

    progress = fixture.get("progress_samples")
    if not isinstance(progress, list) or len(progress) < 2:
        errors.append("PROGRESS_SAMPLES")
        progress = []
    previous_completed = -1
    previous_time: datetime | None = None
    bound_input_hash: str | None = None
    bound_implementation_hash: str | None = None
    allowed_phases = {
        "CACHE_MATERIALIZATION",
        "PRODUCER",
        "INDEPENDENT_VALIDATOR",
        "TERMINAL",
    }
    allowed_statuses = {
        "STARTING",
        "RUNNING",
        "VALIDATING",
        "VALID",
        "INVALID",
        "FAILED",
    }
    for index, sample in enumerate(progress):
        if not isinstance(sample, dict):
            errors.append(f"PROGRESS_OBJECT:{index}")
            continue
        missing = REQUIRED_PROGRESS_FIELDS - set(sample)
        if missing:
            errors.append(f"PROGRESS_FIELDS:{index}:{','.join(sorted(missing))}")
            continue
        completed = sample.get("completed")
        total = sample.get("total")
        if not isinstance(completed, int) or not isinstance(total, int):
            errors.append(f"PROGRESS_COUNT_TYPE:{index}")
        elif completed <= previous_completed or completed > total:
            errors.append(f"PROGRESS_NOT_ADVANCING:{index}")
        previous_completed = completed if isinstance(completed, int) else previous_completed
        phase = sample.get("phase")
        status = sample.get("status")
        if phase not in allowed_phases:
            errors.append(f"PROGRESS_PHASE:{index}")
        if status not in allowed_statuses:
            errors.append(f"PROGRESS_STATUS:{index}")
        pid = sample.get("pid")
        if status in {"STARTING", "RUNNING", "VALIDATING"} and (
            not isinstance(pid, int) or pid <= 0
        ):
            errors.append(f"PROGRESS_PID:{index}")
        throughput = sample.get("throughput_per_s")
        if not isinstance(throughput, (int, float)) or not math.isfinite(
            float(throughput)
        ) or float(throughput) < 0:
            errors.append(f"PROGRESS_THROUGHPUT:{index}")
        eta = sample.get("eta_s")
        if completed != total and (
            not isinstance(eta, (int, float))
            or not math.isfinite(float(eta))
            or float(eta) < 0
        ):
            errors.append(f"PROGRESS_ETA:{index}")
        timestamp = _parse_rfc3339_utc(sample.get("last_progress_at"))
        if timestamp is None:
            errors.append(f"PROGRESS_TIMESTAMP:{index}")
        elif previous_time is not None:
            delta_s = (timestamp - previous_time).total_seconds()
            if delta_s <= 0 or delta_s > 120:
                errors.append(f"PROGRESS_FRESHNESS:{index}")
        if timestamp is not None:
            previous_time = timestamp
        input_hash = sample.get("input_sha256")
        implementation_hash = sample.get("implementation_sha256")
        if not _is_hex64(input_hash):
            errors.append(f"PROGRESS_INPUT_HASH:{index}")
        elif bound_input_hash is None:
            bound_input_hash = input_hash
        elif input_hash != bound_input_hash:
            errors.append(f"PROGRESS_INPUT_HASH_DRIFT:{index}")
        if not _is_hex64(implementation_hash):
            errors.append(f"PROGRESS_IMPLEMENTATION_HASH:{index}")
        elif bound_implementation_hash is None:
            bound_implementation_hash = implementation_hash
        elif implementation_hash != bound_implementation_hash:
            errors.append(f"PROGRESS_IMPLEMENTATION_HASH_DRIFT:{index}")

    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--synthetic-fixture", type=Path)
    args = parser.parse_args()

    try:
        repo_root = args.repo_root.resolve()
        contract = _load_object(args.contract.resolve())
        manifest = _load_object(args.manifest.resolve())
        errors = validate_design_objects(repo_root, contract, manifest)
        if args.synthetic_fixture:
            fixture = _load_object(args.synthetic_fixture.resolve())
            errors.extend(validate_synthetic_fixture(fixture))
        errors = sorted(set(errors))
        payload = {
            "status": "VALID" if not errors else "INVALID",
            "protocol_id": PROTOCOL_ID,
            "algorithm_outcome_content_read": False,
            "errors": errors,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        payload = {
            "status": "INVALID",
            "protocol_id": PROTOCOL_ID,
            "algorithm_outcome_content_read": False,
            "errors": [f"LOAD:{type(error).__name__}:{error}"],
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
