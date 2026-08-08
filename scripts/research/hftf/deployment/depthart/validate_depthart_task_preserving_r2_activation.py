#!/usr/bin/env python3
"""Validate a DepthART task-preserving R2 manifest before outcome access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_deployment_r2_protocol_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_deployment_r2_activation_manifest_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2"
PRIOR_TERMINAL = (
    "CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED"
)
PREPARED_STATUS = "PREPARED_NOT_ACTIVATED"
SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require_sha256(value: Any, name: str) -> None:
    _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
             f"{name} must be a 64-character SHA-256")


def validate(protocol: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema mismatch")
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id mismatch")
    _require(protocol.get("immutable_prior_terminal") == PRIOR_TERMINAL,
             "strict G4-D terminal changed")
    _require(
        protocol.get("status")
        == "PROTOCOL_FROZEN_EXECUTION_NOT_ACTIVATED_NO_OUTCOME_ACCESSED",
        "protocol is not frozen in the pre-outcome state",
    )

    _require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "manifest protocol id mismatch")
    _require(manifest.get("status") == PREPARED_STATUS, "manifest must remain unactivated")
    _require(manifest.get("execution_authorized") is False,
             "this validator cannot authorize execution")
    _require(manifest.get("outcome_access") == "NONE", "outcome access must remain NONE")
    for forbidden in ("results", "metrics", "terminal", "quality_passed"):
        _require(forbidden not in manifest, f"pre-outcome manifest contains forbidden key: {forbidden}")

    implementation = manifest.get("implementation")
    _require(isinstance(implementation, dict), "implementation must be an object")
    _require(implementation.get("candidate_count") == 1, "exactly one candidate is required")
    allowed = set(protocol["candidate"]["allowed_representation_families"])
    _require(implementation.get("candidate_representation_family") in allowed,
             "candidate representation is outside the frozen allowlist")
    for name in (
        "reference_checkpoint_sha256",
        "reference_graph_sha256",
        "candidate_graph_sha256",
        "task_postprocess_sha256",
        "runtime_config_sha256",
    ):
        _require_sha256(implementation.get(name), f"implementation.{name}")
    _require(
        implementation["reference_graph_sha256"].upper()
        != implementation["candidate_graph_sha256"].upper(),
        "reference and candidate graph identities must differ",
    )
    _require(implementation.get("task_postprocess_identical") is True,
             "reference and candidate must share the frozen task postprocess")

    cohort = manifest.get("cohort")
    _require(isinstance(cohort, dict), "cohort must be an object")
    _require(cohort.get("role") == protocol["cohort"]["required_role"],
             "cohort role must be SEALED_UNSEEN")
    _require(cohort.get("outcome_files_registered_not_opened") is True,
             "outcome files must be registered but unopened")
    _require_sha256(cohort.get("manifest_sha256"), "cohort.manifest_sha256")
    for key in ("cohort_id", "provenance"):
        _require(isinstance(cohort.get(key), str) and bool(cohort[key].strip()),
                 f"cohort.{key} must be non-empty")
    for key in ("parent_ids", "session_ids"):
        values = cohort.get(key)
        _require(isinstance(values, list) and bool(values), f"cohort.{key} must be non-empty")
        _require(all(isinstance(value, str) and value.strip() for value in values),
                 f"cohort.{key} contains an invalid id")
        _require(len(values) == len(set(values)), f"cohort.{key} contains duplicates")
    exclusions = set(cohort.get("excluded_cohort_ids", []))
    required_exclusions = set(protocol["cohort"]["required_excluded_cohort_ids"])
    _require(required_exclusions.issubset(exclusions), "required prior cohorts are not excluded")

    _require(manifest.get("gates") == protocol.get("gates"),
             "manifest gates differ from the frozen protocol")
    return {
        "schema": "blindassist_depthart_task_preserving_deployment_r2_preoutcome_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PRE_OUTCOME_CONTRACT_VALID_EXECUTION_NOT_ACTIVATED",
        "checks": {
            "strict_g4d_terminal_immutable": True,
            "single_candidate_frozen": True,
            "reference_candidate_and_postprocess_bound": True,
            "sealed_parent_session_cohort_declared": True,
            "consumed_and_synthetic_cohorts_excluded": True,
            "task_gates_exactly_match_protocol": True,
            "outcome_access_none": True,
        },
        "authority": "PRE_OUTCOME_CONTRACT_ONLY",
        "execution_authorized": False,
        "next_action": "EXPLICIT_USER_ACTIVATION_REQUIRED_BEFORE_OUTCOME_ACCESS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = validate(_load_json(args.protocol), _load_json(args.manifest))
    receipt["protocol_sha256"] = _sha256(args.protocol)
    receipt["manifest_sha256"] = _sha256(args.manifest)
    rendered = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
