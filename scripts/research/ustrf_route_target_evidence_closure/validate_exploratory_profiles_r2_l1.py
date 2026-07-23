#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from exploratory_profiles_r2_l1 import (
    TERMINAL_SCHEMA,
    TERMINAL_STATES,
    collect_verified_input_artifacts,
    implementation_bindings,
    identity,
    load_and_verify_config,
    load_json,
    load_route_map,
    sha256_file,
    stable_slug,
    validate_compact_ledger,
    validate_exhausted_resource_guard_receipt,
    validate_mask_contract,
    validate_profile_contract,
)


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def forbidden_key_fragments(value: Any, path: str = "$") -> list[str]:
    forbidden = ("winner", "rank", "best_candidate", "tie_break", "promotion")
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(fragment in lowered for fragment in forbidden):
                found.append(f"{path}.{key}")
            found.extend(forbidden_key_fragments(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_key_fragments(child, f"{path}[{index}]"))
    return found


def _is_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValidationError(f"unsupported schema type: {expected}")


def validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str = "$",
) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        require(reference.startswith("#/$defs/"), f"unsupported schema ref: {reference}")
        name = reference.rsplit("/", 1)[-1]
        validate_json_schema(value, root_schema["$defs"][name], root_schema, path)
        return
    if "const" in schema:
        require(value == schema["const"], f"schema const mismatch at {path}")
    if "enum" in schema:
        require(value in schema["enum"], f"schema enum mismatch at {path}")
    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        require(
            any(_is_type(value, item) for item in types),
            f"schema type mismatch at {path}",
        )
    if isinstance(value, str):
        if "minLength" in schema:
            require(len(value) >= schema["minLength"], f"schema minLength at {path}")
        if "pattern" in schema:
            require(re.fullmatch(schema["pattern"], value) is not None, f"schema pattern at {path}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema:
            require(value >= schema["minimum"], f"schema minimum at {path}")
        if "maximum" in schema:
            require(value <= schema["maximum"], f"schema maximum at {path}")
    if isinstance(value, list):
        if "minItems" in schema:
            require(len(value) >= schema["minItems"], f"schema minItems at {path}")
        if "maxItems" in schema:
            require(len(value) <= schema["maxItems"], f"schema maxItems at {path}")
        if "items" in schema:
            for index, child in enumerate(value):
                validate_json_schema(
                    child, schema["items"], root_schema, f"{path}[{index}]"
                )
    if isinstance(value, dict):
        required_keys = schema.get("required", [])
        for key in required_keys:
            require(key in value, f"schema required field missing at {path}.{key}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            require(not extras, f"schema additional properties at {path}: {sorted(extras)}")
        for key, child_schema in properties.items():
            if key in value:
                validate_json_schema(
                    value[key], child_schema, root_schema, f"{path}.{key}"
                )
    for child_schema in schema.get("allOf", []):
        validate_json_schema(value, child_schema, root_schema, path)
    if "if" in schema:
        try:
            validate_json_schema(value, schema["if"], root_schema, path)
        except ValidationError:
            condition = False
        else:
            condition = True
        branch = schema.get("then" if condition else "else")
        if branch is not None:
            validate_json_schema(value, branch, root_schema, path)


def validate(repo: Path, config_path: Path, receipt_path: Path) -> dict[str, Any]:
    config, expected_bindings = load_and_verify_config(repo, config_path)
    mask = load_json(repo / config["parent_bindings"]["eligibility_mask"]["path"])
    groups, resets = validate_mask_contract(config, mask)
    route_map = load_route_map(config, repo)
    receipt = load_json(receipt_path)
    schema_path = repo / config["parent_bindings"]["terminal_receipt_schema"]["path"]
    schema = load_json(schema_path)
    validate_json_schema(receipt, schema, schema)
    require(receipt.get("schema") == TERMINAL_SCHEMA, "terminal receipt schema mismatch")
    require(receipt.get("stage") == "R2-L1E", "stage mismatch")
    require(receipt.get("terminal_state") in TERMINAL_STATES, "illegal terminal state")
    require(receipt.get("discontinuity_resets") == resets, "discontinuity reset drift")
    require(not forbidden_key_fragments(receipt), "selection/comparison field present")
    claim = receipt["claim_boundary"]
    require(all(value is False for value in claim.values()), "claim boundary opened")
    expected_bindings.update(implementation_bindings(repo))
    for key, value in expected_bindings.items():
        require(receipt["bindings"].get(key) == value, f"binding mismatch: {key}")
    require(
        set(receipt["bindings"]) == set(expected_bindings),
        "terminal binding inventory drift",
    )
    gaps = receipt["gap_matrix"]
    require(len(gaps) == len(groups) == 41, "gap matrix ledger count mismatch")
    expected_missing_frames = 0
    verified_ledgers = 0
    verified_frames = 0
    for (descriptor, rows), gap in zip(groups, gaps, strict=True):
        require(
            (gap["source_id"], gap["sequence_id"])
            == (descriptor["source_id"], descriptor["sequence_id"]),
            "gap matrix sequence order mismatch",
        )
        require(gap["frame_mask_sha256"] == descriptor["frame_mask_sha256"], "frame mask hash drift")
        ledger = (
            repo
            / config["resource_guards"]["output_root"]
            / "detector-ledgers"
            / (
                stable_slug(descriptor["source_id"], descriptor["sequence_id"])
                + ".json"
            )
        )
        successor = ledger.with_name(ledger.stem + ".successor-receipt.json")
        complete = validate_compact_ledger(ledger, successor, descriptor, rows)
        require(gap["canonical_detector_frame_count"] == (len(rows) if complete else 0), "detector coverage drift")
        require(gap["causal_route_frame_count"] == len(rows), "causal route is incomplete")
        require(gap["capture_timestamp_frame_count"] == len(rows), "capture timestamp is incomplete")
        if complete:
            require(not gap["missing_fields"], "complete ledger marked missing")
            require(not gap["missing_frame_rows"], "complete ledger has frame gaps")
            verified_ledgers += 1
            verified_frames += len(rows)
        else:
            require(
                gap["missing_fields"] == ["android_canvas_canonical_detector_raw_successor"],
                "unexpected missing field taxonomy",
            )
            require(len(gap["missing_frame_rows"]) == len(rows), "frame gap expansion mismatch")
            require(
                [item["unit_id"] for item in gap["missing_frame_rows"]]
                == [row["unit_id"] for row in rows],
                "frame gap membership/order mismatch",
            )
            expected_missing_frames += len(rows)
        for row in rows:
            require(
                identity(row) in route_map,
                "route identity missing",
            )
    scope = receipt["verified_scope"]
    require(scope["expected_frames"] == 62229, "expected frame count drift")
    require(scope["fully_input_verified_sequence_ledgers"] == verified_ledgers, "verified ledger count drift")
    require(scope["fully_input_verified_frames"] == verified_frames, "verified frame count drift")
    output_root = repo / config["resource_guards"]["output_root"]
    expected_artifacts = collect_verified_input_artifacts(repo, groups, output_root)
    require(
        receipt["verified_input_artifacts"] == expected_artifacts,
        "verified compact/successor artifact binding drift",
    )
    if receipt["terminal_state"] == "FAIL_CLOSED_EXECUTION_ABORTED":
        require(receipt["candidate_execution"]["started"] is False, "candidate execution started")
        require(receipt["candidate_execution"]["authoritative_trace_count"] == 0, "partial trace authorized")
        require(receipt["profiles"] == [], "profile emitted after execution abort")
        require(expected_missing_frames == 57635, "unexpected raw gap after resource abort")
        require("available_memory_guard" in scope["first_blocker"], "resource blocker missing")
        guard_path = repo / receipt["guards"]["attempt_receipt_path"]
        require(
            sha256_file(guard_path) == receipt["guards"]["attempt_receipt_sha256"],
            "resource guard receipt hash drift",
        )
        current_implementations = {
            key: value
            for key, value in expected_bindings.items()
            if key.endswith("_implementation_sha256")
            or key == "terminal_schema_sha256"
        }
        guard = validate_exhausted_resource_guard_receipt(
            guard_path,
            expected_bindings["config_sha256"],
            current_implementations,
            int(
                config["resource_guards"][
                    "minimum_system_available_physical_memory_bytes"
                ]
            ),
            1
            + int(
                config["resource_guards"][
                    "maximum_retry_count_after_initial_attempt"
                ]
            ),
        )
        require(
            guard["automatic_retry_allowed_after_receipt"] is False,
            "fourth retry remains open",
        )
        require(not (output_root / "attempts").exists(), "device attempt exists after pre-device guard")
    elif receipt["terminal_state"] == "FAIL_CLOSED_INPUT_BLOCKED":
        require(receipt["candidate_execution"]["started"] is False, "candidate execution started on input block")
        require(receipt["profiles"] == [], "profile emitted on input block")
        require(expected_missing_frames > 0, "input-blocked receipt has no exact gap")
    else:
        require(expected_missing_frames == 0, "complete terminal has input gaps")
        require(verified_ledgers == 41 and verified_frames == 62229, "complete coverage mismatch")
        execution = receipt["candidate_execution"]
        require(execution["started"] is True, "complete receipt did not start candidates")
        require(execution["authoritative_trace_count"] == 123, "trace count mismatch")
        require(len(execution["trace_receipts"]) == 123, "trace receipt inventory mismatch")
        require(len(receipt["profiles"]) == 3, "profile count mismatch")
        for profile in receipt["profiles"]:
            validate_profile_contract(profile, config)
    return {
        "status": "VALID",
        "terminal_state": receipt["terminal_state"],
        "sequence_ledgers": len(groups),
        "frames": sum(len(rows) for _, rows in groups),
        "verified_sequence_ledgers": verified_ledgers,
        "verified_frames": verified_frames,
        "missing_frames": expected_missing_frames,
        "terminal_receipt_sha256": sha256_file(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = args.receipt or (
        repo
        / "artifacts.local/evidence/ustrf-route-target-l1-exploratory-profile-r1/terminal-receipt-r1.json"
    )
    result = validate(repo, args.config.resolve(), receipt.resolve())
    validation_path = receipt.with_name("validation-receipt-r1.json")
    __import__("exploratory_profiles_r2_l1").atomic_write_json(validation_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
