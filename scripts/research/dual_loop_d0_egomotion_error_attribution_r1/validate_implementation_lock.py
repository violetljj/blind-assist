#!/usr/bin/env python3
"""Validate the frozen D0-R1 pre-execution implementation envelope.

This module deliberately stops before activation.  It may hash frozen files and
parse already-burned JSON metadata, but it never opens Vicon bag messages,
computes a D0 metric, creates an activation, or grants execution authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
PROTOCOL_ID = "D0_EGOMOTION_ERROR_ATTRIBUTION_R1"
LOCK_SCHEMA = (
    "blindassist.d0_egomotion_error_attribution.implementation_lock.v1"
)
ACTIVATION_SCHEMA = (
    "blindassist.d0_egomotion_error_attribution.activation.v1"
)
IMPLEMENTATION_REVIEW_SCHEMA = (
    "blindassist.d0_egomotion_error_attribution.implementation_review.v1"
)
PROTOCOL_PATH = (
    "docs/research/dual-loop/"
    "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R1_PROTOCOL_2026-07-30.json"
)
PROTOCOL_SHA256 = (
    "87931369f912fdd054783db9decb2a1813080d0a961c3526b83ce686d1a48183"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON forbids non-finite numbers")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            _reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item)


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted UTF-8 JSON without a BOM."""
    _reject_nonfinite(value)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_exclusive_fsync_json(path: Path, value: Any) -> None:
    """Exclusively create and fsync one canonical JSON file.

    A partial file is intentionally retained after any post-create failure:
    once exclusive creation succeeds, the one-shot namespace is consumed.
    """
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive marker write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_formal_start_marker(path: Path, value: Any) -> None:
    """Create the consumed-run marker without creating its parent namespace."""
    if path.name != "formal_start.json":
        raise ValueError("formal start marker must be named formal_start.json")
    if not path.parent.is_dir():
        raise FileNotFoundError(
            "formal output namespace must be prepared before marker creation"
        )
    write_exclusive_fsync_json(path, value)


def _repository_path(repository_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("repository binding path must be a non-empty string")
    normalized = relative.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or pure.drive or ".." in pure.parts:
        raise ValueError(f"unsafe repository binding path: {relative!r}")
    resolved_root = repository_root.resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"repository binding escapes root: {relative!r}"
        ) from error
    return resolved


def _count_jsonl(path: Path) -> tuple[int, int | None, bool]:
    row_count = 0
    primary_count = 0
    saw_primary_field = False
    last_byte = b""
    with path.open("rb") as stream:
        for raw in stream:
            last_byte = raw[-1:] if raw else last_byte
            if not raw.strip():
                continue
            row_count += 1
            row = json.loads(raw.decode("utf-8"))
            if "primary_event_eligible" in row:
                saw_primary_field = True
                if row["primary_event_eligible"] is True:
                    primary_count += 1
    return row_count, primary_count if saw_primary_field else None, last_byte == b"\n"


def _normalized_input_binding(specification: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(specification)
    if "sha256" not in binding and "content_identity_sha256" in binding:
        binding["sha256"] = binding["content_identity_sha256"]
    return binding


def _expected_implementation_paths(
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> set[str]:
    planned = protocol["planned_implementation"]
    module_root = str(planned["module_root"]).replace("\\", "/").rstrip("/")
    module_directory = _repository_path(repository_root, module_root)
    expected = {
        f"{module_root}/{path.name}"
        for path in module_directory.iterdir()
        if path.is_file() and (path.suffix == ".py" or path.name == "README.md")
    }
    expected.add(str(planned["stable_adapter"]).replace("\\", "/"))
    for module in planned["modules"]:
        expected.add(f"{module_root}/{module}")
    return expected


def validate_activation_identity(
    activation: Mapping[str, Any],
    implementation_lock_path: Path,
    repository_root: Path,
    *,
    expected_formal_output_root: str,
) -> dict[str, Any]:
    """Validate a later activation's identity bindings, not its authority."""
    failures: list[str] = []
    if activation.get("schema_version") != ACTIVATION_SCHEMA:
        failures.append("ACTIVATION_SCHEMA")
    if activation.get("protocol_id") != PROTOCOL_ID:
        failures.append("ACTIVATION_PROTOCOL_ID")
    if activation.get("execution_state") != "NOT_RUN":
        failures.append("ACTIVATION_EXECUTION_STATE")
    if activation.get("formal_output_root") != expected_formal_output_root:
        failures.append("ACTIVATION_OUTPUT_ROOT")

    binding = activation.get("implementation_lock", {})
    try:
        bound_path = _repository_path(repository_root, binding.get("path"))
    except (TypeError, ValueError):
        bound_path = None
        failures.append("ACTIVATION_LOCK_PATH")
    expected_lock = implementation_lock_path.resolve()
    try:
        expected_lock.relative_to(repository_root.resolve())
    except ValueError:
        failures.append("ACTIVATION_LOCK_OUTSIDE_REPOSITORY")
        expected_lock_allowed = False
    else:
        expected_lock_allowed = True
    if bound_path is not None and bound_path != expected_lock:
        failures.append("ACTIVATION_LOCK_PATH")
    actual_hash = (
        sha256_file(expected_lock)
        if expected_lock_allowed and expected_lock.is_file()
        else None
    )
    if binding.get("sha256") != actual_hash:
        failures.append("ACTIVATION_LOCK_SHA256")
    try:
        lock = json.loads(expected_lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        lock = {}
        failures.append("ACTIVATION_LOCK_PARSE")
    if activation.get("protocol_sha256") != lock.get("protocol", {}).get(
        "sha256"
    ):
        failures.append("ACTIVATION_PROTOCOL_SHA256")
    if activation.get("repository") != lock.get("repository"):
        failures.append("ACTIVATION_REPOSITORY")
    review = activation.get("implementation_review", {})
    try:
        review_path = _repository_path(repository_root, review.get("path"))
    except (TypeError, ValueError):
        review_path = None
        failures.append("ACTIVATION_REVIEW_PATH")
    if review_path is None or not review_path.is_file():
        failures.append("ACTIVATION_REVIEW_PATH")
    else:
        if review.get("sha256") != sha256_file(review_path):
            failures.append("ACTIVATION_REVIEW_SHA256")
        try:
            review_bytes = review_path.read_bytes()
            review_payload = json.loads(review_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            failures.append("ACTIVATION_REVIEW_PARSE")
        else:
            if review_bytes != canonical_json_bytes(review_payload) + b"\n":
                failures.append("ACTIVATION_REVIEW_CANONICAL")
            checks = review_payload.get("checks")
            if (
                review_payload.get("schema_version")
                != IMPLEMENTATION_REVIEW_SCHEMA
                or review_payload.get("status") != "PASS"
                or review_payload.get("reviewer_role")
                != "INDEPENDENT_READ_ONLY_REVIEW"
                or review_payload.get("protocol")
                != {
                    "protocol_id": PROTOCOL_ID,
                    "sha256": lock.get("protocol", {}).get("sha256"),
                }
                or review_payload.get("implementation_lock")
                != {
                    "path": binding.get("path"),
                    "sha256": actual_hash,
                }
                or review_payload.get("repository") != lock.get("repository")
                or review_payload.get("formal_execution_authorized") is not False
                or not isinstance(checks, list)
                or not checks
                or any(
                    not isinstance(check, dict)
                    or check.get("passed") is not True
                    for check in checks
                )
            ):
                failures.append("ACTIVATION_REVIEW_STATUS")
    return {
        "status": "VALID_IDENTITY" if not failures else "INVALID_IDENTITY",
        "failures": sorted(set(failures)),
        "implementation_lock_sha256": actual_hash,
        "authorization_evaluated": False,
    }


def validate(
    lock_path: Path,
    repository_root: Path = REPO_ROOT,
    *,
    expected_protocol_path: str = PROTOCOL_PATH,
    expected_protocol_sha256: str = PROTOCOL_SHA256,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "detail": detail}
        )

    try:
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {
            "status": "INVALID",
            "failures": ["LOCK_PARSE"],
            "checks": [
                {
                    "name": "lock_parse",
                    "passed": False,
                    "detail": type(error).__name__,
                }
            ],
            "formal_execution_authorized": False,
            "vicon_bag_messages_opened": False,
        }

    check(
        "lock_canonical_json",
        lock_bytes == canonical_json_bytes(lock) + b"\n",
    )
    check("schema", lock.get("schema_version") == LOCK_SCHEMA)
    check("protocol_id", lock.get("protocol_id") == PROTOCOL_ID)
    check(
        "implementation_status",
        lock.get("implementation_status")
        == "FROZEN_FOR_INDEPENDENT_REVIEW",
    )
    authority = lock.get("authority", {})
    check(
        "not_run_no_authority",
        lock.get("execution_state") == "NOT_RUN"
        and authority.get("activation_authorized") is False
        and authority.get("formal_execution_authorized") is False
        and authority.get("scientific_exit_authorized") is False,
        {
            "execution_state": lock.get("execution_state"),
            "authority": authority,
        },
    )

    protocol_binding = lock.get("protocol", {})
    check(
        "protocol_binding_identity",
        protocol_binding
        == {
            "path": expected_protocol_path,
            "sha256": expected_protocol_sha256,
        },
        protocol_binding,
    )
    try:
        protocol_path = _repository_path(
            repository_root, expected_protocol_path
        )
        protocol_hash = (
            sha256_file(protocol_path) if protocol_path.is_file() else None
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        protocol_hash = None
        protocol = {}
        check("protocol_parse", False, type(error).__name__)
    else:
        check("protocol_parse", True)
    check(
        "protocol_sha256",
        protocol_hash == expected_protocol_sha256,
        {"expected": expected_protocol_sha256, "actual": protocol_hash},
    )
    check(
        "protocol_frozen_not_run",
        protocol.get("protocol_id") == PROTOCOL_ID
        and protocol.get("status") == "CONTRACT_FROZEN"
        and protocol.get("execution_status") == "NOT_RUN"
        and protocol.get("execution_authorized") is False,
    )

    predecessor = protocol.get("predecessor_gate", {})
    expected_predecessor = {
        "result": dict(predecessor.get("result", {})),
        "independent_validation": dict(
            predecessor.get("independent_validation", {})
        ),
        "seal": dict(predecessor.get("seal", {})),
    }
    actual_predecessor = lock.get("predecessor_bindings", {})
    check(
        "predecessor_binding_set",
        actual_predecessor == expected_predecessor,
    )
    for name, specification in expected_predecessor.items():
        try:
            path = _repository_path(repository_root, specification.get("path"))
        except (TypeError, ValueError) as error:
            check(f"predecessor_{name}", False, str(error))
            continue
        actual_hash = sha256_file(path) if path.is_file() else None
        check(
            f"predecessor_{name}_sha256",
            actual_hash == specification.get("sha256"),
            {
                "expected": specification.get("sha256"),
                "actual": actual_hash,
            },
        )
        if path.suffix == ".json" and path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, expected in specification.items():
                if not key.startswith("required_"):
                    continue
                actual_key = key.removeprefix("required_")
                check(
                    f"predecessor_{name}_{actual_key}",
                    payload.get(actual_key) == expected,
                    {
                        "expected": expected,
                        "actual": payload.get(actual_key),
                    },
                )

    expected_inputs = {
        name: _normalized_input_binding(specification)
        for name, specification in protocol.get("frozen_inputs", {}).items()
    }
    actual_inputs = lock.get("frozen_inputs", {})
    check(
        "frozen_input_binding_set",
        actual_inputs == expected_inputs,
        {
            "expected": sorted(expected_inputs),
            "actual": (
                sorted(actual_inputs) if isinstance(actual_inputs, dict) else None
            ),
        },
    )
    for name, specification in expected_inputs.items():
        try:
            path = _repository_path(repository_root, specification.get("path"))
        except (TypeError, ValueError) as error:
            check(f"input_{name}_path", False, str(error))
            continue
        if not path.is_file():
            check(f"input_{name}_exists", False, str(path))
            continue
        expected_size = specification.get("bytes")
        if expected_size is not None:
            check(
                f"input_{name}_bytes",
                path.stat().st_size == expected_size,
                {
                    "expected": expected_size,
                    "actual": path.stat().st_size,
                },
            )
        actual_hash = sha256_file(path)
        check(
            f"input_{name}_sha256",
            actual_hash == specification.get("sha256"),
            {
                "expected": specification.get("sha256"),
                "actual": actual_hash,
            },
        )
        if path.suffix == ".jsonl":
            try:
                rows, primary_rows, final_lf = _count_jsonl(path)
            except (UnicodeError, json.JSONDecodeError) as error:
                check(f"input_{name}_jsonl", False, type(error).__name__)
            else:
                check(f"input_{name}_final_lf", final_lf)
                if "rows" in specification:
                    check(
                        f"input_{name}_rows",
                        rows == specification["rows"],
                        {
                            "expected": specification["rows"],
                            "actual": rows,
                        },
                    )
                if "primary_rows" in specification:
                    check(
                        f"input_{name}_primary_rows",
                        primary_rows == specification["primary_rows"],
                        {
                            "expected": specification["primary_rows"],
                            "actual": primary_rows,
                        },
                    )
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                check(f"input_{name}_json", False, type(error).__name__)
            else:
                for key in (
                    "schema_version",
                    "status",
                    "primary_event_count",
                    "cross_target_overlap_pair_count",
                    "same_target_overlap_pair_count",
                    "exact_overlap_component_count",
                ):
                    if key in specification:
                        check(
                            f"input_{name}_{key}",
                            payload.get(key) == specification[key],
                            {
                                "expected": specification[key],
                                "actual": payload.get(key),
                            },
                        )

    canonical_contract = (
        protocol.get("planned_implementation", {})
        .get("canonical_serialization", {})
    )
    check(
        "canonical_serialization_sha256",
        lock.get("canonical_serialization_sha256")
        == canonical_sha256(canonical_contract),
    )
    check(
        "frozen_inputs_sha256",
        lock.get("frozen_inputs_sha256")
        == canonical_sha256(expected_inputs),
    )

    try:
        expected_paths = _expected_implementation_paths(
            repository_root, protocol
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        expected_paths = set()
        check("implementation_path_inventory", False, type(error).__name__)
    implementation_hashes = lock.get("implementation_file_hashes", {})
    check(
        "implementation_file_set",
        isinstance(implementation_hashes, dict)
        and set(implementation_hashes) == expected_paths,
        {
            "expected": sorted(expected_paths),
            "actual": (
                sorted(implementation_hashes)
                if isinstance(implementation_hashes, dict)
                else None
            ),
        },
    )
    if isinstance(implementation_hashes, dict):
        for relative, expected_hash in sorted(implementation_hashes.items()):
            try:
                path = _repository_path(repository_root, relative)
            except (TypeError, ValueError) as error:
                check(f"implementation_{relative}", False, str(error))
                continue
            actual_hash = sha256_file(path) if path.is_file() else None
            check(
                f"implementation_{relative}",
                actual_hash == expected_hash,
                {"expected": expected_hash, "actual": actual_hash},
            )

    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "status": "VALID" if not failures else "INVALID",
        "failures": failures,
        "checks": checks,
        "lock_sha256": sha256_file(lock_path),
        "formal_execution_authorized": False,
        "activation_authorized": False,
        "vicon_bag_messages_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPO_ROOT,
    )
    args = parser.parse_args()
    result = validate(
        args.implementation_lock.resolve(),
        args.repository_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
