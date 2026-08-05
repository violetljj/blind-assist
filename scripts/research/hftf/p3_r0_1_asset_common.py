#!/usr/bin/env python3
"""Fail-closed primitives shared by P3 R0.1 frozen-asset producers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


SHA_CHARS = frozenset("0123456789ABCDEF")
PROTOCOL_SCHEMA = "blindassist_dav2_temporal_392_student_p3_r0_1_protocol"
ROLE_MANIFEST_SCHEMA = "blindassist_dav2_temporal_392_student_p3_r0_1_role_manifest"
STATES = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
TRANSITIONS = tuple(f"{left}_TO_{right}" for left in STATES for right in STATES)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def valid_sha(value: Any) -> bool:
    normalized = str(value).upper()
    return len(normalized) == 64 and set(normalized) <= SHA_CHARS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_fields(value: dict[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    observed = set(value)
    require(observed == expected, f"{label} exact fields drift: {sorted(observed ^ expected)}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def resolve_inside(repo_root: Path, value: str) -> Path:
    root = repo_root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository: {value}") from error
    return path


def verify_bound_file(repo_root: Path, binding: dict[str, Any], label: str) -> Path:
    exact_fields(binding, {"path", "sha256"}, label)
    require(valid_sha(binding["sha256"]), f"{label} SHA invalid")
    path = resolve_inside(repo_root, str(binding["path"]))
    require(path.is_file(), f"{label} file missing")
    require(sha256_file(path) == str(binding["sha256"]).upper(), f"{label} SHA mismatch")
    return path


def verify_producer_sha(expected: Any, source_path: Path) -> str:
    require(valid_sha(expected), "producer SHA invalid")
    actual = sha256_file(source_path)
    require(actual == str(expected).upper(), "producer SHA mismatch")
    return actual


def assert_outputs_absent(repo_root: Path, relative_paths: Iterable[str]) -> list[Path]:
    paths = [resolve_inside(repo_root, value) for value in relative_paths]
    require(len(paths) == len(set(paths)), "output paths must be unique")
    for path in paths:
        require(not path.exists(), f"overwrite forbidden: {path}")
    return paths


def exclusive_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def output_receipt(
    *,
    schema: str,
    producer_sha256: str,
    request_sha256: str,
    input_sha256: dict[str, str],
    outputs: dict[str, tuple[str, bytes]],
) -> dict[str, Any]:
    return {
        "schema": schema,
        "producer_sha256": producer_sha256,
        "request_sha256": request_sha256,
        "input_sha256": dict(sorted(input_sha256.items())),
        "outputs": {
            name: {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
            for name, (relative, payload) in sorted(outputs.items())
        },
        "overwrite_permitted": False,
        "p3_model_constructed": False,
        "optimizer_constructed": False,
        "training_started": False,
        "legacy_p1_outcomes_read": False,
        "terminal": "P3_R0_1_FROZEN_ASSET_MATERIALIZED",
    }


def commit_outputs(
    repo_root: Path,
    *,
    outputs: dict[str, tuple[str, bytes]],
    receipt_relative: str,
    receipt: dict[str, Any],
) -> None:
    relative_paths = [relative for relative, _ in outputs.values()] + [receipt_relative]
    paths = assert_outputs_absent(repo_root, relative_paths)
    for path, (_, payload) in zip(paths[:-1], outputs.values()):
        exclusive_write(path, payload)
    exclusive_write(paths[-1], pretty_bytes(receipt))


def request_sha256(request: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(request))


def reject_outcome_keys(value: Any, *, label: str) -> None:
    forbidden = {
        "outcome",
        "outcomes",
        "candidate",
        "baseline",
        "metric",
        "metrics",
        "score",
        "scores",
        "result",
        "results",
        "prediction",
        "predictions",
    }
    if isinstance(value, dict):
        collided = {str(key).lower() for key in value} & forbidden
        require(not collided, f"{label} contains forbidden outcome keys: {sorted(collided)}")
        for nested in value.values():
            reject_outcome_keys(nested, label=label)
    elif isinstance(value, list):
        for nested in value:
            reject_outcome_keys(nested, label=label)


def validate_protocol(repo_root: Path, binding: dict[str, Any]) -> tuple[Path, str]:
    path = verify_bound_file(repo_root, binding, "protocol")
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "P3 R0.1 protocol schema drift")
    return path, sha256_file(path)
