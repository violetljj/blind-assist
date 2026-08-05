#!/usr/bin/env python3
"""Fail-closed helpers for P3 R0.2.1 private sealing producers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


STATES = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
TRANSITIONS = tuple(f"{left}_TO_{right}" for left in STATES for right in STATES)
SHA_CHARS = frozenset("0123456789ABCDEF")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_fields(value: dict[str, Any], expected: Iterable[str], label: str) -> None:
    drift = set(value) ^ set(expected)
    require(not drift, f"{label} exact fields drift: {sorted(drift)}")


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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


def resolve_inside(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"path leaves repository: {relative}")
    return path


def verify_bound_file(repo_root: Path, binding: dict[str, Any], label: str) -> Path:
    exact_fields(binding, {"path", "sha256"}, label)
    require(valid_sha(binding["sha256"]), f"{label} SHA invalid")
    path = resolve_inside(repo_root, str(binding["path"]))
    require(path.is_file(), f"{label} missing")
    require(sha256_file(path) == str(binding["sha256"]).upper(), f"{label} SHA mismatch")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def exclusive_write(path: Path, value: bytes) -> None:
    require(not path.exists(), f"overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def materialization_receipt(schema: str, producer_sha: str, inputs: dict[str, str], outputs: dict[str, tuple[str, bytes]]) -> dict[str, Any]:
    return {
        "schema": schema,
        "producer_sha256": producer_sha,
        "input_sha256": dict(sorted(inputs.items())),
        "outputs": {
            name: {"path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
            for name, (relative, payload) in sorted(outputs.items())
        },
        "overwrite_permitted": False,
        "holdout_outcomes_opened": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
    }
