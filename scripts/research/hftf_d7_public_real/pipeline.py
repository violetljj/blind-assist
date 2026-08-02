#!/usr/bin/env python3
"""Small standard-library helpers for the HFTF D7 intake module."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractError(ValueError):
    """Raised when a D7 input violates the frozen schema or firewall."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:20]}"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot load JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row).decode("utf-8"))


def read_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"expected integer-like value, got {value!r}") from exc


def dataset_id_for_ledger_row(row: Mapping[str, str]) -> str | None:
    dataset = (row.get("dataset") or "").strip()
    if dataset == "SANPO":
        root = (row.get("session_root") or "").lower()
        return "SANPO-Synthetic" if "synthetic" in root else "SANPO-Real"
    if dataset == "EgoWalk":
        return "EgoWalk"
    if dataset == "JRDB":
        return "JRDB"
    if dataset == "THOR":
        return "THOR"
    if dataset == "PublicVideo":
        return "PublicVideo-Auxiliary"
    return None


def role_for_ledger_row(row: Mapping[str, str]) -> str:
    history = (row.get("history_roles") or "").lower()
    if (row.get("is_consumed") or "").lower() == "true" or (row.get("is_burned") or "").lower() == "true":
        return "THESIS_DEVELOPMENT_CONSUMED"
    if "reserved" in history or (row.get("is_reserved") or "").lower() == "true":
        return "HOLD_ROLE_REVIEW"
    if "fresh" in history or (row.get("is_fresh") or "").lower() == "true":
        return "HOLD_ROLE_REVIEW"
    return "DEVELOPMENT_CANDIDATE_DISCOVERY"

