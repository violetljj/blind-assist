#!/usr/bin/env python3
"""BlindAssist zero-copy master asset catalog.

The catalog inventories durable assets in place, resolves them through stable
ids, records every declared consumer and derivation, and keeps evidence
authority separate from storage lifecycle. Byte hashing is explicit and
incremental: discovery gives every asset a locator identity immediately,
while ``hash`` promotes selected immutable assets to content identity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = DEFAULT_REPO_ROOT / "artifacts.local"
DEFAULT_POLICY_PATH = DEFAULT_REPO_ROOT / "data" / "asset-management-policy.json"
DEFAULT_DATABASE_RELATIVE = Path("evidence/resource-fabric/catalog/master-assets.sqlite3")
DEFAULT_REPORT_RELATIVE = Path("evidence/resource-fabric/reports/assets/current")
VALID_EVIDENCE_STATUSES = {
    "reserved",
    "fresh",
    "development_consumed",
    "sealed_final",
    "diagnostic",
    "unknown",
}
VALID_STORAGE_STATUSES = {
    "active",
    "shared",
    "sealed_cold",
    "rebuildable",
    "unknown",
}


class CatalogError(RuntimeError):
    """A user-correctable asset catalog error."""


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Cannot read JSON {path}: {exc}") from exc


def parse_json(value: str | None, *, default: Any) -> Any:
    if value is None:
        return default
    if value.startswith("@"):
        return read_json(Path(value[1:]).resolve())
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Invalid JSON argument: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def is_reparse_point(path: Path) -> bool:
    info = path.lstat()
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & marker)


def normalize_locator(value: str) -> str:
    locator = value.replace("\\", "/").strip("/")
    if not locator or locator == ".":
        raise CatalogError("Asset locator cannot be empty")
    if any(part in {"", ".", ".."} for part in locator.split("/")):
        raise CatalogError(f"Invalid asset locator: {value}")
    return locator


def locator_for_path(path: Path, artifact_root: Path) -> str:
    resolved = path.resolve()
    root = artifact_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise CatalogError(f"Asset path escapes artifacts.local: {resolved}") from exc


def path_for_locator(locator: str, artifact_root: Path) -> Path:
    root = artifact_root.resolve()
    candidate = root.joinpath(*normalize_locator(locator).split("/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CatalogError(f"Catalog locator escapes artifacts.local: {locator}") from exc
    return candidate


def asset_id_for_locator(locator: str) -> str:
    identity = f"blindassist-asset-locator-v1\0{normalize_locator(locator).casefold()}"
    return f"asset:{sha256_bytes(identity.encode('utf-8'))}"


def load_policy(path: Path) -> dict[str, Any]:
    policy = read_json(path)
    if policy.get("schema") != "blindassist-asset-management-policy-v1":
        raise CatalogError(f"Unsupported asset policy schema: {policy.get('schema')}")
    for name, rule in policy.get("roots", {}).items():
        validate_rule(name, rule)
    validate_rule("fallback", policy.get("fallback", {}))
    return policy


def validate_rule(name: str, rule: dict[str, Any]) -> None:
    required = {
        "asset_kind",
        "asset_class",
        "evidence_status",
        "storage_status",
        "owner",
        "retention_reason",
    }
    missing = sorted(required - set(rule))
    if missing:
        raise CatalogError(f"Policy rule {name} is missing: {missing}")
    if rule["evidence_status"] not in VALID_EVIDENCE_STATUSES:
        raise CatalogError(f"Invalid evidence status in policy rule {name}")
    if rule["storage_status"] not in VALID_STORAGE_STATUSES:
        raise CatalogError(f"Invalid storage status in policy rule {name}")


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    scan_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    assets_seen INTEGER NOT NULL DEFAULT 0,
    files_seen INTEGER NOT NULL DEFAULT 0,
    bytes_seen INTEGER NOT NULL DEFAULT 0,
    vanished_entries INTEGER NOT NULL DEFAULT 0,
    reparse_entries INTEGER NOT NULL DEFAULT 0,
    errors_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS roots (
    root_name TEXT PRIMARY KEY,
    disposition TEXT NOT NULL,
    asset_kind TEXT,
    asset_class TEXT,
    reason TEXT NOT NULL,
    state TEXT NOT NULL,
    asset_count INTEGER NOT NULL DEFAULT 0,
    file_count INTEGER NOT NULL DEFAULT 0,
    bytes INTEGER NOT NULL DEFAULT 0,
    vanished_entries INTEGER NOT NULL DEFAULT 0,
    reparse_entries INTEGER NOT NULL DEFAULT 0,
    last_scan_id TEXT,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    locator TEXT NOT NULL UNIQUE,
    logical_name TEXT NOT NULL,
    root_name TEXT NOT NULL,
    asset_kind TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    state TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    metadata_sha256 TEXT,
    content_id TEXT,
    identity_strength TEXT NOT NULL,
    evidence_status TEXT NOT NULL,
    storage_status TEXT NOT NULL,
    owner TEXT NOT NULL,
    retention_reason TEXT NOT NULL,
    rebuild_command TEXT,
    rebuild_cost TEXT,
    source_uri TEXT,
    license_id TEXT,
    claim_ceiling TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_scan_id TEXT,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS assets_locator_index ON assets(locator);
CREATE INDEX IF NOT EXISTS assets_name_index ON assets(logical_name);
CREATE INDEX IF NOT EXISTS assets_content_index ON assets(content_id);
CREATE INDEX IF NOT EXISTS assets_root_index ON assets(root_name, state);

CREATE TABLE IF NOT EXISTS asset_files (
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT,
    PRIMARY KEY (asset_id, relative_path)
);

CREATE INDEX IF NOT EXISTS asset_files_size_index ON asset_files(bytes);
CREATE INDEX IF NOT EXISTS asset_files_hash_index ON asset_files(sha256);

CREATE TABLE IF NOT EXISTS usage_events (
    event_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    consumer TEXT NOT NULL,
    purpose TEXT NOT NULL,
    experiment_id TEXT,
    access_mode TEXT NOT NULL,
    evidence_effect TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS usage_asset_index ON usage_events(asset_id, recorded_at);
CREATE INDEX IF NOT EXISTS usage_consumer_index ON usage_events(consumer, recorded_at);

CREATE TABLE IF NOT EXISTS asset_references (
    reference_id TEXT PRIMARY KEY,
    asset_id TEXT REFERENCES assets(asset_id),
    root_name TEXT,
    source_file TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    raw_locator TEXT NOT NULL,
    resolution TEXT NOT NULL,
    scan_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS reference_asset_index ON asset_references(asset_id);
CREATE INDEX IF NOT EXISTS reference_source_index ON asset_references(source_file, line_number);

CREATE TABLE IF NOT EXISTS lifecycle_events (
    event_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    evidence_status TEXT NOT NULL,
    storage_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    claim_ceiling TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS lifecycle_asset_index ON lifecycle_events(asset_id, recorded_at);

CREATE TABLE IF NOT EXISTS derivations (
    derivation_id TEXT PRIMARY KEY,
    output_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    transform TEXT NOT NULL,
    transform_version TEXT NOT NULL,
    producer TEXT NOT NULL,
    code_sha256 TEXT,
    config_sha256 TEXT,
    parameters_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS derivation_inputs (
    derivation_id TEXT NOT NULL REFERENCES derivations(derivation_id) ON DELETE CASCADE,
    input_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    input_role TEXT NOT NULL,
    PRIMARY KEY (derivation_id, input_asset_id, input_role)
);
"""


def open_catalog(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA_SQL)
    reference_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(asset_references)")
    }
    if "root_name" not in reference_columns:
        connection.execute("ALTER TABLE asset_references ADD COLUMN root_name TEXT")
    if "resolution" not in reference_columns:
        connection.execute(
            "ALTER TABLE asset_references ADD COLUMN resolution TEXT NOT NULL DEFAULT 'unknown_root'"
        )
    connection.execute(
        "INSERT OR REPLACE INTO catalog_meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


def scan_asset(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if is_reparse_point(path):
        raise CatalogError(f"Asset unit cannot be a reparse point: {path}")

    entries: list[dict[str, Any]] = []
    vanished = 0
    reparse_skipped = 0
    if path.is_file():
        info = path.stat()
        entries.append(
            {
                "relative_path": ".",
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "sha256": None,
            }
        )
        entry_type = "file"
    elif path.is_dir():
        entry_type = "directory"
        stack = [path]
        while stack:
            directory = stack.pop()
            try:
                children = sorted(os.scandir(directory), key=lambda item: item.name.casefold())
            except FileNotFoundError:
                vanished += 1
                continue
            for child in reversed(children):
                child_path = Path(child.path)
                try:
                    if is_reparse_point(child_path):
                        reparse_skipped += 1
                        continue
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child_path)
                        continue
                    if not child.is_file(follow_symlinks=False):
                        continue
                    info = child.stat(follow_symlinks=False)
                except FileNotFoundError:
                    vanished += 1
                    continue
                entries.append(
                    {
                        "relative_path": child_path.relative_to(path).as_posix(),
                        "bytes": info.st_size,
                        "mtime_ns": info.st_mtime_ns,
                        "sha256": None,
                    }
                )
    else:
        raise CatalogError(f"Unsupported asset unit: {path}")

    entries.sort(key=lambda item: item["relative_path"].casefold())
    fingerprint = hashlib.sha256()
    fingerprint.update(b"blindassist-asset-metadata-v1\0")
    for item in entries:
        fingerprint.update(item["relative_path"].encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(str(item["bytes"]).encode("ascii"))
        fingerprint.update(b"\0")
        fingerprint.update(str(item["mtime_ns"]).encode("ascii"))
        fingerprint.update(b"\n")
    return {
        "entry_type": entry_type,
        "bytes": sum(int(item["bytes"]) for item in entries),
        "file_count": len(entries),
        "metadata_sha256": fingerprint.hexdigest(),
        "entries": entries,
        "vanished_entries": vanished,
        "reparse_entries": reparse_skipped,
    }


ASSET_COLUMNS = [
    "asset_id",
    "locator",
    "logical_name",
    "root_name",
    "asset_kind",
    "asset_class",
    "entry_type",
    "state",
    "bytes",
    "file_count",
    "metadata_sha256",
    "content_id",
    "identity_strength",
    "evidence_status",
    "storage_status",
    "owner",
    "retention_reason",
    "rebuild_command",
    "rebuild_cost",
    "source_uri",
    "license_id",
    "claim_ceiling",
    "first_seen_at",
    "last_seen_at",
    "last_scan_id",
    "metadata_json",
]


def upsert_asset(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    preserve_lifecycle: bool = True,
) -> None:
    existing = connection.execute(
        "SELECT * FROM assets WHERE asset_id = ?",
        (record["asset_id"],),
    ).fetchone()
    if existing:
        record["first_seen_at"] = existing["first_seen_at"]
        if preserve_lifecycle and existing["asset_class"] != "legacy_unclassified":
            record["evidence_status"] = existing["evidence_status"]
            record["storage_status"] = existing["storage_status"]
            record["claim_ceiling"] = existing["claim_ceiling"]
            if existing["owner"] != "unclassified":
                record["owner"] = existing["owner"]
            if existing["retention_reason"]:
                record["retention_reason"] = existing["retention_reason"]
            for field in (
                "rebuild_command",
                "rebuild_cost",
                "source_uri",
                "license_id",
            ):
                if existing[field] is not None:
                    record[field] = existing[field]
        if record.get("content_id") is None and existing["content_id"] is not None:
            if record["metadata_sha256"] == existing["metadata_sha256"]:
                record["content_id"] = existing["content_id"]
                record["identity_strength"] = existing["identity_strength"]
                existing_hashes = {
                    row["relative_path"]: row["sha256"]
                    for row in connection.execute(
                        "SELECT relative_path, sha256 FROM asset_files WHERE asset_id = ?",
                        (record["asset_id"],),
                    )
                    if row["sha256"] is not None
                }
                for item in entries:
                    item["sha256"] = existing_hashes.get(item["relative_path"])

    placeholders = ", ".join(f":{column}" for column in ASSET_COLUMNS)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in ASSET_COLUMNS
        if column not in {"asset_id", "first_seen_at"}
    )
    connection.execute(
        f"""
        INSERT INTO assets ({', '.join(ASSET_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(asset_id) DO UPDATE SET {updates}
        """,
        {column: record.get(column) for column in ASSET_COLUMNS},
    )
    connection.execute("DELETE FROM asset_files WHERE asset_id = ?", (record["asset_id"],))
    connection.executemany(
        """
        INSERT INTO asset_files(asset_id, relative_path, bytes, mtime_ns, sha256)
        VALUES(?, ?, ?, ?, ?)
        """,
        [
            (
                record["asset_id"],
                item["relative_path"],
                int(item["bytes"]),
                int(item.get("mtime_ns", 0)),
                item.get("sha256"),
            )
            for item in entries
        ],
    )


def discovered_record(
    locator: str,
    root_name: str,
    rule: dict[str, Any],
    scan: dict[str, Any],
    scan_id: str,
    now: str,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id_for_locator(locator),
        "locator": locator,
        "logical_name": Path(locator).name,
        "root_name": root_name,
        "asset_kind": rule["asset_kind"],
        "asset_class": rule["asset_class"],
        "entry_type": scan["entry_type"],
        "state": "present",
        "bytes": scan["bytes"],
        "file_count": scan["file_count"],
        "metadata_sha256": scan["metadata_sha256"],
        "content_id": None,
        "identity_strength": "metadata",
        "evidence_status": rule["evidence_status"],
        "storage_status": rule["storage_status"],
        "owner": rule["owner"],
        "retention_reason": rule["retention_reason"],
        "rebuild_command": rule.get("rebuild_command"),
        "rebuild_cost": rule.get("rebuild_cost"),
        "source_uri": rule.get("source_uri"),
        "license_id": rule.get("license_id"),
        "claim_ceiling": rule.get("claim_ceiling", "UNCLASSIFIED"),
        "first_seen_at": now,
        "last_seen_at": now,
        "last_scan_id": scan_id,
        "metadata_json": json.dumps(
            {
                "discovery": "zero-copy",
                "vanished_entries": scan["vanished_entries"],
                "reparse_entries": scan["reparse_entries"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def upsert_root(
    connection: sqlite3.Connection,
    *,
    root_name: str,
    disposition: str,
    rule: dict[str, Any] | None,
    reason: str,
    state: str,
    asset_count: int,
    file_count: int,
    bytes_count: int,
    vanished: int,
    reparse_count: int,
    scan_id: str,
    now: str,
) -> None:
    connection.execute(
        """
        INSERT INTO roots(
            root_name, disposition, asset_kind, asset_class, reason, state,
            asset_count, file_count, bytes, vanished_entries, reparse_entries,
            last_scan_id, last_seen_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(root_name) DO UPDATE SET
            disposition=excluded.disposition,
            asset_kind=excluded.asset_kind,
            asset_class=excluded.asset_class,
            reason=excluded.reason,
            state=excluded.state,
            asset_count=excluded.asset_count,
            file_count=excluded.file_count,
            bytes=excluded.bytes,
            vanished_entries=excluded.vanished_entries,
            reparse_entries=excluded.reparse_entries,
            last_scan_id=excluded.last_scan_id,
            last_seen_at=excluded.last_seen_at
        """,
        (
            root_name,
            disposition,
            rule.get("asset_kind") if rule else None,
            rule.get("asset_class") if rule else None,
            reason,
            state,
            asset_count,
            file_count,
            bytes_count,
            vanished,
            reparse_count,
            scan_id,
            now,
        ),
    )


def record_usage(
    connection: sqlite3.Connection,
    *,
    asset_id: str,
    consumer: str,
    purpose: str,
    experiment_id: str | None,
    access_mode: str,
    evidence_effect: str,
    metadata: Any,
    event_id: str | None = None,
    recorded_at: str | None = None,
    ignore_existing: bool = False,
) -> str:
    if evidence_effect not in {"none", "development_consumed", "sealed_final"}:
        raise CatalogError(f"Invalid evidence effect: {evidence_effect}")
    if connection.execute(
        "SELECT 1 FROM assets WHERE asset_id = ?", (asset_id,)
    ).fetchone() is None:
        raise CatalogError(f"Unknown asset id: {asset_id}")
    event_id = event_id or f"use:{uuid.uuid4()}"
    values = (
        event_id,
        asset_id,
        consumer,
        purpose,
        experiment_id,
        access_mode,
        evidence_effect,
        recorded_at or utc_now(),
        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
    connection.execute(
        f"""
        {verb} INTO usage_events(
            event_id, asset_id, consumer, purpose, experiment_id,
            access_mode, evidence_effect, recorded_at, metadata_json
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    if evidence_effect != "none":
        transition_asset(
            connection,
            asset_id=asset_id,
            evidence_status=evidence_effect,
            storage_status=None,
            reason=f"Usage by {consumer}: {purpose}",
            claim_ceiling=None,
            event_id=f"{event_id}:lifecycle",
            recorded_at=recorded_at,
            ignore_existing=ignore_existing,
        )
    return event_id


def transition_asset(
    connection: sqlite3.Connection,
    *,
    asset_id: str,
    evidence_status: str | None,
    storage_status: str | None,
    reason: str,
    claim_ceiling: str | None,
    event_id: str | None = None,
    recorded_at: str | None = None,
    ignore_existing: bool = False,
) -> str:
    asset = connection.execute(
        "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    if asset is None:
        raise CatalogError(f"Unknown asset id: {asset_id}")
    evidence_status = evidence_status or asset["evidence_status"]
    storage_status = storage_status or asset["storage_status"]
    claim_ceiling = claim_ceiling or asset["claim_ceiling"]
    if evidence_status not in VALID_EVIDENCE_STATUSES:
        raise CatalogError(f"Invalid evidence status: {evidence_status}")
    if storage_status not in VALID_STORAGE_STATUSES:
        raise CatalogError(f"Invalid storage status: {storage_status}")
    statuses = {
        row[0]
        for row in connection.execute(
            "SELECT evidence_status FROM lifecycle_events WHERE asset_id = ?",
            (asset_id,),
        )
    }
    statuses.add(asset["evidence_status"])
    if "development_consumed" in statuses and evidence_status != "development_consumed":
        raise CatalogError(
            "A development_consumed asset cannot regain another evidence status"
        )
    if "sealed_final" in statuses and evidence_status != "sealed_final":
        raise CatalogError("A sealed_final asset must retain sealed_final evidence status")
    event_id = event_id or f"lifecycle:{uuid.uuid4()}"
    verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
    connection.execute(
        f"""
        {verb} INTO lifecycle_events(
            event_id, asset_id, evidence_status, storage_status,
            reason, claim_ceiling, recorded_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            asset_id,
            evidence_status,
            storage_status,
            reason,
            claim_ceiling,
            recorded_at or utc_now(),
        ),
    )
    connection.execute(
        """
        UPDATE assets
        SET evidence_status = ?, storage_status = ?, claim_ceiling = ?
        WHERE asset_id = ?
        """,
        (evidence_status, storage_status, claim_ceiling, asset_id),
    )
    return event_id


def resolve_asset(connection: sqlite3.Connection, key: str) -> sqlite3.Row:
    normalized = key.replace("\\", "/").strip("/")
    rows = connection.execute(
        """
        SELECT * FROM assets
        WHERE asset_id = ? OR locator = ? OR content_id = ? OR logical_name = ?
        ORDER BY CASE
            WHEN asset_id = ? THEN 0
            WHEN locator = ? THEN 1
            WHEN content_id = ? THEN 2
            ELSE 3
        END, locator
        """,
        (key, normalized, key, key, key, normalized, key),
    ).fetchall()
    if not rows:
        raise CatalogError(f"Asset not found: {key}")
    best_rank = (
        0 if rows[0]["asset_id"] == key else
        1 if rows[0]["locator"] == normalized else
        2 if rows[0]["content_id"] == key else 3
    )
    equally_ranked = []
    for row in rows:
        rank = (
            0 if row["asset_id"] == key else
            1 if row["locator"] == normalized else
            2 if row["content_id"] == key else 3
        )
        if rank == best_rank:
            equally_ranked.append(row)
    if len(equally_ranked) > 1:
        choices = [row["locator"] for row in equally_ranked[:10]]
        raise CatalogError(f"Ambiguous asset key {key!r}; use asset_id or locator: {choices}")
    return equally_ranked[0]


REFERENCE_EXTENSIONS = {
    ".bat",
    ".cmd",
    ".gradle",
    ".json",
    ".jsonl",
    ".kts",
    ".kt",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ARTIFACT_REFERENCE_PATTERN = re.compile(
    r"artifacts\.local[\\/][^\s'\"`<>(){}\[\],;]+",
    re.IGNORECASE,
)


def repository_text_files(repo_root: Path, artifact_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        relative_paths = [
            Path(value.decode("utf-8", errors="surrogateescape"))
            for value in result.stdout.split(b"\0")
            if value
        ]
        candidates = [repo_root / relative for relative in relative_paths]
    except (OSError, subprocess.CalledProcessError):
        candidates = list(repo_root.rglob("*"))
    artifact_resolved = artifact_root.resolve()
    selected: list[Path] = []
    for path in candidates:
        if not path.is_file() or path.suffix.casefold() not in REFERENCE_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(artifact_resolved)
            continue
        except ValueError:
            pass
        try:
            if path.stat().st_size > 16 * 1024 * 1024:
                continue
        except FileNotFoundError:
            continue
        selected.append(path)
    return sorted(selected, key=lambda item: str(item).casefold())


def import_repository_references(
    connection: sqlite3.Connection,
    repo_root: Path,
    artifact_root: Path,
    scan_id: str,
) -> dict[str, int]:
    assets = connection.execute(
        "SELECT asset_id, locator FROM assets WHERE state = 'present'"
    ).fetchall()
    locator_rows = sorted(
        [(row["locator"].casefold(), row["asset_id"]) for row in assets],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    root_names = {
        row["root_name"]
        for row in connection.execute("SELECT root_name FROM roots")
    }
    observations: list[
        tuple[str, str | None, str | None, str, int, str, str, str, str]
    ] = []
    now = utc_now()
    for path in repository_text_files(repo_root, artifact_root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            source_file = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except (OSError, ValueError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for occurrence, match in enumerate(ARTIFACT_REFERENCE_PATTERN.finditer(line)):
                raw = match.group(0).rstrip(".:")
                marker_index = raw.casefold().index("artifacts.local")
                locator = raw[marker_index + len("artifacts.local"):].lstrip("\\/")
                locator = locator.replace("\\", "/").strip("/")
                if not locator:
                    continue
                asset_id = None
                locator_folded = locator.casefold()
                for candidate, candidate_id in locator_rows:
                    if locator_folded == candidate or locator_folded.startswith(candidate + "/"):
                        asset_id = candidate_id
                        break
                root_name = locator.split("/", 1)[0]
                known_root = root_name if root_name in root_names else None
                if asset_id is not None:
                    resolution = "asset"
                elif (
                    any(marker in locator for marker in ("*", "${", "$(`", "YYYY", "<"))
                    or any(part.casefold().startswith("my-") for part in locator.split("/"))
                ):
                    resolution = "template"
                elif any(
                    locator_folded == prefix
                    or locator_folded.startswith(prefix + "/")
                    for prefix in (
                        "downloads/resource-store",
                        "evidence/resource-fabric",
                        "evidence/resource-store",
                        "models/resource-store",
                        "work/resource-cache",
                    )
                ):
                    resolution = "root"
                elif known_root and locator == known_root:
                    resolution = "root"
                elif known_root:
                    resolution = "missing_within_root"
                else:
                    resolution = "unknown_root"
                identity = canonical_json_bytes(
                    {
                        "source_file": source_file,
                        "line": line_number,
                        "occurrence": occurrence,
                        "locator": locator,
                    }
                )
                reference_id = f"reference:{sha256_bytes(identity)}"
                observations.append(
                    (
                        reference_id,
                        asset_id,
                        known_root,
                        source_file,
                        line_number,
                        locator,
                        resolution,
                        scan_id,
                        now,
                    )
                )
    with connection:
        connection.execute("DELETE FROM asset_references")
        connection.executemany(
            """
            INSERT INTO asset_references(
                reference_id, asset_id, root_name, source_file, line_number,
                raw_locator, resolution, scan_id, recorded_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            observations,
        )
    resolved = sum(1 for item in observations if item[6] == "asset")
    root_scoped = sum(1 for item in observations if item[6] == "root")
    missing = sum(1 for item in observations if item[6] == "missing_within_root")
    templates = sum(1 for item in observations if item[6] == "template")
    unknown = sum(1 for item in observations if item[6] == "unknown_root")
    return {
        "references": len(observations),
        "resolved_assets": resolved,
        "root_scoped": root_scoped,
        "missing_within_root": missing,
        "templates": templates,
        "unknown_roots": unknown,
        "unresolved": missing + unknown,
        "referenced_assets": len({item[1] for item in observations if item[1] is not None}),
    }


def fabric_records(root: Path, relative: str) -> list[dict[str, Any]]:
    directory = root / relative
    if not directory.exists():
        return []
    return [read_json(path) for path in sorted(directory.rglob("*.json"))]


def sync_resource_fabric(
    connection: sqlite3.Connection,
    artifact_root: Path,
    scan_id: str,
    now: str,
) -> dict[str, int]:
    fabric = artifact_root / "evidence" / "resource-fabric"
    if not fabric.exists():
        return {"resources": 0, "caches": 0, "uses": 0, "derivations": 0}

    registrations = fabric_records(fabric, "catalog/registrations")
    registration_by_resource: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for registration in registrations:
        registration_by_resource[registration.get("resource_id", "")].append(registration)
    lifecycle = fabric_records(fabric, "catalog/lifecycle")
    latest_lifecycle: dict[str, dict[str, Any]] = {}
    for event in sorted(lifecycle, key=lambda item: (item.get("recorded_at", ""), item.get("event_id", ""))):
        latest_lifecycle[event.get("resource_id", "")] = event

    resources = 0
    caches_count = 0
    uses = 0
    derivations_count = 0
    for obj in fabric_records(fabric, "catalog/objects"):
        resource_id = obj["resource_id"]
        locator = normalize_locator(obj["canonical_path"])
        regs = registration_by_resource.get(resource_id, [])
        latest_registration = sorted(
            regs, key=lambda item: item.get("registered_at", "")
        )[-1] if regs else {}
        latest_event = latest_lifecycle.get(resource_id, {})
        inventory_path = artifact_root / obj["inventory_path"]
        inventory = read_json(inventory_path) if inventory_path.is_file() else {"files": []}
        entries = [
            {
                "relative_path": item.get("path", "."),
                "bytes": int(item.get("bytes", 0)),
                "mtime_ns": 0,
                "sha256": item.get("sha256"),
            }
            for item in inventory.get("files", [])
        ]
        record = {
            "asset_id": resource_id,
            "locator": locator,
            "logical_name": latest_registration.get("name", resource_id),
            "root_name": "resource-fabric",
            "asset_kind": latest_registration.get("kind", "resource"),
            "asset_class": "managed_resource",
            "entry_type": "file" if obj.get("payload_type") == "file" else "directory",
            "state": "present" if path_for_locator(locator, artifact_root).exists() else "missing",
            "bytes": int(obj.get("bytes", 0)),
            "file_count": int(obj.get("file_count", 0)),
            "metadata_sha256": obj.get("sha256"),
            "content_id": resource_id,
            "identity_strength": "content",
            "evidence_status": latest_event.get("evidence_status", "unknown"),
            "storage_status": latest_event.get("storage_status", "unknown"),
            "owner": latest_registration.get("owner", "shared-research"),
            "retention_reason": latest_registration.get(
                "retention_reason", "Managed immutable resource"
            ),
            "rebuild_command": latest_registration.get("rebuild_command"),
            "rebuild_cost": latest_registration.get("rebuild_cost"),
            "source_uri": latest_registration.get("source_uri"),
            "license_id": latest_registration.get("license_id"),
            "claim_ceiling": "RESOURCE_LIFECYCLE_BOUNDARY",
            "first_seen_at": obj.get("created_at", now),
            "last_seen_at": now,
            "last_scan_id": scan_id,
            "metadata_json": json.dumps(
                {"object": obj, "registrations": regs},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        upsert_asset(connection, record, entries, preserve_lifecycle=False)
        resources += 1
        for registration in regs:
            consumer = registration.get("consumer")
            if not consumer:
                continue
            record_usage(
                connection,
                asset_id=resource_id,
                consumer=consumer,
                purpose=registration.get("evidence_role", "registered-consumer"),
                experiment_id=None,
                access_mode="input",
                evidence_effect="none",
                metadata={"source": "resource-fabric-registration"},
                event_id=f"fabric-registration:{registration['registration_id']}",
                recorded_at=registration.get("registered_at"),
                ignore_existing=True,
            )
            uses += 1
        for event in [item for item in lifecycle if item.get("resource_id") == resource_id]:
            connection.execute(
                """
                INSERT OR IGNORE INTO lifecycle_events(
                    event_id, asset_id, evidence_status, storage_status,
                    reason, claim_ceiling, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"fabric-lifecycle:{resource_id}:{event.get('event_id')}",
                    resource_id,
                    event.get("evidence_status", "unknown"),
                    event.get("storage_status", "unknown"),
                    event.get("reason", "Imported resource-fabric lifecycle"),
                    "RESOURCE_LIFECYCLE_BOUNDARY",
                    event.get("recorded_at", now),
                ),
            )

    for cache in fabric_records(fabric, "catalog/caches"):
        cache_key = cache["cache_key"]
        asset_id = f"cache:{cache_key}"
        locator = normalize_locator(cache["payload_path"])
        content_prefix = "sha256" if cache.get("payload_type") == "file" else "tree-sha256"
        record = {
            "asset_id": asset_id,
            "locator": locator,
            "logical_name": f"{cache.get('layer', 'cache')}-{cache_key[:12]}",
            "root_name": "resource-fabric",
            "asset_kind": cache.get("layer", "cache"),
            "asset_class": "shared_cache",
            "entry_type": "file" if cache.get("payload_type") == "file" else "directory",
            "state": "present" if path_for_locator(locator, artifact_root).exists() else "missing",
            "bytes": int(cache.get("payload_bytes", 0)),
            "file_count": int(cache.get("payload_file_count", 0)),
            "metadata_sha256": cache.get("payload_sha256"),
            "content_id": f"{content_prefix}:{cache.get('payload_sha256')}",
            "identity_strength": "content",
            "evidence_status": "diagnostic",
            "storage_status": "shared",
            "owner": cache.get("producer", "shared-research"),
            "retention_reason": "Deterministic shared cache with recorded lineage",
            "rebuild_command": cache.get("producer"),
            "rebuild_cost": None,
            "source_uri": None,
            "license_id": None,
            "claim_ceiling": "DERIVED_CACHE_ONLY",
            "first_seen_at": cache.get("created_at", now),
            "last_seen_at": now,
            "last_scan_id": scan_id,
            "metadata_json": json.dumps(cache, ensure_ascii=False, sort_keys=True),
        }
        entries = [
            {
                "relative_path": ".",
                "bytes": record["bytes"],
                "mtime_ns": 0,
                "sha256": cache.get("payload_sha256"),
            }
        ]
        upsert_asset(connection, record, entries, preserve_lifecycle=False)
        caches_count += 1
        input_ids = list(cache.get("source_ids", [])) + [
            f"cache:{key}" for key in cache.get("parent_cache_keys", [])
        ] + list(cache.get("model_ids", []))
        present_inputs = [
            item
            for item in input_ids
            if connection.execute("SELECT 1 FROM assets WHERE asset_id = ?", (item,)).fetchone()
        ]
        derivation_id = f"fabric-cache:{cache_key}"
        connection.execute(
            """
            INSERT OR IGNORE INTO derivations(
                derivation_id, output_asset_id, transform, transform_version,
                producer, code_sha256, config_sha256, parameters_json, recorded_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                derivation_id,
                asset_id,
                cache.get("transform", "unknown"),
                cache.get("transform_version", "unknown"),
                cache.get("producer", "unknown"),
                cache.get("code_sha256"),
                cache.get("config_sha256"),
                json.dumps(cache.get("parameters", {}), ensure_ascii=False, sort_keys=True),
                cache.get("created_at", now),
            ),
        )
        for input_id in present_inputs:
            connection.execute(
                """
                INSERT OR IGNORE INTO derivation_inputs(
                    derivation_id, input_asset_id, input_role
                ) VALUES(?, ?, 'source')
                """,
                (derivation_id, input_id),
            )
        derivations_count += 1

    experiment_root = fabric / "experiments"
    if experiment_root.exists():
        for manifest_path in sorted(experiment_root.rglob("manifest.json")):
            manifest = read_json(manifest_path)
            consumer = manifest.get("id", manifest_path.parent.name)
            referenced = list(manifest.get("source_ids", [])) + [
                f"cache:{key}" for key in manifest.get("cache_keys", [])
            ]
            for asset_id in referenced:
                if connection.execute(
                    "SELECT 1 FROM assets WHERE asset_id = ?", (asset_id,)
                ).fetchone() is None:
                    continue
                record_usage(
                    connection,
                    asset_id=asset_id,
                    consumer=consumer,
                    purpose="thin-experiment-input",
                    experiment_id=consumer,
                    access_mode="input",
                    evidence_effect="none",
                    metadata={"manifest": manifest_path.relative_to(artifact_root).as_posix()},
                    event_id=f"fabric-experiment:{consumer}:{asset_id}",
                    recorded_at=manifest.get("created_at", now),
                    ignore_existing=True,
                )
                uses += 1

    hard_case_root = fabric / "hard-cases"
    if hard_case_root.exists():
        for case_path in sorted(hard_case_root.rglob("*.json")):
            case = read_json(case_path)
            consumer = f"hard-case:{case.get('id', case_path.stem)}"
            referenced = list(case.get("source_ids", [])) + [
                f"cache:{key}" for key in case.get("cache_keys", [])
            ]
            for asset_id in referenced:
                if connection.execute(
                    "SELECT 1 FROM assets WHERE asset_id = ?", (asset_id,)
                ).fetchone() is None:
                    continue
                record_usage(
                    connection,
                    asset_id=asset_id,
                    consumer=consumer,
                    purpose="hard-case-reference",
                    experiment_id=None,
                    access_mode="diagnostic",
                    evidence_effect="none",
                    metadata={"case": case_path.relative_to(artifact_root).as_posix()},
                    event_id=f"fabric-hard-case:{consumer}:{asset_id}",
                    recorded_at=case.get("created_at", now),
                    ignore_existing=True,
                )
                uses += 1

    return {
        "resources": resources,
        "caches": caches_count,
        "uses": uses,
        "derivations": derivations_count,
    }


def discover_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    policy = load_policy(args.policy.resolve())
    database = args.database.resolve()
    connection = open_catalog(database)
    scan_id = f"scan:{utc_now()}:{uuid.uuid4().hex[:8]}"
    now = utc_now()
    policy_sha256 = sha256_bytes(canonical_json_bytes(policy))
    connection.execute(
        """
        INSERT INTO scan_runs(scan_id, started_at, status, policy_sha256)
        VALUES(?, ?, 'running', ?)
        """,
        (scan_id, now, policy_sha256),
    )
    connection.commit()

    assets_seen = 0
    files_seen = 0
    bytes_seen = 0
    vanished_total = 0
    reparse_total = 0
    errors: list[dict[str, str]] = []
    scanned_roots: set[str] = set()
    top_level_file_assets: list[Path] = []
    try:
        actual_entries = {
            entry.name: Path(entry.path)
            for entry in os.scandir(artifact_root)
        }
        root_names = sorted(
            set(actual_entries)
            | set(policy.get("roots", {}))
            | set(policy.get("excluded_roots", {})),
            key=str.casefold,
        )
        for root_name in root_names:
            path = actual_entries.get(root_name)
            rule = policy.get("roots", {}).get(root_name)
            exclusion_reason = policy.get("excluded_roots", {}).get(root_name)
            if path is None:
                connection.execute(
                    "UPDATE assets SET state = 'missing' WHERE root_name = ?",
                    (root_name,),
                )
                upsert_root(
                    connection,
                    root_name=root_name,
                    disposition="missing",
                    rule=rule,
                    reason="Configured root is absent",
                    state="missing",
                    asset_count=0,
                    file_count=0,
                    bytes_count=0,
                    vanished=0,
                    reparse_count=0,
                    scan_id=scan_id,
                    now=now,
                )
                continue
            try:
                if is_reparse_point(path):
                    connection.execute(
                        "UPDATE assets SET state = 'excluded_by_policy' WHERE root_name = ?",
                        (root_name,),
                    )
                    upsert_root(
                        connection,
                        root_name=root_name,
                        disposition="reparse_excluded",
                        rule=rule,
                        reason="Top-level reparse alias is not followed",
                        state="present",
                        asset_count=0,
                        file_count=0,
                        bytes_count=0,
                        vanished=0,
                        reparse_count=1,
                        scan_id=scan_id,
                        now=now,
                    )
                    reparse_total += 1
                    continue
            except FileNotFoundError:
                vanished_total += 1
                continue
            if path.is_file():
                if any(
                    fnmatch.fnmatch(root_name.casefold(), pattern.casefold())
                    for pattern in policy.get("top_level_file_exclude_globs", [])
                ):
                    upsert_root(
                        connection,
                        root_name=root_name,
                        disposition="excluded",
                        rule=None,
                        reason="Top-level file matches an exclusion pattern",
                        state="present",
                        asset_count=0,
                        file_count=0,
                        bytes_count=0,
                        vanished=0,
                        reparse_count=0,
                        scan_id=scan_id,
                        now=now,
                    )
                else:
                    top_level_file_assets.append(path)
                continue
            if exclusion_reason:
                connection.execute(
                    "UPDATE assets SET state = 'excluded_by_policy' WHERE root_name = ?",
                    (root_name,),
                )
                upsert_root(
                    connection,
                    root_name=root_name,
                    disposition="excluded",
                    rule=None,
                    reason=exclusion_reason,
                    state="present",
                    asset_count=0,
                    file_count=0,
                    bytes_count=0,
                    vanished=0,
                    reparse_count=0,
                    scan_id=scan_id,
                    now=now,
                )
                print(f"[asset-catalog] excluded root={root_name} reason={exclusion_reason}", file=sys.stderr, flush=True)
                continue
            rule = rule or policy["fallback"]
            excluded_children = {name.casefold() for name in rule.get("exclude_children", [])}
            scanned_roots.add(root_name)
            root_assets = 0
            root_files = 0
            root_bytes = 0
            root_vanished = 0
            root_reparse = 0
            print(f"[asset-catalog] scanning root={root_name}", file=sys.stderr, flush=True)
            try:
                children = sorted(os.scandir(path), key=lambda item: item.name.casefold())
            except FileNotFoundError:
                children = []
                root_vanished += 1
            for child in children:
                if child.name.casefold() in excluded_children:
                    continue
                child_path = Path(child.path)
                try:
                    if is_reparse_point(child_path):
                        root_reparse += 1
                        continue
                    scan = scan_asset(child_path)
                    locator = locator_for_path(child_path, artifact_root)
                    record = discovered_record(locator, root_name, rule, scan, scan_id, now)
                    with connection:
                        upsert_asset(connection, record, scan["entries"])
                    root_assets += 1
                    root_files += scan["file_count"]
                    root_bytes += scan["bytes"]
                    root_vanished += scan["vanished_entries"]
                    root_reparse += scan["reparse_entries"]
                    if root_assets % 25 == 0:
                        print(
                            f"[asset-catalog] root={root_name} assets={root_assets} files={root_files} bytes={root_bytes}",
                            file=sys.stderr,
                            flush=True,
                        )
                except FileNotFoundError:
                    root_vanished += 1
                except (CatalogError, OSError) as exc:
                    errors.append({"locator": str(child_path), "error": str(exc)})
            upsert_root(
                connection,
                root_name=root_name,
                disposition="cataloged",
                rule=rule,
                reason=rule["retention_reason"],
                state="present",
                asset_count=root_assets,
                file_count=root_files,
                bytes_count=root_bytes,
                vanished=root_vanished,
                reparse_count=root_reparse,
                scan_id=scan_id,
                now=now,
            )
            connection.commit()
            assets_seen += root_assets
            files_seen += root_files
            bytes_seen += root_bytes
            vanished_total += root_vanished
            reparse_total += root_reparse
            print(
                f"[asset-catalog] complete root={root_name} assets={root_assets} files={root_files} bytes={root_bytes}",
                file=sys.stderr,
                flush=True,
            )

        if top_level_file_assets:
            root_name = "__root_files__"
            rule = policy.get("top_level_file_rule", policy["fallback"])
            scanned_roots.add(root_name)
            root_bytes = 0
            root_files = 0
            for path in top_level_file_assets:
                try:
                    scan = scan_asset(path)
                    locator = locator_for_path(path, artifact_root)
                    record = discovered_record(locator, root_name, rule, scan, scan_id, now)
                    with connection:
                        upsert_asset(connection, record, scan["entries"])
                    assets_seen += 1
                    root_files += scan["file_count"]
                    root_bytes += scan["bytes"]
                except (CatalogError, OSError) as exc:
                    errors.append({"locator": str(path), "error": str(exc)})
            files_seen += root_files
            bytes_seen += root_bytes
            upsert_root(
                connection,
                root_name=root_name,
                disposition="cataloged",
                rule=rule,
                reason="Stable top-level files",
                state="present",
                asset_count=len(top_level_file_assets),
                file_count=root_files,
                bytes_count=root_bytes,
                vanished=0,
                reparse_count=0,
                scan_id=scan_id,
                now=now,
            )

        if scanned_roots:
            placeholders = ",".join("?" for _ in scanned_roots)
            connection.execute(
                f"""
                UPDATE assets SET state = 'missing'
                WHERE root_name IN ({placeholders}) AND COALESCE(last_scan_id, '') != ?
                """,
                (*sorted(scanned_roots), scan_id),
            )
        fabric_summary = sync_resource_fabric(connection, artifact_root, scan_id, now)
        reference_summary = import_repository_references(
            connection,
            args.repo_root.resolve(),
            artifact_root,
            scan_id,
        )
        connection.execute(
            """
            UPDATE scan_runs SET
                completed_at = ?, status = ?, assets_seen = ?, files_seen = ?,
                bytes_seen = ?, vanished_entries = ?, reparse_entries = ?,
                errors_json = ?
            WHERE scan_id = ?
            """,
            (
                utc_now(),
                "complete" if not errors else "partial",
                assets_seen + fabric_summary["resources"] + fabric_summary["caches"],
                files_seen,
                bytes_seen,
                vanished_total,
                reparse_total,
                json.dumps(errors, ensure_ascii=False, sort_keys=True),
                scan_id,
            ),
        )
        connection.commit()
        return {
            "status": "PASS" if not errors else "PARTIAL",
            "scan_id": scan_id,
            "database": str(database),
            "assets_seen": assets_seen,
            "files_seen": files_seen,
            "bytes_seen": bytes_seen,
            "vanished_entries": vanished_total,
            "reparse_entries": reparse_total,
            "errors": errors,
            "resource_fabric": fabric_summary,
            "repository_references": reference_summary,
        }
    except Exception:
        connection.execute(
            "UPDATE scan_runs SET completed_at = ?, status = 'failed' WHERE scan_id = ?",
            (utc_now(), scan_id),
        )
        connection.commit()
        raise
    finally:
        connection.close()


def register_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    path = args.path.resolve()
    locator = locator_for_path(path, artifact_root)
    scan = scan_asset(path)
    now = utc_now()
    rule = {
        "asset_kind": args.kind,
        "asset_class": args.asset_class,
        "evidence_status": args.evidence_status,
        "storage_status": args.storage_status,
        "owner": args.owner,
        "retention_reason": args.retention_reason,
        "claim_ceiling": args.claim_ceiling,
        "source_uri": args.source_uri,
        "license_id": args.license_id,
        "rebuild_command": args.rebuild_command,
        "rebuild_cost": args.rebuild_cost,
    }
    record = discovered_record(locator, locator.split("/", 1)[0], rule, scan, "manual", now)
    connection = open_catalog(args.database.resolve())
    try:
        with connection:
            upsert_asset(connection, record, scan["entries"])
        return {key: record[key] for key in ("asset_id", "locator", "bytes", "file_count", "identity_strength")}
    finally:
        connection.close()


def references_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        return import_repository_references(
            connection,
            args.repo_root.resolve(),
            args.artifact_root.resolve(),
            f"reference-scan:{utc_now()}:{uuid.uuid4().hex[:8]}",
        )
    finally:
        connection.close()


def list_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        filters = {
            "root_name": args.root,
            "asset_kind": args.kind,
            "asset_class": args.asset_class,
            "state": args.state,
            "evidence_status": args.evidence_status,
            "storage_status": args.storage_status,
            "identity_strength": args.identity_strength,
        }
        clauses: list[str] = []
        parameters: list[Any] = []
        for field, value in filters.items():
            if value is not None:
                clauses.append(f"{field} = ?")
                parameters.append(value)
        if args.contains:
            clauses.append("(locator LIKE ? OR logical_name LIKE ?)")
            pattern = f"%{args.contains}%"
            parameters.extend([pattern, pattern])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        total = connection.execute(
            f"SELECT COUNT(*) FROM assets{where}", parameters
        ).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT asset_id, locator, logical_name, root_name, asset_kind,
                   asset_class, state, bytes, file_count, identity_strength,
                   content_id, evidence_status, storage_status, owner,
                   claim_ceiling
            FROM assets{where}
            ORDER BY bytes DESC, locator LIMIT ?
            """,
            (*parameters, args.limit),
        ).fetchall()
        return {
            "total": int(total),
            "returned": len(rows),
            "assets": [dict(row) for row in rows],
        }
    finally:
        connection.close()


def resolve_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        asset = resolve_asset(connection, args.key)
        if asset["state"] != "present" and not args.allow_missing:
            raise CatalogError(f"Asset is not present: {asset['locator']} state={asset['state']}")
        path = path_for_locator(asset["locator"], args.artifact_root.resolve())
        if not path.exists() and not args.allow_missing:
            raise CatalogError(f"Asset locator is missing: {path}")
        event_id = None
        if args.consumer or args.purpose:
            if not args.consumer or not args.purpose:
                raise CatalogError("--consumer and --purpose must be supplied together")
            with connection:
                event_id = record_usage(
                    connection,
                    asset_id=asset["asset_id"],
                    consumer=args.consumer,
                    purpose=args.purpose,
                    experiment_id=args.experiment_id,
                    access_mode=args.access_mode,
                    evidence_effect=args.evidence_effect,
                    metadata=parse_json(args.metadata_json, default={}),
                )
            asset = resolve_asset(connection, asset["asset_id"])
        return {
            "asset_id": asset["asset_id"],
            "logical_name": asset["logical_name"],
            "locator": asset["locator"],
            "path": str(path),
            "content_id": asset["content_id"],
            "identity_strength": asset["identity_strength"],
            "evidence_status": asset["evidence_status"],
            "storage_status": asset["storage_status"],
            "claim_ceiling": asset["claim_ceiling"],
            "usage_event_id": event_id,
        }
    finally:
        connection.close()


def consume_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        asset = resolve_asset(connection, args.key)
        with connection:
            event_id = record_usage(
                connection,
                asset_id=asset["asset_id"],
                consumer=args.consumer,
                purpose=args.purpose,
                experiment_id=args.experiment_id,
                access_mode=args.access_mode,
                evidence_effect=args.evidence_effect,
                metadata=parse_json(args.metadata_json, default={}),
                event_id=args.event_id,
            )
        updated = resolve_asset(connection, asset["asset_id"])
        return {
            "event_id": event_id,
            "asset_id": asset["asset_id"],
            "evidence_status": updated["evidence_status"],
            "storage_status": updated["storage_status"],
        }
    finally:
        connection.close()


def transition_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        asset = resolve_asset(connection, args.key)
        with connection:
            event_id = transition_asset(
                connection,
                asset_id=asset["asset_id"],
                evidence_status=args.evidence_status,
                storage_status=args.storage_status,
                reason=args.reason,
                claim_ceiling=args.claim_ceiling,
                event_id=args.event_id,
            )
        updated = resolve_asset(connection, asset["asset_id"])
        return {
            "event_id": event_id,
            "asset_id": asset["asset_id"],
            "evidence_status": updated["evidence_status"],
            "storage_status": updated["storage_status"],
            "claim_ceiling": updated["claim_ceiling"],
        }
    finally:
        connection.close()


def hash_asset_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    connection = open_catalog(args.database.resolve())
    try:
        asset = resolve_asset(connection, args.key)
        if asset["state"] != "present":
            raise CatalogError(f"Cannot hash non-present asset: {asset['locator']}")
        path = path_for_locator(asset["locator"], artifact_root)
        scan = scan_asset(path)
        hashed_entries: list[dict[str, Any]] = []
        for index, item in enumerate(scan["entries"], start=1):
            file_path = path if scan["entry_type"] == "file" else path / item["relative_path"]
            digest = sha256_file(file_path)
            hashed_entries.append({**item, "sha256": digest})
            if index % 1000 == 0:
                print(
                    f"[asset-catalog] hashing asset={asset['asset_id']} files={index}/{scan['file_count']}",
                    file=sys.stderr,
                    flush=True,
                )
        if scan["entry_type"] == "file":
            content_id = f"sha256:{hashed_entries[0]['sha256']}"
        else:
            tree_material = {
                "algorithm": "blindassist-tree-sha256-v1",
                "files": [
                    {
                        "path": item["relative_path"],
                        "bytes": item["bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in hashed_entries
                ],
            }
            content_id = f"tree-sha256:{sha256_bytes(canonical_json_bytes(tree_material))}"
        with connection:
            connection.execute(
                """
                UPDATE assets SET
                    bytes = ?, file_count = ?, metadata_sha256 = ?,
                    content_id = ?, identity_strength = 'content',
                    last_seen_at = ?
                WHERE asset_id = ?
                """,
                (
                    scan["bytes"],
                    scan["file_count"],
                    scan["metadata_sha256"],
                    content_id,
                    utc_now(),
                    asset["asset_id"],
                ),
            )
            connection.execute("DELETE FROM asset_files WHERE asset_id = ?", (asset["asset_id"],))
            connection.executemany(
                """
                INSERT INTO asset_files(asset_id, relative_path, bytes, mtime_ns, sha256)
                VALUES(?, ?, ?, ?, ?)
                """,
                [
                    (
                        asset["asset_id"],
                        item["relative_path"],
                        item["bytes"],
                        item["mtime_ns"],
                        item["sha256"],
                    )
                    for item in hashed_entries
                ],
            )
        duplicates = connection.execute(
            "SELECT asset_id, locator FROM assets WHERE content_id = ? ORDER BY locator",
            (content_id,),
        ).fetchall()
        return {
            "asset_id": asset["asset_id"],
            "content_id": content_id,
            "bytes": scan["bytes"],
            "file_count": scan["file_count"],
            "duplicate_assets": [dict(row) for row in duplicates if row["asset_id"] != asset["asset_id"]],
        }
    finally:
        connection.close()


def derive_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        output = resolve_asset(connection, args.output)
        inputs = [resolve_asset(connection, key) for key in args.input]
        for label, value in (("code", args.code_sha256), ("config", args.config_sha256)):
            if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise CatalogError(f"Invalid {label} SHA-256: {value}")
        parameters = parse_json(args.parameters_json, default={})
        spec = {
            "output_asset_id": output["asset_id"],
            "input_asset_ids": sorted(row["asset_id"] for row in inputs),
            "transform": args.transform,
            "transform_version": args.transform_version,
            "producer": args.producer,
            "code_sha256": args.code_sha256,
            "config_sha256": args.config_sha256,
            "parameters": parameters,
        }
        derivation_id = args.derivation_id or f"derive:{sha256_bytes(canonical_json_bytes(spec))}"
        with connection:
            existing = connection.execute(
                "SELECT * FROM derivations WHERE derivation_id = ?", (derivation_id,)
            ).fetchone()
            if existing:
                raise CatalogError(f"Derivation already exists: {derivation_id}")
            connection.execute(
                """
                INSERT INTO derivations(
                    derivation_id, output_asset_id, transform, transform_version,
                    producer, code_sha256, config_sha256, parameters_json, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    derivation_id,
                    output["asset_id"],
                    args.transform,
                    args.transform_version,
                    args.producer,
                    args.code_sha256,
                    args.config_sha256,
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )
            for input_asset in inputs:
                connection.execute(
                    """
                    INSERT INTO derivation_inputs(
                        derivation_id, input_asset_id, input_role
                    ) VALUES(?, ?, 'source')
                    """,
                    (derivation_id, input_asset["asset_id"]),
                )
                record_usage(
                    connection,
                    asset_id=input_asset["asset_id"],
                    consumer=output["asset_id"],
                    purpose=f"derive:{args.transform}",
                    experiment_id=args.experiment_id,
                    access_mode="derivation_input",
                    evidence_effect=args.evidence_effect,
                    metadata={"derivation_id": derivation_id},
                )
        return {
            "derivation_id": derivation_id,
            "output_asset_id": output["asset_id"],
            "input_asset_ids": [row["asset_id"] for row in inputs],
        }
    finally:
        connection.close()


def grouped_counts(connection: sqlite3.Connection, field: str) -> dict[str, dict[str, int]]:
    if field not in {
        "root_name",
        "asset_kind",
        "asset_class",
        "evidence_status",
        "storage_status",
        "identity_strength",
        "state",
    }:
        raise CatalogError(f"Unsupported grouping field: {field}")
    rows = connection.execute(
        f"""
        SELECT {field} AS name, COUNT(*) AS assets,
               COALESCE(SUM(bytes), 0) AS bytes,
               COALESCE(SUM(file_count), 0) AS files
        FROM assets GROUP BY {field} ORDER BY bytes DESC, name
        """
    ).fetchall()
    return {
        row["name"]: {
            "assets": int(row["assets"]),
            "bytes": int(row["bytes"]),
            "files": int(row["files"]),
        }
        for row in rows
    }


def report_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        catalog_records = connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        nonpresent_assets = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE state != 'present'"
        ).fetchone()[0]
        summary_row = connection.execute(
            """
            SELECT COUNT(*) AS assets,
                   COALESCE(SUM(bytes), 0) AS logical_bytes,
                   COALESCE(SUM(file_count), 0) AS files,
                   SUM(CASE WHEN identity_strength = 'content' THEN 1 ELSE 0 END) AS content_assets,
                   SUM(CASE WHEN asset_class = 'legacy_unclassified' THEN 1 ELSE 0 END) AS unclassified_assets,
                   SUM(CASE WHEN evidence_status = 'unknown' THEN 1 ELSE 0 END) AS unknown_evidence_assets,
                   SUM(CASE WHEN storage_status = 'unknown' THEN 1 ELSE 0 END) AS unknown_storage_assets
            FROM assets WHERE state = 'present'
            """
        ).fetchone()
        usage_count = connection.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
        consumers = connection.execute("SELECT COUNT(DISTINCT consumer) FROM usage_events").fetchone()[0]
        derivations = connection.execute("SELECT COUNT(*) FROM derivations").fetchone()[0]
        reference_row = connection.execute(
            """
            SELECT COUNT(*) AS reference_count,
                   SUM(CASE WHEN resolution = 'asset' THEN 1 ELSE 0 END) AS resolved_assets,
                   SUM(CASE WHEN resolution = 'root' THEN 1 ELSE 0 END) AS root_scoped,
                   SUM(CASE WHEN resolution = 'missing_within_root' THEN 1 ELSE 0 END) AS missing_within_root,
                   SUM(CASE WHEN resolution = 'template' THEN 1 ELSE 0 END) AS templates,
                   SUM(CASE WHEN resolution = 'unknown_root' THEN 1 ELSE 0 END) AS unknown_roots,
                   COUNT(DISTINCT asset_id) AS referenced_assets
            FROM asset_references
            """
        ).fetchone()
        unused_count = connection.execute(
            """
            SELECT COUNT(*) FROM assets a
            WHERE a.state = 'present'
              AND NOT EXISTS(SELECT 1 FROM usage_events u WHERE u.asset_id = a.asset_id)
              AND NOT EXISTS(SELECT 1 FROM asset_references r WHERE r.asset_id = a.asset_id)
            """
        ).fetchone()[0]
        unconsumed_count = connection.execute(
            """
            SELECT COUNT(*) FROM assets a
            WHERE a.state = 'present' AND NOT EXISTS(
                SELECT 1 FROM usage_events u WHERE u.asset_id = a.asset_id
            )
            """
        ).fetchone()[0]
        unconsumed = connection.execute(
            """
            SELECT a.asset_id, a.locator, a.asset_kind, a.bytes
            FROM assets a LEFT JOIN usage_events u ON u.asset_id = a.asset_id
            WHERE a.state = 'present' AND u.asset_id IS NULL
            GROUP BY a.asset_id ORDER BY a.bytes DESC LIMIT 100
            """
        ).fetchall()
        duplicate_count = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT content_id FROM assets
                WHERE content_id IS NOT NULL AND state = 'present'
                GROUP BY content_id HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        duplicates = connection.execute(
            """
            SELECT content_id, COUNT(*) AS assets, MAX(bytes) AS bytes,
                   GROUP_CONCAT(locator, ' | ') AS locators
            FROM assets
            WHERE content_id IS NOT NULL AND state = 'present'
            GROUP BY content_id HAVING COUNT(*) > 1
            ORDER BY bytes DESC LIMIT 100
            """
        ).fetchall()
        largest = connection.execute(
            """
            SELECT asset_id, locator, asset_kind, asset_class, bytes,
                   file_count, identity_strength, evidence_status, storage_status
            FROM assets WHERE state = 'present'
            ORDER BY bytes DESC LIMIT 100
            """
        ).fetchall()
        unresolved_references = connection.execute(
            """
            SELECT source_file, line_number, raw_locator, resolution, root_name
            FROM asset_references
            WHERE resolution IN ('missing_within_root', 'unknown_root')
            ORDER BY source_file, line_number LIMIT 200
            """
        ).fetchall()
        roots = [dict(row) for row in connection.execute("SELECT * FROM roots ORDER BY root_name")]
        scans = [dict(row) for row in connection.execute(
            "SELECT * FROM scan_runs ORDER BY started_at DESC LIMIT 10"
        )]
        summary = {
            "assets": int(summary_row["assets"] or 0),
            "catalog_records": int(catalog_records),
            "logical_bytes": int(summary_row["logical_bytes"] or 0),
            "files": int(summary_row["files"] or 0),
            "content_verified_assets": int(summary_row["content_assets"] or 0),
            "metadata_identity_assets": int(summary_row["assets"] or 0) - int(summary_row["content_assets"] or 0),
            "unclassified_assets": int(summary_row["unclassified_assets"] or 0),
            "unknown_evidence_assets": int(summary_row["unknown_evidence_assets"] or 0),
            "unknown_storage_assets": int(summary_row["unknown_storage_assets"] or 0),
            "nonpresent_assets": int(nonpresent_assets),
            "usage_events": int(usage_count),
            "consumers": int(consumers),
            "derivations": int(derivations),
            "repository_references": int(reference_row["reference_count"] or 0),
            "resolved_repository_references": int(reference_row["resolved_assets"] or 0),
            "root_scoped_repository_references": int(reference_row["root_scoped"] or 0),
            "template_repository_references": int(reference_row["templates"] or 0),
            "missing_locator_references": int(reference_row["missing_within_root"] or 0),
            "unknown_root_references": int(reference_row["unknown_roots"] or 0),
            "unresolved_repository_references": int(reference_row["missing_within_root"] or 0)
            + int(reference_row["unknown_roots"] or 0),
            "referenced_assets": int(reference_row["referenced_assets"] or 0),
            "unconsumed_assets": int(unconsumed_count),
            "unused_and_unreferenced_assets": int(unused_count),
            "duplicate_content_groups": int(duplicate_count),
        }
        report = {
            "schema": "blindassist-master-asset-report-v1",
            "generated_at": utc_now(),
            "database": str(args.database.resolve()),
            "summary": summary,
            "groups": {
                field: grouped_counts(connection, field)
                for field in (
                    "root_name",
                    "asset_kind",
                    "asset_class",
                    "evidence_status",
                    "storage_status",
                    "identity_strength",
                    "state",
                )
            },
            "roots": roots,
            "recent_scans": scans,
            "largest_assets": [dict(row) for row in largest],
            "unconsumed_assets": [dict(row) for row in unconsumed],
            "unresolved_repository_references": [dict(row) for row in unresolved_references],
            "duplicate_content_groups": [dict(row) for row in duplicates],
        }
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "master-asset-report.json"
        markdown_path = output_dir / "master-asset-report.md"
        atomic_write_json(json_path, report)
        lines = [
            "# BlindAssist master asset report",
            "",
            f"Generated: `{report['generated_at']}`",
            "",
            "Logical bytes count references, not deduplicated physical blocks.",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
        for key, value in summary.items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.extend([
            "",
            "## Root coverage",
            "",
            "| Root | Disposition | State | Assets | Files | Bytes | Reason |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ])
        for root in roots:
            reason = str(root["reason"]).replace("|", "\\|")
            lines.append(
                f"| `{root['root_name']}` | `{root['disposition']}` | `{root['state']}` | "
                f"`{root['asset_count']}` | `{root['file_count']}` | `{root['bytes']}` | {reason} |"
            )
        lines.extend([
            "",
            "## Largest assets",
            "",
            "| Locator | Kind | Class | Bytes | Files | Identity | Evidence | Storage |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
        ])
        for item in largest[:50]:
            lines.append(
                f"| `{item['locator']}` | `{item['asset_kind']}` | `{item['asset_class']}` | "
                f"`{item['bytes']}` | `{item['file_count']}` | `{item['identity_strength']}` | "
                f"`{item['evidence_status']}` | `{item['storage_status']}` |"
            )
        atomic_write_text(markdown_path, "\n".join(lines).rstrip() + "\n")
        return {
            "report_json": str(json_path),
            "report_markdown": str(markdown_path),
            **summary,
        }
    finally:
        connection.close()


def verify_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    connection = open_catalog(args.database.resolve())
    errors: list[str] = []
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            errors.append(f"sqlite integrity: {integrity}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            errors.append(f"foreign key failures: {len(foreign_keys)}")
        assets = connection.execute("SELECT * FROM assets ORDER BY locator").fetchall()
        deep_checked = 0
        for asset in assets:
            try:
                path = path_for_locator(asset["locator"], artifact_root)
                if asset["state"] == "present" and not path.exists():
                    errors.append(f"missing present asset: {asset['locator']}")
                    continue
                if not args.deep or asset["identity_strength"] != "content" or not path.exists():
                    continue
                scan = scan_asset(path)
                hashed: list[dict[str, Any]] = []
                for item in scan["entries"]:
                    file_path = path if scan["entry_type"] == "file" else path / item["relative_path"]
                    hashed.append({
                        "path": item["relative_path"],
                        "bytes": item["bytes"],
                        "sha256": sha256_file(file_path),
                    })
                if scan["entry_type"] == "file":
                    actual = f"sha256:{hashed[0]['sha256']}"
                else:
                    actual = "tree-sha256:" + sha256_bytes(canonical_json_bytes({
                        "algorithm": "blindassist-tree-sha256-v1",
                        "files": hashed,
                    }))
                if actual != asset["content_id"]:
                    errors.append(
                        f"content drift {asset['locator']}: expected={asset['content_id']} actual={actual}"
                    )
                deep_checked += 1
            except (CatalogError, OSError) as exc:
                errors.append(f"asset {asset['asset_id']}: {exc}")
        result = {
            "status": "PASS" if not errors else "FAIL",
            "assets": len(assets),
            "deep": bool(args.deep),
            "deep_checked": deep_checked,
            "errors": errors,
        }
        if errors:
            raise CatalogError(json.dumps(result, ensure_ascii=False))
        return result
    finally:
        connection.close()


def add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / DEFAULT_DATABASE_RELATIVE,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Zero-copy catalog every stable asset root")
    add_catalog_arguments(discover)
    discover.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    discover.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    discover.set_defaults(func=discover_command)

    register = subparsers.add_parser("register", help="Zero-copy register one asset unit")
    add_catalog_arguments(register)
    register.add_argument("path", type=Path)
    register.add_argument("--kind", required=True)
    register.add_argument("--asset-class", required=True)
    register.add_argument("--evidence-status", choices=sorted(VALID_EVIDENCE_STATUSES), required=True)
    register.add_argument("--storage-status", choices=sorted(VALID_STORAGE_STATUSES), required=True)
    register.add_argument("--owner", required=True)
    register.add_argument("--retention-reason", required=True)
    register.add_argument("--claim-ceiling", default="UNCLASSIFIED")
    register.add_argument("--source-uri")
    register.add_argument("--license-id")
    register.add_argument("--rebuild-command")
    register.add_argument("--rebuild-cost")
    register.set_defaults(func=register_command)

    references = subparsers.add_parser(
        "references", help="Refresh tracked repository references to cataloged assets"
    )
    add_catalog_arguments(references)
    references.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    references.set_defaults(func=references_command)

    list_parser = subparsers.add_parser("list", help="Query assets by lifecycle or classification")
    add_catalog_arguments(list_parser)
    list_parser.add_argument("--root")
    list_parser.add_argument("--kind")
    list_parser.add_argument("--asset-class")
    list_parser.add_argument("--state")
    list_parser.add_argument("--evidence-status")
    list_parser.add_argument("--storage-status")
    list_parser.add_argument("--identity-strength")
    list_parser.add_argument("--contains")
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.set_defaults(func=list_command)

    resolve = subparsers.add_parser("resolve", help="Resolve an asset and optionally record its consumer")
    add_catalog_arguments(resolve)
    resolve.add_argument("key")
    resolve.add_argument("--consumer")
    resolve.add_argument("--purpose")
    resolve.add_argument("--experiment-id")
    resolve.add_argument("--access-mode", default="input")
    resolve.add_argument(
        "--evidence-effect",
        choices=("none", "development_consumed", "sealed_final"),
        default="none",
    )
    resolve.add_argument("--metadata-json")
    resolve.add_argument("--allow-missing", action="store_true")
    resolve.set_defaults(func=resolve_command)

    consume = subparsers.add_parser("consume", help="Record an asset use without resolving a path")
    add_catalog_arguments(consume)
    consume.add_argument("key")
    consume.add_argument("--consumer", required=True)
    consume.add_argument("--purpose", required=True)
    consume.add_argument("--experiment-id")
    consume.add_argument("--access-mode", default="input")
    consume.add_argument(
        "--evidence-effect",
        choices=("none", "development_consumed", "sealed_final"),
        default="none",
    )
    consume.add_argument("--metadata-json")
    consume.add_argument("--event-id")
    consume.set_defaults(func=consume_command)

    transition = subparsers.add_parser("transition", help="Append an evidence/storage lifecycle event")
    add_catalog_arguments(transition)
    transition.add_argument("key")
    transition.add_argument("--evidence-status", choices=sorted(VALID_EVIDENCE_STATUSES))
    transition.add_argument("--storage-status", choices=sorted(VALID_STORAGE_STATUSES))
    transition.add_argument("--reason", required=True)
    transition.add_argument("--claim-ceiling")
    transition.add_argument("--event-id")
    transition.set_defaults(func=transition_command)

    hash_parser = subparsers.add_parser("hash", help="Promote one immutable asset to content identity")
    add_catalog_arguments(hash_parser)
    hash_parser.add_argument("key")
    hash_parser.set_defaults(func=hash_asset_command)

    derive = subparsers.add_parser("derive", help="Record output lineage and consume its inputs")
    add_catalog_arguments(derive)
    derive.add_argument("--output", required=True)
    derive.add_argument("--input", action="append", required=True)
    derive.add_argument("--transform", required=True)
    derive.add_argument("--transform-version", required=True)
    derive.add_argument("--producer", required=True)
    derive.add_argument("--code-sha256")
    derive.add_argument("--config-sha256")
    derive.add_argument("--parameters-json")
    derive.add_argument("--experiment-id")
    derive.add_argument(
        "--evidence-effect",
        choices=("none", "development_consumed"),
        default="none",
    )
    derive.add_argument("--derivation-id")
    derive.set_defaults(func=derive_command)

    report = subparsers.add_parser("report", help="Generate the master asset and utilization report")
    add_catalog_arguments(report)
    report.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / DEFAULT_REPORT_RELATIVE,
    )
    report.set_defaults(func=report_command)

    verify = subparsers.add_parser("verify", help="Verify catalog integrity and asset resolution")
    add_catalog_arguments(verify)
    verify.add_argument("--deep", action="store_true")
    verify.set_defaults(func=verify_command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.artifact_root = args.artifact_root.resolve()
        if hasattr(args, "database"):
            if args.database == DEFAULT_ARTIFACT_ROOT / DEFAULT_DATABASE_RELATIVE:
                args.database = args.artifact_root / DEFAULT_DATABASE_RELATIVE
            args.database = args.database.resolve()
        if hasattr(args, "output_dir"):
            if args.output_dir == DEFAULT_ARTIFACT_ROOT / DEFAULT_REPORT_RELATIVE:
                args.output_dir = args.artifact_root / DEFAULT_REPORT_RELATIVE
            args.output_dir = args.output_dir.resolve()
        result = args.func(args)
    except (CatalogError, OSError, sqlite3.Error) as exc:
        print(f"ASSET_CATALOG_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
