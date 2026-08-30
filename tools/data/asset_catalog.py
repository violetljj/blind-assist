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


SCHEMA_VERSION = 3
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
    "source_material",
    "not_applicable",
    "unknown",
}
AUTOMATIC_AUTHORITY_STATUSES = {
    "development_consumed",
    "diagnostic",
    "source_material",
    "not_applicable",
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
        overrides = rule.get("asset_overrides", {})
        if not isinstance(overrides, dict):
            raise CatalogError(f"Policy rule {name} asset_overrides must be an object")
        for child_name, child_rule in overrides.items():
            if (
                not isinstance(child_name, str)
                or not child_name
                or "/" in child_name
                or "\\" in child_name
                or child_name in {".", ".."}
            ):
                raise CatalogError(
                    f"Policy rule {name} has invalid direct-child override {child_name!r}"
                )
            validate_rule(f"{name}/asset_overrides/{child_name}", child_rule)
    managed_assets = policy.get("managed_assets", {})
    if not isinstance(managed_assets, dict):
        raise CatalogError("managed_assets must be an object keyed by locator")
    for locator, rule in managed_assets.items():
        normalize_locator(locator)
        validate_rule(f"managed_assets/{locator}", rule)
    validate_rule("fallback", policy.get("fallback", {}))
    validate_authority_classification(policy.get("authority_classification"))
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
    semantic_profile = rule.get("semantic_profile")
    if semantic_profile is not None and (
        not isinstance(semantic_profile, str) or not semantic_profile.strip()
    ):
        raise CatalogError(
            f"Policy rule {name} semantic_profile must be a repository-relative path"
        )


def validate_authority_classification(config: Any) -> None:
    if config is None:
        return
    if not isinstance(config, dict):
        raise CatalogError("authority_classification must be an object")
    if config.get("schema") != "blindassist-evidence-authority-classification-v1":
        raise CatalogError(
            "Unsupported authority classification schema: "
            f"{config.get('schema')}"
        )
    sources = config.get("current_authority_sources", [])
    if not isinstance(sources, list) or not all(
        isinstance(item, str) and item for item in sources
    ):
        raise CatalogError("current_authority_sources must be a list of paths")
    markers = config.get("outcome_reference_markers", [])
    if not isinstance(markers, list) or not all(
        isinstance(item, str) and item for item in markers
    ):
        raise CatalogError("outcome_reference_markers must be a list of tokens")
    for collection_name in (
        "static_type_rules",
        "artifact_assertion_rules",
        "exact_rules",
    ):
        rules = config.get(collection_name, [])
        if not isinstance(rules, list):
            raise CatalogError(f"{collection_name} must be a list")
        seen: set[str] = set()
        for rule in rules:
            if not isinstance(rule, dict):
                raise CatalogError(f"{collection_name} entries must be objects")
            rule_id = rule.get("rule_id")
            if not isinstance(rule_id, str) or not rule_id or rule_id in seen:
                raise CatalogError(f"Invalid or duplicate rule_id in {collection_name}")
            seen.add(rule_id)
            status = rule.get("evidence_status")
            if status not in AUTOMATIC_AUTHORITY_STATUSES:
                raise CatalogError(
                    f"Automatic authority rule {rule_id} cannot assign {status!r}; "
                    "reserved, fresh, and sealed_final require explicit manual authority"
                )
            for required in ("reason", "claim_ceiling"):
                if not isinstance(rule.get(required), str) or not rule[required]:
                    raise CatalogError(
                        f"Authority rule {rule_id} requires non-empty {required}"
                    )
            if collection_name == "static_type_rules":
                selectors = (
                    "root_names",
                    "asset_kinds",
                    "asset_classes",
                    "entry_types",
                )
                if not any(rule.get(name) for name in selectors):
                    raise CatalogError(
                        f"Static authority rule {rule_id} needs at least one selector"
                    )
            elif collection_name == "artifact_assertion_rules":
                tokens = rule.get("tokens", [])
                if not isinstance(tokens, list) or not all(
                    isinstance(item, str) and item for item in tokens
                ):
                    raise CatalogError(
                        f"Artifact assertion rule {rule_id} needs token strings"
                    )
            else:
                for required in ("locator", "authority_source", "authority_anchors"):
                    if required not in rule:
                        raise CatalogError(
                            f"Exact authority rule {rule_id} is missing {required}"
                        )
                anchors = rule["authority_anchors"]
                if not isinstance(anchors, list) or not all(
                    isinstance(item, str) and item for item in anchors
                ):
                    raise CatalogError(
                        f"Exact authority rule {rule_id} needs authority_anchors"
                    )


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

CREATE TABLE IF NOT EXISTS asset_semantic_profiles (
    asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL UNIQUE,
    profile_path TEXT NOT NULL,
    profile_sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    facts_json TEXT NOT NULL,
    last_scan_id TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS asset_components (
    component_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    component_key TEXT NOT NULL,
    logical_name TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    component_kind TEXT NOT NULL,
    data_role TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    state TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    content_id TEXT,
    identity_strength TEXT NOT NULL,
    evidence_status TEXT NOT NULL,
    claim_ceiling TEXT NOT NULL,
    description TEXT NOT NULL,
    facts_json TEXT NOT NULL,
    profile_sha256 TEXT NOT NULL,
    last_scan_id TEXT NOT NULL,
    UNIQUE(asset_id, component_key)
);

CREATE INDEX IF NOT EXISTS asset_component_asset_index
ON asset_components(asset_id, component_key);
CREATE INDEX IF NOT EXISTS asset_component_kind_index
ON asset_components(component_kind, data_role);
CREATE INDEX IF NOT EXISTS asset_component_content_index
ON asset_components(content_id);

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

CREATE TABLE IF NOT EXISTS authority_classifications (
    classification_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    prior_evidence_status TEXT NOT NULL,
    assigned_evidence_status TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    source TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS authority_classification_asset_index
ON authority_classifications(asset_id, applied_at);
CREATE INDEX IF NOT EXISTS authority_classification_rule_index
ON authority_classifications(rule_id, applied_at);

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
    *,
    discovery: str = "zero-copy",
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
                "discovery": discovery,
                "vanished_entries": scan["vanished_entries"],
                "reparse_entries": scan["reparse_entries"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def normalize_component_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if normalized in {"", "."}:
        return "."
    if re.match(r"^[A-Za-z]:", normalized) or normalized.startswith("/"):
        raise CatalogError(f"Semantic component path must be relative: {value}")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise CatalogError(f"Invalid semantic component path: {value}")
    return normalized


def sync_semantic_profile(
    connection: sqlite3.Connection,
    *,
    record: dict[str, Any],
    scan: dict[str, Any],
    rule: dict[str, Any],
    asset_path: Path,
    repo_root: Path,
    scan_id: str,
    now: str,
) -> int:
    profile_reference = rule.get("semantic_profile")
    if profile_reference is None:
        connection.execute(
            "DELETE FROM asset_components WHERE asset_id = ?",
            (record["asset_id"],),
        )
        connection.execute(
            "DELETE FROM asset_semantic_profiles WHERE asset_id = ?",
            (record["asset_id"],),
        )
        return 0

    repository = repo_root.resolve()
    profile_path = (repository / profile_reference).resolve()
    try:
        relative_profile = profile_path.relative_to(repository).as_posix()
    except ValueError as exc:
        raise CatalogError(
            f"Semantic profile escapes repository: {profile_reference}"
        ) from exc
    profile = read_json(profile_path)
    if not isinstance(profile, dict) or profile.get("schema") != "blindassist-asset-semantic-profile-v1":
        raise CatalogError(
            f"Unsupported semantic profile schema in {relative_profile}: "
            f"{profile.get('schema') if isinstance(profile, dict) else type(profile).__name__}"
        )
    for required in ("profile_id", "asset_locator", "title", "summary", "components"):
        if required not in profile:
            raise CatalogError(f"Semantic profile {relative_profile} is missing {required}")
    if normalize_locator(str(profile["asset_locator"])) != record["locator"]:
        raise CatalogError(
            f"Semantic profile locator mismatch: {relative_profile} targets "
            f"{profile['asset_locator']!r}, discovered {record['locator']!r}"
        )
    if not all(
        isinstance(profile.get(name), str) and profile[name]
        for name in ("profile_id", "title", "summary")
    ):
        raise CatalogError(f"Semantic profile {relative_profile} has empty identity fields")
    components = profile["components"]
    if not isinstance(components, list) or not components:
        raise CatalogError(f"Semantic profile {relative_profile} needs components")

    profile_sha256 = sha256_file(profile_path)
    asset_resolved = asset_path.resolve()
    component_rows: list[tuple[Any, ...]] = []
    seen_keys: set[str] = set()
    for component in components:
        if not isinstance(component, dict):
            raise CatalogError(f"Semantic profile {relative_profile} has a non-object component")
        component_key = component.get("component_key")
        if (
            not isinstance(component_key, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", component_key)
            or component_key in seen_keys
        ):
            raise CatalogError(
                f"Semantic profile {relative_profile} has invalid or duplicate component_key "
                f"{component_key!r}"
            )
        seen_keys.add(component_key)
        for required in (
            "path",
            "component_kind",
            "data_role",
            "evidence_status",
            "claim_ceiling",
            "description",
        ):
            if not isinstance(component.get(required), str) or not component[required]:
                raise CatalogError(
                    f"Semantic component {component_key} requires non-empty {required}"
                )
        evidence_status = component["evidence_status"]
        if evidence_status not in AUTOMATIC_AUTHORITY_STATUSES:
            raise CatalogError(
                f"Semantic profile component {component_key} cannot grant protected "
                f"evidence status {evidence_status!r}"
            )
        facts = component.get("facts", {})
        if not isinstance(facts, dict):
            raise CatalogError(f"Semantic component {component_key} facts must be an object")
        relative_path = normalize_component_path(component["path"])
        component_path = (
            asset_resolved
            if relative_path == "."
            else asset_resolved.joinpath(*relative_path.split("/")).resolve()
        )
        try:
            component_path.relative_to(asset_resolved)
        except ValueError as exc:
            raise CatalogError(
                f"Semantic component escapes asset {record['locator']}: {relative_path}"
            ) from exc
        required_component = bool(component.get("required", True))
        present = component_path.exists()
        if required_component and not present:
            raise CatalogError(
                f"Required semantic component is missing: {record['locator']}#{component_key} "
                f"path={relative_path}"
            )
        entry_type = (
            "file" if present and component_path.is_file()
            else "directory" if present and component_path.is_dir()
            else "missing"
        )
        prefix = "" if relative_path == "." else relative_path.rstrip("/") + "/"
        matched_entries: list[dict[str, Any]] = []
        for item in scan["entries"]:
            item_path = item["relative_path"]
            if relative_path == ".":
                component_relative = item_path
            elif item_path.casefold() == relative_path.casefold():
                component_relative = "."
            elif item_path.casefold().startswith(prefix.casefold()):
                component_relative = item_path[len(prefix):]
            else:
                continue
            matched_entries.append({**item, "component_relative_path": component_relative})
        if present and entry_type == "file" and len(matched_entries) != 1:
            raise CatalogError(
                f"Semantic file component is absent from asset inventory: "
                f"{record['locator']}#{component_key}"
            )
        fingerprint = hashlib.sha256()
        fingerprint.update(b"blindassist-component-metadata-v1\0")
        for item in matched_entries:
            fingerprint.update(item["component_relative_path"].encode("utf-8"))
            fingerprint.update(b"\0")
            fingerprint.update(str(item["bytes"]).encode("ascii"))
            fingerprint.update(b"\0")
            fingerprint.update(str(item.get("mtime_ns", 0)).encode("ascii"))
            fingerprint.update(b"\n")
        content_id = None
        if matched_entries and all(item.get("sha256") for item in matched_entries):
            if entry_type == "file":
                content_id = f"sha256:{matched_entries[0]['sha256']}"
            else:
                content_id = "tree-sha256:" + sha256_bytes(
                    canonical_json_bytes(
                        {
                            "algorithm": "blindassist-tree-sha256-v1",
                            "files": [
                                {
                                    "path": item["component_relative_path"],
                                    "bytes": item["bytes"],
                                    "sha256": item["sha256"],
                                }
                                for item in matched_entries
                            ],
                        }
                    )
                )
        component_id = f"{record['asset_id']}#component:{component_key}"
        component_rows.append(
            (
                component_id,
                record["asset_id"],
                component_key,
                component.get("logical_name", component_key),
                relative_path,
                component["component_kind"],
                component["data_role"],
                entry_type,
                "present" if present else "missing",
                sum(int(item["bytes"]) for item in matched_entries),
                len(matched_entries),
                fingerprint.hexdigest(),
                content_id,
                "content" if content_id else "metadata",
                evidence_status,
                component["claim_ceiling"],
                component["description"],
                json.dumps(facts, ensure_ascii=False, sort_keys=True),
                profile_sha256,
                scan_id,
            )
        )

    profile_facts = {
        key: value
        for key, value in profile.items()
        if key not in {"schema", "profile_id", "asset_locator", "title", "summary", "components"}
    }
    connection.execute("DELETE FROM asset_components WHERE asset_id = ?", (record["asset_id"],))
    connection.execute(
        """
        INSERT INTO asset_semantic_profiles(
            asset_id, profile_id, profile_path, profile_sha256, title,
            summary, facts_json, last_scan_id, last_seen_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
            profile_id=excluded.profile_id,
            profile_path=excluded.profile_path,
            profile_sha256=excluded.profile_sha256,
            title=excluded.title,
            summary=excluded.summary,
            facts_json=excluded.facts_json,
            last_scan_id=excluded.last_scan_id,
            last_seen_at=excluded.last_seen_at
        """,
        (
            record["asset_id"],
            profile["profile_id"],
            relative_profile,
            profile_sha256,
            profile["title"],
            profile["summary"],
            json.dumps(profile_facts, ensure_ascii=False, sort_keys=True),
            scan_id,
            now,
        ),
    )
    connection.executemany(
        """
        INSERT INTO asset_components(
            component_id, asset_id, component_key, logical_name, relative_path,
            component_kind, data_role, entry_type, state, bytes, file_count,
            metadata_sha256, content_id, identity_strength, evidence_status,
            claim_ceiling, description, facts_json, profile_sha256, last_scan_id
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        component_rows,
    )
    return len(component_rows)


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


def resolve_asset_target(
    connection: sqlite3.Connection, key: str
) -> tuple[sqlite3.Row, sqlite3.Row | None]:
    if "#" not in key:
        return resolve_asset(connection, key), None
    asset_key, component_key = key.rsplit("#", 1)
    if not asset_key or not component_key:
        raise CatalogError(f"Invalid semantic component key: {key}")
    asset = resolve_asset(connection, asset_key)
    component = connection.execute(
        """
        SELECT * FROM asset_components
        WHERE asset_id = ? AND component_key = ?
        """,
        (asset["asset_id"], component_key),
    ).fetchone()
    if component is None:
        choices = [
            row["component_key"]
            for row in connection.execute(
                """
                SELECT component_key FROM asset_components
                WHERE asset_id = ? ORDER BY component_key LIMIT 30
                """,
                (asset["asset_id"],),
            )
        ]
        raise CatalogError(
            f"Semantic component not found: {key}; available components: {choices}"
        )
    return asset, component


def component_usage_metadata(
    component: sqlite3.Row | None, metadata: Any
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise CatalogError("Usage metadata must be a JSON object")
    if component is None:
        return metadata
    return {
        **metadata,
        "semantic_component": {
            "component_id": component["component_id"],
            "component_key": component["component_key"],
            "component_kind": component["component_kind"],
            "data_role": component["data_role"],
            "relative_path": component["relative_path"],
        },
    }


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
    r"(?:test-)?artifacts\.local[\\/][^\s'\"`<>(){}\[\],;]+",
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
    root_rows = connection.execute(
        "SELECT root_name, disposition FROM roots"
    ).fetchall()
    root_names = {row["root_name"] for row in root_rows}
    excluded_roots = {
        row["root_name"]
        for row in root_rows
        if row["disposition"] == "excluded"
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
                marker = (
                    "test-artifacts.local"
                    if raw.casefold().startswith("test-artifacts.local")
                    else "artifacts.local"
                )
                marker_index = raw.casefold().index(marker)
                raw_locator = raw[marker_index + len(marker):].lstrip("\\/")
                raw_locator = raw_locator.replace("\\", "/").strip("/")
                if not raw_locator:
                    continue
                locator = re.sub(r"/+", "/", raw_locator)
                if marker == "test-artifacts.local":
                    locator = f"evidence/{locator}"
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
                elif (
                    known_root in excluded_roots
                    and (artifact_root / Path(locator)).exists()
                ):
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
                        "locator": raw_locator,
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
                        raw_locator,
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
    cache_accesses = fabric_records(fabric, "catalog/accesses")
    accesses_by_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for access in cache_accesses:
        accesses_by_cache[access.get("cache_key", "")].append(access)
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
        ] + list(cache.get("model_ids", [])) + [
            item["asset_id"]
            for item in cache.get("asset_inputs", [])
            if item.get("asset_id")
        ]
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
        for access in accesses_by_cache.get(cache_key, []):
            consumer = access.get("consumer")
            event_id = access.get("event_id")
            if not consumer or not event_id:
                continue
            record_usage(
                connection,
                asset_id=asset_id,
                consumer=consumer,
                purpose=access.get("purpose", "shared-cache-input"),
                experiment_id=access.get("experiment_id"),
                access_mode="cache_hit",
                evidence_effect="none",
                metadata={
                    "source": "resource-fabric-cache-access",
                    "payload_bytes": int(access.get("payload_bytes", 0)),
                },
                event_id=f"fabric-cache-access:{cache_key}:{event_id}",
                recorded_at=access.get("recorded_at"),
                ignore_existing=True,
            )
            uses += 1

    experiment_root = fabric / "experiments"
    if experiment_root.exists():
        for manifest_path in sorted(experiment_root.rglob("manifest.json")):
            manifest = read_json(manifest_path)
            consumer = manifest.get("id", manifest_path.parent.name)
            referenced = list(manifest.get("source_ids", [])) + [
                f"cache:{key}" for key in manifest.get("cache_keys", [])
            ] + [
                f"cache:{key}" for key in manifest.get("produced_cache_keys", [])
            ] + [
                item["asset_id"]
                for item in manifest.get("asset_inputs", [])
                if item.get("asset_id")
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
            ] + [
                item["asset_id"]
                for item in case.get("asset_inputs", [])
                if item.get("asset_id")
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
    managed_assets_seen = 0
    semantic_components_seen = 0
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
                    child_rule = next(
                        (
                            override
                            for name, override in rule.get("asset_overrides", {}).items()
                            if name.casefold() == child.name.casefold()
                        ),
                        rule,
                    )
                    record = discovered_record(
                        locator, root_name, child_rule, scan, scan_id, now
                    )
                    with connection:
                        upsert_asset(connection, record, scan["entries"])
                        semantic_components_seen += sync_semantic_profile(
                            connection,
                            record=record,
                            scan=scan,
                            rule=child_rule,
                            asset_path=child_path,
                            repo_root=args.repo_root.resolve(),
                            scan_id=scan_id,
                            now=now,
                        )
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

        # A stable asset may deliberately live below an otherwise excluded root.
        # Exact policy locators let us govern that asset without opening mutable
        # siblings such as runtime caches, environments, or process state.
        for locator, rule in sorted(
            policy.get("managed_assets", {}).items(),
            key=lambda item: item[0].casefold(),
        ):
            normalized = normalize_locator(locator)
            path = path_for_locator(normalized, artifact_root)
            asset_id = asset_id_for_locator(normalized)
            if not path.exists():
                connection.execute(
                    "UPDATE assets SET state = 'missing' WHERE asset_id = ?",
                    (asset_id,),
                )
                connection.execute(
                    "DELETE FROM asset_components WHERE asset_id = ?",
                    (asset_id,),
                )
                connection.execute(
                    "DELETE FROM asset_semantic_profiles WHERE asset_id = ?",
                    (asset_id,),
                )
                continue
            try:
                if is_reparse_point(path):
                    raise CatalogError(
                        f"Managed asset cannot be a reparse point: {normalized}"
                    )
                scan = scan_asset(path)
                record = discovered_record(
                    normalized,
                    normalized.split("/", 1)[0],
                    rule,
                    scan,
                    scan_id,
                    now,
                    discovery="policy-managed-excluded-root",
                )
                with connection:
                    upsert_asset(connection, record, scan["entries"])
                    semantic_components_seen += sync_semantic_profile(
                        connection,
                        record=record,
                        scan=scan,
                        rule=rule,
                        asset_path=path,
                        repo_root=args.repo_root.resolve(),
                        scan_id=scan_id,
                        now=now,
                    )
                assets_seen += 1
                managed_assets_seen += 1
                files_seen += scan["file_count"]
                bytes_seen += scan["bytes"]
                vanished_total += scan["vanished_entries"]
                reparse_total += scan["reparse_entries"]
                print(
                    f"[asset-catalog] managed asset={normalized} "
                    f"files={scan['file_count']} bytes={scan['bytes']}",
                    file=sys.stderr,
                    flush=True,
                )
            except (CatalogError, OSError) as exc:
                errors.append({"locator": normalized, "error": str(exc)})

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
            "managed_assets_seen": managed_assets_seen,
            "semantic_components_seen": semantic_components_seen,
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


def governed_asset_unit_for_path(
    path: Path,
    artifact_root: Path,
    policy: dict[str, Any],
) -> tuple[str, dict[str, Any], Path]:
    """Return the catalog unit that owns one concrete artifact path.

    Full discovery catalogs stable roots by direct child.  Event-driven runs
    use the same unit boundary so an output refresh never creates overlapping
    one-file assets or walks unrelated roots.  Exact managed assets are the
    only allowed units below an otherwise excluded root.
    """

    locator = locator_for_path(path, artifact_root)
    parts = locator.split("/")
    managed_matches = [
        normalize_locator(candidate)
        for candidate in policy.get("managed_assets", {})
        if locator.casefold() == normalize_locator(candidate).casefold()
        or locator.casefold().startswith(normalize_locator(candidate).casefold() + "/")
    ]
    if managed_matches:
        unit_locator = max(managed_matches, key=lambda item: len(item.split("/")))
        rule = next(
            value
            for candidate, value in policy["managed_assets"].items()
            if normalize_locator(candidate).casefold() == unit_locator.casefold()
        )
        return unit_locator, rule, path_for_locator(unit_locator, artifact_root)

    root_name = parts[0]
    if root_name in policy.get("excluded_roots", {}):
        raise CatalogError(
            f"Artifact path is below excluded root {root_name!r} and has no "
            f"managed asset rule: {locator}"
        )
    root_rule = policy.get("roots", {}).get(root_name, policy["fallback"])
    if len(parts) == 1:
        unit_locator = root_name
        unit_path = path_for_locator(unit_locator, artifact_root)
        if unit_path.is_dir():
            raise CatalogError(
                f"A stable root is not an asset unit; select one child below {root_name}"
            )
        return unit_locator, root_rule, unit_path

    child_name = parts[1]
    excluded_children = {
        value.casefold() for value in root_rule.get("exclude_children", [])
    }
    if child_name.casefold() in excluded_children:
        raise CatalogError(
            f"Artifact path is below policy-excluded child {root_name}/{child_name}: "
            f"{locator}"
        )
    rule = next(
        (
            override
            for name, override in root_rule.get("asset_overrides", {}).items()
            if name.casefold() == child_name.casefold()
        ),
        root_rule,
    )
    unit_locator = f"{root_name}/{child_name}"
    return unit_locator, rule, path_for_locator(unit_locator, artifact_root)


def reconcile_asset_path(
    connection: sqlite3.Connection,
    *,
    path: Path,
    artifact_root: Path,
    repo_root: Path,
    policy: dict[str, Any],
    scan_id: str,
    now: str,
) -> dict[str, Any]:
    requested = path.resolve()
    if not requested.exists():
        raise CatalogError(f"Cannot reconcile missing artifact path: {requested}")
    unit_locator, rule, unit_path = governed_asset_unit_for_path(
        requested, artifact_root, policy
    )
    if is_reparse_point(unit_path):
        raise CatalogError(f"Reconciled asset unit cannot be a reparse point: {unit_path}")
    scan = scan_asset(unit_path)
    record = discovered_record(
        unit_locator,
        unit_locator.split("/", 1)[0],
        rule,
        scan,
        scan_id,
        now,
        discovery="event-reconcile",
    )
    upsert_asset(connection, record, scan["entries"])
    component_count = sync_semantic_profile(
        connection,
        record=record,
        scan=scan,
        rule=rule,
        asset_path=unit_path,
        repo_root=repo_root,
        scan_id=scan_id,
        now=now,
    )
    requested_relative = (
        "." if requested == unit_path else requested.relative_to(unit_path).as_posix()
    )
    return {
        "asset_id": record["asset_id"],
        "locator": unit_locator,
        "path": str(unit_path),
        "requested_path": str(requested),
        "requested_relative_path": requested_relative,
        "bytes": record["bytes"],
        "file_count": record["file_count"],
        "identity_strength": record["identity_strength"],
        "semantic_components": component_count,
    }


def reconcile_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    repo_root = args.repo_root.resolve()
    policy = load_policy(args.policy.resolve())
    scan_id = f"reconcile:{utc_now()}:{uuid.uuid4().hex[:8]}"
    now = utc_now()
    connection = open_catalog(args.database.resolve())
    reconciled: list[dict[str, Any]] = []
    try:
        with connection:
            for path in args.path:
                reconciled.append(
                    reconcile_asset_path(
                        connection,
                        path=path,
                        artifact_root=artifact_root,
                        repo_root=repo_root,
                        policy=policy,
                        scan_id=scan_id,
                        now=now,
                    )
                )
            fabric_summary = (
                sync_resource_fabric(connection, artifact_root, scan_id, now)
                if args.sync_fabric
                else None
            )
    finally:
        connection.close()

    references = None
    if args.references:
        references = references_command(
            argparse.Namespace(
                artifact_root=artifact_root,
                database=args.database.resolve(),
                repo_root=repo_root,
            )
        )
    report = None
    if args.report:
        report = report_command(
            argparse.Namespace(
                artifact_root=artifact_root,
                database=args.database.resolve(),
                output_dir=artifact_root / DEFAULT_REPORT_RELATIVE,
            )
        )
    verification = None
    if args.verify:
        verification = verify_command(
            argparse.Namespace(
                artifact_root=artifact_root,
                database=args.database.resolve(),
                repo_root=repo_root,
                deep=False,
            )
        )
    return {
        "status": "PASS",
        "scan_id": scan_id,
        "reconciled": reconciled,
        "resource_fabric": fabric_summary,
        "references": references,
        "report": report,
        "verification": verification,
    }


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


def components_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        asset = resolve_asset(connection, args.key)
        profile = connection.execute(
            "SELECT * FROM asset_semantic_profiles WHERE asset_id = ?",
            (asset["asset_id"],),
        ).fetchone()
        if profile is None:
            raise CatalogError(f"Asset has no semantic profile: {asset['locator']}")
        components = connection.execute(
            """
            SELECT component_id, component_key, logical_name, relative_path,
                   component_kind, data_role, entry_type, state, bytes,
                   file_count, content_id, identity_strength, evidence_status,
                   claim_ceiling, description, facts_json
            FROM asset_components WHERE asset_id = ?
            ORDER BY component_kind, component_key
            """,
            (asset["asset_id"],),
        ).fetchall()
        return {
            "asset_id": asset["asset_id"],
            "locator": asset["locator"],
            "profile": {
                "profile_id": profile["profile_id"],
                "profile_path": profile["profile_path"],
                "profile_sha256": profile["profile_sha256"],
                "title": profile["title"],
                "summary": profile["summary"],
                "facts": json.loads(profile["facts_json"]),
            },
            "components": [
                {**dict(row), "facts": json.loads(row["facts_json"])}
                for row in components
            ],
        }
    finally:
        connection.close()


def resolve_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        asset, component = resolve_asset_target(connection, args.key)
        if asset["state"] != "present" and not args.allow_missing:
            raise CatalogError(f"Asset is not present: {asset['locator']} state={asset['state']}")
        path = path_for_locator(asset["locator"], args.artifact_root.resolve())
        if component is not None:
            if component["state"] != "present" and not args.allow_missing:
                raise CatalogError(
                    f"Semantic component is not present: {args.key} state={component['state']}"
                )
            if component["relative_path"] != ".":
                path = path.joinpath(*component["relative_path"].split("/"))
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
                    metadata=component_usage_metadata(
                        component, parse_json(args.metadata_json, default={})
                    ),
                )
            asset = resolve_asset(connection, asset["asset_id"])
        result = {
            "asset_id": asset["asset_id"],
            "logical_name": asset["logical_name"],
            "locator": asset["locator"],
            "path": str(path),
            "content_id": component["content_id"] if component is not None else asset["content_id"],
            "identity_strength": (
                component["identity_strength"] if component is not None
                else asset["identity_strength"]
            ),
            "evidence_status": (
                component["evidence_status"] if component is not None
                else asset["evidence_status"]
            ),
            "storage_status": asset["storage_status"],
            "claim_ceiling": (
                component["claim_ceiling"] if component is not None
                else asset["claim_ceiling"]
            ),
            "usage_event_id": event_id,
        }
        if component is not None:
            result["component"] = {
                "component_id": component["component_id"],
                "component_key": component["component_key"],
                "logical_name": component["logical_name"],
                "component_kind": component["component_kind"],
                "data_role": component["data_role"],
            }
        return result
    finally:
        connection.close()


def consume_command(args: argparse.Namespace) -> dict[str, Any]:
    connection = open_catalog(args.database.resolve())
    try:
        asset, component = resolve_asset_target(connection, args.key)
        with connection:
            event_id = record_usage(
                connection,
                asset_id=asset["asset_id"],
                consumer=args.consumer,
                purpose=args.purpose,
                experiment_id=args.experiment_id,
                access_mode=args.access_mode,
                evidence_effect=args.evidence_effect,
                metadata=component_usage_metadata(
                    component, parse_json(args.metadata_json, default={})
                ),
                event_id=args.event_id,
            )
        updated = resolve_asset(connection, asset["asset_id"])
        return {
            "event_id": event_id,
            "asset_id": asset["asset_id"],
            "component_key": component["component_key"] if component is not None else None,
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


def authority_queue_reason(asset: sqlite3.Row) -> tuple[str, str]:
    if int(asset["file_count"]) == 0:
        return (
            "empty_asset_no_authority_evidence",
            "The catalog unit is empty, so it has no inspectable authority assertion.",
        )
    if asset["asset_class"] == "evidence":
        return (
            "evidence_authority_not_cited_or_asserted",
            "No current authority outcome reference or explicit consumed/diagnostic assertion was found.",
        )
    if asset["asset_class"] in {"archive", "legacy_unclassified"}:
        return (
            "legacy_authority_requires_manual_adjudication",
            "Legacy/archive contents need an owner and an explicit authority decision.",
        )
    return (
        "no_matching_auditable_authority_rule",
        "No high-confidence static type, current authority reference, or artifact assertion matched.",
    )


def build_authority_queue(
    connection: sqlite3.Connection,
    *,
    exclude_asset_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude_asset_ids or set()
    rows = connection.execute(
        """
        SELECT asset_id, locator, root_name, asset_kind, asset_class,
               entry_type, bytes, file_count, owner, claim_ceiling
        FROM assets
        WHERE state = 'present' AND evidence_status = 'unknown'
        ORDER BY locator
        """
    ).fetchall()
    groups: dict[str, dict[str, Any]] = {}
    assets: list[dict[str, Any]] = []
    for asset in rows:
        if asset["asset_id"] in excluded:
            continue
        reason_code, reason = authority_queue_reason(asset)
        item = dict(asset)
        item["reason_code"] = reason_code
        item["reason"] = reason
        assets.append(item)
        group = groups.setdefault(
            reason_code,
            {
                "reason": reason,
                "assets": 0,
                "bytes": 0,
                "files": 0,
                "locators": [],
            },
        )
        group["assets"] += 1
        group["bytes"] += int(asset["bytes"])
        group["files"] += int(asset["file_count"])
        group["locators"].append(asset["locator"])
    return {
        "assets": len(assets),
        "bytes": sum(int(item["bytes"]) for item in assets),
        "files": sum(int(item["file_count"]) for item in assets),
        "groups": dict(sorted(groups.items())),
        "items": assets,
    }


def authority_queue_markdown(queue: dict[str, Any], generated_at: str) -> str:
    lines = [
        "# BlindAssist evidence authority adjudication queue",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This is a non-destructive decision queue, not a deletion list. Every locator",
        "below remains `unknown`; no name-only inference grants fresh, sealed, or final authority.",
        "",
        "## Summary",
        "",
        "| Reason | Assets | Files | Bytes |",
        "| --- | ---: | ---: | ---: |",
    ]
    for reason_code, group in queue["groups"].items():
        lines.append(
            f"| `{reason_code}` | `{group['assets']}` | `{group['files']}` | `{group['bytes']}` |"
        )
    for reason_code, group in queue["groups"].items():
        lines.extend(
            [
                "",
                f"## `{reason_code}`",
                "",
                group["reason"],
                "",
            ]
        )
        lines.extend(f"- `{locator}`" for locator in group["locators"])
    return "\n".join(lines).rstrip() + "\n"


def authority_marker_match(value: str, markers: Iterable[str]) -> bool:
    for component in value.replace("\\", "/").split("/"):
        for marker in markers:
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])",
                component,
                re.IGNORECASE,
            ):
                return True
    return False


def authority_token_match(text: str, tokens: Iterable[str]) -> str | None:
    for token in tokens:
        matches = re.finditer(
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        )
        for match in matches:
            if token.casefold() == "consumed":
                prefix = text[max(0, match.start() - 48):match.start()]
                if re.search(
                    r"\b(?:not|never|without)\b[^.;:\n]{0,32}$",
                    prefix,
                    re.IGNORECASE,
                ):
                    continue
            return match.group(0)
    return None


def static_authority_rule_matches(rule: dict[str, Any], asset: sqlite3.Row) -> bool:
    selectors = {
        "root_names": "root_name",
        "asset_kinds": "asset_kind",
        "asset_classes": "asset_class",
        "entry_types": "entry_type",
    }
    for selector, column in selectors.items():
        allowed = rule.get(selector)
        if allowed and asset[column] not in allowed:
            return False
    if rule.get("requires_nonempty") and int(asset["file_count"]) == 0:
        return False
    return True


def authority_classification_decisions(
    connection: sqlite3.Connection,
    *,
    config: dict[str, Any],
    policy_sha256: str,
    artifact_root: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets = {
        row["asset_id"]: row
        for row in connection.execute(
            """
            SELECT * FROM assets
            WHERE state = 'present' AND evidence_status = 'unknown'
            ORDER BY locator
            """
        )
    }
    assets_by_locator = {row["locator"]: row for row in assets.values()}
    decisions: dict[str, dict[str, Any]] = {}
    holds: list[dict[str, Any]] = []

    def propose(
        asset: sqlite3.Row,
        rule: dict[str, Any],
        *,
        source: str,
        evidence: dict[str, Any],
        priority: int,
    ) -> None:
        existing = decisions.get(asset["asset_id"])
        if existing and int(existing["priority"]) >= priority:
            return
        decisions[asset["asset_id"]] = {
            "asset_id": asset["asset_id"],
            "locator": asset["locator"],
            "prior_evidence_status": asset["evidence_status"],
            "evidence_status": rule["evidence_status"],
            "claim_ceiling": rule["claim_ceiling"],
            "rule_id": rule["rule_id"],
            "source": source,
            "reason": rule["reason"],
            "evidence": evidence,
            "policy_sha256": policy_sha256,
            "priority": priority,
        }

    for rule in config.get("exact_rules", []):
        source_path = (repo_root / rule["authority_source"]).resolve()
        try:
            source_path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise CatalogError(
                f"Exact authority source escapes repository: {rule['authority_source']}"
            ) from exc
        try:
            authority_text = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CatalogError(
                f"Cannot read exact authority source {rule['authority_source']}: {exc}"
            ) from exc
        missing = [
            anchor for anchor in rule["authority_anchors"] if anchor not in authority_text
        ]
        if missing:
            raise CatalogError(
                f"Exact authority rule {rule['rule_id']} has stale anchors: {missing}"
            )
        asset = assets_by_locator.get(normalize_locator(rule["locator"]))
        if asset is None:
            continue
        propose(
            asset,
            rule,
            source=rule["authority_source"],
            evidence={
                "authority_anchors": rule["authority_anchors"],
                "authority_source_sha256": sha256_file(source_path),
            },
            priority=400,
        )

    authority_sources = set(config.get("current_authority_sources", []))
    outcome_markers = config.get("outcome_reference_markers", [])
    reference_rule = config.get("current_outcome_reference_rule")
    source_snapshots: dict[str, tuple[str, list[str]]] = {}
    if reference_rule:
        for reference in connection.execute(
            """
            SELECT a.*, r.source_file, r.line_number, r.raw_locator
            FROM assets a JOIN asset_references r ON r.asset_id = a.asset_id
            WHERE a.state = 'present' AND a.evidence_status = 'unknown'
              AND a.asset_class = 'evidence' AND r.resolution = 'asset'
            ORDER BY a.locator, r.source_file, r.line_number
            """
        ):
            if reference["source_file"] not in authority_sources:
                continue
            if not authority_marker_match(reference["raw_locator"], outcome_markers):
                continue
            source_file = reference["source_file"]
            if source_file not in source_snapshots:
                source_path = (repo_root / source_file).resolve()
                try:
                    source_path.relative_to(repo_root.resolve())
                except ValueError as exc:
                    raise CatalogError(
                        f"Authority source escapes repository: {source_file}"
                    ) from exc
                source_text = source_path.read_text(encoding="utf-8")
                source_snapshots[source_file] = (
                    sha256_file(source_path),
                    source_text.splitlines(),
                )
            source_sha256, source_lines = source_snapshots[source_file]
            needle = re.sub(
                r"/+",
                "/",
                reference["raw_locator"].replace("\\", "/"),
            ).casefold()
            actual_line_number = next(
                (
                    index
                    for index, line in enumerate(source_lines, start=1)
                    if needle
                    in re.sub(r"/+", "/", line.replace("\\", "/")).casefold()
                ),
                None,
            )
            if actual_line_number is None:
                continue
            propose(
                reference,
                reference_rule,
                source=f"{source_file}:{actual_line_number}",
                evidence={
                    "raw_locator": reference["raw_locator"],
                    "authority_source_sha256": source_sha256,
                },
                priority=300,
            )

    assertion_rules = config.get("artifact_assertion_rules", [])
    if assertion_rules:
        text_extensions = {
            item.casefold()
            for item in config.get(
                "artifact_assertion_text_extensions",
                [".json", ".jsonl", ".md", ".txt", ".csv", ".toml", ".yaml", ".yml", ".log"],
            )
        }
        maximum_bytes = int(config.get("artifact_assertion_max_bytes", 1_048_576))
        protected_tokens = config.get("protected_authority_tokens", ["sealed_final"])
        assertions: dict[str, dict[str, Any]] = {}
        protected_assets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        file_rows = connection.execute(
            """
            SELECT a.asset_id, a.locator, a.entry_type, f.relative_path, f.bytes
            FROM assets a JOIN asset_files f ON f.asset_id = a.asset_id
            WHERE a.state = 'present' AND a.evidence_status = 'unknown'
              AND a.asset_class = 'evidence' AND f.bytes <= ?
            ORDER BY a.locator, f.relative_path
            """,
            (maximum_bytes,),
        ).fetchall()
        for item in file_rows:
            candidate_name = (
                Path(item["locator"]).name
                if item["entry_type"] == "file"
                else Path(item["relative_path"]).name
            )
            if Path(candidate_name).suffix.casefold() not in text_extensions:
                continue
            if not authority_marker_match(candidate_name, outcome_markers):
                continue
            asset_path = path_for_locator(item["locator"], artifact_root)
            evidence_path = (
                asset_path
                if item["entry_type"] == "file"
                else asset_path.joinpath(*item["relative_path"].split("/"))
            )
            try:
                text = evidence_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            protected = authority_token_match(text, protected_tokens)
            if protected:
                protected_assets[item["asset_id"]].append(
                    {
                        "relative_path": item["relative_path"],
                        "token": protected,
                    }
                )
            for index, rule in enumerate(assertion_rules):
                matched = authority_token_match(text, rule["tokens"])
                if not matched:
                    continue
                priority = 250 if rule["evidence_status"] == "development_consumed" else 240
                current = assertions.get(item["asset_id"])
                if current and int(current["priority"]) >= priority:
                    continue
                assertions[item["asset_id"]] = {
                    "rule": rule,
                    "priority": priority,
                    "source": f"asset:{item['locator']}/{item['relative_path']}",
                    "evidence": {
                        "relative_path": item["relative_path"],
                        "matched_token": matched,
                        "assertion_file_sha256": sha256_file(evidence_path),
                    },
                }
        for asset_id, assertion in assertions.items():
            if asset_id in protected_assets:
                holds.append(
                    {
                        "asset_id": asset_id,
                        "locator": assets[asset_id]["locator"],
                        "reason_code": "conflicting_protected_authority_assertion",
                        "reason": "A protected authority token conflicts with an automatic assertion.",
                        "evidence": protected_assets[asset_id],
                    }
                )
                continue
            propose(
                assets[asset_id],
                assertion["rule"],
                source=assertion["source"],
                evidence=assertion["evidence"],
                priority=int(assertion["priority"]),
            )

    for asset in assets.values():
        for rule in config.get("static_type_rules", []):
            if static_authority_rule_matches(rule, asset):
                propose(
                    asset,
                    rule,
                    source=(
                        "data/asset-management-policy.json#"
                        f"authority_classification/static_type_rules/{rule['rule_id']}"
                    ),
                    evidence={
                        "asset_kind": asset["asset_kind"],
                        "asset_class": asset["asset_class"],
                        "entry_type": asset["entry_type"],
                    },
                    priority=100,
                )
                break

    ordered = sorted(decisions.values(), key=lambda item: item["locator"])
    for item in ordered:
        item.pop("priority", None)
    return ordered, sorted(holds, key=lambda item: item["locator"])


def classify_authority_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy.resolve())
    config = policy.get("authority_classification")
    if not config:
        raise CatalogError("Policy does not define authority_classification")
    policy_sha256 = sha256_bytes(canonical_json_bytes(policy))
    connection = open_catalog(args.database.resolve())
    try:
        before = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE state = 'present' AND evidence_status = 'unknown'"
        ).fetchone()[0]
        decisions, holds = authority_classification_decisions(
            connection,
            config=config,
            policy_sha256=policy_sha256,
            artifact_root=args.artifact_root.resolve(),
            repo_root=args.repo_root.resolve(),
        )
        by_status: dict[str, int] = defaultdict(int)
        by_rule: dict[str, int] = defaultdict(int)
        for decision in decisions:
            by_status[decision["evidence_status"]] += 1
            by_rule[decision["rule_id"]] += 1
        applied = 0
        if args.apply:
            applied_at = utc_now()
            with connection:
                for decision in decisions:
                    material = {
                        "asset_id": decision["asset_id"],
                        "rule_id": decision["rule_id"],
                        "assigned_evidence_status": decision["evidence_status"],
                        "source": decision["source"],
                        "evidence": decision["evidence"],
                        "policy_sha256": policy_sha256,
                    }
                    classification_id = (
                        "authority-classification:"
                        + sha256_bytes(canonical_json_bytes(material))
                    )
                    transition_asset(
                        connection,
                        asset_id=decision["asset_id"],
                        evidence_status=decision["evidence_status"],
                        storage_status=None,
                        reason=(
                            f"Authority classification {decision['rule_id']}: "
                            f"{decision['reason']} Source: {decision['source']}"
                        ),
                        claim_ceiling=decision["claim_ceiling"],
                        event_id=f"{classification_id}:lifecycle",
                        recorded_at=applied_at,
                    )
                    connection.execute(
                        """
                        INSERT INTO authority_classifications(
                            classification_id, asset_id, prior_evidence_status,
                            assigned_evidence_status, rule_id, source, reason,
                            evidence_json, policy_sha256, applied_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            classification_id,
                            decision["asset_id"],
                            decision["prior_evidence_status"],
                            decision["evidence_status"],
                            decision["rule_id"],
                            decision["source"],
                            decision["reason"],
                            json.dumps(
                                decision["evidence"],
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            policy_sha256,
                            applied_at,
                        ),
                    )
                    applied += 1
        after = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE state = 'present' AND evidence_status = 'unknown'"
        ).fetchone()[0]
        queue = build_authority_queue(
            connection,
            exclude_asset_ids=(
                {item["asset_id"] for item in decisions} if not args.apply else None
            ),
        )
        return {
            "status": "APPLIED" if args.apply else "DRY_RUN",
            "policy_sha256": policy_sha256,
            "unknown_before": int(before),
            "unknown_after": int(after),
            "projected_unknown_after": int(before) - len(decisions),
            "classified": len(decisions),
            "applied": applied,
            "classified_by_status": dict(sorted(by_status.items())),
            "classified_by_rule": dict(sorted(by_rule.items())),
            "protected_holds": holds,
            "remaining_queue": {
                "assets": queue["assets"],
                "groups": {
                    key: {
                        "assets": value["assets"],
                        "bytes": value["bytes"],
                        "files": value["files"],
                        "reason": value["reason"],
                    }
                    for key, value in queue["groups"].items()
                },
            },
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
        semantic_profile_count = connection.execute(
            "SELECT COUNT(*) FROM asset_semantic_profiles"
        ).fetchone()[0]
        semantic_component_count = connection.execute(
            "SELECT COUNT(*) FROM asset_components"
        ).fetchone()[0]
        authority_classification_count = connection.execute(
            "SELECT COUNT(*) FROM authority_classifications"
        ).fetchone()[0]
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
        authority_classifications = [dict(row) for row in connection.execute(
            """
            SELECT classification_id, asset_id, prior_evidence_status,
                   assigned_evidence_status, rule_id, source, reason,
                   evidence_json, policy_sha256, applied_at
            FROM authority_classifications
            ORDER BY applied_at DESC, classification_id LIMIT 1000
            """
        )]
        authority_queue = build_authority_queue(connection)
        semantic_profiles: list[dict[str, Any]] = []
        for profile_row in connection.execute(
            """
            SELECT p.*, a.locator FROM asset_semantic_profiles p
            JOIN assets a ON a.asset_id = p.asset_id
            ORDER BY a.locator
            """
        ):
            component_rows = connection.execute(
                """
                SELECT component_id, component_key, logical_name, relative_path,
                       component_kind, data_role, entry_type, state, bytes,
                       file_count, content_id, identity_strength, evidence_status,
                       claim_ceiling, description, facts_json
                FROM asset_components WHERE asset_id = ?
                ORDER BY component_kind, component_key
                """,
                (profile_row["asset_id"],),
            ).fetchall()
            semantic_profiles.append(
                {
                    "asset_id": profile_row["asset_id"],
                    "locator": profile_row["locator"],
                    "profile_id": profile_row["profile_id"],
                    "profile_path": profile_row["profile_path"],
                    "profile_sha256": profile_row["profile_sha256"],
                    "title": profile_row["title"],
                    "summary": profile_row["summary"],
                    "facts": json.loads(profile_row["facts_json"]),
                    "components": [
                        {**dict(row), "facts": json.loads(row["facts_json"])}
                        for row in component_rows
                    ],
                }
            )
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
            "semantic_profiles": int(semantic_profile_count),
            "semantic_components": int(semantic_component_count),
            "authority_classification_events": int(authority_classification_count),
            "remaining_authority_queue_assets": int(authority_queue["assets"]),
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
            "authority_classifications": authority_classifications,
            "evidence_authority_queue": authority_queue,
            "semantic_profiles": semantic_profiles,
        }
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "master-asset-report.json"
        markdown_path = output_dir / "master-asset-report.md"
        authority_queue_json_path = output_dir / "evidence-authority-queue.json"
        authority_queue_markdown_path = output_dir / "evidence-authority-queue.md"
        atomic_write_json(json_path, report)
        atomic_write_json(
            authority_queue_json_path,
            {
                "schema": "blindassist-evidence-authority-queue-v1",
                "generated_at": report["generated_at"],
                **authority_queue,
            },
        )
        atomic_write_text(
            authority_queue_markdown_path,
            authority_queue_markdown(authority_queue, report["generated_at"]),
        )
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
        lines.extend([
            "",
            "## Semantic asset components",
            "",
            "| Asset | Component | Kind | Role | Bytes | Files | Identity | Evidence |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ])
        for profile in semantic_profiles:
            for component in profile["components"]:
                lines.append(
                    f"| `{profile['locator']}` | `{component['component_key']}` | "
                    f"`{component['component_kind']}` | `{component['data_role']}` | "
                    f"`{component['bytes']}` | `{component['file_count']}` | "
                    f"`{component['identity_strength']}` | `{component['evidence_status']}` |"
                )
        lines.extend([
            "",
            "## Evidence authority adjudication queue",
            "",
            "Unknown assets remain fail-closed and are grouped in "
            "`evidence-authority-queue.json` and `evidence-authority-queue.md`.",
            "",
            "| Reason | Assets | Files | Bytes |",
            "| --- | ---: | ---: | ---: |",
        ])
        for reason_code, group in authority_queue["groups"].items():
            lines.append(
                f"| `{reason_code}` | `{group['assets']}` | `{group['files']}` | `{group['bytes']}` |"
            )
        atomic_write_text(markdown_path, "\n".join(lines).rstrip() + "\n")
        return {
            "report_json": str(json_path),
            "report_markdown": str(markdown_path),
            "authority_queue_json": str(authority_queue_json_path),
            "authority_queue_markdown": str(authority_queue_markdown_path),
            **summary,
        }
    finally:
        connection.close()


def verify_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    repo_root = args.repo_root.resolve()
    connection = open_catalog(args.database.resolve())
    errors: list[str] = []
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            errors.append(f"sqlite integrity: {integrity}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            errors.append(f"foreign key failures: {len(foreign_keys)}")
        profile_checked = 0
        component_checked = 0
        for profile in connection.execute(
            """
            SELECT p.*, a.locator FROM asset_semantic_profiles p
            JOIN assets a ON a.asset_id = p.asset_id
            ORDER BY a.locator
            """
        ):
            try:
                profile_path = (repo_root / profile["profile_path"]).resolve()
                profile_path.relative_to(repo_root)
                if not profile_path.is_file():
                    errors.append(f"missing semantic profile: {profile['profile_path']}")
                    continue
                if sha256_file(profile_path) != profile["profile_sha256"]:
                    errors.append(f"semantic profile drift: {profile['profile_path']}")
                asset_path = path_for_locator(profile["locator"], artifact_root)
                for component in connection.execute(
                    "SELECT * FROM asset_components WHERE asset_id = ? ORDER BY component_key",
                    (profile["asset_id"],),
                ):
                    if component["evidence_status"] not in AUTOMATIC_AUTHORITY_STATUSES:
                        errors.append(
                            f"invalid semantic component evidence status "
                            f"{profile['locator']}#{component['component_key']}: "
                            f"{component['evidence_status']}"
                        )
                    component_path = (
                        asset_path
                        if component["relative_path"] == "."
                        else asset_path.joinpath(*component["relative_path"].split("/"))
                    )
                    if component["state"] == "present" and not component_path.exists():
                        errors.append(
                            f"missing semantic component: "
                            f"{profile['locator']}#{component['component_key']}"
                        )
                    if (
                        component["state"] == "present"
                        and component["entry_type"] == "file"
                        and component_path.is_file()
                        and int(component_path.stat().st_size) != int(component["bytes"])
                    ):
                        errors.append(
                            f"semantic component size drift: "
                            f"{profile['locator']}#{component['component_key']}"
                        )
                    component_checked += 1
                profile_checked += 1
            except (CatalogError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"semantic profile {profile['profile_id']}: {exc}")
        assets = connection.execute("SELECT * FROM assets ORDER BY locator").fetchall()
        deep_checked = 0
        for asset in assets:
            try:
                if asset["evidence_status"] not in VALID_EVIDENCE_STATUSES:
                    errors.append(
                        f"invalid evidence status {asset['locator']}: {asset['evidence_status']}"
                    )
                if asset["storage_status"] not in VALID_STORAGE_STATUSES:
                    errors.append(
                        f"invalid storage status {asset['locator']}: {asset['storage_status']}"
                    )
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
            "semantic_profiles_checked": profile_checked,
            "semantic_components_checked": component_checked,
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

    reconcile = subparsers.add_parser(
        "reconcile",
        help="Incrementally refresh run-touched asset units and resource lineage",
    )
    add_catalog_arguments(reconcile)
    reconcile.add_argument("--path", type=Path, action="append", default=[])
    reconcile.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    reconcile.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    reconcile.add_argument(
        "--sync-fabric",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    reconcile.add_argument(
        "--references",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    reconcile.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    reconcile.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    reconcile.set_defaults(func=reconcile_command)

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

    components = subparsers.add_parser(
        "components", help="Inspect the semantic components of one physical asset"
    )
    add_catalog_arguments(components)
    components.add_argument("key")
    components.set_defaults(func=components_command)

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

    classify_authority = subparsers.add_parser(
        "classify-authority",
        help="Audit unknown evidence authority and optionally append classification events",
    )
    add_catalog_arguments(classify_authority)
    classify_authority.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    classify_authority.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    classify_authority.add_argument(
        "--apply",
        action="store_true",
        help="Apply high-confidence decisions; default is a read-only dry run",
    )
    classify_authority.set_defaults(func=classify_authority_command)

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
    verify.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
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
