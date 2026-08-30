#!/usr/bin/env python3
"""Content-addressed resource reuse for BlindAssist research.

The fabric keeps raw data and models in one logical catalog, derives shared
normalized/feature caches from immutable resource ids, records reusable hard
cases without copying their media, and constrains experiment directories to a
small manifest/parameters/result/evidence-boundary surface.

All generated state stays below ``artifacts.local``.  Existing payloads are
never removed by this tool; adoption is copy-by-default and can use an explicit
hardlink mode for immutable files on the same volume.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = DEFAULT_REPO_ROOT / "artifacts.local"
DEFAULT_CONSUMED_ALLOWED = [
    "diagnostics",
    "training",
    "feature_cache",
    "hard_case_mining",
    "regression",
    "development_replay",
]
DEFAULT_CONSUMED_FORBIDDEN = [
    "fresh_confirmation",
    "generalization_claim",
    "safety_claim",
]
ALLOWED_EXPERIMENT_FILES = {
    "manifest.json",
    "parameters.json",
    "result.json",
    "evidence-boundary.md",
}


class FabricError(RuntimeError):
    """A user-correctable resource-fabric failure."""


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def slug(value: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not candidate:
        raise FabricError(f"Value cannot form a stable id: {value!r}")
    return candidate


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
        raise FabricError(f"Cannot read JSON {path}: {exc}") from exc


def parse_json_value(value: str | None, *, default: Any) -> Any:
    if value is None:
        return default
    if value.startswith("@"):
        return read_json(Path(value[1:]).resolve())
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise FabricError(f"Invalid JSON argument: {exc}") from exc


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


def write_once_json(path: Path, value: Any) -> bool:
    if path.exists():
        if read_json(path) != value:
            raise FabricError(f"Immutable record already exists with different content: {path}")
        return False
    atomic_write_json(path, value)
    return True


def write_once_text(path: Path, value: str) -> bool:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise FabricError(f"Immutable record already exists with different content: {path}")
        return False
    atomic_write_text(path, value)
    return True


def write_timestamped_record(
    path: Path,
    value_without_timestamp: dict[str, Any],
    *,
    timestamp_field: str,
) -> tuple[dict[str, Any], bool]:
    if path.exists():
        existing = read_json(path)
        comparable = dict(existing)
        comparable.pop(timestamp_field, None)
        if comparable != value_without_timestamp:
            raise FabricError(f"Immutable record already exists with different content: {path}")
        return existing, True
    record = {**value_without_timestamp, timestamp_field: utc_now()}
    write_once_json(path, record)
    return record, False


def is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise FabricError(f"Cannot stat {path}: {exc}") from exc
    attributes = getattr(info, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & marker)


def ensure_plain_path(path: Path, *, label: str) -> None:
    if is_reparse_point(path):
        raise FabricError(f"{label} cannot be a symlink/junction/reparse point: {path}")


def artifact_relative(path: Path, artifact_root: Path) -> str:
    resolved = path.resolve()
    root = artifact_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise FabricError(f"Path escapes artifacts.local: {resolved}") from exc


def resource_token(resource_id: str) -> str:
    if resource_id.startswith("sha256:"):
        kind = "file-sha256"
        digest = resource_id.split(":", 1)[1]
    elif resource_id.startswith("tree-sha256:"):
        kind = "tree-sha256"
        digest = resource_id.split(":", 1)[1]
    else:
        raise FabricError(f"Invalid resource id: {resource_id}")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise FabricError(f"Invalid resource digest: {resource_id}")
    return f"{kind}--{digest}"


def scan_payload(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FabricError(f"Payload does not exist: {path}")
    ensure_plain_path(path, label="Payload")
    if path.is_file():
        size = path.stat().st_size
        digest = sha256_file(path)
        return {
            "payload_type": "file",
            "resource_id": f"sha256:{digest}",
            "sha256": digest,
            "bytes": size,
            "file_count": 1,
            "entries": [{"path": path.name, "bytes": size, "sha256": digest}],
        }
    if not path.is_dir():
        raise FabricError(f"Payload must be a file or directory: {path}")

    entries: list[dict[str, Any]] = []
    stack = [path]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name.lower())
        except OSError as exc:
            raise FabricError(f"Cannot enumerate {directory}: {exc}") from exc
        for child in children:
            child_path = Path(child.path)
            ensure_plain_path(child_path, label="Payload entry")
            if child.is_dir(follow_symlinks=False):
                stack.append(child_path)
                continue
            if not child.is_file(follow_symlinks=False):
                raise FabricError(f"Unsupported payload entry: {child_path}")
            relative = child_path.relative_to(path).as_posix()
            size = child.stat(follow_symlinks=False).st_size
            entries.append(
                {
                    "path": relative,
                    "bytes": size,
                    "sha256": sha256_file(child_path),
                }
            )
    entries.sort(key=lambda item: item["path"])
    tree_material = {
        "algorithm": "blindassist-tree-sha256-v1",
        "files": entries,
    }
    digest = sha256_bytes(canonical_json_bytes(tree_material))
    return {
        "payload_type": "tree",
        "resource_id": f"tree-sha256:{digest}",
        "sha256": digest,
        "bytes": sum(int(item["bytes"]) for item in entries),
        "file_count": len(entries),
        "entries": entries,
    }


def store_root(artifact_root: Path, storage_class: str) -> Path:
    mapping = {
        "download": artifact_root / "downloads" / "resource-store",
        "model": artifact_root / "models" / "resource-store",
        "sealed": artifact_root / "evidence" / "resource-store",
    }
    try:
        return mapping[storage_class]
    except KeyError as exc:
        raise FabricError(f"Unknown storage class: {storage_class}") from exc


def fabric_root(artifact_root: Path) -> Path:
    return artifact_root / "evidence" / "resource-fabric"


def object_manifest_path(artifact_root: Path, resource_id: str) -> Path:
    return fabric_root(artifact_root) / "catalog" / "objects" / f"{resource_token(resource_id)}.json"


def object_payload_path(object_manifest: dict[str, Any], artifact_root: Path) -> Path:
    return (artifact_root / object_manifest["canonical_path"]).resolve()


def load_object(artifact_root: Path, resource_id: str) -> dict[str, Any]:
    path = object_manifest_path(artifact_root, resource_id)
    if not path.is_file():
        raise FabricError(f"Unknown resource id: {resource_id}")
    record = read_json(path)
    if record.get("resource_id") != resource_id:
        raise FabricError(f"Resource catalog identity mismatch: {path}")
    return record


def materialize_payload(
    source: Path,
    destination: Path,
    scan: dict[str, Any],
    *,
    mode: str,
) -> None:
    if mode not in {"copy", "hardlink"}:
        raise FabricError(f"Unsupported materialization mode: {mode}")
    if scan["payload_type"] == "file":
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError as exc:
                raise FabricError(f"Hardlink failed for immutable payload {source}: {exc}") from exc
        else:
            shutil.copy2(source, destination)
        return

    destination.mkdir(parents=True, exist_ok=True)
    for entry in scan["entries"]:
        source_file = source / entry["path"]
        target_file = destination / entry["path"]
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(source_file, target_file)
            except OSError as exc:
                raise FabricError(f"Hardlink failed for immutable payload {source_file}: {exc}") from exc
        else:
            shutil.copy2(source_file, target_file)


def ensure_object(
    source: Path,
    artifact_root: Path,
    *,
    storage_class: str,
    mode: str,
) -> tuple[dict[str, Any], bool]:
    scan = scan_payload(source)
    existing_manifest = object_manifest_path(artifact_root, scan["resource_id"])
    if existing_manifest.exists():
        existing = load_object(artifact_root, scan["resource_id"])
        payload = object_payload_path(existing, artifact_root)
        if not payload.exists():
            raise FabricError(f"Cataloged object payload is missing: {payload}")
        return existing, True

    digest = scan["sha256"]
    kind_dir = "sha256" if scan["payload_type"] == "file" else "tree-sha256"
    root = store_root(artifact_root, storage_class)
    object_dir = root / "objects" / kind_dir / digest[:2] / digest
    payload = object_dir / "payload"
    stage = object_dir.parent / f".{digest[:12]}.tmp-{uuid.uuid4().hex[:8]}"
    stage_payload = stage / "payload"
    if object_dir.exists():
        raise FabricError(f"Orphan object exists without catalog entry: {object_dir}")
    try:
        materialize_payload(source, stage_payload, scan, mode=mode)
        atomic_write_json(
            stage / "inventory.json",
            {
                "schema": "blindassist-resource-object-inventory-v1",
                "resource_id": scan["resource_id"],
                "files": scan["entries"],
            },
        )
        object_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, object_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    record = {
        "schema": "blindassist-resource-object-v1",
        "schema_version": SCHEMA_VERSION,
        "resource_id": scan["resource_id"],
        "payload_type": scan["payload_type"],
        "sha256": digest,
        "bytes": scan["bytes"],
        "file_count": scan["file_count"],
        "storage_class": storage_class,
        "canonical_path": artifact_relative(payload, artifact_root),
        "inventory_path": artifact_relative(object_dir / "inventory.json", artifact_root),
        "created_at": utc_now(),
    }
    write_once_json(existing_manifest, record)
    return record, False


def lifecycle_directory(artifact_root: Path, resource_id: str) -> Path:
    return fabric_root(artifact_root) / "catalog" / "lifecycle" / resource_token(resource_id)


def record_transition(
    artifact_root: Path,
    resource_id: str,
    *,
    event_id: str,
    evidence_status: str,
    storage_status: str,
    reason: str,
    allowed_uses: list[str],
    forbidden_uses: list[str],
    experiment_id: str | None = None,
) -> dict[str, Any]:
    load_object(artifact_root, resource_id)
    base_record = {
        "schema": "blindassist-resource-lifecycle-event-v1",
        "schema_version": SCHEMA_VERSION,
        "event_id": slug(event_id),
        "resource_id": resource_id,
        "evidence_status": evidence_status,
        "storage_status": storage_status,
        "reason": reason,
        "allowed_uses": sorted(set(allowed_uses)),
        "forbidden_uses": sorted(set(forbidden_uses)),
        "experiment_id": experiment_id,
    }
    path = lifecycle_directory(artifact_root, resource_id) / f"{base_record['event_id']}.json"
    if not path.exists():
        history_root = lifecycle_directory(artifact_root, resource_id)
        history = [
            read_json(event_path)
            for event_path in sorted(history_root.glob("*.json"))
        ] if history_root.exists() else []
        historical_statuses = {item.get("evidence_status") for item in history}
        if (
            "development_consumed" in historical_statuses
            and evidence_status != "development_consumed"
        ):
            raise FabricError(
                "A development_consumed resource cannot regain another evidence status; "
                "change storage_status while retaining development_consumed"
            )
        if "sealed_final" in historical_statuses and evidence_status != "sealed_final":
            raise FabricError(
                "A sealed_final resource must retain sealed_final evidence status"
            )
    record, _ = write_timestamped_record(
        path,
        base_record,
        timestamp_field="recorded_at",
    )
    return record


def ingest_resource(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    source = args.path.resolve()
    for parent_resource_id in args.parent_resource_id:
        load_object(artifact_root, parent_resource_id)
    record, reused = ensure_object(
        source,
        artifact_root,
        storage_class=args.storage_class,
        mode=args.mode,
    )
    registration_id = f"{slug(args.name)}-{record['sha256'][:12]}"
    registration_base = {
        "schema": "blindassist-resource-registration-v1",
        "schema_version": SCHEMA_VERSION,
        "registration_id": registration_id,
        "name": args.name,
        "resource_id": record["resource_id"],
        "kind": args.kind,
        "route": args.route,
        "consumer": args.consumer,
        "evidence_role": args.evidence_role,
        "dataset_id": args.dataset_id,
        "source_unit_ids": sorted(set(args.source_unit_id)),
        "disjoint_key": args.disjoint_key,
        "parent_resource_ids": sorted(set(args.parent_resource_id)),
        "owner": args.owner,
        "retention_reason": args.retention_reason,
        "rebuild_command": args.rebuild_command,
        "rebuild_cost": args.rebuild_cost,
        "source_uri": args.source_uri,
        "license_id": args.license_id,
        "original_path": str(source),
        "canonical_path": record["canonical_path"],
        "ingest_mode": args.mode,
    }
    registration_path = (
        fabric_root(artifact_root)
        / "catalog"
        / "registrations"
        / f"{registration_id}.json"
    )
    _, registration_reused = write_timestamped_record(
        registration_path,
        registration_base,
        timestamp_field="registered_at",
    )
    allowed = args.allowed_use
    forbidden = args.forbidden_use
    if args.evidence_status == "development_consumed":
        allowed = allowed or DEFAULT_CONSUMED_ALLOWED
        forbidden = forbidden or DEFAULT_CONSUMED_FORBIDDEN
    transition = record_transition(
        artifact_root,
        record["resource_id"],
        event_id=f"{registration_id}-admitted",
        evidence_status=args.evidence_status,
        storage_status=args.storage_status,
        reason=args.reason,
        allowed_uses=allowed,
        forbidden_uses=forbidden,
    )
    return {
        "resource_id": record["resource_id"],
        "canonical_path": record["canonical_path"],
        "bytes": record["bytes"],
        "file_count": record["file_count"],
        "object_reused": reused,
        "registration_reused": registration_reused,
        "registration": artifact_relative(registration_path, artifact_root),
        "evidence_status": transition["evidence_status"],
        "storage_status": transition["storage_status"],
    }


def transition_command(args: argparse.Namespace) -> dict[str, Any]:
    allowed = args.allowed_use
    forbidden = args.forbidden_use
    if args.evidence_status == "development_consumed":
        allowed = allowed or DEFAULT_CONSUMED_ALLOWED
        forbidden = forbidden or DEFAULT_CONSUMED_FORBIDDEN
    return record_transition(
        args.artifact_root.resolve(),
        args.resource_id,
        event_id=args.event_id,
        evidence_status=args.evidence_status,
        storage_status=args.storage_status,
        reason=args.reason,
        allowed_uses=allowed,
        forbidden_uses=forbidden,
        experiment_id=args.experiment_id,
    )


def cache_catalog_path(artifact_root: Path, cache_key: str) -> Path:
    return fabric_root(artifact_root) / "catalog" / "caches" / f"{cache_key}.json"


def load_cache(artifact_root: Path, cache_key: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
        raise FabricError(f"Invalid cache key: {cache_key}")
    path = cache_catalog_path(artifact_root, cache_key)
    if not path.is_file():
        raise FabricError(f"Unknown cache key: {cache_key}")
    return read_json(path)


def create_cache(
    artifact_root: Path,
    *,
    layer: str,
    source_ids: list[str],
    parent_cache_keys: list[str],
    model_ids: list[str],
    transform: str,
    transform_version: str,
    parameters: Any,
    producer: str,
    payload_source: Path,
    mode: str,
    code_sha256: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    for resource_id in source_ids:
        load_object(artifact_root, resource_id)
    for resource_id in model_ids:
        load_object(artifact_root, resource_id)
    for cache_key in parent_cache_keys:
        load_cache(artifact_root, cache_key)
    for label, digest in (("code", code_sha256), ("config", config_sha256)):
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise FabricError(f"Invalid {label} SHA-256: {digest}")
    spec = {
        "schema": "blindassist-resource-cache-key-v1",
        "layer": layer,
        "source_ids": sorted(set(source_ids)),
        "parent_cache_keys": sorted(set(parent_cache_keys)),
        "model_ids": sorted(set(model_ids)),
        "transform": transform,
        "transform_version": transform_version,
        "parameters": parameters,
        "producer": producer,
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
    }
    cache_key = sha256_bytes(canonical_json_bytes(spec))
    existing_path = cache_catalog_path(artifact_root, cache_key)
    if existing_path.exists():
        existing = read_json(existing_path)
        payload = (artifact_root / existing["payload_path"]).resolve()
        if not payload.exists():
            raise FabricError(f"Cataloged cache payload is missing: {payload}")
        return {**existing, "reused": True}

    payload_scan = scan_payload(payload_source)
    cache_dir = artifact_root / "work" / "resource-cache" / layer / cache_key[:2] / cache_key
    if cache_dir.exists():
        raise FabricError(f"Orphan cache exists without catalog entry: {cache_dir}")
    stage = cache_dir.parent / f".{cache_key[:12]}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        materialize_payload(payload_source.resolve(), stage / "payload", payload_scan, mode=mode)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, cache_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    manifest = {
        "schema": "blindassist-resource-cache-v1",
        "schema_version": SCHEMA_VERSION,
        "cache_key": cache_key,
        **spec,
        "payload_type": payload_scan["payload_type"],
        "payload_sha256": payload_scan["sha256"],
        "payload_bytes": payload_scan["bytes"],
        "payload_file_count": payload_scan["file_count"],
        "payload_path": artifact_relative(cache_dir / "payload", artifact_root),
        "created_at": utc_now(),
    }
    atomic_write_json(cache_dir / "manifest.json", manifest)
    write_once_json(existing_path, manifest)
    return {**manifest, "reused": False}


def cache_put_command(args: argparse.Namespace) -> dict[str, Any]:
    return create_cache(
        args.artifact_root.resolve(),
        layer=args.layer,
        source_ids=args.source_id,
        parent_cache_keys=args.parent_cache_key,
        model_ids=args.model_id,
        transform=args.transform,
        transform_version=args.transform_version,
        parameters=parse_json_value(args.parameters_json, default={}),
        producer=args.producer,
        payload_source=args.payload.resolve(),
        mode=args.mode,
        code_sha256=args.code_sha256,
        config_sha256=args.config_sha256,
    )


def cache_json_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    resource = load_object(artifact_root, args.source_id)
    payload = object_payload_path(resource, artifact_root)
    if resource["payload_type"] != "file":
        raise FabricError("cache-json requires a single-file JSON resource")
    value = read_json(payload)
    temporary_dir = artifact_root / "tmp" / "resource-fabric" / uuid.uuid4().hex
    temporary = temporary_dir / "normalized.json"
    try:
        atomic_write_json(temporary, value)
        return create_cache(
            artifact_root,
            layer="normalized",
            source_ids=[args.source_id],
            parent_cache_keys=[],
            model_ids=[],
            transform="json-canonical",
            transform_version="v1",
            parameters={},
            producer="tools/data/resource_fabric.py cache-json",
            payload_source=temporary,
            mode="copy",
        )
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def hard_case_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    for resource_id in args.source_id:
        load_object(artifact_root, resource_id)
    for cache_key in args.cache_key:
        load_cache(artifact_root, cache_key)
    forbidden = args.forbidden_use or DEFAULT_CONSUMED_FORBIDDEN
    base_record = {
        "schema": "blindassist-hard-case-v1",
        "schema_version": SCHEMA_VERSION,
        "id": slug(args.id),
        "route": slug(args.route),
        "case_kind": args.case_kind,
        "failure_layer": args.failure_layer,
        "evidence_split": args.evidence_split,
        "source_ids": sorted(set(args.source_id)),
        "cache_keys": sorted(set(args.cache_key)),
        "selector": parse_json_value(args.selector_json, default={}),
        "truth_authority": args.truth_authority,
        "selected_by": args.selected_by,
        "observed_outcome": args.observed_outcome,
        "claim_ceiling": args.claim_ceiling,
        "reusable_for": sorted(set(args.allowed_use or DEFAULT_CONSUMED_ALLOWED)),
        "forbidden_for": sorted(set(forbidden)),
    }
    path = (
        fabric_root(artifact_root)
        / "hard-cases"
        / base_record["route"]
        / f"{base_record['id']}.json"
    )
    record, reused = write_timestamped_record(
        path,
        base_record,
        timestamp_field="created_at",
    )
    return {"hard_case": artifact_relative(path, artifact_root), "reused": reused, **record}


def experiment_directory(artifact_root: Path, route: str, experiment_id: str) -> Path:
    return fabric_root(artifact_root) / "experiments" / slug(route) / slug(experiment_id)


def experiment_create_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    for resource_id in args.source_id:
        load_object(artifact_root, resource_id)
    for cache_key in args.cache_key:
        load_cache(artifact_root, cache_key)
    hard_case_paths = []
    for hard_case_id in args.hard_case:
        path = fabric_root(artifact_root) / "hard-cases" / slug(args.route) / f"{slug(hard_case_id)}.json"
        if not path.is_file():
            raise FabricError(f"Unknown hard case for route {args.route}: {hard_case_id}")
        hard_case_paths.append(artifact_relative(path, artifact_root))

    directory = experiment_directory(artifact_root, args.route, args.id)
    parameters = parse_json_value(args.parameters_json, default={})
    if not isinstance(parameters, dict):
        raise FabricError("Experiment parameters must be a JSON object")
    boundary = args.boundary
    if args.boundary_file:
        boundary = args.boundary_file.read_text(encoding="utf-8")
    if not boundary:
        raise FabricError("Experiment requires an explicit evidence boundary")
    boundary_text = boundary.rstrip() + "\n"
    manifest_base = {
        "schema": "blindassist-thin-experiment-v1",
        "schema_version": SCHEMA_VERSION,
        "id": slug(args.id),
        "route": slug(args.route),
        "question": args.question,
        "evaluator": args.evaluator,
        "status": args.status,
        "source_ids": sorted(set(args.source_id)),
        "cache_keys": sorted(set(args.cache_key)),
        "hard_cases": sorted(hard_case_paths),
        "parameters_path": "parameters.json",
        "evidence_boundary_path": "evidence-boundary.md",
        "result": None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    parameters_reused = not write_once_json(directory / "parameters.json", parameters)
    boundary_reused = not write_once_text(directory / "evidence-boundary.md", boundary_text)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        immutable_keys = {
            "schema",
            "schema_version",
            "id",
            "route",
            "question",
            "evaluator",
            "source_ids",
            "cache_keys",
            "hard_cases",
            "parameters_path",
            "evidence_boundary_path",
        }
        if {key: manifest.get(key) for key in immutable_keys} != {
            key: manifest_base.get(key) for key in immutable_keys
        }:
            raise FabricError(f"Experiment already exists with different immutable inputs: {manifest_path}")
        manifest_reused = True
    else:
        timestamp = utc_now()
        manifest = {**manifest_base, "created_at": timestamp, "updated_at": timestamp}
        write_once_json(manifest_path, manifest)
        manifest_reused = False
    reused = parameters_reused and boundary_reused and manifest_reused
    return {
        "experiment": artifact_relative(directory, artifact_root),
        "reused": reused,
        **manifest,
    }


def experiment_finalize_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    directory = experiment_directory(artifact_root, args.route, args.id)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FabricError(f"Experiment does not exist: {directory}")
    if args.result_json.stat().st_size > 32 * 1024 * 1024:
        raise FabricError("Result is larger than 32 MiB; register heavy output as a resource/cache reference")
    result = read_json(args.result_json.resolve())
    result_path = directory / "result.json"
    atomic_write_json(result_path, result)
    manifest = read_json(manifest_path)
    manifest["status"] = args.status
    manifest["result"] = {
        "path": "result.json",
        "sha256": sha256_file(result_path),
        "bytes": result_path.stat().st_size,
    }
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return {
        "experiment": artifact_relative(directory, artifact_root),
        "status": args.status,
        "result": manifest["result"],
    }


def load_records(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    return [read_json(path) for path in sorted(directory.rglob("*.json"))]


def inventory_top_level(root: Path) -> dict[str, Any]:
    root = root.resolve()
    ensure_plain_path(root, label="Inventory root")
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    reparse_skipped = 0
    vanished_entries_skipped = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except FileNotFoundError:
            vanished_entries_skipped += 1
            continue
        for entry in entries:
            path = Path(entry.path)
            try:
                if is_reparse_point(path):
                    reparse_skipped += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                size = entry.stat(follow_symlinks=False).st_size
            except FabricError as exc:
                if not isinstance(exc.__cause__, FileNotFoundError):
                    raise
                vanished_entries_skipped += 1
                continue
            except FileNotFoundError:
                vanished_entries_skipped += 1
                continue
            relative = path.relative_to(root)
            top = relative.parts[0]
            stats[top]["files"] += 1
            stats[top]["bytes"] += size
    rows = [
        {"name": name, "files": values["files"], "bytes": values["bytes"]}
        for name, values in stats.items()
    ]
    rows.sort(key=lambda item: item["bytes"], reverse=True)
    return {
        "root": str(root),
        "files": sum(item["files"] for item in rows),
        "bytes": sum(item["bytes"] for item in rows),
        "reparse_points_skipped": reparse_skipped,
        "vanished_entries_skipped": vanished_entries_skipped,
        "top_level": rows,
    }


def current_lifecycle(artifact_root: Path) -> dict[str, dict[str, Any]]:
    events = load_records(fabric_root(artifact_root) / "catalog" / "lifecycle")
    latest: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=lambda item: (item.get("recorded_at", ""), item.get("event_id", ""))):
        latest[event["resource_id"]] = event
    return latest


def report_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    root = fabric_root(artifact_root)
    objects = load_records(root / "catalog" / "objects")
    registrations = load_records(root / "catalog" / "registrations")
    caches = load_records(root / "catalog" / "caches")
    hard_cases = load_records(root / "hard-cases")
    experiments = [
        read_json(path)
        for path in sorted((root / "experiments").rglob("manifest.json"))
    ] if (root / "experiments").exists() else []
    lifecycle = current_lifecycle(artifact_root)

    object_by_id = {item["resource_id"]: item for item in objects}
    referenced_resources: set[str] = set()
    referenced_caches: set[str] = set()
    for cache in caches:
        referenced_resources.update(cache.get("source_ids", []))
        referenced_resources.update(cache.get("model_ids", []))
        referenced_caches.update(cache.get("parent_cache_keys", []))
    for case in hard_cases:
        referenced_resources.update(case.get("source_ids", []))
        referenced_caches.update(case.get("cache_keys", []))
    for experiment in experiments:
        referenced_resources.update(experiment.get("source_ids", []))
        referenced_caches.update(experiment.get("cache_keys", []))

    evidence_status_bytes: dict[str, int] = defaultdict(int)
    storage_status_bytes: dict[str, int] = defaultdict(int)
    for resource_id, obj in object_by_id.items():
        event = lifecycle.get(resource_id, {})
        evidence_status = event.get("evidence_status", "unknown")
        storage_status = event.get("storage_status", "unknown")
        evidence_status_bytes[evidence_status] += int(obj.get("bytes", 0))
        storage_status_bytes[storage_status] += int(obj.get("bytes", 0))

    report = {
        "schema": "blindassist-resource-utilization-report-v1",
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "summary": {
            "unique_resources": len(objects),
            "unique_resource_bytes": sum(int(item.get("bytes", 0)) for item in objects),
            "registrations": len(registrations),
            "shared_caches": len(caches),
            "shared_cache_bytes": sum(int(item.get("payload_bytes", 0)) for item in caches),
            "hard_cases": len(hard_cases),
            "thin_experiments": len(experiments),
            "referenced_resources": len(referenced_resources),
            "unreferenced_resources": len(set(object_by_id) - referenced_resources),
            "referenced_caches": len(referenced_caches),
            "evidence_status_bytes": dict(sorted(evidence_status_bytes.items())),
            "storage_status_bytes": dict(sorted(storage_status_bytes.items())),
        },
        "unreferenced_resource_ids": sorted(set(object_by_id) - referenced_resources),
        "resources": objects,
        "registrations": registrations,
        "caches": caches,
        "hard_cases": hard_cases,
        "experiments": experiments,
    }
    if args.inventory_root:
        report["logical_inventory"] = inventory_top_level(args.inventory_root)

    output_dir = args.output_dir or (root / "reports" / "current")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "resource-utilization.json"
    markdown_path = output_dir / "resource-utilization.md"
    atomic_write_json(json_path, report)

    lines = [
        "# BlindAssist resource utilization",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Fabric summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["summary"].items():
        if key in {"evidence_status_bytes", "storage_status_bytes"}:
            continue
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Evidence-status bytes", "", "| Status | Bytes |", "| --- | ---: |"])
    for status_name, value in report["summary"]["evidence_status_bytes"].items():
        lines.append(f"| `{status_name}` | `{value}` |")
    lines.extend(["", "## Storage-status bytes", "", "| Status | Bytes |", "| --- | ---: |"])
    for status_name, value in report["summary"]["storage_status_bytes"].items():
        lines.append(f"| `{status_name}` | `{value}` |")
    if "logical_inventory" in report:
        inventory = report["logical_inventory"]
        lines.extend(
            [
                "",
                "## Live logical inventory",
                "",
                f"Root: `{inventory['root']}`",
                f"Reparse points skipped: `{inventory['reparse_points_skipped']}`",
                f"Entries vanished during live scan: `{inventory['vanished_entries_skipped']}`",
                "",
                "| Top-level path | Files | Bytes |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in inventory["top_level"]:
            lines.append(f"| `{row['name']}` | `{row['files']}` | `{row['bytes']}` |")
    atomic_write_text(markdown_path, "\n".join(lines).rstrip() + "\n")
    return {
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        **report["summary"],
    }


def verify_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    root = fabric_root(artifact_root)
    errors: list[str] = []
    objects = load_records(root / "catalog" / "objects")
    object_ids = {item.get("resource_id") for item in objects}
    for item in objects:
        try:
            payload = object_payload_path(item, artifact_root)
            if not payload.exists():
                raise FabricError(f"missing payload {payload}")
            if args.deep:
                scan = scan_payload(payload)
                for field in ("resource_id", "bytes", "file_count", "payload_type"):
                    if scan[field] != item.get(field):
                        raise FabricError(
                            f"{field} mismatch {scan[field]!r} != {item.get(field)!r}"
                        )
        except (FabricError, OSError) as exc:
            errors.append(f"object {item.get('resource_id')}: {exc}")

    caches = load_records(root / "catalog" / "caches")
    cache_keys = {item.get("cache_key") for item in caches}
    for item in caches:
        for resource_id in item.get("source_ids", []) + item.get("model_ids", []):
            if resource_id not in object_ids:
                errors.append(f"cache {item.get('cache_key')}: missing source {resource_id}")
        for parent_cache_key in item.get("parent_cache_keys", []):
            if parent_cache_key not in cache_keys:
                errors.append(
                    f"cache {item.get('cache_key')}: missing parent cache {parent_cache_key}"
                )
        payload = (artifact_root / item.get("payload_path", "")).resolve()
        if not payload.exists():
            errors.append(f"cache {item.get('cache_key')}: missing payload {payload}")
        elif args.deep:
            try:
                scan = scan_payload(payload)
                if scan["sha256"] != item.get("payload_sha256"):
                    errors.append(f"cache {item.get('cache_key')}: payload digest mismatch")
            except FabricError as exc:
                errors.append(f"cache {item.get('cache_key')}: {exc}")

    hard_cases = load_records(root / "hard-cases")
    hard_case_paths = {
        artifact_relative(path, artifact_root)
        for path in (root / "hard-cases").rglob("*.json")
    } if (root / "hard-cases").exists() else set()
    for item in hard_cases:
        for resource_id in item.get("source_ids", []):
            if resource_id not in object_ids:
                errors.append(f"hard case {item.get('id')}: missing source {resource_id}")
        for cache_key in item.get("cache_keys", []):
            if cache_key not in cache_keys:
                errors.append(f"hard case {item.get('id')}: missing cache {cache_key}")

    experiment_count = 0
    if (root / "experiments").exists():
        for manifest_path in sorted((root / "experiments").rglob("manifest.json")):
            experiment_count += 1
            directory = manifest_path.parent
            manifest = read_json(manifest_path)
            extras = {path.name for path in directory.iterdir() if path.is_file()} - ALLOWED_EXPERIMENT_FILES
            if extras:
                errors.append(f"experiment {manifest.get('id')}: non-thin files {sorted(extras)}")
            for resource_id in manifest.get("source_ids", []):
                if resource_id not in object_ids:
                    errors.append(f"experiment {manifest.get('id')}: missing source {resource_id}")
            for cache_key in manifest.get("cache_keys", []):
                if cache_key not in cache_keys:
                    errors.append(f"experiment {manifest.get('id')}: missing cache {cache_key}")
            for hard_case in manifest.get("hard_cases", []):
                if hard_case not in hard_case_paths:
                    errors.append(f"experiment {manifest.get('id')}: missing hard case {hard_case}")
            result = manifest.get("result")
            if result:
                result_path = directory / result.get("path", "")
                if not result_path.is_file():
                    errors.append(f"experiment {manifest.get('id')}: missing result")
                elif sha256_file(result_path) != result.get("sha256"):
                    errors.append(f"experiment {manifest.get('id')}: result digest mismatch")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "objects": len(objects),
        "caches": len(caches),
        "hard_cases": len(hard_cases),
        "experiments": experiment_count,
        "deep": bool(args.deep),
        "errors": errors,
    }
    if errors:
        raise FabricError(json.dumps(result, ensure_ascii=False))
    return result


def add_common_uses(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allowed-use", action="append", default=[])
    parser.add_argument("--forbidden-use", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Add one immutable data/model object to the unique store")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--name", required=True)
    ingest.add_argument("--kind", choices=("data", "model"), required=True)
    ingest.add_argument("--storage-class", choices=("download", "model", "sealed"), required=True)
    ingest.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    ingest.add_argument("--route", required=True)
    ingest.add_argument("--consumer", required=True)
    ingest.add_argument("--evidence-role", required=True)
    ingest.add_argument("--dataset-id")
    ingest.add_argument("--source-unit-id", action="append", default=[])
    ingest.add_argument("--disjoint-key")
    ingest.add_argument("--parent-resource-id", action="append", default=[])
    ingest.add_argument("--owner", required=True)
    ingest.add_argument("--retention-reason", required=True)
    ingest.add_argument("--rebuild-command")
    ingest.add_argument("--rebuild-cost")
    ingest.add_argument(
        "--evidence-status",
        choices=("reserved", "fresh", "development_consumed", "sealed_final", "diagnostic", "unknown"),
        default="unknown",
    )
    ingest.add_argument(
        "--storage-status",
        choices=("active", "shared", "sealed_cold", "rebuildable", "unknown"),
        default="active",
    )
    ingest.add_argument("--reason", required=True)
    ingest.add_argument("--source-uri")
    ingest.add_argument("--license-id")
    add_common_uses(ingest)
    ingest.set_defaults(func=ingest_resource)

    transition = subparsers.add_parser("transition", help="Append a lifecycle event without rewriting history")
    transition.add_argument("resource_id")
    transition.add_argument("--event-id", required=True)
    transition.add_argument(
        "--evidence-status",
        choices=("reserved", "fresh", "development_consumed", "sealed_final", "diagnostic", "unknown"),
        required=True,
    )
    transition.add_argument(
        "--storage-status",
        choices=("active", "shared", "sealed_cold", "rebuildable", "unknown"),
        required=True,
    )
    transition.add_argument("--reason", required=True)
    transition.add_argument("--experiment-id")
    add_common_uses(transition)
    transition.set_defaults(func=transition_command)

    cache_put = subparsers.add_parser("cache-put", help="Create or reuse a normalized/feature cache entry")
    cache_put.add_argument("--layer", choices=("normalized", "features"), required=True)
    cache_put.add_argument("--source-id", action="append", required=True)
    cache_put.add_argument("--parent-cache-key", action="append", default=[])
    cache_put.add_argument("--model-id", action="append", default=[])
    cache_put.add_argument("--transform", required=True)
    cache_put.add_argument("--transform-version", required=True)
    cache_put.add_argument("--parameters-json")
    cache_put.add_argument("--producer", required=True)
    cache_put.add_argument("--code-sha256")
    cache_put.add_argument("--config-sha256")
    cache_put.add_argument("--payload", type=Path, required=True)
    cache_put.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    cache_put.set_defaults(func=cache_put_command)

    cache_json = subparsers.add_parser("cache-json", help="Canonicalize a JSON resource into the shared normalized cache")
    cache_json.add_argument("source_id")
    cache_json.set_defaults(func=cache_json_command)

    hard_case = subparsers.add_parser("hard-case", help="Register a reusable failure/hard-case slice by reference")
    hard_case.add_argument("--id", required=True)
    hard_case.add_argument("--route", required=True)
    hard_case.add_argument("--case-kind", choices=("failure", "hard_case", "evidence_gap"), required=True)
    hard_case.add_argument("--failure-layer", required=True)
    hard_case.add_argument("--evidence-split", required=True)
    hard_case.add_argument("--source-id", action="append", default=[])
    hard_case.add_argument("--cache-key", action="append", default=[])
    hard_case.add_argument("--selector-json")
    hard_case.add_argument("--truth-authority", required=True)
    hard_case.add_argument("--selected-by", required=True)
    hard_case.add_argument("--observed-outcome", required=True)
    hard_case.add_argument("--claim-ceiling", required=True)
    add_common_uses(hard_case)
    hard_case.set_defaults(func=hard_case_command)

    experiment = subparsers.add_parser("experiment-create", help="Create a thin experiment reference directory")
    experiment.add_argument("--id", required=True)
    experiment.add_argument("--route", required=True)
    experiment.add_argument("--question", required=True)
    experiment.add_argument("--evaluator", required=True)
    experiment.add_argument("--status", default="prepared")
    experiment.add_argument("--source-id", action="append", default=[])
    experiment.add_argument("--cache-key", action="append", default=[])
    experiment.add_argument("--hard-case", action="append", default=[])
    experiment.add_argument("--parameters-json")
    experiment.add_argument("--boundary")
    experiment.add_argument("--boundary-file", type=Path)
    experiment.set_defaults(func=experiment_create_command)

    finalize = subparsers.add_parser("experiment-finalize", help="Attach one small JSON result to a thin experiment")
    finalize.add_argument("--id", required=True)
    finalize.add_argument("--route", required=True)
    finalize.add_argument("--result-json", type=Path, required=True)
    finalize.add_argument("--status", required=True)
    finalize.set_defaults(func=experiment_finalize_command)

    report = subparsers.add_parser("report", help="Generate a live utilization report from resource references")
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--inventory-root", type=Path)
    report.set_defaults(func=report_command)

    verify = subparsers.add_parser("verify", help="Verify resource lineage and thin experiment invariants")
    verify.add_argument("--deep", action="store_true")
    verify.set_defaults(func=verify_command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.artifact_root = args.artifact_root.resolve()
        args.artifact_root.mkdir(parents=True, exist_ok=True)
        result = args.func(args)
    except (FabricError, OSError) as exc:
        print(f"RESOURCE_FABRIC_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
