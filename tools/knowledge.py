#!/usr/bin/env python3
"""Query, validate, and update the BlindAssist research knowledge reserve."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "research" / "knowledge"
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")

ITEM_KINDS = {
    "paper",
    "algorithm",
    "project",
    "dataset",
    "tool",
    "survey",
    "other",
}
USE_STATES = {"candidate", "planned", "active", "adopted", "rejected", "retired"}
ADOPTION_MODES = {
    "reference",
    "mechanism_adaptation",
    "reimplementation",
    "direct_replication",
    "component",
    "dataset",
    "evaluator",
}
REPRODUCTION_STATUSES = {
    "not_attempted",
    "mechanics_only",
    "partial",
    "reproduced",
    "failed",
    "not_applicable",
}
VERDICTS = {
    "not_run",
    "positive",
    "negative",
    "mixed",
    "falsified",
    "not_evaluable",
    "unknown",
}
EVIDENCE_KINDS = {"repo", "experiment", "git", "artifact", "external"}
REFERENCE_PREFIXES = ("https://", "http://", "doi:", "repo:", "git:")
MIGRATION_DISPOSITIONS = {"migrated", "deduplicated", "synthesis"}
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class KnowledgeError(RuntimeError):
    """User-facing knowledge library error."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise KnowledgeError(f"{path}: top-level JSON value must be an object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_records(directory: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not directory.is_dir():
        return records, [f"missing directory: {directory}"]
    for path in sorted(directory.glob("*.json")):
        try:
            record = _read_json(path)
        except KnowledgeError as exc:
            errors.append(str(exc))
            continue
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{path}: missing string id")
            continue
        if record_id in records:
            errors.append(f"duplicate id {record_id}: {path}")
            continue
        if path.stem != record_id:
            errors.append(f"{path}: file stem must equal id {record_id}")
        record["_path"] = path
        records[record_id] = record
    return records, errors


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_string(
    errors: list[str],
    record: dict[str, Any],
    field: str,
    context: str,
    *,
    allow_empty: bool = False,
) -> None:
    value = record.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "string" if allow_empty else "non-empty string"
        errors.append(f"{context}: {field} must be a {qualifier}")


def _check_string_list(
    errors: list[str],
    value: Any,
    field: str,
    context: str,
    *,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list) or any(not _is_nonempty_string(item) for item in value):
        errors.append(f"{context}: {field} must be a list of non-empty strings")
        return []
    if not allow_empty and not value:
        errors.append(f"{context}: {field} must not be empty")
    if len(value) != len(set(value)):
        errors.append(f"{context}: {field} contains duplicates")
    return value


def _check_date(errors: list[str], value: Any, field: str, context: str) -> None:
    if not isinstance(value, str) or not DATE_PATTERN.fullmatch(value):
        errors.append(f"{context}: {field} must use YYYY-MM-DD")
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        errors.append(f"{context}: {field} is not a valid date")


def _check_reference(errors: list[str], value: Any, field: str, context: str) -> None:
    if not _is_nonempty_string(value) or not value.startswith(REFERENCE_PREFIXES):
        errors.append(
            f"{context}: {field} must start with https://, http://, doi:, repo:, or git:"
        )


def _is_safe_repo_relative(reference: str) -> bool:
    if WINDOWS_ABSOLUTE_PATTERN.match(reference) or reference.startswith("/"):
        return False
    if "\\" in reference:
        return False
    parts = PurePosixPath(reference).parts
    return bool(parts) and ".." not in parts and "." not in parts


def _experiment_ids(repo_root: Path) -> set[str]:
    ledger = repo_root / "experiments" / "index.jsonl"
    if not ledger.is_file():
        return set()
    result: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        identifier = payload.get("id")
        if isinstance(identifier, str):
            result.add(identifier)
    return result


def _validate_item(record: dict[str, Any], context: str) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != 1:
        errors.append(f"{context}: schema_version must be 1")
    identifier = record.get("id")
    if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
        errors.append(f"{context}: id must match {ID_PATTERN.pattern}")
    kind = record.get("kind")
    if kind not in ITEM_KINDS:
        errors.append(f"{context}: kind must be one of {sorted(ITEM_KINDS)}")
    _check_string(errors, record, "title", context)
    _check_reference(errors, record.get("canonical_ref"), "canonical_ref", context)
    _check_string_list(errors, record.get("authors"), "authors", context)
    year = record.get("year")
    if year is not None and (not isinstance(year, int) or not 1900 <= year <= 2200):
        errors.append(f"{context}: year must be null or an integer from 1900 to 2200")
    _check_string(errors, record, "venue", context, allow_empty=True)
    _check_string(errors, record, "summary", context)
    _check_string_list(errors, record.get("tags"), "tags", context)
    aliases = _check_string_list(errors, record.get("aliases"), "aliases", context)

    mechanisms = record.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        errors.append(f"{context}: mechanisms must be a non-empty list")
    else:
        mechanism_ids: list[str] = []
        for index, mechanism in enumerate(mechanisms):
            mechanism_context = f"{context}.mechanisms[{index}]"
            if not isinstance(mechanism, dict):
                errors.append(f"{mechanism_context}: must be an object")
                continue
            mechanism_id = mechanism.get("id")
            if not isinstance(mechanism_id, str) or not ID_PATTERN.fullmatch(mechanism_id):
                errors.append(f"{mechanism_context}: id must match {ID_PATTERN.pattern}")
            else:
                mechanism_ids.append(mechanism_id)
            for field in ("name", "description", "limitations"):
                _check_string(errors, mechanism, field, mechanism_context)
            _check_string_list(
                errors, mechanism.get("inputs"), "inputs", mechanism_context
            )
            _check_string_list(
                errors, mechanism.get("outputs"), "outputs", mechanism_context
            )
        if len(mechanism_ids) != len(set(mechanism_ids)):
            errors.append(f"{context}: mechanism ids contain duplicates")

    links = record.get("links")
    if not isinstance(links, list):
        errors.append(f"{context}: links must be a list")
    else:
        for index, link in enumerate(links):
            link_context = f"{context}.links[{index}]"
            if not isinstance(link, dict):
                errors.append(f"{link_context}: must be an object")
                continue
            _check_string(errors, link, "label", link_context)
            _check_reference(errors, link.get("ref"), "ref", link_context)

    for field in ("added_at", "updated_at"):
        _check_date(errors, record.get(field), field, context)
    if isinstance(identifier, str) and identifier in aliases:
        errors.append(f"{context}: aliases must not repeat the canonical id")
    return errors


def _validate_use(
    record: dict[str, Any],
    context: str,
    items: dict[str, dict[str, Any]],
    repo_root: Path,
    known_experiments: set[str],
) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != 1:
        errors.append(f"{context}: schema_version must be 1")
    identifier = record.get("id")
    if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
        errors.append(f"{context}: id must match {ID_PATTERN.pattern}")
    item_id = record.get("item_id")
    if not _is_nonempty_string(item_id):
        errors.append(f"{context}: item_id must be a non-empty string")
    elif item_id not in items:
        errors.append(f"{context}: unknown item_id {item_id}")
    _check_string(errors, record, "route", context)
    mechanisms = _check_string_list(
        errors,
        record.get("mechanism_ids"),
        "mechanism_ids",
        context,
        allow_empty=False,
    )
    if isinstance(item_id, str) and item_id in items:
        valid_mechanisms = {
            mechanism["id"]
            for mechanism in items[item_id].get("mechanisms", [])
            if isinstance(mechanism, dict) and isinstance(mechanism.get("id"), str)
        }
        for mechanism_id in mechanisms:
            if mechanism_id not in valid_mechanisms:
                errors.append(
                    f"{context}: mechanism {mechanism_id} is not defined by {item_id}"
                )
    if record.get("use_state") not in USE_STATES:
        errors.append(f"{context}: use_state must be one of {sorted(USE_STATES)}")
    if record.get("adoption_mode") not in ADOPTION_MODES:
        errors.append(
            f"{context}: adoption_mode must be one of {sorted(ADOPTION_MODES)}"
        )

    usage = record.get("usage")
    if not isinstance(usage, dict):
        errors.append(f"{context}: usage must be an object")
    else:
        for field in (
            "source_scope",
            "project_application",
            "modifications",
            "expected_effect",
        ):
            _check_string(errors, usage, field, f"{context}.usage")

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append(f"{context}: evaluation must be an object")
    else:
        if evaluation.get("reproduction_status") not in REPRODUCTION_STATUSES:
            errors.append(
                f"{context}.evaluation: reproduction_status must be one of "
                f"{sorted(REPRODUCTION_STATUSES)}"
            )
        if evaluation.get("verdict") not in VERDICTS:
            errors.append(
                f"{context}.evaluation: verdict must be one of {sorted(VERDICTS)}"
            )
        for field in ("setup", "effect"):
            _check_string(
                errors, evaluation, field, f"{context}.evaluation", allow_empty=True
            )
        _check_string(errors, evaluation, "claim_boundary", f"{context}.evaluation")
        _check_string_list(
            errors, evaluation.get("metrics"), "metrics", f"{context}.evaluation"
        )

    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        errors.append(f"{context}: evidence must be a list")
    else:
        for index, entry in enumerate(evidence):
            evidence_context = f"{context}.evidence[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{evidence_context}: must be an object")
                continue
            kind = entry.get("kind")
            reference = entry.get("ref")
            if kind not in EVIDENCE_KINDS:
                errors.append(
                    f"{evidence_context}: kind must be one of {sorted(EVIDENCE_KINDS)}"
                )
            _check_string(errors, entry, "ref", evidence_context)
            _check_string(errors, entry, "summary", evidence_context)
            if not isinstance(reference, str):
                continue
            if kind == "repo":
                if not _is_safe_repo_relative(reference):
                    errors.append(f"{evidence_context}: unsafe repository reference")
                elif not (repo_root / reference).is_file():
                    errors.append(
                        f"{evidence_context}: repository reference does not exist: "
                        f"{reference}"
                    )
            elif kind == "experiment" and reference not in known_experiments:
                errors.append(
                    f"{evidence_context}: unknown experiments/index.jsonl id "
                    f"{reference}"
                )
            elif kind == "git":
                revision, separator, git_path = reference.partition(":")
                if (
                    not separator
                    or not revision
                    or not _is_safe_repo_relative(git_path)
                ):
                    errors.append(
                        f"{evidence_context}: git ref must be REVISION:repo/path"
                    )
            elif kind == "artifact":
                if not _is_safe_repo_relative(reference) or not reference.startswith(
                    "artifacts.local/"
                ):
                    errors.append(
                        f"{evidence_context}: artifact ref must be below artifacts.local/"
                    )
            elif kind == "external" and not reference.startswith(
                ("https://", "http://")
            ):
                errors.append(f"{evidence_context}: external ref must be an HTTP URL")

    history = record.get("history")
    if not isinstance(history, list) or not history:
        errors.append(f"{context}: history must be a non-empty list")
    else:
        for index, event in enumerate(history):
            event_context = f"{context}.history[{index}]"
            if not isinstance(event, dict):
                errors.append(f"{event_context}: must be an object")
                continue
            _check_date(errors, event.get("date"), "date", event_context)
            _check_string(errors, event, "change", event_context)

    for field in ("added_at", "updated_at"):
        _check_date(errors, record.get(field), field, context)
    return errors


def _validate_migrations(
    root: Path,
    items: dict[str, dict[str, Any]],
    uses: dict[str, dict[str, Any]],
) -> tuple[list[str], int, int]:
    """Validate auditable coverage receipts for one-time legacy migrations."""

    directory = root / "migrations"
    if not directory.exists():
        return [], 0, 0
    if not directory.is_dir():
        return [f"migration path is not a directory: {directory}"], 0, 0

    errors: list[str] = []
    manifest_count = 0
    mapping_count = 0
    seen_legacy_ids: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        manifest_count += 1
        try:
            manifest = _read_json(path)
        except KnowledgeError as exc:
            errors.append(str(exc))
            continue
        context = f"migration {path.stem}"
        if manifest.get("schema_version") != 1:
            errors.append(f"{context}: schema_version must be 1")
        identifier = manifest.get("id")
        if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
            errors.append(f"{context}: id must match {ID_PATTERN.pattern}")
        elif identifier != path.stem:
            errors.append(f"{context}: file stem must equal id {identifier}")
        _check_date(errors, manifest.get("created_at"), "created_at", context)
        _check_string(errors, manifest, "scope", context)

        source_groups = manifest.get("source_groups")
        if not isinstance(source_groups, list) or not source_groups:
            errors.append(f"{context}: source_groups must be a non-empty list")
            continue
        group_ids: set[str] = set()
        for group_index, group in enumerate(source_groups):
            group_context = f"{context}.source_groups[{group_index}]"
            if not isinstance(group, dict):
                errors.append(f"{group_context}: must be an object")
                continue
            group_id = group.get("id")
            if not isinstance(group_id, str) or not ID_PATTERN.fullmatch(group_id):
                errors.append(f"{group_context}: id must match {ID_PATTERN.pattern}")
            elif group_id in group_ids:
                errors.append(f"{group_context}: duplicate source-group id {group_id}")
            else:
                group_ids.add(group_id)
            _check_string(errors, group, "source_ref", group_context)
            source_sha256 = group.get("source_sha256")
            if source_sha256 is not None and (
                not isinstance(source_sha256, str)
                or not SHA256_PATTERN.fullmatch(source_sha256)
            ):
                errors.append(
                    f"{group_context}: source_sha256 must be null or 64 hex digits"
                )
            expected_entries = group.get("expected_entries")
            mappings = group.get("mappings")
            if not isinstance(expected_entries, int) or expected_entries < 0:
                errors.append(f"{group_context}: expected_entries must be >= 0")
            if not isinstance(mappings, list):
                errors.append(f"{group_context}: mappings must be a list")
                continue
            if isinstance(expected_entries, int) and expected_entries != len(mappings):
                errors.append(
                    f"{group_context}: expected_entries={expected_entries} but "
                    f"mappings={len(mappings)}"
                )
            mapping_count += len(mappings)
            for mapping_index, mapping in enumerate(mappings):
                mapping_context = f"{group_context}.mappings[{mapping_index}]"
                if not isinstance(mapping, dict):
                    errors.append(f"{mapping_context}: must be an object")
                    continue
                legacy_id = mapping.get("legacy_id")
                if not _is_nonempty_string(legacy_id):
                    errors.append(f"{mapping_context}: legacy_id must be non-empty")
                elif legacy_id in seen_legacy_ids:
                    errors.append(
                        f"{mapping_context}: legacy_id {legacy_id} already mapped by "
                        f"{seen_legacy_ids[legacy_id]}"
                    )
                else:
                    seen_legacy_ids[legacy_id] = mapping_context
                if mapping.get("disposition") not in MIGRATION_DISPOSITIONS:
                    errors.append(
                        f"{mapping_context}: disposition must be one of "
                        f"{sorted(MIGRATION_DISPOSITIONS)}"
                    )
                item_id = mapping.get("item_id")
                if not _is_nonempty_string(item_id) or item_id not in items:
                    errors.append(f"{mapping_context}: unknown item_id {item_id}")
                elif isinstance(legacy_id, str) and legacy_id not in items[item_id].get(
                    "aliases", []
                ):
                    errors.append(
                        f"{mapping_context}: item {item_id} does not carry alias "
                        f"{legacy_id}"
                    )
                use_ids = _check_string_list(
                    errors,
                    mapping.get("use_ids"),
                    "use_ids",
                    mapping_context,
                    allow_empty=False,
                )
                for use_id in use_ids:
                    if use_id not in uses:
                        errors.append(f"{mapping_context}: unknown use_id {use_id}")
                    elif isinstance(item_id, str) and uses[use_id].get("item_id") != item_id:
                        errors.append(
                            f"{mapping_context}: use {use_id} belongs to "
                            f"{uses[use_id].get('item_id')}, not {item_id}"
                        )
                _check_string(errors, mapping, "note", mapping_context)

        exclusions = manifest.get("exclusions")
        if not isinstance(exclusions, list):
            errors.append(f"{context}: exclusions must be a list")
        else:
            for exclusion_index, exclusion in enumerate(exclusions):
                exclusion_context = f"{context}.exclusions[{exclusion_index}]"
                if not isinstance(exclusion, dict):
                    errors.append(f"{exclusion_context}: must be an object")
                    continue
                _check_string(errors, exclusion, "scope", exclusion_context)
                _check_string(errors, exclusion, "reason", exclusion_context)
    return errors, manifest_count, mapping_count


def validate_library(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    items, errors = _load_records(root / "items")
    uses, use_load_errors = _load_records(root / "uses")
    errors.extend(use_load_errors)
    for identifier, item in items.items():
        errors.extend(_validate_item(item, f"item {identifier}"))

    aliases: dict[str, str] = {}
    canonical_refs: dict[str, str] = {}
    for identifier in sorted(set(items).intersection(uses)):
        errors.append(f"id is shared by an item and a use: {identifier}")
    for identifier, item in items.items():
        for alias in item.get("aliases", []):
            owner = aliases.get(alias)
            if owner is not None:
                errors.append(f"alias {alias} is shared by {owner} and {identifier}")
            elif alias in items:
                errors.append(f"alias {alias} collides with canonical item id")
            elif alias in uses:
                errors.append(f"alias {alias} collides with use id")
            else:
                aliases[alias] = identifier
        canonical_ref = item.get("canonical_ref")
        if isinstance(canonical_ref, str):
            normalized = canonical_ref.casefold().rstrip("/")
            owner = canonical_refs.get(normalized)
            if owner is not None:
                errors.append(
                    f"canonical_ref is duplicated by {owner} and {identifier}: "
                    f"{canonical_ref}"
                )
            else:
                canonical_refs[normalized] = identifier

    repo_root = root.parents[1]
    known_experiments = _experiment_ids(repo_root)
    for identifier, use in uses.items():
        errors.extend(
            _validate_use(
                use,
                f"use {identifier}",
                items,
                repo_root,
                known_experiments,
            )
        )
    migration_errors, _, _ = _validate_migrations(root, items, uses)
    errors.extend(migration_errors)
    return items, uses, errors


def _clean_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in record.items() if key != "_path"}
        for record in records
    ]


def _require_valid(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    items, uses, errors = validate_library(root)
    if errors:
        raise KnowledgeError("knowledge library is invalid:\n - " + "\n - ".join(errors))
    return items, uses


def _resolve_item(
    requested: str, items: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if requested in items:
        return items[requested]
    for item in items.values():
        if requested in item.get("aliases", []):
            return item
    return None


def _filtered_rows(
    items: dict[str, dict[str, Any]],
    uses: dict[str, dict[str, Any]],
    *,
    route: str | None,
    state: str | None,
    verdict: str | None,
    tag: str | None,
    query: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_folded = query.casefold() if query else None
    for item in sorted(items.values(), key=lambda value: value["id"]):
        linked_uses = [
            use for use in uses.values() if use.get("item_id") == item.get("id")
        ]
        filtered_uses = [
            use
            for use in linked_uses
            if (route is None or use.get("route") == route)
            and (state is None or use.get("use_state") == state)
            and (
                verdict is None
                or use.get("evaluation", {}).get("verdict") == verdict
            )
        ]
        if route is not None or state is not None or verdict is not None:
            if not filtered_uses:
                continue
        else:
            filtered_uses = linked_uses
        if tag is not None and tag not in item.get("tags", []):
            continue
        row = {
            "item": {key: value for key, value in item.items() if key != "_path"},
            "uses": _clean_records(sorted(filtered_uses, key=lambda value: value["id"])),
        }
        if query_folded is not None:
            searchable = json.dumps(row, ensure_ascii=False).casefold()
            if query_folded not in searchable:
                continue
        rows.append(row)
    return rows


def _print_rows(rows: list[dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    print("ID\tKIND\tUSES\tROUTES\tTITLE")
    for row in rows:
        item = row["item"]
        uses = row["uses"]
        routes = ",".join(sorted({use["route"] for use in uses})) or "-"
        print(
            f"{item['id']}\t{item['kind']}\t{len(uses)}\t{routes}\t{item['title']}"
        )


def _command_validate(args: argparse.Namespace) -> int:
    items, uses, errors = validate_library(args.root)
    if errors:
        print("Knowledge library validation failed:", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1
    routes = {use["route"] for use in uses.values()}
    _, migration_count, mapping_count = _validate_migrations(args.root, items, uses)
    print(
        f"PASS knowledge library: items={len(items)} uses={len(uses)} "
        f"routes={len(routes)} migrations={migration_count} "
        f"legacy_mappings={mapping_count}"
    )
    return 0


def _command_list(args: argparse.Namespace) -> int:
    items, uses = _require_valid(args.root)
    rows = _filtered_rows(
        items,
        uses,
        route=args.route,
        state=args.state,
        verdict=args.verdict,
        tag=args.tag,
        query=None,
    )
    _print_rows(rows, args.json)
    return 0


def _command_search(args: argparse.Namespace) -> int:
    items, uses = _require_valid(args.root)
    rows = _filtered_rows(
        items,
        uses,
        route=args.route,
        state=None,
        verdict=None,
        tag=None,
        query=args.query,
    )
    _print_rows(rows, args.json)
    return 0


def _command_show(args: argparse.Namespace) -> int:
    items, uses = _require_valid(args.root)
    if args.id in uses:
        use = uses[args.id]
        payload = {
            "item": {
                key: value
                for key, value in items[use["item_id"]].items()
                if key != "_path"
            },
            "use": {key: value for key, value in use.items() if key != "_path"},
        }
    else:
        item = _resolve_item(args.id, items)
        if item is None:
            raise KnowledgeError(f"unknown item, alias, or use id: {args.id}")
        payload = {
            "item": {key: value for key, value in item.items() if key != "_path"},
            "uses": _clean_records(
                sorted(
                    (
                        use
                        for use in uses.values()
                        if use.get("item_id") == item["id"]
                    ),
                    key=lambda value: value["id"],
                )
            ),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _command_new_item(args: argparse.Namespace) -> int:
    items, _ = _require_valid(args.root)
    if args.id in items or _resolve_item(args.id, items) is not None:
        raise KnowledgeError(f"item id or alias already exists: {args.id}")
    today = date.today().isoformat()
    record: dict[str, Any] = {
        "schema_version": 1,
        "id": args.id,
        "kind": args.kind,
        "title": args.title,
        "canonical_ref": args.canonical_ref,
        "authors": args.author or [],
        "year": args.year,
        "venue": args.venue,
        "summary": args.summary,
        "mechanisms": [
            {
                "id": args.mechanism_id,
                "name": args.mechanism_name,
                "description": args.mechanism_description,
                "inputs": args.mechanism_input or [],
                "outputs": args.mechanism_output or [],
                "limitations": args.mechanism_limitations,
            }
        ],
        "tags": args.tag or [],
        "aliases": args.alias or [],
        "links": [
            {"label": label, "ref": reference}
            for label, reference in (args.link or [])
        ],
        "added_at": today,
        "updated_at": today,
    }
    candidate_errors = _validate_item(record, f"item {args.id}")
    if candidate_errors:
        raise KnowledgeError("new item is invalid:\n - " + "\n - ".join(candidate_errors))
    path = args.root / "items" / f"{args.id}.json"
    if path.exists():
        raise KnowledgeError(f"refusing to overwrite {path}")
    _write_json_atomic(path, record)
    _, _, errors = validate_library(args.root)
    if errors:
        path.unlink(missing_ok=True)
        raise KnowledgeError("new item rejected:\n - " + "\n - ".join(errors))
    print(f"CREATED {path}")
    return 0


def _command_new_use(args: argparse.Namespace) -> int:
    items, uses = _require_valid(args.root)
    if args.id in uses:
        raise KnowledgeError(f"use id already exists: {args.id}")
    item = _resolve_item(args.item, items)
    if item is None:
        raise KnowledgeError(f"unknown item or alias: {args.item}")
    today = date.today().isoformat()
    record: dict[str, Any] = {
        "schema_version": 1,
        "id": args.id,
        "item_id": item["id"],
        "route": args.route,
        "mechanism_ids": args.mechanism,
        "use_state": args.state,
        "adoption_mode": args.mode,
        "usage": {
            "source_scope": args.source_scope,
            "project_application": args.project_application,
            "modifications": args.modifications,
            "expected_effect": args.expected_effect,
        },
        "evaluation": {
            "reproduction_status": "not_attempted",
            "verdict": "not_run",
            "setup": "",
            "effect": "",
            "metrics": [],
            "claim_boundary": args.claim_boundary,
        },
        "evidence": [],
        "history": [{"date": today, "change": args.note}],
        "added_at": today,
        "updated_at": today,
    }
    repo_root = args.root.parents[1]
    candidate_errors = _validate_use(
        record,
        f"use {args.id}",
        items,
        repo_root,
        _experiment_ids(repo_root),
    )
    if candidate_errors:
        raise KnowledgeError("new use is invalid:\n - " + "\n - ".join(candidate_errors))
    path = args.root / "uses" / f"{args.id}.json"
    if path.exists():
        raise KnowledgeError(f"refusing to overwrite {path}")
    _write_json_atomic(path, record)
    _, _, errors = validate_library(args.root)
    if errors:
        path.unlink(missing_ok=True)
        raise KnowledgeError("new use rejected:\n - " + "\n - ".join(errors))
    print(f"CREATED {path}")
    return 0


def _command_update_use(args: argparse.Namespace) -> int:
    items, uses = _require_valid(args.root)
    if args.id not in uses:
        raise KnowledgeError(f"unknown use id: {args.id}")
    updates_requested = any(
        value is not None
        for value in (
            args.state,
            args.mode,
            args.reproduction,
            args.verdict,
            args.setup,
            args.effect,
            args.claim_boundary,
            args.source_scope,
            args.project_application,
            args.modifications,
            args.expected_effect,
            args.mechanism,
            args.metric,
            args.evidence,
        )
    )
    if not updates_requested:
        raise KnowledgeError("update-use needs at least one field change")

    original = {
        key: value for key, value in uses[args.id].items() if key != "_path"
    }
    record = json.loads(json.dumps(original, ensure_ascii=False))
    if args.state is not None:
        record["use_state"] = args.state
    if args.mode is not None:
        record["adoption_mode"] = args.mode
    if args.mechanism is not None:
        record["mechanism_ids"] = args.mechanism
    usage_updates = {
        "source_scope": args.source_scope,
        "project_application": args.project_application,
        "modifications": args.modifications,
        "expected_effect": args.expected_effect,
    }
    for field, value in usage_updates.items():
        if value is not None:
            record["usage"][field] = value
    evaluation_updates = {
        "reproduction_status": args.reproduction,
        "verdict": args.verdict,
        "setup": args.setup,
        "effect": args.effect,
        "claim_boundary": args.claim_boundary,
    }
    for field, value in evaluation_updates.items():
        if value is not None:
            record["evaluation"][field] = value
    if args.metric:
        for metric in args.metric:
            if metric not in record["evaluation"]["metrics"]:
                record["evaluation"]["metrics"].append(metric)
    if args.evidence:
        for kind, reference, summary in args.evidence:
            candidate = {"kind": kind, "ref": reference, "summary": summary}
            if candidate not in record["evidence"]:
                record["evidence"].append(candidate)
    today = date.today().isoformat()
    record["updated_at"] = today
    record["history"].append({"date": today, "change": args.note})

    repo_root = args.root.parents[1]
    candidate_errors = _validate_use(
        record,
        f"use {args.id}",
        items,
        repo_root,
        _experiment_ids(repo_root),
    )
    if candidate_errors:
        raise KnowledgeError(
            "updated use is invalid:\n - " + "\n - ".join(candidate_errors)
        )
    path = uses[args.id]["_path"]
    _write_json_atomic(path, record)
    _, _, errors = validate_library(args.root)
    if errors:
        _write_json_atomic(path, original)
        raise KnowledgeError("update rejected:\n - " + "\n - ".join(errors))
    print(f"UPDATED {path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the BlindAssist research knowledge reserve."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Knowledge directory (defaults to research/knowledge).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate all records.")
    validate_parser.set_defaults(handler=_command_validate)

    list_parser = subparsers.add_parser("list", help="List knowledge items.")
    list_parser.add_argument("--route")
    list_parser.add_argument("--state", choices=sorted(USE_STATES))
    list_parser.add_argument("--verdict", choices=sorted(VERDICTS))
    list_parser.add_argument("--tag")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(handler=_command_list)

    search_parser = subparsers.add_parser("search", help="Full-text search records.")
    search_parser.add_argument("query")
    search_parser.add_argument("--route")
    search_parser.add_argument("--json", action="store_true")
    search_parser.set_defaults(handler=_command_search)

    show_parser = subparsers.add_parser(
        "show", help="Show an item, alias, or use record."
    )
    show_parser.add_argument("id")
    show_parser.set_defaults(handler=_command_show)

    item_parser = subparsers.add_parser("new-item", help="Add one knowledge item.")
    item_parser.add_argument("--id", required=True)
    item_parser.add_argument("--kind", required=True, choices=sorted(ITEM_KINDS))
    item_parser.add_argument("--title", required=True)
    item_parser.add_argument("--canonical-ref", required=True)
    item_parser.add_argument("--author", action="append")
    item_parser.add_argument("--year", type=int)
    item_parser.add_argument("--venue", default="")
    item_parser.add_argument("--summary", required=True)
    item_parser.add_argument("--mechanism-id", required=True)
    item_parser.add_argument("--mechanism-name", required=True)
    item_parser.add_argument("--mechanism-description", required=True)
    item_parser.add_argument("--mechanism-input", action="append")
    item_parser.add_argument("--mechanism-output", action="append")
    item_parser.add_argument("--mechanism-limitations", required=True)
    item_parser.add_argument("--tag", action="append")
    item_parser.add_argument("--alias", action="append")
    item_parser.add_argument(
        "--link",
        action="append",
        nargs=2,
        metavar=("LABEL", "REF"),
    )
    item_parser.set_defaults(handler=_command_new_item)

    use_parser = subparsers.add_parser("new-use", help="Add one route-specific use.")
    use_parser.add_argument("--id", required=True)
    use_parser.add_argument("--item", required=True)
    use_parser.add_argument("--route", required=True)
    use_parser.add_argument("--mechanism", action="append", required=True)
    use_parser.add_argument("--state", choices=sorted(USE_STATES), default="candidate")
    use_parser.add_argument(
        "--mode",
        choices=sorted(ADOPTION_MODES),
        default="mechanism_adaptation",
    )
    use_parser.add_argument("--source-scope", required=True)
    use_parser.add_argument("--project-application", required=True)
    use_parser.add_argument("--modifications", required=True)
    use_parser.add_argument("--expected-effect", required=True)
    use_parser.add_argument("--claim-boundary", required=True)
    use_parser.add_argument(
        "--note",
        default="Created as a route-specific candidate use.",
    )
    use_parser.set_defaults(handler=_command_new_use)

    update_parser = subparsers.add_parser(
        "update-use", help="Update a route-specific use and append its history."
    )
    update_parser.add_argument("id")
    update_parser.add_argument("--state", choices=sorted(USE_STATES))
    update_parser.add_argument("--mode", choices=sorted(ADOPTION_MODES))
    update_parser.add_argument("--mechanism", action="append")
    update_parser.add_argument(
        "--reproduction", choices=sorted(REPRODUCTION_STATUSES)
    )
    update_parser.add_argument("--verdict", choices=sorted(VERDICTS))
    update_parser.add_argument("--setup")
    update_parser.add_argument("--effect")
    update_parser.add_argument("--claim-boundary")
    update_parser.add_argument("--source-scope")
    update_parser.add_argument("--project-application")
    update_parser.add_argument("--modifications")
    update_parser.add_argument("--expected-effect")
    update_parser.add_argument("--metric", action="append")
    update_parser.add_argument(
        "--evidence",
        action="append",
        nargs=3,
        choices=None,
        metavar=("KIND", "REF", "SUMMARY"),
    )
    update_parser.add_argument("--note", required=True)
    update_parser.set_defaults(handler=_command_update_use)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows may inherit a legacy code page even though the knowledge records are
    # UTF-8 and can contain Chinese, mathematical symbols, and author names.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    if getattr(args, "evidence", None):
        invalid_kinds = [
            evidence[0]
            for evidence in args.evidence
            if evidence[0] not in EVIDENCE_KINDS
        ]
        if invalid_kinds:
            parser.error(
                f"--evidence KIND must be one of {sorted(EVIDENCE_KINDS)}; "
                f"got {invalid_kinds}"
            )
    try:
        return int(args.handler(args))
    except KnowledgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
