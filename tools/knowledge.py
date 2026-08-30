#!/usr/bin/env python3
"""Query, validate, and update the BlindAssist research knowledge reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
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
PROMOTED_USE_STATES = {"planned", "active"}
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
CONTEXT_DEFAULT_LIMIT = 4
CONTEXT_VERDICT_PRIORITY = {
    "falsified": 0,
    "negative": 1,
    "mixed": 2,
    "not_evaluable": 3,
    "unknown": 4,
    "positive": 5,
    "not_run": 6,
}
ROUTE_FAMILIES: dict[str, frozenset[str]] = {
    "obstacle-avoidance": frozenset({"obstacle-avoidance", "dtr-r0"}),
    "ten-meter-copilot": frozenset({"ten-meter-copilot", "l10-r0"}),
}
ROUTE_CANONICAL = {
    alias: canonical
    for canonical, aliases in ROUTE_FAMILIES.items()
    for alias in aliases
}
DECISION_SCHEMA_VERSION = 1
DECISION_INDEX_SCHEMA_VERSION = 2
DECISION_DEFAULT_MECHANISM_LIMIT = 2
DECISION_DEFAULT_ATTEMPT_LIMIT = 4
DECISION_CONFIG_RELATIVE = Path("decision") / "config.json"
DECISION_INDEX_RELATIVE = Path("decision") / "index.json"
DECISION_TERMINALS_RELATIVE = Path("decision") / "terminals.json"
DECISION_GOLDEN_RELATIVE = Path("decision") / "golden_cases.json"


class KnowledgeError(RuntimeError):
    """User-facing knowledge library error."""


def _canonical_route(route: str) -> str:
    return ROUTE_CANONICAL.get(route, route)


def _route_family(route: str) -> frozenset[str]:
    canonical = _canonical_route(route)
    return ROUTE_FAMILIES.get(canonical, frozenset({canonical}))


def _route_matches(candidate: Any, requested: str) -> bool:
    return isinstance(candidate, str) and candidate in _route_family(requested)


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
    temporary.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
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
    valid_mechanisms: dict[str, dict[str, Any]] = {}
    if isinstance(item_id, str) and item_id in items:
        valid_mechanisms = {
            mechanism["id"]: mechanism
            for mechanism in items[item_id].get("mechanisms", [])
            if isinstance(mechanism, dict) and isinstance(mechanism.get("id"), str)
        }
        for mechanism_id in mechanisms:
            if mechanism_id not in valid_mechanisms:
                errors.append(
                    f"{context}: mechanism {mechanism_id} is not defined by {item_id}"
                )
    use_state = record.get("use_state")
    if use_state not in USE_STATES:
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
        if use_state in PROMOTED_USE_STATES:
            _check_string(errors, usage, "applicability", f"{context}.usage")
        elif "applicability" in usage:
            _check_string(errors, usage, "applicability", f"{context}.usage")

    if use_state in PROMOTED_USE_STATES:
        for mechanism_id in mechanisms:
            mechanism = valid_mechanisms.get(mechanism_id)
            if mechanism is None:
                continue
            for field in ("inputs", "outputs"):
                values = mechanism.get(field)
                if not isinstance(values, list) or not values:
                    errors.append(
                        f"{context}: promoted mechanism {item_id}#{mechanism_id} "
                        f"must declare non-empty {field}"
                    )

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


def _normalize_search_text(value: str) -> str:
    return re.sub(
        r"[^a-z0-9\u3400-\u9fff]+", " ", value.casefold()
    ).strip()


def _query_match_score(query: str, searchable: str) -> int:
    """Require every meaningful query term and reward an exact phrase match."""
    normalized_query = _normalize_search_text(query)
    normalized_searchable = _normalize_search_text(searchable)
    if not normalized_query or not normalized_searchable:
        return 0

    query_terms = list(
        dict.fromkeys(
            term
            for term in normalized_query.split()
            if len(term) >= 2 or re.search(r"[\u3400-\u9fff]", term)
        )
    )
    if not query_terms:
        return 0

    searchable_terms = set(normalized_searchable.split())

    def term_matches(term: str) -> bool:
        if re.search(r"[\u3400-\u9fff]", term):
            return term in normalized_searchable
        return term in searchable_terms

    if not all(term_matches(term) for term in query_terms):
        return 0

    phrase_bonus = 10_000 if normalized_query in normalized_searchable else 0
    return phrase_bonus + 100 * len(query_terms) + sum(map(len, query_terms))


def _record_query_score(query: str, record: dict[str, Any], primary: str) -> int:
    score = _query_match_score(query, json.dumps(record, ensure_ascii=False))
    if score:
        score += 2 * _query_match_score(query, primary)
    return score


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
    scored_rows: list[tuple[int, dict[str, Any]]] = []
    for item in sorted(items.values(), key=lambda value: value["id"]):
        linked_uses = [
            use for use in uses.values() if use.get("item_id") == item.get("id")
        ]
        filtered_uses = [
            use
            for use in linked_uses
            if (route is None or _route_matches(use.get("route"), route))
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
        score = 0
        if query is not None:
            primary = " ".join(
                [
                    item["id"],
                    item["title"],
                    *item.get("aliases", []),
                    *(
                        mechanism["name"]
                        for mechanism in item.get("mechanisms", [])
                    ),
                ]
            )
            score = _record_query_score(query, row, primary)
            if score == 0:
                continue
        scored_rows.append((score, row))
    if query is not None:
        scored_rows.sort(
            key=lambda value: (
                -value[0],
                value[1]["item"]["title"].casefold(),
                value[1]["item"]["id"],
            )
        )
    return [row for _, row in scored_rows]


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
    if not errors:
        errors.extend(_decision_index_validation_errors(args.root, items))
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


def _context_bucket(use: dict[str, Any]) -> int:
    state = use["use_state"]
    verdict = use["evaluation"]["verdict"]
    if verdict != "not_run" or state == "rejected":
        return 0
    if state in PROMOTED_USE_STATES:
        return 1
    return {
        "adopted": 2,
        "candidate": 3,
        "retired": 4,
    }.get(state, 6)


def _context_entry(item: dict[str, Any], use: dict[str, Any]) -> dict[str, Any]:
    mechanisms = {
        mechanism["id"]: mechanism for mechanism in item.get("mechanisms", [])
    }
    latest_change = use.get("history", [])[-1] if use.get("history") else None
    return {
        "item": {
            "id": item["id"],
            "kind": item["kind"],
            "title": item["title"],
            "canonical_ref": item["canonical_ref"],
            "summary": item["summary"],
            "aliases": item.get("aliases", []),
        },
        "mechanisms": [
            mechanisms[mechanism_id]
            for mechanism_id in use.get("mechanism_ids", [])
            if mechanism_id in mechanisms
        ],
        "use": {
            "id": use["id"],
            "use_state": use["use_state"],
            "adoption_mode": use["adoption_mode"],
            "usage": use["usage"],
            "evaluation": use["evaluation"],
            "evidence": use["evidence"],
            "latest_change": latest_change,
            "updated_at": use["updated_at"],
        },
    }


def _context_sort_key(
    entry: dict[str, Any], query_score: int = 0
) -> tuple[int, int, int, str, str]:
    use = entry["use"]
    return (
        _context_bucket(use),
        CONTEXT_VERDICT_PRIORITY[use["evaluation"]["verdict"]],
        -query_score,
        entry["item"]["title"].casefold(),
        use["id"],
    )


def _select_context_entries(
    entries: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        return []
    if limit is None or len(entries) <= limit:
        return entries

    selected_ids: set[str] = set()
    represented_buckets: set[int] = set()
    for entry in entries:
        bucket = _context_bucket(entry["use"])
        if bucket in represented_buckets:
            continue
        selected_ids.add(entry["use"]["id"])
        represented_buckets.add(bucket)
        if len(selected_ids) == limit:
            break
    for entry in entries:
        if len(selected_ids) == limit:
            break
        selected_ids.add(entry["use"]["id"])
    return [entry for entry in entries if entry["use"]["id"] in selected_ids]


def _context_terminal_entry(
    terminal: dict[str, Any],
    association: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": terminal["id"],
        "status": terminal.get("status"),
        "decision": terminal.get("decision"),
        "summary": terminal.get("question"),
        "successor_requires": terminal.get("successor_requires"),
        "forbidden_repeats": terminal.get("forbidden_repeats", []),
        "evidence": terminal.get("evidence", []),
        "commit": terminal.get("commit"),
        "association": association,
    }


def _context_terminals(
    decision_index: dict[str, Any] | None,
    *,
    route: str,
    query: str | None,
) -> tuple[int, list[dict[str, Any]]]:
    if decision_index is None:
        return 0, []
    associations = {
        association.get("decision_id"): association
        for association in decision_index.get("associations", [])
        if isinstance(association, dict)
        and _is_nonempty_string(association.get("decision_id"))
    }
    route_terminals: list[tuple[int, dict[str, Any]]] = []
    for position, terminal in enumerate(decision_index.get("experiments", [])):
        if not isinstance(terminal, dict) or terminal.get("kind") != "current_terminal":
            continue
        if not any(
            _route_matches(candidate, route)
            for candidate in terminal.get("routes", [])
        ):
            continue
        route_terminals.append(
            (
                position,
                _context_terminal_entry(
                    terminal, associations.get(terminal.get("id"))
                ),
            )
        )

    route_total = len(route_terminals)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for position, terminal in route_terminals:
        primary = " ".join(
            str(terminal.get(field) or "")
            for field in (
                "id",
                "status",
                "decision",
                "summary",
                "successor_requires",
            )
        )
        score = _record_query_score(query, terminal, primary) if query else 0
        if query and score == 0:
            continue
        ranked.append((score, position, terminal))
    ranked.sort(key=lambda value: (-value[0], -value[1], value[2]["id"]))
    return route_total, [value[2] for value in ranked]


def _build_context(
    items: dict[str, dict[str, Any]],
    uses: dict[str, dict[str, Any]],
    *,
    route: str,
    query: str | None,
    limit: int | None,
    decision_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_route = _canonical_route(route)
    route_uses = [
        use for use in uses.values() if _route_matches(use.get("route"), route)
    ]
    route_terminal_count, terminal_entries = _context_terminals(
        decision_index, route=route, query=query
    )
    if not route_uses and route_terminal_count == 0:
        raise KnowledgeError(f"no knowledge records found for route: {route}")

    entries = [
        _context_entry(items[use["item_id"]], use) for use in route_uses
    ]
    query_scores: dict[str, int] = {}
    if query:
        matched_entries: list[dict[str, Any]] = []
        for entry in entries:
            primary = " ".join(
                [
                    entry["item"]["id"],
                    entry["item"]["title"],
                    *entry["item"].get("aliases", []),
                    *(mechanism["name"] for mechanism in entry["mechanisms"]),
                ]
            )
            score = _record_query_score(query, entry, primary)
            if score == 0:
                continue
            query_scores[entry["use"]["id"]] = score
            matched_entries.append(entry)
        entries = matched_entries
    entries.sort(
        key=lambda entry: _context_sort_key(
            entry, query_scores.get(entry["use"]["id"], 0)
        )
    )
    state_counts = Counter(entry["use"]["use_state"] for entry in entries)
    verdict_counts = Counter(
        entry["use"]["evaluation"]["verdict"] for entry in entries
    )
    matched_count = len(entries)
    matched_terminal_count = len(terminal_entries)
    if limit is None:
        selected_terminals = terminal_entries
        selected = entries
    else:
        selected_terminals = terminal_entries[:1]
        selected = _select_context_entries(entries, limit - len(selected_terminals))
    returned_records = len(selected_terminals) + len(selected)
    matched_records = matched_terminal_count + matched_count
    return {
        "route": canonical_route,
        "requested_route": route,
        "query": query,
        "summary": {
            "route_total_terminals": route_terminal_count,
            "matched_terminals": matched_terminal_count,
            "returned_terminals": len(selected_terminals),
            "omitted_terminals": matched_terminal_count - len(selected_terminals),
            "route_total_uses": len(route_uses),
            "matched_uses": matched_count,
            "returned_uses": len(selected),
            "omitted_uses": matched_count - len(selected),
            "matched_records": matched_records,
            "returned_records": returned_records,
            "omitted_records": matched_records - returned_records,
            "states": dict(sorted(state_counts.items())),
            "verdicts": dict(sorted(verdict_counts.items())),
            "selection_policy": (
                "return the newest current terminal first, then represent every "
                "present use tier once before filling: evaluated or rejected; "
                "active or planned; adopted; candidate; retired; "
                + (
                    "query relevance then stable title order within each tier"
                    if query
                    else "stable title order within each tier"
                )
            ),
        },
        "terminals": selected_terminals,
        "entries": selected,
    }


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


def _print_context(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    summary = payload["summary"]
    print(f"# Route knowledge context: {payload['route']}")
    if payload["requested_route"] != payload["route"]:
        print(f"Requested route alias: {payload['requested_route']}")
    if payload["query"]:
        print(f"Query: {payload['query']}")
    print(
        f"Records: matched={summary['matched_records']} "
        f"returned={summary['returned_records']} omitted={summary['omitted_records']}"
    )
    print(
        f"Terminals: route={summary['route_total_terminals']} "
        f"matched={summary['matched_terminals']} "
        f"returned={summary['returned_terminals']} "
        f"omitted={summary['omitted_terminals']}"
    )
    print(
        f"Uses: route={summary['route_total_uses']} matched={summary['matched_uses']} "
        f"returned={summary['returned_uses']} omitted={summary['omitted_uses']}"
    )
    states = ", ".join(
        f"{name}={count}" for name, count in summary["states"].items()
    ) or "none"
    verdicts = ", ".join(
        f"{name}={count}" for name, count in summary["verdicts"].items()
    ) or "none"
    print(f"States: {states}")
    print(f"Verdicts: {verdicts}")

    for terminal in payload["terminals"]:
        print()
        print(
            f"## current terminal / {terminal['status']} — {terminal['id']}"
        )
        print(f"- Decision: {_one_line(terminal['decision'])}")
        print(f"- Summary: {_one_line(terminal['summary'])}")
        print(f"- Successor requires: {_one_line(terminal['successor_requires'])}")
        print(
            "- Forbidden repeats: "
            + ("; ".join(terminal["forbidden_repeats"]) or "-")
        )
        association = terminal.get("association")
        if association:
            print(
                f"- Run association: {association['run_id']} | "
                f"uses={','.join(association['use_ids']) or '-'}"
            )
            print(
                f"- Revision/input: {association['code_revision'] or '-'} | "
                f"{association['input_fingerprint'] or '-'}"
            )
        else:
            print(f"- Revision: {terminal['commit'] or '-'}")
        print(f"- Evidence: {'; '.join(terminal['evidence']) or '-'}")

    for entry in payload["entries"]:
        item = entry["item"]
        use = entry["use"]
        evaluation = use["evaluation"]
        print()
        print(
            f"## {use['use_state']} / {evaluation['verdict']} — {item['title']}"
        )
        aliases = ", ".join(item["aliases"]) or "-"
        print(f"- IDs: {item['id']} | {use['id']} | aliases: {aliases}")
        print(f"- Source: {item['canonical_ref']}")
        if entry["mechanisms"]:
            mechanism_text = "; ".join(
                f"{mechanism['name']}: {mechanism['description']}"
                for mechanism in entry["mechanisms"]
            )
            print(f"- Mechanism: {_one_line(mechanism_text)}")
        usage = use["usage"]
        if usage.get("applicability"):
            print(f"- Applicability: {_one_line(usage['applicability'])}")
        print(f"- Application: {_one_line(usage['project_application'])}")
        print(f"- Modification: {_one_line(usage['modifications'])}")
        print(f"- Expected effect: {_one_line(usage['expected_effect'])}")
        print(
            f"- Evaluation: {evaluation['reproduction_status']} / "
            f"{evaluation['verdict']}"
        )
        if evaluation["effect"]:
            print(f"- Observed effect: {_one_line(evaluation['effect'])}")
        if evaluation["metrics"]:
            print(f"- Metrics: {'; '.join(evaluation['metrics'])}")
        print(f"- Claim boundary: {_one_line(evaluation['claim_boundary'])}")
        evidence_refs = "; ".join(
            evidence["ref"] for evidence in use["evidence"]
        ) or "-"
        print(f"- Evidence: {evidence_refs}")
        if use["latest_change"]:
            latest = use["latest_change"]
            print(
                f"- Latest change: {latest['date']} — "
                f"{_one_line(latest['change'])}"
            )

    if summary["omitted_records"]:
        print()
        print(
            f"Omitted {summary['omitted_records']} lower-priority matches. "
            "Refine with --query or request the complete route with --all."
        )


def _command_context(args: argparse.Namespace) -> int:
    items, uses = _require_valid(args.root)
    if not args.include_all and args.limit < 1:
        raise KnowledgeError("--limit must be at least 1")
    decision_index = (
        _load_decision_index(args.root)
        if _decision_index_path(args.root).is_file()
        else None
    )
    payload = _build_context(
        items,
        uses,
        route=args.route,
        query=args.query,
        limit=None if args.include_all else args.limit,
        decision_index=decision_index,
    )
    _print_context(payload, args.json)
    return 0


def _decision_config_path(root: Path) -> Path:
    return root / DECISION_CONFIG_RELATIVE


def _decision_index_path(root: Path) -> Path:
    return root / DECISION_INDEX_RELATIVE


def _decision_terminals_path(root: Path) -> Path:
    return root / DECISION_TERMINALS_RELATIVE


def _validate_decision_config(
    config: dict[str, Any],
    items: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != DECISION_SCHEMA_VERSION:
        errors.append(
            f"decision config: schema_version must be {DECISION_SCHEMA_VERSION}"
        )
    _check_string(errors, config, "engine_version", "decision config")
    layers = config.get("failure_layers")
    layer_ids: set[str] = set()
    if not isinstance(layers, list) or not layers:
        errors.append("decision config: failure_layers must be a non-empty list")
        layers = []
    for index, layer in enumerate(layers):
        context = f"decision config.failure_layers[{index}]"
        if not isinstance(layer, dict):
            errors.append(f"{context}: must be an object")
            continue
        identifier = layer.get("id")
        if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
            errors.append(f"{context}: id must match {ID_PATTERN.pattern}")
        elif identifier in layer_ids:
            errors.append(f"{context}: duplicate layer id {identifier}")
        else:
            layer_ids.add(identifier)
        for field in ("name", "description"):
            _check_string(errors, layer, field, context)
        for field in (
            "symptom_signals",
            "mechanism_signals",
            "required_evidence",
        ):
            _check_string_list(
                errors, layer.get(field), field, context, allow_empty=False
            )
        experiment = layer.get("experiment")
        if not isinstance(experiment, dict):
            errors.append(f"{context}.experiment: must be an object")
        else:
            for field in (
                "hypothesis",
                "baseline",
                "single_change",
                "cohort",
                "primary_metric",
                "claim_ceiling",
            ):
                _check_string(errors, experiment, field, f"{context}.experiment")
            for field in ("stop_conditions", "not_evaluable_conditions"):
                _check_string_list(
                    errors,
                    experiment.get(field),
                    field,
                    f"{context}.experiment",
                    allow_empty=False,
                )

    route_profiles = config.get("route_profiles")
    if not isinstance(route_profiles, dict):
        errors.append("decision config: route_profiles must be an object")
    else:
        for route, aliases in route_profiles.items():
            if not _is_nonempty_string(route):
                errors.append("decision config: route profile key must be non-empty")
                continue
            _check_string_list(
                errors,
                aliases,
                f"route_profiles.{route}",
                "decision config",
                allow_empty=False,
            )

    overrides = config.get("mechanism_overrides")
    if not isinstance(overrides, dict):
        errors.append("decision config: mechanism_overrides must be an object")
    else:
        known_mechanisms: set[str] = set()
        if items is not None:
            for item in items.values():
                for mechanism in item.get("mechanisms", []):
                    known_mechanisms.add(f"{item['id']}#{mechanism['id']}")
        for mechanism_key, override in overrides.items():
            context = f"decision config.mechanism_overrides.{mechanism_key}"
            if items is not None and mechanism_key not in known_mechanisms:
                errors.append(f"{context}: unknown item#mechanism key")
            if not isinstance(override, dict):
                errors.append(f"{context}: must be an object")
                continue
            override_layers = _check_string_list(
                errors,
                override.get("failure_layers"),
                "failure_layers",
                context,
                allow_empty=False,
            )
            for layer_id in override_layers:
                if layer_id not in layer_ids:
                    errors.append(f"{context}: unknown failure layer {layer_id}")
            _check_string_list(
                errors,
                override.get("signatures"),
                "signatures",
                context,
                allow_empty=False,
            )

    _check_string_list(
        errors,
        config.get("global_guardrails"),
        "global_guardrails",
        "decision config",
        allow_empty=False,
    )
    return errors


def _load_decision_config(
    root: Path,
    items: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = _decision_config_path(root)
    if not path.is_file():
        raise KnowledgeError(f"missing decision config: {path}")
    config = _read_json(path)
    errors = _validate_decision_config(config, items)
    if errors:
        raise KnowledgeError("invalid decision config:\n - " + "\n - ".join(errors))
    return config


def _decision_source_fingerprint(root: Path) -> str:
    repo_root = root.parents[1]
    paths = [_decision_config_path(root)]
    terminals_path = _decision_terminals_path(root)
    if terminals_path.is_file():
        paths.append(terminals_path)
    paths.extend(sorted((root / "items").glob("*.json")))
    paths.extend(sorted((root / "uses").glob("*.json")))
    experiment_ledger = repo_root / "experiments" / "index.jsonl"
    if experiment_ledger.is_file():
        paths.append(experiment_ledger)
    digest = hashlib.sha256()
    for path in paths:
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = path.as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _phrase_hits(text: str, phrases: Iterable[str]) -> list[str]:
    folded = re.sub(
        r"[^a-z0-9\u3400-\u9fff]+", " ", text.casefold()
    ).strip()
    return sorted(
        {
            phrase
            for phrase in phrases
            if re.sub(
                r"[^a-z0-9\u3400-\u9fff]+", " ", phrase.casefold()
            ).strip()
            in folded
        },
        key=lambda value: (-len(value), value.casefold()),
    )


def _layer_scores_for_text(
    text: str,
    layers: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, int]:
    scores: dict[str, int] = {}
    for layer in layers:
        phrases: list[str] = []
        for field in fields:
            phrases.extend(layer.get(field, []))
        hits = _phrase_hits(text, phrases)
        if hits:
            scores[layer["id"]] = sum(2 + min(len(hit.split()), 3) for hit in hits)
    return scores


def _read_experiment_rows(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "experiments" / "index.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict) or not _is_nonempty_string(row.get("id")):
            raise KnowledgeError(f"{path}:{line_number}: experiment row needs string id")
        context = f"{path}:{line_number}"
        for field in ("protocol_id", "decision_id"):
            if field in row and row[field] is not None and not _is_nonempty_string(
                row[field]
            ):
                raise KnowledgeError(f"{context}: {field} must be null or non-empty")
        fingerprint = row.get("input_fingerprint")
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or not SHA256_PATTERN.fullmatch(fingerprint)
        ):
            raise KnowledgeError(
                f"{context}: input_fingerprint must be null or a SHA-256 hex digest"
            )
        for field in ("use_ids", "artifact_refs"):
            if field not in row:
                continue
            values = row[field]
            if not isinstance(values, list) or any(
                not _is_nonempty_string(value) for value in values
            ):
                raise KnowledgeError(
                    f"{context}: {field} must be a list of non-empty strings"
                )
            if len(values) != len(set(values)):
                raise KnowledgeError(f"{context}: {field} contains duplicates")
        rows.append(row)
    return rows


def _read_current_terminals(
    root: Path,
    layer_ids: set[str],
) -> list[dict[str, Any]]:
    path = _decision_terminals_path(root)
    if not path.is_file():
        return []
    payload = _read_json(path)
    errors: list[str] = []
    if payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        errors.append(
            f"{path}: schema_version must be {DECISION_SCHEMA_VERSION}"
        )
    _check_date(errors, payload.get("updated_at"), "updated_at", str(path))
    terminals = payload.get("terminals")
    if not isinstance(terminals, list):
        errors.append(f"{path}: terminals must be a list")
        terminals = []
    seen_ids: set[str] = set()
    repo_root = root.parents[1]
    for index, terminal in enumerate(terminals):
        context = f"{path}.terminals[{index}]"
        if not isinstance(terminal, dict):
            errors.append(f"{context}: must be an object")
            continue
        identifier = terminal.get("id")
        if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
            errors.append(f"{context}: id must match {ID_PATTERN.pattern}")
        elif identifier in seen_ids:
            errors.append(f"{context}: duplicate id {identifier}")
        else:
            seen_ids.add(identifier)
        for field in (
            "route",
            "status",
            "decision",
            "summary",
            "successor_requires",
            "commit",
        ):
            _check_string(errors, terminal, field, context)
        failure_layers = _check_string_list(
            errors,
            terminal.get("failure_layers"),
            "failure_layers",
            context,
            allow_empty=False,
        )
        for layer_id in failure_layers:
            if layer_id not in layer_ids:
                errors.append(f"{context}: unknown failure layer {layer_id}")
        _check_string_list(
            errors,
            terminal.get("forbidden_repeats"),
            "forbidden_repeats",
            context,
            allow_empty=False,
        )
        evidence = _check_string_list(
            errors,
            terminal.get("evidence"),
            "evidence",
            context,
            allow_empty=False,
        )
        for reference in evidence:
            if not _is_safe_repo_relative(reference):
                errors.append(f"{context}: unsafe evidence path {reference}")
            elif not (repo_root / PurePosixPath(reference)).is_file():
                errors.append(f"{context}: missing evidence path {reference}")
    if errors:
        raise KnowledgeError("invalid decision terminals:\n - " + "\n - ".join(errors))
    return terminals


def _infer_routes(text: str, route_profiles: dict[str, list[str]]) -> list[str]:
    folded = text.casefold()
    result: list[str] = []
    for route, aliases in route_profiles.items():
        candidates = [route, *aliases]
        if any(candidate.casefold() in folded for candidate in candidates):
            result.append(route)
    return sorted(result)


def _compact_use(use: dict[str, Any]) -> dict[str, Any]:
    evaluation = use["evaluation"]
    compact = {
        "id": use["id"],
        "route": use["route"],
        "use_state": use["use_state"],
        "reproduction_status": evaluation["reproduction_status"],
        "verdict": evaluation["verdict"],
        "project_application": use["usage"]["project_application"],
        "expected_effect": use["usage"]["expected_effect"],
        "observed_effect": evaluation["effect"],
        "metrics": evaluation["metrics"],
        "claim_boundary": evaluation["claim_boundary"],
        "evidence": [
            {
                "kind": evidence["kind"],
                "ref": evidence["ref"],
                "summary": evidence["summary"],
            }
            for evidence in use.get("evidence", [])
        ],
        "updated_at": use["updated_at"],
    }
    applicability = use["usage"].get("applicability")
    if applicability:
        compact["applicability"] = applicability
    return compact


def _one_run_value(
    run_id: str,
    rows: list[dict[str, Any]],
    field: str,
) -> str | None:
    values = {
        value
        for row in rows
        if _is_nonempty_string(value := row.get(field))
    }
    if len(values) > 1:
        raise KnowledgeError(
            f"experiment run {run_id}: conflicting {field} values {sorted(values)}"
        )
    return next(iter(values), None)


def _build_run_associations(
    experiment_rows: list[dict[str, Any]],
    uses: dict[str, dict[str, Any]],
    current_terminals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for row in experiment_rows:
        rows_by_run.setdefault(row["id"], []).append(row)

    uses_by_run: dict[str, set[str]] = {}
    for use in uses.values():
        for evidence in use.get("evidence", []):
            if evidence.get("kind") == "experiment" and _is_nonempty_string(
                evidence.get("ref")
            ):
                uses_by_run.setdefault(evidence["ref"], set()).add(use["id"])

    terminals_by_decision: dict[str, list[str]] = {}
    for terminal in current_terminals:
        terminals_by_decision.setdefault(terminal["decision"], []).append(
            terminal["id"]
        )

    associations: list[dict[str, Any]] = []
    for run_id, rows in sorted(rows_by_run.items()):
        explicit_use_ids = {
            use_id
            for row in rows
            for use_id in row.get("use_ids", [])
        }
        unknown_use_ids = sorted(explicit_use_ids - uses.keys())
        if unknown_use_ids:
            raise KnowledgeError(
                f"experiment run {run_id}: unknown use_ids {unknown_use_ids}"
            )
        use_ids = sorted(explicit_use_ids | uses_by_run.get(run_id, set()))

        explicit_decision_ids = {
            row["decision_id"]
            for row in rows
            if _is_nonempty_string(row.get("decision_id"))
        }
        inferred_decision_ids = {
            terminal_ids[0]
            for row in rows
            if len(
                terminal_ids := terminals_by_decision.get(row.get("decision"), [])
            )
            == 1
        }
        decision_ids = explicit_decision_ids | inferred_decision_ids
        if len(decision_ids) > 1:
            raise KnowledgeError(
                f"experiment run {run_id}: conflicting decision links "
                f"{sorted(decision_ids)}"
            )

        artifact_refs: list[str] = []
        for row in rows:
            for value in (
                *row.get("artifact_refs", []),
                row.get("artifacts"),
                row.get("report"),
            ):
                if _is_nonempty_string(value) and value not in artifact_refs:
                    artifact_refs.append(value)

        associations.append(
            {
                "run_id": run_id,
                "use_ids": use_ids,
                "protocol_id": _one_run_value(run_id, rows, "protocol_id"),
                "code_revision": _one_run_value(run_id, rows, "commit"),
                "input_fingerprint": _one_run_value(
                    run_id, rows, "input_fingerprint"
                ),
                "artifact_refs": artifact_refs,
                "decision_id": next(iter(decision_ids), None),
                "source_rows": len(rows),
            }
        )
    return associations


def _build_decision_index_payload(
    root: Path,
    items: dict[str, dict[str, Any]],
    uses: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    layers = config["failure_layers"]
    overrides = config["mechanism_overrides"]
    uses_by_item: dict[str, list[dict[str, Any]]] = {}
    for use in uses.values():
        uses_by_item.setdefault(use["item_id"], []).append(use)

    mechanisms: list[dict[str, Any]] = []
    for item in sorted(items.values(), key=lambda value: value["id"]):
        linked_uses = uses_by_item.get(item["id"], [])
        for mechanism in item.get("mechanisms", []):
            key = f"{item['id']}#{mechanism['id']}"
            mechanism_uses = [
                use
                for use in linked_uses
                if mechanism["id"] in use.get("mechanism_ids", [])
            ]
            override = overrides.get(key, {})
            search_parts: list[Any] = [
                item["title"],
                item["summary"],
                *item.get("tags", []),
                mechanism["name"],
                mechanism["description"],
                *mechanism.get("inputs", []),
                *mechanism.get("outputs", []),
                mechanism["limitations"],
                *override.get("signatures", []),
            ]
            for use in mechanism_uses:
                search_parts.extend(
                    [
                        use["route"],
                        use["usage"].get("applicability", ""),
                        use["usage"]["project_application"],
                        use["usage"]["expected_effect"],
                        use["evaluation"]["effect"],
                        *use["evaluation"]["metrics"],
                        use["evaluation"]["claim_boundary"],
                    ]
                )
            search_text = " ".join(str(part) for part in search_parts if part)
            layer_scores = _layer_scores_for_text(
                search_text, layers, ("mechanism_signals",)
            )
            for layer_id in override.get("failure_layers", []):
                layer_scores[layer_id] = max(layer_scores.get(layer_id, 0), 20)
            mechanisms.append(
                {
                    "id": key,
                    "item_id": item["id"],
                    "mechanism_id": mechanism["id"],
                    "kind": item["kind"],
                    "title": item["title"],
                    "canonical_ref": item["canonical_ref"],
                    "summary": item["summary"],
                    "tags": item.get("tags", []),
                    "name": mechanism["name"],
                    "description": mechanism["description"],
                    "inputs": mechanism.get("inputs", []),
                    "outputs": mechanism.get("outputs", []),
                    "limitations": mechanism["limitations"],
                    "signatures": override.get("signatures", []),
                    "layer_scores": dict(sorted(layer_scores.items())),
                    "routes": sorted({use["route"] for use in mechanism_uses}),
                    "uses": [
                        _compact_use(use)
                        for use in sorted(mechanism_uses, key=lambda value: value["id"])
                    ],
                    "search_text": search_text,
                }
            )

    experiments: list[dict[str, Any]] = []
    repo_root = root.parents[1]
    current_terminals = _read_current_terminals(
        root, {layer["id"] for layer in layers}
    )
    experiment_rows = _read_experiment_rows(repo_root)
    associations = _build_run_associations(
        experiment_rows, uses, current_terminals
    )
    association_by_decision = {
        association["decision_id"]: association
        for association in associations
        if association["decision_id"] is not None
    }
    for terminal in current_terminals:
        search_text = " ".join(
            [
                terminal["id"],
                terminal["route"],
                terminal["decision"],
                terminal["summary"],
                terminal["successor_requires"],
                *terminal["forbidden_repeats"],
            ]
        )
        experiments.append(
            {
                "kind": "current_terminal",
                "id": terminal["id"],
                "status": terminal["status"],
                "question": terminal["summary"],
                "baseline": None,
                "change": terminal["successor_requires"],
                "primary_metric": None,
                "decision": terminal["decision"],
                "report": terminal["evidence"][0],
                "source": "decision/terminals.json",
                "commit": terminal["commit"],
                "routes": [terminal["route"]],
                "layer_scores": {
                    layer_id: 30 for layer_id in terminal["failure_layers"]
                },
                "successor_requires": terminal["successor_requires"],
                "forbidden_repeats": terminal["forbidden_repeats"],
                "evidence": terminal["evidence"],
                "association_id": (
                    association_by_decision[terminal["id"]]["run_id"]
                    if terminal["id"] in association_by_decision
                    else None
                ),
                "search_text": search_text,
            }
        )
    for row in experiment_rows:
        search_text = " ".join(
            str(row.get(field) or "")
            for field in (
                "id",
                "question",
                "baseline",
                "change",
                "primary_metric",
                "decision",
                "report",
                "source",
            )
        )
        experiments.append(
            {
                "kind": "experiment",
                "id": row["id"],
                "status": row.get("status"),
                "question": row.get("question"),
                "baseline": row.get("baseline"),
                "change": row.get("change"),
                "primary_metric": row.get("primary_metric"),
                "decision": row.get("decision"),
                "report": row.get("report"),
                "source": row.get("source"),
                "commit": row.get("commit"),
                "association_id": row["id"],
                "routes": _infer_routes(search_text, config["route_profiles"]),
                "layer_scores": _layer_scores_for_text(
                    search_text,
                    layers,
                    ("symptom_signals", "mechanism_signals"),
                ),
                "search_text": search_text,
            }
        )

    return {
        "schema_version": DECISION_INDEX_SCHEMA_VERSION,
        "engine_version": config["engine_version"],
        "generated_at": date.today().isoformat(),
        "source_fingerprint": _decision_source_fingerprint(root),
        "counts": {
            "failure_layers": len(layers),
            "mechanisms": len(mechanisms),
            "uses": len(uses),
            "experiments": len(experiments) - len(current_terminals),
            "current_terminals": len(current_terminals),
            "run_associations": len(associations),
        },
        "failure_layers": layers,
        "route_profiles": config["route_profiles"],
        "global_guardrails": config["global_guardrails"],
        "mechanisms": mechanisms,
        "experiments": experiments,
        "associations": associations,
    }


def _command_build_decision_index(args: argparse.Namespace) -> int:
    items, uses = _require_valid(args.root)
    config = _load_decision_config(args.root, items)
    payload = _build_decision_index_payload(args.root, items, uses, config)
    path = _decision_index_path(args.root)
    _write_json_atomic(path, payload)
    counts = payload["counts"]
    print(
        f"BUILT {path}: layers={counts['failure_layers']} "
        f"mechanisms={counts['mechanisms']} uses={counts['uses']} "
        f"experiments={counts['experiments']} "
        f"current_terminals={counts['current_terminals']} "
        f"run_associations={counts['run_associations']}"
    )
    return 0


def _decision_association_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    associations = payload.get("associations")
    if not isinstance(associations, list):
        return ["decision index: associations must be a list"]
    run_ids: set[str] = set()
    decision_owners: dict[str, str] = {}
    for index, association in enumerate(associations):
        context = f"decision index.associations[{index}]"
        if not isinstance(association, dict):
            errors.append(f"{context}: must be an object")
            continue
        run_id = association.get("run_id")
        if not _is_nonempty_string(run_id):
            errors.append(f"{context}: run_id must be a non-empty string")
        elif run_id in run_ids:
            errors.append(f"{context}: duplicate run_id {run_id}")
        else:
            run_ids.add(run_id)
        for field in ("use_ids", "artifact_refs"):
            _check_string_list(
                errors, association.get(field), field, context
            )
        for field in (
            "protocol_id",
            "code_revision",
            "input_fingerprint",
            "decision_id",
        ):
            value = association.get(field)
            if value is not None and not _is_nonempty_string(value):
                errors.append(f"{context}: {field} must be null or non-empty")
        fingerprint = association.get("input_fingerprint")
        if isinstance(fingerprint, str) and not SHA256_PATTERN.fullmatch(fingerprint):
            errors.append(f"{context}: input_fingerprint must be a SHA-256 digest")
        decision_id = association.get("decision_id")
        if isinstance(decision_id, str):
            owner = decision_owners.get(decision_id)
            if owner is not None and owner != run_id:
                errors.append(
                    f"{context}: decision_id {decision_id} is already linked to {owner}"
                )
            elif isinstance(run_id, str):
                decision_owners[decision_id] = run_id
        source_rows = association.get("source_rows")
        if not isinstance(source_rows, int) or source_rows < 1:
            errors.append(f"{context}: source_rows must be an integer >= 1")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        errors.append("decision index: experiments must be a list")
        experiments = []
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            continue
        association_id = experiment.get("association_id")
        if association_id is not None and association_id not in run_ids:
            errors.append(
                f"decision index.experiments[{index}]: unknown association_id "
                f"{association_id}"
            )
    return errors


def _decision_index_validation_errors(
    root: Path,
    items: dict[str, dict[str, Any]],
) -> list[str]:
    config_path = _decision_config_path(root)
    if not config_path.is_file():
        return []
    errors: list[str] = []
    try:
        config = _read_json(config_path)
        errors.extend(_validate_decision_config(config, items))
    except KnowledgeError as exc:
        return [str(exc)]
    if not errors:
        try:
            _read_current_terminals(
                root,
                {
                    layer["id"]
                    for layer in config.get("failure_layers", [])
                    if isinstance(layer, dict) and isinstance(layer.get("id"), str)
                },
            )
        except KnowledgeError as exc:
            errors.append(str(exc))
    index_path = _decision_index_path(root)
    if not index_path.is_file():
        errors.append(
            f"missing decision index: {index_path}; run build-decision-index"
        )
        return errors
    try:
        payload = _read_json(index_path)
    except KnowledgeError as exc:
        errors.append(str(exc))
        return errors
    if payload.get("schema_version") != DECISION_INDEX_SCHEMA_VERSION:
        errors.append(
            "decision index: schema_version must be "
            f"{DECISION_INDEX_SCHEMA_VERSION}"
        )
    errors.extend(_decision_association_errors(payload))
    expected = _decision_source_fingerprint(root)
    if payload.get("source_fingerprint") != expected:
        errors.append("decision index is stale; run build-decision-index")
    return errors


def _load_decision_index(root: Path) -> dict[str, Any]:
    path = _decision_index_path(root)
    if not path.is_file():
        raise KnowledgeError(
            f"missing decision index: {path}; run build-decision-index"
        )
    payload = _read_json(path)
    if payload.get("schema_version") != DECISION_INDEX_SCHEMA_VERSION:
        raise KnowledgeError(
            f"unsupported decision index schema: {payload.get('schema_version')}"
        )
    for field in ("failure_layers", "mechanisms", "experiments", "associations"):
        if not isinstance(payload.get(field), list):
            raise KnowledgeError(f"decision index: {field} must be a list")
    association_errors = _decision_association_errors(payload)
    if association_errors:
        raise KnowledgeError(
            "invalid decision index associations:\n - "
            + "\n - ".join(association_errors)
        )
    expected_fingerprint = _decision_source_fingerprint(root)
    if payload.get("source_fingerprint") != expected_fingerprint:
        raise KnowledgeError("decision index is stale; run build-decision-index")
    return payload


def _diagnose_layers(
    symptom: str,
    observations: list[str],
    layers: list[dict[str, Any]],
) -> dict[str, Any]:
    symptom_scores = _layer_scores_for_text(
        symptom, layers, ("symptom_signals",)
    )
    observation_text = " ".join(observations)
    observation_scores = _layer_scores_for_text(
        observation_text, layers, ("symptom_signals",)
    )
    layer_by_id = {layer["id"]: layer for layer in layers}
    ranked: list[dict[str, Any]] = []
    combined_text = f"{symptom} {observation_text}"
    for layer in layers:
        score = symptom_scores.get(layer["id"], 0) * 3
        score += observation_scores.get(layer["id"], 0) * 2
        if score <= 0:
            continue
        hits = _phrase_hits(combined_text, layer["symptom_signals"])
        ranked.append(
            {
                "id": layer["id"],
                "name": layer["name"],
                "score": score,
                "matched_signals": hits,
                "description": layer["description"],
                "required_evidence": layer["required_evidence"],
            }
        )
    ranked.sort(key=lambda value: (-value["score"], value["id"]))
    ranked = ranked[:3]
    if not ranked:
        return {
            "status": "unlocalized",
            "confidence": "unknown",
            "layers": [],
            "next_evidence": [
                "原始故障文本/退出码或错误样本",
                "最后一个确认正确的接口输出",
                "UNKNOWN、缺失与负例的独立计数",
            ],
        }
    top_score = ranked[0]["score"]
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0
    if top_score >= 18 and second_score < top_score * 0.65:
        status = "localized"
        confidence = "high"
    elif second_score >= top_score * 0.8:
        status = "ambiguous"
        confidence = "medium"
    else:
        status = "localized"
        confidence = "medium"
    next_evidence: list[str] = []
    for result in ranked[:2]:
        next_evidence.extend(layer_by_id[result["id"]]["required_evidence"])
    return {
        "status": status,
        "confidence": confidence,
        "layers": ranked,
        "next_evidence": list(dict.fromkeys(next_evidence))[:4],
    }


def _query_terms(text: str) -> list[str]:
    folded = text.casefold()
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9._-]+", folded):
        if len(token) >= 3:
            terms.add(token)
        terms.update(
            part for part in re.split(r"[._-]+", token) if len(part) >= 3
        )
    for span in re.findall(r"[\u3400-\u9fff]+", folded):
        if len(span) <= 4:
            terms.add(span)
        else:
            for size in (2, 3, 4):
                for start in range(0, len(span) - size + 1):
                    terms.add(span[start : start + size])
    return sorted(terms, key=lambda value: (-len(value), value))


def _signature_hits(text: str, signatures: Iterable[str]) -> list[str]:
    exact = set(_phrase_hits(text, signatures))
    normalized_text = re.sub(
        r"[^a-z0-9\u3400-\u9fff]+", " ", text.casefold()
    ).strip()
    text_tokens = set(normalized_text.split())
    for signature in signatures:
        normalized_signature = re.sub(
            r"[^a-z0-9\u3400-\u9fff]+", " ", signature.casefold()
        ).strip()
        signature_tokens = {
            token for token in normalized_signature.split() if len(token) >= 3
        }
        if len(signature_tokens) >= 2 and signature_tokens.issubset(text_tokens):
            exact.add(signature)
    return sorted(exact, key=lambda value: (-len(value), value.casefold()))


def _maximal_text_hits(hits: Iterable[str], limit: int = 5) -> list[str]:
    selected: list[str] = []
    for hit in sorted(set(hits), key=lambda value: (-len(value), value)):
        if any(hit in existing for existing in selected):
            continue
        selected.append(hit)
        if len(selected) == limit:
            break
    return selected


def _route_use_preference(use: dict[str, Any]) -> int:
    score = {
        "active": 16,
        "planned": 12,
        "adopted": 10,
        "candidate": 6,
        "retired": -6,
        "rejected": -14,
    }.get(use["use_state"], 0)
    verdict = use["verdict"]
    score += {
        "positive": 10,
        "mixed": 3,
        "not_run": 0,
        "unknown": -1,
        "not_evaluable": -4,
        "negative": -12,
        "falsified": -20,
    }.get(verdict, 0)
    return score


def _rank_mechanisms(
    index: dict[str, Any],
    route: str,
    query_text: str,
    diagnosis: dict[str, Any],
    available: list[str],
    missing: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    terms = _query_terms(query_text)
    route_family = _route_family(route)
    diagnosed_layers = [layer["id"] for layer in diagnosis["layers"]]
    layer_weights = (70, 42, 24)
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for mechanism in index["mechanisms"]:
        score = 0
        reasons: list[str] = []
        for rank, layer_id in enumerate(diagnosed_layers):
            layer_score = mechanism["layer_scores"].get(layer_id, 0)
            if layer_score:
                score += layer_weights[rank] + min(layer_score, 20)
                reasons.append(f"匹配故障层 {layer_id}")
        direct_hits = _signature_hits(
            query_text, mechanism.get("signatures", [])
        )
        if direct_hits:
            score += 55 + 6 * len(direct_hits)
            reasons.append("直接故障签名: " + ", ".join(direct_hits[:3]))
        folded_search = mechanism["search_text"].casefold()
        lexical_hits = _maximal_text_hits(
            (term for term in terms if term in folded_search), limit=8
        )
        if lexical_hits:
            score += min(36, len(lexical_hits) * 3)
            reasons.append("命中: " + ", ".join(lexical_hits[:5]))
        route_uses = [
            use for use in mechanism["uses"] if use["route"] in route_family
        ]
        if route_family.intersection(mechanism["routes"]):
            score += 28
            reasons.append(f"已有 {route} 使用记录")
        selected_use = None
        if route_uses:
            selected_use = max(route_uses, key=_route_use_preference)
            score += _route_use_preference(selected_use)
        input_text = " ".join(mechanism.get("inputs", [])).casefold()
        available_hits = [value for value in available if value.casefold() in input_text]
        missing_hits = [value for value in missing if value.casefold() in input_text]
        if available_hits:
            score += min(12, len(available_hits) * 4)
            reasons.append("已有输入: " + ", ".join(available_hits))
        if missing_hits:
            score -= 35
        if score <= 0:
            continue
        contraindications: list[str] = []
        if missing_hits:
            contraindications.append("缺少前置输入: " + ", ".join(missing_hits))
        if selected_use and selected_use["use_state"] in {"rejected", "retired"}:
            contraindications.append(
                f"路线状态为 {selected_use['use_state']}，不得原样重开"
            )
        if selected_use and selected_use["verdict"] in {
            "negative",
            "falsified",
            "not_evaluable",
        }:
            contraindications.append(
                f"既有 verdict={selected_use['verdict']}；新实验必须满足 successor 条件"
            )
        if mechanism["limitations"]:
            contraindications.append(mechanism["limitations"])
        result = {
            "id": mechanism["id"],
            "item_id": mechanism["item_id"],
            "mechanism_id": mechanism["mechanism_id"],
            "name": mechanism["name"],
            "source_title": mechanism["title"],
            "canonical_ref": mechanism["canonical_ref"],
            "score": score,
            "why": list(dict.fromkeys(reasons)),
            "description": mechanism["description"],
            "inputs": mechanism["inputs"],
            "outputs": mechanism["outputs"],
            "route_history": selected_use,
            "contraindications": contraindications,
        }
        ranked.append((score, mechanism["id"], result))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return [result for _, _, result in ranked[:limit]]


def _terminal_markers(text: str) -> list[str]:
    candidates = re.findall(r"[A-Z][A-Z0-9_]{4,}", text or "")
    terminal_terms = (
        "STOP",
        "CLOSE",
        "NOT_EVALUABLE",
        "NO_FINAL",
        "CONSUMED",
        "FAILED",
        "GATE_NOT_MET",
        "HOLD",
    )
    return list(
        dict.fromkeys(
            candidate
            for candidate in candidates
            if any(term in candidate for term in terminal_terms)
        )
    )[:8]


def _rank_prior_attempts(
    index: dict[str, Any],
    route: str,
    query_text: str,
    diagnosis: dict[str, Any],
    mechanisms: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    terms = _query_terms(query_text)
    route_family = _route_family(route)
    diagnosed_layers = [layer["id"] for layer in diagnosis["layers"]]
    attempts: list[tuple[int, str, dict[str, Any]]] = []
    seen_uses: set[str] = set()
    for mechanism in mechanisms:
        use = mechanism.get("route_history")
        if not use or use["id"] in seen_uses:
            continue
        if use["verdict"] == "not_run" and use["use_state"] not in {
            "rejected",
            "retired",
        }:
            continue
        seen_uses.add(use["id"])
        summary = use["observed_effect"] or use["expected_effect"]
        markers = _terminal_markers(f"{summary} {use['claim_boundary']}")
        attempts.append(
            (
                120 + mechanism["score"],
                use["id"],
                {
                    "kind": "route_use",
                    "id": use["id"],
                    "status": use["use_state"],
                    "verdict": use["verdict"],
                    "summary": summary,
                    "metrics": use["metrics"],
                    "evidence": use["evidence"],
                    "terminal_markers": markers,
                    "do_not_repeat": (
                        use["use_state"] in {"rejected", "retired"}
                        or use["verdict"]
                        in {"negative", "falsified", "not_evaluable"}
                    ),
                },
            )
        )

    for experiment in index["experiments"]:
        experiment_routes = set(experiment["routes"])
        if route and experiment_routes and not route_family.intersection(
            experiment_routes
        ):
            continue
        score = 0
        if route_family.intersection(experiment_routes):
            score += 45
        for rank, layer_id in enumerate(diagnosed_layers):
            if experiment["layer_scores"].get(layer_id, 0):
                score += (45, 28, 16)[rank]
        folded = experiment["search_text"].casefold()
        lexical_hits = [term for term in terms if term in folded]
        score += min(36, len(lexical_hits) * 4)
        if experiment.get("kind") == "current_terminal":
            score += 60 if lexical_hits else 0
        markers = _terminal_markers(str(experiment.get("decision") or ""))
        if markers:
            score += 16
        if score < 40:
            continue
        attempts.append(
            (
                score,
                experiment["id"],
                {
                    "kind": experiment.get("kind", "experiment"),
                    "id": experiment["id"],
                    "status": experiment.get("status"),
                    "question": experiment.get("question"),
                    "decision": experiment.get("decision"),
                    "report": experiment.get("report"),
                    "commit": experiment.get("commit"),
                    "terminal_markers": markers,
                    "do_not_repeat": bool(markers),
                    "successor_requires": experiment.get("successor_requires"),
                    "forbidden_repeats": experiment.get("forbidden_repeats", []),
                    "evidence": experiment.get("evidence", []),
                },
            )
        )
    attempts.sort(key=lambda value: (-value[0], value[1]))
    return [attempt for _, _, attempt in attempts[:limit]]


def _slugify(value: str, fallback: str = "fault") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return (slug[:64] or fallback).strip("-")


def _mechanism_execution_block(
    mechanism: dict[str, Any], attempts: list[dict[str, Any]]
) -> str | None:
    history = mechanism.get("route_history")
    if history and history["use_state"] in {"rejected", "retired"}:
        return f"route use {history['id']} is {history['use_state']}"
    if history and history["verdict"] in {
        "negative",
        "falsified",
        "not_evaluable",
    }:
        return f"route use {history['id']} has verdict={history['verdict']}"
    mechanism_terms = {
        token
        for token in re.findall(
            r"[a-z0-9][a-z0-9_-]+",
            f"{mechanism['name']} {mechanism['description']}".casefold(),
        )
        if len(token) >= 5
    }
    ignored = {
        "route",
        "target",
        "mechanism",
        "development",
        "result",
        "current",
        "prediction",
    }
    mechanism_terms.difference_update(ignored)
    for attempt in attempts:
        if attempt.get("kind") != "current_terminal" or not attempt.get(
            "do_not_repeat"
        ):
            continue
        attempt_text = " ".join(
            [
                str(attempt.get("decision") or ""),
                str(attempt.get("question") or ""),
                " ".join(attempt.get("forbidden_repeats", [])),
            ]
        ).casefold()
        overlapping = {term for term in mechanism_terms if term in attempt_text}
        if len(overlapping) >= 2:
            return (
                f"current terminal {attempt['id']} already consumed the same "
                f"mechanism family ({', '.join(sorted(overlapping)[:4])})"
            )
    return None


def _build_minimum_experiment(
    index: dict[str, Any],
    route: str,
    symptom: str,
    diagnosis: dict[str, Any],
    mechanisms: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_by_id = {layer["id"]: layer for layer in index["failure_layers"]}
    if diagnosis["layers"]:
        layer = layer_by_id[diagnosis["layers"][0]["id"]]
        template = dict(layer["experiment"])
        layer_id = layer["id"]
        layer_name = layer["name"]
    else:
        layer_id = "unlocalized"
        layer_name = "尚未定位"
        template = {
            "hypothesis": "故障尚未定位；最后一个正确接口与第一个错误接口之间存在单一断点。",
            "baseline": "不改变算法，保存原始输入、输出和错误。",
            "single_change": "只加入逐接口观测，定位第一个错误层。",
            "cohort": "一个最小复现样本和一个成功对照。",
            "primary_metric": "first_divergent_interface 被唯一定位",
            "stop_conditions": ["需要同时修改多个接口", "缺少原始错误或输入 identity"],
            "not_evaluable_conditions": ["故障不可复现", "没有成功对照"],
            "claim_ceiling": "只能定位故障层，不能判断机制效果。",
        }
    blocked_candidates: list[dict[str, str]] = []
    selected_mechanism = None
    blocking_terminal_id: str | None = None
    for mechanism in mechanisms:
        blocked_reason = _mechanism_execution_block(mechanism, attempts)
        if blocked_reason:
            blocked_candidates.append(
                {"id": mechanism["id"], "reason": blocked_reason}
            )
            if blocked_reason.startswith("current terminal "):
                blocking_terminal_id = blocked_reason.split()[2]
                break
            continue
        selected_mechanism = mechanism
        break
    factor = (
        selected_mechanism["name"]
        if selected_mechanism is not None
        else "interface instrumentation"
    )
    terminal_markers: list[str] = []
    for attempt in attempts:
        terminal_markers.extend(attempt.get("terminal_markers", []))
    plan_id = f"minexp-{_slugify(route)}-{_slugify(symptom)}"
    successor_attempts = [
        attempt for attempt in attempts if attempt.get("successor_requires")
    ]
    successor_attempts.sort(
        key=lambda attempt: 0 if attempt["id"] == blocking_terminal_id else 1
    )
    successor_requirements = list(
        dict.fromkeys(
            attempt["successor_requires"] for attempt in successor_attempts
        )
    )
    if selected_mechanism is not None:
        single_change = f"{template['single_change']} 候选机制：{factor}。"
        plan_status = "ready"
    else:
        successor_text = (
            successor_requirements[0]
            if successor_requirements
            else "需要新的信息源、表示或新鲜协议后才能执行。"
        )
        single_change = (
            "不重开已消费机制；先把 successor requirement 变成一个 source-admission "
            f"canary：{successor_text}"
        )
        plan_status = "successor_required"
    return {
        "id": plan_id,
        "status": plan_status,
        "route": route,
        "failure_layer": {"id": layer_id, "name": layer_name},
        "selected_mechanism": (
            {
                "id": selected_mechanism["id"],
                "name": selected_mechanism["name"],
            }
            if selected_mechanism
            else None
        ),
        "hypothesis": template["hypothesis"],
        "baseline": template["baseline"],
        "single_change": single_change,
        "cohort": template["cohort"],
        "primary_metric": template["primary_metric"],
        "stop_conditions": template["stop_conditions"],
        "not_evaluable_conditions": template["not_evaluable_conditions"],
        "claim_ceiling": template["claim_ceiling"],
        "guardrails": index["global_guardrails"],
        "blocked_candidates": blocked_candidates,
        "successor_requirements": successor_requirements,
        "prior_terminals_to_preserve": list(dict.fromkeys(terminal_markers)),
        "default_output": (
            f"artifacts.local/knowledge/decision/{plan_id}.json"
        ),
    }


def _build_decision_card(
    index: dict[str, Any],
    *,
    route: str,
    symptom: str,
    observations: list[str],
    available: list[str],
    missing: list[str],
    mechanism_limit: int,
    attempt_limit: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    route = _canonical_route(route)
    query_text = " ".join([symptom, *observations])
    diagnosis = _diagnose_layers(symptom, observations, index["failure_layers"])
    mechanisms = _rank_mechanisms(
        index,
        route,
        query_text,
        diagnosis,
        available,
        missing,
        mechanism_limit,
    )
    attempts = _rank_prior_attempts(
        index,
        route,
        query_text,
        diagnosis,
        mechanisms,
        attempt_limit,
    )
    experiment = _build_minimum_experiment(
        index, route, symptom, diagnosis, mechanisms, attempts
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "engine_version": index["engine_version"],
        "route": route,
        "symptom": symptom,
        "observations": observations,
        "input_availability": {"available": available, "missing": missing},
        "diagnosis": diagnosis,
        "mechanisms": mechanisms,
        "prior_attempts": attempts,
        "minimum_experiment": experiment,
        "runtime_ms": elapsed_ms,
        "source_fingerprint": index["source_fingerprint"],
    }


def _short_text(value: Any, limit: int = 360) -> str:
    text_value = _one_line(value or "")
    if len(text_value) <= limit:
        return text_value
    return text_value[: limit - 1].rstrip() + "…"


def _print_decision_card(card: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return
    diagnosis = card["diagnosis"]
    print(f"# Research decision card — {card['route']}")
    print(f"Symptom: {_one_line(card['symptom'])}")
    print(
        f"Diagnosis: {diagnosis['status']} / {diagnosis['confidence']} "
        f"({card['runtime_ms']} ms engine time)"
    )
    if diagnosis["layers"]:
        for rank, layer in enumerate(diagnosis["layers"], start=1):
            signals = ", ".join(layer["matched_signals"]) or "-"
            print(
                f"  {rank}. {layer['name']} [{layer['id']}] "
                f"score={layer['score']} signals={signals}"
            )
    else:
        print("  No layer has enough evidence; run the localization experiment below.")
    print("Evidence needed: " + "; ".join(diagnosis["next_evidence"]))

    print("\n## Most relevant mechanisms")
    for rank, mechanism in enumerate(card["mechanisms"], start=1):
        print(f"{rank}. {mechanism['name']} ({mechanism['id']})")
        print(f"   Why: {'; '.join(mechanism['why'])}")
        print(f"   Action: {_short_text(mechanism['description'], 260)}")
        if mechanism["route_history"]:
            history = mechanism["route_history"]
            print(
                f"   History: {history['use_state']} / {history['verdict']} "
                f"— {_short_text(history['observed_effect'] or history['expected_effect'], 220)}"
            )
        if mechanism["contraindications"]:
            print(
                "   Limits: "
                + _short_text("; ".join(mechanism["contraindications"]), 300)
            )

    print("\n## What was already tried")
    if not card["prior_attempts"]:
        print("No sufficiently related route attempt was found in the compiled ledger.")
    for attempt in card["prior_attempts"]:
        if attempt["kind"] == "route_use":
            print(
                f"- {attempt['id']}: {attempt['status']} / {attempt['verdict']} "
                f"— {_short_text(attempt['summary'])}"
            )
        else:
            prefix = (
                "CURRENT TERMINAL"
                if attempt["kind"] == "current_terminal"
                else "experiment"
            )
            print(
                f"- [{prefix}] {attempt['id']}: "
                f"{_short_text(attempt['decision'] or attempt['question'])}"
            )
            if attempt.get("successor_requires"):
                print(
                    "  Successor requires: "
                    + _short_text(attempt["successor_requires"])
                )
            if attempt.get("forbidden_repeats"):
                print(
                    "  Do not repeat: "
                    + "; ".join(attempt["forbidden_repeats"])
                )
        if attempt["terminal_markers"]:
            print("  Preserve: " + ", ".join(attempt["terminal_markers"]))

    experiment = card["minimum_experiment"]
    print("\n## Minimum falsifiable experiment")
    print(f"ID: {experiment['id']} ({experiment['status']})")
    for blocked in experiment["blocked_candidates"]:
        print(f"Blocked candidate: {blocked['id']} — {blocked['reason']}")
    print(f"Hypothesis: {_one_line(experiment['hypothesis'])}")
    print(f"Baseline: {_one_line(experiment['baseline'])}")
    print(f"Only change: {_one_line(experiment['single_change'])}")
    print(f"Cohort: {_one_line(experiment['cohort'])}")
    print(f"Primary metric: {_one_line(experiment['primary_metric'])}")
    print("Stop: " + "; ".join(experiment["stop_conditions"]))
    print("Not evaluable: " + "; ".join(experiment["not_evaluable_conditions"]))
    print(f"Claim ceiling: {_one_line(experiment['claim_ceiling'])}")
    if experiment["prior_terminals_to_preserve"]:
        print(
            "Frozen terminals: "
            + ", ".join(experiment["prior_terminals_to_preserve"])
        )
    print(f"Default plan path: {experiment['default_output']}")


def _write_decision_plan(
    root: Path, relative_path: str, card: dict[str, Any]
) -> Path:
    if not _is_safe_repo_relative(relative_path):
        raise KnowledgeError("--write-plan must be a safe repo-relative path")
    repo_root = root.parents[1]
    path = repo_root / PurePosixPath(relative_path)
    _write_json_atomic(path, card)
    return path


def _command_diagnose(args: argparse.Namespace) -> int:
    if not 2 <= args.mechanism_limit <= 4:
        raise KnowledgeError("--mechanism-limit must be from 2 to 4")
    if not 0 <= args.attempt_limit <= 8:
        raise KnowledgeError("--attempt-limit must be from 0 to 8")
    index = _load_decision_index(args.root)
    card = _build_decision_card(
        index,
        route=args.route,
        symptom=args.symptom,
        observations=args.observed or [],
        available=args.available or [],
        missing=args.missing or [],
        mechanism_limit=args.mechanism_limit,
        attempt_limit=args.attempt_limit,
    )
    _print_decision_card(card, args.json)
    if args.write_plan:
        path = _write_decision_plan(args.root, args.write_plan, card)
        if not args.json:
            print(f"WROTE {path}")
    return 0


def _experiment_plan_is_valid(plan: dict[str, Any]) -> bool:
    required_strings = (
        "hypothesis",
        "baseline",
        "single_change",
        "cohort",
        "primary_metric",
        "claim_ceiling",
    )
    if any(not _is_nonempty_string(plan.get(field)) for field in required_strings):
        return False
    for field in ("stop_conditions", "not_evaluable_conditions", "guardrails"):
        value = plan.get(field)
        if not isinstance(value, list) or not value or any(
            not _is_nonempty_string(item) for item in value
        ):
            return False
    guardrail_text = " ".join(plan["guardrails"])
    return (
        "UNKNOWN" in guardrail_text
        and "已消费" in guardrail_text
        and "一个" in guardrail_text
    )


def _load_golden_cases(root: Path) -> dict[str, Any]:
    path = root / DECISION_GOLDEN_RELATIVE
    if not path.is_file():
        raise KnowledgeError(f"missing golden cases: {path}")
    payload = _read_json(path)
    if payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise KnowledgeError(
            f"{path}: schema_version must be {DECISION_SCHEMA_VERSION}"
        )
    cases = payload.get("cases")
    gates = payload.get("gates")
    if not isinstance(cases, list) or not cases:
        raise KnowledgeError(f"{path}: cases must be a non-empty list")
    if not isinstance(gates, dict):
        raise KnowledgeError(f"{path}: gates must be an object")
    return payload


def _evaluate_golden_cases(
    index: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    layer_passes = 0
    mechanism_passes = 0
    mechanism_cases = 0
    history_passes = 0
    history_cases = 0
    invalid_experiments = 0
    max_runtime_ms = 0.0
    for case in payload["cases"]:
        card = _build_decision_card(
            index,
            route=case["route"],
            symptom=case["symptom"],
            observations=case.get("observed", []),
            available=case.get("available", []),
            missing=case.get("missing", []),
            mechanism_limit=4,
            attempt_limit=4,
        )
        actual_layers = [
            layer["id"] for layer in card["diagnosis"]["layers"][:2]
        ]
        layer_pass = bool(set(case["expected_layers"]).intersection(actual_layers))
        layer_passes += int(layer_pass)
        actual_mechanisms = [mechanism["id"] for mechanism in card["mechanisms"]]
        expected_mechanisms = case.get("expected_mechanisms", [])
        mechanism_pass = None
        if expected_mechanisms:
            mechanism_cases += 1
            mechanism_pass = all(
                mechanism in actual_mechanisms for mechanism in expected_mechanisms
            )
            mechanism_passes += int(mechanism_pass)
        actual_attempts = [attempt["id"] for attempt in card["prior_attempts"]]
        expected_attempts = case.get("expected_attempts", [])
        history_pass = None
        if expected_attempts:
            history_cases += 1
            history_pass = all(
                attempt in actual_attempts for attempt in expected_attempts
            )
            history_passes += int(history_pass)
        experiment_valid = _experiment_plan_is_valid(card["minimum_experiment"])
        invalid_experiments += int(not experiment_valid)
        max_runtime_ms = max(max_runtime_ms, float(card["runtime_ms"]))
        case_results.append(
            {
                "id": case["id"],
                "layer_pass": layer_pass,
                "expected_layers": case["expected_layers"],
                "actual_layers": actual_layers,
                "mechanism_pass": mechanism_pass,
                "expected_mechanisms": expected_mechanisms,
                "actual_mechanisms": actual_mechanisms,
                "history_pass": history_pass,
                "expected_attempts": expected_attempts,
                "actual_attempts": actual_attempts,
                "experiment_valid": experiment_valid,
                "runtime_ms": card["runtime_ms"],
            }
        )

    layer_recall = layer_passes / len(case_results)
    mechanism_recall = (
        mechanism_passes / mechanism_cases if mechanism_cases else 1.0
    )
    history_recall = history_passes / history_cases if history_cases else 1.0
    gates = payload["gates"]
    gate_results = {
        "top2_layer_recall": layer_recall >= gates["top2_layer_recall"],
        "top4_mechanism_recall": (
            mechanism_recall >= gates["top4_mechanism_recall"]
        ),
        "top4_history_recall": history_recall >= gates["top4_history_recall"],
        "max_engine_runtime_ms": (
            max_runtime_ms <= gates["max_engine_runtime_ms"]
        ),
        "invalid_experiment_count": (
            invalid_experiments <= gates["invalid_experiment_count"]
        ),
    }
    return {
        "id": payload["id"],
        "cases": len(case_results),
        "metrics": {
            "top2_layer_recall": round(layer_recall, 4),
            "top4_mechanism_recall": round(mechanism_recall, 4),
            "top4_history_recall": round(history_recall, 4),
            "max_engine_runtime_ms": round(max_runtime_ms, 2),
            "invalid_experiment_count": invalid_experiments,
        },
        "required_gates": gates,
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "case_results": case_results,
    }


def _command_evaluate_decision_engine(args: argparse.Namespace) -> int:
    index = _load_decision_index(args.root)
    golden = _load_golden_cases(args.root)
    result = _evaluate_golden_cases(index, golden)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} decision engine: cases={result['cases']}")
        for metric, value in result["metrics"].items():
            gate = result["required_gates"][metric]
            gate_pass = result["gate_results"][metric]
            print(f"- {metric}={value} gate={gate} pass={gate_pass}")
        failures = [
            case
            for case in result["case_results"]
            if not case["layer_pass"]
            or case["mechanism_pass"] is False
            or case["history_pass"] is False
            or not case["experiment_valid"]
        ]
        for case in failures:
            print(
                f"- CASE {case['id']}: layers={case['actual_layers']} "
                f"mechanisms={case['actual_mechanisms']} "
                f"attempts={case['actual_attempts']}"
            )
    return 0 if result["passed"] else 1


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
        "route": _canonical_route(args.route),
        "mechanism_ids": args.mechanism,
        "use_state": args.state,
        "adoption_mode": args.mode,
        "usage": {
            "source_scope": args.source_scope,
            "project_application": args.project_application,
            "modifications": args.modifications,
            "expected_effect": args.expected_effect,
            **(
                {"applicability": args.applicability}
                if args.applicability is not None
                else {}
            ),
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
            args.applicability,
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
        "applicability": args.applicability,
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

    context_parser = subparsers.add_parser(
        "context",
        help="Build a compact route-specific research context.",
    )
    context_parser.add_argument("--route", required=True)
    context_parser.add_argument("--query")
    context_size = context_parser.add_mutually_exclusive_group()
    context_size.add_argument(
        "--limit",
        type=int,
        default=CONTEXT_DEFAULT_LIMIT,
        help=f"Maximum returned terminal + use records (default: {CONTEXT_DEFAULT_LIMIT}).",
    )
    context_size.add_argument(
        "--all",
        action="store_true",
        dest="include_all",
        help="Return every matching terminal and use for the route.",
    )
    context_parser.add_argument("--json", action="store_true")
    context_parser.set_defaults(handler=_command_context)

    build_decision_parser = subparsers.add_parser(
        "build-decision-index",
        help="Compile items, uses, and experiments into the fast decision index.",
    )
    build_decision_parser.set_defaults(handler=_command_build_decision_index)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Locate a failure layer and generate a research decision card.",
    )
    diagnose_parser.add_argument("--route", required=True)
    diagnose_parser.add_argument("--symptom", required=True)
    diagnose_parser.add_argument(
        "--observed",
        action="append",
        help="Observed fact; repeat for multiple observations.",
    )
    diagnose_parser.add_argument(
        "--available",
        action="append",
        help="Available input or capability; repeat as needed.",
    )
    diagnose_parser.add_argument(
        "--missing",
        action="append",
        help="Known missing input or capability; repeat as needed.",
    )
    diagnose_parser.add_argument(
        "--mechanism-limit",
        type=int,
        default=DECISION_DEFAULT_MECHANISM_LIMIT,
        help="Number of mechanisms to return (2-4).",
    )
    diagnose_parser.add_argument(
        "--attempt-limit",
        type=int,
        default=DECISION_DEFAULT_ATTEMPT_LIMIT,
        help="Maximum historical attempts to return (0-8).",
    )
    diagnose_parser.add_argument("--write-plan")
    diagnose_parser.add_argument("--json", action="store_true")
    diagnose_parser.set_defaults(handler=_command_diagnose)

    evaluate_decision_parser = subparsers.add_parser(
        "evaluate-decision-engine",
        help="Run the frozen fault-localization and retrieval golden cases.",
    )
    evaluate_decision_parser.add_argument("--json", action="store_true")
    evaluate_decision_parser.set_defaults(handler=_command_evaluate_decision_engine)

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
    use_parser.add_argument(
        "--applicability",
        help="Required when the new use starts in planned or active state.",
    )
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
    update_parser.add_argument("--applicability")
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
