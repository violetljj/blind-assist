#!/usr/bin/env python3
"""Refresh decision templates without rebuilding or validating experiment evidence.

Only engine_version, failure_layers[*].experiment and global_guardrails may
change. Every other source byte must still match the cached index fingerprint
when the previous config bytes are substituted. This is not a ledger repair.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

if __package__:
    from . import knowledge
else:
    import knowledge


SourceSnapshot = tuple[tuple[str, bytes], ...]


def _json_object(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise knowledge.KnowledgeError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise knowledge.KnowledgeError(f"{path}: expected a JSON object")
    return payload


def _source_snapshot(root: Path) -> SourceSnapshot:
    """Use knowledge._decision_source_fingerprint's exact path order and bytes."""
    repo_root = root.parents[1]
    paths = [knowledge._decision_config_path(root)]
    optional_paths = [knowledge._decision_terminals_path(root)]
    # Schema-2 knowledge.py predates inheritance and excludes it from its
    # fingerprint even if an unrelated WIP inheritance file exists locally.
    inheritance_path = getattr(knowledge, "_decision_inheritance_path", None)
    if callable(inheritance_path):
        optional_paths.append(inheritance_path(root))
    for path in optional_paths:
        if path.is_file():
            paths.append(path)
    paths.extend(sorted((root / "items").glob("*.json")))
    paths.extend(sorted((root / "uses").glob("*.json")))
    ledger = repo_root / "experiments" / "index.jsonl"
    if ledger.is_file():
        paths.append(ledger)
    snapshot = []
    for path in paths:
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = path.as_posix()
        snapshot.append((relative, path.read_bytes()))
    return tuple(snapshot)


def _fingerprint(snapshot: SourceSnapshot, config_bytes: bytes | None = None) -> str:
    digest = hashlib.sha256()
    for position, (relative, raw) in enumerate(snapshot):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(config_bytes if position == 0 and config_bytes is not None else raw)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_template_change(previous: dict[str, Any], current: dict[str, Any]) -> None:
    for label, config in (("previous", previous), ("current", current)):
        errors = knowledge._validate_decision_config(config)
        if errors:
            raise knowledge.KnowledgeError(
                f"invalid {label} decision config:\n - " + "\n - ".join(errors)
            )
    comparable = []
    for config in (previous, current):
        fixed = copy.deepcopy(config)
        fixed.pop("engine_version")
        fixed.pop("global_guardrails")
        for layer in fixed["failure_layers"]:
            layer.pop("experiment")
        comparable.append(fixed)
    if comparable[0] != comparable[1]:
        raise knowledge.KnowledgeError(
            "non-template config change (including retrieval signals, layers, order "
            "or routes); a full decision-index rebuild is required"
        )


def refresh_templates(root: Path, previous_config: Path, *, check_only: bool = False) -> dict[str, Any]:
    root = root.resolve()
    previous_config = previous_config.resolve()
    config_path = knowledge._decision_config_path(root)
    index_path = knowledge._decision_index_path(root)
    previous_bytes = previous_config.read_bytes()
    index_bytes = index_path.read_bytes()
    snapshot = _source_snapshot(root)
    previous = _json_object(previous_bytes, previous_config)
    current = _json_object(snapshot[0][1], config_path)
    cached = _json_object(index_bytes, index_path)
    _validate_template_change(previous, current)

    if cached.get("schema_version") != knowledge.DECISION_INDEX_SCHEMA_VERSION:
        raise knowledge.KnowledgeError("unsupported cached index schema; no schema upgrades are allowed")
    for field in ("mechanisms", "experiments", "associations"):
        if not isinstance(cached.get(field), list):
            raise knowledge.KnowledgeError(f"cached index: {field} must be a list")
    for field in ("engine_version", "failure_layers", "global_guardrails", "route_profiles"):
        if cached.get(field) != previous[field]:
            raise knowledge.KnowledgeError(f"cached index {field} does not match the previous config")

    old_fingerprint = _fingerprint(snapshot, previous_bytes)
    if cached.get("source_fingerprint") != old_fingerprint:
        raise knowledge.KnowledgeError(
            "previous-config source fingerprint does not match the cached index; "
            "unrelated source drift requires a full decision-index rebuild"
        )
    new_fingerprint = _fingerprint(snapshot)
    if new_fingerprint != knowledge._decision_source_fingerprint(root):
        raise knowledge.KnowledgeError(
            "source snapshot or fingerprint algorithm changed; refusing template refresh"
        )

    refreshed = copy.deepcopy(cached)
    for layer, template in zip(refreshed["failure_layers"], current["failure_layers"]):
        layer["experiment"] = copy.deepcopy(template["experiment"])
    refreshed["global_guardrails"] = copy.deepcopy(current["global_guardrails"])
    refreshed["engine_version"] = current["engine_version"]
    refreshed["generated_at"] = date.today().isoformat()
    refreshed["source_fingerprint"] = new_fingerprint
    refreshed["template_only_refresh"] = {
        "mode": "templates_only",
        "experiment_ledger_validated": False,
        "experiment_outcomes_recomputed": False,
        "previous_source_fingerprint": old_fingerprint,
        "previous_config_sha256": hashlib.sha256(previous_bytes).hexdigest(),
        "current_config_sha256": hashlib.sha256(snapshot[0][1]).hexdigest(),
        "previous_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "note": "Cached evidence is preserved; this refresh did NOT validate the experiment ledger.",
    }

    # Recheck all inputs immediately before the existing atomic writer. A changed
    # ledger, source inventory, config, previous-config file or index must not be
    # overwritten or silently incorporated into this narrowly scoped refresh.
    if previous_config.read_bytes() != previous_bytes or _source_snapshot(root) != snapshot:
        raise knowledge.KnowledgeError("sources changed during refresh; no index was written")
    if index_path.read_bytes() != index_bytes:
        raise knowledge.KnowledgeError("cached index changed during refresh; no index was written")
    if not check_only:
        knowledge._write_json_atomic(index_path, refreshed)
    return refreshed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=knowledge.DEFAULT_ROOT)
    parser.add_argument("--previous-config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="check eligibility without writing the index")
    args = parser.parse_args(argv)
    try:
        result = refresh_templates(args.root, args.previous_config, check_only=args.check)
    except (knowledge.KnowledgeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    action = "ELIGIBLE" if args.check else "REFRESHED"
    print(f"{action} templates only: {result['source_fingerprint']}")
    print("Did NOT validate the experiment ledger or recompute outcomes; cached evidence is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
