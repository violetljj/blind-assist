#!/usr/bin/env python3
"""Audit local public-video lineage before claiming a prospective positive source."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_prospective_source_inventory_audit_v1"
INVENTORY_SCHEMA = "blindassist_public_video_local_source_inventory_v1"
ALLOWED_CONTINUITY = {"continuous", "static_or_panning", "edited_montage"}
ALLOWED_VIEWPOINTS = {"pedestrian", "vehicle", "static", "mixed"}
ALLOWED_POSITIVE_STATUS = {
    "derivation_positive_exit",
    "held_out_positive_exit",
    "persistent_positive_only",
    "no_positive_exit_observed",
    "rejected_noncausal",
}


def resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def require_bool(entry: dict[str, Any], key: str) -> bool:
    value = entry.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean for {entry.get('source_id')}")
    return value


def audit_inventory(inventory_path: Path, contract_path: Path) -> dict[str, Any]:
    mil.reject_independent_direction(inventory_path)
    mil.reject_independent_direction(contract_path)
    inventory = common.load_json(inventory_path)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"unsupported inventory schema: {inventory.get('schema')}")
    expected_contract_sha = str(inventory.get("frozen_contract_sha256", ""))
    actual_contract_sha = common.sha256_file(contract_path)
    if expected_contract_sha != actual_contract_sha:
        raise ValueError(
            "frozen contract hash mismatch: "
            f"expected {expected_contract_sha}, got {actual_contract_sha}"
        )

    inventory_base = inventory_path.parent
    registry_sources: dict[str, dict[str, Any]] = {}
    registry_attestations: list[dict[str, str]] = []
    registry_paths = inventory.get("source_registry_paths")
    if not isinstance(registry_paths, list) or not registry_paths:
        raise ValueError("source_registry_paths must be a non-empty list")
    for registry_value in registry_paths:
        registry_path = resolve_path(str(registry_value), base=inventory_base)
        mil.reject_independent_direction(registry_path)
        registry = common.load_json(registry_path)
        sources = registry.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"registry sources must be a list: {registry_path}")
        registry_attestations.append({
            "path": str(registry_path),
            "sha256": common.sha256_file(registry_path),
        })
        for source in sources:
            source_id = str(source.get("source_id", ""))
            if not source_id:
                raise ValueError(f"registry source_id is missing: {registry_path}")
            if source_id in registry_sources:
                raise ValueError(f"duplicate source_id across registries: {source_id}")
            registry_sources[source_id] = source

    entries = inventory.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("inventory sources must be a non-empty list")
    inventory_ids = [str(entry.get("source_id", "")) for entry in entries]
    if any(not source_id for source_id in inventory_ids):
        raise ValueError("inventory source_id is missing")
    if len(inventory_ids) != len(set(inventory_ids)):
        raise ValueError("inventory source_id values must be unique")
    missing = sorted(set(registry_sources) - set(inventory_ids))
    extra = sorted(set(inventory_ids) - set(registry_sources))
    if missing or extra:
        raise ValueError(f"inventory coverage mismatch: missing={missing}, extra={extra}")

    blocked_contract_ids = inventory.get("prospective_gate", {}).get(
        "forbidden_influence_contract_ids"
    )
    if not isinstance(blocked_contract_ids, list) or not blocked_contract_ids:
        raise ValueError("prospective gate must list forbidden influence contract ids")
    blocked_contract_ids = {str(value) for value in blocked_contract_ids}
    results: list[dict[str, Any]] = []
    seen_video_shas: dict[str, str] = {}
    for entry in entries:
        source_id = str(entry["source_id"])
        registry_source = registry_sources[source_id]
        inventory_video_path = resolve_path(
            str(entry.get("local_video_path", "")), base=inventory_base
        )
        registry_video_path = resolve_path(
            str(registry_source.get("local_video_path", "")),
            base=inventory_base,
        )
        if inventory_video_path != registry_video_path:
            raise ValueError(f"video path disagrees with registry for {source_id}")
        if not inventory_video_path.is_file():
            raise FileNotFoundError(inventory_video_path)
        actual_video_sha = common.sha256_file(inventory_video_path)
        expected_video_sha = str(entry.get("video_sha256", ""))
        if actual_video_sha != expected_video_sha:
            raise ValueError(
                f"video hash mismatch for {source_id}: "
                f"expected {expected_video_sha}, got {actual_video_sha}"
            )
        if actual_video_sha in seen_video_shas:
            raise ValueError(
                "duplicate video bytes under multiple source ids: "
                f"{seen_video_shas[actual_video_sha]} and {source_id}"
            )
        seen_video_shas[actual_video_sha] = source_id

        continuity = str(entry.get("temporal_continuity", ""))
        viewpoint = str(entry.get("viewpoint", ""))
        positive_status = str(entry.get("positive_exit_status", ""))
        if continuity not in ALLOWED_CONTINUITY:
            raise ValueError(f"invalid temporal_continuity for {source_id}: {continuity}")
        if viewpoint not in ALLOWED_VIEWPOINTS:
            raise ValueError(f"invalid viewpoint for {source_id}: {viewpoint}")
        if positive_status not in ALLOWED_POSITIVE_STATUS:
            raise ValueError(f"invalid positive_exit_status for {source_id}: {positive_status}")
        influence_ids = entry.get("influenced_contract_ids")
        if not isinstance(influence_ids, list):
            raise ValueError(f"influenced_contract_ids must be a list for {source_id}")
        influence_ids = {str(value) for value in influence_ids}
        forbidden_influence = sorted(influence_ids & blocked_contract_ids)
        license_usable = require_bool(entry, "item_level_license_usable")
        original_order = require_bool(entry, "original_temporal_order")
        continuous = continuity == "continuous"
        held_out_positive = positive_status == "held_out_positive_exit"
        eligible = (
            license_usable
            and original_order
            and continuous
            and held_out_positive
            and not forbidden_influence
        )
        pedestrian_eligible = eligible and viewpoint == "pedestrian"
        results.append({
            **entry,
            "local_video_path": str(inventory_video_path),
            "video_sha256_verified": True,
            "forbidden_influence_contract_ids": forbidden_influence,
            "eligible_prospective_positive_exit": eligible,
            "eligible_prospective_pedestrian_exit": pedestrian_eligible,
        })

    eligible = [row for row in results if row["eligible_prospective_positive_exit"]]
    pedestrian = [row for row in results if row["eligible_prospective_pedestrian_exit"]]
    gate = inventory["prospective_gate"]
    minimum_positive = int(gate.get("minimum_independent_positive_exit_sources", 1))
    minimum_pedestrian = int(gate.get("minimum_pedestrian_positive_exit_sources", 1))
    positive_passed = len(eligible) >= minimum_positive
    pedestrian_passed = len(pedestrian) >= minimum_pedestrian
    passed = positive_passed and pedestrian_passed
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": common.sha256_file(inventory_path),
        },
        "frozen_contract": {
            "path": str(contract_path.resolve()),
            "sha256": actual_contract_sha,
        },
        "source_registries": registry_attestations,
        "sources": results,
        "summary": {
            "unique_video_source_count": len(results),
            "eligible_prospective_positive_exit_source_count": len(eligible),
            "eligible_prospective_pedestrian_exit_source_count": len(pedestrian),
            "eligible_source_ids": [row["source_id"] for row in eligible],
            "eligible_pedestrian_source_ids": [row["source_id"] for row in pedestrian],
        },
        "gate": {
            "minimum_independent_positive_exit_sources": minimum_positive,
            "minimum_pedestrian_positive_exit_sources": minimum_pedestrian,
            "independent_positive_exit_gate_passed": positive_passed,
            "pedestrian_positive_exit_gate_passed": pedestrian_passed,
            "passed": passed,
        },
        "decision": (
            "Prospective positive-source gate passed."
            if passed
            else "Keep training, calibration, blind evaluation, Android runtime integration, and production promotion closed until a new eligible positive exit source is added."
        ),
        "evidence_limit": "Large-model source-lineage audit only; classifications are discovery evidence, not human truth, calibration, blind evaluation, training truth, Android authorization, or production evidence.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local public-video lineage and the prospective positive-source gate."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    report = audit_inventory(args.inventory, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
