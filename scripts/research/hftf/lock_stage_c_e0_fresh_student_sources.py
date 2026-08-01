#!/usr/bin/env python3
"""Validate the frozen HFTF Stage C E0 fresh-source lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_hftf_stage_c_fresh_foot_ground_student_canary_e0"
STATUS = "FROZEN_BEFORE_FRESH_RGB_DEPTH_OR_GEOMETRY_LABEL_OUTCOME"
RESULT_SCHEMA = "blindassist_hftf_stage_c_e0_fresh_source_lock_result"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve_inside(base: Path, value: str) -> Path:
    resolved = (base / value).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"Frozen path leaves repository: {resolved}") from error
    return resolved


def _validate_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected={expected} actual={actual}"
        )


def _select_sources(
    ledger: list[dict[str, Any]],
    consumed: set[str],
    count: int = 6,
) -> list[dict[str, Any]]:
    healthy = sorted(
        (
            item
            for item in ledger
            if item.get("metadata_healthy")
            and item.get("trajectory") not in consumed
        ),
        key=lambda item: (
            int(item["total_bytes"]),
            str(item["trajectory"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    dates: set[str] = set()
    for item in healthy:
        recording_date = str(item["recording_date"])
        if recording_date in dates:
            continue
        dates.add(recording_date)
        selected.append(item)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Inventory cannot satisfy six unique-date sources")
    return selected


def _canonical_inventory_source(
    item: dict[str, Any], role: str
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for kind in ("pose", "rgb", "depth"):
        metadata = item["files"][kind]
        files[kind] = {
            "path": item["repo_paths"][kind],
            "size_bytes": int(metadata["size_bytes"]),
            "sha256": str(metadata["sha256"]),
        }
    return {
        "role": role,
        "trajectory": str(item["trajectory"]),
        "recording_date": str(item["recording_date"]),
        "rows": int(item["rows"]),
        "camera_height_m": float(item["camera_height_m"]),
        "total_bytes": int(item["total_bytes"]),
        "files": files,
    }


def _validate_selection(
    protocol: dict[str, Any],
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    binding = protocol["dataset_binding"]
    if inventory.get("terminal") != "C0_EGOWALK_METADATA_COHORT_LOCKED":
        raise ValueError("Inventory is not the locked C0 metadata inventory")
    if inventory.get("dataset_repo") != binding["dataset_repo"]:
        raise ValueError("Dataset repository binding mismatch")
    if inventory.get("dataset_revision") != binding["dataset_revision"]:
        raise ValueError("Dataset revision binding mismatch")
    if int(inventory.get("metadata_healthy_count", -1)) != 95:
        raise ValueError("Frozen healthy inventory count mismatch")
    if inventory.get("rgb_or_depth_media_content_read"):
        raise ValueError("Inventory unexpectedly read RGB/depth content")

    roles = ["train", "train", "train", "train", "dev", "heldout"]
    selected = _select_sources(
        inventory["inventory_ledger"],
        set(binding["consumed_trajectory_exclusions"]),
    )
    canonical = [
        _canonical_inventory_source(item, role)
        for item, role in zip(selected, roles)
    ]
    if canonical != protocol["frozen_sources"]:
        raise ValueError("Frozen E0 source cohort does not recompute exactly")
    if len({item["trajectory"] for item in canonical}) != 6:
        raise ValueError("Frozen trajectory IDs are not unique")
    if len({item["recording_date"] for item in canonical}) != 6:
        raise ValueError("Frozen recording dates are not unique")
    return canonical


def _validate_protocol_bindings(
    protocol_path: Path,
    protocol: dict[str, Any],
    inventory_path: Path,
    pretrained_weight_path: Path,
    repo_root: Path,
) -> None:
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise ValueError("Stage C E0 protocol is not frozen")

    parents = protocol["parent_bindings"]
    protocol_dir = protocol_path.parent
    parent_specs = (
        (
            protocol_dir / parents["d1_result_path"],
            parents["d1_result_sha256"],
            "D1 result document",
        ),
        (
            repo_root / parents["d1_report_path"],
            parents["d1_report_sha256"],
            "D1 formal report",
        ),
        (
            protocol_dir / parents["d1_protocol_path"],
            parents["d1_protocol_sha256"],
            "D1 protocol",
        ),
        (
            protocol_dir / parents["d1_runner_path"],
            parents["d1_runner_sha256"],
            "D1 runner",
        ),
        (
            protocol_dir / parents["d0_runner_path"],
            parents["d0_runner_sha256"],
            "D0 runner",
        ),
    )
    for path, expected, label in parent_specs:
        resolved = path.resolve()
        try:
            resolved.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"{label} leaves repository") from error
        _validate_hash(resolved, expected, label)

    binding = protocol["dataset_binding"]
    _validate_hash(
        inventory_path,
        binding["inventory_report_sha256"],
        "C0 inventory report",
    )
    inventory = _load_json(inventory_path)
    metadata_root = Path(str(inventory["metadata_root"])).resolve()
    for relative, expected in binding["metadata_files"].items():
        _validate_hash(
            metadata_root / relative,
            expected,
            f"metadata {relative}",
        )

    weights = protocol["student_contract"]["pretrained_encoder"]
    if pretrained_weight_path.stat().st_size != int(weights["size_bytes"]):
        raise ValueError("Pretrained encoder size mismatch")
    _validate_hash(
        pretrained_weight_path,
        weights["sha256"],
        "Pretrained encoder",
    )


def run(
    protocol_path: Path,
    inventory_path: Path,
    pretrained_weight_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    protocol_path = _resolve_inside(repo_root, str(protocol_path))
    inventory_path = _resolve_inside(repo_root, str(inventory_path))
    pretrained_weight_path = _resolve_inside(
        repo_root, str(pretrained_weight_path)
    )
    protocol = _load_json(protocol_path)
    _validate_protocol_bindings(
        protocol_path,
        protocol,
        inventory_path,
        pretrained_weight_path,
        repo_root,
    )
    inventory = _load_json(inventory_path)
    selected = _validate_selection(protocol, inventory)
    return {
        "schema": RESULT_SCHEMA,
        "terminal": "E0_FRESH_SOURCE_LOCK_VALIDATED",
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "dataset_repo": protocol["dataset_binding"]["dataset_repo"],
        "dataset_revision": protocol["dataset_binding"]["dataset_revision"],
        "selection_rule": protocol["dataset_binding"]["selection_rule"],
        "role_rule": protocol["dataset_binding"]["role_rule"],
        "selected_sources": selected,
        "selected_source_count": len(selected),
        "selected_total_bytes": sum(
            int(item["total_bytes"]) for item in selected
        ),
        "recording_dates_unique": len(
            {item["recording_date"] for item in selected}
        )
        == len(selected),
        "pretrained_encoder_path": str(pretrained_weight_path),
        "pretrained_encoder_sha256": _sha256(pretrained_weight_path),
        "rgb_or_depth_media_content_read": False,
        "fresh_geometry_label_outcome_read": False,
        "student_output_read": False,
        "exact_selected_media_acquisition_authorized": True,
        "teacher_corpus_generation_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_artifacts_output(path: Path, repo_root: Path) -> Path:
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("Output must stay under artifacts.local") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--pretrained-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    first = run(
        args.protocol,
        args.inventory,
        args.pretrained_weight,
        repo_root,
    )
    second = run(
        args.protocol,
        args.inventory,
        args.pretrained_weight,
        repo_root,
    )
    deterministic = (
        json.dumps(first, sort_keys=True, separators=(",", ":"))
        == json.dumps(second, sort_keys=True, separators=(",", ":"))
    )
    if not deterministic:
        raise ValueError("Source-lock result is not byte deterministic")
    first["determinism_check"] = {
        "second_run_payload_byte_exact": True
    }
    output = _require_artifacts_output(args.output, repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(first, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "terminal": first["terminal"],
                "selected_source_count": first["selected_source_count"],
                "selected_total_bytes": first["selected_total_bytes"],
                "deterministic": True,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
