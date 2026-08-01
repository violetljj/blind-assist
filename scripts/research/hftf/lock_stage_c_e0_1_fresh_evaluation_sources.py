#!/usr/bin/env python3
"""Validate HFTF Stage C E0.1 fresh dev/heldout source lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_hftf_stage_c_foot_ground_student_canary_e0_1"
STATUS = "FROZEN_BEFORE_FRESH_EVALUATION_RGB_DEPTH_OR_LABEL_OUTCOME"
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_e0_1_fresh_evaluation_source_lock"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _select(
    ledger: list[dict[str, Any]], excluded: set[str]
) -> list[dict[str, Any]]:
    eligible = sorted(
        (
            item
            for item in ledger
            if item.get("metadata_healthy")
            and item.get("trajectory") not in excluded
        ),
        key=lambda item: (
            int(item["total_bytes"]),
            str(item["trajectory"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    dates: set[str] = set()
    for item in eligible:
        date = str(item["recording_date"])
        if date in dates:
            continue
        dates.add(date)
        selected.append(item)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError("Cannot select two unique-date evaluation sources")
    return selected


def _canonical(item: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "role": role,
        "trajectory": item["trajectory"],
        "recording_date": item["recording_date"],
        "rows": int(item["rows"]),
        "camera_height_m": float(item["camera_height_m"]),
        "total_bytes": int(item["total_bytes"]),
        "files": {
            kind: {
                "path": item["repo_paths"][kind],
                "size_bytes": int(item["files"][kind]["size_bytes"]),
                "sha256": item["files"][kind]["sha256"],
            }
            for kind in ("pose", "rgb", "depth")
        },
    }


def _validate_parents(
    protocol: dict[str, Any],
    protocol_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    parents = protocol["parent_bindings"]
    specs = (
        (
            protocol_path.parent / parents["e0_protocol_path"],
            parents["e0_protocol_sha256"],
            "E0 protocol",
        ),
        (
            protocol_path.parent / parents["e0_result_path"],
            parents["e0_result_sha256"],
            "E0 result",
        ),
        (
            repo_root / parents["e0_teacher_opportunity_path"],
            parents["e0_teacher_opportunity_sha256"],
            "E0 teacher opportunity",
        ),
        (
            repo_root / parents["e0_acquisition_manifest_path"],
            parents["e0_acquisition_manifest_sha256"],
            "E0 acquisition",
        ),
        (
            repo_root / parents["e0_transport_path"],
            parents["e0_transport_sha256"],
            "E0 transport",
        ),
        (
            repo_root / parents["inventory_path"],
            parents["inventory_sha256"],
            "C0 inventory",
        ),
    )
    for path, expected, label in specs:
        actual = _sha256(path.resolve())
        if actual != expected:
            raise ValueError(
                f"{label} hash mismatch: expected={expected} actual={actual}"
            )
    opportunity = _load_json(
        (repo_root / parents["e0_teacher_opportunity_path"]).resolve()
    )
    if (
        opportunity.get("terminal")
        != "E0_FRESH_TEACHER_MECHANICS_NOT_EVALUABLE"
        or opportunity.get("all_teacher_mechanics_gates_pass")
        or not opportunity.get("all_role_opportunity_gates_pass")
    ):
        raise ValueError("E0 predecessor terminal does not justify E0.1")
    return _load_json((repo_root / parents["inventory_path"]).resolve())


def run(
    protocol_path: Path,
    pretrained_weight_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise ValueError("Stage C E0.1 protocol is not frozen")
    inventory = _validate_parents(protocol, protocol_path, repo_root)
    if inventory.get("metadata_healthy_count") != 95:
        raise ValueError("E0.1 inventory healthy count mismatch")
    selection = protocol["fresh_evaluation_selection"]
    selected = _select(
        inventory["inventory_ledger"],
        set(selection["excluded_consumed_trajectories"]),
    )
    canonical = [
        _canonical(item, role)
        for item, role in zip(selected, ("dev", "heldout"))
    ]
    if canonical != protocol["fresh_evaluation_sources"]:
        raise ValueError("E0.1 evaluation cohort does not recompute exactly")
    if len(set(selection["excluded_consumed_trajectories"])) != 8:
        raise ValueError("E0.1 consumed exclusion set mismatch")
    if (
        set(protocol["reused_training_sources"]["trajectory_ids"])
        & {item["trajectory"] for item in canonical}
    ):
        raise ValueError("E0.1 training/evaluation source overlap")
    weights = protocol["student_contract"]["pretrained_encoder"]
    if pretrained_weight_path.stat().st_size != int(weights["size_bytes"]):
        raise ValueError("E0.1 pretrained weight size mismatch")
    if _sha256(pretrained_weight_path) != weights["sha256"]:
        raise ValueError("E0.1 pretrained weight hash mismatch")
    return {
        "schema": RESULT_SCHEMA,
        "terminal": "E0_1_FRESH_EVALUATION_SOURCE_LOCK_VALIDATED",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "inventory_sha256": protocol["parent_bindings"][
            "inventory_sha256"
        ],
        "reused_training_trajectory_ids": protocol[
            "reused_training_sources"
        ]["trajectory_ids"],
        "fresh_evaluation_sources": canonical,
        "fresh_evaluation_total_bytes": sum(
            item["total_bytes"] for item in canonical
        ),
        "recording_dates_unique": len(
            {item["recording_date"] for item in canonical}
        )
        == 2,
        "pretrained_weight_sha256": _sha256(pretrained_weight_path),
        "fresh_evaluation_rgb_or_depth_read": False,
        "fresh_evaluation_geometry_label_outcome_read": False,
        "student_output_read": False,
        "exact_fresh_evaluation_media_acquisition_authorized": True,
        "teacher_corpus_generation_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _new_artifacts_output(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as error:
        raise ValueError("E0.1 lock output must stay under artifacts.local") from error
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite report: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pretrained-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = (repo_root / args.protocol).resolve()
    weight = (repo_root / args.pretrained_weight).resolve()
    output = _new_artifacts_output(repo_root / args.output, repo_root)
    first = run(protocol, weight, repo_root)
    second = run(protocol, weight, repo_root)
    if json.dumps(first, sort_keys=True) != json.dumps(
        second, sort_keys=True
    ):
        raise ValueError("E0.1 source lock is not deterministic")
    first["determinism_check"] = {
        "second_run_payload_byte_exact": True
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(first, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "terminal": first["terminal"],
                "fresh_evaluation_sources": [
                    {
                        "role": item["role"],
                        "trajectory": item["trajectory"],
                    }
                    for item in first["fresh_evaluation_sources"]
                ],
                "fresh_evaluation_total_bytes": first[
                    "fresh_evaluation_total_bytes"
                ],
                "deterministic": True,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
