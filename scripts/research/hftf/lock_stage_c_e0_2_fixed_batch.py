#!/usr/bin/env python3
"""Validate the fixed HFTF Stage C E0.2 six-source batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lock_stage_c_e0_1_fresh_evaluation_sources as e01_lock  # noqa: E402


SCHEMA = (
    "blindassist_hftf_stage_c_multi_source_evaluation_qualification_e0_2"
)
STATUS = "FROZEN_BEFORE_FIXED_BATCH_RGB_DEPTH_OR_LABEL_OUTCOME"
RESULT_SCHEMA = "blindassist_hftf_stage_c_e0_2_fixed_batch_source_lock"


def _select(
    ledger: list[dict[str, Any]],
    excluded_trajectories: set[str],
    excluded_dates: set[str],
) -> list[dict[str, Any]]:
    eligible = sorted(
        (
            item
            for item in ledger
            if item.get("metadata_healthy")
            and item.get("trajectory") not in excluded_trajectories
            and item.get("recording_date") not in excluded_dates
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
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError("Cannot form fixed six-source unique-date batch")
    return selected


def run(
    protocol_path: Path, repo_root: Path
) -> dict[str, Any]:
    protocol = e01_lock._load_json(protocol_path)
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise ValueError("Stage C E0.2 protocol is not frozen")
    parents = protocol["parent_bindings"]
    specs = (
        (
            protocol_path.parent / parents["e0_1_protocol_path"],
            parents["e0_1_protocol_sha256"],
            "E0.1 protocol",
        ),
        (
            protocol_path.parent / parents["e0_1_result_path"],
            parents["e0_1_result_sha256"],
            "E0.1 result",
        ),
        (
            repo_root / parents["e0_1_teacher_opportunity_path"],
            parents["e0_1_teacher_opportunity_sha256"],
            "E0.1 teacher opportunity",
        ),
        (
            repo_root / parents["inventory_path"],
            parents["inventory_sha256"],
            "C0 inventory",
        ),
    )
    for path, expected, label in specs:
        actual = e01_lock._sha256(path.resolve())
        if actual != expected:
            raise ValueError(
                f"{label} hash mismatch: expected={expected} actual={actual}"
            )
    predecessor = e01_lock._load_json(
        repo_root / parents["e0_1_teacher_opportunity_path"]
    )
    if (
        predecessor.get("terminal")
        != "E0_1_FOOT_GROUND_STUDENT_CANARY_NOT_EVALUABLE"
        or predecessor.get("student_training_authorized")
    ):
        raise ValueError("E0.1 predecessor does not justify E0.2")
    inventory = e01_lock._load_json(
        repo_root / parents["inventory_path"]
    )
    selection = protocol["fixed_batch_selection"]
    selected = _select(
        inventory["inventory_ledger"],
        set(selection["excluded_consumed_trajectories"]),
        set(selection["excluded_consumed_recording_dates"]),
    )
    roles = ("dev", "heldout", "dev", "heldout", "dev", "heldout")
    canonical = [
        e01_lock._canonical(item, role)
        for item, role in zip(selected, roles)
    ]
    if canonical != protocol["frozen_evaluation_sources"]:
        raise ValueError("E0.2 fixed batch does not recompute exactly")
    if len({item["recording_date"] for item in canonical}) != 6:
        raise ValueError("E0.2 fixed batch dates are not unique")
    if {
        item["recording_date"] for item in canonical
    } & set(selection["excluded_consumed_recording_dates"]):
        raise ValueError("E0.2 fixed batch date overlaps consumed date")
    return {
        "schema": RESULT_SCHEMA,
        "terminal": "E0_2_FIXED_BATCH_SOURCE_LOCK_VALIDATED",
        "protocol_path": str(protocol_path),
        "protocol_sha256": e01_lock._sha256(protocol_path),
        "inventory_sha256": parents["inventory_sha256"],
        "frozen_evaluation_sources": canonical,
        "fixed_batch_total_bytes": sum(
            item["total_bytes"] for item in canonical
        ),
        "roles": {"dev": 3, "heldout": 3},
        "recording_dates_unique_and_disjoint_from_consumed": True,
        "rgb_or_depth_read": False,
        "geometry_label_outcome_read": False,
        "student_output_read": False,
        "exact_fixed_batch_acquisition_authorized": True,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _new_output(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as error:
        raise ValueError("E0.2 source-lock output must stay under artifacts.local") from error
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite report: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = (repo_root / args.protocol).resolve()
    output = _new_output(repo_root / args.output, repo_root)
    first = run(protocol, repo_root)
    second = run(protocol, repo_root)
    if json.dumps(first, sort_keys=True) != json.dumps(
        second, sort_keys=True
    ):
        raise ValueError("E0.2 source-lock result is not deterministic")
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
                "fixed_batch_total_bytes": first[
                    "fixed_batch_total_bytes"
                ],
                "sources": [
                    {
                        "role": item["role"],
                        "trajectory": item["trajectory"],
                    }
                    for item in first["frozen_evaluation_sources"]
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
