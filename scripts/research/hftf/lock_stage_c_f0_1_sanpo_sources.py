#!/usr/bin/env python3
"""Validate and lock the exact HFTF Stage C F0.1 SANPO source set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plan_stage_c_f0_1_sanpo_cross_split_inventory import (
    READY as PLAN_READY,
    SCHEMA as PLAN_SCHEMA,
    _validate_protocol_and_f0_plan,
)
from plan_stage_c_f0_sanpo_inventory import _load_json, _sha256


SCHEMA = "blindassist_hftf_stage_c_f0_1_sanpo_source_lock"
TERMINAL = "F0_1_SANPO_CROSS_SPLIT_SOURCE_LOCK_VALIDATED"


def _receipt_ready(value: dict[str, Any]) -> bool:
    return (
        isinstance(value.get("name"), str)
        and bool(value["name"])
        and isinstance(value.get("generation"), str)
        and bool(value["generation"])
        and isinstance(value.get("size"), int)
        and int(value["size"]) > 0
        and isinstance(value.get("md5_base64"), str)
        and bool(value["md5_base64"])
        and isinstance(value.get("crc32c_base64"), str)
        and bool(value["crc32c_base64"])
    )


def _expected_frames(source_fps: float) -> list[int]:
    if source_fps == 5.0:
        return list(range(25))
    if source_fps == 20.0:
        return list(range(0, 50, 2))
    raise ValueError("F0.1 locked source fps must be 5 or 20")


def lock(
    protocol_path: Path,
    ledger_path: Path,
    f0_plan_path: Path,
    cross_split_plan_path: Path,
) -> dict[str, Any]:
    protocol, _, burned = _validate_protocol_and_f0_plan(
        protocol_path, ledger_path, f0_plan_path
    )
    plan = _load_json(cross_split_plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("terminal") != PLAN_READY
        or plan.get("protocol_sha256") != _sha256(protocol_path)
        or plan.get("parent_f0_plan_sha256") != _sha256(f0_plan_path)
        or plan.get("burn_ledger_sha256") != _sha256(ledger_path)
    ):
        raise ValueError("F0.1 cross-split plan binding mismatch")
    for key in (
        "geometry_outcome_read",
        "teacher_outcome_read",
        "student_outcome_read",
    ):
        if plan.get(key) is not False:
            raise ValueError(f"F0.1 plan outcome firewall opened: {key}")
    if (
        plan.get("source_acquisition_authorized") is not True
        or plan.get("teacher_corpus_authorized") is not False
        or plan.get("student_training_authorized") is not False
    ):
        raise ValueError("F0.1 plan authorization boundary mismatch")
    test = protocol["heldout_selection"]
    test_object = plan["test_split_object"]
    if (
        str(test_object.get("generation"))
        != str(test["split_object_generation"])
        or plan.get("test_split_text_sha256")
        != str(test["split_text_sha256"])
    ):
        raise ValueError("F0.1 test split receipt mismatch")
    sources = plan.get("sources", [])
    roles = [str(item.get("role")) for item in sources]
    expected_roles = ["train"] * 6 + ["dev"] * 3 + ["heldout"] * 3
    ids = [str(item.get("session_id")) for item in sources]
    if (
        len(sources) != 12
        or roles != expected_roles
        or len(ids) != len(set(ids))
        or any(session_id in burned for session_id in ids)
        or plan.get("parent_session_disjoint") is not True
    ):
        raise ValueError("F0.1 exact role/session set mismatch")
    locked_sources: list[dict[str, Any]] = []
    for index, item in enumerate(sources):
        split = str(item.get("official_split"))
        expected_split = "test" if index >= 9 else "train"
        source_fps = float(item["source_fps"])
        target_fps = float(item["target_fps"])
        expected_target = min(10.0, source_fps)
        if (
            split != expected_split
            or int(item["aligned_source_frame_count"]) < 50
            or target_fps != expected_target
            or list(item["selected_source_frames"])
            != _expected_frames(source_fps)
            or not _receipt_ready(item["description_object"])
            or not _receipt_ready(item["camera_poses_object"])
        ):
            raise ValueError(
                f"F0.1 source receipt/timeline mismatch: {ids[index]}"
            )
        locked_sources.append(
            {
                "role": roles[index],
                "official_split": split,
                "session_id": ids[index],
                "source_fps": source_fps,
                "target_fps": target_fps,
                "selected_source_frames": list(
                    item["selected_source_frames"]
                ),
                "description_object": item["description_object"],
                "camera_poses_object": item["camera_poses_object"],
                "camera": item["camera"],
            }
        )
    return {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "workflow_profile": protocol["workflow_profile"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": _sha256(ledger_path),
        "parent_f0_plan_path": str(f0_plan_path.resolve()),
        "parent_f0_plan_sha256": _sha256(f0_plan_path),
        "cross_split_plan_path": str(cross_split_plan_path.resolve()),
        "cross_split_plan_sha256": _sha256(cross_split_plan_path),
        "role_counts": {"train": 6, "dev": 3, "heldout": 3},
        "sources": locked_sources,
        "outcome_firewall": {
            "geometry_outcome_read": False,
            "teacher_outcome_read": False,
            "student_outcome_read": False,
        },
        "exact_media_acquisition_authorized": True,
        "teacher_label_or_corpus_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--burn-ledger", type=Path, required=True)
    parser.add_argument("--f0-plan", type=Path, required=True)
    parser.add_argument("--cross-split-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = lock(
            args.protocol.resolve(),
            args.burn_ledger.resolve(),
            args.f0_plan.resolve(),
            args.cross_split_plan.resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "role_counts": report["role_counts"],
                    "output": str(output),
                }
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
