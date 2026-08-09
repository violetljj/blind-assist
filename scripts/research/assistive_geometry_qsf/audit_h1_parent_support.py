#!/usr/bin/env python3
"""Audit TRAIN-only parent support for a new AG-QSF H1 mechanics split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELATIVE = Path(
    "docs/research/assistive-geometry-qsf/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_PARENT_SUPPORT_AUDIT_PROTOCOL_2026-08-09.json"
)
REQUIRED_SUPPORT_KEYS = (
    "event_count",
    "right_censor_count",
    "occupied_known_count",
    "clearance_event_count",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def flatten_manifest(manifest: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    parents: list[str] = []
    frames: list[dict[str, Any]] = []
    for video in manifest.get("videos", []):
        parent = str(video["video_id"])
        parents.append(parent)
        for frame in video.get("frames", []):
            frames.append({**frame, "video_id": parent})
    return parents, frames


def select_parent_frames(
    frames: list[dict[str, Any]],
    parent_ids: list[str],
    frames_per_parent: int,
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_parent[str(frame["video_id"])].append(frame)
    selected: list[dict[str, Any]] = []
    for parent in parent_ids:
        values = by_parent[parent]
        require(len(values) >= frames_per_parent, f"insufficient frames for {parent}")
        indices = np.linspace(0, len(values) - 1, frames_per_parent, dtype=np.int64)
        require(len(set(int(value) for value in indices)) == frames_per_parent, "duplicate frame")
        selected.extend(values[int(index)] for index in indices)
    return selected


def support_counts(frames: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in REQUIRED_SUPPORT_KEYS}
    for frame in frames:
        receipt = frame.get("target", {})
        path = Path(str(receipt.get("path", "")))
        frame_id = f"{frame.get('video_id')}/{frame.get('frame_stem')}"
        require(path.is_file(), f"missing target: {frame_id}")
        require(path.stat().st_size == int(receipt.get("bytes", -1)), f"target bytes drift: {frame_id}")
        require(sha256_file(path) == str(receipt.get("sha256", "")).upper(), f"target SHA drift: {frame_id}")
        with np.load(path, allow_pickle=False) as target:
            clearance = np.asarray(target["clearance_m"])
            clearance_valid = np.asarray(target["clearance_valid"], dtype=bool)
            occupancy = np.asarray(target["occupancy"])
            occupancy_valid = np.asarray(target["occupancy_valid"], dtype=bool)
        require(clearance.shape == (3,) and occupancy.shape == (3, 3), f"target shape drift: {frame_id}")
        event = clearance_valid & (clearance <= 2.0)
        fully_clear = occupancy_valid.all(axis=-1) & (~(occupancy >= 0.5)).all(axis=-1)
        right_censor = (clearance_valid & (clearance > 2.0)) | (~clearance_valid & fully_clear)
        counts["event_count"] += int(event.sum())
        counts["right_censor_count"] += int(right_censor.sum())
        counts["occupied_known_count"] += int((occupancy_valid & (occupancy >= 0.5)).sum())
        counts["clearance_event_count"] += int(event.sum())
    return counts


def choose_eval_parents(
    parent_order: list[str],
    selected_support: dict[str, dict[str, int]],
    count: int,
) -> tuple[list[str], list[str]]:
    eligible = [
        parent
        for parent in parent_order
        if all(selected_support[parent][key] > 0 for key in REQUIRED_SUPPORT_KEYS)
    ]
    return eligible[:count], eligible


def add_counts(values: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(value[key] for value in values) for key in REQUIRED_SUPPORT_KEYS}


def validate_protocol(protocol: dict[str, Any]) -> None:
    require(
        protocol.get("schema") == "blindassist.assistive_geometry_qsf.h1_parent_support_audit_protocol.v1",
        "protocol schema drift",
    )
    require(protocol.get("status") == "TRAIN_SUPPORT_AUDIT_LOCKED_NOT_RUN", "protocol status drift")
    require(protocol.get("model_or_feature_access") is False, "model/feature access leaked")
    require(protocol.get("development_or_confirmation_access") is False, "protected access leaked")
    require(protocol.get("frames_per_parent") == 64, "frame count drift")
    require(protocol.get("eval_parent_count") == 4, "eval parent count drift")
    require(
        protocol.get("eval_selection_rule")
        == "FIRST_FOUR_MANIFEST_ORDER_PARENTS_WITH_NONZERO_SELECTED64_EVENT_CENSOR_OCCUPIED_CLEARANCE_EVENT",
        "eval selection rule drift",
    )
    implementation = protocol.get("implementation", {})
    expected_paths = {
        "scripts/research/assistive_geometry_qsf/audit_h1_parent_support.py",
        "scripts/research/assistive_geometry_qsf/test_audit_h1_parent_support.py",
    }
    require(set(implementation) == expected_paths, "implementation path set drift")
    for logical, expected_sha in implementation.items():
        require(sha256_file(REPO_ROOT / logical) == expected_sha, f"implementation SHA drift: {logical}")
    source = protocol.get("input", {})
    require(source.get("data_role") == "TRAIN", "input role drift")
    require(source.get("outcome_access") == "CONTENT_INSPECTED", "input access drift")
    require(source.get("claim_use") == "TRAIN_SUPPORT_AUDIT_ONLY", "input claim-use drift")
    source_path = REPO_ROOT / str(source.get("path"))
    require(source_path.is_file(), "input manifest missing")
    require(sha256_file(source_path) == source.get("sha256"), "input manifest SHA drift")


def execute(protocol_path: Path, output: Path) -> int:
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    require(output.resolve() == (REPO_ROOT / protocol["output"]).resolve(), "output path drift")
    require(not output.exists(), "output collision")
    manifest = load_json(REPO_ROOT / protocol["input"]["path"])
    parent_order, frames = flatten_manifest(manifest)
    require(parent_order == protocol["parent_order"], "parent order drift")
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_parent[str(frame["video_id"])].append(frame)
    selected = select_parent_frames(frames, parent_order, int(protocol["frames_per_parent"]))
    selected_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in selected:
        selected_by_parent[str(frame["video_id"])].append(frame)

    per_parent: dict[str, dict[str, Any]] = {}
    for parent in parent_order:
        per_parent[parent] = {
            "all_frame_count": len(by_parent[parent]),
            "all_frame_support": support_counts(by_parent[parent]),
            "selected64_support": support_counts(selected_by_parent[parent]),
        }
    selected_support = {parent: value["selected64_support"] for parent, value in per_parent.items()}
    eval_parents, eligible = choose_eval_parents(
        parent_order,
        selected_support,
        int(protocol["eval_parent_count"]),
    )
    fit_parents = [parent for parent in parent_order if parent not in set(eval_parents)]
    fit_support = add_counts([selected_support[parent] for parent in fit_parents])
    eval_support = add_counts([selected_support[parent] for parent in eval_parents])
    qualified = (
        len(eval_parents) == int(protocol["eval_parent_count"])
        and all(value > 0 for value in fit_support.values())
        and all(value > 0 for value in eval_support.values())
    )
    result = {
        "schema": "blindassist.assistive_geometry_qsf.h1_parent_support_audit_result.v1",
        "protocol_sha256": sha256_file(protocol_path),
        "input_manifest_sha256": protocol["input"]["sha256"],
        "model_or_feature_access": False,
        "development_or_confirmation_access": False,
        "parent_order": parent_order,
        "per_parent": per_parent,
        "eligible_eval_parents": eligible,
        "selected_eval_parents": eval_parents,
        "selected_fit_parents": fit_parents,
        "selected64_combined_support": {"fit": fit_support, "eval": eval_support},
        "support_based_roster_selection_disclosed": True,
        "terminal": (
            "H1_PARENT_SUPPORT_AUDIT_PASS_RELOCK_ALLOWED"
            if qualified
            else "H1_PARENT_SUPPORT_AUDIT_NOT_EVALUABLE"
        ),
        "claim_ceiling": (
            "TRAIN-only target-support and roster-mechanics audit; no model, learnability, "
            "Development, Confirmation, device, product, production, or safety authority."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if qualified else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / PROTOCOL_RELATIVE)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts.local/evidence/assistive-geometry-qsf/"
            "h1-parent-support-audit-r0/result.json"
        ),
    )
    args = parser.parse_args()
    return execute(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
