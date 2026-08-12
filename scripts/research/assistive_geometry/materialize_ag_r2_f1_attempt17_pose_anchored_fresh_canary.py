#!/usr/bin/env python3
"""Materialize the identity-locked Attempt-17 pose-anchored fresh canary.

The source-native TUM depth is Tier A evidence.  Support and boundary are
Tier B geometry-anchored teacher labels derived from source depth/pose and the
independently evidenced Freiburg world-up convention.  Student outputs are not
loaded by this program.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from materialize_ag_r2_f1_attempt12_pose_anchored_fresh_canary import (  # noqa: E402
    GEOMETRY_ANCHORED_TEACHER_TIER,
    SOURCE_NATIVE_TIER,
    WORLD_UP,
    validate_world_up_anchor,
)
from materialize_ag_r2_f1_source_native_labels import (  # noqa: E402
    FORBIDDEN_TASK_FIELD_TOKENS,
    arrays_equal,
    build_payload,
    load_parent_frames,
    require,
    sha256_file,
    sha256_json,
    support_identity,
)


DEFAULT_SOURCE_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT17_FRESH_CANARY_SOURCE_LOCK_2026-08-12.json"
SOURCE_CONTRACT = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK_2026-08-11.json"
DEFAULT_SOURCE_DIR = REPO_ROOT / "artifacts.local/downloads/ag-r2-f1-attempt17-fresh-canary-r0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt17-pose-anchored-fresh-canary-labels-r0"


def locked_rows(lock: dict[str, Any], source_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for locked in sorted(lock["parents"], key=lambda row: int(row["order"])):
        parent_id = str(locked["parent_id"])
        path = (source_dir / f"{parent_id}.tgz").resolve()
        require(path.is_file(), f"Attempt17 archive missing: {parent_id}")
        require(path.stat().st_size == int(locked["content_length"]), f"Attempt17 archive length drift: {parent_id}")
        rows.append(
            {
                "parent_id": parent_id,
                "family": "freiburg3",
                "role": "FRESH_CANARY",
                "orientation": str(locked["orientation"]),
                "content_length": int(locked["content_length"]),
                "source_sha256": sha256_file(path),
                "resolved_url": str(locked["resolved_url"]),
                "_source_path": path,
            }
        )
    require(len(rows) == 4 and len({row["parent_id"] for row in rows}) == 4, "Attempt17 roster drift")
    return rows


def pose_anchor_frames(frames: list[Any]) -> list[Any]:
    anchored = []
    for frame in frames:
        rotation_camera_to_world = np.asarray(frame.camera_to_world[:3, :3], dtype=np.float64)
        gravity_camera = rotation_camera_to_world.T @ WORLD_UP
        gravity_camera /= np.linalg.norm(gravity_camera)
        anchored.append(replace(frame, gravity_up_camera=gravity_camera))
    return anchored


def run(source_lock_path: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"output exists: {output_dir}")
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    require(
        lock["status"]
        == "ATTEMPT17_FRESH_CANARY_SOURCE_DOWNLOAD_AND_POSE_ANCHORED_LABEL_MATERIALIZATION_AUTHORIZED_MODEL_EVALUATION_FORBIDDEN",
        "Attempt17 source lock invalid",
    )
    require(not lock["execution_authority"]["model_evaluation"], "Attempt17 model evaluation unexpectedly authorized")
    require(not lock["execution_authority"]["factor_tensor_serialization"], "Attempt17 factor serialization unexpectedly authorized")
    source = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))["source_contract"]
    anchor_receipt = validate_world_up_anchor()
    rows = locked_rows(lock, source_dir)

    all_frames = []
    source_receipts = []
    identities: dict[str, Any] = {}
    identity_receipts: dict[str, Any] = {}
    for row in rows:
        source_path = row.pop("_source_path")
        frames, receipt = load_parent_frames(
            row,
            source_path.parent,
            source,
            str(lock["selection_token"]),
        )
        require(len(frames) == 3, "Attempt17 selected frame count drift")
        require(receipt["source_archive_sha256"] == row["source_sha256"], "Attempt17 source receipt drift")
        frames = pose_anchor_frames(frames)
        identity, identity_receipt = support_identity(frames)
        if identity is None:
            print(
                json.dumps(
                    {
                        "loaded_parent": row["parent_id"],
                        "support_identity": identity_receipt,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        identity_receipt["gravity"]["status"] = "POSE_CONDITIONED_CROSS_PARENT_WORLD_UP_VALID"
        identity_receipt["gravity"]["anchor_world_up"] = WORLD_UP.tolist()
        identity_receipt["gravity"]["source_native_anchor_parent_count"] = int(anchor_receipt["source_native_parent_count"])
        identity_receipt["supervision_tier"] = "B_GEOMETRY_ANCHORED_TEACHER"
        all_frames.extend(frames)
        source_receipts.append(receipt)
        identities[row["parent_id"]] = identity
        identity_receipts[row["parent_id"]] = identity_receipt
        print(json.dumps({"loaded_parent": row["parent_id"], "support_identity": identity_receipt["status"]}), flush=True)
    require(len(all_frames) == 12, "Attempt17 total frame count drift")
    unavailable = sorted(parent for parent, identity in identities.items() if identity is None)
    if unavailable:
        output_dir.mkdir(parents=True, exist_ok=False)
        gates = {
            "A17_S00_ALL_SUPPORT_IDENTITIES_MATERIALIZABLE": False,
            "A17_S07_MODEL_METRICS_UNOPENED": True,
            "A17_S08_TASK_AND_REDUCER_FIREWALL": True,
        }
        result = {
            "schema": "blindassist_ag_r2_f1_attempt17_pose_anchored_fresh_canary_labels_result_v1",
            "status": "ATTEMPT17_POSE_ANCHORED_LABELS_FAIL_SOURCE_REPLACEMENT_REQUIRED_NO_MODEL_EVALUATION",
            "passed": False,
            "source_lock": {"path": str(source_lock_path.resolve()), "sha256": sha256_file(source_lock_path)},
            "source_contract": {"path": str(SOURCE_CONTRACT.resolve()), "sha256": sha256_file(SOURCE_CONTRACT)},
            "world_up_anchor_receipt": anchor_receipt,
            "source_receipts": source_receipts,
            "support_identity_receipts": identity_receipts,
            "unavailable_support_identity_parents": unavailable,
            "frame_count": 0,
            "parent_count": len(identities),
            "parent_joint": {parent: False for parent in identities},
            "gates": gates,
            "frames": [],
            "decision": {
                "model_metrics_opened": False,
                "optimizer_use_of_fresh_canary_authorized": False,
                "factor_labels_written": False,
                "next_action_if_fail": "Lock the next unused source identity by the predeclared hash order; do not inspect model outputs.",
            },
        }
        with (output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        return result

    output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts = []
    for frame in all_frames:
        payload, report = build_payload(
            frame,
            identities[frame.parent_id],
            sha256_json(identity_receipts[frame.parent_id]),
            sha256_file(source_lock_path),
        )
        payload["metric_depth_supervision_tier_code"] = np.asarray(SOURCE_NATIVE_TIER, dtype=np.uint8)
        payload["support_supervision_tier_code"] = np.asarray(GEOMETRY_ANCHORED_TEACHER_TIER, dtype=np.uint8)
        payload["boundary_supervision_tier_code"] = np.asarray(GEOMETRY_ANCHORED_TEACHER_TIER, dtype=np.uint8)
        payload["gravity_anchor_world_up_xyz"] = WORLD_UP.astype(np.float32)
        require(
            not any(token in key.lower() for key in payload for token in FORBIDDEN_TASK_FIELD_TOKENS),
            "task field leaked into Attempt17 payload",
        )
        output_path = output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "Attempt17 output field drift")
            require(
                all(arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()),
                "Attempt17 output roundtrip drift",
            )
        frame_receipts.append(
            {
                "sample_id": frame.frame_id,
                "parent_id": frame.parent_id,
                "role": "FRESH_CANARY",
                "orientation": frame.orientation,
                "output": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": sha256_file(output_path),
                "source_archive_sha256": frame.source_archive_sha256,
                "metric_depth_supervision_tier": "A_SOURCE_NATIVE",
                "support_and_boundary_supervision_tier": "B_GEOMETRY_ANCHORED_TEACHER",
                **report,
            }
        )

    parent_joint = {}
    for parent in identities:
        members = [row for row in frame_receipts if row["parent_id"] == parent]
        parent_joint[parent] = (
            sum(int(row["metric_depth_valid_pixels"]) for row in members) > 0
            and any(bool(row["support_plane_valid"]) for row in members)
            and sum(int(row["support_valid_pixels"]) for row in members) > 0
            and sum(int(row["support_positive_pixels_ge_0_5"]) for row in members) > 0
            and sum(int(row["evidence_valid_pixels"]) for row in members) > 0
            and sum(int(row["boundary_seed_pixels"]) for row in members) > 0
        )
    orientations = {row["orientation"] for row in frame_receipts}
    gates = {
        "A17_S01_ARCHIVES_EXACT_AND_HASHED": len(source_receipts) == 4
        and all(len(row["source_archive_sha256"]) == 64 for row in source_receipts),
        "A17_S02_FRAME_PARENT_EXACT": len(frame_receipts) == 12
        and all(sum(row["parent_id"] == parent for row in frame_receipts) == 3 for parent in identities),
        "A17_S03_BOTH_ORIENTATIONS": orientations == {"LANDSCAPE_IDENTITY", "PORTRAIT_ROT90_CLOCKWISE"},
        "A17_S04_WORLD_UP_INDEPENDENTLY_ANCHORED": anchor_receipt["source_native_parent_count"] == 18
        and anchor_receipt["maximum_angle_to_anchor_deg"] <= 10.0,
        "A17_S05_ALL_PARENTS_JOINT_FACTOR": all(parent_joint.values()),
        "A17_S06_TIERED_PROVENANCE_EXPLICIT": all(
            row["support_and_boundary_supervision_tier"] == "B_GEOMETRY_ANCHORED_TEACHER"
            for row in frame_receipts
        ),
        "A17_S07_MODEL_METRICS_UNOPENED": True,
        "A17_S08_TASK_AND_REDUCER_FIREWALL": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt17_pose_anchored_fresh_canary_labels_result_v1",
        "status": "ATTEMPT17_POSE_ANCHORED_LABELS_PASS_EXECUTION_LOCK_REQUIRED"
        if passed
        else "ATTEMPT17_POSE_ANCHORED_LABELS_FAIL_NO_MODEL_EVALUATION",
        "passed": passed,
        "source_lock": {"path": str(source_lock_path.resolve()), "sha256": sha256_file(source_lock_path)},
        "source_contract": {"path": str(SOURCE_CONTRACT.resolve()), "sha256": sha256_file(SOURCE_CONTRACT)},
        "world_up_anchor_receipt": anchor_receipt,
        "source_receipts": source_receipts,
        "support_identity_receipts": identity_receipts,
        "frame_count": len(frame_receipts),
        "parent_count": len(parent_joint),
        "parent_joint": parent_joint,
        "gates": gates,
        "frames": frame_receipts,
        "decision": {
            "model_metrics_opened": False,
            "optimizer_use_of_fresh_canary_authorized": False,
            "truth_claim": "FORBIDDEN_FOR_TIER_B_SUPPORT_AND_BOUNDARY",
            "next_action_if_pass": "Bind exact labels, checkpoint, calibration and implementation in a one-shot Attempt17 execution lock.",
        },
    }
    with (output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.source_lock.resolve(), args.source_dir.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "parent_joint": result["parent_joint"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
