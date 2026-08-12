#!/usr/bin/env python3
"""Upgrade the frozen Attempt-12 cohort with a pose-conditioned gravity anchor.

Metric depth remains source-native.  Support and boundary factors are explicitly
Tier-B geometry-anchored labels: the TUM pose rotates a cross-parent +Z world-up
anchor into each camera frame.  No student/model output participates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from materialize_ag_r2_f1_attempt12_fresh_canary import locked_rows  # noqa: E402
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


DEFAULT_SOURCE_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT12_FRESH_CANARY_SOURCE_LOCK_2026-08-11.json"
DEFAULT_ANCHOR_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT12_POSE_ANCHORED_LABEL_LOCK_2026-08-11.json"
SOURCE_CONTRACT = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK_2026-08-11.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt12-pose-anchored-fresh-canary-labels-r1"

ANCHOR_EVIDENCE = (
    (
        REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0/result.json",
        "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422",
    ),
    (
        REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-fresh-ag-held-labels-r0/result.json",
        "4DBF0E85F45357C613221DF9F2C5A5E3B0971C314EB29D1967C02E0D6FAEB7CC",
    ),
    (
        REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt06-fresh-selection-labels-r0/result.json",
        "F67A9A000A4A82C180B9E875DEE976C80E00584AB8A6CEBCC603DFAAEE1E90A5",
    ),
)
ATTEMPT12_SOURCE_ONLY_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt12-fresh-canary-labels-r0/result.json",
    "FFB1E2CA36CA6918E9C197B453CE1B7860566B2C50CB7B24BFA024EEB5795B93",
)
WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
SOURCE_NATIVE_TIER = 1
GEOMETRY_ANCHORED_TEACHER_TIER = 2


def validate_world_up_anchor() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    receipts = []
    for path, expected_sha in ANCHOR_EVIDENCE:
        require(path.is_file() and sha256_file(path) == expected_sha, f"anchor evidence drift: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        receipts.append({"path": str(path.resolve()), "sha256": expected_sha})
        for parent_id, identity in result["support_identity_receipts"].items():
            gravity = identity["gravity"]
            if gravity["status"] != "SOURCE_GRAVITY_VALID":
                continue
            value = np.asarray(gravity["world_up_unit"], dtype=np.float64)
            value /= np.linalg.norm(value)
            angle = math.degrees(math.acos(float(np.clip(value @ WORLD_UP, -1.0, 1.0))))
            rows.append({"parent_id": parent_id, "angle_to_plus_z_deg": angle})
    require(len(rows) == 18, "source-native world-up parent count drift")
    require(len({row["parent_id"] for row in rows}) == len(rows), "world-up parent overlap")
    maximum = max(float(row["angle_to_plus_z_deg"]) for row in rows)
    require(maximum <= 10.0, "source-native evidence does not support +Z world-up anchor")
    return {
        "anchor": WORLD_UP.tolist(),
        "source_native_parent_count": len(rows),
        "maximum_angle_to_anchor_deg": maximum,
        "parent_angles": sorted(rows, key=lambda row: row["parent_id"]),
        "evidence": receipts,
        "provenance": "CROSS_PARENT_SOURCE_NATIVE_GRAVITY_PLUS_SOURCE_POSE",
    }


def pose_anchor_frames(frames: list[Any]) -> list[Any]:
    anchored = []
    for frame in frames:
        rotation_camera_to_world = np.asarray(frame.camera_to_world[:3, :3], dtype=np.float64)
        gravity_camera = rotation_camera_to_world.T @ WORLD_UP
        gravity_camera /= np.linalg.norm(gravity_camera)
        anchored.append(replace(frame, gravity_up_camera=gravity_camera))
    return anchored


def run(anchor_lock_path: Path, source_lock_path: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"output exists: {output_dir}")
    anchor_lock = json.loads(anchor_lock_path.read_text(encoding="utf-8"))
    require(anchor_lock["status"] == "ATTEMPT12_POSE_ANCHORED_LABEL_MATERIALIZATION_AUTHORIZED_MODEL_EVALUATION_FORBIDDEN", "anchor lock invalid")
    source_only_path, source_only_sha = ATTEMPT12_SOURCE_ONLY_RESULT
    require(sha256_file(source_only_path) == source_only_sha, "Attempt12 source-only result drift")
    source_only = json.loads(source_only_path.read_text(encoding="utf-8"))
    require(not source_only["decision"]["attempt12_model_metrics_opened"], "Attempt12 model outcome already opened")
    require(not source_only["decision"]["attempt12_optimizer_use_authorized"], "Attempt12 labels already authorized for optimization")

    anchor_receipt = validate_world_up_anchor()
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))["source_contract"]
    rows = locked_rows(source_lock)
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
            str(source_lock["selection_token"]),
        )
        require(len(frames) == 3, "Attempt12 frame identity drift")
        frames = pose_anchor_frames(frames)
        identity, identity_receipt = support_identity(frames)
        identity_receipt["gravity"]["status"] = "POSE_CONDITIONED_CROSS_PARENT_WORLD_UP_VALID"
        identity_receipt["gravity"]["anchor_world_up"] = WORLD_UP.tolist()
        identity_receipt["gravity"]["source_native_anchor_parent_count"] = int(anchor_receipt["source_native_parent_count"])
        identity_receipt["supervision_tier"] = "B_GEOMETRY_ANCHORED_TEACHER"
        all_frames.extend(frames)
        source_receipts.append(receipt)
        identities[row["parent_id"]] = identity
        identity_receipts[row["parent_id"]] = identity_receipt
        print(json.dumps({"loaded_parent": row["parent_id"], "support_identity": identity_receipt["status"]}), flush=True)
    require(len(all_frames) == 12, "Attempt12 frame count drift")

    output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts = []
    for frame in all_frames:
        payload, report = build_payload(
            frame,
            identities[frame.parent_id],
            sha256_json(identity_receipts[frame.parent_id]),
            sha256_file(anchor_lock_path),
        )
        payload["metric_depth_supervision_tier_code"] = np.asarray(SOURCE_NATIVE_TIER, dtype=np.uint8)
        payload["support_supervision_tier_code"] = np.asarray(GEOMETRY_ANCHORED_TEACHER_TIER, dtype=np.uint8)
        payload["boundary_supervision_tier_code"] = np.asarray(GEOMETRY_ANCHORED_TEACHER_TIER, dtype=np.uint8)
        payload["gravity_anchor_world_up_xyz"] = WORLD_UP.astype(np.float32)
        require(not any(token in key.lower() for key in payload for token in FORBIDDEN_TASK_FIELD_TOKENS), "task field leaked")
        output_path = output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "output field drift")
            require(all(arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()), "output roundtrip drift")
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
    gates = {
        "A12B_S01_SOURCE_AND_FRAME_IDENTITIES_UNCHANGED": len(frame_receipts) == 12 and len(parent_joint) == 4,
        "A12B_S02_WORLD_UP_INDEPENDENTLY_ANCHORED": anchor_receipt["source_native_parent_count"] == 18 and anchor_receipt["maximum_angle_to_anchor_deg"] <= 10.0,
        "A12B_S03_ALL_PARENTS_JOINT_FACTOR": all(parent_joint.values()),
        "A12B_S04_TIERED_PROVENANCE_EXPLICIT": all(row["support_and_boundary_supervision_tier"] == "B_GEOMETRY_ANCHORED_TEACHER" for row in frame_receipts),
        "A12B_S05_MODEL_METRICS_UNOPENED": True,
        "A12B_S06_TASK_FIREWALL": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt12_pose_anchored_fresh_canary_labels_result_v1",
        "status": "ATTEMPT12_POSE_ANCHORED_LABELS_PASS_COMPLETE_FACTOR_RETRAIN_REQUIRED" if passed else "ATTEMPT12_POSE_ANCHORED_LABELS_FAIL_NO_MODEL_EVALUATION",
        "passed": passed,
        "source_lock": {"path": str(source_lock_path.resolve()), "sha256": sha256_file(source_lock_path)},
        "anchor_lock": {"path": str(anchor_lock_path.resolve()), "sha256": sha256_file(anchor_lock_path)},
        "source_only_result": {"path": str(source_only_path.resolve()), "sha256": source_only_sha},
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
            "next_action_if_pass": "Retrain all factor heads on consumed parents, freeze exact checkpoint/calibration, then evaluate this fresh cohort exactly once.",
        },
    }
    with (output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-lock", type=Path, default=DEFAULT_ANCHOR_LOCK)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.anchor_lock.resolve(), args.source_lock.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "passed": result["passed"], "parent_joint": result["parent_joint"], "gates": result["gates"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
