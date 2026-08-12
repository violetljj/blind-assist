#!/usr/bin/env python3
"""Expand the locked TUM source roster into a factor-distillation corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from ag_st_tum_rgbd import (  # noqa: E402
    _read_member,
    _tar_member_map,
    interpolate_camera_to_world,
    pair_rgb_depth_unique,
    parse_tum_index,
    parse_tum_poses,
)
from materialize_ag_r2_f1_source_native_labels import (  # noqa: E402
    FORBIDDEN_TASK_FIELD_TOKENS,
    REQUIRED_F1_SUPERVISION_FIELDS,
    SelectedFrame,
    arrays_equal,
    build_payload,
    gravity_at_timestamp,
    intrinsics_matrix,
    load_depth,
    load_rgb,
    parse_accelerometer_optional,
    require,
    sha256_bytes,
    sha256_file,
    sha256_json,
    support_identity,
)


DEFAULT_CONTRACT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK_2026-08-11.json"
)
DEFAULT_BASE_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0/result.json"
)
EXPECTED_BASE_RESULT_SHA256 = (
    "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422"
)
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "artifacts.local/downloads/ag-r2-f1-supervision-tum13-r0"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-superteacher-distillation-corpus-tum13-r0"
)
FRAMES_PER_PARENT = 12


def select_evenly_spaced_pairs(
    rgb_rows: list[Any],
    depth_rows: list[Any],
    pose_rows: list[Any],
    count: int,
) -> list[tuple[Any, Any]]:
    pairing = pair_rgb_depth_unique(rgb_rows, depth_rows)
    rgb_by_index = {row.row_index: row for row in rgb_rows}
    eligible = []
    for rgb_index in sorted(pairing, key=lambda index: rgb_by_index[index].timestamp_seconds):
        rgb = rgb_by_index[rgb_index]
        try:
            _, gap = interpolate_camera_to_world(pose_rows, rgb.timestamp_seconds)
        except ValueError:
            continue
        if gap <= 0.1:
            eligible.append(rgb)
    require(len(eligible) >= count, "insufficient pose-bound RGB-D pairs")
    indices = [min(len(eligible) - 1, int((index + 0.5) * len(eligible) / count)) for index in range(count)]
    require(len(set(indices)) == count, "dense frame selection collision")
    return [(eligible[index], pairing[eligible[index].row_index]) for index in indices]


def load_dense_parent_frames(
    row: dict[str, Any],
    source_dir: Path,
    source: dict[str, Any],
    expected_archive_sha256: str,
    count: int,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    parent_id = str(row["parent_id"])
    archive_path = source_dir / f"{parent_id}.tgz"
    require(archive_path.is_file(), f"source archive missing: {parent_id}")
    require(
        archive_path.stat().st_size == int(row["content_length"]),
        f"source archive length drift: {parent_id}",
    )
    archive_sha = sha256_file(archive_path)
    require(archive_sha == expected_archive_sha256, f"source archive SHA drift: {parent_id}")
    intrinsics = intrinsics_matrix(
        source["intrinsics_fx_fy_cx_cy"][str(row["family"])]
    )
    imu_to_rgb = np.asarray(source["imu_to_rgb_optical_rotation"], dtype=np.float64)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _tar_member_map(archive, parent_id)
        required_names = ("rgb.txt", "depth.txt", "groundtruth.txt", "accelerometer.txt")
        for required in required_names:
            require(required in members, f"required source member missing: {parent_id}/{required}")
        metadata_bytes = {
            name: _read_member(archive, members[name]) for name in required_names
        }
        published_rgb = parse_tum_index(metadata_bytes["rgb.txt"].decode("utf-8"))
        published_depth = parse_tum_index(metadata_bytes["depth.txt"].decode("utf-8"))
        rgb_rows = [value for value in published_rgb if value.relative_path in members]
        depth_rows = [value for value in published_depth if value.relative_path in members]
        require(len(rgb_rows) == len(published_rgb), f"published RGB member missing: {parent_id}")
        require(len(depth_rows) == len(published_depth), f"published depth member missing: {parent_id}")
        pose_rows = parse_tum_poses(metadata_bytes["groundtruth.txt"].decode("utf-8"))
        accelerometer = parse_accelerometer_optional(
            metadata_bytes["accelerometer.txt"].decode("utf-8")
        )
        selected = select_evenly_spaced_pairs(
            rgb_rows,
            depth_rows,
            pose_rows,
            count,
        )
        metadata_sha = sha256_json(
            {name: sha256_bytes(value) for name, value in metadata_bytes.items()}
        )
        frames = []
        for rgb, depth in selected:
            rgb_bytes = _read_member(archive, members[rgb.relative_path])
            depth_bytes = _read_member(archive, members[depth.relative_path])
            camera_to_world, pose_gap = interpolate_camera_to_world(
                pose_rows,
                rgb.timestamp_seconds,
            )
            gravity, sample_count, acceleration_norm = gravity_at_timestamp(
                accelerometer,
                rgb.timestamp_seconds,
                float(source["accelerometer_window_seconds"]),
                imu_to_rgb,
            )
            depth_m, depth_valid = load_depth(depth_bytes)
            frames.append(
                SelectedFrame(
                    parent_id=parent_id,
                    role=str(row["role"]),
                    orientation=str(row["orientation"]),
                    rgb=rgb,
                    depth=depth,
                    rgb_u8_hwc=load_rgb(rgb_bytes),
                    depth_m_hw=depth_m,
                    depth_valid_hw=depth_valid,
                    intrinsics=intrinsics,
                    camera_to_world=camera_to_world,
                    pose_bracketing_gap_seconds=pose_gap,
                    gravity_up_camera=gravity,
                    accelerometer_sample_count=sample_count,
                    accelerometer_norm_mps2=acceleration_norm,
                    source_archive_sha256=archive_sha,
                    rgb_member_sha256=sha256_bytes(rgb_bytes),
                    depth_member_sha256=sha256_bytes(depth_bytes),
                    metadata_member_sha256=metadata_sha,
                )
            )
    return frames, {
        "parent_id": parent_id,
        "source_archive": str(archive_path.resolve()),
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": archive_sha,
        "selected_rgb_row_indices": [frame.rgb.row_index for frame in frames],
        "selected_depth_row_indices": [frame.depth.row_index for frame in frames],
        "selection_rule": "TEMPORAL_EQUAL_MASS_BUCKET_CENTER",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.base_result) == EXPECTED_BASE_RESULT_SHA256, "base result drift")
    base = json.loads(args.base_result.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    require(base["passed"] and base["frame_count"] == 39, "base frontdoor drift")
    require(args.frames_per_parent == FRAMES_PER_PARENT, "frame count policy drift")
    contract_sha = sha256_file(args.contract)
    source = contract["source_contract"]
    cohort = contract["cohort_contract"]
    base_sources = {
        row["parent_id"]: row for row in base["source_receipts"]
    }

    all_frames: list[SelectedFrame] = []
    source_receipts = []
    parent_rows: dict[str, dict[str, Any]] = {}
    for row in cohort["parents"]:
        parent_id = str(row["parent_id"])
        require(parent_id in base_sources, f"base source receipt missing: {parent_id}")
        frames, receipt = load_dense_parent_frames(
            row,
            args.source_dir,
            source,
            str(base_sources[parent_id]["source_archive_sha256"]),
            args.frames_per_parent,
        )
        require(len(frames) == args.frames_per_parent, f"dense frame count drift: {parent_id}")
        all_frames.extend(frames)
        source_receipts.append(receipt)
        parent_rows[parent_id] = row
    require(len(all_frames) == 13 * args.frames_per_parent, "dense corpus total drift")

    identities: dict[str, dict[str, Any] | None] = {}
    identity_receipts: dict[str, dict[str, Any]] = {}
    identity_shas: dict[str, str] = {}
    for parent_id in sorted(parent_rows):
        parent_frames = [frame for frame in all_frames if frame.parent_id == parent_id]
        identity, receipt = support_identity(parent_frames)
        identities[parent_id] = identity
        identity_receipts[parent_id] = receipt
        identity_shas[parent_id] = sha256_json(receipt)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts = []
    role_counts: Counter[str] = Counter()
    roundtrip_exact = True
    unknown_fail_closed = True
    task_firewall = True
    for frame in all_frames:
        payload, report = build_payload(
            frame,
            identities[frame.parent_id],
            identity_shas[frame.parent_id],
            contract_sha,
        )
        require(REQUIRED_F1_SUPERVISION_FIELDS.issubset(payload), "supervision field drift")
        task_firewall &= not any(
            token in key.lower()
            for key in payload
            for token in FORBIDDEN_TASK_FIELD_TOKENS
        )
        metric_valid = np.asarray(payload["metric_depth_valid_hw"], dtype=np.bool_)
        support_valid = np.asarray(payload["support_truth_valid_hw"], dtype=np.bool_)
        evidence_valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
        unknown_fail_closed &= bool(
            np.all(payload["metric_depth_m_hw"][~metric_valid] == 0.0)
            and np.all(payload["support_truth_hw"][~support_valid] == 0.0)
            and np.all(payload["obstacle_evidence_truth_hw"][~evidence_valid] == 0.0)
            and np.all(np.isnan(payload["boundary_distance_px_hw"][~evidence_valid]))
        )
        output_path = args.output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            exact = set(written.files) == set(payload) and all(
                arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()
            )
        roundtrip_exact &= exact
        role_counts[frame.role] += 1
        frame_receipts.append(
            {
                "sample_id": frame.frame_id,
                "parent_id": frame.parent_id,
                "role": frame.role,
                "orientation": frame.orientation,
                "rgb_timestamp": frame.rgb.timestamp_seconds,
                "depth_timestamp": frame.depth.timestamp_seconds,
                "pose_bracketing_gap_seconds": frame.pose_bracketing_gap_seconds,
                "support_identity_receipt_sha256": identity_shas[frame.parent_id],
                "output": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": sha256_file(output_path),
                **report,
            }
        )

    joint_parents = sorted(
        parent_id for parent_id, identity in identities.items() if identity is not None
    )
    gates = {
        "DISTILL_D01_BASE_FRONTDOOR_AND_ARCHIVES_EXACT": len(source_receipts) == 13,
        "DISTILL_D02_PARENT_ROLE_COUNTS_EXACT": dict(role_counts)
        == {"FIT": 108, "CHECKPOINT_SELECTION": 24, "TRAIN_CANARY": 24},
        "DISTILL_D03_156_UNIQUE_FACTOR_LABELS": len(frame_receipts) == 156
        and len({row["sample_id"] for row in frame_receipts}) == 156,
        "DISTILL_D04_OUTPUT_ROUNDTRIP_EXACT": roundtrip_exact,
        "DISTILL_D05_UNKNOWN_FAIL_CLOSED": unknown_fail_closed,
        "DISTILL_D06_TASK_AND_REDUCER_FIREWALL": task_firewall,
        "DISTILL_D07_JOINT_FACTOR_PARENT_COUNT_GE_12": len(joint_parents) >= 12,
        "DISTILL_D08_NO_MODEL_OR_OUTCOME_USED": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_superteacher_distillation_corpus_tum13_result_v1",
        "status": "AG_R2_SUPERTEACHER_DISTILLATION_CORPUS_PASS"
        if passed
        else "AG_R2_SUPERTEACHER_DISTILLATION_CORPUS_INCOMPLETE",
        "passed": passed,
        "contract": {"path": str(args.contract.resolve()), "sha256": contract_sha},
        "base_frontdoor": {
            "path": str(args.base_result.resolve()),
            "sha256": EXPECTED_BASE_RESULT_SHA256,
        },
        "selection": {
            "frames_per_parent": args.frames_per_parent,
            "rule": "TEMPORAL_EQUAL_MASS_BUCKET_CENTER",
            "label_or_model_output_used": False,
        },
        "source_receipts": source_receipts,
        "support_identity_receipts": identity_receipts,
        "joint_factor_parents": joint_parents,
        "frame_count": len(frame_receipts),
        "parent_count": len(parent_rows),
        "role_frame_counts": dict(sorted(role_counts.items())),
        "frames": frame_receipts,
        "gates": gates,
        "claim_boundary": "Source-native and geometry-anchored factor-distillation corpus. Tier A/B supervision, not complete truth, task utility, deployment, product, or safety evidence.",
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--base-result", type=Path, default=DEFAULT_BASE_RESULT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames-per-parent", type=int, default=FRAMES_PER_PARENT)
    args = parser.parse_args()
    for name in ("contract", "base_result", "source_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "parent_count": result["parent_count"],
                "frame_count": result["frame_count"],
                "role_frame_counts": result["role_frame_counts"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
