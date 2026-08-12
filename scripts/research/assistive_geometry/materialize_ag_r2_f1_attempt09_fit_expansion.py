#!/usr/bin/env python3
"""Expand the frozen source-native FIT parents to 12 metadata-selected frames each."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from ag_st_tum_rgbd import (  # noqa: E402
    TumIndexRow,
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
    digest_text,
    gravity_at_timestamp,
    intrinsics_matrix,
    load_depth,
    load_rgb,
    parse_accelerometer_optional,
    require,
    sha256_bytes,
    sha256_json,
    support_identity,
)
from validate_ag_r2_f1_supervision_contract import sha256_file  # noqa: E402


CONTRACT = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK_2026-08-11.json"
SOURCE_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0/result.json"
EXPECTED_SOURCE_RESULT_SHA256 = "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422"
SOURCE_DIR = REPO_ROOT / "artifacts.local/downloads/ag-r2-f1-supervision-tum13-r0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-fit-expansion-labels-r0"
SELECTION_TOKEN = "AG_R2_F1_ATTEMPT09_FIT_EXPANSION_2026-08-11"
FRAMES_PER_PARENT = 12


def select_metadata_frames(
    parent_id: str,
    rgb_rows: list[TumIndexRow],
    depth_rows: list[TumIndexRow],
    pose_rows: list[Any],
) -> list[tuple[TumIndexRow, TumIndexRow]]:
    pairing = pair_rgb_depth_unique(rgb_rows, depth_rows)
    rgb_by_index = {row.row_index: row for row in rgb_rows}
    eligible = []
    for rgb_index in sorted(pairing, key=lambda index: rgb_by_index[index].timestamp_seconds):
        rgb = rgb_by_index[rgb_index]
        try:
            interpolate_camera_to_world(pose_rows, rgb.timestamp_seconds)
        except ValueError:
            continue
        eligible.append(rgb)
    require(len(eligible) >= FRAMES_PER_PARENT, f"insufficient eligible pairs: {parent_id}")
    selected = []
    for bucket_index in range(FRAMES_PER_PARENT):
        start = bucket_index * len(eligible) // FRAMES_PER_PARENT
        end = (bucket_index + 1) * len(eligible) // FRAMES_PER_PARENT
        bucket = eligible[start:end]
        require(bool(bucket), f"empty metadata bucket: {parent_id}/{bucket_index}")
        rgb = min(
            bucket,
            key=lambda value: digest_text(
                f"{SELECTION_TOKEN}:{parent_id}:{bucket_index}:"
                f"{value.relative_path}:{pairing[value.row_index].relative_path}"
            ),
        )
        selected.append((rgb, pairing[rgb.row_index]))
    require(len({rgb.row_index for rgb, _ in selected}) == FRAMES_PER_PARENT, "selected RGB identity collision")
    return selected


def load_parent(
    row: dict[str, Any],
    source: dict[str, Any],
    expected_archive_sha: str,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    parent_id = str(row["parent_id"])
    archive_path = SOURCE_DIR / f"{parent_id}.tgz"
    require(archive_path.is_file(), f"archive missing: {parent_id}")
    require(archive_path.stat().st_size == int(row["content_length"]), f"archive length drift: {parent_id}")
    archive_sha = sha256_file(archive_path)
    require(archive_sha == expected_archive_sha, f"archive sha drift: {parent_id}")
    intrinsics = intrinsics_matrix(source["intrinsics_fx_fy_cx_cy"][str(row["family"])])
    imu_to_rgb = np.asarray(source["imu_to_rgb_optical_rotation"], dtype=np.float64)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _tar_member_map(archive, parent_id)
        for required in ("rgb.txt", "depth.txt", "groundtruth.txt", "accelerometer.txt"):
            require(required in members, f"required member missing: {parent_id}/{required}")
        metadata_bytes = {
            name: _read_member(archive, members[name])
            for name in ("rgb.txt", "depth.txt", "groundtruth.txt", "accelerometer.txt")
        }
        rgb_rows = parse_tum_index(metadata_bytes["rgb.txt"].decode("utf-8"))
        depth_rows = parse_tum_index(metadata_bytes["depth.txt"].decode("utf-8"))
        require(all(row.relative_path in members for row in rgb_rows), f"RGB member drift: {parent_id}")
        require(all(row.relative_path in members for row in depth_rows), f"depth member drift: {parent_id}")
        pose_rows = parse_tum_poses(metadata_bytes["groundtruth.txt"].decode("utf-8"))
        accelerometer = parse_accelerometer_optional(metadata_bytes["accelerometer.txt"].decode("utf-8"))
        selected = select_metadata_frames(parent_id, rgb_rows, depth_rows, pose_rows)
        metadata_sha = sha256_json({name: sha256_bytes(value) for name, value in metadata_bytes.items()})
        frames = []
        for rgb, depth in selected:
            rgb_bytes = _read_member(archive, members[rgb.relative_path])
            depth_bytes = _read_member(archive, members[depth.relative_path])
            camera_to_world, pose_gap = interpolate_camera_to_world(pose_rows, rgb.timestamp_seconds)
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
                    role="FIT",
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
    }


def run(output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(sha256_file(SOURCE_RESULT) == EXPECTED_SOURCE_RESULT_SHA256, "source result drift")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    source = contract["source_contract"]
    fit_rows = [row for row in contract["cohort_contract"]["parents"] if row["role"] == "FIT"]
    require(len(fit_rows) == 9, "frozen FIT parent count drift")
    expected_shas = {
        row["parent_id"]: row["source_archive_sha256"] for row in source_result["source_receipts"]
    }
    all_frames = []
    source_receipts = []
    for row in fit_rows:
        frames, receipt = load_parent(row, source, expected_shas[str(row["parent_id"])])
        require(len(frames) == FRAMES_PER_PARENT, "expanded frame count drift")
        all_frames.extend(frames)
        source_receipts.append(receipt)
        print(json.dumps({"loaded_parent": row["parent_id"], "frames": len(frames)}), flush=True)
    require(len(all_frames) == 9 * FRAMES_PER_PARENT, "expanded total frame count drift")

    identities = {}
    identity_receipts = {}
    identity_shas = {}
    for parent_id in sorted({frame.parent_id for frame in all_frames}):
        identity, receipt = support_identity([frame for frame in all_frames if frame.parent_id == parent_id])
        identities[parent_id] = identity
        identity_receipts[parent_id] = receipt
        identity_shas[parent_id] = sha256_json(receipt)

    output_dir.mkdir(parents=True, exist_ok=False)
    contract_sha = sha256_file(CONTRACT)
    frame_receipts = []
    coverage = defaultdict(Counter)
    checks = {
        "schema_complete": True,
        "unknown_fail_closed": True,
        "task_firewall": True,
        "roundtrip_exact": True,
    }
    for index, frame in enumerate(all_frames):
        payload, report = build_payload(frame, identities[frame.parent_id], identity_shas[frame.parent_id], contract_sha)
        checks["schema_complete"] &= REQUIRED_F1_SUPERVISION_FIELDS.issubset(payload)
        checks["task_firewall"] &= not any(
            token in key.lower() for key in payload for token in FORBIDDEN_TASK_FIELD_TOKENS
        )
        metric_valid = payload["metric_depth_valid_hw"].astype(np.bool_)
        support_valid = payload["support_truth_valid_hw"].astype(np.bool_)
        evidence_valid = payload["evidence_truth_valid_hw"].astype(np.bool_)
        checks["unknown_fail_closed"] &= bool(
            np.all(payload["metric_depth_m_hw"][~metric_valid] == 0.0)
            and np.all(payload["support_truth_hw"][~support_valid] == 0.0)
            and np.all(payload["obstacle_evidence_truth_hw"][~evidence_valid] == 0.0)
            and np.all(np.isnan(payload["boundary_distance_px_hw"][~evidence_valid]))
        )
        output_path = output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            exact = set(written.files) == set(payload) and all(
                arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()
            )
        checks["roundtrip_exact"] &= exact
        receipt = {
            "sample_id": frame.frame_id,
            "parent_id": frame.parent_id,
            "role": "FIT_EXPANDED",
            "orientation": frame.orientation,
            "rgb_timestamp": frame.rgb.timestamp_seconds,
            "depth_timestamp": frame.depth.timestamp_seconds,
            "association_delta_seconds": abs(frame.rgb.timestamp_seconds - frame.depth.timestamp_seconds),
            "pose_bracketing_gap_seconds": frame.pose_bracketing_gap_seconds,
            "accelerometer_sample_count": frame.accelerometer_sample_count,
            "accelerometer_norm_mps2": frame.accelerometer_norm_mps2,
            "source_archive_sha256": frame.source_archive_sha256,
            "rgb_member_sha256": frame.rgb_member_sha256,
            "depth_member_sha256": frame.depth_member_sha256,
            "metadata_member_sha256": frame.metadata_member_sha256,
            "support_identity_receipt_sha256": identity_shas[frame.parent_id],
            "label_transform_contract_sha256": contract_sha,
            "output": str(output_path.resolve()),
            "output_bytes": output_path.stat().st_size,
            "output_sha256": sha256_file(output_path),
            "field_count": len(payload),
            **report,
        }
        frame_receipts.append(receipt)
        for key in ("metric_depth_valid_pixels", "support_valid_pixels", "support_positive_pixels_ge_0_5", "evidence_valid_pixels", "boundary_seed_pixels"):
            coverage[frame.parent_id][key] += int(report[key])
        if (index + 1) % 12 == 0:
            print(json.dumps({"materialized_frames": index + 1, "total": len(all_frames)}), flush=True)
    joint = sorted(
        parent for parent, counts in coverage.items()
        if all(counts[key] > 0 for key in ("metric_depth_valid_pixels", "support_valid_pixels", "support_positive_pixels_ge_0_5", "evidence_valid_pixels", "boundary_seed_pixels"))
    )
    gates = {
        "fit_parent_count_9": len(source_receipts) == 9,
        "frames_12_each_108_total": len(frame_receipts) == 108 and all(sum(row["parent_id"] == parent for row in frame_receipts) == 12 for parent in coverage),
        "joint_factor_coverage_9": len(joint) == 9,
        **checks,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_f1_attempt09_fit_expansion_labels_result_v1",
        "status": "ATTEMPT09_FIT_EXPANSION_LABELS_PASS_AG_RETRAIN_AUTHORIZED" if passed else "ATTEMPT09_FIT_EXPANSION_LABELS_FAIL_NO_RETRAIN",
        "passed": passed,
        "selection_token": SELECTION_TOKEN,
        "selection_rule": "metadata-only 12 contiguous time buckets; SHA256 minimum per bucket",
        "preserved_canary_metrics_opened": False,
        "contract": {"path": str(CONTRACT.resolve()), "sha256": contract_sha},
        "source_result": {"path": str(SOURCE_RESULT.resolve()), "sha256": EXPECTED_SOURCE_RESULT_SHA256},
        "source_receipts": source_receipts,
        "support_identity_receipts": identity_receipts,
        "frame_count": len(frame_receipts),
        "parent_count": len(source_receipts),
        "joint_factor_parents": joint,
        "coverage_by_parent": {parent: dict(values) for parent, values in coverage.items()},
        "gates": gates,
        "frames": frame_receipts,
        "claim_boundary": "Additional source-native FIT supervision only; held parents remain unopened and unchanged.",
    }
    with (output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "passed": result["passed"], "parent_count": result["parent_count"], "frame_count": result["frame_count"], "gates": result["gates"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
