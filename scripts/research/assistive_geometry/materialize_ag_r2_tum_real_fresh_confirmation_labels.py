#!/usr/bin/env python3
"""Materialize checkpoint-unseen real TUM factors for one-shot AG confirmation.

Metric depth and poses are source-native.  TUM fr3 does not carry usable IMU
samples, so support uses a geometry-anchored world-up hypothesis (+Z in the
motion-capture frame) that must be corroborated by persistent horizontal depth
surfaces across all twelve selected frames.  The resulting support/boundary
factors are tier-B Teacher evidence, not complete ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from ag_st_tum_rgbd import (  # noqa: E402
    TumIndexRow,
    TumPoseRow,
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
    intrinsics_matrix,
    load_depth,
    load_rgb,
    sha256_bytes,
    sha256_json,
    support_identity,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    require,
    sha256_file,
)


DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "artifacts.local/datasets/fresh-tf-r1a/rgbd_dataset_freiburg3_sitting_static"
)
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "artifacts.local/datasets/fresh-tf-r1a/rgbd_dataset_freiburg3_sitting_static.tgz"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-real-fresh-confirmation-labels-r0"
)
METRIC_STUDENT_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-metric-depth-student-r0/result.json"
)
FACTOR_STUDENT_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-f1-attempt18-consumed-cross-domain-adaptation-r0/result.json"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "6239D4B7F6EDCAB719B51C76434ACDC238D03DAD666758B51E6046992A7C3914"
)
EXPECTED_METRIC_STUDENT_RESULT_SHA256 = (
    "49549F4D46A70AA56EC55695C1822060A5B8A73709534BB2F045BC2353107DB6"
)
EXPECTED_FACTOR_STUDENT_RESULT_SHA256 = (
    "5898A02FC2B475EB6670BFBA8BF9717C1434DCE5F3756DED9698887229DB4BCA"
)
PARENT_ID = "rgbd_dataset_freiburg3_sitting_static"
ROLE = "CHECKPOINT_UNSEEN_REAL_CONFIRMATION"
ORIENTATION = "LANDSCAPE_IDENTITY"
SELECTION_TOKEN = "AG_R2_TUM_FR3_SITTING_STATIC_REAL_CONFIRMATION_R0"
FRAME_COUNT = 12
INTRINSICS = [535.4, 539.2, 320.1, 247.6]
WORLD_UP_HYPOTHESIS = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
TIER_A_SOURCE_NATIVE = np.uint8(1)
TIER_B_GEOMETRY_ANCHORED_TEACHER = np.uint8(2)


def checkpoint_parent_ids(
    metric_result: dict[str, Any], factor_result: dict[str, Any]
) -> set[str]:
    parents = set(str(value) for value in factor_result["fit_parents"])
    parents.update(str(value) for value in factor_result["internal_validation_parents"])
    for row in metric_result["roles"].values():
        parents.update(str(value) for value in row["parents"])
    return parents


def select_twelve_metadata_frames(
    rgb_rows: list[TumIndexRow],
    depth_rows: list[TumIndexRow],
    pose_rows: list[TumPoseRow],
    source_root: Path,
    *,
    selection_token: str = SELECTION_TOKEN,
) -> list[tuple[TumIndexRow, TumIndexRow]]:
    pairing = pair_rgb_depth_unique(rgb_rows, depth_rows)
    rgb_by_index = {row.row_index: row for row in rgb_rows}
    eligible: list[TumIndexRow] = []
    for rgb_index in sorted(
        pairing, key=lambda index: rgb_by_index[index].timestamp_seconds
    ):
        rgb = rgb_by_index[rgb_index]
        depth = pairing[rgb_index]
        if not (source_root / rgb.relative_path).is_file():
            continue
        if not (source_root / depth.relative_path).is_file():
            continue
        try:
            interpolate_camera_to_world(pose_rows, rgb.timestamp_seconds)
        except ValueError:
            continue
        eligible.append(rgb)
    require(len(eligible) >= FRAME_COUNT, "insufficient pose-bound RGB-D pairs")
    selected: list[tuple[TumIndexRow, TumIndexRow]] = []
    for bucket_index in range(FRAME_COUNT):
        start = bucket_index * len(eligible) // FRAME_COUNT
        end = (bucket_index + 1) * len(eligible) // FRAME_COUNT
        bucket = eligible[start:end]
        require(bool(bucket), f"empty temporal bucket: {bucket_index}")
        rgb = min(
            bucket,
            key=lambda value: digest_text(
                f"{selection_token}:{bucket_index}:{value.relative_path}:"
                f"{pairing[value.row_index].relative_path}"
            ),
        )
        selected.append((rgb, pairing[rgb.row_index]))
    require(
        len({rgb.row_index for rgb, _ in selected}) == FRAME_COUNT,
        "selected frame identity collision",
    )
    return selected


def load_selected_frames(
    source_root: Path,
    archive_sha256: str,
    *,
    parent_id: str = PARENT_ID,
    role: str = ROLE,
    orientation: str = ORIENTATION,
    selection_token: str = SELECTION_TOKEN,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    metadata_names = ("rgb.txt", "depth.txt", "groundtruth.txt", "accelerometer.txt")
    metadata_bytes = {
        name: (source_root / name).read_bytes() for name in metadata_names
    }
    rgb_rows = parse_tum_index(metadata_bytes["rgb.txt"].decode("utf-8"))
    depth_rows = parse_tum_index(metadata_bytes["depth.txt"].decode("utf-8"))
    pose_rows = parse_tum_poses(metadata_bytes["groundtruth.txt"].decode("utf-8"))
    accelerometer_rows = [
        line
        for line in metadata_bytes["accelerometer.txt"].decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(not accelerometer_rows, "fr3 source unexpectedly gained IMU rows")
    selected = select_twelve_metadata_frames(
        rgb_rows,
        depth_rows,
        pose_rows,
        source_root,
        selection_token=selection_token,
    )
    metadata_sha = sha256_json(
        {name: sha256_bytes(value) for name, value in metadata_bytes.items()}
    )
    intrinsics = intrinsics_matrix(INTRINSICS)
    frames: list[SelectedFrame] = []
    for rgb, depth in selected:
        rgb_bytes = (source_root / rgb.relative_path).read_bytes()
        depth_bytes = (source_root / depth.relative_path).read_bytes()
        camera_to_world, pose_gap = interpolate_camera_to_world(
            pose_rows, rgb.timestamp_seconds
        )
        gravity_up_camera = camera_to_world[:3, :3].T @ WORLD_UP_HYPOTHESIS
        gravity_up_camera /= np.linalg.norm(gravity_up_camera)
        depth_m, depth_valid = load_depth(depth_bytes)
        frames.append(
            SelectedFrame(
                parent_id=parent_id,
                role=role,
                orientation=orientation,
                rgb=rgb,
                depth=depth,
                rgb_u8_hwc=load_rgb(rgb_bytes),
                depth_m_hw=depth_m,
                depth_valid_hw=depth_valid,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                pose_bracketing_gap_seconds=pose_gap,
                gravity_up_camera=gravity_up_camera,
                accelerometer_sample_count=0,
                accelerometer_norm_mps2=None,
                source_archive_sha256=archive_sha256,
                rgb_member_sha256=sha256_bytes(rgb_bytes),
                depth_member_sha256=sha256_bytes(depth_bytes),
                metadata_member_sha256=metadata_sha,
            )
        )
    return frames, {
        "selection_token": selection_token,
        "selection_rule": "SHA256 minimum inside twelve equal temporal buckets after RGB-depth-pose metadata admission",
        "selected_rgb_row_indices": [frame.rgb.row_index for frame in frames],
        "selected_depth_row_indices": [frame.depth.row_index for frame in frames],
        "selected_rgb_timestamps": [frame.rgb.timestamp_seconds for frame in frames],
        "accelerometer_rows": 0,
        "world_up_hypothesis": WORLD_UP_HYPOTHESIS.tolist(),
        "world_up_evidence": "TUM_FR3_MOTION_CAPTURE_FRAME_PLUS_MULTI_FRAME_PERSISTENT_HORIZONTAL_DEPTH_CORROBORATION",
    }


def run(
    args: argparse.Namespace,
    *,
    expected_source_archive_sha256: str = EXPECTED_SOURCE_ARCHIVE_SHA256,
    parent_id: str = PARENT_ID,
    role: str = ROLE,
    orientation: str = ORIENTATION,
    selection_token: str = SELECTION_TOKEN,
    dataset_name: str = "TUM RGB-D fr3 sitting_static",
    globally_unopened_claim: bool = False,
    extra_checkpoint_receipts: dict[Path, str] | None = None,
) -> dict[str, Any]:
    require(args.source_root.is_dir(), f"source root missing: {args.source_root}")
    require(args.source_archive.is_file(), f"source archive missing: {args.source_archive}")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    archive_sha = sha256_file(args.source_archive)
    require(archive_sha == expected_source_archive_sha256, "source archive drift")
    require(
        sha256_file(METRIC_STUDENT_RESULT) == EXPECTED_METRIC_STUDENT_RESULT_SHA256,
        "metric student result drift",
    )
    require(
        sha256_file(FACTOR_STUDENT_RESULT) == EXPECTED_FACTOR_STUDENT_RESULT_SHA256,
        "factor student result drift",
    )
    metric_result = json.loads(METRIC_STUDENT_RESULT.read_text(encoding="utf-8"))
    factor_result = json.loads(FACTOR_STUDENT_RESULT.read_text(encoding="utf-8"))
    frozen_parents = checkpoint_parent_ids(metric_result, factor_result)
    require(parent_id not in frozen_parents, "real confirmation parent leaked into checkpoint")
    checked_extra_receipts: dict[str, str] = {}
    for path, expected_sha in (extra_checkpoint_receipts or {}).items():
        require(path.is_file(), f"extra current recipe receipt missing: {path}")
        require(sha256_file(path) == expected_sha, f"extra receipt drift: {path}")
        require(
            parent_id not in path.read_text(encoding="utf-8"),
            f"real confirmation parent leaked into current recipe: {path}",
        )
        checked_extra_receipts[str(path.resolve())] = expected_sha

    frames, selection_receipt = load_selected_frames(
        args.source_root,
        archive_sha,
        parent_id=parent_id,
        role=role,
        orientation=orientation,
        selection_token=selection_token,
    )
    require(len(frames) == FRAME_COUNT, "real confirmation frame count drift")
    identity, identity_receipt = support_identity(frames)
    require(identity is not None, "real confirmation support identity unavailable")
    require(
        identity_receipt["status"] == "SOURCE_SEQUENCE_SUPPORT_IDENTITY_VALID",
        "support identity receipt invalid",
    )
    camera_heights = [
        float(
            np.dot(frame.camera_to_world[:3, 3], WORLD_UP_HYPOTHESIS)
            - float(identity["support_world_height_m"])
        )
        for frame in frames
    ]
    require(
        all(0.45 <= value <= 2.20 for value in camera_heights),
        "geometry-anchored camera height implausible",
    )

    contract = {
        "schema": "blindassist_ag_r2_tum_real_fresh_confirmation_label_contract_v1",
        "parent_id": parent_id,
        "source_archive_sha256": archive_sha,
        "selection": selection_receipt,
        "intrinsics_fx_fy_cx_cy": INTRINSICS,
        "metric_depth_tier": "A_SOURCE_NATIVE",
        "support_boundary_tier": "B_GEOMETRY_ANCHORED_TEACHER",
        "complete_truth_required": False,
        "current_student_or_reducer_output_used": False,
    }
    contract_sha = sha256_json(contract)
    identity_receipt = {
        **identity_receipt,
        "supervision_tier": "B_GEOMETRY_ANCHORED_TEACHER",
        "world_up_hypothesis": WORLD_UP_HYPOTHESIS.tolist(),
        "camera_height_range_m": [min(camera_heights), max(camera_heights)],
    }
    identity_sha = sha256_json(identity_receipt)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts: list[dict[str, Any]] = []
    roundtrip = True
    fail_closed = True
    task_firewall = True
    for frame in frames:
        payload, report = build_payload(
            frame,
            identity,
            identity_sha,
            contract_sha,
        )
        payload.update(
            {
                "metric_depth_supervision_tier_code": np.asarray(
                    TIER_A_SOURCE_NATIVE, dtype=np.uint8
                ),
                "support_supervision_tier_code": np.asarray(
                    TIER_B_GEOMETRY_ANCHORED_TEACHER, dtype=np.uint8
                ),
                "boundary_supervision_tier_code": np.asarray(
                    TIER_B_GEOMETRY_ANCHORED_TEACHER, dtype=np.uint8
                ),
                "gravity_anchor_world_up_xyz": WORLD_UP_HYPOTHESIS.astype(np.float32),
            }
        )
        task_firewall &= REQUIRED_F1_SUPERVISION_FIELDS.issubset(payload)
        task_firewall &= not any(
            token in key.lower()
            for key in payload
            for token in FORBIDDEN_TASK_FIELD_TOKENS
        )
        metric_valid = payload["metric_depth_valid_hw"].astype(np.bool_)
        support_valid = payload["support_truth_valid_hw"].astype(np.bool_)
        evidence_valid = payload["evidence_truth_valid_hw"].astype(np.bool_)
        fail_closed &= bool(np.all(payload["metric_depth_m_hw"][~metric_valid] == 0.0))
        fail_closed &= bool(np.all(payload["support_truth_hw"][~support_valid] == 0.0))
        fail_closed &= bool(
            np.all(payload["obstacle_evidence_truth_hw"][~evidence_valid] == 0.0)
        )
        fail_closed &= bool(
            np.all(np.isnan(payload["boundary_distance_px_hw"][~evidence_valid]))
        )
        output_path = args.output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            roundtrip &= set(written.files) == set(payload)
            roundtrip &= all(
                arrays_equal(np.asarray(written[key]), value)
                for key, value in payload.items()
            )
        frame_receipts.append(
            {
                "sample_id": frame.frame_id,
                "parent_id": frame.parent_id,
                "role": frame.role,
                "orientation": frame.orientation,
                "rgb_timestamp": frame.rgb.timestamp_seconds,
                "depth_timestamp": frame.depth.timestamp_seconds,
                "association_delta_seconds": abs(
                    frame.rgb.timestamp_seconds - frame.depth.timestamp_seconds
                ),
                "pose_bracketing_gap_seconds": frame.pose_bracketing_gap_seconds,
                "source_archive_sha256": frame.source_archive_sha256,
                "rgb_member_sha256": frame.rgb_member_sha256,
                "depth_member_sha256": frame.depth_member_sha256,
                "metadata_member_sha256": frame.metadata_member_sha256,
                "support_identity_receipt_sha256": identity_sha,
                "label_transform_contract_sha256": contract_sha,
                "output": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": sha256_file(output_path),
                "field_count": len(payload),
                **report,
            }
        )

    counts = Counter()
    for row in frame_receipts:
        for key in (
            "metric_depth_valid_pixels",
            "support_valid_pixels",
            "support_positive_pixels_ge_0_5",
            "evidence_valid_pixels",
            "boundary_seed_pixels",
        ):
            counts[key] += int(row[key])
    gates = {
        "TUMREAL_C01_EXACT_SOURCE_AND_CHECKPOINT_EXCLUSION": bool(
            archive_sha == expected_source_archive_sha256
            and parent_id not in frozen_parents
            and len(checked_extra_receipts) == len(extra_checkpoint_receipts or {})
        ),
        "TUMREAL_C02_TWELVE_DETERMINISTIC_UNIQUE_FRAMES": bool(
            len(frame_receipts) == FRAME_COUNT
            and len({row["sample_id"] for row in frame_receipts}) == FRAME_COUNT
        ),
        "TUMREAL_C03_SOURCE_NATIVE_METRIC_DEPTH_ALL_FRAMES": bool(
            all(row["metric_depth_valid_pixels"] > 0 for row in frame_receipts)
        ),
        "TUMREAL_C04_GEOMETRY_ANCHORED_SUPPORT_ALL_FRAMES": bool(
            all(
                row["support_plane_valid"] and row["support_valid_pixels"] > 0
                for row in frame_receipts
            )
        ),
        "TUMREAL_C05_BOUNDARY_EVIDENCE_ALL_FRAMES": bool(
            all(
                row["evidence_valid_pixels"] > 0 and row["boundary_seed_pixels"] > 0
                for row in frame_receipts
            )
        ),
        "TUMREAL_C06_PAYLOAD_ROUNDTRIP_AND_HASHES": bool(
            roundtrip
            and all(
                sha256_file(Path(row["output"])) == row["output_sha256"]
                for row in frame_receipts
            )
        ),
        "TUMREAL_C07_UNKNOWN_FAIL_CLOSED": fail_closed,
        "TUMREAL_C08_NO_TASK_OR_CURRENT_MODEL_OUTPUT_USED": task_firewall,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_tum_real_fresh_confirmation_labels_result_v1",
        "status": "AG_R2_TUM_REAL_FRESH_CONFIRMATION_LABELS_PASS"
        if passed
        else "AG_R2_TUM_REAL_FRESH_CONFIRMATION_LABELS_NOT_EVALUABLE",
        "passed": passed,
        "parent_count": 1,
        "frame_count": len(frame_receipts),
        "source": {
            "dataset": dataset_name,
            "real_world": True,
            "globally_unopened_claim": globally_unopened_claim,
            "checkpoint_unseen_by_current_factor_and_metric_students": True,
            "source_root": str(args.source_root.resolve()),
            "source_archive": str(args.source_archive.resolve()),
            "source_archive_sha256": archive_sha,
        },
        "extra_current_recipe_receipts": checked_extra_receipts,
        "label_contract": contract,
        "label_contract_sha256": contract_sha,
        "support_identity_receipt": identity_receipt,
        "support_identity_receipt_sha256": identity_sha,
        "coverage": dict(counts),
        "gates": gates,
        "frames": frame_receipts,
        "decision": {
            "complete_truth_required": False,
            "current_student_or_reducer_output_opened_during_materialization": False,
            "next_action": "Run the already frozen r10 student and reducer recipe once; never recalibrate on this source outcome.",
        },
        "claim_ceiling": "Checkpoint-unseen real TUM source-native metric depth plus geometry-anchored tier-B support/boundary factors; not complete truth, task utility, deployment, product, or safety proof.",
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name in ("source_root", "source_archive", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "frame_count": result["frame_count"],
                "coverage": result["coverage"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
