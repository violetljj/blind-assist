#!/usr/bin/env python3
"""Materialize the frozen-route walking_xyz labels for the final AG seam.

The TUM sequence was consumed by unrelated historical experiments, so this
script deliberately does not call it globally fresh.  It is absent from every
checkpoint, factor fit, uncertainty bank, and consumed V2 seam that defines the
current frozen recipe.  Frame selection is metadata-only and happens before
any current model or reducer output is opened.

Metric depth and poses are source-native (tier A).  Support and boundary are
deterministic geometry-anchored teacher labels (tier B), using the TUM fr3
motion-capture +Z world-up hypothesis corroborated by persistent horizontal
depth geometry.  Unsupported pixels remain UNKNOWN/fail-closed.
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

from ag_st_tum_rgbd import parse_tum_index, parse_tum_poses  # noqa: E402
from materialize_ag_r2_f1_source_native_labels import (  # noqa: E402
    FORBIDDEN_TASK_FIELD_TOKENS,
    REQUIRED_F1_SUPERVISION_FIELDS,
    arrays_equal,
    build_payload,
    sha256_bytes,
    sha256_json,
    support_identity,
)
from materialize_ag_r2_tum_real_fresh_confirmation_labels import (  # noqa: E402
    INTRINSICS,
    TIER_A_SOURCE_NATIVE,
    TIER_B_GEOMETRY_ANCHORED_TEACHER,
    WORLD_UP_HYPOTHESIS,
    load_selected_frames,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    require,
    sha256_file,
)


PARENT_ID = "rgbd_dataset_freiburg3_walking_xyz"
ROLE = "CHECKPOINT_UNSEEN_ROUTE_FINAL_REAL_CONFIRMATION"
ORIENTATION = "LANDSCAPE_IDENTITY"
SELECTION_TOKEN = "AG_R2_TUM_FR3_WALKING_XYZ_FINAL_CONFIRMATION_R0"
FRAME_COUNT = 12

DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "artifacts.local/datasets/model-variant-gate-r0"
    / PARENT_ID
)
DEFAULT_SOURCE_PROTOCOL = (
    REPO_ROOT
    / "docs/research/hftf"
    / "DAV2_MODEL_VARIANT_VALIDATION_R0_PROTOCOL_2026-08-05.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-tum-walking-xyz-final-confirmation-labels-r0"
)

EXPECTED_SOURCE_PROTOCOL_SHA256 = (
    "7AD758A829DB3AA07EC295F31090DDF8A8B1E7E6D6943B86C2A25FE063EBC664"
)
EXPECTED_INHERITED_ARCHIVE_SHA256 = (
    "1459E9488AC0E61A2EC80DFBC35CFB77942F6D8EABDED1C8D26A70BE650D0E1D"
)
EXPECTED_METADATA_SHA256 = {
    "accelerometer.txt": "D4E7B19B9B619A1ED41253BB0BFCE69CEB51ACFBAC164119379F1B4D7164C627",
    "depth.txt": "01C02C0DA8CCCECBC66D36B91C549BC2A30284BF41A4706EB90477D8CCBA37CF",
    "groundtruth.txt": "4117699F2B027A6E305F84D3B17F1928AA5361DB300D5DF10639824540D92B6D",
    "rgb.txt": "5866AB98D620D1163C8C8F01D90BC0A0B3BC7DCD223BCA8BC5F238F1734C35C0",
}
EXPECTED_METADATA_ROW_COUNTS = {
    "accelerometer.txt": 0,
    "depth.txt": 833,
    "groundtruth.txt": 2884,
    "rgb.txt": 859,
}
EXPECTED_PAYLOAD_COUNTS = {"depth": 833, "rgb": 859}

FACTOR_STUDENT_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-f1-attempt18-consumed-cross-domain-adaptation-r0/result.json"
)
METRIC_STUDENT_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-multisource-metric-depth-student-r0/result.json"
)
SCALE_BANK_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-metric-scale-residual-bank-r2/result.json"
)
FROZEN_V2_CONSUMED_SEAM_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-multisource-v2-consumed-seam-r0/result.json"
)
EXACT_CURRENT_RECIPE_RECEIPTS = {
    FACTOR_STUDENT_RESULT: "5898A02FC2B475EB6670BFBA8BF9717C1434DCE5F3756DED9698887229DB4BCA",
    METRIC_STUDENT_RESULT: "F0703357B0F25C7ABF209EE53DE9B04E588BEDE3629C1B1F5273D9E31D41BFF3",
    SCALE_BANK_RESULT: "68188B00FB4951771443B4706526F6BAAED1AA85E77C4AE4D73958600AB4C0E5",
    FROZEN_V2_CONSUMED_SEAM_RESULT: "106E64706632D19BF327513DE927A8BA68277F800F66DA3379EB2298EE724528",
}


def non_comment_row_count(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def source_receipt(source_root: Path, source_protocol: Path) -> dict[str, Any]:
    require(source_root.is_dir(), f"source root missing: {source_root}")
    require(source_protocol.is_file(), f"source protocol missing: {source_protocol}")
    require(
        sha256_file(source_protocol) == EXPECTED_SOURCE_PROTOCOL_SHA256,
        "source protocol drift",
    )
    protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
    archive_receipt = protocol["cohort"]["source_archives"][f"{PARENT_ID}.tgz"]
    require(
        archive_receipt == EXPECTED_INHERITED_ARCHIVE_SHA256,
        "inherited archive receipt drift",
    )

    metadata_hashes: dict[str, str] = {}
    metadata_rows: dict[str, int] = {}
    for name in sorted(EXPECTED_METADATA_SHA256):
        path = source_root / name
        require(path.is_file(), f"source metadata missing: {path}")
        raw = path.read_bytes()
        metadata_hashes[name] = sha256_bytes(raw)
        metadata_rows[name] = non_comment_row_count(raw.decode("utf-8"))
    require(metadata_hashes == EXPECTED_METADATA_SHA256, "source metadata drift")
    require(metadata_rows == EXPECTED_METADATA_ROW_COUNTS, "metadata row count drift")

    payload_counts = {
        name: sum(1 for path in (source_root / name).iterdir() if path.is_file())
        for name in sorted(EXPECTED_PAYLOAD_COUNTS)
    }
    require(payload_counts == EXPECTED_PAYLOAD_COUNTS, "source payload count drift")

    rgb_rows = parse_tum_index((source_root / "rgb.txt").read_text(encoding="utf-8"))
    depth_rows = parse_tum_index(
        (source_root / "depth.txt").read_text(encoding="utf-8")
    )
    pose_rows = parse_tum_poses(
        (source_root / "groundtruth.txt").read_text(encoding="utf-8")
    )
    require(len(rgb_rows) == 859 and len(depth_rows) == 833, "parsed index drift")
    require(len(pose_rows) == 2884, "parsed pose count drift")

    return {
        "schema": "blindassist_ag_r2_tum_extracted_source_receipt_v1",
        "parent_id": PARENT_ID,
        "source_root": str(source_root.resolve()),
        "source_protocol": str(source_protocol.resolve()),
        "source_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "inherited_archive_sha256": EXPECTED_INHERITED_ARCHIVE_SHA256,
        "archive_live_rehash_available": False,
        "metadata_sha256": metadata_hashes,
        "metadata_row_counts": metadata_rows,
        "payload_file_counts": payload_counts,
        "selected_payload_members_bound_separately": True,
    }


def verify_checkpoint_exclusion() -> dict[str, str]:
    receipts: dict[str, str] = {}
    for path, expected_sha in EXACT_CURRENT_RECIPE_RECEIPTS.items():
        require(path.is_file(), f"current recipe receipt missing: {path}")
        actual_sha = sha256_file(path)
        require(actual_sha == expected_sha, f"current recipe receipt drift: {path}")
        text = path.read_text(encoding="utf-8")
        require(PARENT_ID not in text, f"parent leaked into current recipe: {path}")
        receipts[str(path.resolve())] = actual_sha
    return receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    source = source_receipt(args.source_root, args.source_protocol)
    recipe_receipts = verify_checkpoint_exclusion()

    frames, selection = load_selected_frames(
        args.source_root,
        EXPECTED_INHERITED_ARCHIVE_SHA256,
        parent_id=PARENT_ID,
        role=ROLE,
        orientation=ORIENTATION,
        selection_token=SELECTION_TOKEN,
    )
    require(len(frames) == FRAME_COUNT, "final confirmation frame count drift")
    identity, identity_receipt = support_identity(frames)
    require(identity is not None, "final confirmation support identity unavailable")
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
        "schema": "blindassist_ag_r2_tum_walking_xyz_final_label_contract_v1",
        "parent_id": PARENT_ID,
        "global_data_role": "CONSUMED_BY_UNRELATED_HISTORICAL_ROUTES",
        "current_recipe_role": "CHECKPOINT_UNSEEN_AFTER_V2_RECIPE_FREEZE",
        "selection": selection,
        "source_receipt_sha256": sha256_json(source),
        "intrinsics_fx_fy_cx_cy": INTRINSICS,
        "metric_depth_tier": "A_SOURCE_NATIVE",
        "support_boundary_tier": "B_GEOMETRY_ANCHORED_TEACHER",
        "complete_truth_required": False,
        "current_student_reducer_or_task_outcome_used": False,
        "no_recalibration_after_open": True,
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
        payload, report = build_payload(frame, identity, identity_sha, contract_sha)
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
                "source_archive_sha256_inherited": frame.source_archive_sha256,
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

    counts: Counter[str] = Counter()
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
        "WALKXYZ_C01_EXACT_EXTRACTED_SOURCE_RECEIPT": bool(
            source["metadata_sha256"] == EXPECTED_METADATA_SHA256
            and source["payload_file_counts"] == EXPECTED_PAYLOAD_COUNTS
        ),
        "WALKXYZ_C02_CURRENT_RECIPE_CHECKPOINT_EXCLUSION": bool(
            len(recipe_receipts) == len(EXACT_CURRENT_RECIPE_RECEIPTS)
        ),
        "WALKXYZ_C03_TWELVE_METADATA_ONLY_UNIQUE_FRAMES": bool(
            len(frame_receipts) == FRAME_COUNT
            and len({row["sample_id"] for row in frame_receipts}) == FRAME_COUNT
        ),
        "WALKXYZ_C04_SOURCE_NATIVE_METRIC_DEPTH_ALL_FRAMES": all(
            row["metric_depth_valid_pixels"] > 0 for row in frame_receipts
        ),
        "WALKXYZ_C05_GEOMETRY_ANCHORED_SUPPORT_ALL_FRAMES": all(
            row["support_plane_valid"] and row["support_valid_pixels"] > 0
            for row in frame_receipts
        ),
        "WALKXYZ_C06_BOUNDARY_EVIDENCE_ALL_FRAMES": all(
            row["evidence_valid_pixels"] > 0 and row["boundary_seed_pixels"] > 0
            for row in frame_receipts
        ),
        "WALKXYZ_C07_PAYLOAD_ROUNDTRIP_AND_HASHES": bool(
            roundtrip
            and all(
                sha256_file(Path(row["output"])) == row["output_sha256"]
                for row in frame_receipts
            )
        ),
        "WALKXYZ_C08_UNKNOWN_FAIL_CLOSED": fail_closed,
        "WALKXYZ_C09_FACTOR_ONLY_NO_CURRENT_MODEL_OR_TASK_OUTPUT": task_firewall,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_tum_walking_xyz_final_labels_result_v1",
        "status": (
            "AG_R2_TUM_WALKING_XYZ_FINAL_LABELS_PASS"
            if passed
            else "AG_R2_TUM_WALKING_XYZ_FINAL_LABELS_NOT_EVALUABLE"
        ),
        "passed": passed,
        "parent_count": 1,
        "frame_count": len(frame_receipts),
        "source": {
            "dataset": "TUM RGB-D fr3 walking_xyz",
            "real_world": True,
            "globally_unopened_claim": False,
            "checkpoint_unseen_by_current_frozen_recipe": True,
            "source_receipt": source,
            "source_receipt_sha256": sha256_json(source),
        },
        "current_recipe_receipts": recipe_receipts,
        "label_contract": contract,
        "label_contract_sha256": contract_sha,
        "support_identity_receipt": identity_receipt,
        "support_identity_receipt_sha256": identity_sha,
        "coverage": dict(counts),
        "gates": gates,
        "frames": frame_receipts,
        "decision": {
            "complete_truth_required": False,
            "current_model_or_reducer_output_opened_during_materialization": False,
            "next_action": "Run the already frozen V2 recipe exactly once; never fit or recalibrate on this source outcome.",
        },
        "claim_ceiling": "Current-recipe-checkpoint-unseen real TUM metric depth plus geometry-anchored tier-B support/boundary; not globally fresh, complete truth, cross-sensor generalization, deployment, product, or safety proof.",
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
    parser.add_argument(
        "--source-protocol", type=Path, default=DEFAULT_SOURCE_PROTOCOL
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name in ("source_root", "source_protocol", "output_dir"):
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
