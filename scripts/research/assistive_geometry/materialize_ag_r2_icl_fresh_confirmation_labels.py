#!/usr/bin/env python3
"""Materialize full factor supervision for checkpoint-unseen ICL trajectory 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from build_ag_st_factor_labels import (  # noqa: E402
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    compute_geometric_factors,
)
from materialize_ag_r2_f1_source_native_labels import (  # noqa: E402
    FORBIDDEN_TASK_FIELD_TOKENS,
    METRIC_PROVENANCE,
    REQUIRED_F1_SUPERVISION_FIELDS,
    UNKNOWN_PROVENANCE,
    sha256_json,
)
from run_ag_st_icl_fresh_depth_boundary_canary import (  # noqa: E402
    SELECTED_INDICES,
)
from run_ag_st_icl_mesh_support_identity import parse_global_pose_text  # noqa: E402
from run_ag_st_icl_pixel_boundary_canary import (  # noqa: E402
    canonical_camera_to_world,
    downsample_exact_depth,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    require,
    sha256_file,
)


DEFAULT_SOURCE_ROOT = REPO_ROOT / "artifacts.local/downloads/ag-st-icl-boundary-r1"
DEFAULT_ARCHIVE = DEFAULT_SOURCE_ROOT / "living_room_traj1_frei_png.tar.gz"
DEFAULT_SELECTED_ROOT = DEFAULT_SOURCE_ROOT / "selected12"
DEFAULT_POSES = DEFAULT_SOURCE_ROOT / "livingRoom1n.gt.sim"
DEFAULT_IDENTITY_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-icl-mesh-support-identity-r2/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-icl-fresh-confirmation-labels-r0"
)
EXPECTED_ARCHIVE_SHA256 = (
    "75D5F87EBAF313F6DDF9D1750815C277E3B16DB8ABD68A950F6A3665A49F2403"
)
EXPECTED_POSE_SHA256 = (
    "672FB9BFAB2FF7B4CA3A1CD5DC06DF3EFE16370DAAC654463AE3082F9851AFEB"
)
EXPECTED_IDENTITY_RESULT_SHA256 = (
    "8D88F92E0B4D50CA75F2BC03B0E4597428AA899188F9C5EF8716B5D36372E8EC"
)
PARENT_ID = "icl_living_room_kt1"
ROLE = "FRESH_CONFIRMATION"


def load_rgb(path: Path) -> np.ndarray:
    require(path.is_file(), f"ICL RGB missing: {path}")
    with Image.open(path) as image:
        raw = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(raw.shape == (480, 640, 3), "ICL RGB shape drift")
    output = np.ascontiguousarray(np.flipud(raw)[2::4, 2::4])
    require(output.shape == (120, 160, 3), "ICL RGB transform drift")
    return output


def build_payload(
    index: int,
    rgb_path: Path,
    depth_path: Path,
    pose_icl: np.ndarray,
    exact_floor_height_m: float,
    identity_receipt_sha256: str,
    transform_contract_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rgb = load_rgb(rgb_path)
    depth, valid, intrinsics = downsample_exact_depth(depth_path)
    pose = canonical_camera_to_world(pose_icl)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    gravity = pose[:3, :3].T @ world_up
    gravity /= np.linalg.norm(gravity)
    camera_height = float(pose[2, 3]) - exact_floor_height_m
    sample_id = f"{PARENT_ID}__rgb{index:06d}"
    quality = np.where(valid, 0.99, 0.0).astype(np.float32)
    tiers = np.where(valid, TIER_A_SOURCE, 0).astype(np.uint8)
    provenance = np.where(valid, PROVENANCE_SOURCE_NATIVE, 0).astype(np.uint8)
    uncertainty = np.where(valid, 0.002, np.inf).astype(np.float32)
    factors = compute_geometric_factors(
        depth,
        valid,
        intrinsics,
        pose,
        quality,
        tiers,
        provenance,
        uncertainty,
        support_camera_height_override_m=camera_height,
        support_plane_residual_override_m=0.01,
    )
    support_valid = np.asarray(factors["support_truth_valid_hw"], dtype=np.bool_)
    evidence_valid = np.asarray(factors["evidence_truth_valid_hw"], dtype=np.bool_)
    signed_residual = np.asarray(
        factors["height_above_support_m_hw"], dtype=np.float32
    ).copy()
    signed_residual[~support_valid] = np.nan
    camera_receipt = {
        "sample_id": sample_id,
        "source": "ICL_NUIM_LIVING_ROOM_TRAJECTORY_1",
        "pose_index": index,
        "intrinsics": intrinsics.tolist(),
        "camera_to_world": pose.tolist(),
        "rgb_sha256": sha256_file(rgb_path),
        "depth_sha256": sha256_file(depth_path),
    }
    camera_receipt_sha256 = sha256_json(camera_receipt)
    metric_provenance = np.where(
        valid, METRIC_PROVENANCE, UNKNOWN_PROVENANCE
    ).astype(np.uint8)
    payload: dict[str, np.ndarray] = {
        "sample_id": np.asarray(sample_id),
        "parent_id": np.asarray(PARENT_ID),
        "role": np.asarray(ROLE),
        "orientation": np.asarray("LANDSCAPE_IDENTITY"),
        "rgb_u8_hwc": rgb,
        "metric_depth_m_hw": depth.astype(np.float32),
        "metric_depth_valid_hw": valid.astype(np.bool_),
        "metric_depth_provenance_hw": metric_provenance,
        "intrinsics_output": intrinsics.astype(np.float64),
        "camera_to_world_output": pose.astype(np.float64),
        "gravity_up_camera_xyz": gravity.astype(np.float32),
        "camera_geometry_receipt_sha256": np.asarray(camera_receipt_sha256),
        "support_identity_receipt_sha256": np.asarray(identity_receipt_sha256),
        "label_transform_contract_sha256": np.asarray(transform_contract_sha256),
        "dense_normal_diagnostic_camera_xyz_hwc": np.asarray(
            factors["dense_normal_diagnostic_camera_xyz_hwc"], dtype=np.float32
        ),
        "normal_valid_diagnostic_hw": np.asarray(
            factors["normal_valid_hw"], dtype=np.bool_
        ),
        "support_truth_hw": np.asarray(factors["support_truth_hw"], dtype=np.float32),
        "support_truth_valid_hw": support_valid,
        "support_plane_normal_camera_xyz": np.asarray(
            factors["support_plane_normal_camera_xyz"], dtype=np.float32
        ),
        "camera_height_m": np.asarray(camera_height, dtype=np.float32),
        "support_plane_valid": np.asarray(
            factors["support_plane_valid"], dtype=np.bool_
        ),
        "support_signed_plane_residual_m_hw": signed_residual,
        "support_plane_fit_residual_diagnostic_m": np.asarray(0.01, dtype=np.float32),
        "obstacle_evidence_truth_hw": np.asarray(
            factors["obstacle_evidence_truth_hw"], dtype=np.float32
        ),
        "boundary_probability_diagnostic_hw": np.asarray(
            factors["boundary_probability_pseudo_hw"], dtype=np.float32
        ),
        "boundary_distance_px_hw": np.asarray(
            factors["boundary_distance_px_hw"], dtype=np.float32
        ),
        "boundary_seed_diagnostic_hw": (
            evidence_valid
            & (np.asarray(factors["boundary_probability_pseudo_hw"]) >= 0.5)
        ).astype(np.bool_),
        "evidence_truth_valid_hw": evidence_valid,
        "support_provenance_hw": np.asarray(
            factors["support_provenance_code_hw"], dtype=np.uint8
        ),
        "support_plane_provenance_code": np.asarray(
            factors["support_plane_provenance_code"], dtype=np.uint8
        ),
        "evidence_provenance_hw": np.asarray(
            factors["evidence_provenance_code_hw"], dtype=np.uint8
        ),
        "metric_depth_supervision_tier_code": np.asarray(1, dtype=np.uint8),
        "support_supervision_tier_code": np.asarray(2, dtype=np.uint8),
        "boundary_supervision_tier_code": np.asarray(2, dtype=np.uint8),
        "gravity_anchor_world_up_xyz": world_up.astype(np.float32),
    }
    report = {
        "sample_id": sample_id,
        "parent_id": PARENT_ID,
        "role": ROLE,
        "orientation": "LANDSCAPE_IDENTITY",
        "pose_index": index,
        "shape_hw": list(depth.shape),
        "metric_depth_valid_pixels": int(valid.sum()),
        "support_plane_valid": bool(factors["support_plane_valid"]),
        "support_valid_pixels": int(support_valid.sum()),
        "support_positive_pixels_ge_0_5": int(
            np.sum(support_valid & (payload["support_truth_hw"] >= 0.5))
        ),
        "evidence_valid_pixels": int(evidence_valid.sum()),
        "boundary_seed_pixels": int(payload["boundary_seed_diagnostic_hw"].sum()),
        "camera_height_m": camera_height,
        "camera_geometry_receipt_sha256": camera_receipt_sha256,
        "rgb_sha256": sha256_file(rgb_path),
        "depth_sha256": sha256_file(depth_path),
    }
    return payload, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.archive) == EXPECTED_ARCHIVE_SHA256, "ICL archive drift")
    require(sha256_file(args.poses) == EXPECTED_POSE_SHA256, "ICL pose drift")
    require(
        sha256_file(args.identity_result) == EXPECTED_IDENTITY_RESULT_SHA256,
        "ICL support identity drift",
    )
    identity_result = json.loads(args.identity_result.read_text(encoding="utf-8"))
    exact_floor_height = float(identity_result["mesh"]["exact_floor_world_height_m"])
    poses = parse_global_pose_text(args.poses.read_text(encoding="utf-8"))
    require(len(poses) == 965, "ICL pose count drift")
    identity_receipt = {
        "source": "ICL_EXACT_MESH_FLOOR",
        "world_up": [0.0, 0.0, 1.0],
        "support_world_height_m": exact_floor_height,
        "identity_result_sha256": EXPECTED_IDENTITY_RESULT_SHA256,
    }
    identity_receipt_sha256 = sha256_json(identity_receipt)
    transform_contract = {
        "source_raster": "ICL positive-fy vertical flip",
        "downsample": "offset-2 stride-4 exact RGB-D",
        "world_transform": "ICL +Y up to canonical +Z up",
        "factor_derivation": "source-native exact depth plus exact pose/floor; no task/reducer output",
    }
    transform_contract_sha256 = sha256_json(transform_contract)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in SELECTED_INDICES:
        rgb_path = args.selected_root / f"rgb/{index}.png"
        depth_path = args.selected_root / f"depth/{index}.png"
        payload, report = build_payload(
            index,
            rgb_path,
            depth_path,
            poses[index],
            exact_floor_height,
            identity_receipt_sha256,
            transform_contract_sha256,
        )
        require(
            REQUIRED_F1_SUPERVISION_FIELDS.issubset(payload),
            "ICL required factor fields missing",
        )
        require(
            not any(
                token in key.lower()
                for key in payload
                for token in FORBIDDEN_TASK_FIELD_TOKENS
            ),
            "ICL task field firewall violated",
        )
        path = args.output_dir / f"{report['sample_id']}.npz"
        np.savez_compressed(path, **payload)
        with np.load(path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "ICL payload field drift")
            require(
                str(np.asarray(written["sample_id"]).item()) == report["sample_id"],
                "ICL payload identity drift",
            )
        rows.append(
            {
                **report,
                "output": str(path.resolve()),
                "output_sha256": sha256_file(path),
                "output_bytes": path.stat().st_size,
                "field_count": len(payload),
            }
        )
    unknown_fail_closed = True
    for row in rows:
        with np.load(row["output"], allow_pickle=False) as payload:
            metric_valid = np.asarray(payload["metric_depth_valid_hw"], dtype=np.bool_)
            support_valid = np.asarray(payload["support_truth_valid_hw"], dtype=np.bool_)
            evidence_valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
            unknown_fail_closed &= bool(
                np.all(np.asarray(payload["metric_depth_m_hw"])[~metric_valid] == 0.0)
                and np.all(np.asarray(payload["support_truth_hw"])[~support_valid] == 0.0)
                and np.all(
                    np.asarray(payload["obstacle_evidence_truth_hw"])[~evidence_valid]
                    == 0.0
                )
            )
    gates = {
        "ICLFC_C01_EXACT_SOURCE_RECEIPTS": True,
        "ICLFC_C02_TWELVE_UNIQUE_CHECKPOINT_UNSEEN_FRAMES": len(rows) == 12
        and len({row["sample_id"] for row in rows}) == 12,
        "ICLFC_C03_SOURCE_NATIVE_METRIC_DEPTH_ALL_FRAMES": all(
            row["metric_depth_valid_pixels"] > 0 for row in rows
        ),
        "ICLFC_C04_POSE_AND_EXACT_FLOOR_SUPPORT_ALL_FRAMES": all(
            row["support_plane_valid"] for row in rows
        ),
        "ICLFC_C05_BOUNDARY_EVIDENCE_ALL_FRAMES": all(
            row["evidence_valid_pixels"] > 0 for row in rows
        ),
        "ICLFC_C06_PAYLOAD_ROUNDTRIP_AND_HASHES": all(
            Path(row["output"]).is_file()
            and sha256_file(Path(row["output"])) == row["output_sha256"]
            for row in rows
        ),
        "ICLFC_C07_UNKNOWN_FAIL_CLOSED": unknown_fail_closed,
        "ICLFC_C08_NO_TASK_OR_REDUCER_OUTPUT_USED": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_icl_fresh_confirmation_labels_result_v1",
        "status": "AG_R2_ICL_FRESH_CONFIRMATION_LABELS_PASS"
        if passed
        else "AG_R2_ICL_FRESH_CONFIRMATION_LABELS_FAIL",
        "passed": passed,
        "source": {
            "dataset": "ICL-NUIM living room trajectory 1",
            "archive": str(args.archive.resolve()),
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "poses": str(args.poses.resolve()),
            "poses_sha256": EXPECTED_POSE_SHA256,
            "identity_result": str(args.identity_result.resolve()),
            "identity_result_sha256": EXPECTED_IDENTITY_RESULT_SHA256,
            "checkpoint_unseen_by_current_student": True,
            "synthetic_exact_not_real_world": True,
        },
        "selection": {
            "indices": list(SELECTED_INDICES),
            "rule": "12 evenly spaced indices frozen by the preexisting ICL trajectory-1 source contract before the current student existed",
            "model_or_task_outcome_used": False,
        },
        "support_identity_receipt": identity_receipt,
        "support_identity_receipt_sha256": identity_receipt_sha256,
        "label_transform_contract": transform_contract,
        "label_transform_contract_sha256": transform_contract_sha256,
        "parent_count": 1,
        "frame_count": len(rows),
        "frames": rows,
        "gates": gates,
        "decision": {
            "current_student_or_reducer_output_opened_during_materialization": False,
            "complete_truth_required": False,
            "next_action": "Run the already frozen student and AG seam once; do not fit or recalibrate on ICL.",
        },
        "claim_ceiling": "Checkpoint-unseen synthetic-exact factor confirmation; not real-world, mobile, product, or safety evidence.",
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--identity-result", type=Path, default=DEFAULT_IDENTITY_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name in ("archive", "selected_root", "poses", "identity_result", "output_dir"):
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
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
