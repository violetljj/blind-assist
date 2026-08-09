#!/usr/bin/env python3
"""Materialize the frozen DEVELOPMENT_SELECTION geometry targets after activation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    parse_trajectory,
    load_manifest_frame,
)
from scripts.research.assistive_geometry.audit_b1_orientation_geometry import (  # noqa: E402
    tensor_hw_for_orientation,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)
from scripts.research.assistive_geometry.materialize_b1_train_targets import (  # noqa: E402
    SOURCE_MANIFEST_SHA256,
    _save_npz_exclusive,
    build_band_targets,
    tensor_intrinsics,
)


MANIFEST_SCHEMA = "blindassist_assistive_geometry_b1_development_target_manifest_v1"
ROLE = "DEVELOPMENT_SELECTION"


def select_frozen_videos(
    source: dict[str, Any], expected: list[dict[str, str]]
) -> list[dict[str, Any]]:
    lookup = {
        str(video["video_id"]): video
        for video in source["videos"]
        if video["role"] == "DEVELOPMENT"
    }
    require(len(lookup) == 8, "source DEVELOPMENT count drift")
    require(len(expected) == 4, "DEVELOPMENT_SELECTION count drift")
    rows: list[dict[str, Any]] = []
    for identity in expected:
        video_id = str(identity["video_id"])
        require(video_id in lookup, f"DEVELOPMENT_SELECTION video missing: {video_id}")
        video = lookup[video_id]
        require(
            str(video["visit_id"]) == str(identity["visit_id"]),
            f"DEVELOPMENT_SELECTION visit drift: {video_id}",
        )
        rows.append(video)
    require(len({str(video["visit_id"]) for video in rows}) == 4, "Development visit overlap")
    return rows


def _validate_video_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json(path)
    require(receipt.get("evaluation_role") == ROLE, f"wrong Development role: {path}")
    for frame in receipt["frames"]:
        target = Path(frame["target"]["path"])
        require(
            target.is_file() and target.stat().st_size == int(frame["target"]["bytes"]),
            f"target size drift: {target}",
        )
        require(sha256_file(target) == frame["target"]["sha256"], f"target SHA drift: {target}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    source_path = args.source_manifest.resolve()
    protocol = load_json(protocol_path)
    source = load_json(source_path)
    require(
        protocol.get("schema")
        == "blindassist_assistive_geometry_b1_a0_development_evaluation_protocol_v1",
        "Development evaluation protocol schema drift",
    )
    require(
        protocol["authority"].get("development_selection_target_materialization") is True,
        "Development target materialization is not activated",
    )
    for seed in (17, 29, 43):
        result_path = Path(protocol["training_outputs"][str(seed)]["result_path"])
        require(result_path.is_file(), f"seed {seed} formal train result is missing")
        train_result = load_json(result_path)
        require(int(train_result.get("seed", -1)) == seed, f"seed {seed} train result identity drift")
        require(
            train_result.get("terminal") == "B1_A0_DEPTH_ONLY_FORMAL_TRAIN_SEED_COMPLETE"
            and int(train_result.get("completed_optimizer_steps", -1)) == 6000,
            f"seed {seed} formal train result is incomplete",
        )
        require(
            train_result.get("development_or_confirmation_content_opened") is False,
            f"seed {seed} train result crossed the data firewall",
        )
    binding = protocol["implementation_bindings"]["target_materializer"]
    require(binding["sha256"] == sha256_file(Path(__file__)), "target materializer binding drift")
    require(sha256_file(source_path) == SOURCE_MANIFEST_SHA256, "source manifest hash drift")
    expected = protocol["data_role"]["identities"]
    videos = select_frozen_videos(source, expected)

    attempt = {
        "schema": "blindassist_assistive_geometry_b1_development_target_attempt_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "source_manifest_sha256": sha256_file(source_path),
        "producer_sha256": sha256_file(Path(__file__)),
        "data_role": ROLE,
        "output_root": str(args.output_root.resolve()),
    }
    attempt_path = args.output_root / "attempt.json"
    if args.output_root.exists():
        require(args.resume, "existing Development target root requires --resume")
        require(attempt_path.is_file() and load_json(attempt_path) == attempt, "resume attempt drift")
        require(not (args.output_root / "manifest.json").exists(), "Development target manifest already complete")
    else:
        args.output_root.mkdir(parents=True)
        write_json_exclusive(attempt_path, attempt)

    receipts: list[dict[str, Any]] = []
    for video_index, video in enumerate(videos, start=1):
        video_id = str(video["video_id"])
        receipt_path = args.output_root / "receipts" / f"{video_index:02d}-{ROLE}-{video_id}.json"
        if receipt_path.exists():
            receipts.append(_validate_video_receipt(receipt_path))
            print(json.dumps({"completed": video_index, "total": 4, "video_id": video_id, "resumed": True}), flush=True)
            continue
        trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
        frames: list[dict[str, Any]] = []
        for frame_index in range(len(video["selected_frame_stems"])):
            frame = load_manifest_frame(video, frame_index, trajectory)
            identity = frame["identity"]
            truth = frame["truth"]
            orientation = int(frame["orientation"]["rotation_index"])
            target_hw = tensor_hw_for_orientation(orientation)
            depth = np.asarray(frame["depth_m_upright"], dtype=np.float32)
            depth_valid = np.asarray(truth["depth_valid"], dtype=np.bool_)
            ground_plane_valid = truth["ground_plane"] is not None
            bands = build_band_targets(truth)
            target_values = {
                "depth_m_source": depth,
                "depth_valid_source": depth_valid,
                "ground_probability_source": np.asarray(truth["ground_probability"], dtype=np.float32),
                "ground_label_valid_source": depth_valid if ground_plane_valid else np.zeros_like(depth_valid),
                "intrinsics_source": np.asarray(frame["intrinsics_upright"], dtype=np.float32),
                "intrinsics_tensor": tensor_intrinsics(frame["intrinsics_upright"], depth.shape, target_hw),
                "up_camera": np.asarray(frame["orientation"]["up_camera"], dtype=np.float32),
                "camera_height_m": np.asarray(
                    float(truth["ground_plane"]["camera_height_m"]) if ground_plane_valid else np.nan,
                    dtype=np.float32,
                ),
                "ground_plane_valid": np.asarray(ground_plane_valid, dtype=np.bool_),
                "clearance_m": bands["clearance_m"],
                "clearance_valid": bands["clearance_valid"],
                "occupancy": bands["occupancy"],
                "occupancy_valid": bands["occupancy_valid"],
                "target_hw": np.asarray(target_hw, dtype=np.int32),
                "orientation_index": np.asarray(orientation, dtype=np.int8),
            }
            output = args.output_root / "targets" / video_id / f"{identity['frame_stem']}.npz"
            rgb_entry = video["extracted"]["lowres_wide"][frame_index]
            frames.append(
                {
                    "frame_index": frame_index,
                    "frame_stem": identity["frame_stem"],
                    "orientation_index": orientation,
                    "orientation_family": "portrait" if orientation in (1, 3) else "landscape",
                    "source_hw": list(depth.shape),
                    "target_hw": list(target_hw),
                    "truth_status": truth["status"],
                    "ground_plane_valid": ground_plane_valid,
                    "rgb_source": rgb_entry,
                    "target": _save_npz_exclusive(output, target_values),
                }
            )
        receipt = {
            "source_role": "DEVELOPMENT",
            "evaluation_role": ROLE,
            "visit_id": str(video["visit_id"]),
            "video_id": video_id,
            "frame_count": len(frames),
            "portrait_frame_count": sum(row["orientation_family"] == "portrait" for row in frames),
            "landscape_frame_count": sum(row["orientation_family"] == "landscape" for row in frames),
            "frames": frames,
        }
        write_json_exclusive(receipt_path, receipt)
        receipts.append(receipt)
        print(json.dumps({"completed": video_index, "total": 4, "video_id": video_id, "frames": len(frames)}), flush=True)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        **{key: value for key, value in attempt.items() if key != "schema"},
        "video_count": len(receipts),
        "frame_count": sum(row["frame_count"] for row in receipts),
        "portrait_frame_count": sum(row["portrait_frame_count"] for row in receipts),
        "landscape_frame_count": sum(row["landscape_frame_count"] for row in receipts),
        "videos": receipts,
        "development_content_opened": True,
        "development_calibration_content_opened": False,
        "confirmation_content_opened": False,
        "model_outputs_read": False,
        "terminal": "B1_A0_DEVELOPMENT_SELECTION_TARGETS_MATERIALIZED",
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
