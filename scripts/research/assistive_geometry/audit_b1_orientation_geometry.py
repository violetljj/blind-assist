#!/usr/bin/env python3
"""Audit B1 full-FOV orientation geometry without opening task outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    TruthReaderPolicy,
    interpolate_camera_to_world,
    orientation_index,
    parse_trajectory,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)


def tensor_hw_for_orientation(index: int) -> tuple[int, int]:
    require(index in (0, 1, 2, 3), "orientation index drift")
    return (608, 448) if index in (1, 3) else (448, 608)


def audit(manifest: dict[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema") == "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_media_manifest_v1", "manifest schema drift")
    require(manifest.get("task_outcome_opened") is False, "task outcome firewall drift")
    require(manifest.get("model_outputs_read") is False, "model output firewall drift")
    policy = TruthReaderPolicy()
    roles: dict[str, Any] = {}
    for role in ("TRAIN", "DEVELOPMENT", "CONFIRMATION"):
        videos = [video for video in manifest["videos"] if video["role"] == role]
        rows: list[dict[str, Any]] = []
        total = {str(index): 0 for index in range(4)}
        tensor_shapes = {"608x448": 0, "448x608": 0}
        for video in videos:
            trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
            counts = {str(index): 0 for index in range(4)}
            for stem in video["selected_frame_stems"]:
                timestamp = float(str(stem).rsplit("_", 1)[1])
                pose, _ = interpolate_camera_to_world(
                    trajectory, timestamp, policy.maximum_pose_bracketing_gap_seconds
                )
                index = orientation_index(pose)
                counts[str(index)] += 1
                total[str(index)] += 1
                height, width = tensor_hw_for_orientation(index)
                tensor_shapes[f"{height}x{width}"] += 1
            rows.append({
                "visit_id": str(video["visit_id"]),
                "video_id": str(video["video_id"]),
                "orientation_counts": counts,
                "portrait_frame_count": counts["1"] + counts["3"],
                "landscape_frame_count": counts["0"] + counts["2"],
            })
        frame_count = sum(total.values())
        portrait_count = total["1"] + total["3"]
        roles[role] = {
            "video_count": len(videos),
            "frame_count": frame_count,
            "orientation_counts": total,
            "full_fov_tensor_shape_counts": tensor_shapes,
            "portrait_frame_count": portrait_count,
            "portrait_fraction": portrait_count / frame_count,
            "landscape_frame_count": frame_count - portrait_count,
            "landscape_fraction": (frame_count - portrait_count) / frame_count,
            "videos": rows,
        }
    return roles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    require(sha256_file(manifest_path) == args.expected_manifest_sha256, "manifest SHA drift")
    roles = audit(load_json(manifest_path))
    train = roles["TRAIN"]
    receipt = {
        "schema": "blindassist_assistive_geometry_b1_orientation_geometry_audit_v1",
        "manifest_sha256": args.expected_manifest_sha256,
        "truth_reader_sha256": sha256_file(Path(__file__).with_name("arkitscenes_truth_reader.py")),
        "roles": roles,
        "attempt_01_fixed_portrait": {
            "admitted_train_frames": train["portrait_frame_count"],
            "excluded_train_frames": train["landscape_frame_count"],
            "excluded_train_fraction": train["landscape_fraction"],
            "full_fov_preserved_for_all_frames": False,
            "scientifically_admissible": False,
        },
        "required_correction": {
            "tensor_shapes_nchw": [[1, 3, 608, 448], [1, 3, 448, 608]],
            "orientation_bucketed_batches": True,
            "full_fov_preserved": True,
            "independent_sx_sy_k_update": True,
            "product_claims_must_report_portrait_stratum": True,
        },
        "firewalls": {
            "image_or_depth_pixels_opened": False,
            "task_outcome_opened": False,
            "model_outputs_read": False,
            "confirmation_identity_only": True,
        },
        "terminal": "B1_PROTOCOL_ATTEMPT_01_SUPERSEDED_PRE_OUTCOME_ORIENTATION_GEOMETRY_CONFLICT",
    }
    write_json_exclusive(args.output.resolve(), receipt)
    print(json.dumps({
        "terminal": receipt["terminal"],
        "train_portrait_frames": train["portrait_frame_count"],
        "train_landscape_frames": train["landscape_frame_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
