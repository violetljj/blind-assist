"""Convert the X1b symmetric Floxel oracle into a two-scan-lag causal source.

At wall-clock frame ``t`` the five required scans for reference ``t-2`` are
all available.  X1c transports the sealed reference-frame cells through the
measured two-scan delay and expresses them in the current ego frame.  This
removes future information from the online decision boundary without changing
the source estimator, motion bounds, route geometry, or scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x1b_symmetric_floxel_oracle as x1b  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import _box_history, load_native_boxes  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    _causal_pose,
    _rotate_world_velocity_to_ego,
    _world_to_ego_xy,
    atomic_npz,
)
from jrdb_rgb_bridge import read_bag_pose_and_rgb  # noqa: E402


SCHEMA = "blindassist-dtr-x1c-lag-compensated-floxel-source-v1"
LEDGER_SCHEMA = "blindassist-dtr-x1c-lag-compensated-floxel-ledger-v1"
LAG_SCANS = 2


def _paths(root: Path) -> dict[str, Path]:
    return {
        "ledger": root / "lag-compensated-floxel.npz",
        "manifest": root / "lag-compensated-floxel.json",
        "result": root / "result.json",
    }


def _ego_velocity_to_world(velocity: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.column_stack(
        (
            cosine * velocity[:, 0] - sine * velocity[:, 1],
            sine * velocity[:, 0] + cosine * velocity[:, 1],
        )
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.symmetric_ledger.resolve(strict=True)
    source_manifest_path = args.symmetric_manifest.resolve(strict=True)
    source = x1._load_sealed(source_path, source_manifest_path, x1b.LEDGER_SCHEMA)
    frames = [int(value) for value in source["frames"]]
    timestamps = {
        int(frame): float(stamp)
        for frame, stamp in zip(source["frames"], source["frame_time_s"])
    }
    pose_samples, _rgb, authority = read_bag_pose_and_rgb(args.bag.resolve(strict=True))
    poses = {
        frame: _causal_pose(pose_samples, round(timestamps[frame] * 1e9))
        for frame in frames
    }
    index_by_frame = {frame: index for index, frame in enumerate(frames)}
    output_rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    delay_rows = []
    for reference in range(x1.TARGET_FIRST_FRAME, x1.TARGET_LAST_FRAME + 1):
        reference_index = index_by_frame[reference]
        output_index = reference_index + LAG_SCANS
        require(output_index < len(frames), f"x1c_output_frame_missing:{reference}")
        output_frame = frames[output_index]
        start = int(source["offsets"][reference_index])
        stop = int(source["offsets"][reference_index + 1])
        positions = np.column_stack(
            (source["forward_m"][start:stop], source["left_m"][start:stop])
        ).astype(np.float64)
        velocities = np.column_stack(
            (
                source["velocity_forward_mps"][start:stop],
                source["velocity_left_mps"][start:stop],
            )
        ).astype(np.float64)
        counts = source["source_point_count"][start:stop].astype(np.int32)
        delay_s = timestamps[output_frame] - timestamps[reference]
        require(delay_s > 0.0, f"x1c_nonpositive_delay:{reference}")
        reference_pose = poses[reference]
        output_pose = poses[output_frame]
        world_position = np.column_stack(
            (
                reference_pose["x_m"]
                + math.cos(reference_pose["yaw_rad"]) * positions[:, 0]
                - math.sin(reference_pose["yaw_rad"]) * positions[:, 1],
                reference_pose["y_m"]
                + math.sin(reference_pose["yaw_rad"]) * positions[:, 0]
                + math.cos(reference_pose["yaw_rad"]) * positions[:, 1],
            )
        )
        world_velocity = _ego_velocity_to_world(
            velocities, float(reference_pose["yaw_rad"])
        )
        transported_world = world_position + world_velocity * delay_s
        transported_position = _world_to_ego_xy(transported_world, output_pose)
        transported_velocity = _rotate_world_velocity_to_ego(
            world_velocity, output_pose
        )
        output_rows[output_frame] = (
            transported_position.astype(np.float32),
            transported_velocity.astype(np.float32),
            counts,
        )
        delay_rows.append(
            {
                "reference_frame": reference,
                "output_frame": output_frame,
                "delay_s": delay_s,
                "cells": int(len(positions)),
            }
        )
    empty = (
        np.empty((0, 2), np.float32),
        np.empty((0, 2), np.float32),
        np.empty(0, np.int32),
    )
    rows = [output_rows.get(frame, empty) for frame in frames]
    offsets = np.cumsum([0] + [len(row[0]) for row in rows], dtype=np.int64)
    arrays = {
        "frames": np.asarray(frames, dtype=np.int32),
        "frame_time_s": np.asarray([timestamps[frame] for frame in frames], dtype=np.float64),
        "offsets": offsets,
        "forward_m": np.concatenate([row[0][:, 0] for row in rows]).astype(np.float32),
        "left_m": np.concatenate([row[0][:, 1] for row in rows]).astype(np.float32),
        "velocity_forward_mps": np.concatenate([row[1][:, 0] for row in rows]).astype(np.float32),
        "velocity_left_mps": np.concatenate([row[1][:, 1] for row in rows]).astype(np.float32),
        "component_id": np.concatenate([np.arange(len(row[0]), dtype=np.int32) for row in rows]),
        "source_point_count": np.concatenate([row[2] for row in rows]),
        "flow_support": np.concatenate([np.ones(len(row[0]), dtype=np.float32) for row in rows]),
    }
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = _paths(root)
    atomic_npz(paths["ledger"], **arrays)
    manifest = {
        "schema": LEDGER_SCHEMA,
        "truth_blind": True,
        "oracle": False,
        "sequence": x1.SEQUENCE,
        "motion_source": "two-scan-lag causal transport of symmetric five-scan voxel flow",
        "online_information_boundary": (
            "output frame t consumes sealed source reference t-2 estimated from scans t-4 through t"
        ),
        "lag_scans": LAG_SCANS,
        "delay_s": {
            "minimum": min(row["delay_s"] for row in delay_rows),
            "median": float(np.median([row["delay_s"] for row in delay_rows])),
            "maximum": max(row["delay_s"] for row in delay_rows),
        },
        "frozen_downstream": {
            "route_entry_geometry": "UNCHANGED_R7",
            "motion_bounds": "UNCHANGED_R7",
            "event_scorer": "UNCHANGED_R7",
        },
        "source": {
            "symmetric_ledger": str(source_path),
            "symmetric_ledger_sha256": sha256_file(source_path),
            "symmetric_manifest": str(source_manifest_path),
            "symmetric_manifest_sha256": sha256_file(source_manifest_path),
            "bag_sha256": sha256_file(args.bag.resolve(strict=True)),
            "bag_pose_authority": authority,
        },
        "diagnostics": {
            "transported_frames": delay_rows,
            "output_cells": int(len(arrays["forward_m"])),
        },
        "ledger": str(paths["ledger"]),
        "ledger_sha256": sha256_file(paths["ledger"]),
    }
    write_json(paths["manifest"], manifest)
    return manifest


def score(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.root.resolve(strict=True))
    ledger = x1._load_sealed(paths["ledger"], paths["manifest"], LEDGER_SCHEMA)
    frames = [int(value) for value in ledger["frames"]]
    timestamps = {
        int(frame): float(stamp)
        for frame, stamp in zip(ledger["frames"], ledger["frame_time_s"])
    }
    pose_samples, _rgb, _authority = read_bag_pose_and_rgb(args.bag.resolve(strict=True))
    poses = {
        frame: _causal_pose(pose_samples, round(timestamps[frame] * 1e9))
        for frame in frames
    }
    boxes = load_native_boxes(
        args.labels.resolve(strict=True), timestamps, poses, sequence=x1.SEQUENCE
    )
    history = _box_history(boxes)
    lagged = x1._diagnose(
        ledger,
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=boxes,
        history=history,
        poses=poses,
    )
    gate = {
        "at_least_two_correct_frames": lagged["correct_frames"] >= 2,
        "at_least_two_correct_route_entry_frames": (
            lagged["correct_route_entry_frames"] >= 2
        ),
    }
    met = all(gate.values())
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X1C_LAG_COMPENSATED_FLOXEL_SOURCE_HEADROOM_MET"
            if met
            else "DTR_X1C_LAG_COMPENSATED_FLOXEL_SOURCE_HEADROOM_NOT_MET"
        ),
        "question": (
            "Can the symmetric five-scan source retain its two-frame motion headroom after "
            "being re-anchored to a wall-clock-causal output with measured lag compensation?"
        ),
        "lag_compensated_source": lagged,
        "delay_s": manifest["delay_s"],
        "gate": gate,
        "decision": {
            "headroom_met": met,
            "next": (
                "FREEZE_AND_RUN_X0_ERROR_SLICE_WITH_LAG_COMPENSATED_SOURCE"
                if met
                else "FLOXEL_HEADROOM_IS_NONCAUSAL_ONLY_CLOSE_SOURCE_FAMILY"
            ),
        },
        "claim_limits": [
            "Opened-truth single-event Development canary; not source-disjoint confirmation or six-sequence performance evidence.",
            "Online information causality is established by the two-scan lag; real-time compute latency is not established.",
            "The upstream estimator is an independent Floxels-inspired adapter, not official Floxels code.",
            "False-segment suppression remains unevaluated.",
        ],
        "sources": {
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
    }
    write_json(paths["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    symmetric_root = (
        REPO / "artifacts.local" / "evidence" / "dtr-x1b" / "symmetric-floxel-oracle"
    )
    root = (
        REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x1c"
        / "lag-compensated-floxel-source"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--symmetric-ledger", type=Path, default=symmetric_root / "symmetric-floxel.npz"
    )
    parser.add_argument(
        "--symmetric-manifest", type=Path, default=symmetric_root / "symmetric-floxel.json"
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "datasets"
        / "dtr-c31-jrdb-fresh-confirmation"
        / f"{x1.SEQUENCE}.bag",
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"materialize", "run"}:
        manifest = materialize(args)
        print(
            json.dumps(
                {
                    "materialized": True,
                    "delay_s": manifest["delay_s"],
                    "output_cells": manifest["diagnostics"]["output_cells"],
                },
                sort_keys=True,
            )
        )
    if args.mode in {"score", "run"}:
        result = score(args)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "gate": result["gate"],
                    "delay_s": result["delay_s"],
                    "lag_compensated_source": {
                        key: result["lag_compensated_source"][key]
                        for key in (
                            "associated_frames",
                            "correct_frames",
                            "correct_route_entry_frames",
                            "minimum_associated_error_mps",
                        )
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
