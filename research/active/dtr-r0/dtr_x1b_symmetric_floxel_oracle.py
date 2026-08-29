"""Run a five-scan symmetric Floxel oracle after the causal X1 miss.

This diagnostic restores the published adjacent-frame information pattern
(two past and two future scans) while keeping the X1 rear source crop, voxel
representation, losses, motion bounds, aggregation, and scorer unchanged.  It
is deliberately non-causal and therefore cannot be a deployable avoidance
source.  Its only purpose is to decide whether the Floxel source family has
information headroom worth causal distillation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import (  # noqa: E402
    _box_history,
    load_native_boxes,
    load_world_clouds,
)
from dtr_r7_occupancy_flow_canary import _causal_pose, atomic_npz  # noqa: E402
from jrdb_rgb_bridge import read_bag_pose_and_rgb  # noqa: E402


SCHEMA = "blindassist-dtr-x1b-symmetric-floxel-oracle-v1"
LEDGER_SCHEMA = "blindassist-dtr-x1b-symmetric-floxel-ledger-v1"
FIRST_SOURCE_FRAME = x1.SUPPORT_FIRST_FRAME
LAST_SOURCE_FRAME = x1.TARGET_LAST_FRAME + 2
SCAN_OFFSETS = (-2, -1, 1, 2)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "ledger": root / "symmetric-floxel.npz",
        "manifest": root / "symmetric-floxel.json",
        "result": root / "result.json",
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    require(torch.cuda.is_available(), "x1b_cuda_unavailable")
    device = torch.device("cuda:0")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = _paths(root)
    with np.load(args.baseline_ledger.resolve(strict=True), allow_pickle=False) as values:
        timestamps = {
            int(frame): float(stamp)
            for frame, stamp in zip(values["frames"], values["frame_time_s"])
            if FIRST_SOURCE_FRAME <= int(frame) <= LAST_SOURCE_FRAME
        }
    frames, frame_times, poses, world_clouds, lidar = load_world_clouds(
        bag_path=args.bag.resolve(strict=True),
        timestamps_path=args.timestamps.resolve(strict=True),
        calibration_dir=args.calibration_dir.resolve(strict=True),
        timestamps_override=timestamps,
    )
    require(
        frames.tolist() == list(range(FIRST_SOURCE_FRAME, LAST_SOURCE_FRAME + 1)),
        "x1b_frame_window",
    )
    rows = []
    diagnostics = []
    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        if not x1.TARGET_FIRST_FRAME <= frame <= x1.TARGET_LAST_FRAME:
            rows.append(
                (
                    np.empty((0, 2), np.float32),
                    np.empty((0, 2), np.float32),
                    np.empty(0, np.int32),
                )
            )
            continue
        pose = poses[frame]
        current, counts = x1._voxel_centroids(x1._local_cloud(world_clouds[index], pose))
        supports = []
        for offset in SCAN_OFFSETS:
            support, _counts = x1._voxel_centroids(
                x1._local_cloud(world_clouds[index + offset], pose, margin_m=1.5)
            )
            supports.append(support)
        require(len(current) > 0 and all(len(row) > 0 for row in supports), f"x1b_empty_cloud:{frame}")
        displacement, detail = x1._optimize_frame(
            current, supports, scan_offsets=SCAN_OFFSETS, device=device
        )
        local_times = [
            frame_times[int(frames[position])]
            for position in range(index - 2, index + 3)
        ]
        one_step_s = float(np.median(np.diff(local_times)))
        require(one_step_s > 0.0, f"x1b_nonpositive_step:{frame}")
        positions, velocities, source_counts = x1._aggregate(
            current, displacement / one_step_s, counts
        )
        rows.append((positions, velocities, source_counts))
        diagnostics.append(
            {
                "frame": frame,
                "one_step_s": one_step_s,
                "output_cells": int(len(positions)),
                **detail,
            }
        )
    offsets = np.cumsum([0] + [len(row[0]) for row in rows], dtype=np.int64)
    arrays = {
        "frames": np.asarray(frames, dtype=np.int32),
        "frame_time_s": np.asarray([frame_times[int(frame)] for frame in frames], dtype=np.float64),
        "offsets": offsets,
        "forward_m": np.concatenate([row[0][:, 0] for row in rows]).astype(np.float32),
        "left_m": np.concatenate([row[0][:, 1] for row in rows]).astype(np.float32),
        "velocity_forward_mps": np.concatenate([row[1][:, 0] for row in rows]).astype(np.float32),
        "velocity_left_mps": np.concatenate([row[1][:, 1] for row in rows]).astype(np.float32),
        "component_id": np.concatenate([np.arange(len(row[0]), dtype=np.int32) for row in rows]),
        "source_point_count": np.concatenate([row[2] for row in rows]),
        "flow_support": np.concatenate([np.ones(len(row[0]), dtype=np.float32) for row in rows]),
    }
    atomic_npz(paths["ledger"], **arrays)
    manifest = {
        "schema": LEDGER_SCHEMA,
        "truth_blind": True,
        "oracle": True,
        "oracle_reason": "two future scans are consumed by the source optimizer",
        "sequence": x1.SEQUENCE,
        "motion_source": "symmetric five-scan explicit voxel scene flow oracle",
        "scan_offsets": list(SCAN_OFFSETS),
        "target_window": [x1.TARGET_FIRST_FRAME, x1.TARGET_LAST_FRAME],
        "source_config_inherited_from": str(Path(x1.__file__).resolve()),
        "backend": {
            "required": "cuda",
            "verified": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
        },
        "frozen_downstream": {
            "route_entry_geometry": "UNCHANGED_R7",
            "motion_bounds": "UNCHANGED_R7",
            "event_scorer": "UNCHANGED_R7",
        },
        "source": {
            "bag_sha256": sha256_file(args.bag.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "baseline_ledger_sha256": sha256_file(args.baseline_ledger.resolve(strict=True)),
        },
        "diagnostics": {
            "frames": diagnostics,
            "total_seconds": sum(float(row["seconds"]) for row in diagnostics),
            "output_cells": int(len(arrays["forward_m"])),
            "lidar": lidar,
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
    symmetric = x1._diagnose(
        ledger,
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=boxes,
        history=history,
        poses=poses,
    )
    causal_result = json.loads(args.causal_result.resolve(strict=True).read_text(encoding="utf-8"))
    causal = causal_result["source_only_comparison"]["causal_floxel"]
    gate = {
        "at_least_two_correct_frames": symmetric["correct_frames"] >= 2,
        "at_least_two_correct_route_entry_frames": (
            symmetric["correct_route_entry_frames"] >= 2
        ),
        "correct_frames_above_causal_adapter": (
            symmetric["correct_frames"] > int(causal["correct_frames"])
        ),
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X1B_SYMMETRIC_FLOXEL_ORACLE_HEADROOM_MET"
            if met
            else "DTR_X1B_SYMMETRIC_FLOXEL_ORACLE_HEADROOM_NOT_MET"
        ),
        "question": (
            "Does restoring the non-causal symmetric five-scan information pattern reveal "
            "Floxel-family source headroom after the past-only adapter missed its gate?"
        ),
        "causal_adapter": {
            "correct_frames": int(causal["correct_frames"]),
            "correct_route_entry_frames": int(causal["correct_route_entry_frames"]),
            "minimum_associated_error_mps": causal["minimum_associated_error_mps"],
        },
        "symmetric_oracle": symmetric,
        "gate": gate,
        "decision": {
            "headroom_met": met,
            "next": (
                "FLOXEL_TEACHER_HAS_HEADROOM_BUT_REQUIRES_CAUSAL_DISTILLATION"
                if met
                else "CLOSE_FLOXEL_SOURCE_FAMILY"
            ),
        },
        "claim_limits": [
            "Opened-truth single-event oracle diagnostic; not deployable performance evidence.",
            "Two future scans violate the online causal contract and cannot be used by BlindAssist at decision time.",
            "The implementation is an independent Floxels-inspired adapter, not official Floxels code or a benchmark reproduction.",
            "False-segment suppression remains unevaluated.",
        ],
        "sources": {
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "causal_result": str(args.causal_result.resolve(strict=True)),
            "causal_result_sha256": sha256_file(args.causal_result.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
    }
    write_json(paths["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    evidence = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    root = REPO / "artifacts.local" / "evidence" / "dtr-x1b" / "symmetric-floxel-oracle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "datasets"
        / "dtr-c31-jrdb-fresh-confirmation"
        / f"{x1.SEQUENCE}.bag",
    )
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "datasets"
        / "ustrf-canonical-observation-source-authority-data-pack-r0"
        / "jrdb_toolkit"
        / "calibration",
    )
    parser.add_argument(
        "--baseline-ledger",
        type=Path,
        default=evidence
        / "baseline-ledgers"
        / x1.SEQUENCE
        / "m1-pd.raw-point-direct-velocity.npz",
    )
    parser.add_argument(
        "--causal-result",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x1"
        / "causal-floxel-source-canary"
        / "result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"materialize", "run"}:
        manifest = materialize(args)
        print(
            json.dumps(
                {
                    "materialized": True,
                    "output_cells": manifest["diagnostics"]["output_cells"],
                    "seconds": manifest["diagnostics"]["total_seconds"],
                    "backend": manifest["backend"],
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
                    "causal_adapter": result["causal_adapter"],
                    "symmetric_oracle": {
                        key: result["symmetric_oracle"][key]
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
