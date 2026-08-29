"""Materialize truth-blind empty X3 checkpoints when source support is absent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x1c_lag_compensated_floxel_source as x1c  # noqa: E402
import dtr_x3_full_lag_floxel_replay as x3  # noqa: E402
from dtr_c1_global_obb_cohort_admission import sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import load_world_clouds  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-c31"
        / "fresh-confirmation"
        / "baseline-predictions.json",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "datasets"
        / "dtr-r0-jrdb-rgb-bridge-v1"
        / "train_timestamps.zip",
    )
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
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    freeze = root / "freeze.json"
    baseline_path = args.baseline_predictions.resolve(strict=True)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = {str(row["sequence"]): row for row in baseline["sequences"]}
    row = rows[args.sequence]
    baseline_ledger = Path(row["sources"]["ledgers"]["M1_PD_GLOBAL"]["ledger"]).resolve(strict=True)
    with np.load(baseline_ledger, allow_pickle=False) as values:
        frames = [int(value) for value in values["frames"]]
        timestamps = {
            int(frame): float(stamp)
            for frame, stamp in zip(values["frames"], values["frame_time_s"])
        }
    bag_path = Path(row["sources"]["bag"]).resolve(strict=True)
    loaded_frames, frame_times, poses, world_clouds, _lidar = load_world_clouds(
        bag_path=bag_path,
        timestamps_path=args.timestamps.resolve(strict=True),
        calibration_dir=args.calibration_dir.resolve(strict=True),
        timestamps_override=timestamps,
    )
    if loaded_frames.tolist() != frames:
        raise RuntimeError(f"x3_amendment_loaded_frames:{args.sequence}")
    cloud_by_frame = {int(frame): cloud for frame, cloud in zip(loaded_frames, world_clouds)}
    checkpoints = x3._paths(root, args.sequence)["checkpoints"]
    checkpoints.mkdir(parents=True, exist_ok=True)
    empty_frames = []
    for output_index, output_frame in enumerate(frames[4:], start=4):
        reference = output_frame - x1c.LAG_SCANS
        pose = poses[reference]
        current, _counts = x1._voxel_centroids(x1._local_cloud(cloud_by_frame[reference], pose))
        supports = [
            x1._voxel_centroids(
                x1._local_cloud(cloud_by_frame[reference + offset], pose, margin_m=1.5)
            )[0]
            for offset in (-2, -1, 1, 2)
        ]
        if len(current) > 0 and all(len(points) > 0 for points in supports):
            continue
        empty_frames.append(output_frame)
        delay_s = frame_times[output_frame] - frame_times[reference]
        atomic_npz(
            checkpoints / f"{output_frame:06d}.npz",
            output_frame=np.asarray([output_frame], dtype=np.int32),
            forward_m=np.empty(0, np.float32),
            left_m=np.empty(0, np.float32),
            velocity_forward_mps=np.empty(0, np.float32),
            velocity_left_mps=np.empty(0, np.float32),
            source_point_count=np.empty(0, np.int32),
            optimization_s=np.asarray([0.0], dtype=np.float64),
            delay_s=np.asarray([delay_s], dtype=np.float64),
        )
    receipt = {
        "schema": "blindassist-dtr-x3-empty-support-amendment-v1",
        "status": "EMPTY_SUPPORT_FAIL_CLOSED",
        "truth_blind": True,
        "sequence": args.sequence,
        "empty_frames": empty_frames,
        "empty_frame_count": len(empty_frames),
        "semantics": "No complete five-scan source support yields zero dynamic cells.",
        "source": {
            "freeze_sha256": sha256_file(freeze),
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "bag_sha256": sha256_file(bag_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    receipt_path = root / "sequences" / args.sequence / "empty-support-amendment.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
