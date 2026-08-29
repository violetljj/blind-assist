"""Run the lag-compensated Floxel source on the frozen X0 error slice.

The truth-blind materializer receives only the 35 post-outcome diagnostic frame
indices selected by X0.  For each output frame ``t`` it estimates a symmetric
five-scan Floxel field at ``t-2`` from scans ``t-4..t`` and transports it to
``t``.  Scoring then asks whether any cell can still enter the unchanged R7
route tube at each representative false frame.  The already sealed X1c
positive canary is carried forward without rerunning or tuning it.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x1c_lag_compensated_floxel_source as x1c  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import load_world_clouds  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    _entry_s,
    _rotate_world_velocity_to_ego,
    _world_to_ego_xy,
    atomic_npz,
)


SCHEMA = "blindassist-dtr-x2-floxel-error-slice-canary-v1"
LEDGER_SCHEMA = "blindassist-dtr-x2-floxel-error-slice-ledger-v1"
SOURCE_FAILURES = {"BAD_FLOW", "STATIC_PSEUDO_MOTION"}
MINIMUM_SUPPRESSION_RATE = 0.70


def _sequence_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    sequence_root = root / sequence
    return sequence_root / "lag-floxel.npz", sequence_root / "lag-floxel.json"


def _transport(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    reference_pose: Mapping[str, Any],
    output_pose: Mapping[str, Any],
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    yaw = float(reference_pose["yaw_rad"])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    world_position = np.column_stack(
        (
            float(reference_pose["x_m"])
            + cosine * positions[:, 0]
            - sine * positions[:, 1],
            float(reference_pose["y_m"])
            + sine * positions[:, 0]
            + cosine * positions[:, 1],
        )
    )
    world_velocity = x1c._ego_velocity_to_world(velocities, yaw)
    transported_world = world_position + world_velocity * delay_s
    return (
        _world_to_ego_xy(transported_world, dict(output_pose)).astype(np.float32),
        _rotate_world_velocity_to_ego(world_velocity, dict(output_pose)).astype(np.float32),
    )


def _selected_units(x0: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(x0["pdc_false_segments"]) + list(x0["c31_incremental_false_segments"])
    return [
        {
            "unit_id": str(row["segment_id"]),
            "sequence": str(row["sequence"]),
            "frame": int(row["diagnostic_frame"]),
            "primary_cause": str(row["primary_cause"]),
        }
        for row in rows
    ]


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    require(torch.cuda.is_available(), "x2_cuda_unavailable")
    device = torch.device("cuda:0")
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    units = _selected_units(x0)
    require(len(units) == 35, "x2_error_unit_count")
    by_sequence: dict[str, set[int]] = defaultdict(set)
    for row in units:
        by_sequence[row["sequence"]].add(int(row["frame"]))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sequence_manifests = []
    started = time.perf_counter()
    for sequence in sorted(by_sequence):
        output_frames = sorted(by_sequence[sequence])
        required_frames = sorted(
            {
                frame + offset
                for frame in output_frames
                for offset in (-4, -3, -2, -1, 0)
            }
        )
        baseline_npz = (
            args.baseline_root.resolve(strict=True)
            / sequence
            / "m1-pd.raw-point-direct-velocity.npz"
        )
        with np.load(baseline_npz, allow_pickle=False) as values:
            available = {
                int(frame): float(stamp)
                for frame, stamp in zip(values["frames"], values["frame_time_s"])
            }
        require(set(required_frames) <= set(available), f"x2_required_frames:{sequence}")
        timestamps = {frame: available[frame] for frame in required_frames}
        bag_path = args.bag_root.resolve(strict=True) / f"{sequence}.bag"
        frames, frame_times, poses, world_clouds, lidar = load_world_clouds(
            bag_path=bag_path,
            timestamps_path=args.timestamps.resolve(strict=True),
            calibration_dir=args.calibration_dir.resolve(strict=True),
            timestamps_override=timestamps,
        )
        cloud_by_frame = {
            int(frame): cloud for frame, cloud in zip(frames, world_clouds)
        }
        diagnostics = []
        output_rows = []
        for output_frame in output_frames:
            reference = output_frame - x1c.LAG_SCANS
            pose = poses[reference]
            current, counts = x1._voxel_centroids(
                x1._local_cloud(cloud_by_frame[reference], pose)
            )
            scan_offsets = (-2, -1, 1, 2)
            supports = []
            for offset in scan_offsets:
                support, _counts = x1._voxel_centroids(
                    x1._local_cloud(
                        cloud_by_frame[reference + offset], pose, margin_m=1.5
                    )
                )
                supports.append(support)
            require(
                len(current) > 0 and all(len(row) > 0 for row in supports),
                f"x2_empty_cloud:{sequence}:{output_frame}",
            )
            displacement, detail = x1._optimize_frame(
                current, supports, scan_offsets=scan_offsets, device=device
            )
            local_times = [frame_times[frame] for frame in range(output_frame - 4, output_frame + 1)]
            one_step_s = float(np.median(np.diff(local_times)))
            require(one_step_s > 0.0, f"x2_nonpositive_step:{sequence}:{output_frame}")
            positions, velocities, source_counts = x1._aggregate(
                current, displacement / one_step_s, counts
            )
            delay_s = frame_times[output_frame] - frame_times[reference]
            positions, velocities = _transport(
                positions,
                velocities,
                reference_pose=poses[reference],
                output_pose=poses[output_frame],
                delay_s=delay_s,
            )
            output_rows.append((positions, velocities, source_counts))
            diagnostics.append(
                {
                    "output_frame": output_frame,
                    "reference_frame": reference,
                    "delay_s": delay_s,
                    "one_step_s": one_step_s,
                    "output_cells": int(len(positions)),
                    **detail,
                }
            )
        offsets = np.cumsum(
            [0] + [len(row[0]) for row in output_rows], dtype=np.int64
        )
        arrays = {
            "frames": np.asarray(output_frames, dtype=np.int32),
            "frame_time_s": np.asarray([frame_times[frame] for frame in output_frames], dtype=np.float64),
            "offsets": offsets,
            "forward_m": np.concatenate([row[0][:, 0] for row in output_rows]).astype(np.float32),
            "left_m": np.concatenate([row[0][:, 1] for row in output_rows]).astype(np.float32),
            "velocity_forward_mps": np.concatenate([row[1][:, 0] for row in output_rows]).astype(np.float32),
            "velocity_left_mps": np.concatenate([row[1][:, 1] for row in output_rows]).astype(np.float32),
            "component_id": np.concatenate([np.arange(len(row[0]), dtype=np.int32) for row in output_rows]),
            "source_point_count": np.concatenate([row[2] for row in output_rows]),
            "flow_support": np.concatenate([np.ones(len(row[0]), dtype=np.float32) for row in output_rows]),
        }
        ledger_path, manifest_path = _sequence_paths(root, sequence)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(ledger_path, **arrays)
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "oracle": False,
            "sequence": sequence,
            "selection": "post-outcome X0 diagnostic frame indices only",
            "online_information_boundary": (
                "output frame t consumes reference t-2 estimated from scans t-4 through t"
            ),
            "output_frames": output_frames,
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
                "bag": str(bag_path),
                "bag_sha256": sha256_file(bag_path),
                "baseline_ledger_sha256": sha256_file(baseline_npz),
                "x0_result_sha256": sha256_file(x0_path),
            },
            "diagnostics": {
                "frames": diagnostics,
                "output_cells": int(len(arrays["forward_m"])),
                "lidar": lidar,
            },
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
        write_json(manifest_path, manifest)
        sequence_manifests.append(manifest)
    receipt = {
        "schema": "blindassist-dtr-x2-floxel-error-slice-materialization-v1",
        "truth_blind": True,
        "selection_post_outcome": True,
        "units": len(units),
        "sequences": len(sequence_manifests),
        "elapsed_s": time.perf_counter() - started,
        "backend": sequence_manifests[0]["backend"],
        "x0_result": str(x0_path),
        "x0_result_sha256": sha256_file(x0_path),
        "sequence_manifests": [
            {
                "sequence": row["sequence"],
                "manifest": str(_sequence_paths(root, row["sequence"])[1]),
                "manifest_sha256": sha256_file(_sequence_paths(root, row["sequence"])[1]),
            }
            for row in sequence_manifests
        ],
    }
    write_json(root / "materialization.json", receipt)
    return receipt


def _risk_cells(ledger: Mapping[str, np.ndarray], frame: int) -> int:
    return sum(
        _entry_s(
            cell["forward_m"],
            cell["left_m"],
            cell["velocity_forward_mps"],
            cell["velocity_left_mps"],
        )
        is not None
        for cell in x1._cells(ledger, frame)
    )


def score(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    units = _selected_units(x0)
    ledgers = {}
    manifest_hashes = {}
    for sequence in sorted({row["sequence"] for row in units}):
        ledger_path, manifest_path = _sequence_paths(root, sequence)
        ledgers[sequence] = x1._load_sealed(
            ledger_path, manifest_path, LEDGER_SCHEMA
        )
        manifest_hashes[sequence] = sha256_file(manifest_path)
    rows = []
    for unit in units:
        risk_cells = _risk_cells(ledgers[unit["sequence"]], int(unit["frame"]))
        rows.append({**unit, "lag_floxel_route_risk_cells": risk_cells, "suppressed": risk_cells == 0})
    source_rows = [row for row in rows if row["primary_cause"] in SOURCE_FAILURES]
    require(len(source_rows) == 34, "x2_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    required = math.ceil(MINIMUM_SUPPRESSION_RATE * len(source_rows))
    positive_path = args.positive_result.resolve(strict=True)
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    positive_gate = positive.get("status") == "DTR_X1C_LAG_COMPENSATED_FLOXEL_SOURCE_HEADROOM_MET"
    gate = {
        "positive_rear_route_headroom_carried_forward": positive_gate,
        "source_error_representative_suppression_at_least_70_percent": suppressed >= required,
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X2_FLOXEL_ERROR_SLICE_GATE_MET"
            if met
            else "DTR_X2_FLOXEL_ERROR_SLICE_GATE_NOT_MET"
        ),
        "question": (
            "Does the lag-compensated source retain the rear-route positive headroom while "
            "suppressing at least 70% of X0 source-error representative false frames?"
        ),
        "positive_canary": {
            "status": positive["status"],
            "correct_frames": positive["lag_compensated_source"]["correct_frames"],
            "correct_route_entry_frames": positive["lag_compensated_source"]["correct_route_entry_frames"],
            "median_information_delay_s": positive["delay_s"]["median"],
        },
        "false_error_slice": {
            "all_units": len(rows),
            "source_error_units": len(source_rows),
            "suppressed_source_error_units": suppressed,
            "retained_source_error_units": len(source_rows) - suppressed,
            "suppression_rate": suppressed / len(source_rows),
            "required_suppression_units": required,
            "by_cause": {
                cause: {
                    "units": sum(row["primary_cause"] == cause for row in source_rows),
                    "suppressed": sum(
                        row["primary_cause"] == cause and row["suppressed"]
                        for row in source_rows
                    ),
                }
                for cause in sorted(SOURCE_FAILURES)
            },
        },
        "gate": gate,
        "decision": {
            "headroom_met": met,
            "next": (
                "RUN_FULL_SIX_SEQUENCE_LAG_FLOXEL_SOURCE_COMPARISON"
                if met
                else "CLOSE_LAG_FLOXEL_SOURCE_BEFORE_FULL_REPLAY"
            ),
        },
        "units": rows,
        "sources": {
            "x0_result": str(x0_path),
            "x0_result_sha256": sha256_file(x0_path),
            "positive_result": str(positive_path),
            "positive_result_sha256": sha256_file(positive_path),
            "sequence_manifest_sha256": manifest_hashes,
        },
        "claim_limits": [
            "Post-outcome representative-frame Development slice; not full event-lifecycle performance or source-disjoint confirmation.",
            "Suppression is conservative frame-local absence of any route-entering source cell, not proof that a complete false segment disappears.",
            "The positive and false slices are already opened X0 evidence and cannot authorize a confirmation claim.",
            "Online information causality is established; real-time compute latency is not established.",
        ],
    }
    write_json(root / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    root = REPO / "artifacts.local" / "evidence" / "dtr-x2" / "floxel-error-slice"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--x0-result",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x0"
        / "motion-source-attribution"
        / "result.json",
    )
    parser.add_argument(
        "--positive-result",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x1c"
        / "lag-compensated-floxel-source"
        / "result.json",
    )
    parser.add_argument(
        "--bag-root",
        type=Path,
        default=REPO / "artifacts.local" / "datasets" / "dtr-c31-jrdb-fresh-confirmation",
    )
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
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
    parser.add_argument("--baseline-root", type=Path, default=c31 / "baseline-ledgers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"materialize", "run"}:
        receipt = materialize(args)
        print(
            json.dumps(
                {
                    "materialized": True,
                    "units": receipt["units"],
                    "sequences": receipt["sequences"],
                    "elapsed_s": receipt["elapsed_s"],
                    "backend": receipt["backend"],
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
                    "positive_canary": result["positive_canary"],
                    "false_error_slice": result["false_error_slice"],
                    "gate": result["gate"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
