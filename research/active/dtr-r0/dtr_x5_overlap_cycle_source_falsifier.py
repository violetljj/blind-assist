"""Falsify a causal overlap-cycle local-flow source from sealed X3 ledgers.

X3 retained local motion and reached 6/6 Development recall, but its dense
per-cell freedom also produced 94 false segments.  X4 made an entire spatial
DBSCAN component rigid: it suppressed 31/34 opened source-error frames but
lost every positive motion frame.  This diagnostic tests exactly one source
successor between those extremes: retain X3 local cells only when the previous
overlapping five-scan window predicts a reciprocal space-and-velocity match in
the current window.

At wall-clock frame t, both the t-1 and t lagged ledgers are already available,
so the check is causal.  It is source correspondence, not alert persistence,
fusion, or scorer/lifecycle tuning.  Matching uses the frozen BEV-cell diagonal
and the already frozen flow-identity tolerance once; there is no sweep.  The
materializer reads only sealed X3 ledgers, their sealed timestamps, public bag
poses, and the already opened X0 frame roster.  Labels are opened only by
``score`` after the diagnostic ledgers are sealed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x1b_symmetric_floxel_oracle as x1b  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import _box_history, load_native_boxes  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    _causal_pose,
    atomic_npz,
)
from jrdb_rgb_bridge import read_bag_pose_and_rgb  # noqa: E402


SCHEMA = "blindassist-dtr-x5-overlap-cycle-source-falsifier-v1"
LEDGER_SCHEMA = "blindassist-dtr-x5-overlap-cycle-source-ledger-v1"
X3_LEDGER_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-ledger-v1"
POSITION_TOLERANCE_M = FROZEN_FLOW_CONFIG.voxel_size_m * math.sqrt(2.0)
VELOCITY_TOLERANCE_MPS = x1.FLOW_ERROR_LIMIT_MPS


def _sequence_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "overlap-cycle.npz", base / "overlap-cycle.json"


def _x3_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _frame_arrays(
    ledger: Mapping[str, np.ndarray], frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matches = np.flatnonzero(ledger["frames"] == int(frame))
    require(len(matches) == 1, f"x5_frame:{frame}")
    index = int(matches[0])
    start = int(ledger["offsets"][index])
    stop = int(ledger["offsets"][index + 1])
    positions = np.column_stack(
        (ledger["forward_m"][start:stop], ledger["left_m"][start:stop])
    ).astype(np.float64)
    velocities = np.column_stack(
        (
            ledger["velocity_forward_mps"][start:stop],
            ledger["velocity_left_mps"][start:stop],
        )
    ).astype(np.float64)
    counts = ledger["source_point_count"][start:stop].astype(np.int32)
    support = ledger["flow_support"][start:stop].astype(np.float32)
    return positions, velocities, counts, support


def _ego_to_world(
    positions: np.ndarray,
    velocities: np.ndarray,
    pose: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    yaw = float(pose["yaw_rad"])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    world_positions = np.column_stack(
        (
            float(pose["x_m"]) + cosine * positions[:, 0] - sine * positions[:, 1],
            float(pose["y_m"]) + sine * positions[:, 0] + cosine * positions[:, 1],
        )
    )
    world_velocities = np.column_stack(
        (
            cosine * velocities[:, 0] - sine * velocities[:, 1],
            sine * velocities[:, 0] + cosine * velocities[:, 1],
        )
    )
    return world_positions, world_velocities


def _reciprocal_current_indices(
    previous_positions: np.ndarray,
    previous_velocities: np.ndarray,
    current_positions: np.ndarray,
    current_velocities: np.ndarray,
    *,
    delta_s: float,
) -> np.ndarray:
    """Return current cells reciprocally explained by the preceding window."""
    from scipy.spatial import cKDTree

    if not len(previous_positions) or not len(current_positions):
        return np.empty(0, dtype=np.int64)
    predicted = previous_positions + previous_velocities * float(delta_s)
    current_tree = cKDTree(current_positions)
    predicted_tree = cKDTree(predicted)
    forward_distance, forward_index = current_tree.query(predicted, k=1, workers=1)
    _reverse_distance, reverse_index = predicted_tree.query(
        current_positions, k=1, workers=1
    )
    previous_index = np.arange(len(predicted), dtype=np.int64)
    reciprocal = reverse_index[forward_index] == previous_index
    velocity_error = np.linalg.norm(
        previous_velocities - current_velocities[forward_index], axis=1
    )
    keep = (
        reciprocal
        & (forward_distance <= POSITION_TOLERANCE_M + 1e-12)
        & (velocity_error <= VELOCITY_TOLERANCE_MPS + 1e-12)
    )
    return np.unique(forward_index[keep]).astype(np.int64)


def _pack_rows(
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    offsets = np.cumsum([0] + [len(row[0]) for row in rows], dtype=np.int64)
    return {
        "frames": np.asarray(frames, dtype=np.int32),
        "frame_time_s": np.asarray(
            [timestamps[int(frame)] for frame in frames], dtype=np.float64
        ),
        "offsets": offsets,
        "forward_m": np.concatenate([row[0][:, 0] for row in rows]).astype(np.float32),
        "left_m": np.concatenate([row[0][:, 1] for row in rows]).astype(np.float32),
        "velocity_forward_mps": np.concatenate([row[1][:, 0] for row in rows]).astype(
            np.float32
        ),
        "velocity_left_mps": np.concatenate([row[1][:, 1] for row in rows]).astype(
            np.float32
        ),
        "component_id": np.concatenate(
            [np.arange(len(row[0]), dtype=np.int32) for row in rows]
        ),
        "source_point_count": np.concatenate([row[2] for row in rows]).astype(np.int32),
        "flow_support": np.concatenate([row[3] for row in rows]).astype(np.float32),
    }


def _empty_row() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), np.float64),
        np.empty((0, 2), np.float64),
        np.empty(0, np.int32),
        np.empty(0, np.float32),
    )


def _selected_frames(x0: Mapping[str, Any]) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = defaultdict(set)
    for unit in x2._selected_units(x0):
        selected[str(unit["sequence"])].add(int(unit["frame"]))
    selected[x1.SEQUENCE].update(
        range(x1b.FIRST_SOURCE_FRAME, x1b.LAST_SOURCE_FRAME + 1)
    )
    return selected


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    selected = _selected_frames(x0)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for sequence in sorted(selected):
        source_path, source_manifest_path = _x3_paths(
            args.x3_root.resolve(strict=True), sequence
        )
        source = x1._load_sealed(
            source_path, source_manifest_path, X3_LEDGER_SCHEMA
        )
        source_frames = [int(frame) for frame in source["frames"]]
        timestamp_by_frame = {
            int(frame): float(stamp)
            for frame, stamp in zip(source["frames"], source["frame_time_s"])
        }
        output_frames = sorted(selected[sequence])
        require(
            set(output_frames) | {frame - 1 for frame in output_frames}
            <= set(source_frames),
            f"x5_source_frames:{sequence}",
        )
        bag_path = args.bag_root.resolve(strict=True) / f"{sequence}.bag"
        pose_samples, _rgb, pose_authority = read_bag_pose_and_rgb(bag_path)
        pose_frames = sorted(set(output_frames) | {frame - 1 for frame in output_frames})
        poses = {
            frame: _causal_pose(
                pose_samples, round(timestamp_by_frame[frame] * 1e9)
            )
            for frame in pose_frames
        }
        rows = []
        diagnostics = []
        for frame in output_frames:
            previous = frame - 1
            previous_row = _frame_arrays(source, previous)
            current_row = _frame_arrays(source, frame)
            previous_world = _ego_to_world(previous_row[0], previous_row[1], poses[previous])
            current_world = _ego_to_world(current_row[0], current_row[1], poses[frame])
            delta_s = timestamp_by_frame[frame] - timestamp_by_frame[previous]
            require(delta_s > 0.0, f"x5_delta:{sequence}:{frame}")
            keep = _reciprocal_current_indices(
                previous_world[0],
                previous_world[1],
                current_world[0],
                current_world[1],
                delta_s=delta_s,
            )
            rows.append(
                (
                    current_row[0][keep],
                    current_row[1][keep],
                    current_row[2][keep],
                    current_row[3][keep],
                )
            )
            diagnostics.append(
                {
                    "frame": frame,
                    "previous_frame": previous,
                    "delta_s": delta_s,
                    "input_cells": int(len(current_row[0])),
                    "reciprocal_cells": int(len(keep)),
                }
            )
        arrays = _pack_rows(output_frames, timestamp_by_frame, rows)
        ledger_path, manifest_path = _sequence_paths(root, sequence)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(ledger_path, **arrays)
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "selection_post_outcome": True,
            "sequence": sequence,
            "motion_source_proxy": (
                "causal reciprocal overlap-window local space-velocity correspondence"
            ),
            "online_information_boundary": (
                "frame t uses only sealed lagged source windows available at t-1 and t"
            ),
            "fixed_correspondence": {
                "position_tolerance_m": POSITION_TOLERANCE_M,
                "position_basis": "ONE_FROZEN_BEV_CELL_DIAGONAL",
                "velocity_tolerance_mps": VELOCITY_TOLERANCE_MPS,
                "velocity_basis": "FROZEN_FLOW_IDENTITY_TOLERANCE",
                "nearest_rule": "RECIPROCAL_ONE_TO_ONE",
                "tree_workers": 1,
            },
            "frozen_downstream": {
                "motion_bounds": "UNCHANGED_X3",
                "route_entry_geometry": "UNCHANGED_R7",
                "event_scorer": "UNCHANGED_X1C_X2",
                "alert_lifecycle": "NOT_USED_BY_THIS_FRAME_LOCAL_FALSIFIER",
            },
            "source": {
                "x3_ledger": str(source_path),
                "x3_ledger_sha256": sha256_file(source_path),
                "x3_manifest": str(source_manifest_path),
                "x3_manifest_sha256": sha256_file(source_manifest_path),
                "bag": str(bag_path),
                "bag_sha256": sha256_file(bag_path),
                "bag_pose_authority": pose_authority,
                "x0_result_sha256": sha256_file(x0_path),
            },
            "diagnostics": {
                "frames": diagnostics,
                "input_cells": sum(row["input_cells"] for row in diagnostics),
                "reciprocal_cells": sum(
                    row["reciprocal_cells"] for row in diagnostics
                ),
            },
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
        write_json(manifest_path, manifest)
        receipts.append(manifest)
    receipt = {
        "schema": "blindassist-dtr-x5-overlap-cycle-materialization-v1",
        "truth_blind": True,
        "selection_post_outcome": True,
        "sequences": len(receipts),
        "frames": sum(len(row["diagnostics"]["frames"]) for row in receipts),
        "input_cells": sum(row["diagnostics"]["input_cells"] for row in receipts),
        "reciprocal_cells": sum(
            row["diagnostics"]["reciprocal_cells"] for row in receipts
        ),
        "sequence_manifests": {
            row["sequence"]: sha256_file(_sequence_paths(root, row["sequence"])[1])
            for row in receipts
        },
    }
    write_json(root / "materialization.json", receipt)
    return receipt


def _load_ledgers(
    args: argparse.Namespace, sequences: Sequence[str]
) -> dict[str, dict[str, np.ndarray]]:
    ledgers = {}
    for sequence in sorted(set(sequences)):
        ledger_path, manifest_path = _sequence_paths(args.root.resolve(strict=True), sequence)
        ledgers[sequence] = x1._load_sealed(
            ledger_path, manifest_path, LEDGER_SCHEMA
        )
    return ledgers


def _score_positive(
    args: argparse.Namespace, ledger: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    frames = [int(frame) for frame in ledger["frames"]]
    timestamps = {
        int(frame): float(stamp)
        for frame, stamp in zip(ledger["frames"], ledger["frame_time_s"])
    }
    bag = args.bag_root.resolve(strict=True) / f"{x1.SEQUENCE}.bag"
    pose_samples, _rgb, _authority = read_bag_pose_and_rgb(bag)
    poses = {
        frame: _causal_pose(pose_samples, round(timestamps[frame] * 1e9))
        for frame in frames
    }
    boxes = load_native_boxes(
        args.labels.resolve(strict=True), timestamps, poses, sequence=x1.SEQUENCE
    )
    return x1._diagnose(
        ledger,
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=boxes,
        history=_box_history(boxes),
        poses=poses,
    )


def score(args: argparse.Namespace) -> dict[str, Any]:
    x0 = json.loads(args.x0_result.resolve(strict=True).read_text(encoding="utf-8"))
    units = x2._selected_units(x0)
    sequences = [str(unit["sequence"]) for unit in units] + [x1.SEQUENCE]
    ledgers = _load_ledgers(args, sequences)
    positive = _score_positive(args, ledgers[x1.SEQUENCE])
    unit_rows = []
    for unit in units:
        risk_cells = x2._risk_cells(
            ledgers[str(unit["sequence"])], int(unit["frame"])
        )
        unit_rows.append(
            {
                **unit,
                "overlap_cycle_route_risk_cells": risk_cells,
                "suppressed": risk_cells == 0,
            }
        )
    source_rows = [
        row for row in unit_rows if str(row["primary_cause"]) in x2.SOURCE_FAILURES
    ]
    require(len(source_rows) == 34, "x5_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    required = math.ceil(x2.MINIMUM_SUPPRESSION_RATE * len(source_rows))
    gate = {
        "positive_correct_frames_at_least_two": positive["correct_frames"] >= 2,
        "positive_correct_route_frames_at_least_two": (
            positive["correct_route_entry_frames"] >= 2
        ),
        "source_error_suppression_at_least_24_of_34": suppressed >= required,
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_MET"
            if met
            else "DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_NOT_MET"
        ),
        "question": (
            "Do X3's true local motions survive a causal reciprocal overlap-window "
            "correspondence while opened source-error frames disappear?"
        ),
        "positive": positive,
        "error_slice": {
            "all_units": len(unit_rows),
            "source_error_units": len(source_rows),
            "suppressed_source_error_units": suppressed,
            "retained_source_error_units": len(source_rows) - suppressed,
            "suppression_rate": suppressed / len(source_rows),
            "required_suppression_units": required,
        },
        "units": unit_rows,
        "gate": gate,
        "decision": {
            "mechanism_headroom": met,
            "next": (
                "IMPLEMENT_RAW_FIVE_SCAN_OVERLAP_CYCLE_SOURCE_CANARY"
                if met
                else "CLOSE_OVERLAP_CYCLE_SUCCESSOR_WITHOUT_THRESHOLD_SWEEP"
            ),
        },
        "claim_limits": [
            "Post-outcome diagnostic using opened Development frames; not confirmation.",
            "This filters sealed X3 cells as a mechanism proxy; it is not a new raw-source performance result.",
            "The 35-unit outcome is frame-local suppression, not full event-lifecycle scoring.",
            "A pass authorizes one raw-source canary only, not fusion with X3 or default promotion.",
        ],
        "sources": {
            "x3_result": str(args.x3_result.resolve(strict=True)),
            "x3_result_sha256": sha256_file(args.x3_result.resolve(strict=True)),
            "x4_result": str(args.x4_result.resolve(strict=True)),
            "x4_result_sha256": sha256_file(args.x4_result.resolve(strict=True)),
            "x0_result_sha256": sha256_file(args.x0_result.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
    }
    write_json(args.root.resolve(strict=True) / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    x3_root = (
        REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x3"
        / "full-lag-floxel-replay-mp"
    )
    x4_root = (
        REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x4"
        / "deterministic-cluster-vote-repeat3-20260829"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x5"
        / "overlap-cycle-source-falsifier-20260829",
    )
    parser.add_argument("--x3-root", type=Path, default=x3_root)
    parser.add_argument("--x3-result", type=Path, default=x3_root / "result.json")
    parser.add_argument(
        "--x4-result", type=Path, default=x4_root / "run-01" / "result.json"
    )
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
        "--bag-root",
        type=Path,
        default=REPO / "artifacts.local" / "datasets" / "dtr-c31-jrdb-fresh-confirmation",
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] | None = None
    if args.mode in {"materialize", "run"}:
        payload = materialize(args)
    if args.mode in {"score", "run"}:
        payload = score(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
