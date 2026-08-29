"""Falsify a causal static-world persistence anchor on sealed X3 cells.

X3's lagged Floxel source recovered all six Development contacts but most of
its false segments were static pseudo-motion.  This diagnostic changes only
the information available to the source: an X3 cell is removed when the same
world location is occupied both recently and across enough elapsed time that a
cell moving at the frozen dynamic-speed floor should have crossed one frozen
BEV-cell diagonal.  Ego poses come from the public bag and all history precedes
the scored frame.  Velocity, route geometry, and scorers remain unchanged.

The materializer is truth blind.  Labels are opened only by ``score`` after the
derived ledgers and manifests have been hash sealed.  This is one fixed
source-only falsifier, not a threshold search.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import load_world_clouds  # noqa: E402
from dtr_r7_occupancy_flow_canary import FROZEN_FLOW_CONFIG, atomic_npz  # noqa: E402


SCHEMA = "blindassist-dtr-x6-static-world-persistence-falsifier-v1"
LEDGER_SCHEMA = "blindassist-dtr-x6-static-world-persistence-ledger-v1"
X3_LEDGER_SCHEMA = x5.X3_LEDGER_SCHEMA

# A truly dynamic cell at the frozen minimum speed should move farther than a
# frozen BEV-cell diagonal over this span.  The recent observation bound is the
# already-frozen raw-flow history maximum; their sum bounds stale-map memory.
POSITION_TOLERANCE_M = FROZEN_FLOW_CONFIG.voxel_size_m * math.sqrt(2.0)
MINIMUM_PERSISTENCE_SPAN_S = (
    POSITION_TOLERANCE_M / FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps
)
RECENT_MATCH_MAX_AGE_S = FROZEN_FLOW_CONFIG.history_max_s
MAXIMUM_ANCHOR_AGE_S = MINIMUM_PERSISTENCE_SPAN_S + RECENT_MATCH_MAX_AGE_S
SOURCE_COMPUTE_BUDGET_S = 0.06961


def _sequence_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "static-anchor.npz", base / "static-anchor.json"


def _selected_frames(x0: Mapping[str, Any]) -> dict[str, set[int]]:
    return x5._selected_frames(x0)


def _within_any(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Whether each query point has a fixed-radius reference neighbour."""
    if not len(query) or not len(reference):
        return np.zeros(len(query), dtype=bool)
    from scipy.spatial import cKDTree

    distance, _index = cKDTree(reference).query(query, k=1, workers=1)
    return distance <= POSITION_TOLERANCE_M + 1e-12


def _history_raw_world_positions(
    raw_world_xy: Mapping[int, np.ndarray],
    frames: Sequence[int],
) -> np.ndarray:
    chunks = [raw_world_xy[int(frame)] for frame in frames]
    nonempty = [chunk for chunk in chunks if len(chunk)]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 2), np.float64)


def _static_mask(
    current_world: np.ndarray,
    recent_world: np.ndarray,
    spanning_world: np.ndarray,
) -> np.ndarray:
    # Two independent past observations are required: one recent enough to
    # establish continued occupancy and one old enough to reject motion at the
    # frozen source's minimum dynamic speed.
    return _within_any(current_world, recent_world) & _within_any(
        current_world, spanning_world
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    selected = _selected_frames(x0)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = []

    for sequence in sorted(selected):
        source_path, source_manifest_path = x5._x3_paths(
            args.x3_root.resolve(strict=True), sequence
        )
        source = x1._load_sealed(source_path, source_manifest_path, X3_LEDGER_SCHEMA)
        source_frames = [int(frame) for frame in source["frames"]]
        timestamp_by_frame = {
            int(frame): float(stamp)
            for frame, stamp in zip(source["frames"], source["frame_time_s"])
        }
        output_frames = sorted(selected[sequence])
        require(set(output_frames) <= set(source_frames), f"x6_source_frames:{sequence}")

        # Load raw LiDAR only for output frames and the bounded causal history
        # they can actually consume.  X3 contributes candidate cells only; the
        # anchor occupancy is flow-independent raw bag evidence.
        history_by_frame: dict[int, tuple[list[int], list[int]]] = {}
        required_pose_frames = set(output_frames)
        for frame in output_frames:
            now = timestamp_by_frame[frame]
            recent = [
                candidate
                for candidate in source_frames
                if 0.0 < now - timestamp_by_frame[candidate] <= RECENT_MATCH_MAX_AGE_S + 1e-12
            ]
            spanning = [
                candidate
                for candidate in source_frames
                if MINIMUM_PERSISTENCE_SPAN_S - 1e-12
                <= now - timestamp_by_frame[candidate]
                <= MAXIMUM_ANCHOR_AGE_S + 1e-12
            ]
            history_by_frame[frame] = (recent, spanning)
            required_pose_frames.update(recent)
            required_pose_frames.update(spanning)

        bag_path = args.bag_root.resolve(strict=True) / f"{sequence}.bag"
        raw_frames, _raw_timestamps, poses, world_clouds, lidar = load_world_clouds(
            bag_path=bag_path,
            timestamps_path=args.timestamps.resolve(strict=True),
            calibration_dir=args.calibration_dir.resolve(strict=True),
            timestamps_override={
                frame: timestamp_by_frame[frame] for frame in sorted(required_pose_frames)
            },
        )
        require(
            raw_frames.tolist() == sorted(required_pose_frames),
            f"x6_raw_frames:{sequence}",
        )
        raw_world_xy = {}
        for frame_value, world_cloud in zip(raw_frames, world_clouds):
            frame = int(frame_value)
            local = x1._local_cloud(world_cloud, poses[frame])
            raw_world_xy[frame] = x5._ego_to_world(
                local[:, :2], np.empty((len(local), 2), np.float64), poses[frame]
            )[0]

        rows = []
        diagnostics = []
        for frame in output_frames:
            started = time.perf_counter()
            current = x5._frame_arrays(source, frame)
            current_world = x5._ego_to_world(current[0], current[1], poses[frame])[0]
            recent_frames, spanning_frames = history_by_frame[frame]
            recent_world = _history_raw_world_positions(raw_world_xy, recent_frames)
            spanning_world = _history_raw_world_positions(raw_world_xy, spanning_frames)
            static = _static_mask(current_world, recent_world, spanning_world)
            keep = np.flatnonzero(~static)
            seconds = time.perf_counter() - started
            rows.append(
                (
                    current[0][keep],
                    current[1][keep],
                    current[2][keep],
                    current[3][keep],
                )
            )
            diagnostics.append(
                {
                    "frame": frame,
                    "input_cells": int(len(current[0])),
                    "static_cells_removed": int(np.count_nonzero(static)),
                    "retained_cells": int(len(keep)),
                    "seconds": seconds,
                    "recent_history_frames": recent_frames,
                    "spanning_history_frames": spanning_frames,
                }
            )

        arrays = x5._pack_rows(output_frames, timestamp_by_frame, rows)
        ledger_path, manifest_path = _sequence_paths(root, sequence)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(ledger_path, **arrays)
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "selection_post_outcome": True,
            "sequence": sequence,
            "motion_source_proxy": "x3 lag-floxel minus causal static-world persistence anchor",
            "online_information_boundary": "frame t uses only sealed X3 cells and bag poses strictly before or at t",
            "fixed_anchor": {
                "position_tolerance_m": POSITION_TOLERANCE_M,
                "position_basis": "ONE_FROZEN_BEV_CELL_DIAGONAL",
                "minimum_persistence_span_s": MINIMUM_PERSISTENCE_SPAN_S,
                "span_basis": "CELL_DIAGONAL_DIVIDED_BY_FROZEN_MINIMUM_DYNAMIC_SPEED",
                "recent_match_max_age_s": RECENT_MATCH_MAX_AGE_S,
                "recent_basis": "FROZEN_FLOW_HISTORY_MAXIMUM",
                "maximum_anchor_age_s": MAXIMUM_ANCHOR_AGE_S,
                "maximum_age_basis": "PERSISTENCE_SPAN_PLUS_FROZEN_HISTORY_MAXIMUM",
                "required_past_matches": "ONE_RECENT_AND_ONE_SPANNING",
                "tree_workers": 1,
            },
            "frozen_downstream": {
                "cell_velocity": "UNCHANGED_X3",
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
                "bag_pose_authority": lidar["bag_authority"],
                "timestamps": str(args.timestamps.resolve(strict=True)),
                "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
                "calibration_lidars_sha256": sha256_file(
                    args.calibration_dir.resolve(strict=True) / "lidars.yaml"
                ),
                "selected_raw_lidar_payload_sha256": lidar[
                    "selected_lidar_payload_sha256"
                ],
                "x0_result_sha256": sha256_file(x0_path),
            },
            "diagnostics": {
                "frames": diagnostics,
                "input_cells": sum(row["input_cells"] for row in diagnostics),
                "static_cells_removed": sum(row["static_cells_removed"] for row in diagnostics),
                "retained_cells": sum(row["retained_cells"] for row in diagnostics),
            },
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
        write_json(manifest_path, manifest)
        receipts.append(manifest)

    receipt = {
        "schema": "blindassist-dtr-x6-static-world-persistence-materialization-v1",
        "truth_blind": True,
        "selection_post_outcome": True,
        "sequences": len(receipts),
        "frames": sum(len(row["diagnostics"]["frames"]) for row in receipts),
        "input_cells": sum(row["diagnostics"]["input_cells"] for row in receipts),
        "static_cells_removed": sum(row["diagnostics"]["static_cells_removed"] for row in receipts),
        "retained_cells": sum(row["diagnostics"]["retained_cells"] for row in receipts),
        "source_compute_p95_s": float(
            np.quantile(
                np.asarray(
                    [
                        frame["seconds"]
                        for row in receipts
                        for frame in row["diagnostics"]["frames"]
                    ],
                    dtype=np.float64,
                ),
                0.95,
                method="higher",
            )
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
        ledgers[sequence] = x1._load_sealed(ledger_path, manifest_path, LEDGER_SCHEMA)
    return ledgers


def score(args: argparse.Namespace) -> dict[str, Any]:
    x0 = json.loads(args.x0_result.resolve(strict=True).read_text(encoding="utf-8"))
    units = x2._selected_units(x0)
    sequences = [str(unit["sequence"]) for unit in units] + [x1.SEQUENCE]
    ledgers = _load_ledgers(args, sequences)
    positive = x5._score_positive(args, ledgers[x1.SEQUENCE])
    unit_rows = []
    for unit in units:
        risk_cells = x2._risk_cells(ledgers[str(unit["sequence"])], int(unit["frame"]))
        unit_rows.append(
            {
                **unit,
                "static_anchor_route_risk_cells": risk_cells,
                "suppressed": risk_cells == 0,
            }
        )
    source_rows = [
        row for row in unit_rows if str(row["primary_cause"]) in x2.SOURCE_FAILURES
    ]
    require(len(source_rows) == 34, "x6_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    required = math.ceil(x2.MINIMUM_SUPPRESSION_RATE * len(source_rows))
    materialization = json.loads(
        (args.root.resolve(strict=True) / "materialization.json").read_text(
            encoding="utf-8"
        )
    )
    source_compute_p95_s = float(materialization["source_compute_p95_s"])
    gate = {
        "positive_correct_frames_at_least_two": positive["correct_frames"] >= 2,
        "positive_correct_route_frames_at_least_two": positive["correct_route_entry_frames"] >= 2,
        "source_error_suppression_at_least_24_of_34": suppressed >= required,
        "source_compute_p95_within_one_scan_period": (
            source_compute_p95_s <= SOURCE_COMPUTE_BUDGET_S
        ),
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_MET"
            if met
            else "DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_NOT_MET"
        ),
        "question": "Can a causal world-frame persistence anchor remove static pseudo-motion while preserving X3 true route motion?",
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
        "runtime": {
            "frames": int(materialization["frames"]),
            "source_compute_p95_s": source_compute_p95_s,
            "median_observed_scan_period_s": SOURCE_COMPUTE_BUDGET_S,
            "p95_within_one_observed_scan_period": (
                source_compute_p95_s <= SOURCE_COMPUTE_BUDGET_S
            ),
            "boundary": "raw bag decoding is excluded; per-frame anchor construction and queries are included",
        },
        "decision": {
            "mechanism_headroom": met,
            "next": (
                "IMPLEMENT_RAW_STATIC_WORLD_ANCHORED_SOURCE_CANARY"
                if met
                else "CLOSE_X3_CELL_PERSISTENCE_ANCHOR_WITHOUT_THRESHOLD_SWEEP"
            ),
        },
        "claim_limits": [
            "Post-outcome diagnostic using opened Development frames; not confirmation.",
            "This filters sealed X3 cells as a mechanism proxy; it is not a new raw-source performance result.",
            "The 35-unit outcome is frame-local suppression, not full event-lifecycle scoring.",
            "A pass authorizes one raw-source canary only, not default promotion.",
        ],
        "sources": {
            "x3_result": str(args.x3_result.resolve(strict=True)),
            "x3_result_sha256": sha256_file(args.x3_result.resolve(strict=True)),
            "x0_result_sha256": sha256_file(args.x0_result.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
    }
    write_json(args.root.resolve(strict=True) / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    x3_root = REPO / "artifacts.local" / "evidence" / "dtr-x3" / "full-lag-floxel-replay-mp"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "artifacts.local" / "evidence" / "dtr-x6" / "static-world-persistence-falsifier-20260829",
    )
    parser.add_argument("--x3-root", type=Path, default=x3_root)
    parser.add_argument("--x3-result", type=Path, default=x3_root / "result.json")
    parser.add_argument(
        "--x0-result",
        type=Path,
        default=REPO / "artifacts.local" / "evidence" / "dtr-x0" / "motion-source-attribution" / "result.json",
    )
    parser.add_argument(
        "--bag-root",
        type=Path,
        default=REPO / "artifacts.local" / "datasets" / "dtr-c31-jrdb-fresh-confirmation",
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
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
