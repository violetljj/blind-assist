"""Run the frozen X6 static-world anchor over the full X3 Development replay.

Each sequence worker reads its sealed X3 lag-Floxel ledger and the matching raw
bag LiDAR/ego poses.  X3 cells are removed only when flow-independent raw
occupancy persists at the same world location under the constants frozen by
X6.  Cell velocity, motion bounds, route geometry, lifecycle, and the full X3
scorer are unchanged.  Materialization and predictions are truth blind; native
OBB labels are opened only by ``score``.

Use ``prepare`` once, run one ``materialize --sequence`` process per sequence,
then call ``assemble``, ``predict``, and ``score`` in order.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x3_full_lag_floxel_replay as x3  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x6_static_world_persistence_falsifier as x6  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import load_world_clouds  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402


SCHEMA = "blindassist-dtr-x7-full-static-world-anchor-replay-v1"
LEDGER_SCHEMA = "blindassist-dtr-x7-full-static-world-anchor-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x7-full-static-world-anchor-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x7-full-static-world-anchor-freeze-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x7-full-static-world-anchor-materialization-v1"
SEQUENCE_SCHEMA = "blindassist-dtr-x7-sequence-materialization-v1"


def _paths(root: Path, sequence: str | None = None) -> dict[str, Path]:
    base = root if sequence is None else root / "sequences" / sequence
    return {
        "freeze": root / "freeze.json",
        "lock": base / "materialize.lock.json",
        "ledger": base / "lag-floxel.npz",
        "manifest": base / "lag-floxel.json",
        "predictions": root / "predictions.json",
        "result": root / "result.json",
        "materialization": root / "materialization.json",
    }


def _x3_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _baseline_rows(args: argparse.Namespace) -> dict[str, Any]:
    _baseline, rows = x3._load_baseline(args)
    require(len(rows) == 6, "x7_sequence_count")
    return rows


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    rows = _baseline_rows(args)
    x3_inputs = {}
    for sequence in sorted(rows):
        ledger, manifest = _x3_paths(args.x3_root.resolve(strict=True), sequence)
        value = json.loads(manifest.resolve(strict=True).read_text(encoding="utf-8"))
        require(value.get("schema") == x5.X3_LEDGER_SCHEMA, f"x7_x3_schema:{sequence}")
        require(value.get("truth_blind") is True, f"x7_x3_truth:{sequence}")
        require(value.get("ledger_sha256") == sha256_file(ledger.resolve(strict=True)), f"x7_x3_hash:{sequence}")
        x3_inputs[sequence] = {
            "ledger_sha256": sha256_file(ledger),
            "manifest_sha256": sha256_file(manifest),
        }
    return {
        "schema": FREEZE_SCHEMA,
        "truth_blind_materialization": True,
        "algorithm_files": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in (__file__, x6.__file__, x3.__file__)
        ],
        "source_config": {
            "position_tolerance_m": x6.POSITION_TOLERANCE_M,
            "minimum_persistence_span_s": x6.MINIMUM_PERSISTENCE_SPAN_S,
            "recent_match_max_age_s": x6.RECENT_MATCH_MAX_AGE_S,
            "maximum_anchor_age_s": x6.MAXIMUM_ANCHOR_AGE_S,
            "required_past_matches": "ONE_RECENT_AND_ONE_SPANNING_RAW_LIDAR_OCCUPANCY",
            "x3_candidate_velocity": "UNCHANGED",
        },
        "frozen_downstream": {
            "motion_bounds": "UNCHANGED_X3",
            "route_entry_geometry": "UNCHANGED_R7",
            "event_lifecycle": "UNCHANGED_X3",
            "full_score_gate": {
                "minimum_contact_recall": x3.MINIMUM_CONTACT_RECALL,
                "maximum_false_segments": x3.MAXIMUM_FALSE_SEGMENTS,
                "minimum_event_f1": x3.MINIMUM_EVENT_F1,
                "minimum_median_lead_s": x3.MINIMUM_MEDIAN_LEAD_S,
                "minimum_dropout_recovery": x3.MINIMUM_DROPOUT_RECOVERY,
            },
        },
        "inputs": {
            "x3_result_sha256": sha256_file(args.x3_result.resolve(strict=True)),
            "x6_result_sha256": sha256_file(args.x6_result.resolve(strict=True)),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "calibration_sha256": sha256_file(args.calibration_dir.resolve(strict=True) / "lidars.yaml"),
            "x3_sequences": x3_inputs,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = _paths(root)["freeze"]
    fingerprint = _fingerprint(args)
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == fingerprint, "x7_freeze_drift")
    else:
        write_json(path, fingerprint)
    return {
        "schema": FREEZE_SCHEMA,
        "status": "READY",
        "sequences": sorted(_baseline_rows(args)),
        "freeze": str(path),
        "freeze_sha256": sha256_file(path),
    }


def _raw_world_xy(
    world_cloud: np.ndarray, pose: Mapping[str, Any]
) -> np.ndarray:
    local = x1._local_cloud(world_cloud, pose)
    return x5._ego_to_world(
        local[:, :2], np.zeros((len(local), 2), dtype=np.float64), pose
    )[0]


def _history_by_frame(
    frames: Sequence[int], timestamps: Mapping[int, float]
) -> dict[int, tuple[list[int], list[int]]]:
    result = {}
    for frame in frames:
        now = timestamps[frame]
        recent = [
            candidate
            for candidate in frames
            if 0.0 < now - timestamps[candidate] <= x6.RECENT_MATCH_MAX_AGE_S + 1e-12
        ]
        spanning = [
            candidate
            for candidate in frames
            if x6.MINIMUM_PERSISTENCE_SPAN_S - 1e-12
            <= now - timestamps[candidate]
            <= x6.MAXIMUM_ANCHOR_AGE_S + 1e-12
        ]
        result[frame] = (recent, spanning)
    return result


def materialize_sequence(args: argparse.Namespace) -> dict[str, Any]:
    require(args.sequence is not None, "x7_sequence_required")
    root = args.root.resolve()
    freeze_path = _paths(root)["freeze"].resolve(strict=True)
    require(json.loads(freeze_path.read_text(encoding="utf-8")) == _fingerprint(args), "x7_freeze_drift")
    rows = _baseline_rows(args)
    require(args.sequence in rows, f"x7_unknown_sequence:{args.sequence}")
    sequence = args.sequence
    paths = _paths(root, sequence)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    if paths["ledger"].exists() and paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("ledger_sha256") == sha256_file(paths["ledger"]):
            return {
                "schema": SEQUENCE_SCHEMA,
                "status": "SEQUENCE_COMPLETE",
                "sequence": sequence,
                "frames": int(manifest["frames"]),
                "resumed_from_sealed_ledger": True,
                "manifest_sha256": sha256_file(paths["manifest"]),
            }
    x3._acquire_lock(paths["lock"])
    started_all = time.perf_counter()
    try:
        source_path, source_manifest_path = _x3_paths(
            args.x3_root.resolve(strict=True), sequence
        )
        source = x1._load_sealed(
            source_path.resolve(strict=True),
            source_manifest_path.resolve(strict=True),
            x5.X3_LEDGER_SCHEMA,
        )
        frames = [int(frame) for frame in source["frames"]]
        timestamps = {
            int(frame): float(stamp)
            for frame, stamp in zip(source["frames"], source["frame_time_s"])
        }
        require(frames == list(range(frames[0], frames[-1] + 1)), f"x7_noncontiguous:{sequence}")
        history = _history_by_frame(frames, timestamps)
        bag_path = Path(rows[sequence]["sources"]["bag"]).resolve(strict=True)
        raw_frames, _raw_times, poses, clouds, lidar = load_world_clouds(
            bag_path=bag_path,
            timestamps_path=args.timestamps.resolve(strict=True),
            calibration_dir=args.calibration_dir.resolve(strict=True),
            timestamps_override=timestamps,
        )
        require(raw_frames.tolist() == frames, f"x7_raw_frames:{sequence}")
        raw_xy = {
            int(frame): _raw_world_xy(cloud, poses[int(frame)])
            for frame, cloud in zip(raw_frames, clouds)
        }
        output_rows = []
        seconds = []
        input_cells = removed_cells = 0
        for index, frame in enumerate(frames):
            started = time.perf_counter()
            current = x5._frame_arrays(source, frame)
            current_world = x5._ego_to_world(current[0], current[1], poses[frame])[0]
            recent_frames, spanning_frames = history[frame]
            recent_world = x6._history_raw_world_positions(raw_xy, recent_frames)
            spanning_world = x6._history_raw_world_positions(raw_xy, spanning_frames)
            static = x6._static_mask(current_world, recent_world, spanning_world)
            keep = np.flatnonzero(~static)
            output_rows.append(
                (current[0][keep], current[1][keep], current[2][keep], current[3][keep])
            )
            seconds.append(time.perf_counter() - started)
            input_cells += len(current[0])
            removed_cells += int(np.count_nonzero(static))
            if (index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "completed": index + 1,
                            "total": len(frames),
                            "percent": round(100.0 * (index + 1) / len(frames), 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        arrays = x5._pack_rows(frames, timestamps, output_rows)
        atomic_npz(paths["ledger"], **arrays)
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "oracle": False,
            "sequence": sequence,
            "frames": len(frames),
            "online_information_boundary": "frame t uses X3 candidate cells at t and raw bag LiDAR/poses strictly through t",
            "fixed_anchor": {
                "position_tolerance_m": x6.POSITION_TOLERANCE_M,
                "minimum_persistence_span_s": x6.MINIMUM_PERSISTENCE_SPAN_S,
                "recent_match_max_age_s": x6.RECENT_MATCH_MAX_AGE_S,
                "maximum_anchor_age_s": x6.MAXIMUM_ANCHOR_AGE_S,
                "required_past_matches": "ONE_RECENT_AND_ONE_SPANNING",
            },
            "frozen_downstream": {
                "cell_velocity": "UNCHANGED_X3",
                "motion_bounds": "UNCHANGED_X3",
                "route_entry_geometry": "UNCHANGED_R7",
                "event_lifecycle": "UNCHANGED_X3",
            },
            "source": {
                "freeze_sha256": sha256_file(freeze_path),
                "x3_ledger": str(source_path.resolve()),
                "x3_ledger_sha256": sha256_file(source_path.resolve()),
                "x3_manifest": str(source_manifest_path.resolve()),
                "x3_manifest_sha256": sha256_file(source_manifest_path.resolve()),
                "bag": str(bag_path),
                "bag_sha256": sha256_file(bag_path),
                "bag_pose_authority": lidar["bag_authority"],
                "selected_raw_lidar_payload_sha256": lidar["selected_lidar_payload_sha256"],
            },
            "diagnostics": {
                "input_cells": int(input_cells),
                "static_cells_removed": int(removed_cells),
                "retained_cells": int(input_cells - removed_cells),
                "source_compute_p95_s": float(np.quantile(np.asarray(seconds), 0.95, method="higher")),
                "elapsed_s": time.perf_counter() - started_all,
            },
            "ledger": str(paths["ledger"]),
            "ledger_sha256": sha256_file(paths["ledger"]),
        }
        write_json(paths["manifest"], manifest)
        return {
            "schema": SEQUENCE_SCHEMA,
            "status": "SEQUENCE_COMPLETE",
            "sequence": sequence,
            "frames": len(frames),
            "input_cells": input_cells,
            "static_cells_removed": removed_cells,
            "retained_cells": input_cells - removed_cells,
            "elapsed_s": time.perf_counter() - started_all,
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
        }
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    freeze = _paths(root)["freeze"]
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x7_freeze_drift")
    receipts = []
    total_frames = 0
    total_input = total_removed = 0
    for sequence in sorted(_baseline_rows(args)):
        paths = _paths(root, sequence)
        manifest = json.loads(paths["manifest"].resolve(strict=True).read_text(encoding="utf-8"))
        require(manifest.get("schema") == LEDGER_SCHEMA, f"x7_schema:{sequence}")
        require(manifest.get("truth_blind") is True, f"x7_truth:{sequence}")
        require(manifest.get("ledger_sha256") == sha256_file(paths["ledger"].resolve(strict=True)), f"x7_hash:{sequence}")
        total_frames += int(manifest["frames"])
        total_input += int(manifest["diagnostics"]["input_cells"])
        total_removed += int(manifest["diagnostics"]["static_cells_removed"])
        receipts.append(
            {
                "sequence": sequence,
                "frames": int(manifest["frames"]),
                "manifest": str(paths["manifest"]),
                "manifest_sha256": sha256_file(paths["manifest"]),
            }
        )
    require(total_frames == 4787, f"x7_full_frame_count:{total_frames}")
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "COMPLETE",
        "truth_blind": True,
        "sequences": len(receipts),
        "frames": total_frames,
        "input_cells": total_input,
        "static_cells_removed": total_removed,
        "retained_cells": total_input - total_removed,
        "backend": {
            "python": platform.python_version(),
            "pid": os.getpid(),
            "raw_lidar_decode": "CPU",
            "anchor_query": "SCIPY_CKDTREE_WORKERS_1_PER_SEQUENCE_PROCESS",
        },
        "freeze": str(freeze),
        "freeze_sha256": sha256_file(freeze),
        "sequence_manifests": receipts,
    }
    write_json(_paths(root)["materialization"], receipt)
    return receipt


def predict(args: argparse.Namespace) -> dict[str, Any]:
    previous_ledger, previous_prediction = x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.LEDGER_SCHEMA = LEDGER_SCHEMA
        x3.PREDICTION_SCHEMA = PREDICTION_SCHEMA
        result = x3.predict(args)
    finally:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous_ledger, previous_prediction
    result["prediction_boundary"] = (
        "sealed full X7 static-world-anchored ledgers and frozen global route lifecycle only; "
        "no labels, roster event details, evaluator identity, or outcomes"
    )
    result["scorer_compatibility_arm_key"] = {
        "X3_LAG_FLOXEL": "X7_STATIC_WORLD_ANCHOR"
    }
    write_json(_paths(args.root.resolve(strict=True))["predictions"], result)
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    previous = (x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA)
    try:
        x3.SCHEMA = SCHEMA
        x3.LEDGER_SCHEMA = LEDGER_SCHEMA
        x3.PREDICTION_SCHEMA = PREDICTION_SCHEMA
        result = x3.score(args)
    finally:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    met = bool(result["gate"]["passed"])
    result["schema"] = SCHEMA
    result["status"] = (
        "DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_MET"
        if met
        else "DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_NOT_MET"
    )
    result["metrics"]["X7_STATIC_WORLD_ANCHOR"] = result["metrics"].pop("X3_LAG_FLOXEL")
    result["decision"]["next"] = (
        "FREEZE_X7_AND_CONFIRM_ON_NEW_SOURCE_DISJOINT_COHORT"
        if met
        else "CLOSE_OR_ATTRIBUTE_X7_FULL_REPLAY_WITHOUT_PARAMETER_SWEEP"
    )
    result["evidence_boundary"] = [
        "Full six-sequence replay on the already opened C31/X0 Development cohort; not new source-disjoint confirmation.",
        "All X7 static-world-anchored ledgers and predictions were sealed before native OBB truth was opened by this scorer.",
        "The raw LiDAR occupancy anchor is flow-independent; X3 velocity, motion bounds, route geometry, and lifecycle are unchanged.",
        "Real-device latency and Android deployment are not established.",
    ]
    write_json(_paths(args.root.resolve(strict=True))["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    x3_root = REPO / "artifacts.local" / "evidence" / "dtr-x3" / "full-lag-floxel-replay-mp"
    x6_root = REPO / "artifacts.local" / "evidence" / "dtr-x6" / "static-world-persistence-falsifier-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "materialize", "assemble", "predict", "score"))
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829",
    )
    parser.add_argument("--sequence")
    parser.add_argument("--x3-root", type=Path, default=x3_root)
    parser.add_argument("--x3-result", type=Path, default=x3_root / "result.json")
    parser.add_argument("--x6-result", type=Path, default=x6_root / "result.json")
    parser.add_argument("--baseline-predictions", type=Path, default=c31 / "baseline-predictions.json")
    parser.add_argument("--baseline-result", type=Path, default=c31 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c31_fresh_confirmation_roster.json")
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
    if args.mode == "prepare":
        payload = prepare(args)
    elif args.mode == "materialize":
        payload = materialize_sequence(args)
    elif args.mode == "assemble":
        payload = assemble(args)
    elif args.mode == "predict":
        payload = predict(args)
    else:
        payload = score(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
