"""Run the full six-sequence lag-compensated Floxel Development replay.

X3 freezes the X1/X2 source after the positive and representative false-error
gates passed.  For each output frame ``t`` it estimates a symmetric explicit
voxel flow field at ``t-2`` from scans ``t-4..t`` and transports the result to
the current ego frame.  Materialization is truth-blind and resumable per frame;
native OBB truth is opened only after complete candidate predictions are
sealed.  The R7 motion bounds, route-entry geometry, and lifecycle are fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_c27_persistent_point_support as c27  # noqa: E402
import dtr_c4_detector_independent_global_risk as c4  # noqa: E402
import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x1c_lag_compensated_floxel_source as x1c  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
from dtr_c1_global_obb_cohort_admission import (  # noqa: E402
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import (  # noqa: E402
    _tracks,
    aggregate_scores,
    score_sequence,
)
from dtr_m1_point_velocity_oracle import load_world_clouds  # noqa: E402
from dtr_r5_dropout_canary import cases_from_tracks  # noqa: E402
from dtr_r7_occupancy_flow_canary import FlowLedger, atomic_npz, load_flow_ledger  # noqa: E402


SCHEMA = "blindassist-dtr-x3-full-lag-floxel-replay-v1"
LEDGER_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x3-full-lag-floxel-freeze-v1"
MINIMUM_CONTACT_RECALL = 5
MAXIMUM_FALSE_SEGMENTS = 16
MINIMUM_EVENT_F1 = 0.35
MINIMUM_MEDIAN_LEAD_S = 2.0
MINIMUM_DROPOUT_RECOVERY = 5


def _paths(root: Path, sequence: str | None = None) -> dict[str, Path]:
    base = root if sequence is None else root / "sequences" / sequence
    return {
        "freeze": root / "freeze.json",
        "lock": root / "materialize.lock.json",
        "progress": root / "progress.json",
        "predictions": root / "predictions.json",
        "result": root / "result.json",
        "ledger": base / "lag-floxel.npz",
        "manifest": base / "lag-floxel.json",
        "checkpoints": base / "checkpoints",
    }


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    files = [Path(__file__).resolve(), Path(x1.__file__).resolve(), Path(x1c.__file__).resolve(), Path(x2.__file__).resolve()]
    return {
        "schema": FREEZE_SCHEMA,
        "truth_blind_materialization": True,
        "algorithm_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in files
        ],
        "source_config": {
            "scan_offsets_at_reference": [-2, -1, 1, 2],
            "output_lag_scans": x1c.LAG_SCANS,
            "rear_roi_forward_m": list(x1.REAR_ROI_FORWARD_M),
            "flow_grid_m": x1.FLOW_GRID_M,
            "distance_grid_m": x1.DT_GRID_M,
            "learning_rate": x1.LEARNING_RATE,
            "maximum_epochs": x1.MAX_EPOCHS,
            "early_patience": x1.EARLY_PATIENCE,
            "early_minimum_delta": x1.EARLY_MIN_DELTA,
            "cluster_weight": x1.CLUSTER_WEIGHT,
            "motion_bounds_mps": [
                x1.FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps,
                x1.FROZEN_FLOW_CONFIG.maximum_dynamic_speed_mps,
            ],
        },
        "full_gate": {
            "minimum_contact_recall": MINIMUM_CONTACT_RECALL,
            "maximum_false_segments": MAXIMUM_FALSE_SEGMENTS,
            "minimum_event_f1": MINIMUM_EVENT_F1,
            "minimum_median_lead_s": MINIMUM_MEDIAN_LEAD_S,
            "minimum_dropout_recovery": MINIMUM_DROPOUT_RECOVERY,
        },
        "inputs": {
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "calibration_sha256": sha256_file(args.calibration_dir.resolve(strict=True) / "lidars.yaml"),
            "x2_result_sha256": sha256_file(args.x2_result.resolve(strict=True)),
        },
    }


def _acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        pid = int(value.get("pid", -1))
        try:
            import psutil

            active = pid > 0 and psutil.pid_exists(pid)
        except ImportError:
            active = True
        require(not active, f"x3_materializer_active:{pid}")
        path.unlink()
    write_json(path, {"pid": os.getpid(), "created_unix_s": time.time()})


def _load_baseline(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    path = args.baseline_predictions.resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("truth_blind") is True, "x3_baseline_not_truth_blind")
    rows = {str(row["sequence"]): row for row in value["sequences"]}
    require(len(rows) == 6, "x3_sequence_count")
    return value, rows


def _checkpoint_valid(path: Path, frame: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            return (
                int(values["output_frame"][0]) == frame
                and {"forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps", "source_point_count"}
                <= set(values.files)
            )
    except Exception:
        return False


def _write_progress(
    path: Path,
    *,
    completed: int,
    total: int,
    sequence: str,
    frame: int | None,
    started: float,
    completed_at_start: int,
) -> None:
    elapsed = time.perf_counter() - started
    processed = completed - completed_at_start
    rate = processed / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - completed) / rate if rate > 0.0 else None
    write_json(
        path,
        {
            "schema": "blindassist-dtr-x3-progress-v1",
            "completed": completed,
            "total": total,
            "percent": 100.0 * completed / total if total else 100.0,
            "active_sequence": sequence,
            "active_frame": frame,
            "elapsed_s_this_process": elapsed,
            "eta_s_this_process_rate": remaining,
            "last_activity_unix_s": time.time(),
            "failures": 0,
        },
    )


def _completed_count(root: Path, frames_by_sequence: Mapping[str, Sequence[int]]) -> int:
    total = 0
    for sequence, frames in frames_by_sequence.items():
        paths = _paths(root, sequence)
        if paths["ledger"].exists() and paths["manifest"].exists():
            try:
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                if manifest.get("ledger_sha256") == sha256_file(paths["ledger"]):
                    total += max(0, len(frames) - 4)
                    continue
            except Exception:
                pass
        total += sum(
            _checkpoint_valid(paths["checkpoints"] / f"{frame:06d}.npz", frame)
            for frame in frames[4:]
        )
    return total


def _assemble_sequence(
    *,
    root: Path,
    sequence: str,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    bag_path: Path,
    baseline_ledger: Path,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = _paths(root, sequence)
    rows = []
    empty = {
        "forward_m": np.empty(0, np.float32),
        "left_m": np.empty(0, np.float32),
        "velocity_forward_mps": np.empty(0, np.float32),
        "velocity_left_mps": np.empty(0, np.float32),
        "source_point_count": np.empty(0, np.int32),
    }
    for frame in frames:
        checkpoint = paths["checkpoints"] / f"{frame:06d}.npz"
        if frame in frames[:4]:
            rows.append(empty)
            continue
        require(_checkpoint_valid(checkpoint, frame), f"x3_checkpoint_missing:{sequence}:{frame}")
        with np.load(checkpoint, allow_pickle=False) as values:
            rows.append({name: values[name].copy() for name in empty})
    offsets = np.cumsum([0] + [len(row["forward_m"]) for row in rows], dtype=np.int64)
    arrays = {
        "frames": np.asarray(frames, dtype=np.int32),
        "frame_time_s": np.asarray([timestamps[frame] for frame in frames], dtype=np.float64),
        "offsets": offsets,
        "forward_m": np.concatenate([row["forward_m"] for row in rows]),
        "left_m": np.concatenate([row["left_m"] for row in rows]),
        "velocity_forward_mps": np.concatenate([row["velocity_forward_mps"] for row in rows]),
        "velocity_left_mps": np.concatenate([row["velocity_left_mps"] for row in rows]),
        "component_id": np.concatenate([np.arange(len(row["forward_m"]), dtype=np.int32) for row in rows]),
        "source_point_count": np.concatenate([row["source_point_count"] for row in rows]),
        "flow_support": np.concatenate([np.ones(len(row["forward_m"]), dtype=np.float32) for row in rows]),
    }
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(paths["ledger"], **arrays)
    manifest = {
        "schema": LEDGER_SCHEMA,
        "truth_blind": True,
        "oracle": False,
        "sequence": sequence,
        "frames": len(frames),
        "online_information_boundary": (
            "output frame t consumes reference t-2 estimated from scans t-4 through t"
        ),
        "frozen_downstream": {
            "route_entry_geometry": "UNCHANGED_R7",
            "motion_bounds": "UNCHANGED_R7",
            "event_lifecycle": "UNCHANGED_R7",
        },
        "source": {
            "freeze_sha256": sha256_file(_paths(root)["freeze"]),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "baseline_ledger": str(baseline_ledger),
            "baseline_ledger_sha256": sha256_file(baseline_ledger),
        },
        "diagnostics": {
            "optimized_frames": len(frames) - 4,
            "output_cells": int(len(arrays["forward_m"])),
            "median_optimization_s": float(np.median([row["seconds"] for row in diagnostics])) if diagnostics else None,
            "median_information_delay_s": float(np.median([row["delay_s"] for row in diagnostics])) if diagnostics else None,
        },
        "ledger": str(paths["ledger"]),
        "ledger_sha256": sha256_file(paths["ledger"]),
    }
    write_json(paths["manifest"], manifest)
    return manifest


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    require(torch.cuda.is_available(), "x3_cuda_unavailable")
    x2_result = json.loads(args.x2_result.resolve(strict=True).read_text(encoding="utf-8"))
    require(x2_result.get("status") == "DTR_X2_FLOXEL_ERROR_SLICE_GATE_MET", "x3_x2_gate")
    _baseline, baseline_rows = _load_baseline(args)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = _paths(root)
    fingerprint = _fingerprint(args)
    if paths["freeze"].exists():
        require(json.loads(paths["freeze"].read_text(encoding="utf-8")) == fingerprint, "x3_freeze_drift")
    else:
        write_json(paths["freeze"], fingerprint)
    _acquire_lock(paths["lock"])
    started = time.perf_counter()
    frames_by_sequence: dict[str, list[int]] = {}
    for sequence, row in baseline_rows.items():
        source = row["sources"]["ledgers"]["M1_PD_GLOBAL"]
        with np.load(Path(source["ledger"]).resolve(strict=True), allow_pickle=False) as values:
            frames_by_sequence[sequence] = [int(value) for value in values["frames"]]
    total = sum(max(0, len(frames) - 4) for frames in frames_by_sequence.values())
    completed = _completed_count(root, frames_by_sequence)
    completed_at_start = completed
    sequence_receipts = []
    try:
        for sequence in sorted(baseline_rows):
            seq_paths = _paths(root, sequence)
            if seq_paths["ledger"].exists() and seq_paths["manifest"].exists():
                manifest = json.loads(seq_paths["manifest"].read_text(encoding="utf-8"))
                if manifest.get("ledger_sha256") == sha256_file(seq_paths["ledger"]):
                    sequence_receipts.append(manifest)
                    continue
            baseline_source = baseline_rows[sequence]["sources"]["ledgers"]["M1_PD_GLOBAL"]
            baseline_ledger = Path(baseline_source["ledger"]).resolve(strict=True)
            with np.load(baseline_ledger, allow_pickle=False) as values:
                frames = [int(value) for value in values["frames"]]
                timestamps = {
                    int(frame): float(stamp)
                    for frame, stamp in zip(values["frames"], values["frame_time_s"])
                }
            require(frames == list(range(frames[0], frames[-1] + 1)), f"x3_noncontiguous_frames:{sequence}")
            bag_path = Path(baseline_rows[sequence]["sources"]["bag"]).resolve(strict=True)
            loaded_frames, frame_times, poses, world_clouds, _lidar = load_world_clouds(
                bag_path=bag_path,
                timestamps_path=args.timestamps.resolve(strict=True),
                calibration_dir=args.calibration_dir.resolve(strict=True),
                timestamps_override=timestamps,
            )
            require(loaded_frames.tolist() == frames, f"x3_loaded_frames:{sequence}")
            cloud_by_frame = {int(frame): cloud for frame, cloud in zip(loaded_frames, world_clouds)}
            seq_paths["checkpoints"].mkdir(parents=True, exist_ok=True)
            diagnostics = []
            for output_index, output_frame in enumerate(frames[4:], start=4):
                checkpoint = seq_paths["checkpoints"] / f"{output_frame:06d}.npz"
                if _checkpoint_valid(checkpoint, output_frame):
                    with np.load(checkpoint, allow_pickle=False) as values:
                        diagnostics.append(
                            {
                                "output_frame": output_frame,
                                "seconds": float(values["optimization_s"][0]),
                                "delay_s": float(values["delay_s"][0]),
                            }
                        )
                    continue
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
                    f"x3_empty_cloud:{sequence}:{output_frame}",
                )
                displacement, detail = x1._optimize_frame(
                    current, supports, scan_offsets=scan_offsets, device=torch.device("cuda:0")
                )
                local_times = [frame_times[frame] for frame in frames[output_index - 4 : output_index + 1]]
                one_step_s = float(np.median(np.diff(local_times)))
                positions, velocities, source_counts = x1._aggregate(
                    current, displacement / one_step_s, counts
                )
                delay_s = frame_times[output_frame] - frame_times[reference]
                positions, velocities = x2._transport(
                    positions,
                    velocities,
                    reference_pose=poses[reference],
                    output_pose=poses[output_frame],
                    delay_s=delay_s,
                )
                atomic_npz(
                    checkpoint,
                    output_frame=np.asarray([output_frame], dtype=np.int32),
                    forward_m=positions[:, 0].astype(np.float32),
                    left_m=positions[:, 1].astype(np.float32),
                    velocity_forward_mps=velocities[:, 0].astype(np.float32),
                    velocity_left_mps=velocities[:, 1].astype(np.float32),
                    source_point_count=source_counts.astype(np.int32),
                    optimization_s=np.asarray([detail["seconds"]], dtype=np.float64),
                    delay_s=np.asarray([delay_s], dtype=np.float64),
                )
                completed += 1
                diagnostics.append(
                    {
                        "output_frame": output_frame,
                        "seconds": detail["seconds"],
                        "delay_s": delay_s,
                    }
                )
                _write_progress(
                    paths["progress"],
                    completed=completed,
                    total=total,
                    sequence=sequence,
                    frame=output_frame,
                    started=started,
                    completed_at_start=completed_at_start,
                )
                if completed % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "progress": f"{completed}/{total}",
                                "percent": round(100.0 * completed / total, 2),
                                "sequence": sequence,
                                "frame": output_frame,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            manifest = _assemble_sequence(
                root=root,
                sequence=sequence,
                frames=frames,
                timestamps=timestamps,
                bag_path=bag_path,
                baseline_ledger=baseline_ledger,
                diagnostics=diagnostics,
            )
            sequence_receipts.append(manifest)
        receipt = {
            "schema": "blindassist-dtr-x3-full-materialization-v1",
            "status": "COMPLETE",
            "truth_blind": True,
            "sequences": len(sequence_receipts),
            "optimized_frames": total,
            "elapsed_s_this_process": time.perf_counter() - started,
            "backend": {
                "required": "cuda",
                "verified": True,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "python": platform.python_version(),
            },
            "freeze": str(paths["freeze"]),
            "freeze_sha256": sha256_file(paths["freeze"]),
            "sequence_manifests": [
                {
                    "sequence": row["sequence"],
                    "manifest": str(_paths(root, row["sequence"])["manifest"]),
                    "manifest_sha256": sha256_file(_paths(root, row["sequence"])["manifest"]),
                }
                for row in sequence_receipts
            ],
        }
        write_json(root / "materialization.json", receipt)
        _write_progress(
            paths["progress"],
            completed=total,
            total=total,
            sequence="COMPLETE",
            frame=None,
            started=started,
            completed_at_start=completed_at_start,
        )
        return receipt
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def _load_candidate(path: Path, manifest_path: Path, *, sequence: str, frames: Sequence[int]) -> FlowLedger:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == LEDGER_SCHEMA, f"x3_ledger_schema:{sequence}")
    require(manifest.get("sequence") == sequence, f"x3_ledger_sequence:{sequence}")
    require(manifest.get("truth_blind") is True, f"x3_ledger_truth:{sequence}")
    require(manifest.get("ledger_sha256") == sha256_file(path), f"x3_ledger_hash:{sequence}")
    with np.load(path, allow_pickle=False) as values:
        require(values["frames"].tolist() == list(frames), f"x3_ledger_frames:{sequence}")
        return FlowLedger(
            frames=values["frames"].copy(),
            offsets=values["offsets"].copy(),
            forward_m=values["forward_m"].copy(),
            left_m=values["left_m"].copy(),
            velocity_forward_mps=values["velocity_forward_mps"].copy(),
            velocity_left_mps=values["velocity_left_mps"].copy(),
            component_id=values["component_id"].copy(),
            manifest=manifest,
        )


def predict(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    materialization = json.loads((root / "materialization.json").read_text(encoding="utf-8"))
    require(materialization.get("status") == "COMPLETE", "x3_materialization_incomplete")
    baseline, baseline_rows = _load_baseline(args)
    predictions = []
    for sequence in sorted(baseline_rows):
        baseline_row = baseline_rows[sequence]
        source = baseline_row["sources"]["ledgers"]["M1_PD_GLOBAL"]
        with np.load(Path(source["ledger"]).resolve(strict=True), allow_pickle=False) as values:
            frames = [int(value) for value in values["frames"]]
            timestamps = {
                int(frame): float(stamp)
                for frame, stamp in zip(values["frames"], values["frame_time_s"])
            }
        seq_paths = _paths(root, sequence)
        ledger = _load_candidate(
            seq_paths["ledger"], seq_paths["manifest"], sequence=sequence, frames=frames
        )
        predictions.append(
            {
                "sequence": sequence,
                "arms": {
                    "M1_PDC_GLOBAL": baseline_row["arms"]["M1_PDC_GLOBAL"],
                    "X3_LAG_FLOXEL": c4._predict_arm(
                        ledger=ledger, frames=frames, timestamps=timestamps
                    ),
                },
                "candidate_ledger": {
                    "ledger": str(seq_paths["ledger"]),
                    "ledger_sha256": sha256_file(seq_paths["ledger"]),
                    "manifest": str(seq_paths["manifest"]),
                    "manifest_sha256": sha256_file(seq_paths["manifest"]),
                },
                "baseline_sources": baseline_row["sources"],
            }
        )
    output = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "prediction_boundary": (
            "sealed full lag-Floxel ledgers and frozen global route lifecycle only; "
            "no labels, roster event details, evaluator identity, or outcomes"
        ),
        "sequences": predictions,
        "source": {
            "materialization_sha256": sha256_file(root / "materialization.json"),
            "freeze_sha256": sha256_file(_paths(root)["freeze"]),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
        },
    }
    write_json(_paths(root)["predictions"], output)
    return output


def _aggregate_dropout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    misses = sum(int(row["track_only_window_misses"]) for row in rows)
    recovered = sum(int(row["m1_ct_recovered_track_only_window_misses"]) for row in rows)
    return {
        "trials": sum(int(row["trials"]) for row in rows),
        "track_only_window_misses": misses,
        "dropout_recovery": recovered,
        "dropout_recovery_rate": recovered / misses if misses else None,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    prediction_path = _paths(root)["predictions"]
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    require(predictions.get("schema") == PREDICTION_SCHEMA, "x3_prediction_schema")
    require(predictions.get("truth_blind") is True, "x3_prediction_truth")
    roster_path = args.roster.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    baseline_result_path = args.baseline_result.resolve(strict=True)
    baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8"))
    require(
        baseline_result.get("terminal_status")
        == "DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_NOT_MET",
        "x3_baseline_result_status",
    )
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "x3_labels_hash")
    require(roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path), "x3_timestamps_hash")
    prediction_rows = {str(row["sequence"]): row for row in predictions["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    require(set(prediction_rows) == set(roster_rows), "x3_score_coverage")
    scores_by_arm: dict[str, list[dict[str, Any]]] = {"M1_PDC_GLOBAL": [], "X3_LAG_FLOXEL": []}
    dropout_rows = []
    per_sequence = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamp_zip:
        for sequence in sorted(roster_rows):
            timestamps = _load_timestamps(timestamp_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(frames=frames, timestamps=timestamps, boxes_by_frame=boxes)
            row = prediction_rows[sequence]
            sequence_scores = {}
            for arm in scores_by_arm:
                arm_score = score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=c4._prediction_frames(frames, row["arms"][arm]),
                )
                scores_by_arm[arm].append(arm_score)
                sequence_scores[arm] = arm_score
            sources = row["baseline_sources"]["ledgers"]
            pd = c27._load_arrays(
                Path(sources["M1_PD_GLOBAL"]["ledger"]),
                Path(sources["M1_PD_GLOBAL"]["manifest"]),
                {"frames", "frame_time_s", "frame_ego_x_m", "frame_ego_y_m", "frame_ego_yaw_rad"},
            )
            frame_poses = {frame: c27._pose(pd, index) for index, frame in enumerate(frames)}
            cases = cases_from_tracks(
                _tracks(boxes_by_frame=boxes, timestamps=timestamps, frame_poses=frame_poses)
            )
            cases_by_key = {(case.label_id, case.segment_index): case for case in cases}
            r7_ledger = load_flow_ledger(
                Path(sources["R7_P_GLOBAL"]["ledger"]),
                Path(sources["R7_P_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            pd_ledger = c27.load_point_ledger(
                Path(sources["M1_PD_GLOBAL"]["ledger"]),
                Path(sources["M1_PD_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            candidate = _load_candidate(
                Path(row["candidate_ledger"]["ledger"]),
                Path(row["candidate_ledger"]["manifest"]),
                sequence=sequence,
                frames=frames,
            )
            stress = c27.dropout_stress(
                roster_sequence=roster_rows[sequence],
                cases=cases_by_key,
                r7=r7_ledger,
                m1=pd_ledger,
                m1_ct=candidate,
            )
            dropout_rows.append(stress)
            per_sequence.append(
                {"sequence": sequence, "scores": sequence_scores, "dropout_stress": stress}
            )
    aggregate = {arm: aggregate_scores(rows) for arm, rows in scores_by_arm.items()}
    dropout = _aggregate_dropout(dropout_rows)
    baseline = aggregate["M1_PDC_GLOBAL"]
    candidate = aggregate["X3_LAG_FLOXEL"]
    baseline_dropout = baseline_result["metrics"]["M1_PDC_GLOBAL"]
    lead = candidate["median_first_alert_lead_s"]
    gate = {
        "contact_recall_at_least_5_of_6": int(candidate["bounded_contact_events_recalled"]) >= MINIMUM_CONTACT_RECALL,
        "false_segments_at_most_16": int(candidate["false_alert_segments"]) <= MAXIMUM_FALSE_SEGMENTS,
        "event_f1_at_least_0_35": float(candidate["bounded_contact_event_f1"]) >= MINIMUM_EVENT_F1,
        "median_lead_at_least_2s": lead is not None and float(lead) >= MINIMUM_MEDIAN_LEAD_S,
        "dropout_recovery_at_least_5_of_18": int(dropout["dropout_recovery"]) >= MINIMUM_DROPOUT_RECOVERY,
        "false_segments_below_pdc": int(candidate["false_alert_segments"]) < int(baseline["false_alert_segments"]),
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": "DTR_X3_FULL_LAG_FLOXEL_GATE_MET" if met else "DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET",
        "truth_blind_predictions": True,
        "metrics": {
            arm: {
                "contact_recall": int(value["bounded_contact_events_recalled"]),
                "contact_events": int(value["bounded_contact_events"]),
                "false_alert_segments": int(value["false_alert_segments"]),
                "event_f1": float(value["bounded_contact_event_f1"]),
                "median_first_alert_lead_s": value["median_first_alert_lead_s"],
                "dropout_recovery": (
                    int(dropout["dropout_recovery"])
                    if arm == "X3_LAG_FLOXEL"
                    else int(baseline_dropout["dropout_recovery"])
                ),
                "dropout_trials": int(baseline_dropout["dropout_trials"]),
            }
            for arm, value in aggregate.items()
        },
        "gain_vs_pdc": {
            "contact_recall": int(candidate["bounded_contact_events_recalled"]) - int(baseline["bounded_contact_events_recalled"]),
            "false_alert_segments": int(candidate["false_alert_segments"]) - int(baseline["false_alert_segments"]),
            "event_f1": float(candidate["bounded_contact_event_f1"]) - float(baseline["bounded_contact_event_f1"]),
            "median_first_alert_lead_s": (
                float(candidate["median_first_alert_lead_s"]) - float(baseline["median_first_alert_lead_s"])
                if candidate["median_first_alert_lead_s"] is not None and baseline["median_first_alert_lead_s"] is not None
                else None
            ),
        },
        "gate": {"passed": met, "checks": gate},
        "per_sequence": per_sequence,
        "decision": {
            "next": (
                "FREEZE_SOURCE_AND_CONFIRM_ON_NEW_DISJOINT_COHORT"
                if met
                else "CLOSE_X3_AND_ATTRIBUTE_FULL_REPLAY_FAILURES"
            )
        },
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": sha256_file(prediction_path),
            "freeze_sha256": sha256_file(_paths(root)["freeze"]),
            "roster_sha256": sha256_file(roster_path),
            "baseline_result_sha256": sha256_file(baseline_result_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "evidence_boundary": [
            "Full six-sequence replay on the already opened C31/X0 Development cohort; not new source-disjoint confirmation.",
            "All lag-Floxel ledgers and predictions were sealed before native OBB truth was opened by this scorer.",
            "Information causality includes a measured two-scan lag; real-time compute latency and Android deployment are not established.",
            "The source is an independent Floxels-inspired implementation, not official Floxels code or a reproduction of published benchmark metrics.",
        ],
    }
    write_json(_paths(root)["result"], result)
    return result


def status(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    _baseline, baseline_rows = _load_baseline(args)
    frames_by_sequence = {}
    for sequence, row in baseline_rows.items():
        source = row["sources"]["ledgers"]["M1_PD_GLOBAL"]
        with np.load(Path(source["ledger"]).resolve(strict=True), allow_pickle=False) as values:
            frames_by_sequence[sequence] = [int(value) for value in values["frames"]]
    total = sum(max(0, len(frames) - 4) for frames in frames_by_sequence.values())
    completed = _completed_count(root, frames_by_sequence)
    progress_path = _paths(root)["progress"]
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    return {
        "completed": completed,
        "total": total,
        "percent": 100.0 * completed / total if total else 100.0,
        "active_sequence": progress.get("active_sequence"),
        "active_frame": progress.get("active_frame"),
        "last_activity_unix_s": progress.get("last_activity_unix_s"),
        "eta_s": progress.get("eta_s_this_process_rate"),
        "failures": progress.get("failures", 0),
        "materialization_complete": (root / "materialization.json").exists(),
    }


def cleanup_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    materialization = json.loads((root / "materialization.json").read_text(encoding="utf-8"))
    require(materialization.get("status") == "COMPLETE", "x3_cleanup_before_complete")
    removed = []
    for row in materialization["sequence_manifests"]:
        sequence = str(row["sequence"])
        paths = _paths(root, sequence)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        require(manifest.get("ledger_sha256") == sha256_file(paths["ledger"]), f"x3_cleanup_ledger:{sequence}")
        checkpoint_dir = paths["checkpoints"].resolve()
        expected_parent = (root / "sequences" / sequence).resolve()
        require(checkpoint_dir.parent == expected_parent, f"x3_cleanup_scope:{sequence}")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            require(not checkpoint_dir.exists(), f"x3_cleanup_failed:{sequence}")
            removed.append(str(checkpoint_dir))
    return {"removed_checkpoint_directories": removed, "recoverable_from": "sealed sequence ledgers"}


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    root = REPO / "artifacts.local" / "evidence" / "dtr-x3" / "full-lag-floxel-replay"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "predict", "score", "run", "status", "cleanup"))
    parser.add_argument("--root", type=Path, default=root)
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
    parser.add_argument(
        "--x2-result",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x2"
        / "floxel-error-slice"
        / "result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "status":
        print(json.dumps(status(args), indent=2, sort_keys=True))
        return
    if args.mode == "cleanup":
        print(json.dumps(cleanup_checkpoints(args), indent=2, sort_keys=True))
        return
    if args.mode in {"materialize", "run"}:
        value = materialize(args)
        print(json.dumps({"materialization": value["status"], "frames": value["optimized_frames"], "elapsed_s": value["elapsed_s_this_process"]}, sort_keys=True))
    if args.mode in {"predict", "run"}:
        value = predict(args)
        print(json.dumps({"predictions": len(value["sequences"]), "truth_blind": value["truth_blind"]}, sort_keys=True))
    if args.mode in {"score", "run"}:
        value = score(args)
        print(json.dumps({"status": value["status"], "metrics": value["metrics"], "gain_vs_pdc": value["gain_vs_pdc"], "gate": value["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
