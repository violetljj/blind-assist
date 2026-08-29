"""Run a deterministic cluster-vote successor to the lagged Floxel source.

X1b/X1c showed that a symmetric five-scan source has useful motion
information, while independent reruns of the CUDA-autograd implementation
produced different source ledgers.  X4 keeps the same rear ROI, reference
frame ``t-2``, scans ``t-4..t``, two-scan transport, motion bounds, route
geometry, and scorers.  It changes only the motion-source representation:
each deterministic DBSCAN component receives one rigid displacement selected
from fixed multi-scan nearest-occupancy votes.

The estimator is CPU float64, single-threaded, closed form, and has no random
seed, autograd, optimizer, convergence threshold, or early stopping.  DBSCAN
noise is emitted as UNKNOWN (no motion cell).  ``repeat3`` runs the opened X1c
positive window and the 35 opened X2 diagnostic units into three independent
roots; ``verify3`` requires identical canonical array hashes before the source
can advance to a full replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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
import dtr_x1b_symmetric_floxel_oracle as x1b  # noqa: E402
import dtr_x1c_lag_compensated_floxel_source as x1c  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import (  # noqa: E402
    _box_history,
    load_native_boxes,
    load_world_clouds,
)
from dtr_r7_occupancy_flow_canary import _causal_pose, atomic_npz  # noqa: E402
from jrdb_rgb_bridge import read_bag_pose_and_rgb  # noqa: E402


SCHEMA = "blindassist-dtr-x4-deterministic-cluster-vote-canary-v1"
LEDGER_SCHEMA = "blindassist-dtr-x4-deterministic-cluster-vote-ledger-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x4-deterministic-cluster-vote-materialization-v1"
VERIFY_SCHEMA = "blindassist-dtr-x4-deterministic-cluster-vote-repeatability-v1"
SCAN_OFFSETS = (-2, -1, 1, 2)
REPEATS = 3


def _positive_paths(root: Path) -> dict[str, Path]:
    base = root / "positive"
    return {
        "ledger": base / "lag-cluster-vote.npz",
        "manifest": base / "lag-cluster-vote.json",
        "result": base / "result.json",
    }


def _error_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "error-slice" / sequence
    return base / "lag-cluster-vote.npz", base / "lag-cluster-vote.json"


def _canonical_array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _empty_row() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), np.float32),
        np.empty((0, 2), np.float32),
        np.empty(0, np.int32),
    )


def _pack_rows(
    frames: Sequence[int],
    frame_times: Mapping[int, float],
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    offsets = np.cumsum([0] + [len(row[0]) for row in rows], dtype=np.int64)
    return {
        "frames": np.asarray(frames, dtype=np.int32),
        "frame_time_s": np.asarray(
            [frame_times[int(frame)] for frame in frames], dtype=np.float64
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
        "flow_support": np.concatenate(
            [np.ones(len(row[0]), dtype=np.float32) for row in rows]
        ),
    }


def _sorted_candidates(cluster: np.ndarray, supports: Sequence[np.ndarray]) -> np.ndarray:
    """Return fixed rigid-displacement hypotheses from per-scan L1 votes."""
    from scipy.spatial import cKDTree

    candidates = [np.zeros(3, dtype=np.float64)]
    scan_votes = []
    for offset, support in zip(SCAN_OFFSETS, supports):
        tree = cKDTree(np.asarray(support, dtype=np.float64))
        _distance, indices = tree.query(cluster, k=1, workers=1)
        votes = (np.asarray(support, dtype=np.float64)[indices] - cluster) / float(offset)
        scan_vote = np.median(votes, axis=0)
        candidates.append(scan_vote)
        scan_votes.append(scan_vote)
    candidates.append(np.median(np.asarray(scan_votes, dtype=np.float64), axis=0))
    values = np.unique(np.asarray(candidates, dtype=np.float64), axis=0)
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    return values[order]


def _select_cluster_displacement(
    cluster: np.ndarray,
    supports: Sequence[np.ndarray],
) -> tuple[np.ndarray, int, float]:
    """Select one deterministic cluster displacement with the frozen DT objective."""
    from scipy.spatial import cKDTree

    candidates = _sorted_candidates(cluster, supports)
    trees = [cKDTree(np.asarray(support, dtype=np.float64)) for support in supports]
    scores = np.zeros(len(candidates), dtype=np.float64)
    for offset, tree in zip(SCAN_OFFSETS, trees):
        projected = cluster[None, :, :] + float(offset) * candidates[:, None, :]
        distances, _indices = tree.query(
            projected.reshape(-1, 3), k=1, workers=1
        )
        distances = distances.reshape(len(candidates), len(cluster))
        distances = np.minimum(distances, x1.TRUNCATE_DISTANCE_M)
        scores += distances.mean(axis=1) / float(abs(offset) ** 2)
    scores += (
        float(len(supports))
        * x1.FLOW_NORM_END
        * np.linalg.norm(candidates, axis=1)
    )
    norms = np.linalg.norm(candidates, axis=1)
    order = np.lexsort(
        (
            candidates[:, 2],
            candidates[:, 1],
            candidates[:, 0],
            norms,
            scores,
        )
    )
    selected = int(order[0])
    return candidates[selected], int(len(candidates)), float(scores[selected])


def _cluster_vote_frame(
    current: np.ndarray,
    counts: np.ndarray,
    supports: Sequence[np.ndarray],
    *,
    one_step_s: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    require(one_step_s > 0.0, "x4_nonpositive_step")
    require(len(supports) == len(SCAN_OFFSETS), "x4_support_count")
    started = time.perf_counter()
    points = np.asarray(current, dtype=np.float64)
    labels = x1._cluster_labels(points)
    selected_points = []
    selected_velocity = []
    selected_counts = []
    component_rows = []
    for label in sorted(int(value) for value in np.unique(labels) if int(value) >= 0):
        keep = np.flatnonzero(labels == label)
        cluster = points[keep]
        displacement, candidate_count, objective = _select_cluster_displacement(
            cluster, supports
        )
        velocity = displacement / float(one_step_s)
        selected_points.append(cluster)
        selected_velocity.append(np.repeat(velocity[None, :], len(cluster), axis=0))
        selected_counts.append(np.asarray(counts, dtype=np.int32)[keep])
        component_rows.append(
            {
                "label": label,
                "points": int(len(cluster)),
                "candidates": candidate_count,
                "objective": objective,
                "displacement_per_scan_m": displacement.tolist(),
            }
        )
    if selected_points:
        row = x1._aggregate(
            np.concatenate(selected_points, axis=0),
            np.concatenate(selected_velocity, axis=0),
            np.concatenate(selected_counts, axis=0),
        )
    else:
        row = _empty_row()
    return row, {
        "seconds": time.perf_counter() - started,
        "input_points": int(len(points)),
        "clusters": int(len(component_rows)),
        "noise_points": int(np.count_nonzero(labels < 0)),
        "output_cells": int(len(row[0])),
        "components": component_rows,
    }


def _transport_row(
    row: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    reference_pose: Mapping[str, Any],
    output_pose: Mapping[str, Any],
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, velocities = x2._transport(
        row[0].astype(np.float64),
        row[1].astype(np.float64),
        reference_pose=reference_pose,
        output_pose=output_pose,
        delay_s=delay_s,
    )
    return positions, velocities, row[2]


def _backend() -> dict[str, Any]:
    import scipy

    return {
        "required": "cpu_float64",
        "verified": True,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "autograd": False,
        "early_stopping": False,
        "tree_workers": 1,
    }


def _materialize_positive(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    paths = _positive_paths(root)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.positive_baseline.resolve(strict=True), allow_pickle=False) as values:
        timestamps = {
            int(frame): float(stamp)
            for frame, stamp in zip(values["frames"], values["frame_time_s"])
            if x1b.FIRST_SOURCE_FRAME <= int(frame) <= x1b.LAST_SOURCE_FRAME
        }
    frames, frame_times, poses, world_clouds, lidar = load_world_clouds(
        bag_path=args.positive_bag.resolve(strict=True),
        timestamps_path=args.timestamps.resolve(strict=True),
        calibration_dir=args.calibration_dir.resolve(strict=True),
        timestamps_override=timestamps,
    )
    require(
        frames.tolist()
        == list(range(x1b.FIRST_SOURCE_FRAME, x1b.LAST_SOURCE_FRAME + 1)),
        "x4_positive_frame_window",
    )
    cloud_by_frame = {int(frame): cloud for frame, cloud in zip(frames, world_clouds)}
    rows_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    diagnostics = []
    for reference in range(x1.TARGET_FIRST_FRAME, x1.TARGET_LAST_FRAME + 1):
        pose = poses[reference]
        current, counts = x1._voxel_centroids(
            x1._local_cloud(cloud_by_frame[reference], pose)
        )
        supports = []
        for offset in SCAN_OFFSETS:
            support, _support_counts = x1._voxel_centroids(
                x1._local_cloud(
                    cloud_by_frame[reference + offset], pose, margin_m=1.5
                )
            )
            supports.append(support)
        require(
            len(current) > 0 and all(len(value) > 0 for value in supports),
            f"x4_positive_empty_cloud:{reference}",
        )
        local_times = [frame_times[reference + offset] for offset in range(-2, 3)]
        one_step_s = float(np.median(np.diff(local_times)))
        row, detail = _cluster_vote_frame(
            current, counts, supports, one_step_s=one_step_s
        )
        output_frame = reference + x1c.LAG_SCANS
        delay_s = frame_times[output_frame] - frame_times[reference]
        require(delay_s > 0.0, f"x4_positive_delay:{reference}")
        rows_by_frame[output_frame] = _transport_row(
            row,
            reference_pose=poses[reference],
            output_pose=poses[output_frame],
            delay_s=delay_s,
        )
        diagnostics.append(
            {
                "reference_frame": reference,
                "output_frame": output_frame,
                "one_step_s": one_step_s,
                "delay_s": delay_s,
                **detail,
            }
        )
    frame_list = [int(frame) for frame in frames]
    arrays = _pack_rows(
        frame_list,
        frame_times,
        [rows_by_frame.get(frame, _empty_row()) for frame in frame_list],
    )
    atomic_npz(paths["ledger"], **arrays)
    manifest = {
        "schema": LEDGER_SCHEMA,
        "truth_blind": True,
        "oracle": False,
        "sequence": x1.SEQUENCE,
        "motion_source": "deterministic rigid-cluster multi-scan occupancy vote",
        "online_information_boundary": (
            "output frame t consumes reference t-2 estimated from scans t-4 through t"
        ),
        "scan_offsets": list(SCAN_OFFSETS),
        "lag_scans": x1c.LAG_SCANS,
        "source_contract": {
            "roi": "UNCHANGED_X1_REAR_ROI",
            "dbscan": "UNCHANGED_X1",
            "distance_truncation": "UNCHANGED_X1",
            "flow_norm": "UNCHANGED_X1_FINAL_WEIGHT",
            "component_representation": "ONE_RIGID_DISPLACEMENT_PER_DBSCAN_COMPONENT",
            "noise": "UNKNOWN_NO_MOTION_CELL",
            "candidate_rule": "ZERO_PLUS_FOUR_SCAN_MEDIANS_PLUS_MEDIAN_OF_SCAN_MEDIANS",
            "candidate_tie_break": "OBJECTIVE_NORM_FORWARD_LEFT_HEIGHT_LEXICOGRAPHIC",
        },
        "backend": _backend(),
        "frozen_downstream": {
            "lag_transport": "UNCHANGED_X1C",
            "motion_bounds": "UNCHANGED_R7",
            "route_entry_geometry": "UNCHANGED_R7",
            "event_scorer": "UNCHANGED_X1C",
        },
        "source": {
            "bag_sha256": sha256_file(args.positive_bag.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "baseline_ledger_sha256": sha256_file(
                args.positive_baseline.resolve(strict=True)
            ),
        },
        "diagnostics": {
            "frames": diagnostics,
            "source_seconds": sum(float(row["seconds"]) for row in diagnostics),
            "output_cells": int(len(arrays["forward_m"])),
            "lidar": lidar,
        },
        "canonical_array_sha256": _canonical_array_hash(arrays),
        "ledger": str(paths["ledger"]),
        "ledger_sha256": sha256_file(paths["ledger"]),
    }
    write_json(paths["manifest"], manifest)
    return manifest


def _materialize_error_slice(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    units = x2._selected_units(x0)
    require(len(units) == 35, "x4_error_unit_count")
    by_sequence: dict[str, set[int]] = defaultdict(set)
    for unit in units:
        by_sequence[str(unit["sequence"])].add(int(unit["frame"]))
    manifests = []
    for sequence in sorted(by_sequence):
        output_frames = sorted(by_sequence[sequence])
        required_frames = sorted(
            {
                frame + offset
                for frame in output_frames
                for offset in (-4, -3, -2, -1, 0)
            }
        )
        baseline = (
            args.baseline_root.resolve(strict=True)
            / sequence
            / "m1-pd.raw-point-direct-velocity.npz"
        )
        with np.load(baseline, allow_pickle=False) as values:
            available = {
                int(frame): float(stamp)
                for frame, stamp in zip(values["frames"], values["frame_time_s"])
            }
        require(set(required_frames) <= set(available), f"x4_frames:{sequence}")
        timestamps = {frame: available[frame] for frame in required_frames}
        bag = args.bag_root.resolve(strict=True) / f"{sequence}.bag"
        frames, frame_times, poses, world_clouds, lidar = load_world_clouds(
            bag_path=bag,
            timestamps_path=args.timestamps.resolve(strict=True),
            calibration_dir=args.calibration_dir.resolve(strict=True),
            timestamps_override=timestamps,
        )
        cloud_by_frame = {int(frame): cloud for frame, cloud in zip(frames, world_clouds)}
        rows = []
        diagnostics = []
        for output_frame in output_frames:
            reference = output_frame - x1c.LAG_SCANS
            pose = poses[reference]
            current, counts = x1._voxel_centroids(
                x1._local_cloud(cloud_by_frame[reference], pose)
            )
            supports = []
            for offset in SCAN_OFFSETS:
                support, _support_counts = x1._voxel_centroids(
                    x1._local_cloud(
                        cloud_by_frame[reference + offset], pose, margin_m=1.5
                    )
                )
                supports.append(support)
            require(
                len(current) > 0 and all(len(value) > 0 for value in supports),
                f"x4_empty_cloud:{sequence}:{output_frame}",
            )
            local_times = [
                frame_times[frame] for frame in range(output_frame - 4, output_frame + 1)
            ]
            one_step_s = float(np.median(np.diff(local_times)))
            row, detail = _cluster_vote_frame(
                current, counts, supports, one_step_s=one_step_s
            )
            delay_s = frame_times[output_frame] - frame_times[reference]
            require(delay_s > 0.0, f"x4_delay:{sequence}:{output_frame}")
            rows.append(
                _transport_row(
                    row,
                    reference_pose=poses[reference],
                    output_pose=poses[output_frame],
                    delay_s=delay_s,
                )
            )
            diagnostics.append(
                {
                    "reference_frame": reference,
                    "output_frame": output_frame,
                    "one_step_s": one_step_s,
                    "delay_s": delay_s,
                    **detail,
                }
            )
        arrays = _pack_rows(output_frames, frame_times, rows)
        ledger_path, manifest_path = _error_paths(root, sequence)
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
            "motion_source": "deterministic rigid-cluster multi-scan occupancy vote",
            "backend": _backend(),
            "frozen_downstream": {
                "lag_transport": "UNCHANGED_X1C",
                "motion_bounds": "UNCHANGED_R7",
                "route_entry_geometry": "UNCHANGED_R7",
                "event_scorer": "UNCHANGED_X2",
            },
            "source": {
                "bag_sha256": sha256_file(bag),
                "baseline_ledger_sha256": sha256_file(baseline),
                "x0_result_sha256": sha256_file(x0_path),
            },
            "diagnostics": {
                "frames": diagnostics,
                "source_seconds": sum(float(row["seconds"]) for row in diagnostics),
                "output_cells": int(len(arrays["forward_m"])),
                "lidar": lidar,
            },
            "canonical_array_sha256": _canonical_array_hash(arrays),
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
        write_json(manifest_path, manifest)
        manifests.append(manifest)
    return manifests


def materialize(args: argparse.Namespace, root: Path | None = None) -> dict[str, Any]:
    output_root = (root or args.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    positive = _materialize_positive(args, output_root)
    errors = _materialize_error_slice(args, output_root)
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "truth_blind_source": True,
        "root": str(output_root),
        "positive_frames": len(positive["diagnostics"]["frames"]),
        "error_units": sum(len(row["output_frames"]) for row in errors),
        "sequences": len(errors),
        "elapsed_s": time.perf_counter() - started,
        "canonical_array_sha256": {
            "positive": positive["canonical_array_sha256"],
            **{
                row["sequence"]: row["canonical_array_sha256"] for row in errors
            },
        },
    }
    write_json(output_root / "materialization.json", receipt)
    return receipt


def _score_positive(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    paths = _positive_paths(root)
    ledger = x1._load_sealed(paths["ledger"], paths["manifest"], LEDGER_SCHEMA)
    frames = [int(value) for value in ledger["frames"]]
    timestamps = {
        int(frame): float(stamp)
        for frame, stamp in zip(ledger["frames"], ledger["frame_time_s"])
    }
    pose_samples, _rgb, _authority = read_bag_pose_and_rgb(
        args.positive_bag.resolve(strict=True)
    )
    poses = {
        frame: _causal_pose(pose_samples, round(timestamps[frame] * 1e9))
        for frame in frames
    }
    boxes = load_native_boxes(
        args.labels.resolve(strict=True), timestamps, poses, sequence=x1.SEQUENCE
    )
    result = x1._diagnose(
        ledger,
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=boxes,
        history=_box_history(boxes),
        poses=poses,
    )
    gate = {
        "at_least_two_correct_frames": result["correct_frames"] >= 2,
        "at_least_two_correct_route_entry_frames": (
            result["correct_route_entry_frames"] >= 2
        ),
    }
    payload = {
        "status": (
            "DTR_X4_CLUSTER_VOTE_POSITIVE_GATE_MET"
            if all(gate.values())
            else "DTR_X4_CLUSTER_VOTE_POSITIVE_GATE_NOT_MET"
        ),
        "lag_compensated_source": result,
        "gate": gate,
        "sources": {
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
    }
    write_json(paths["result"], payload)
    return payload


def _score_error_slice(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    x0 = json.loads(args.x0_result.resolve(strict=True).read_text(encoding="utf-8"))
    units = x2._selected_units(x0)
    ledgers = {}
    for sequence in sorted({str(row["sequence"]) for row in units}):
        ledger_path, manifest_path = _error_paths(root, sequence)
        ledgers[sequence] = x1._load_sealed(
            ledger_path, manifest_path, LEDGER_SCHEMA
        )
    rows = []
    for unit in units:
        risk_cells = x2._risk_cells(
            ledgers[str(unit["sequence"])], int(unit["frame"])
        )
        rows.append(
            {
                **unit,
                "lag_cluster_vote_route_risk_cells": risk_cells,
                "suppressed": risk_cells == 0,
            }
        )
    source_rows = [
        row for row in rows if str(row["primary_cause"]) in x2.SOURCE_FAILURES
    ]
    require(len(source_rows) == 34, "x4_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    required = math.ceil(x2.MINIMUM_SUPPRESSION_RATE * len(source_rows))
    return {
        "status": (
            "DTR_X4_CLUSTER_VOTE_ERROR_SLICE_GATE_MET"
            if suppressed >= required
            else "DTR_X4_CLUSTER_VOTE_ERROR_SLICE_GATE_NOT_MET"
        ),
        "false_error_slice": {
            "all_units": len(rows),
            "source_error_units": len(source_rows),
            "suppressed_source_error_units": suppressed,
            "retained_source_error_units": len(source_rows) - suppressed,
            "suppression_rate": suppressed / len(source_rows),
            "required_suppression_units": required,
        },
        "units": rows,
    }


def _runtime_gate(root: Path) -> dict[str, Any]:
    rows = []
    positive = json.loads(
        _positive_paths(root)["manifest"].read_text(encoding="utf-8")
    )
    rows.extend(positive["diagnostics"]["frames"])
    error_root = root / "error-slice"
    for manifest_path in sorted(error_root.glob("*/lag-cluster-vote.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.extend(manifest["diagnostics"]["frames"])
    seconds = np.asarray([float(row["seconds"]) for row in rows], dtype=np.float64)
    periods = np.asarray([float(row["one_step_s"]) for row in rows], dtype=np.float64)
    p95 = float(np.quantile(seconds, 0.95, method="higher"))
    period = float(np.median(periods))
    return {
        "frames": int(len(rows)),
        "source_compute_p95_s": p95,
        "median_observed_scan_period_s": period,
        "p95_within_one_observed_scan_period": p95 <= period,
    }


def _effect_signature(positive: Mapping[str, Any], error: Mapping[str, Any]) -> str:
    lagged = positive["lag_compensated_source"]
    payload = {
        "positive": {
            key: lagged[key]
            for key in (
                "associated_frames",
                "correct_frames",
                "correct_route_entry_frames",
                "minimum_associated_error_mps",
            )
        },
        "units": [
            {
                key: row[key]
                for key in (
                    "unit_id",
                    "sequence",
                    "frame",
                    "lag_cluster_vote_route_risk_cells",
                    "suppressed",
                )
            }
            for row in error["units"]
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score(args: argparse.Namespace, root: Path | None = None) -> dict[str, Any]:
    output_root = (root or args.root).resolve(strict=True)
    positive = _score_positive(args, output_root)
    error = _score_error_slice(args, output_root)
    runtime = _runtime_gate(output_root)
    gate = {
        "positive_correct_route_frames_at_least_two": positive["gate"][
            "at_least_two_correct_route_entry_frames"
        ],
        "positive_correct_frames_at_least_two": positive["gate"][
            "at_least_two_correct_frames"
        ],
        "source_error_suppression_at_least_24_of_34": (
            error["false_error_slice"]["suppressed_source_error_units"]
            >= error["false_error_slice"]["required_suppression_units"]
        ),
        "source_compute_p95_within_one_scan_period": runtime[
            "p95_within_one_observed_scan_period"
        ],
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X4_DETERMINISTIC_CLUSTER_VOTE_CANARY_GATE_MET"
            if met
            else "DTR_X4_DETERMINISTIC_CLUSTER_VOTE_CANARY_GATE_NOT_MET"
        ),
        "question": (
            "Can a deterministic rigid-cluster five-scan source preserve X1c positive "
            "headroom, X2 false-source suppression, and one-period compute cost?"
        ),
        "positive": positive,
        "error_slice": error,
        "runtime": runtime,
        "gate": gate,
        "effect_signature_sha256": _effect_signature(positive, error),
        "decision": {
            "headroom_met": met,
            "next": (
                "REQUIRE_THREE_IDENTICAL_COLD_RUNS_BEFORE_FULL_REPLAY"
                if met
                else "CLOSE_DETERMINISTIC_CLUSTER_VOTE_SOURCE"
            ),
        },
        "claim_limits": [
            "Opened Development positive and error slices; not source-disjoint confirmation.",
            "The 35-unit result is frame-local suppression, not full false-segment scoring.",
            "One-period source compute excludes bag loading and downstream application latency.",
            "Full six-sequence replay is forbidden until three cold roots are identical.",
        ],
    }
    write_json(output_root / "result.json", result)
    return result


def _trial_roots(root: Path) -> list[Path]:
    return [root.resolve() / f"run-{index:02d}" for index in range(1, REPEATS + 1)]


def verify_three(args: argparse.Namespace) -> dict[str, Any]:
    roots = _trial_roots(args.root)
    runs = []
    for trial_root in roots:
        receipt = json.loads(
            (trial_root / "materialization.json").read_text(encoding="utf-8")
        )
        result = json.loads((trial_root / "result.json").read_text(encoding="utf-8"))
        runs.append(
            {
                "root": str(trial_root),
                "canonical_array_sha256": receipt["canonical_array_sha256"],
                "effect_signature_sha256": result["effect_signature_sha256"],
                "gate": result["gate"],
                "status": result["status"],
                "runtime": result["runtime"],
            }
        )
    array_signatures = {
        json.dumps(row["canonical_array_sha256"], sort_keys=True) for row in runs
    }
    effect_signatures = {str(row["effect_signature_sha256"]) for row in runs}
    gate = {
        "three_canonical_array_sets_identical": len(array_signatures) == 1,
        "three_effect_signatures_identical": len(effect_signatures) == 1,
        "all_three_effect_and_runtime_gates_met": all(
            row["status"] == "DTR_X4_DETERMINISTIC_CLUSTER_VOTE_CANARY_GATE_MET"
            for row in runs
        ),
    }
    met = all(gate.values())
    payload = {
        "schema": VERIFY_SCHEMA,
        "status": (
            "DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_MET"
            if met
            else "DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_NOT_MET"
        ),
        "runs": runs,
        "gate": gate,
        "decision": {
            "advance_to_full_replay": met,
            "next": (
                "RUN_FULL_SIX_SEQUENCE_DETERMINISTIC_CLUSTER_VOTE_COMPARISON"
                if met
                else "CLOSE_OR_REDESIGN_CLUSTER_VOTE_SOURCE_WITHOUT_PARAMETER_SWEEP"
            ),
        },
    }
    args.root.resolve().mkdir(parents=True, exist_ok=True)
    write_json(args.root.resolve() / "verify3.json", payload)
    return payload


def repeat_three(args: argparse.Namespace) -> dict[str, Any]:
    for trial_root in _trial_roots(args.root):
        require(not trial_root.exists(), f"x4_cold_root_exists:{trial_root}")
        materialize(args, trial_root)
        score(args, trial_root)
    return verify_three(args)


def smoke() -> dict[str, Any]:
    current = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.3, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    displacement = np.asarray([0.05, -0.02, 0.0], dtype=np.float64)
    supports = [current + float(offset) * displacement for offset in SCAN_OFFSETS]
    counts = np.ones(len(current), dtype=np.int32)
    hashes = []
    rows = []
    for _index in range(REPEATS):
        row, detail = _cluster_vote_frame(
            current, counts, supports, one_step_s=0.1
        )
        arrays = _pack_rows([0], {0: 0.0}, [row])
        hashes.append(_canonical_array_hash(arrays))
        rows.append(detail)
    require(len(set(hashes)) == 1, "x4_smoke_nondeterministic")
    require(len(rows[0]["components"]) == 1, "x4_smoke_cluster_count")
    return {
        "status": "DTR_X4_DETERMINISTIC_CLUSTER_VOTE_SMOKE_MET",
        "canonical_array_sha256": hashes[0],
        "output_cells": rows[0]["output_cells"],
        "clusters": rows[0]["clusters"],
    }


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("materialize", "score", "run", "repeat3", "verify3", "smoke"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "evidence"
        / "dtr-x4"
        / "deterministic-cluster-vote-canary",
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
    parser.add_argument(
        "--positive-bag",
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
    parser.add_argument("--baseline-root", type=Path, default=c31 / "baseline-ledgers")
    parser.add_argument(
        "--positive-baseline",
        type=Path,
        default=c31
        / "baseline-ledgers"
        / x1.SEQUENCE
        / "m1-pd.raw-point-direct-velocity.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        payload = smoke()
    elif args.mode == "verify3":
        payload = verify_three(args)
    elif args.mode == "repeat3":
        payload = repeat_three(args)
    else:
        payload = None
        if args.mode in {"materialize", "run"}:
            payload = materialize(args)
        if args.mode in {"score", "run"}:
            payload = score(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
