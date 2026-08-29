"""Continue only RGB-authorized X7 motion for the frozen 0.50 s grace.

X13 showed strong false-source separation but frame-local dynamic authority
arrived before the positive cell entered the route.  This 60-frame falsifier
keeps X13's unchanged C22 authorization rule, then transports an authorized
cell causally with its frozen X7 velocity for the already frozen R1 clear-grace
duration.  No unauthorised cell can originate a continuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_c22_ego_rigid_visual_motion as c22  # noqa: E402
from dtr_r1 import FROZEN_R1_CONFIG  # noqa: E402
import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x6_static_world_persistence_falsifier as x6  # noqa: E402
import dtr_x7_full_static_world_anchor_replay as x7  # noqa: E402
import dtr_x8_rgb_static_veto_falsifier as x8  # noqa: E402
import dtr_x13_stitched_dynamic_birth_authority_falsifier as x13  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402


SCHEMA = "blindassist-dtr-x14-rgb-authorized-motion-continuation-falsifier-v1"
LEDGER_SCHEMA = "blindassist-dtr-x14-rgb-authorized-motion-continuation-ledger-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x14-rgb-authorized-motion-continuation-materialization-v1"
CONTINUATION_S = FROZEN_R1_CONFIG.clear_grace_s


def _sequence_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "rgb-authorized-continuation.npz", base / "rgb-authorized-continuation.json"


def _source_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _history_frames(frames: list[int], times: dict[int, float], output: int) -> list[int]:
    index = frames.index(output)
    selected = []
    for frame in reversed(frames[: index + 1]):
        if times[output] - times[frame] > CONTINUATION_S + 1e-9:
            break
        selected.append(frame)
    return list(reversed(selected))


def _transport(
    row: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    source_pose: dict[str, Any],
    target_pose: dict[str, Any],
    delta_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions, velocities, counts, support = row
    if not len(positions):
        return positions, velocities, counts, support
    world_positions, world_velocities = x5._ego_to_world(positions.astype(np.float64), velocities.astype(np.float64), source_pose)
    future_world = world_positions + world_velocities * float(delta_s)
    local = c22._world_to_ego_xy(future_world, target_pose)
    yaw = float(target_pose["yaw_rad"])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_velocity = np.column_stack((cosine * world_velocities[:, 0] + sine * world_velocities[:, 1], -sine * world_velocities[:, 0] + cosine * world_velocities[:, 1]))
    return local, local_velocity, counts, support


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    x0_path = args.x0_result.resolve(strict=True)
    selected = x5._selected_frames(json.loads(x0_path.read_text(encoding="utf-8")))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    calibration = c22.load_calibration(args.calibration_dir.resolve(strict=True))
    manifests = []
    frame_seconds = []
    for sequence in sorted(selected):
        source_path, source_manifest_path = _source_paths(args.x7_root.resolve(strict=True), sequence)
        source = x1._load_sealed(source_path, source_manifest_path, x7.LEDGER_SCHEMA)
        source_frames = [int(frame) for frame in source["frames"]]
        times = {int(frame): float(stamp) for frame, stamp in zip(source["frames"], source["frame_time_s"])}
        output_frames = sorted(selected[sequence])
        histories = {frame: _history_frames(source_frames, times, frame) for frame in output_frames}
        authority_frames = sorted({item for values in histories.values() for item in values})
        image_frames = sorted(set(authority_frames) | {frame - 1 for frame in authority_frames})
        require(set(image_frames) <= set(source_frames), f"x14_source_frames:{sequence}")
        target_ns = {frame: round(times[frame] * 1e9) for frame in image_frames}
        bag_path = args.bag_root.resolve(strict=True) / f"{sequence}.bag"
        images, pose_samples, camera = x8._read_visual_context(bag_path, target_ns)
        poses = {}
        for frame in image_frames:
            try:
                poses[frame] = c22.interpolate_pose(pose_samples, target_ns[frame])
            except (AssertionError, RuntimeError, ValueError):
                pass
        authorized = {}
        authorization_diagnostics = []
        for frame in authority_frames:
            current = x5._frame_arrays(source, frame)
            previous = frame - 1
            started = time.perf_counter()
            missing = [name for name, present in (("PREVIOUS_RGB", previous in images), ("CURRENT_RGB", frame in images), ("PREVIOUS_POSE", previous in poses), ("CURRENT_POSE", frame in poses)) if not present]
            if missing:
                confidence = np.zeros(len(current[0]), dtype=np.float32)
                diag = {"tracks": 0, "valid_dynamic_tracks": 0, "dynamic_cells": 0}
            else:
                confidence, diag = x13._dynamic_confidence(
                    previous_gray=images[previous],
                    current_gray=images[frame],
                    previous_pose=poses[previous],
                    current_pose=poses[frame],
                    dt_s=times[frame] - times[previous],
                    row=current,
                    calibration=calibration,
                )
            keep = np.flatnonzero(confidence >= c22.DECISION_CONFIDENCE)
            authorized[frame] = (current[0][keep], current[1][keep], current[2][keep], current[3][keep])
            seconds = time.perf_counter() - started
            frame_seconds.append(seconds)
            authorization_diagnostics.append({"frame": frame, "input_cells": int(len(current[0])), "authorized_cells": int(len(keep)), "missing_visual_evidence": missing, "seconds": seconds, **diag})
        rows = []
        continuation_diagnostics = []
        for output in output_frames:
            pieces = []
            for source_frame in histories[output]:
                if source_frame not in poses or output not in poses:
                    continue
                pieces.append(_transport(authorized[source_frame], source_pose=poses[source_frame], target_pose=poses[output], delta_s=times[output] - times[source_frame]))
            if pieces:
                row = tuple(np.concatenate([piece[index] for piece in pieces], axis=0) for index in range(4))
            else:
                row = (np.empty((0, 2), np.float64), np.empty((0, 2), np.float64), np.empty(0, np.int32), np.empty(0, np.float32))
            rows.append(row)
            continuation_diagnostics.append({"frame": output, "history_first_frame": histories[output][0], "history_frames": len(histories[output]), "continued_cells": int(len(row[0]))})
        arrays = x5._pack_rows(output_frames, times, rows)
        ledger_path, manifest_path = _sequence_paths(root, sequence)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(ledger_path, **arrays)
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "selection_post_outcome": True,
            "sequence": sequence,
            "birth_rule": "UNCHANGED_X13_C22_DYNAMIC_AUTHORITY",
            "continuation_rule": "TRANSPORT_AUTHORIZED_CELL_AT_FROZEN_X7_VELOCITY_FOR_R1_CLEAR_GRACE",
            "continuation_s": CONTINUATION_S,
            "missing_evidence_policy": "NO_NEW_AUTHORITY",
            "frozen_downstream": {"cell_velocity": "UNCHANGED_X7", "route_entry_geometry": "UNCHANGED_R7", "event_scorer": "UNCHANGED_X6_FRAME_LOCAL_FALSIFIER"},
            "camera_audit": camera,
            "source": {"x7_ledger_sha256": sha256_file(source_path), "x7_manifest_sha256": sha256_file(source_manifest_path), "bag": str(bag_path), "bag_sha256": sha256_file(bag_path)},
            "diagnostics": {"authorization_frames": authorization_diagnostics, "output_frames": continuation_diagnostics, "authorized_cells": sum(row["authorized_cells"] for row in authorization_diagnostics), "continued_cells": sum(row["continued_cells"] for row in continuation_diagnostics)},
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
        write_json(manifest_path, manifest)
        manifests.append(manifest)
    receipt = {
        "schema": MATERIALIZATION_SCHEMA,
        "truth_blind": True,
        "selection_post_outcome": True,
        "sequences": len(manifests),
        "output_frames": sum(len(row["diagnostics"]["output_frames"]) for row in manifests),
        "authorization_frames": sum(len(row["diagnostics"]["authorization_frames"]) for row in manifests),
        "authorized_cells": sum(row["diagnostics"]["authorized_cells"] for row in manifests),
        "continued_cells": sum(row["diagnostics"]["continued_cells"] for row in manifests),
        "continuation_s": CONTINUATION_S,
        "source_compute_p95_s": float(np.quantile(np.asarray(frame_seconds), 0.95, method="higher")),
        "runtime_boundary": "projection, LK, dynamic authorization, and bounded cell transport; bag scan/image matching/decode excluded",
        "backend": {"kind": "cpu", "opencv": cv2.__version__, "processor": platform.processor()},
        "sequence_manifests": {row["sequence"]: sha256_file(_sequence_paths(root, row["sequence"])[1]) for row in manifests},
    }
    write_json(root / "materialization.json", receipt)
    return receipt


def score(args: argparse.Namespace) -> dict[str, Any]:
    x0_path = args.x0_result.resolve(strict=True)
    x0 = json.loads(x0_path.read_text(encoding="utf-8"))
    units = x2._selected_units(x0)
    sequences = [str(row["sequence"]) for row in units] + [x1.SEQUENCE]
    ledgers = {}
    for sequence in sorted(set(sequences)):
        path, manifest = _sequence_paths(args.root.resolve(strict=True), sequence)
        ledgers[sequence] = x1._load_sealed(path, manifest, LEDGER_SCHEMA)
    positive = x5._score_positive(args, ledgers[x1.SEQUENCE])
    rows = []
    for unit in units:
        risk_cells = x2._risk_cells(ledgers[str(unit["sequence"])], int(unit["frame"]))
        rows.append({**unit, "authorized_continuation_route_risk_cells": risk_cells, "suppressed": risk_cells == 0})
    source_rows = [row for row in rows if str(row["primary_cause"]) in x2.SOURCE_FAILURES]
    require(len(source_rows) == 34, "x14_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    required = int(np.ceil(x2.MINIMUM_SUPPRESSION_RATE * len(source_rows)))
    materialization = json.loads((args.root.resolve(strict=True) / "materialization.json").read_text(encoding="utf-8"))
    p95_s = float(materialization["source_compute_p95_s"])
    gate = {
        "positive_correct_frames_at_least_two": positive["correct_frames"] >= 2,
        "positive_correct_route_frames_at_least_two": positive["correct_route_entry_frames"] >= 2,
        "source_error_suppression_at_least_24_of_34": suppressed >= required,
        "source_compute_p95_within_one_scan_period": p95_s <= x6.SOURCE_COMPUTE_BUDGET_S,
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": "DTR_X14_RGB_AUTHORIZED_MOTION_CONTINUATION_FALSIFIER_GATE_MET" if met else "DTR_X14_RGB_AUTHORIZED_MOTION_CONTINUATION_FALSIFIER_GATE_NOT_MET",
        "question": "Can a frozen 0.50-second causal continuation turn early RGB dynamic authority into route risk without reopening false births?",
        "positive": positive,
        "error_slice": {"source_error_units": len(source_rows), "suppressed_source_error_units": suppressed, "retained_source_error_units": len(source_rows) - suppressed, "suppression_rate": suppressed / len(source_rows), "required_suppression_units": required},
        "units": rows,
        "gate": gate,
        "runtime": {"source_compute_p95_s": p95_s, "median_observed_scan_period_s": x6.SOURCE_COMPUTE_BUDGET_S, "boundary": materialization["runtime_boundary"]},
        "decision": {"mechanism_headroom": met, "next": "IMPLEMENT_FULL_RGB_AUTHORIZED_CONTINUATION_REPLAY" if met else "CLOSE_RGB_AUTHORIZED_CONTINUATION_WITHOUT_PARAMETER_SWEEP"},
        "claim_limits": ["Post-outcome diagnostic on the opened X6 60-frame roster; not confirmation.", "Only already RGB-authorized motion can continue; continuation cannot originate authority.", "The 0.50-second duration is reused from frozen R1 clear grace, not selected from this outcome."],
        "sources": {"x7_result_sha256": sha256_file(args.x7_result.resolve(strict=True)), "x0_result_sha256": sha256_file(x0_path), "labels_sha256": sha256_file(args.labels.resolve(strict=True))},
    }
    write_json(args.root.resolve(strict=True) / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x14" / "rgb-authorized-motion-continuation-falsifier-20260829")
    parser.add_argument("--x7-root", type=Path, default=x7_root)
    parser.add_argument("--x7-result", type=Path, default=x7_root / "result.json")
    parser.add_argument("--x0-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x0" / "motion-source-attribution" / "result.json")
    parser.add_argument("--bag-root", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-c31-jrdb-fresh-confirmation")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--calibration-dir", type=Path, default=REPO / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = None
    if args.mode in {"materialize", "run"}:
        payload = materialize(args)
    if args.mode in {"score", "run"}:
        payload = score(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
