"""Require independent stitched-RGB dynamic evidence to admit X7 cells.

X8 used RGB only as a negative static veto and retained candidates whenever
visual evidence was absent.  This post-outcome 60-frame falsifier changes the
authority role: a cell is admitted only when valid causal RGB motion is closer
to the velocity-conditioned moving reprojection than to ego-rigid static and
passes the unchanged C22 confidence rule.  It is a frame-local birth-authority
test; lifecycle continuation is intentionally not tested here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_c22_ego_rigid_visual_motion as c22  # noqa: E402
import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x2_floxel_error_slice_canary as x2  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x6_static_world_persistence_falsifier as x6  # noqa: E402
import dtr_x7_full_static_world_anchor_replay as x7  # noqa: E402
import dtr_x8_rgb_static_veto_falsifier as x8  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402


SCHEMA = "blindassist-dtr-x13-stitched-dynamic-birth-authority-falsifier-v1"
LEDGER_SCHEMA = "blindassist-dtr-x13-stitched-dynamic-birth-authority-ledger-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x13-stitched-dynamic-birth-authority-materialization-v1"


def _sequence_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "stitched-dynamic-authority.npz", base / "stitched-dynamic-authority.json"


def _source_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _dynamic_confidence(
    *,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_pose: dict[str, Any],
    current_pose: dict[str, Any],
    dt_s: float,
    row: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    positions, velocities, _counts, _support = row
    count = len(positions)
    if not count:
        return np.empty(0, dtype=np.float32), {"tracks": 0, "valid_dynamic_tracks": 0, "dynamic_cells": 0}
    current_world_xy, velocity_world = x5._ego_to_world(positions.astype(np.float64), velocities.astype(np.float64), current_pose)
    previous_world_xy = current_world_xy - velocity_world * float(dt_s)
    anchors = np.asarray(c22.HEIGHT_ANCHORS_M, dtype=np.float64)

    def expand(xy: np.ndarray) -> np.ndarray:
        return np.column_stack((np.repeat(xy[:, 0], len(anchors)), np.repeat(xy[:, 1], len(anchors)), np.tile(anchors, count)))

    previous_world = expand(previous_world_xy)
    current_world = expand(current_world_xy)
    source = c22._project(previous_world, previous_pose, calibration)
    rigid = c22._project(previous_world, current_pose, calibration)
    moving = c22._project(current_world, current_pose, calibration)
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(rigid), axis=1) & np.all(np.isfinite(moving), axis=1)
    finite &= (source[:, 1] >= 2.0) & (source[:, 1] < c22.IMAGE_HEIGHT - 2.0)
    indices = np.flatnonzero(finite)
    confidence = np.zeros(len(source), dtype=np.float64)
    dynamic_closer = np.zeros(len(source), dtype=bool)
    if len(indices):
        points = source[indices]
        actual, valid, fb_error, _lk_error = c22._lk_tracks(previous_gray, current_gray, points)
        rigid_u = c22._unwrap(rigid[indices, 0], points[:, 0])
        moving_u = c22._unwrap(moving[indices, 0], points[:, 0])
        actual_u = c22._unwrap(actual[:, 0], points[:, 0])
        static_error = np.linalg.norm(np.column_stack((actual_u - rigid_u, actual[:, 1] - rigid[indices, 1])), axis=1)
        moving_error = np.linalg.norm(np.column_stack((actual_u - moving_u, actual[:, 1] - moving[indices, 1])), axis=1)
        proposed_motion = np.linalg.norm(np.column_stack((moving_u - rigid_u, moving[indices, 1] - rigid[indices, 1])), axis=1)
        q = np.exp(-0.5 * (fb_error / c22.FB_SIGMA_PX) ** 2)
        q *= np.exp(-0.5 * (moving_error / c22.AGREEMENT_SIGMA_PX) ** 2)
        q *= 1.0 - np.exp(-0.5 * (proposed_motion / c22.MOTION_SIGMA_PX) ** 2)
        closer = moving_error < static_error
        q[~valid | ~closer] = 0.0
        confidence[indices] = q
        dynamic_closer[indices] = closer & valid
    samples = confidence.reshape(count, len(anchors))
    output = np.zeros(count, dtype=np.float32)
    for index, values in enumerate(samples):
        positive = values[values > 0.0]
        if len(positive):
            output[index] = float(np.quantile(positive, c22.COMPONENT_QUANTILE) * (len(positive) / len(values)))
    return output, {
        "tracks": int(len(indices)),
        "valid_dynamic_tracks": int(np.count_nonzero(confidence > 0.0)),
        "dynamic_closer_tracks": int(np.count_nonzero(dynamic_closer)),
        "dynamic_cells": int(np.count_nonzero(output >= c22.DECISION_CONFIDENCE)),
    }


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
        timestamp_by_frame = {int(frame): float(stamp) for frame, stamp in zip(source["frames"], source["frame_time_s"])}
        output_frames = sorted(selected[sequence])
        needed = sorted(set(output_frames) | {frame - 1 for frame in output_frames})
        require(set(needed) <= set(source_frames), f"x13_source_frames:{sequence}")
        target_ns = {frame: round(timestamp_by_frame[frame] * 1e9) for frame in needed}
        bag_path = args.bag_root.resolve(strict=True) / f"{sequence}.bag"
        images, pose_samples, camera = x8._read_visual_context(bag_path, target_ns)
        poses = {}
        for frame in needed:
            try:
                poses[frame] = c22.interpolate_pose(pose_samples, target_ns[frame])
            except (AssertionError, RuntimeError, ValueError):
                pass
        rows = []
        diagnostics = []
        for frame in output_frames:
            current = x5._frame_arrays(source, frame)
            previous = frame - 1
            started = time.perf_counter()
            missing = []
            if previous not in images:
                missing.append("PREVIOUS_RGB")
            if frame not in images:
                missing.append("CURRENT_RGB")
            if previous not in poses:
                missing.append("PREVIOUS_POSE")
            if frame not in poses:
                missing.append("CURRENT_POSE")
            if missing:
                confidence = np.zeros(len(current[0]), dtype=np.float32)
                diag = {"tracks": 0, "valid_dynamic_tracks": 0, "dynamic_cells": 0}
            else:
                confidence, diag = _dynamic_confidence(
                    previous_gray=images[previous],
                    current_gray=images[frame],
                    previous_pose=poses[previous],
                    current_pose=poses[frame],
                    dt_s=timestamp_by_frame[frame] - timestamp_by_frame[previous],
                    row=current,
                    calibration=calibration,
                )
            admitted = confidence >= c22.DECISION_CONFIDENCE
            keep = np.flatnonzero(admitted)
            seconds = time.perf_counter() - started
            frame_seconds.append(seconds)
            rows.append((current[0][keep], current[1][keep], current[2][keep], current[3][keep]))
            diagnostics.append({
                "frame": frame,
                "previous_frame": previous,
                "input_cells": int(len(current[0])),
                "admitted_dynamic_cells": int(len(keep)),
                "rejected_without_dynamic_authority": int(len(current[0]) - len(keep)),
                "missing_visual_evidence": missing,
                "seconds": seconds,
                **diag,
            })
        arrays = x5._pack_rows(output_frames, timestamp_by_frame, rows)
        ledger_path, manifest_path = _sequence_paths(root, sequence)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(ledger_path, **arrays)
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "selection_post_outcome": True,
            "sequence": sequence,
            "rule": "ADMIT_ONLY_IF_RGB_DYNAMIC_CONFIDENCE_AT_LEAST_C22_DECISION_AND_MOVING_CLOSER_THAN_STATIC",
            "missing_evidence_policy": "REJECT_NEW_DYNAMIC_AUTHORITY",
            "authority_role": "FRAME_LOCAL_BIRTH_ONLY_NO_CONTINUATION_TESTED",
            "fixed_visual_constants": {
                "height_anchors_m": list(c22.HEIGHT_ANCHORS_M),
                "forward_backward_sigma_px": c22.FB_SIGMA_PX,
                "moving_agreement_sigma_px": c22.AGREEMENT_SIGMA_PX,
                "motion_strength_sigma_px": c22.MOTION_SIGMA_PX,
                "cell_quantile": c22.COMPONENT_QUANTILE,
                "decision_confidence": c22.DECISION_CONFIDENCE,
                "lk_window": list(c22.LK_WINDOW),
                "lk_levels": c22.LK_LEVELS,
            },
            "frozen_downstream": {"cell_velocity": "UNCHANGED_X7", "route_entry_geometry": "UNCHANGED_R7", "event_scorer": "UNCHANGED_X6_FRAME_LOCAL_FALSIFIER"},
            "camera_audit": camera,
            "source": {"x7_ledger_sha256": sha256_file(source_path), "x7_manifest_sha256": sha256_file(source_manifest_path), "bag": str(bag_path), "bag_sha256": sha256_file(bag_path)},
            "diagnostics": {
                "frames": diagnostics,
                "input_cells": sum(row["input_cells"] for row in diagnostics),
                "admitted_dynamic_cells": sum(row["admitted_dynamic_cells"] for row in diagnostics),
                "rejected_without_dynamic_authority": sum(row["rejected_without_dynamic_authority"] for row in diagnostics),
            },
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
        "frames": sum(len(row["diagnostics"]["frames"]) for row in manifests),
        "input_cells": sum(row["diagnostics"]["input_cells"] for row in manifests),
        "admitted_dynamic_cells": sum(row["diagnostics"]["admitted_dynamic_cells"] for row in manifests),
        "source_compute_p95_s": float(np.quantile(np.asarray(frame_seconds), 0.95, method="higher")),
        "runtime_boundary": "projection, LK, dynamic confidence, and admission; bag scan/image matching/decode excluded",
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
        rows.append({**unit, "dynamic_authority_route_risk_cells": risk_cells, "suppressed": risk_cells == 0})
    source_rows = [row for row in rows if str(row["primary_cause"]) in x2.SOURCE_FAILURES]
    require(len(source_rows) == 34, "x13_source_error_count")
    suppressed = sum(bool(row["suppressed"]) for row in source_rows)
    required = math.ceil(x2.MINIMUM_SUPPRESSION_RATE * len(source_rows))
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
        "status": "DTR_X13_STITCHED_DYNAMIC_BIRTH_AUTHORITY_FALSIFIER_GATE_MET" if met else "DTR_X13_STITCHED_DYNAMIC_BIRTH_AUTHORITY_FALSIFIER_GATE_NOT_MET",
        "question": "Can sensor-disjoint RGB positively authorize X7 dynamic risk births instead of merely vetoing static cells?",
        "positive": positive,
        "error_slice": {"source_error_units": len(source_rows), "suppressed_source_error_units": suppressed, "retained_source_error_units": len(source_rows) - suppressed, "suppression_rate": suppressed / len(source_rows), "required_suppression_units": required},
        "units": rows,
        "gate": gate,
        "runtime": {"source_compute_p95_s": p95_s, "median_observed_scan_period_s": x6.SOURCE_COMPUTE_BUDGET_S, "boundary": materialization["runtime_boundary"]},
        "decision": {"mechanism_headroom": met, "next": "IMPLEMENT_FULL_DYNAMIC_BIRTH_WITH_BOUNDED_CONTINUATION" if met else "CLOSE_STITCHED_DYNAMIC_BIRTH_AUTHORITY_WITHOUT_PARAMETER_SWEEP"},
        "claim_limits": [
            "Post-outcome diagnostic on the opened X6 60-frame roster; not confirmation.",
            "This tests only frame-local birth authority, not bounded continuation after a previously authorized birth.",
            "Missing images, invalid projections, and failed tracks reject new dynamic authority.",
        ],
        "sources": {"x7_result_sha256": sha256_file(args.x7_result.resolve(strict=True)), "x0_result_sha256": sha256_file(x0_path), "labels_sha256": sha256_file(args.labels.resolve(strict=True))},
    }
    write_json(args.root.resolve(strict=True) / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x13" / "stitched-dynamic-birth-authority-falsifier-20260829")
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
