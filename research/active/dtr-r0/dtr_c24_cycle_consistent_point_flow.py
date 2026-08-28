"""Evaluate cell-local cycle-consistent raw-LiDAR point flow on the R7 canary.

C23 proved that component rigidity can preserve all induced-dropout recovery,
but it still broadcasts one component translation to every cell.  C24 changes
the information representation, not the route decision: reciprocal raw-point
correspondences produce cell-local direct velocity, and an independent prior
flow observation must reproject to the current cell with consistent velocity.

The raw point and temporal-confidence builders are the already exercised M1-PD
and M1-PDC implementations.  This wrapper applies them to the consumed R7
Packard Development window and leaves route geometry, thresholds, lifecycle,
target truth, and fresh cohorts unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_m1_confident_direct_velocity as confident_flow
import dtr_m1_raw_point_direct_velocity as point_flow
import dtr_r7_occupancy_flow_canary as r7
from dtr_r5_dropout_canary import cases_from_tracks
from jrdb_rgb_bridge import (
    FIRST_FRAME,
    LAST_FRAME,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import (
    load_truth_and_associate,
    read_jsonl,
    write_json,
)
from jrdb_range_acquire import sha256_file


SCHEMA = "blindassist-dtr-c24-cycle-consistent-point-flow-v1"
STATUS_MET = "DTR_C24_CYCLE_CONSISTENT_POINT_FLOW_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C24_CYCLE_CONSISTENT_POINT_FLOW_DEVELOPMENT_GATE_NOT_MET"
TARGET_FALSE_SEGMENTS = 14


def _evaluate(cases: Any, ledger: r7.FlowLedger) -> dict[str, Any]:
    original = r7.evaluate_original(cases, ledger)
    stress = r7.evaluate_stress(cases, ledger)
    nuisance = r7.global_nuisance(cases, ledger)
    recovered = sum(
        row["occupancy_flow"]["recovered_track_only_window_misses"]
        for row in stress.values()
    )
    return {
        "original": original,
        "stress": stress,
        "global_nuisance": nuisance,
        "recovered_window_misses": recovered,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    r7_result_path = args.r7_result.resolve(strict=True)
    r7_result = json.loads(r7_result_path.read_text(encoding="utf-8"))
    bag_path = Path(r7_result["source"]["bag"]).resolve(strict=True)
    timestamps_path = Path(r7_result["source"]["timestamps"]).resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    sequence = str(r7_result["source"]["sequence"])
    frames = list(range(FIRST_FRAME, LAST_FRAME + 1))
    timestamps_all = load_image_timestamps(timestamps_path)
    timestamps = {frame: timestamps_all[frame] for frame in frames}

    output_path = args.output.resolve()
    ledger_root = output_path.parent
    point_npz, point_manifest = point_flow.ledger_paths(ledger_root / "m1-pd.json")
    if not (args.reuse_ledgers and point_npz.exists() and point_manifest.exists()):
        point_flow.materialize(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            calibration_dir=calibration_dir,
            output_path=point_npz,
            manifest_path=point_manifest,
            backend_receipt_path=args.backend_receipt.resolve(),
            sequence=sequence,
            timestamps_override=timestamps,
        )
    point = point_flow.load_ledger(
        point_npz,
        point_manifest,
        expected_sequence=sequence,
        expected_frames=frames,
    )

    confident_npz, confident_manifest = confident_flow.ledger_paths(
        ledger_root / "m1-pdc.json"
    )
    if not (
        args.reuse_ledgers and confident_npz.exists() and confident_manifest.exists()
    ):
        confident_flow.materialize(
            source_path=point_npz,
            source_manifest_path=point_manifest,
            output_path=confident_npz,
            manifest_path=confident_manifest,
        )
    confident = confident_flow.load_ledger(
        confident_npz,
        confident_manifest,
        expected_sequence=sequence,
        expected_frames=frames,
    )

    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in frames
    }
    known_tracks = Path(r7_result["source"]["known_height_tracks"]).resolve(
        strict=True
    )
    labels = Path(r7_result["source"]["labels"]).resolve(strict=True)
    tracks, geometry_quality = load_truth_and_associate(
        labels, read_jsonl(known_tracks), context
    )
    cases = cases_from_tracks(tracks)
    raw = _evaluate(cases, point)
    cycle = _evaluate(cases, confident)
    baseline = r7_result["original_cohort"]["r7_p_occupancy_flow"]
    score = cycle["original"]
    checks = {
        "preserves_all_nine_dropout_recoveries": cycle["recovered_window_misses"]
        == 9,
        "critical_event_recall_not_lower": score["critical_event_recall"]
        >= baseline["critical_event_recall"],
        "false_segments_at_most_fourteen": score["false_alert_segments"]
        <= TARGET_FALSE_SEGMENTS,
        "event_f1_higher": score["event_detection_f1"]
        > baseline["event_detection_f1"],
    }
    passed = all(checks.values())
    result = {
        "schema_version": SCHEMA,
        "status": STATUS_MET if passed else STATUS_NOT_MET,
        "question": (
            "Can cell-local reciprocal raw-point velocity plus independent "
            "three-frame cycle consistency retain R7 recovery while removing "
            "component-motion broadcast false alerts?"
        ),
        "source": {
            "r7_result": str(r7_result_path),
            "r7_result_sha256": sha256_file(r7_result_path),
            "bag": str(bag_path),
            "bag_authority": bag_authority,
            "calibration_dir": str(calibration_dir),
            "calibration_sha256": sha256_file(calibration_dir / "lidars.yaml"),
        },
        "arms": {
            "r7_component_translation": baseline,
            "m1_pd_cell_local_direct_velocity": raw,
            "m1_pdc_three_frame_cycle_consistency": cycle,
        },
        "ledgers": {
            "point": json.loads(point_manifest.read_text(encoding="utf-8")),
            "cycle_consistent": json.loads(
                confident_manifest.read_text(encoding="utf-8")
            ),
        },
        "gate": {"passed": passed, "checks": checks},
        "tradeoffs": {
            "median_first_alert_lead_change_s": score["median_first_alert_lead_s"]
            - baseline["median_first_alert_lead_s"],
            "median_escalation_lead_change_s": score["median_escalation_lead_s"]
            - baseline["median_escalation_lead_s"],
            "interpretation": (
                "The M1 gate follows the requested recovery/false-alert target. "
                "Lead remains separately reported and is not hidden inside pass/fail."
            ),
        },
        "evaluator_firewall": {
            "motion": (
                "sealed from causal current/past raw LiDAR, ego pose, and "
                "calibration before labels"
            ),
            "labels": (
                "opened only for target attribution and scoring after both "
                "point-flow ledgers were sealed"
            ),
            "geometry_quality": geometry_quality,
        },
        "limitations": [
            "One consumed 143-frame Development canary with three events and nine repeated induced-dropout trials.",
            "Reciprocal voxel tokens are point-local evidence, not semantic object identity or learned scene flow.",
            "No fresh, product, user-benefit, reliability, or safety claim is authorized.",
        ],
    }
    write_json(output_path, result)
    return result


def main() -> int:
    evidence = REPO / "artifacts.local" / "evidence"
    root = evidence / "dtr-c24" / "cycle-consistent-point-flow"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r7-result",
        type=Path,
        default=evidence / "dtr-r7" / "occupancy-flow-canary" / "result.json",
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
    parser.add_argument("--output", type=Path, default=root / "result.json")
    parser.add_argument(
        "--backend-receipt", type=Path, default=root / "backend-point-match.json"
    )
    parser.add_argument("--reuse-ledgers", action="store_true")
    args = parser.parse_args()
    result = run(args)
    cycle = result["arms"]["m1_pdc_three_frame_cycle_consistency"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "c24": cycle["original"],
                "dropout_recovery": cycle["recovered_window_misses"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
