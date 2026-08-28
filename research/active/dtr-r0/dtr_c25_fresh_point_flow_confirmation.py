"""Freeze and run detector-independent C24 point flow on fresh JRDB sequences.

C25 keeps the C24 representation fixed: reciprocal raw-point direct velocity
followed by independent three-frame position/velocity consistency.  The worker
phase sees only raw bags, timestamps, calibration, and frozen code.  Global
route-conflict predictions are sealed before the score phase opens the roster
or future native OBB truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import zipfile

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_m1_confident_direct_velocity as confident_flow
import dtr_m1_raw_point_direct_velocity as point_flow
import dtr_r7_occupancy_flow_canary as r7
from dtr_c1_global_obb_cohort_admission import (
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_acquire_frozen_bags import SCHEMA as ACQUISITION_SCHEMA
from dtr_c2_fresh_global_obb_replay import (
    _causal_pose,
    _tracks,
    aggregate_scores,
    dropout_stress,
    score_sequence,
)
from dtr_r5_dropout_canary import cases_from_tracks
from dtr_c4_detector_independent_global_risk import (
    _predict_arm,
    _prediction_frames,
)
from dtr_c10_fresh_c9_confirmation import (
    _acquired_bag,
    _select_point_backend,
)
from dtr_c10_fresh_confirmation_admission import _meets, _select, _totals
from jrdb_rgb_bridge import read_bag_pose_and_rgb


SCHEMA = "blindassist-dtr-c25-fresh-point-flow-confirmation-v1"
WORKER_SCHEMA = "blindassist-dtr-c25-sealed-sequence-point-flow-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c25-sealed-point-flow-predictions-v1"
ADMISSION_SCHEMA = "blindassist-dtr-c25-fresh-point-flow-admission-v1"
STATUS_MET = "DTR_C25_POINT_FLOW_ALGORITHM_FRESH_GATE_MET"
STATUS_NOT_MET = "DTR_C25_POINT_FLOW_ALGORITHM_FRESH_GATE_NOT_MET"
ROSTER_STATUS = "DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY"
ARMS = ("R7_P_GLOBAL", "M1_PD_GLOBAL", "M1_PDC_GLOBAL")


def admit(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.c1_result.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    require(
        source.get("schema")
        == "blindassist-dtr-c1-global-obb-cohort-admission-v1",
        "c1_result_schema_drift",
    )
    consumed = set(str(value) for value in source["selected_sequence_names"])
    consumed.update(str(value) for value in source["source"]["excluded_consumed_sequences"])
    consumed_rosters = []
    for path_value in (args.c10_roster, args.c11_roster):
        path = path_value.resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        require(value.get("schema") == ROSTER_SCHEMA, f"roster_schema:{path}")
        consumed.update(str(row["sequence"]) for row in value["selected_sequences"])
        consumed_rosters.append(
            {"path": str(path), "sha256": sha256_file(path)}
        )
    candidates = [
        row for row in source["sequence_scan"] if str(row["sequence"]) not in consumed
    ]
    preferred = source["admission_policy"]["preferred"]
    minimum = source["admission_policy"]["minimum"]
    remaining_totals = _totals(candidates)
    require(not _meets(remaining_totals, preferred), "preferred_gate_still_reachable")
    selected, considered = _select(candidates, minimum)
    totals = _totals(selected)
    require(_meets(totals, minimum), "minimum_gate_not_met")
    frozen_files = [
        Path(__file__).resolve(),
        point_flow.__file__,
        confident_flow.__file__,
        Path(r7.__file__).resolve(),
        REPO
        / "research"
        / "active"
        / "dtr-r0"
        / "dtr_c4_detector_independent_global_risk.py",
    ]
    roster = {
        "schema": ROSTER_SCHEMA,
        "status": ROSTER_STATUS,
        "claim_ceiling": "ALGORITHM_FRESH_PUBLIC_REPLAY_NO_PRODUCT_OR_SAFETY_CLAIM",
        "dataset": "JRDB public train split",
        "contract": source["contract"],
        "selection_policy": {
            "algorithm_consumed_sequences_excluded": sorted(consumed),
            "preferred_gate_unreachable_on_remaining_pool": True,
            "gate": minimum,
            "ordering": (
                "minimum sequence cardinality meeting the existing minimum gate; "
                "then minimum total frames; then lexicographic sequence tuple"
            ),
            "candidate_sequences": len(candidates),
            "combinations_considered": considered,
        },
        "selected_sequences": [
            {
                "sequence": str(row["sequence"]),
                "first_frame": int(row["first_frame"]),
                "last_frame": int(row["last_frame"]),
                "frames": int(row["frames"]),
                "timeline_duration_s": float(row["timeline_duration_s"]),
                "known_non_contact_s": float(row["known_non_contact_s"]),
                "bounded_contact_events": int(row["bounded_contact_events"]),
                "unique_responsible_events": int(row["unique_responsible_events"]),
                "bounded_contact_event_details": list(row["events"]),
            }
            for row in selected
        ],
        "selected_totals": totals,
        "frozen_algorithm": {
            "arms": list(ARMS),
            "representation": (
                "R7 component translation versus reciprocal raw-point direct "
                "velocity versus three-frame point-local consistency"
            ),
            "route_geometry_motion_bounds_and_lifecycle": "UNCHANGED",
            "gate": {
                "m1_pdc_contact_recall_not_lower_than_r7": True,
                "m1_pdc_false_segment_factor_vs_r7": 0.70,
                "m1_pdc_event_f1_higher_than_r7": True,
                "m1_pdc_dropout_recovery_not_lower_than_r7": True,
            },
            "files": [
                {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
                for path in frozen_files
            ],
        },
        "source_authority": {
            "labels_archive_name": Path(source["source"]["labels"]).name,
            "labels_sha256": source["source"]["labels_sha256"],
            "timestamps_archive_name": Path(source["source"]["timestamps"]).name,
            "timestamps_sha256": source["source"]["timestamps_sha256"],
            "c1_result": str(source_path),
            "c1_result_sha256": sha256_file(source_path),
            "consumed_rosters": consumed_rosters,
        },
        "forbidden": [
            "changing point matching, confidence, route, motion bounds, lifecycle, or gate after acquisition",
            "opening selected future OBB labels before all global predictions are sealed",
            "adding an algorithm-exposed sequence",
            "rerunning a failed arm with different parameters",
        ],
    }
    roster_path = args.roster.resolve()
    write_json(roster_path, roster)
    result = {
        "schema": ADMISSION_SCHEMA,
        "status": "DTR_C25_FRESH_POINT_FLOW_COHORT_ADMITTED_METADATA_ONLY",
        "selected_sequences": [str(row["sequence"]) for row in selected],
        "selected_totals": totals,
        "remaining_pool_totals": remaining_totals,
        "artifacts": {
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
        },
    }
    write_json(args.output.resolve(), result)
    return result


def merge_acquisition(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    archive = None
    roster_sha256 = None
    for path_value in args.acquisition_workers:
        path = path_value.resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        require(value.get("schema") == ACQUISITION_SCHEMA, f"acquisition_schema:{path}")
        require(len(value.get("bags", [])) == 1, f"acquisition_cardinality:{path}")
        if archive is None:
            archive = value["archive"]
            roster_sha256 = value["roster_sha256"]
        require(value["archive"] == archive, f"acquisition_archive_drift:{path}")
        require(value["roster_sha256"] == roster_sha256, f"acquisition_roster_drift:{path}")
        row = value["bags"][0]
        bag = Path(row["bag"]).resolve(strict=True)
        require(bag.stat().st_size == int(row["bytes"]), f"bag_size:{bag}")
        require(sha256_file(bag) == row["sha256"], f"bag_hash:{bag}")
        rows.append(row)
    names = [str(row["sequence"]) for row in rows]
    require(len(names) == len(set(names)) and bool(names), "acquisition_duplicate")
    result = {
        "schema": ACQUISITION_SCHEMA,
        "status": "DTR_C2_FROZEN_RAW_BAGS_ACQUIRED",
        "roster": str(args.roster.resolve(strict=True)),
        "roster_sha256": roster_sha256,
        "archive": archive,
        "recovery": {
            "completed_members_are_hash_checked_and_skipped": True,
            "partial_member_resume": False,
            "maximum_lost_work": "one compressed rosbag member",
        },
        "bags": sorted(rows, key=lambda row: str(row["sequence"])),
        "totals": {"sequences": len(rows), "bytes": sum(int(row["bytes"]) for row in rows)},
    }
    write_json(args.output.resolve(), result)
    return result


def worker(args: argparse.Namespace) -> dict[str, Any]:
    sequence = str(args.sequence)
    timestamps_path = args.timestamps.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    bag_path, bag_row = _acquired_bag(args.acquisition.resolve(strict=True), sequence)
    with zipfile.ZipFile(timestamps_path) as archive:
        timestamps = _load_timestamps(archive, sequence)
    frames = sorted(timestamps)
    sequence_root = args.ledger_root.resolve() / sequence

    r7_selection = _select_point_backend(args.r7_backend_receipt.resolve())
    require(
        r7_selection["selected_backend"] == "python-irregular-component-matching",
        "r7_selected_backend_mismatch",
    )
    r7_npz, r7_manifest = r7.ledger_paths(sequence_root / "r7.json")
    if not (r7_npz.exists() and r7_manifest.exists()):
        r7.materialize_flow_ledger(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            calibration_dir=calibration_dir,
            output_path=r7_npz,
            manifest_path=r7_manifest,
            sequence=sequence,
            timestamps_override=timestamps,
        )
    r7_ledger = r7.load_flow_ledger(
        r7_npz, r7_manifest, expected_sequence=sequence, expected_frames=frames
    )

    point_npz, point_manifest = point_flow.ledger_paths(sequence_root / "m1-pd.json")
    if not (point_npz.exists() and point_manifest.exists()):
        point_flow.materialize(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            calibration_dir=calibration_dir,
            output_path=point_npz,
            manifest_path=point_manifest,
            backend_receipt_path=args.point_backend_receipt.resolve(),
            sequence=sequence,
            timestamps_override=timestamps,
        )
    point_ledger = point_flow.load_ledger(
        point_npz, point_manifest, expected_sequence=sequence, expected_frames=frames
    )

    confident_npz, confident_manifest = confident_flow.ledger_paths(
        sequence_root / "m1-pdc.json"
    )
    if not (confident_npz.exists() and confident_manifest.exists()):
        confident_flow.materialize(
            source_path=point_npz,
            source_manifest_path=point_manifest,
            output_path=confident_npz,
            manifest_path=confident_manifest,
        )
    confident_ledger = confident_flow.load_ledger(
        confident_npz,
        confident_manifest,
        expected_sequence=sequence,
        expected_frames=frames,
    )
    ledgers = {
        "R7_P_GLOBAL": (r7_ledger, r7_npz, r7_manifest),
        "M1_PD_GLOBAL": (point_ledger, point_npz, point_manifest),
        "M1_PDC_GLOBAL": (confident_ledger, confident_npz, confident_manifest),
    }
    result = {
        "schema": WORKER_SCHEMA,
        "truth_blind": True,
        "sequence": sequence,
        "frames": len(frames),
        "arms": {
            arm: _predict_arm(ledger=value[0], frames=frames, timestamps=timestamps)
            for arm, value in ledgers.items()
        },
        "sources": {
            "bag": str(bag_path),
            "bag_sha256": bag_row["sha256"],
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "calibration": str(calibration_dir),
            "calibration_sha256": sha256_file(calibration_dir / "lidars.yaml"),
            "ledgers": {
                arm: {
                    "ledger": str(value[1]),
                    "ledger_sha256": sha256_file(value[1]),
                    "manifest": str(value[2]),
                    "manifest_sha256": sha256_file(value[2]),
                }
                for arm, value in ledgers.items()
            },
        },
        "backends": {
            "r7": r7_selection,
            "point_receipt": str(args.point_backend_receipt.resolve()),
            "point_receipt_sha256": sha256_file(args.point_backend_receipt.resolve()),
        },
        "prediction_boundary": (
            "raw bag, timestamps, calibration, and frozen point-flow code only; "
            "no roster, labels, future OBB truth, or prior fresh result"
        ),
    }
    write_json(args.output.resolve(), result)
    return result


def merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for path_value in args.worker_predictions:
        path = path_value.resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        require(value.get("schema") == WORKER_SCHEMA, f"worker_schema:{path}")
        require(value.get("truth_blind") is True, f"worker_truth:{path}")
        for source in value["sources"]["ledgers"].values():
            require(
                sha256_file(Path(source["ledger"]).resolve(strict=True))
                == source["ledger_sha256"],
                f"worker_ledger_hash:{path}",
            )
        rows.append(value)
    names = [str(row["sequence"]) for row in rows]
    require(len(names) == len(set(names)) and bool(names), "worker_duplicate")
    prediction = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "prediction_boundary": (
            "all detector-independent route-conflict timelines sealed before "
            "the C25 roster or future OBB truth is opened"
        ),
        "sequences": sorted(rows, key=lambda row: str(row["sequence"])),
    }
    write_json(args.predictions.resolve(), prediction)
    return prediction


def _aggregate_stress(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    misses = sum(int(row["track_only_window_misses"]) for row in rows)
    recovered = {
        "r7": sum(int(row["r7_recovered_track_only_window_misses"]) for row in rows),
        "m1_pd": sum(int(row["m1_recovered_track_only_window_misses"]) for row in rows),
        "m1_pdc": sum(int(row["m1_ct_recovered_track_only_window_misses"]) for row in rows),
    }
    return {
        "trials": sum(int(row["trials"]) for row in rows),
        "track_only_window_misses": misses,
        **{f"{name}_recovered_track_only_window_misses": value for name, value in recovered.items()},
        **{
            f"{name}_recovery_rate": (value / misses if misses else None)
            for name, value in recovered.items()
        },
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.predictions.resolve(strict=True)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    require(prediction.get("schema") == PREDICTION_SCHEMA, "prediction_schema")
    require(prediction.get("truth_blind") is True, "prediction_truth")
    roster_path = args.roster.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema")
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    require(
        roster["source_authority"]["labels_sha256"] == sha256_file(labels_path),
        "labels_hash",
    )
    require(
        roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path),
        "timestamps_hash",
    )
    by_sequence = {str(row["sequence"]): row for row in prediction["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    require(set(by_sequence) == set(roster_rows), "prediction_roster_coverage")
    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    stress_rows = []
    per_sequence = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence, roster_row in roster_rows.items():
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(
                frames=frames, timestamps=timestamps, boxes_by_frame=boxes
            )
            prediction_row = by_sequence[sequence]
            scores = {}
            for arm in ARMS:
                arm_score = score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=_prediction_frames(
                        frames, prediction_row["arms"][arm]
                    ),
                )
                scores[arm] = arm_score
                per_arm[arm].append(arm_score)

            poses, _rgb, bag_authority = read_bag_pose_and_rgb(
                Path(prediction_row["sources"]["bag"]).resolve(strict=True)
            )
            frame_poses = {
                frame: _causal_pose(poses, round(timestamps[frame] * 1e9))
                for frame in frames
            }
            cases = cases_from_tracks(
                _tracks(
                    boxes_by_frame=boxes,
                    timestamps=timestamps,
                    frame_poses=frame_poses,
                )
            )
            sources = prediction_row["sources"]["ledgers"]
            r7_ledger = r7.load_flow_ledger(
                Path(sources["R7_P_GLOBAL"]["ledger"]),
                Path(sources["R7_P_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            point_ledger = point_flow.load_ledger(
                Path(sources["M1_PD_GLOBAL"]["ledger"]),
                Path(sources["M1_PD_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            pdc_ledger = confident_flow.load_ledger(
                Path(sources["M1_PDC_GLOBAL"]["ledger"]),
                Path(sources["M1_PDC_GLOBAL"]["manifest"]),
                expected_sequence=sequence,
                expected_frames=frames,
            )
            stress = dropout_stress(
                roster_sequence=roster_row,
                cases={(case.label_id, case.segment_index): case for case in cases},
                r7=r7_ledger,
                m1=point_ledger,
                m1_ct=pdc_ledger,
            )
            stress_rows.append(stress)
            per_sequence.append(
                {
                    "sequence": sequence,
                    "scores": scores,
                    "dropout_stress": stress,
                    "bag_authority": bag_authority,
                }
            )
    aggregate = {arm: aggregate_scores(rows) for arm, rows in per_arm.items()}
    stress = _aggregate_stress(stress_rows)
    baseline = aggregate["R7_P_GLOBAL"]
    candidate = aggregate["M1_PDC_GLOBAL"]
    gate_config = roster["frozen_algorithm"]["gate"]
    checks = {
        "contact_recall_not_lower_than_r7": candidate[
            "bounded_contact_events_recalled"
        ]
        >= baseline["bounded_contact_events_recalled"],
        "false_segments_reduced_at_least_thirty_percent": candidate[
            "false_alert_segments"
        ]
        <= gate_config["m1_pdc_false_segment_factor_vs_r7"]
        * baseline["false_alert_segments"],
        "event_f1_higher_than_r7": candidate["bounded_contact_event_f1"]
        > baseline["bounded_contact_event_f1"],
        "dropout_recovery_not_lower_than_r7": stress[
            "m1_pdc_recovered_track_only_window_misses"
        ]
        >= stress["r7_recovered_track_only_window_misses"],
    }
    passed = all(checks.values())
    result = {
        "schema": SCHEMA,
        "status": STATUS_MET if passed else STATUS_NOT_MET,
        "question": (
            "Does detector-independent point-local three-frame motion retain "
            "fresh future-path conflict recall while suppressing R7 pseudo-motion?"
        ),
        "aggregate": aggregate,
        "dropout_stress": stress,
        "per_sequence": per_sequence,
        "gate": {"passed": passed, "checks": checks, "config": gate_config},
        "tradeoffs": {
            "median_first_alert_lead_change_s": candidate[
                "median_first_alert_lead_s"
            ]
            - baseline["median_first_alert_lead_s"],
            "lead_is_reported_not_gated": True,
        },
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": sha256_file(prediction_path),
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "claim_limits": [
            "Five algorithm-fresh public JRDB sequences with twelve bounded CONTACT events.",
            "Global predictions are detector-independent, but induced dropout scoring uses evaluator-side target identity only after prediction sealing.",
            "Constant point velocity is not multimodal future occupancy forecasting.",
            "No product, deployment, user-benefit, reliability, or safety claim follows.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    evidence = REPO / "artifacts.local" / "evidence" / "dtr-c25" / "fresh-point-flow-confirmation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("admit", "merge-acquisition", "worker", "merge", "score")
    )
    parser.add_argument("--sequence")
    parser.add_argument("--c1-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-c1" / "global-obb-cohort-admission" / "result.json")
    parser.add_argument("--c10-roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c10_fresh_confirmation_roster.json"))
    parser.add_argument("--c11-roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c11_fresh_confirmation_roster.json"))
    parser.add_argument("--roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c25_fresh_confirmation_roster.json"))
    parser.add_argument("--acquisition-workers", type=Path, nargs="*")
    parser.add_argument("--worker-predictions", type=Path, nargs="*")
    parser.add_argument("--acquisition", type=Path, default=evidence / "acquisition.json")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--calibration-dir", type=Path, default=REPO / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    parser.add_argument("--ledger-root", type=Path, default=evidence / "ledgers")
    parser.add_argument("--r7-backend-receipt", type=Path)
    parser.add_argument("--point-backend-receipt", type=Path)
    parser.add_argument("--predictions", type=Path, default=evidence / "predictions.json")
    parser.add_argument("--output", type=Path, default=evidence / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "admit":
        result = admit(args)
        print(json.dumps({"status": result["status"], "selected": result["selected_sequences"], "totals": result["selected_totals"]}))
    elif args.mode == "merge-acquisition":
        require(bool(args.acquisition_workers), "acquisition_workers_required")
        result = merge_acquisition(args)
        print(json.dumps({"status": result["status"], "totals": result["totals"]}))
    elif args.mode == "worker":
        require(bool(args.sequence), "worker_sequence_required")
        require(args.r7_backend_receipt is not None, "r7_backend_receipt_required")
        require(args.point_backend_receipt is not None, "point_backend_receipt_required")
        result = worker(args)
        print(json.dumps({"status": "C25_SEQUENCE_PREDICTIONS_SEALED", "sequence": result["sequence"]}))
    elif args.mode == "merge":
        require(bool(args.worker_predictions), "worker_predictions_required")
        result = merge(args)
        print(json.dumps({"status": "C25_PREDICTIONS_SEALED", "sequences": len(result["sequences"])}))
    else:
        result = score(args)
        print(json.dumps({"status": result["status"], "gate": result["gate"], "aggregate": result["aggregate"], "dropout": result["dropout_stress"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
