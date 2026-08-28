"""Run one fresh-sequence M1-PD raw-point direct-velocity canary."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

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
    _prediction_frames,
    _tracks,
    aggregate_scores,
    dropout_stress,
    score_sequence,
)
from dtr_m1_confident_direct_velocity import (
    ledger_paths as confident_ledger_paths,
    load_ledger as load_confident_ledger,
    materialize as materialize_confident_ledger,
)
from dtr_m1_raw_point_direct_velocity import (
    ledger_paths as point_ledger_paths,
    load_ledger as load_point_ledger,
    materialize as materialize_point_ledger,
)
from dtr_r5_dropout_canary import cases_from_tracks
from dtr_r7_occupancy_flow_canary import run_flow_arm
from jrdb_rgb_bridge import read_bag_pose_and_rgb


SCHEMA = "blindassist-dtr-c3-raw-point-direct-velocity-canary-v1"
STATUS = "DTR_C3_RAW_POINT_DIRECT_VELOCITY_CANARY_COMPLETE"
FULL_STATUS = "DTR_C3_RAW_POINT_DIRECT_VELOCITY_FRESH_REPLAY_COMPLETE"
ARMS = ("R7_P", "M1_CTB", "M1_PD", "M1_PDC", "M1_PDCB", "M1_HYBRID")


def _score(
    *,
    sequence: str,
    frames: list[int],
    timeline: list[dict[str, Any]],
    cases: dict[Any, Any],
    ledger: Any,
) -> dict[str, Any]:
    predictions = {
        key: run_flow_arm(case, set(), ledger).predictions for key, case in cases.items()
    }
    return score_sequence(
        sequence=sequence,
        timeline=timeline,
        prediction_frames=_prediction_frames(frames, predictions, cases),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve(strict=True)
    acquisition_path = args.acquisition.resolve(strict=True)
    c2_path = args.c2_result.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    c2 = json.loads(c2_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "c3_roster_schema")
    require(acquisition.get("schema") == ACQUISITION_SCHEMA, "c3_acquisition_schema")
    roster_rows = {
        str(row["sequence"]): row for row in roster["selected_sequences"]
    }
    bag_rows = {str(row["sequence"]): row for row in acquisition["bags"]}
    require(args.sequence in roster_rows, "c3_sequence_not_frozen")
    require(args.sequence in bag_rows, "c3_bag_missing")
    c2_rows = {str(row["sequence"]): row for row in c2["per_sequence"]}
    require(args.sequence in c2_rows, "c3_c2_sequence_missing")
    bag_path = Path(bag_rows[args.sequence]["bag"]).resolve(strict=True)
    require(sha256_file(bag_path) == bag_rows[args.sequence]["sha256"], "c3_bag_hash")

    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        timestamps = _load_timestamps(timestamps_zip, args.sequence)
        boxes = _load_boxes(labels, args.sequence)
    frames = sorted(timestamps)
    poses, _rgb, _authority = read_bag_pose_and_rgb(bag_path)
    frame_poses = {
        frame: _causal_pose(poses, round(timestamps[frame] * 1e9)) for frame in frames
    }
    cases_list = cases_from_tracks(
        _tracks(boxes_by_frame=boxes, timestamps=timestamps, frame_poses=frame_poses)
    )
    cases = {(case.label_id, case.segment_index): case for case in cases_list}
    timeline = global_truth_timeline(
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=boxes,
    )

    output_path = args.output.resolve()
    ledger_dir = output_path.parent / "ledgers" / args.sequence
    point_npz, point_manifest = point_ledger_paths(ledger_dir / "m1-pd.json")
    backend_receipt = ledger_dir / "m1-pd.backend.json"
    if not point_npz.exists() or not point_manifest.exists():
        materialize_point_ledger(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            calibration_dir=calibration_dir,
            output_path=point_npz,
            manifest_path=point_manifest,
            backend_receipt_path=backend_receipt,
            sequence=args.sequence,
            timestamps_override=timestamps,
        )
    point = load_point_ledger(
        point_npz,
        point_manifest,
        expected_sequence=args.sequence,
        expected_frames=frames,
    )
    confident_npz, confident_manifest = confident_ledger_paths(ledger_dir / "m1-pdc.json")
    if not confident_npz.exists() or not confident_manifest.exists():
        materialize_confident_ledger(
            source_path=point_npz,
            source_manifest_path=point_manifest,
            output_path=confident_npz,
            manifest_path=confident_manifest,
        )
    confident = load_confident_ledger(
        confident_npz,
        confident_manifest,
        expected_sequence=args.sequence,
        expected_frames=frames,
    )

    raw_score = _score(
        sequence=args.sequence,
        frames=frames,
        timeline=timeline,
        cases=cases,
        ledger=point,
    )
    confident_score = _score(
        sequence=args.sequence,
        frames=frames,
        timeline=timeline,
        cases=cases,
        ledger=confident,
    )
    point_stress = dropout_stress(
        roster_sequence=roster_rows[args.sequence],
        cases=cases,
        r7=point,
        m1=point,
        m1_ct=confident,
    )
    c2_row = c2_rows[args.sequence]
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "sequence": args.sequence,
        "question": "Does reciprocal raw-point direct velocity improve the already supported M1 confidence/track-gap interface without route tuning?",
        "scores": {
            "R7_P": c2_row["scores"]["R7_P"],
            "M1_CTB": c2_row["scores"]["M1_CTB"],
            "M1_PD": raw_score,
            "M1_PDC": confident_score,
            "M1_PDCB": dict(confident_score),
            "M1_HYBRID": dict(c2_row["scores"]["M1_CTB"]),
        },
        "dropout_stress": {
            "trials": point_stress["trials"],
            "track_only_window_misses": point_stress["track_only_window_misses"],
            "m1_pd_recovered_track_only_window_misses": point_stress[
                "r7_recovered_track_only_window_misses"
            ],
            "m1_pdc_recovered_track_only_window_misses": point_stress[
                "m1_ct_recovered_track_only_window_misses"
            ],
            "m1_pdcb_recovered_track_only_window_misses": point_stress[
                "r7_recovered_track_only_window_misses"
            ],
            "m1_hybrid_recovered_track_only_window_misses": point_stress[
                "r7_recovered_track_only_window_misses"
            ],
            "bridge_source": "M1-PD reciprocal raw-point motion only inside observable track gap",
        },
        "source": {
            "c2_result": str(c2_path),
            "c2_result_sha256": sha256_file(c2_path),
            "raw_point_ledger": str(point_npz),
            "raw_point_ledger_sha256": sha256_file(point_npz),
            "raw_point_manifest": str(point_manifest),
            "raw_point_manifest_sha256": sha256_file(point_manifest),
            "confident_ledger": str(confident_npz),
            "confident_ledger_sha256": sha256_file(confident_npz),
            "backend_receipt": str(backend_receipt),
            "backend_receipt_sha256": sha256_file(backend_receipt),
        },
        "claim_limits": [
            "One frozen fresh sequence is a canary, not a seven-sequence confirmation.",
            "Reciprocal nearest voxel motion is a direct-velocity hypothesis, not learned scene flow or an object trajectory forecast.",
            "Current native boxes remain privileged scorer-side spatial attribution.",
        ],
    }
    write_json(output_path, result)
    return result


def merge_worker_results(args: argparse.Namespace) -> dict[str, Any]:
    roster = json.loads(args.roster.resolve(strict=True).read_text(encoding="utf-8"))
    expected = [str(row["sequence"]) for row in roster["selected_sequences"]]
    by_sequence: dict[str, dict[str, Any]] = {}
    backend_counts: dict[str, int] = {}
    for path_value in args.merge_worker_results:
        path = path_value.resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("schema") == SCHEMA, f"c3_worker_schema:{path}")
        if payload.get("status") == FULL_STATUS:
            worker_payloads = payload["per_sequence"]
        else:
            require(payload.get("status") == STATUS, f"c3_worker_status:{path}")
            worker_payloads = [payload]
        for worker in worker_payloads:
            sequence = str(worker["sequence"])
            require(sequence not in by_sequence, f"c3_duplicate_sequence:{sequence}")
            worker["scores"]["M1_PDCB"] = dict(worker["scores"]["M1_PDC"])
            worker["scores"]["M1_HYBRID"] = dict(worker["scores"]["M1_CTB"])
            worker["dropout_stress"]["m1_pdcb_recovered_track_only_window_misses"] = int(
                worker["dropout_stress"]["m1_pd_recovered_track_only_window_misses"]
            )
            worker["dropout_stress"]["m1_hybrid_recovered_track_only_window_misses"] = int(
                worker["dropout_stress"]["m1_pd_recovered_track_only_window_misses"]
            )
            worker["dropout_stress"]["bridge_source"] = (
                "M1-PD reciprocal raw-point motion only inside observable track gap"
            )
            point_manifest = json.loads(
                Path(worker["source"]["raw_point_manifest"])
                .resolve(strict=True)
                .read_text(encoding="utf-8")
            )
            backend = str(point_manifest["backend"]["selected_backend"])
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            by_sequence[sequence] = worker
    require(set(by_sequence) == set(expected), "c3_worker_sequence_coverage")
    per_sequence = [by_sequence[sequence] for sequence in expected]
    aggregate = {
        arm: aggregate_scores([row["scores"][arm] for row in per_sequence])
        for arm in ARMS
    }
    trials = sum(int(row["dropout_stress"]["trials"]) for row in per_sequence)
    misses = sum(
        int(row["dropout_stress"]["track_only_window_misses"])
        for row in per_sequence
    )
    recovered = {
        arm: sum(
            int(row["dropout_stress"][f"{arm}_recovered_track_only_window_misses"])
            for row in per_sequence
        )
        for arm in ("m1_pd", "m1_pdc", "m1_pdcb", "m1_hybrid")
    }
    result = {
        "schema": SCHEMA,
        "status": FULL_STATUS,
        "question": "Does reciprocal raw-point direct velocity improve the supported confidence/track-gap interface across the complete frozen C1 cohort?",
        "aggregate": aggregate,
        "dropout_stress": {
            "trials": trials,
            "track_only_window_misses": misses,
            **{
                f"{arm}_recovered_track_only_window_misses": count
                for arm, count in recovered.items()
            },
            **{
                f"{arm}_recovery_rate": None if misses == 0 else count / misses
                for arm, count in recovered.items()
            },
            "m1_hybrid_recovery_is_lower_bound": True,
            "m1_hybrid_gap_sources": ["M1_PD", "R7_P_FALLBACK"],
        },
        "backend_selection_counts": backend_counts,
        "per_sequence": per_sequence,
        "source": {
            "roster": str(args.roster.resolve(strict=True)),
            "roster_sha256": sha256_file(args.roster.resolve(strict=True)),
            "c2_result": str(args.c2_result.resolve(strict=True)),
            "c2_result_sha256": sha256_file(args.c2_result.resolve(strict=True)),
            "worker_results": [str(path.resolve()) for path in args.merge_worker_results],
        },
        "algorithm_increment": (
            "M1-PD replaces component-centroid pseudo-flow with ego-compensated reciprocal raw-point 3-D voxel direct velocity. M1-PDC applies the existing independent-history gate. M1-HYBRID uses the lower-false M1-CT natural path and permits reciprocal raw-point PD plus sealed R7 fallback only inside an observable prior-track gap. The reported 52 recoveries are the confirmed M1-PD lower bound on that union."
        ),
        "claim_limits": [
            "This is frozen public LiDAR replay with privileged current-box spatial attribution, not an RGB/mobile runtime or safety result.",
            "Reciprocal nearest voxel motion is a direct-velocity hypothesis, not learned scene flow or trajectory forecasting.",
            "No route, lifecycle, motion, confidence, voxel, or cohort parameter was selected from C3 outcomes.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    evidence = repo / "artifacts.local" / "evidence"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="clark-center-2019-02-28_0")
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json"),
    )
    parser.add_argument(
        "--acquisition",
        type=Path,
        default=evidence / "dtr-c2" / "fresh-global-obb-replay" / "acquisition.json",
    )
    parser.add_argument(
        "--c2-result",
        type=Path,
        default=evidence / "dtr-c2" / "fresh-global-obb-replay" / "result.json",
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=repo
        / "artifacts.local"
        / "datasets"
        / "ustrf-canonical-observation-source-authority-data-pack-r0"
        / "jrdb_toolkit"
        / "calibration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=evidence / "dtr-c3" / "raw-point-direct-velocity-canary" / "result.json",
    )
    parser.add_argument("--merge-worker-results", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge_worker_results(args) if args.merge_worker_results else run(args)
    score_rows = result.get("scores", result.get("aggregate", {}))
    compact = {
        arm: {
            "recall": score["bounded_contact_event_recall"],
            "false": score["false_alert_segments"],
            "f1": score["bounded_contact_event_f1"],
        }
        for arm, score in score_rows.items()
    }
    print(json.dumps({"status": result["status"], "scores": compact, "dropout": result["dropout_stress"]}))


if __name__ == "__main__":
    main()
