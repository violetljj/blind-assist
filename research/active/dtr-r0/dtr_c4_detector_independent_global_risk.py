"""Score detector-independent route risk directly from sealed motion ledgers.

Phase A reads only timestamps and truth-blind motion ledgers.  It evaluates
every admitted motion cell against the unchanged DTR route tube, applies the
unchanged risk lifecycle, and hash-seals a global alert timeline.  Phase B then
opens the frozen C1 roster and future native OBB labels solely for scoring.

This deliberately tests a representation boundary rather than another route
threshold: can scene motion itself originate a useful warning without a
current detector/native-box association?
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from dtr_c1_global_obb_cohort_admission import (
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence
from dtr_m1_confident_direct_velocity import load_ledger as load_confident_ledger
from dtr_m1_raw_point_direct_velocity import load_ledger as load_raw_point_ledger
from dtr_r0 import DTRConfig, Signal
from dtr_r1 import RiskEventLifecycle
from dtr_r2 import FROZEN_R2_CONFIG
from dtr_r5_dropout_canary import ACTIVE_SIGNALS
from dtr_r7_occupancy_flow_canary import (
    HORIZON_S,
    ROUTE_HALF_WIDTH_M,
    _entry_s,
    load_flow_ledger,
)


SCHEMA = "blindassist-dtr-c4-detector-independent-global-risk-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c4-sealed-global-motion-prediction-v1"
STATUS = "DTR_C4_DETECTOR_INDEPENDENT_GLOBAL_RISK_MEASURED"
ARMS = ("R7_P_GLOBAL", "M1_CT_GLOBAL", "M1_PD_GLOBAL", "M1_PDC_GLOBAL")


def _manifest_path(ledger_path: Path) -> Path:
    return ledger_path.with_suffix(".json")


def _load_arm_ledger(
    arm: str,
    ledger_path: Path,
    *,
    sequence: str,
    frames: list[int],
) -> Any:
    manifest_path = _manifest_path(ledger_path)
    require(manifest_path.is_file(), f"ledger_manifest_missing:{arm}:{sequence}")
    if arm == "R7_P_GLOBAL":
        return load_flow_ledger(
            ledger_path,
            manifest_path,
            expected_sequence=sequence,
            expected_frames=frames,
        )
    if arm in {"M1_CT_GLOBAL", "M1_PDC_GLOBAL"}:
        return load_confident_ledger(
            ledger_path,
            manifest_path,
            expected_sequence=sequence,
            expected_frames=frames,
        )
    if arm == "M1_PD_GLOBAL":
        return load_raw_point_ledger(
            ledger_path,
            manifest_path,
            expected_sequence=sequence,
            expected_frames=frames,
        )
    raise ValueError(f"unknown_arm:{arm}")


def _ledger_paths(c2_root: Path, c3_root: Path, sequence: str) -> dict[str, Path]:
    return {
        "R7_P_GLOBAL": c2_root / sequence / "r7.occupancy-flow.npz",
        "M1_CT_GLOBAL": c2_root / sequence / "m1-ct.confident-direct-velocity.npz",
        "M1_PD_GLOBAL": c3_root / sequence / "m1-pd.raw-point-direct-velocity.npz",
        "M1_PDC_GLOBAL": c3_root / sequence / "m1-pdc.confident-direct-velocity.npz",
    }


def _sequence_names(c2_root: Path, c3_root: Path) -> list[str]:
    c2 = {path.name for path in c2_root.iterdir() if path.is_dir()}
    c3 = {path.name for path in c3_root.iterdir() if path.is_dir()}
    shared = sorted(c2 & c3)
    require(bool(shared), "no_shared_sealed_ledger_sequences")
    return shared


def _predict_arm(
    *,
    ledger: Any,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    origin_s = float(timestamps[int(frames[0])])
    guard_boundary_s = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    raw_alert_frames: list[int] = []
    active_alert_frames: list[int] = []
    urgent_frames: list[int] = []
    minimum_entry_s_by_frame: dict[str, float] = {}
    risky_cells_by_frame: dict[str, int] = {}
    total_cells = 0
    risky_cells = 0

    for frame in frames:
        forward, left, velocity_forward, velocity_left, _component = ledger.frame_cells(int(frame))
        total_cells += len(forward)
        entries = []
        for values in zip(forward, left, velocity_forward, velocity_left):
            entry_s = _entry_s(*(float(value) for value in values))
            if entry_s is not None:
                entries.append(float(entry_s))
        minimum_entry_s = min(entries) if entries else None
        raw_alert = minimum_entry_s is not None
        urgent = bool(raw_alert and minimum_entry_s <= guard_boundary_s + 1e-9)
        signal = lifecycle.update(
            float(timestamps[int(frame)]) - origin_s,
            raw_alert,
            urgent=urgent,
        )
        if raw_alert:
            raw_alert_frames.append(int(frame))
            minimum_entry_s_by_frame[str(int(frame))] = float(minimum_entry_s)
            risky_cells_by_frame[str(int(frame))] = len(entries)
            risky_cells += len(entries)
        if urgent:
            urgent_frames.append(int(frame))
        if signal in ACTIVE_SIGNALS:
            active_alert_frames.append(int(frame))

    return {
        "raw_alert_frames": raw_alert_frames,
        "active_alert_frames": active_alert_frames,
        "urgent_frames": urgent_frames,
        "minimum_entry_s_by_frame": minimum_entry_s_by_frame,
        "risky_cells_by_frame": risky_cells_by_frame,
        "diagnostics": {
            "frames": len(frames),
            "frames_with_motion_cells": sum(
                len(ledger.frame_cells(int(frame))[0]) > 0 for frame in frames
            ),
            "frames_with_route_entry": len(raw_alert_frames),
            "active_alert_frames": len(active_alert_frames),
            "total_motion_cells": total_cells,
            "route_entry_cells": risky_cells,
        },
    }


def seal_predictions(args: argparse.Namespace) -> dict[str, Any]:
    """Truth-blind phase: no roster, native boxes, or prior score is opened."""
    timestamps_path = args.timestamps.resolve(strict=True)
    c2_root = args.c2_ledger_root.resolve(strict=True)
    c3_root = args.c3_ledger_root.resolve(strict=True)
    sequence_names = _sequence_names(c2_root, c3_root)
    sequences = []
    with zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in sequence_names:
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            paths = _ledger_paths(c2_root, c3_root, sequence)
            arm_predictions = {}
            sources = {}
            for arm in ARMS:
                ledger_path = paths[arm].resolve(strict=True)
                manifest_path = _manifest_path(ledger_path).resolve(strict=True)
                ledger = _load_arm_ledger(
                    arm,
                    ledger_path,
                    sequence=sequence,
                    frames=frames,
                )
                arm_predictions[arm] = _predict_arm(
                    ledger=ledger,
                    frames=frames,
                    timestamps=timestamps,
                )
                sources[arm] = {
                    "ledger": str(ledger_path),
                    "ledger_sha256": sha256_file(ledger_path),
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                }
            sequences.append(
                {
                    "sequence": sequence,
                    "first_frame": frames[0],
                    "last_frame": frames[-1],
                    "frames": len(frames),
                    "arms": arm_predictions,
                    "sources": sources,
                }
            )
            print(
                json.dumps(
                    {
                        "c4_truth_blind_sequence": sequence,
                        "frames": len(frames),
                        "raw_alert_frames": {
                            arm: len(arm_predictions[arm]["raw_alert_frames"])
                            for arm in ARMS
                        },
                    }
                ),
                flush=True,
            )
    prediction = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "prediction_boundary": (
            "timestamps plus sealed truth-blind scene-motion ledgers only; no roster, "
            "native boxes, detector cases, future labels, or prior scores"
        ),
        "frozen": {
            "arms": list(ARMS),
            "route_thresholds_lifecycle_and_motion_ledgers": "UNCHANGED",
            "training_or_tuning": False,
        },
        "source": {
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "c2_ledger_root": str(c2_root),
            "c3_ledger_root": str(c3_root),
        },
        "sequences": sequences,
    }
    prediction_path = args.predictions.resolve()
    write_json(prediction_path, prediction)
    return prediction


def _prediction_frames(
    frames: Sequence[int],
    arm_prediction: Mapping[str, Any],
) -> dict[int, dict[str, set[str]]]:
    raw = {int(frame) for frame in arm_prediction["raw_alert_frames"]}
    active = {int(frame) for frame in arm_prediction["active_alert_frames"]}
    return {
        int(frame): {
            "raw": {"global-scene-motion"} if int(frame) in raw else set(),
            "active": {"global-scene-motion"} if int(frame) in active else set(),
        }
        for frame in frames
    }


def _write_scorecard(path: Path, aggregate: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "arm",
        "bounded_contact_events_recalled",
        "bounded_contact_events",
        "bounded_contact_event_recall",
        "bounded_contact_event_precision",
        "bounded_contact_event_f1",
        "false_alert_segments",
        "known_non_contact_wearer_minutes",
        "false_alert_segments_per_known_non_contact_minute",
        "median_first_alert_lead_s",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            writer.writerow(
                {"arm": arm, **{field: aggregate[arm][field] for field in fields[1:]}}
            )
    os.replace(partial, path)


def score_sealed_predictions(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluator phase: open roster and future native OBB truth after sealing."""
    prediction_path = args.predictions.resolve(strict=True)
    prediction_sha256 = sha256_file(prediction_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    require(prediction.get("schema") == PREDICTION_SCHEMA, "prediction_schema_drift")
    require(prediction.get("truth_blind") is True, "prediction_not_truth_blind")

    roster_path = args.roster.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema_drift")
    require(
        roster["source_authority"]["labels_sha256"] == sha256_file(labels_path),
        "labels_hash_drift",
    )
    require(
        roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path),
        "timestamps_hash_drift",
    )
    by_sequence = {str(row["sequence"]): row for row in prediction["sequences"]}
    expected = [str(row["sequence"]) for row in roster["selected_sequences"]]
    require(set(by_sequence) == set(expected), "prediction_roster_sequence_drift")

    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    per_sequence = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in expected:
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            require(set(boxes) == set(frames), f"label_timestamp_frame_mismatch:{sequence}")
            timeline = global_truth_timeline(
                frames=frames,
                timestamps=timestamps,
                boxes_by_frame=boxes,
            )
            prediction_row = by_sequence[sequence]
            scores = {}
            for arm in ARMS:
                score = score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=_prediction_frames(frames, prediction_row["arms"][arm]),
                )
                scores[arm] = score
                per_arm[arm].append(score)
            per_sequence.append(
                {
                    "sequence": sequence,
                    "scores": scores,
                    "prediction_diagnostics": {
                        arm: prediction_row["arms"][arm]["diagnostics"] for arm in ARMS
                    },
                }
            )

    aggregate = {arm: aggregate_scores(per_arm[arm]) for arm in ARMS}
    output_path = args.output.resolve()
    scorecard_path = output_path.with_name(output_path.stem + ".scorecard.csv")
    _write_scorecard(scorecard_path, aggregate)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": (
            "Can confidence-aware scene motion originate stable future path-conflict alerts "
            "without any current detector/native-box attribution?"
        ),
        "aggregate": aggregate,
        "per_sequence": per_sequence,
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": prediction_sha256,
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "artifacts": {
            "scorecard_csv": str(scorecard_path),
            "scorecard_csv_sha256": sha256_file(scorecard_path),
        },
        "algorithm_increment": (
            "Every confidence-admitted point-wise scene-motion cell is queried directly "
            "against the unchanged wearer route tube; the global lifecycle runs before "
            "any detector track or native OBB is available."
        ),
        "claim_limits": [
            "This is a detector-independent global-risk measurement on seven curated public JRDB sequences, not product or safety evidence.",
            "The motion ledgers estimate present direct velocity and constant-velocity route entry; they do not forecast multimodal trajectories.",
            "Future native OBBs are opened only after the prediction artifact is hash sealed and are evaluator-only.",
            "Reciprocal nearest-point flow can under-estimate surface-parallel motion; R7 and M1-PD remain diagnostic arms, not universal motion estimators.",
        ],
    }
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c2_evidence = repo / "artifacts.local" / "evidence" / "dtr-c2" / "fresh-global-obb-replay"
    c3_evidence = repo / "artifacts.local" / "evidence" / "dtr-c3" / "raw-point-direct-velocity-canary"
    c4_evidence = repo / "artifacts.local" / "evidence" / "dtr-c4" / "detector-independent-global-risk"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json"),
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--c2-ledger-root", type=Path, default=c2_evidence / "ledgers")
    parser.add_argument("--c3-ledger-root", type=Path, default=c3_evidence / "ledgers")
    parser.add_argument("--predictions", type=Path, default=c4_evidence / "predictions.json")
    parser.add_argument("--output", type=Path, default=c4_evidence / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seal_predictions(args)
    result = score_sealed_predictions(args)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
