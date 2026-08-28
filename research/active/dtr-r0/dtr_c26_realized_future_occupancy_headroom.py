"""Support-conditioned realized-future-occupancy headroom after C25.

Privileged consumed-cohort counterfactual, not a causal predictor. Only cells
already present in sealed M1-PD may use future native OBBs. Unsupported,
ambiguous, or right-censored cells retain their sealed constant-velocity entry.
The combined raw signal passes through the unchanged R2 lifecycle and scorer.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import zipfile

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coda_static_ceiling import point_to_box_clearance  # noqa: E402
from dtr_c1_global_obb_cohort_admission import (  # noqa: E402
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    _obb_clearance,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence  # noqa: E402
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
import dtr_m1_raw_point_direct_velocity as point_flow  # noqa: E402
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    HORIZON_S,
    ROUTE_HALF_WIDTH_M,
    _entry_s,
)

SCHEMA = "blindassist-dtr-c26-supported-realized-future-occupancy-headroom-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c25-sealed-point-flow-predictions-v1"
C25_SCHEMA = "blindassist-dtr-c25-fresh-point-flow-confirmation-v1"
STATUS_MET = "DTR_C26_SUPPORTED_FUTURE_OCCUPANCY_HEADROOM_MET"
STATUS_NOT_MET = "DTR_C26_SUPPORTED_FUTURE_OCCUPANCY_HEADROOM_NOT_MET"
ARMS = ("R7_P_GLOBAL", "M1_PD_GLOBAL", "M1_PDC_GLOBAL")
BASELINE = "M1_PD_GLOBAL"
EPSILON = 1e-9


def _cell_clearance(forward: float, left: float, box: Any) -> float:
    return point_to_box_clearance(
        forward,
        left,
        box.center_forward_m,
        box.center_left_m,
        box.yaw_ego_rad,
        box.length_m,
        box.width_m,
    )


def _realized_entry(
    *,
    label_id: str,
    origin_index: int,
    frames: Sequence[int],
    times: Sequence[float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
) -> tuple[str, float | None]:
    origin_time = float(times[origin_index])
    final_index = bisect.bisect_right(times, origin_time + HORIZON_S + EPSILON) - 1
    if final_index < origin_index or times[-1] - origin_time < HORIZON_S - 0.05:
        return "RIGHT_CENSORED", None
    future = []
    for index in range(origin_index, final_index + 1):
        matches = [
            box
            for box in boxes_by_frame.get(int(frames[index]), ())
            if str(box.label_id) == label_id
        ]
        if len(matches) != 1:
            return "RIGHT_CENSORED", None
        future.append((float(times[index]) - origin_time, matches[0]))
    for delta_s, box in future:
        if _obb_clearance(box, ROUTE_HALF_WIDTH_M) <= EPSILON:
            return "FULL_HORIZON", delta_s
    return "FULL_HORIZON", None


def _predict(
    *,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    ledger: Any,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard_s = HORIZON_S * FROZEN_R2_CONFIG.imminent_horizon_fraction
    origin_s = float(timestamps[int(frames[0])])
    times = [float(timestamps[int(frame)]) for frame in frames]
    margin_m = FROZEN_FLOW_CONFIG.association_margin_cells * FROZEN_FLOW_CONFIG.voxel_size_m
    raw_frames: list[int] = []
    active_frames: list[int] = []
    urgent_frames: list[int] = []
    minimum_entries: dict[str, float] = {}
    risky_cells: dict[str, int] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    future_cache: dict[tuple[int, str], tuple[str, float | None]] = {}

    for origin_index, frame in enumerate(frames):
        forward, left, vf, vl, _component = ledger.frame_cells(int(frame))
        entries: list[float] = []
        categories: Counter[str] = Counter()
        labels_by_category: dict[str, set[str]] = {}
        boxes = boxes_by_frame.get(int(frame), ())
        for values in zip(forward, left, vf, vl):
            cell_forward, cell_left, velocity_forward, velocity_left = (
                float(value) for value in values
            )
            cv_entry = _entry_s(
                cell_forward, cell_left, velocity_forward, velocity_left
            )
            matches = [
                box
                for box in boxes
                if _cell_clearance(cell_forward, cell_left, box) <= margin_m + EPSILON
            ]
            label_id = None
            if len(matches) != 1:
                category = "UNSUPPORTED_OR_AMBIGUOUS"
                chosen_entry = cv_entry
            else:
                label_id = str(matches[0].label_id)
                key = (origin_index, label_id)
                if key not in future_cache:
                    future_cache[key] = _realized_entry(
                        label_id=label_id,
                        origin_index=origin_index,
                        frames=frames,
                        times=times,
                        boxes_by_frame=boxes_by_frame,
                    )
                coverage, oracle_entry = future_cache[key]
                if coverage == "RIGHT_CENSORED":
                    category = "RIGHT_CENSORED"
                    chosen_entry = cv_entry
                else:
                    chosen_entry = oracle_entry
                    if cv_entry is not None and oracle_entry is not None:
                        category = "BOTH_HIT"
                    elif cv_entry is None and oracle_entry is not None:
                        category = "ORACLE_ONLY_HIT"
                    elif cv_entry is not None and oracle_entry is None:
                        category = "CV_ONLY_HIT"
                    else:
                        category = "NEITHER_HIT"
            categories[category] += 1
            if label_id is not None:
                labels_by_category.setdefault(category, set()).add(label_id)
            if chosen_entry is not None:
                entries.append(float(chosen_entry))

        minimum_entry = min(entries) if entries else None
        raw = minimum_entry is not None
        urgent = bool(raw and minimum_entry <= guard_s + EPSILON)
        signal = lifecycle.update(
            float(timestamps[int(frame)]) - origin_s, raw, urgent=urgent
        )
        if raw:
            raw_frames.append(int(frame))
            minimum_entries[str(int(frame))] = float(minimum_entry)
            risky_cells[str(int(frame))] = len(entries)
        if urgent:
            urgent_frames.append(int(frame))
        if signal in ACTIVE_SIGNALS:
            active_frames.append(int(frame))
        diagnostics[int(frame)] = {
            "categories": dict(sorted(categories.items())),
            "labels_by_category": {
                name: sorted(labels) for name, labels in sorted(labels_by_category.items())
            },
            "raw_alert": raw,
            "active_alert": signal in ACTIVE_SIGNALS,
        }
    return (
        {
            "raw_alert_frames": raw_frames,
            "active_alert_frames": active_frames,
            "urgent_frames": urgent_frames,
            "minimum_entry_s_by_frame": minimum_entries,
            "risky_cells_by_frame": risky_cells,
        },
        diagnostics,
    )


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def _frame_range(row: Mapping[str, Any]) -> set[int]:
    return set(range(int(row["first_frame"]), int(row["last_frame"]) + 1))


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve(strict=True)
    predictions_path = args.predictions.resolve(strict=True)
    c25_path = args.c25_result.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    c25 = json.loads(c25_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema")
    require(predictions.get("schema") == PREDICTION_SCHEMA, "predictions_schema")
    require(predictions.get("truth_blind") is True, "predictions_not_sealed")
    require(c25.get("schema") == C25_SCHEMA, "c25_schema")
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "labels_hash")
    require(roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path), "timestamps_hash")
    require(c25["source"]["sealed_predictions_sha256"] == sha256_file(predictions_path), "predictions_hash")
    prediction_rows = {str(row["sequence"]): row for row in predictions["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    require(set(prediction_rows) == set(roster_rows), "sequence_coverage")

    baseline_rows = {arm: [] for arm in ARMS}
    oracle_rows: list[dict[str, Any]] = []
    event_ledger: list[dict[str, Any]] = []
    false_ledger: list[dict[str, Any]] = []
    per_sequence: list[dict[str, Any]] = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamp_zip:
        for sequence in sorted(roster_rows):
            timestamps = _load_timestamps(timestamp_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(frames=frames, timestamps=timestamps, boxes_by_frame=boxes)
            prediction_row = prediction_rows[sequence]
            scores = {}
            for arm in ARMS:
                score = score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=_prediction_frames(frames, prediction_row["arms"][arm]),
                )
                scores[arm] = score
                baseline_rows[arm].append(score)
            source = prediction_row["sources"]["ledgers"][BASELINE]
            ledger = point_flow.load_ledger(
                Path(source["ledger"]), Path(source["manifest"]),
                expected_sequence=sequence, expected_frames=frames,
            )
            oracle_prediction, frame_diagnostics = _predict(
                frames=frames, timestamps=timestamps, boxes_by_frame=boxes, ledger=ledger
            )
            oracle_score = score_sequence(
                sequence=sequence,
                timeline=timeline,
                prediction_frames=_prediction_frames(frames, oracle_prediction),
            )
            oracle_rows.append(oracle_score)
            baseline_events = {row["event_id"]: row for row in scores[BASELINE]["event_rows"]}
            for index, oracle_event in enumerate(oracle_score["event_rows"]):
                event_id = str(oracle_event["event_id"])
                baseline_event = baseline_events[event_id]
                event = roster_rows[sequence]["bounded_contact_event_details"][index]
                categories: Counter[str] = Counter()
                supported_labels: set[str] = set()
                for frame in _frame_range(event):
                    diagnostic = frame_diagnostics.get(frame)
                    if diagnostic is None:
                        continue
                    categories.update(diagnostic["categories"])
                    for values in diagnostic["labels_by_category"].values():
                        supported_labels.update(values)
                baseline_lead = baseline_event["first_alert_lead_s"]
                oracle_lead = oracle_event["first_alert_lead_s"]
                event_ledger.append({
                    "event_id": event_id,
                    "responsible_components": list(event["responsible_components"]),
                    "responsible_component_has_unique_support": any(
                        str(label) in supported_labels for label in event["responsible_components"]
                    ),
                    "m1_pd_recalled": bool(baseline_event["recalled"]),
                    "oracle_recalled": bool(oracle_event["recalled"]),
                    "m1_pd_first_alert_lead_s": baseline_lead,
                    "oracle_first_alert_lead_s": oracle_lead,
                    "lead_delta_s": None if baseline_lead is None or oracle_lead is None else float(oracle_lead) - float(baseline_lead),
                    "categories": dict(sorted(categories.items())),
                })
            oracle_active = set(oracle_prediction["active_alert_frames"])
            for index, false_range in enumerate(scores[BASELINE]["false_alert_ranges"], start=1):
                relevant = _frame_range(false_range)
                categories: Counter[str] = Counter()
                for frame in relevant:
                    if frame in frame_diagnostics:
                        categories.update(frame_diagnostics[frame]["categories"])
                false_ledger.append({
                    "false_segment_id": f"{sequence}:m1-pd-false:{index:03d}",
                    "first_frame": int(false_range["first_frame"]),
                    "last_frame": int(false_range["last_frame"]),
                    "removed_by_oracle": not bool(relevant & oracle_active),
                    "categories": dict(sorted(categories.items())),
                })
            per_sequence.append({"sequence": sequence, "oracle": oracle_score, "baselines": scores})

    baselines = {arm: aggregate_scores(rows) for arm, rows in baseline_rows.items()}
    oracle = aggregate_scores(oracle_rows)
    for arm in ARMS:
        for metric in (
            "bounded_contact_events", "bounded_contact_events_recalled",
            "false_alert_segments", "bounded_contact_event_f1",
            "median_first_alert_lead_s",
        ):
            require(_same(baselines[arm][metric], c25["aggregate"][arm][metric]), f"c25_replay_drift:{arm}:{metric}")
    envelope = {
        "contact_events_recalled": max(int(row["bounded_contact_events_recalled"]) for row in baselines.values()),
        "false_alert_segments": min(int(row["false_alert_segments"]) for row in baselines.values()),
        "median_first_alert_lead_s": max(float(row["median_first_alert_lead_s"]) for row in baselines.values()),
    }
    checks = {
        "recall_reaches_c25_envelope": int(oracle["bounded_contact_events_recalled"]) >= envelope["contact_events_recalled"],
        "false_segments_reach_c25_envelope": int(oracle["false_alert_segments"]) <= envelope["false_alert_segments"],
        "lead_reaches_c25_envelope": float(oracle["median_first_alert_lead_s"]) + EPSILON >= envelope["median_first_alert_lead_s"],
    }
    passed = all(checks.values())
    result = {
        "schema": SCHEMA,
        "status": STATUS_MET if passed else STATUS_NOT_MET,
        "question": "Can perfect future OBB occupancy on uniquely supported sealed M1-PD cells reach the best C25 recall, false, and lead anchors simultaneously?",
        "oracle": oracle,
        "baselines": baselines,
        "gate": {"passed": passed, "checks": checks, "c25_componentwise_envelope": envelope},
        "counterfactual_ledger": {
            "events": event_ledger,
            "m1_pd_false_segments": false_ledger,
            "summary": {
                "events_with_unique_responsible_support": sum(row["responsible_component_has_unique_support"] for row in event_ledger),
                "events_recovered_over_m1_pd": sum((not row["m1_pd_recalled"]) and row["oracle_recalled"] for row in event_ledger),
                "m1_pd_false_segments_removed": sum(row["removed_by_oracle"] for row in false_ledger),
            },
        },
        "per_sequence": per_sequence,
        "source": {
            "roster": str(roster_path), "roster_sha256": sha256_file(roster_path),
            "sealed_predictions": str(predictions_path), "sealed_predictions_sha256": sha256_file(predictions_path),
            "c25_result": str(c25_path), "c25_result_sha256": sha256_file(c25_path),
            "labels_sha256": sha256_file(labels_path), "timestamps_sha256": sha256_file(timestamps_path),
        },
        "fixed_contract": {
            "support_margin_m": FROZEN_FLOW_CONFIG.association_margin_cells * FROZEN_FLOW_CONFIG.voxel_size_m,
            "support_rule": "exactly one current native OBB within frozen cell association margin",
            "future_rule": "same identity has exactly one native OBB at every frame through the full 3 s horizon",
            "unknown_rule": "unsupported, ambiguous, or right-censored cells retain sealed M1-PD constant-velocity entry",
            "replacement_rule": "full-horizon uniquely supported cells use first realized future native-OBB route entry instead of constant velocity",
            "route_half_width_m": ROUTE_HALF_WIDTH_M, "route_horizon_s": HORIZON_S,
            "lifecycle": "unchanged RiskEventLifecycle and R2 urgent boundary",
        },
        "compute": {"backend": "CPU", "reason": "sealed ledger/label replay and small scalar geometry remain CPU by contract; no model, batch tensor, or large point matching is launched"},
        "claim_limits": [
            "Future native OBB geometry and identity are evaluator-only privileged inputs.",
            "Current native OBB association is scorer-side and not deployable attribution.",
            "This consumed-cohort counterfactual can authorize or close training headroom only; it is not causal model performance.",
            "No product, user-benefit, reliability, or safety claim follows.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    c25 = REPO / "artifacts.local" / "evidence" / "dtr-c25" / "fresh-point-flow-confirmation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c25_fresh_confirmation_roster.json")
    parser.add_argument("--predictions", type=Path, default=c25 / "predictions.json")
    parser.add_argument("--c25-result", type=Path, default=c25 / "result.json")
    parser.add_argument("--labels", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_timestamps.zip")
    parser.add_argument("--output", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-c26" / "supported-future-occupancy-headroom" / "result.json")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"], "gate": result["gate"],
        "oracle": result["oracle"],
        "counterfactual_summary": result["counterfactual_ledger"]["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
