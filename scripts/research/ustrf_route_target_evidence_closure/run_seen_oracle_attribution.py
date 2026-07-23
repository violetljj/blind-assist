from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from contract import load_json, sha256_file, validate_prereg

TRACKER_DIR = Path(__file__).resolve().parents[1] / "ustrf_tracker_ttc_ablation"
sys.path.insert(0, str(TRACKER_DIR))
from run_ablation import ArmState, associate, iou, predicted_box, route_hit  # noqa: E402


ACTIVE_ROLES = {"route_intersecting", "approaching_route"}


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q, method="higher")) if values else None


def run_current_evidence(
    frame_ids: list[str], detection_frames: dict[str, dict], route_frames: dict[str, dict],
    target_frames: dict[str, dict], tracker_config: dict, kernel: dict,
) -> list[dict[str, Any]]:
    state = ArmState()
    rows = []
    max_gap = int(tracker_config["arms"]["T0"]["max_gap_frames"])
    for frame_id in frame_ids:
        detection_frame = detection_frames[frame_id]
        detections = [row for row in detection_frame["post_nms_detections_canonical_320"] if row["class_id"] == 0]
        observed = associate(detections, int(frame_id), "T0", state, tracker_config)
        observed_ids = {track.track_id for track, _ in observed}
        route = route_frames[frame_id]
        route_known = route.get("status") == "known" and route.get("uv") is not None
        risk_boxes = []
        ages = []
        if route_known:
            width, height = detection_frame["source_size"]
            for track in state.tracks:
                if track.misses > max_gap:
                    continue
                box = track.box if track.track_id in observed_ids else predicted_box(track, "T0")
                if route_hit(box, route, width, height, float(kernel["route_point_margin_fraction"])):
                    risk_boxes.append(box)
                    ages.append(track.misses * 1000.0 / float(kernel["frame_rate_hz"]))
        target = target_frames.get(frame_id)
        target_risk = bool(target and target["visible_state"].startswith("visible_") and any(
            iou(box, target["target_bbox_xyxy"]) >= float(kernel["target_match_iou"]) for box in risk_boxes
        ))
        rows.append({"frame_id": frame_id, "route_known": route_known, "risk": bool(risk_boxes), "target_risk": target_risk, "ages_ms": ages})
    return rows


def run_o1(
    frame_ids: list[str], detection_frames: dict[str, dict], route_frames: dict[str, dict],
    role_frames: dict[str, list[dict]], kernel: dict,
) -> list[dict[str, Any]]:
    rows = []
    for frame_id in frame_ids:
        detection_frame = detection_frames[frame_id]
        route = route_frames[frame_id]
        route_known = route.get("status") == "known" and route.get("uv") is not None
        risk_people = []
        if route_known:
            width, height = detection_frame["source_size"]
            for person in role_frames.get(frame_id, []):
                if route_hit(person["bbox_xyxy"], route, width, height, float(kernel["route_point_margin_fraction"])):
                    risk_people.append(person)
        rows.append({
            "frame_id": frame_id,
            "route_known": route_known,
            "risk": bool(risk_people),
            "target_risk": any(person["is_frozen_target"] for person in risk_people),
            "ages_ms": [0.0] * len(risk_people),
        })
    return rows


def run_o2(
    frame_ids: list[str], detection_frames: dict[str, dict], route_frames: dict[str, dict],
    role_frames: dict[str, list[dict]], tracker_config: dict, kernel: dict,
) -> list[dict[str, Any]]:
    state = ArmState()
    rows = []
    for frame_id in frame_ids:
        detection_frame = detection_frames[frame_id]
        detections = [row for row in detection_frame["post_nms_detections_canonical_320"] if row["class_id"] == 0]
        observed = associate(detections, int(frame_id), "T0", state, tracker_config)
        people = role_frames.get(frame_id, [])
        candidates = []
        for track_index, (track, _) in enumerate(observed):
            for person_index, person in enumerate(people):
                overlap = iou(track.box, person["bbox_xyxy"])
                if overlap >= float(kernel["target_match_iou"]):
                    candidates.append((overlap, track_index, person_index))
        used_tracks: set[int] = set()
        used_people: set[int] = set()
        matched_people = []
        for _, track_index, person_index in sorted(candidates, reverse=True):
            if track_index not in used_tracks and person_index not in used_people:
                used_tracks.add(track_index)
                used_people.add(person_index)
                matched_people.append(people[person_index])
        route_known = route_frames[frame_id].get("status") == "known"
        active = [person for person in matched_people if person.get("role") in ACTIVE_ROLES]
        rows.append({
            "frame_id": frame_id,
            "route_known": route_known,
            "risk": bool(active) if route_known else False,
            "target_risk": any(person["is_frozen_target"] for person in active) if route_known else False,
            "ages_ms": [0.0] * len(active),
        })
    return rows


def current_kernel_alerts(rows: list[dict[str, Any]], kernel: dict) -> list[dict[str, Any]]:
    min_alert = int(kernel["min_alert_frames"])
    min_clear = int(kernel["min_clear_frames"])
    risk_run = clear_run = 0
    pending_target = False
    active = None
    alerts = []
    for row in rows:
        frame_number = int(row["frame_id"])
        if not row["route_known"]:
            if active is not None:
                alerts.append(active | {"end_frame": frame_number - 1, "closed": True, "reason": "route_unknown"})
                active = None
            risk_run = clear_run = 0
            pending_target = False
            continue
        if row["risk"]:
            risk_run += 1
            clear_run = 0
            pending_target = pending_target or row["target_risk"]
            if active is None and risk_run >= min_alert:
                active = {"start_frame": frame_number - min_alert + 1, "target_matched": pending_target}
            elif active is not None:
                active["target_matched"] = active["target_matched"] or row["target_risk"]
        else:
            risk_run = 0
            pending_target = False
            clear_run += 1
            if active is not None and clear_run >= min_clear:
                alerts.append(active | {"end_frame": frame_number - min_clear, "closed": True, "reason": "min_clear"})
                active = None
                clear_run = 0
    if active is not None:
        alerts.append(active | {"end_frame": int(rows[-1]["frame_id"]), "closed": False, "reason": "window_end"})
    return alerts


def score_window(window: dict, rows: list[dict[str, Any]], alerts: list[dict[str, Any]], *, evaluability: str) -> dict[str, Any]:
    frame_period_ms = 1000.0 / 15.0
    result = {
        "window_id": window["window_id"], "source_id": window["source_id"], "window_type": window["window_type"],
        "critical": bool(window.get("critical", False)), "evaluability": evaluability, "alerts": alerts,
        "duration_s": len(rows) / 15.0,
        "evidence_age_p95_ms": percentile([age for row in rows for age in row["ages_ms"]], 95),
        "alert_active_on_unknown_frame_count": 0,
    }
    if window["window_type"] == "negative":
        result["false_alert_count"] = len(alerts)
        return result
    truth = window["truth_anchors"]
    correct = [alert for alert in alerts if alert["target_matched"] and alert["start_frame"] <= int(truth["passed_or_cleared_frame"]) and alert["end_frame"] >= int(truth["alertable_frame"])]
    result["event_recalled"] = bool(correct)
    result["repeat_alert_count"] = max(0, len(correct) - 1)
    result["non_target_alert_count"] = sum(not alert["target_matched"] for alert in alerts)
    clearance = None
    if correct:
        final = correct[-1]
        if final["closed"] and final["end_frame"] < int(window["end_frame"]):
            clearance = max(0, final["end_frame"] + 1 - int(truth["passed_or_cleared_frame"])) * frame_period_ms
    result["clearance_success"] = clearance is not None
    result["clearance_delay_ms"] = clearance
    return result


def summarize(windows: list[dict[str, Any]], source_ids: list[str]) -> dict[str, Any]:
    sources = {}
    for source_id in source_ids:
        rows = [row for row in windows if row["source_id"] == source_id]
        positives = [row for row in rows if row["window_type"] == "positive"]
        negatives = [row for row in rows if row["window_type"] == "negative"]
        evaluable = [row for row in rows if row["evaluability"] == "evaluable"]
        evaluable_positive = [row for row in positives if row["evaluability"] == "evaluable"]
        negative_minutes = sum(row["duration_s"] for row in negatives) / 60.0
        false_count = sum(row.get("false_alert_count", 0) + row.get("non_target_alert_count", 0) for row in rows)
        clearance = [row["clearance_delay_ms"] for row in evaluable_positive if row.get("clearance_success")]
        sources[source_id] = {
            "status": "evaluable" if len(evaluable) == len(rows) else "not_evaluable_all_person_truth_incomplete",
            "evaluable_window_count": len(evaluable), "window_count": len(rows),
            "event_count": len(positives),
            "event_recall": sum(row.get("event_recalled", False) for row in evaluable_positive) / len(evaluable_positive) if evaluable_positive else None,
            "critical_miss_count": sum(row["critical"] and not row.get("event_recalled", False) for row in evaluable_positive),
            "false_alert_count": false_count,
            "false_alerts_per_minute": false_count / negative_minutes if negative_minutes else None,
            "clearance_rate": sum(row.get("clearance_success", False) for row in evaluable_positive) / len(evaluable_positive) if evaluable_positive else None,
            "clearance_p95_ms": percentile(clearance, 95),
            "repeat_alert_count": sum(row.get("repeat_alert_count", 0) for row in evaluable_positive),
            "evidence_age_p95_ms": percentile([row["evidence_age_p95_ms"] for row in evaluable if row["evidence_age_p95_ms"] is not None], 95),
        }
    return {"source_metrics": sources, "windows": windows, "status": "evaluable" if all(row["status"] == "evaluable" for row in sources.values()) else "not_evaluable_all_person_truth_incomplete"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite oracle result: {args.output}")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    expected_output = repo / config["seen_diagnostic"]["oracle_planned_output_path"]
    if args.output.resolve() != expected_output.resolve():
        raise ValueError("oracle output path differs from preregistration")
    parent = load_json(repo / "configs/ustrf_android_native_association_r1.json")
    tracker_config = load_json(repo / parent["tracker_parent_config_path"])
    windows_payload = load_json(repo / parent["windows_path"])
    ledger = load_json(repo / parent["canonical_detection_ledger_path"])
    target_truth = load_json(repo / parent["truth_path"])
    role_truth_path = repo / config["route_role_truth"]["model_proxy_geometry_contract"]["materialized_output"]["path"]
    role_truth = load_json(role_truth_path)
    scorer = load_json(repo / config["seen_truth_proposal_protocol"]["isolated_scorer_binding"]["path"])
    blind_by_window = {window["window_id"]: window["blind_window_id"] for source in scorer["sources"] for window in source["windows"]}
    quarantine_windows = {row["blind_window_id"] for row in role_truth["quarantined_identity_episodes"]}
    detections: dict[str, dict[str, dict]] = {source_id: {} for source_id in parent["route_sources"]}
    for row in ledger["frames"]:
        detections[row["source_id"]][row["frame_id"]] = row
    routes = {}
    for source_id, binding in parent["route_sources"].items():
        payload = load_json(repo / binding["path"])
        source = next(row for row in payload["sources"] if row["source_id"] == source_id)
        routes[source_id] = {row["frame_id"]: row for row in source["route_predictions"]}
    targets: dict[str, dict[str, dict]] = {source_id: {} for source_id in parent["route_sources"]}
    for source in target_truth["sources"]:
        for event in source["target_events"]:
            for frame in event["frames"]:
                targets[source["source_id"]][frame["frame_id"]] = frame
    role_by_window_frame: dict[str, dict[str, list[dict]]] = {}
    for source in role_truth["sources"]:
        for episode in source["person_episodes"]:
            window_rows = role_by_window_frame.setdefault(episode["blind_window_id"], {})
            for frame in episode["frames"]:
                if frame["visibility"] == "visible":
                    window_rows.setdefault(frame["frame_id"], []).append({
                        "bbox_xyxy": frame["bbox_xyxy"], "role": frame["role"],
                        "is_frozen_target": episode["is_frozen_target"], "person_id": episode["person_id"],
                    })

    results = {arm: [] for arm in ("T0_CURRENT", "O1_ORACLE_PERSON", "O2_ORACLE_ROUTE_RELATION", "O3_ORACLE_LIFECYCLE")}
    kernel = parent["fixed_kernel"]
    for window in windows_payload["windows"]:
        source_id = window["source_id"]
        blind_window_id = blind_by_window[window["window_id"]]
        frame_ids = [f"{number:06d}" for number in range(int(window["start_frame"]), int(window["end_frame"]) + 1)]
        current_rows = run_current_evidence(frame_ids, detections[source_id], routes[source_id], targets[source_id], tracker_config, kernel)
        results["T0_CURRENT"].append(score_window(window, current_rows, current_kernel_alerts(current_rows, kernel), evaluability="evaluable"))
        o1_rows = run_o1(frame_ids, detections[source_id], routes[source_id], role_by_window_frame.get(blind_window_id, {}), kernel)
        o1_eval = "not_evaluable_quarantined_person_episode" if blind_window_id in quarantine_windows else "evaluable"
        results["O1_ORACLE_PERSON"].append(score_window(window, o1_rows, current_kernel_alerts(o1_rows, kernel), evaluability=o1_eval))
        o2_rows = run_o2(frame_ids, detections[source_id], routes[source_id], role_by_window_frame.get(blind_window_id, {}), tracker_config, kernel)
        results["O2_ORACLE_ROUTE_RELATION"].append(score_window(window, o2_rows, current_kernel_alerts(o2_rows, kernel), evaluability="evaluable"))
        if window["window_type"] == "positive":
            truth = window["truth_anchors"]
            evidence = [row for row in current_rows if int(truth["alertable_frame"]) <= int(row["frame_id"]) <= int(truth["passed_or_cleared_frame"]) and row["target_risk"]]
            oracle_alerts = [{"start_frame": int(evidence[0]["frame_id"]), "end_frame": int(truth["passed_or_cleared_frame"]) - 1, "target_matched": True, "closed": True, "reason": "oracle_truth_lifecycle"}] if evidence else []
        else:
            oracle_alerts = []
        results["O3_ORACLE_LIFECYCLE"].append(score_window(window, current_rows, oracle_alerts, evaluability="evaluable"))

    source_ids = list(parent["route_sources"])
    summaries = {arm: summarize(rows, source_ids) for arm, rows in results.items()}
    payload = {
        "schema": "blindassist_ustrf_route_target_seen_oracle_attribution_r1",
        "authority": "seen_diagnostic_failure_attribution_only_no_candidate_selection_or_production_authority",
        "config_sha256": sha256_file(args.config),
        "route_role_truth_sha256": sha256_file(role_truth_path),
        "h2_candidate_depth_used": False,
        "oracle_person_truth_is_model_proxy": True,
        "results": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({arm: {"status": row["status"], "source_metrics": row["source_metrics"]} for arm, row in summaries.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
