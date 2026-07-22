from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

TRACKER_DIR = Path(__file__).resolve().parents[1] / "ustrf_tracker_ttc_ablation"
sys.path.insert(0, str(TRACKER_DIR))
from run_ablation import ArmState, associate, iou, predicted_box, route_hit  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bound(path: Path, expected: str) -> dict:
    if sha256(path) != expected:
        raise ValueError(f"evidence hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q, method="higher")) if values else None


def evaluate_window(
    window: dict,
    arm: str,
    tracker_config: dict,
    kernel: dict,
    detection_frames: dict[str, dict],
    route_frames: dict[str, dict],
    target_frames: dict[str, dict],
) -> dict:
    state = ArmState()
    max_gap = int(tracker_config["arms"][arm]["max_gap_frames"])
    min_alert = int(kernel["min_alert_frames"])
    min_clear = int(kernel["min_clear_frames"])
    frame_period_ms = 1000.0 / float(kernel["frame_rate_hz"])
    risk_run = clear_run = 0
    pending_target_match = False
    pending_track_ids: set[int] = set()
    active: dict | None = None
    alerts: list[dict] = []
    target_track_ids: set[int] = set()
    evidence_ages_ms: list[float] = []
    unknown_route_frames = alert_active_on_unknown_frames = 0
    frame_ids = [f"{value:06d}" for value in range(int(window["start_frame"]), int(window["end_frame"]) + 1)]
    for frame_id in frame_ids:
        detection_frame = detection_frames[frame_id]
        route = route_frames[frame_id]
        detections = [
            row for row in detection_frame["post_nms_detections_canonical_320"]
            if row["class_id"] == 0
        ]
        observed = associate(detections, int(frame_id), arm, state, tracker_config)
        observed_ids = {track.track_id for track, _ in observed}
        target = target_frames.get(frame_id)
        if target is not None and target["visible_state"].startswith("visible_"):
            target_box = target["target_bbox_xyxy"]
            for track, _ in observed:
                if iou(track.box, target_box) >= float(kernel["target_match_iou"]):
                    target_track_ids.add(track.track_id)
        route_known = route.get("status") == "known" and route.get("uv") is not None
        if not route_known:
            unknown_route_frames += 1
        risk_tracks: list[tuple[object, list[float]]] = []
        if route_known:
            width, height = detection_frame["source_size"]
            for track in state.tracks:
                if track.misses > max_gap:
                    continue
                box = track.box if track.track_id in observed_ids else predicted_box(track, arm)
                if route_hit(box, route, width, height, float(kernel["route_point_margin_fraction"])):
                    risk_tracks.append((track, box))
                    evidence_ages_ms.append(track.misses * frame_period_ms)
        target_match = False
        risk_track_ids = {track.track_id for track, _ in risk_tracks}
        if target is not None and target["visible_state"].startswith("visible_"):
            target_match = any(
                iou(box, target["target_bbox_xyxy"]) >= float(kernel["target_match_iou"])
                for _, box in risk_tracks
            )
        risk = bool(risk_tracks)
        frame_number = int(frame_id)
        if not route_known:
            if active is not None:
                alerts.append(active | {"end_frame": frame_number - 1, "closed_by_kernel": True, "closed_reason": "route_unknown_abstain"})
                active = None
            risk_run = clear_run = 0
            pending_target_match = False
            pending_track_ids.clear()
            continue
        if risk:
            clear_run = 0
            risk_run += 1
            pending_target_match = pending_target_match or target_match
            pending_track_ids.update(risk_track_ids)
            if active is None and risk_run >= min_alert:
                active = {
                    "start_frame": frame_number - min_alert + 1,
                    "target_matched": pending_target_match,
                    "risk_track_ids": sorted(pending_track_ids),
                }
            elif active is not None:
                active["target_matched"] = active["target_matched"] or target_match
                active["risk_track_ids"] = sorted(set(active["risk_track_ids"]) | risk_track_ids)
        else:
            risk_run = 0
            pending_target_match = False
            pending_track_ids.clear()
            clear_run += 1
            if active is not None and clear_run >= min_clear:
                alerts.append(active | {
                    "end_frame": frame_number - min_clear,
                    "closed_by_kernel": True,
                    "closed_reason": "min_clear_frames",
                })
                active = None
                clear_run = 0
        if active is not None and not route_known:
            alert_active_on_unknown_frames += 1
    if active is not None:
        alerts.append(active | {
            "end_frame": int(frame_ids[-1]),
            "closed_by_kernel": False,
            "closed_reason": "window_end_still_active",
        })
    result = {
        "window_id": window["window_id"],
        "source_id": window["source_id"],
        "window_type": window["window_type"],
        "critical": bool(window.get("critical", False)),
        "alerts": alerts,
        "unknown_route_frame_count": unknown_route_frames,
        "alert_active_on_unknown_frame_count": alert_active_on_unknown_frames,
        "track_births": state.track_births,
        "track_fragmentations": state.fragmented_track_count,
        "target_track_fragmentation": max(0, len(target_track_ids) - 1),
        "evidence_age_p95_ms": percentile(evidence_ages_ms, 95),
        "duration_s": len(frame_ids) / float(kernel["frame_rate_hz"]),
    }
    if window["window_type"] == "positive":
        truth = window["truth_anchors"]
        correct = [
            alert for alert in alerts
            if alert["target_matched"]
            and alert["start_frame"] <= int(truth["passed_or_cleared_frame"])
            and alert["end_frame"] >= int(truth["alertable_frame"])
        ]
        result["event_recalled"] = bool(correct)
        result["correct_alert_count"] = len(correct)
        result["repeat_alert_count"] = max(0, len(correct) - 1)
        result["first_correct_alert_delay_ms"] = (
            max(0, correct[0]["start_frame"] - int(truth["alertable_frame"])) * frame_period_ms
            if correct else None
        )
        clearance = None
        if correct:
            final_alert = correct[-1]
            first_clear_frame = final_alert["end_frame"] + 1
            if (
                final_alert["closed_by_kernel"]
                and final_alert["end_frame"] < int(window["end_frame"])
            ):
                # An alert already inactive when the truth reaches clear has
                # zero clearance latency; requiring it to persist until clear
                # would reward late clearing and invert the metric.
                clearance = max(0, first_clear_frame - int(truth["passed_or_cleared_frame"])) * frame_period_ms
        result["clearance_success"] = clearance is not None
        result["clearance_delay_ms"] = clearance
        result["non_target_alert_count"] = sum(int(not alert["target_matched"]) for alert in alerts)
    else:
        result["false_alert_count"] = len(alerts)
    return result


def summarize(windows: list[dict], source_ids: list[str]) -> dict:
    positives = [row for row in windows if row["window_type"] == "positive"]
    negatives = [row for row in windows if row["window_type"] == "negative"]
    source_metrics: dict[str, dict] = {}
    for source_id in source_ids:
        source_positive = [row for row in positives if row["source_id"] == source_id]
        source_negative = [row for row in negatives if row["source_id"] == source_id]
        critical = [row for row in source_positive if row["critical"]]
        clearance = [row["clearance_delay_ms"] for row in source_positive if row["clearance_success"]]
        delays = [row["first_correct_alert_delay_ms"] for row in source_positive if row["event_recalled"]]
        evidence = [row["evidence_age_p95_ms"] for row in source_positive + source_negative if row["evidence_age_p95_ms"] is not None]
        negative_minutes = sum(row["duration_s"] for row in source_negative) / 60.0
        source_metrics[source_id] = {
            "event_count": len(source_positive),
            "event_recall": sum(int(row["event_recalled"]) for row in source_positive) / len(source_positive),
            "critical_miss_count": sum(int(not row["event_recalled"]) for row in critical),
            "false_alert_count": sum(row["false_alert_count"] for row in source_negative),
            "false_alerts_per_minute": sum(row["false_alert_count"] for row in source_negative) / negative_minutes,
            "first_correct_alert_delay_p95_ms": percentile(delays, 95),
            "clearance_rate": sum(int(row["clearance_success"]) for row in source_positive) / len(source_positive),
            "clearance_p95_ms": percentile(clearance, 95),
            "repeat_alert_count": sum(row["repeat_alert_count"] for row in source_positive),
            "target_track_fragmentation": sum(row["target_track_fragmentation"] for row in source_positive),
            "evidence_age_p95_ms": percentile(evidence, 95),
            "unknown_route_frame_count": sum(row["unknown_route_frame_count"] for row in source_positive + source_negative),
            "alert_active_on_unknown_frame_count": sum(row["alert_active_on_unknown_frame_count"] for row in source_positive + source_negative),
        }
    clearance = [row["clearance_delay_ms"] for row in positives if row["clearance_success"]]
    delays = [row["first_correct_alert_delay_ms"] for row in positives if row["event_recalled"]]
    evidence = [row["evidence_age_p95_ms"] for row in windows if row["evidence_age_p95_ms"] is not None]
    negative_minutes = sum(row["duration_s"] for row in negatives) / 60.0
    result = {
        "event_recall": sum(int(row["event_recalled"]) for row in positives) / len(positives),
        "critical_miss_count": sum(int(row["critical"] and not row["event_recalled"]) for row in positives),
        "false_alert_count": sum(row["false_alert_count"] for row in negatives),
        "false_alerts_per_minute": sum(row["false_alert_count"] for row in negatives) / negative_minutes,
        "first_correct_alert_delay_p95_ms": percentile(delays, 95),
        "clearance_rate": sum(int(row["clearance_success"]) for row in positives) / len(positives),
        "clearance_p95_ms": percentile(clearance, 95),
        "repeat_alert_count": sum(row["repeat_alert_count"] for row in positives),
        "target_track_fragmentation": sum(row["target_track_fragmentation"] for row in positives),
        "evidence_age_p95_ms": percentile(evidence, 95),
        "source_metrics": source_metrics,
    }
    result["worst_source"] = {
        "event_recall": min(source_metrics, key=lambda key: source_metrics[key]["event_recall"]),
        "critical_miss": max(source_metrics, key=lambda key: source_metrics[key]["critical_miss_count"]),
        "false_alerts_per_minute": max(source_metrics, key=lambda key: source_metrics[key]["false_alerts_per_minute"]),
        "clearance_rate": min(source_metrics, key=lambda key: source_metrics[key]["clearance_rate"]),
    }
    result["windows"] = windows
    return result


def winner_key(item: tuple[str, dict]) -> tuple:
    arm, row = item
    worst_recall = min(value["event_recall"] for value in row["source_metrics"].values())
    clearance_p95 = row["clearance_p95_ms"] if row["clearance_p95_ms"] is not None else math.inf
    delay_p95 = row["first_correct_alert_delay_p95_ms"] if row["first_correct_alert_delay_p95_ms"] is not None else math.inf
    evidence_p95 = row["evidence_age_p95_ms"] if row["evidence_age_p95_ms"] is not None else math.inf
    return (
        row["critical_miss_count"], -worst_recall, row["false_alerts_per_minute"],
        -row["clearance_rate"], clearance_p95, row["repeat_alert_count"], delay_p95,
        row["target_track_fragmentation"], evidence_p95, arm,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite result: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    attribution = read_bound(Path(config["detector_target_attribution_result_path"]), config["detector_target_attribution_result_sha256"])
    if attribution["decision"] != config["required_detector_decision"]:
        raise ValueError("detector target attribution did not reopen T0-T3")
    tracker_config = read_bound(Path(config["tracker_parent_config_path"]), config["tracker_parent_config_sha256"])
    windows_payload = read_bound(Path(config["windows_path"]), config["windows_sha256"])
    truth = read_bound(Path(config["truth_path"]), config["truth_sha256"])
    ledger = read_bound(Path(config["canonical_detection_ledger_path"]), config["canonical_detection_ledger_sha256"])
    source_ids = list(config["route_sources"])
    detection_by_source: dict[str, dict[str, dict]] = {source_id: {} for source_id in source_ids}
    for row in ledger["frames"]:
        detection_by_source[row["source_id"]][row["frame_id"]] = row
    route_by_source: dict[str, dict[str, dict]] = {}
    for source_id, binding in config["route_sources"].items():
        route_payload = read_bound(Path(binding["path"]), binding["sha256"])
        source = next(row for row in route_payload["sources"] if row["source_id"] == source_id)
        route_by_source[source_id] = {row["frame_id"]: row for row in source["route_predictions"]}
    target_by_source: dict[str, dict[str, dict]] = {source_id: {} for source_id in source_ids}
    for source in truth["sources"]:
        for event in source["target_events"]:
            for frame in event["frames"]:
                target_by_source[source["source_id"]][frame["frame_id"]] = frame
    results: dict[str, dict] = {}
    for arm in config["arms"]:
        window_results = []
        for window in windows_payload["windows"]:
            source_id = window["source_id"]
            window_results.append(evaluate_window(
                window, arm, tracker_config, config["fixed_kernel"],
                detection_by_source[source_id], route_by_source[source_id], target_by_source[source_id],
            ))
        results[arm] = summarize(window_results, source_ids)
    winner = min(results.items(), key=winner_key)[0]
    winner_result = results[winner]
    shadow_gate = all(
        row["event_recall"] >= config["shadow_entry_requires"]["event_recall_each_source"]
        and row["critical_miss_count"] <= config["shadow_entry_requires"]["critical_miss_each_source"]
        and row["alert_active_on_unknown_frame_count"] == config["shadow_entry_requires"]["unknown_route_alert_count"]
        for row in winner_result["source_metrics"].values()
    )
    payload = {
        "schema": "blindassist_ustrf_android_native_association_result_r1",
        "authority": "benchmark_only_association_selection_no_app_or_production_authority",
        "config_sha256": sha256(args.config),
        "detector_fixed": True,
        "route_and_event_kernel_fixed": True,
        "ttc_or_depth_used": False,
        "results": results,
        "winner": winner,
        "winner_selection_order": config["winner_lexicographic_order"],
        "shadow_entry_gate_passed": shadow_gate,
        "decision": "ENTER_PRODUCTION_ISOLATED_ANDROID_SHADOW" if shadow_gate else "STOP_BEFORE_ANDROID_SHADOW",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "winner": winner,
        "shadow_entry_gate_passed": shadow_gate,
        "decision": payload["decision"],
        "arms": {arm: {key: row[key] for key in (
            "event_recall", "critical_miss_count", "false_alerts_per_minute",
            "first_correct_alert_delay_p95_ms", "clearance_rate", "clearance_p95_ms",
            "repeat_alert_count", "target_track_fragmentation", "evidence_age_p95_ms",
        )} for arm, row in results.items()},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
