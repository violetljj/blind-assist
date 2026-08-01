"""Independently score the frozen RISKSEG-R0 three-arm parent-event gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SEEDS = (20260801, 20260802, 20260803)
DECISION_SEED = 20260801
ARMS = (
    "A_CURRENT_YOLO_ONLY",
    "B_LEARNED_SEGMENTATION_ONLY",
    "C_TRUTH_MASK_ORACLE_REFERENCE",
)
POSITIVE_BUCKETS = (
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def event_score(
    *,
    event: dict[str, Any],
    arm: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["frame_index"]))
    if [int(row["frame_index"]) for row in rows] != list(range(len(event["frames"]))):
        raise ValueError(f"{event['parent_event_id']}/{arm}: frame membership drift")
    positive = bool(event["positive"])
    if positive:
        alert_start, alert_end = map(int, event["alertable_interval_frames"])
        passed_start, passed_end = map(int, event["passed_interval_frames"])
        hit_frames = [
            index
            for index in range(alert_start, alert_end + 1)
            if rows[index]["actual_alert"] is True
        ]
        passed_alert_frames = [
            index
            for index in range(passed_start, passed_end + 1)
            if rows[index]["actual_alert"] is True
        ]
        runtime_exited = rows[passed_end]["risk_event_active"] is False
    else:
        hit_frames = []
        passed_alert_frames = []
        runtime_exited = False
    return {
        "parent_event_id": event["parent_event_id"],
        "source_session_id": event["source_session_id"],
        "bucket": event["bucket"],
        "positive": positive,
        "arm": arm,
        "event_hit": positive and bool(hit_frames),
        "first_alertable_alert_frame": hit_frames[0] if hit_frames else None,
        "critical_miss": positive and not hit_frames,
        "false_alert_event": (not positive)
        and any(row["actual_alert"] is True for row in rows),
        "passed_cleared": positive and not passed_alert_frames,
        "passed_alert_frame_count": len(passed_alert_frames),
        "runtime_exited_by_passed_end": runtime_exited,
        "delivered_alert_count": sum(row["actual_alert"] is True for row in rows),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [score for score in scores if score["positive"]]
    negatives = [score for score in scores if not score["positive"]]
    hit_count = sum(score["event_hit"] for score in positives)
    cleared_count = sum(score["passed_cleared"] for score in positives)
    return {
        "positive_event_count": len(positives),
        "hit_event_count": hit_count,
        "event_recall": hit_count / len(positives),
        "critical_miss_count": sum(score["critical_miss"] for score in positives),
        "negative_event_count": len(negatives),
        "false_alert_event_count": sum(
            score["false_alert_event"] for score in negatives
        ),
        "passed_event_count": len(positives),
        "cleared_event_count": cleared_count,
        "clearance_rate": cleared_count / len(positives),
        "runtime_exited_by_passed_end_count": sum(
            score["runtime_exited_by_passed_end"] for score in positives
        ),
        "bucket_hit_counts": {
            bucket: sum(
                score["event_hit"] for score in positives if score["bucket"] == bucket
            )
            for bucket in POSITIVE_BUCKETS
        },
        "bucket_miss_counts": {
            bucket: sum(
                score["critical_miss"]
                for score in positives
                if score["bucket"] == bucket
            )
            for bucket in POSITIVE_BUCKETS
        },
    }


def score_report(
    *,
    report_path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    expected_seed: int,
) -> dict[str, Any]:
    report = read_object(report_path)
    if (
        report.get("schema_version")
        != "blindassist.riskseg_r0.three_arm_device_report.v1"
        or report.get("protocol_id") != "RISKSEG_R0_EVENT_EVAL_V1"
        or int(report.get("seed", -1)) != expected_seed
        or report.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError(f"{report_path}: report identity mismatch")
    trace_path = report_path.parent / str(report["trace_file"])
    if sha256_file(trace_path) != report.get("trace_sha256"):
        raise ValueError(f"{report_path}: trace SHA-256 mismatch")
    rows = read_jsonl(trace_path)
    expected_rows = int(manifest["frame_count"]) * len(ARMS)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{report_path}: trace row count {len(rows)} != {expected_rows}"
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("schema_version")
            != "blindassist.riskseg_r0.three_arm_frame_trace.v1"
            or row.get("protocol_id") != "RISKSEG_R0_EVENT_EVAL_V1"
            or int(row.get("seed", -1)) != expected_seed
            or row.get("arm") not in ARMS
        ):
            raise ValueError(f"{report_path}: trace identity mismatch")
        grouped[(str(row["parent_event_id"]), str(row["arm"]))].append(row)

    events = {event["parent_event_id"]: event for event in manifest["events"]}
    scores: list[dict[str, Any]] = []
    for event_id, event in events.items():
        for arm in ARMS:
            key = (event_id, arm)
            if key not in grouped:
                raise ValueError(f"{report_path}: missing trace group {key}")
            scores.append(event_score(event=event, arm=arm, rows=grouped[key]))
    device_scores = {
        (item["parent_event_id"], item["arm"]): item
        for item in report["event_summaries"]
    }
    for score in scores:
        device = device_scores.get((score["parent_event_id"], score["arm"]))
        if device is None:
            raise ValueError(f"{report_path}: missing device event summary")
        normalized_device = {
            key: (None if device.get(key) is None else device.get(key))
            for key in score
        }
        if normalized_device != score:
            raise ValueError(
                f"{report_path}: device/host score mismatch "
                f"{score['parent_event_id']}/{score['arm']}"
            )
    by_arm = {
        arm: aggregate([score for score in scores if score["arm"] == arm])
        for arm in ARMS
    }
    return {
        "seed": expected_seed,
        "model_sha256": report["model_sha256"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "event_scores": scores,
        "arm_aggregates": by_arm,
    }


def quality_gates(scored: dict[str, Any]) -> dict[str, Any]:
    aggregate_by_arm = scored["arm_aggregates"]
    yolo = aggregate_by_arm[ARMS[0]]
    learned = aggregate_by_arm[ARMS[1]]
    score_by_key = {
        (score["parent_event_id"], score["arm"]): score
        for score in scored["event_scores"]
    }
    common: list[tuple[str, int]] = []
    for score in scored["event_scores"]:
        if score["arm"] != ARMS[0] or not score["event_hit"]:
            continue
        learned_score = score_by_key[(score["parent_event_id"], ARMS[1])]
        if learned_score["event_hit"]:
            delay = int(learned_score["first_alertable_alert_frame"]) - int(
                score["first_alertable_alert_frame"]
            )
            common.append((score["parent_event_id"], delay))
    delays = [delay for _, delay in common]
    if len(delays) >= 4:
        median_delay = float(statistics.median(delays))
        late_over_two_rate = sum(delay > 2 for delay in delays) / len(delays)
        timing_pass = median_delay <= 1.0 and late_over_two_rate <= 0.25
        timing_rule = "common_at_least_4"
    elif delays:
        median_delay = float(statistics.median(delays))
        late_over_two_rate = sum(delay > 2 for delay in delays) / len(delays)
        timing_pass = all(delay <= 2 for delay in delays)
        timing_rule = "common_1_to_3"
    else:
        median_delay = None
        late_over_two_rate = None
        timing_pass = True
        timing_rule = "NOT_APPLICABLE_ZERO_COMMON_HITS"

    gates = {
        "recall_gain": {
            "recall_delta": learned["event_recall"] - yolo["event_recall"],
            "additional_hit_events": (
                learned["hit_event_count"] - yolo["hit_event_count"]
            ),
        },
        "critical_miss": {
            "reduction": (
                yolo["critical_miss_count"] - learned["critical_miss_count"]
            ),
            "yolo_bucket_misses": yolo["bucket_miss_counts"],
            "learned_bucket_misses": learned["bucket_miss_counts"],
        },
        "false_alert": {
            "yolo_count": yolo["false_alert_event_count"],
            "learned_count": learned["false_alert_event_count"],
        },
        "clearance": {
            "yolo_count": yolo["cleared_event_count"],
            "learned_count": learned["cleared_event_count"],
            "yolo_rate": yolo["clearance_rate"],
            "learned_rate": learned["clearance_rate"],
        },
        "common_hit_timing": {
            "rule": timing_rule,
            "common_hit_count": len(common),
            "median_delay_frames": median_delay,
            "late_over_two_frames_rate": late_over_two_rate,
            "event_delays": [
                {"parent_event_id": event_id, "delay_frames": delay}
                for event_id, delay in common
            ],
        },
    }
    gates["recall_gain"]["passed"] = (
        gates["recall_gain"]["recall_delta"] >= 0.15
        and gates["recall_gain"]["additional_hit_events"] >= 3
    )
    gates["critical_miss"]["passed"] = (
        gates["critical_miss"]["reduction"] >= 2
        and all(
            learned["bucket_miss_counts"][bucket]
            <= yolo["bucket_miss_counts"][bucket]
            for bucket in POSITIVE_BUCKETS
        )
    )
    gates["false_alert"]["passed"] = (
        learned["false_alert_event_count"] <= yolo["false_alert_event_count"]
    )
    gates["clearance"]["passed"] = (
        learned["cleared_event_count"] >= yolo["cleared_event_count"]
        and learned["clearance_rate"] >= yolo["clearance_rate"]
    )
    gates["common_hit_timing"]["passed"] = timing_pass
    return {
        "seed": scored["seed"],
        "quality_gate_passed": all(
            gates[name]["passed"]
            for name in (
                "recall_gain",
                "critical_miss",
                "false_alert",
                "clearance",
                "common_hit_timing",
            )
        ),
        "gates": gates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-20260801-report", type=Path, required=True)
    parser.add_argument("--seed-20260802-report", type=Path, required=True)
    parser.add_argument("--seed-20260803-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    manifest = read_object(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    if (
        manifest.get("schema_version")
        != "blindassist.riskseg_r0.device_event_view.v1"
        or manifest.get("event_count") != 30
        or manifest.get("source_session_count") != 30
    ):
        raise ValueError("device event manifest contract mismatch")
    paths = {
        20260801: args.seed_20260801_report.resolve(),
        20260802: args.seed_20260802_report.resolve(),
        20260803: args.seed_20260803_report.resolve(),
    }
    scored = {
        seed: score_report(
            report_path=paths[seed],
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            expected_seed=seed,
        )
        for seed in SEEDS
    }
    gates = {seed: quality_gates(scored[seed]) for seed in SEEDS}
    decision_seed_pass = gates[DECISION_SEED]["quality_gate_passed"]
    passing_seed_count = sum(gate["quality_gate_passed"] for gate in gates.values())
    all_seed_guardrails = all(
        gate["gates"]["false_alert"]["passed"]
        and gate["gates"]["clearance"]["passed"]
        for gate in gates.values()
    )
    stability_pass = (
        decision_seed_pass and passing_seed_count >= 2 and all_seed_guardrails
    )
    result = {
        "schema_version": "blindassist.riskseg_r0.three_arm_validation.v1",
        "protocol_id": "RISKSEG_R0_EVENT_EVAL_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "RISKSEG_R0_EVENT_QUALITY_AND_STABILITY_PASS"
            if stability_pass
            else "RISKSEG_R0_EVENT_QUALITY_OR_STABILITY_FAIL"
        ),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "decision_seed": DECISION_SEED,
        "decision_seed_quality_pass": decision_seed_pass,
        "passing_seed_count": passing_seed_count,
        "all_seed_false_alert_and_clearance_guardrails_pass": all_seed_guardrails,
        "stability_pass": stability_pass,
        "per_seed": {
            str(seed): {
                "model_sha256": scored[seed]["model_sha256"],
                "report_sha256": scored[seed]["report_sha256"],
                "trace_sha256": scored[seed]["trace_sha256"],
                "arm_aggregates": scored[seed]["arm_aggregates"],
                **gates[seed],
            }
            for seed in SEEDS
        },
    }
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if stability_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
