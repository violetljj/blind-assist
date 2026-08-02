from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .common import (
    ACTION_REVIEW_SCHEMA,
    ARMS,
    BUCKETS,
    COHORT_SCHEMA,
    EXCLUSION_SCHEMA,
    FULL_EVENT_FACTS_SCHEMA,
    KNOWNNESS,
    NEGATIVE_BUCKETS,
    POSITIVE_BUCKETS,
    PROTOCOL_ID,
    REPRESENTATION_ARMS,
    SCENE_FRAME_SCHEMA,
    THREE_STATE,
    TRACE_MANIFEST_SCHEMA,
    TRACE_SCHEMA,
    read_json,
    read_jsonl,
    require,
    require_finite_nonnegative,
    sha256_file,
    sha256_json,
)


P0_ANCHOR_AGREEMENT_SCHEMA = "blindassist.eval_validity_r0.p0_anchor_agreement.v1"
P1_ACTION_FACTS_SCHEMA = "blindassist.eval_validity_r0.p1_action_facts.v1"


def _validate_contract(contract: dict[str, Any]) -> None:
    require(contract.get("protocol_id") == PROTOCOL_ID, "contract: protocol mismatch")
    require(contract.get("status") == "PRE_OUTPUT_LOCKED", "contract: must be pre-output locked")
    gates = contract.get("gates")
    require(isinstance(gates, dict), "contract: missing gates")
    for key in (
        "minimum_actionability_exact_agreement",
        "minimum_clearance_exact_agreement",
        "minimum_knownness_exact_agreement",
        "minimum_sequence_exact_agreement",
        "maximum_unknown_burden",
        "maximum_response_delay_frames",
    ):
        require(key in gates, f"contract: missing gate {key}")


def _validate_exclusions(registry: dict[str, Any]) -> set[str]:
    require(registry.get("schema_version") == EXCLUSION_SCHEMA, "exclusion registry: schema mismatch")
    require(registry.get("protocol_id") == PROTOCOL_ID, "exclusion registry: protocol mismatch")
    sessions = registry.get("excluded_source_sessions")
    require(isinstance(sessions, list) and all(isinstance(value, str) and value for value in sessions), "exclusion registry: invalid sessions")
    require(len(sessions) == len(set(sessions)), "exclusion registry: duplicate excluded session")
    return set(sessions)


def _load_cohort(cohort: dict[str, Any], excluded_sessions: set[str], contract: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    require(cohort.get("schema_version") == COHORT_SCHEMA, "cohort: schema mismatch")
    require(cohort.get("protocol_id") == PROTOCOL_ID, "cohort: protocol mismatch")
    require(cohort.get("status") in {"SCENE_FACTS_FROZEN_ACTION_REVIEWS_PENDING", "SCENE_AND_EVENT_FACTS_FROZEN_AFTER_P0_P1"}, "cohort: wrong status")
    require(cohort.get("candidate_outputs_opened") is False, "cohort: candidate output access is forbidden before action review")
    items = cohort.get("items")
    require(isinstance(items, list) and items, "cohort: missing items")
    by_event: dict[str, dict[str, Any]] = {}
    sessions: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    for item in items:
        require(isinstance(item, dict), "cohort: item must be object")
        event_id = item.get("parent_event_id")
        session_id = item.get("source_session_id")
        bucket = item.get("bucket")
        frames = item.get("frame_indices")
        anchors = item.get("anchor_frame_indices")
        require(isinstance(event_id, str) and event_id and event_id not in by_event, "cohort: duplicate or missing parent_event_id")
        require(isinstance(session_id, str) and session_id and session_id not in excluded_sessions, f"cohort: excluded source session {session_id}")
        require(isinstance(bucket, str) and bucket in BUCKETS, f"cohort: invalid bucket for {event_id}")
        require(isinstance(frames, list) and len(frames) >= 20 and all(isinstance(value, int) for value in frames), f"cohort: invalid frame indices for {event_id}")
        require(frames == list(range(frames[0], frames[-1] + 1)), f"cohort: frame indices must be contiguous for {event_id}")
        require(isinstance(anchors, list) and anchors and all(isinstance(value, int) for value in anchors), f"cohort: invalid anchors for {event_id}")
        require(anchors == sorted(set(anchors)) and set(anchors).issubset(frames), f"cohort: anchor outside event for {event_id}")
        require(isinstance(item.get("scene_fact_manifest_sha256"), str) and len(item["scene_fact_manifest_sha256"]) == 64, f"cohort: missing scene fact hash for {event_id}")
        by_event[event_id] = item
        sessions.add(session_id)
        bucket_counts[bucket] += 1
    minimum = contract["cohort_requirements"]
    require(len(by_event) >= int(minimum["minimum_parent_events"]), "cohort: insufficient parent events")
    require(len(sessions) >= int(minimum["minimum_source_sessions"]), "cohort: insufficient source sessions")
    if minimum.get("one_event_per_session") is True:
        require(len(sessions) == len(by_event), "cohort: each parent event must have its own source session")
    for bucket, floor in minimum["minimum_bucket_parent_events"].items():
        require(bucket_counts[bucket] >= int(floor), f"cohort: {bucket} below floor")
    return by_event, sha256_json(cohort)


def _load_action_review(path: Path, role: str, cohort_sha256: str) -> dict[str, Any]:
    review = read_json(path)
    require(review.get("schema_version") == ACTION_REVIEW_SCHEMA, f"{path}: review schema mismatch")
    require(review.get("protocol_id") == PROTOCOL_ID, f"{path}: protocol mismatch")
    require(review.get("reviewer_role") == role, f"{path}: reviewer role mismatch")
    require(review.get("cohort_sha256") == cohort_sha256, f"{path}: cohort hash mismatch")
    require(review.get("isolated_context") is True, f"{path}: reviewer isolation missing")
    require(review.get("other_review_visible_before_submission") is False, f"{path}: cross-review visibility")
    require(review.get("model_or_oracle_output_visible") is False, f"{path}: model output visibility")
    items = review.get("items")
    require(isinstance(items, list) and items, f"{path}: missing review items")
    return review


def _review_index(review: dict[str, Any], review_map: dict[str, Any], cohort: dict[str, dict[str, Any]], where: str) -> dict[str, dict[int, dict[str, str]]]:
    result: dict[str, dict[int, dict[str, str]]] = {}
    forbidden_review_fields = {
        "parent_event_id", "source_session_id", "bucket", "positive", "oracle_mask",
        "truth_mask", "truth_box", "yolo_output", "model_output", "feedback_trace",
    }
    require(not (forbidden_review_fields & set(review)), f"{where}: reviewer envelope leaks forbidden context")
    for item in review["items"]:
        require(isinstance(item, dict), f"{where}: invalid review item")
        require(not (forbidden_review_fields & set(item)), f"{where}: reviewer item leaks forbidden context")
        opaque_id = item.get("review_item_id")
        mapping = review_map.get(opaque_id) if isinstance(opaque_id, str) else None
        require(isinstance(mapping, dict), f"{where}: unbound opaque review item")
        event_id, expected_frame = mapping.get("parent_event_id"), mapping.get("anchor_frame_index")
        require(event_id in cohort and isinstance(expected_frame, int), f"{where}: invalid opaque review binding")
        require("anchors" not in item, f"{where}: a review item may expose exactly one causal anchor")
        anchor = item.get("anchor")
        require(isinstance(anchor, dict), f"{where}: missing causal anchor")
        frame = anchor.get("frame_index")
        require(frame == expected_frame and frame in cohort[event_id]["anchor_frame_indices"], f"{where}: anchor identity mismatch")
        per_frame = result.setdefault(event_id, {})
        require(frame not in per_frame, f"{where}: duplicate anchor review")
        actionability = anchor.get("reminder_now")
        clearance = anchor.get("cleared")
        knownness = anchor.get("knownness")
        require(actionability in THREE_STATE, f"{where}: invalid reminder_now")
        require(clearance in THREE_STATE, f"{where}: invalid cleared")
        require(knownness in KNOWNNESS, f"{where}: invalid knownness")
        per_frame[frame] = {"reminder_now": actionability, "cleared": clearance, "knownness": knownness}
    require(set(result) == set(cohort), f"{where}: review coverage mismatch")
    for event_id, per_frame in result.items():
        require(sorted(per_frame) == cohort[event_id]["anchor_frame_indices"], f"{where}: anchor coverage mismatch for {event_id}")
    return result


def _agreement_and_event_facts(
    review_a: dict[str, dict[int, dict[str, str]]], review_b: dict[str, dict[int, dict[str, str]]], cohort: dict[str, dict[str, Any]], contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    matches = Counter()
    total = 0
    sequence_matches = 0
    unknown = 0
    event_facts: dict[str, dict[str, Any]] = {}
    for event_id, item in cohort.items():
        anchor_facts: dict[int, dict[str, str]] = {}
        same_sequence = True
        for frame in item["anchor_frame_indices"]:
            left, right = review_a[event_id][frame], review_b[event_id][frame]
            total += 1
            for field in ("reminder_now", "cleared", "knownness"):
                if left[field] == right[field]:
                    matches[field] += 1
                else:
                    same_sequence = False
            if all(left[field] == right[field] for field in ("reminder_now", "cleared", "knownness")) and "UNKNOWN" not in (left["reminder_now"], left["cleared"], left["knownness"]):
                anchor_facts[frame] = left
            else:
                # A raw abstention and an unresolved disagreement are both unavailable
                # event facts; neither may silently become a no-alert/no-risk label.
                unknown += 1
                anchor_facts[frame] = {"reminder_now": "UNKNOWN", "cleared": "UNKNOWN", "knownness": "UNKNOWN"}
        sequence_matches += int(same_sequence)
        event_facts[event_id] = {"bucket": item["bucket"], "anchors": anchor_facts}
    require(total > 0, "action review: zero anchors")
    metrics = {
        "reminder_now_exact_agreement": matches["reminder_now"] / total,
        "cleared_exact_agreement": matches["cleared"] / total,
        "knownness_exact_agreement": matches["knownness"] / total,
        "parent_event_sequence_exact_agreement": sequence_matches / len(cohort),
        "unknown_anchor_burden": unknown / total,
        "anchor_count": total,
    }
    gates = contract["gates"]
    passed = (
        metrics["reminder_now_exact_agreement"] >= gates["minimum_actionability_exact_agreement"]
        and metrics["cleared_exact_agreement"] >= gates["minimum_clearance_exact_agreement"]
        and metrics["knownness_exact_agreement"] >= gates["minimum_knownness_exact_agreement"]
        and metrics["parent_event_sequence_exact_agreement"] >= gates["minimum_sequence_exact_agreement"]
        and metrics["unknown_anchor_burden"] <= gates["maximum_unknown_burden"]
    )
    return {"metrics": metrics, "passed": passed}, event_facts


def _interval(value: Any, frames: list[int], where: str) -> list[int]:
    require(isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value), f"{where}: invalid interval")
    require(value[0] in frames and value[1] in frames and value[0] <= value[1], f"{where}: interval outside event")
    return value


def _load_full_event_facts(value: dict[str, Any], cohort: dict[str, dict[str, Any]], cohort_sha256: str, anchor_agreement: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(value.get("schema_version") == FULL_EVENT_FACTS_SCHEMA, "full event facts: schema mismatch")
    require(value.get("protocol_id") == PROTOCOL_ID, "full event facts: protocol mismatch")
    require(value.get("cohort_sha256") == cohort_sha256, "full event facts: cohort hash mismatch")
    require(value.get("status") == "FULL_EVENT_FACTS_FROZEN_AFTER_ANCHOR_CONSISTENCY", "full event facts: wrong status")
    require(value.get("anchor_consistency_sha256") == sha256_json(anchor_agreement), "full event facts: anchor agreement binding mismatch")
    evidence = value.get("independent_full_review_evidence")
    require(isinstance(evidence, dict), "full event facts: missing independent review evidence")
    for key in ("review_a_sha256", "review_b_sha256"):
        require(isinstance(evidence.get(key), str) and len(evidence[key]) == 64, f"full event facts: invalid {key}")
    require(evidence.get("reviewers_isolated") is True and evidence.get("model_or_oracle_output_visible") is False, "full event facts: reviewer isolation failure")
    require(evidence.get("agreement_passed") is True and evidence.get("unknown_anchor_or_frame_count") == 0, "full event facts: unresolved review fact")
    items = value.get("items")
    require(isinstance(items, list), "full event facts: missing items")
    facts: dict[str, dict[str, Any]] = {}
    for item in items:
        require(isinstance(item, dict), "full event facts: invalid item")
        event_id = item.get("parent_event_id")
        require(isinstance(event_id, str) and event_id in cohort and event_id not in facts, "full event facts: invalid/duplicate event")
        cohort_item = cohort[event_id]
        bucket = item.get("bucket")
        require(bucket == cohort_item["bucket"], f"full event facts: bucket drift for {event_id}")
        if bucket in POSITIVE_BUCKETS:
            alertable = _interval(item.get("alertable_interval_frames"), cohort_item["frame_indices"], f"full event facts: alertable {event_id}")
            passed = _interval(item.get("passed_interval_frames"), cohort_item["frame_indices"], f"full event facts: passed {event_id}")
            require(passed[0] > alertable[1], f"full event facts: passed must follow alertable for {event_id}")
            facts[event_id] = {"bucket": bucket, "alertable_interval_frames": alertable, "passed_interval_frames": passed}
        else:
            require(item.get("alertable_interval_frames") is None and item.get("passed_interval_frames") is None, f"full event facts: negative interval for {event_id}")
            facts[event_id] = {"bucket": bucket, "alertable_interval_frames": None, "passed_interval_frames": None}
    require(set(facts) == set(cohort), "full event facts: coverage mismatch")
    return facts


def _load_scene_rows(path: Path, cohort: dict[str, dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        require(row.get("schema_version") == SCENE_FRAME_SCHEMA, f"{path}: scene schema mismatch")
        require(row.get("protocol_id") == PROTOCOL_ID, f"{path}: scene protocol mismatch")
        event_id, arm, frame = row.get("parent_event_id"), row.get("arm"), row.get("frame_index")
        require(event_id in cohort and arm in REPRESENTATION_ARMS and isinstance(frame, int), f"{path}: invalid scene row identity")
        require(frame in cohort[event_id]["frame_indices"], f"{path}: scene frame outside cohort")
        require(row.get("scene_fact_manifest_sha256") == cohort[event_id]["scene_fact_manifest_sha256"], f"{path}: scene fact binding mismatch")
        key = (event_id, arm, frame)
        require(key not in result, f"{path}: duplicate scene row")
        area = require_finite_nonnegative(row.get("frame_area_px"), f"{path}: frame area")
        truth = require_finite_nonnegative(row.get("truth_area_px"), f"{path}: truth area")
        intersection = require_finite_nonnegative(row.get("intersection_area_px"), f"{path}: intersection")
        predicted = require_finite_nonnegative(row.get("predicted_area_px"), f"{path}: predicted area")
        require(area > 0 and truth <= area and intersection <= truth and intersection <= predicted and predicted <= area, f"{path}: impossible areas")
        for field in ("truth_component_count", "matched_truth_component_count", "predicted_component_count", "unmatched_predicted_component_count"):
            number = row.get(field)
            require(isinstance(number, int) and number >= 0, f"{path}: invalid {field}")
        require(row["matched_truth_component_count"] <= row["truth_component_count"], f"{path}: matched component count")
        require(row["unmatched_predicted_component_count"] <= row["predicted_component_count"], f"{path}: unmatched component count")
        previous_iou = row.get("previous_prediction_iou")
        require(previous_iou is None or (isinstance(previous_iou, (int, float)) and not isinstance(previous_iou, bool) and 0 <= float(previous_iou) <= 1), f"{path}: previous_prediction_iou")
        result[key] = row
    expected = {(event_id, arm, frame) for event_id, item in cohort.items() for arm in REPRESENTATION_ARMS for frame in item["frame_indices"]}
    require(set(result) == expected, f"{path}: scene rows do not cover exactly the frozen cohort")
    return result


def _representation_metrics(rows: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for arm in REPRESENTATION_ARMS:
        arm_rows = [row for (_, current_arm, _), row in rows.items() if current_arm == arm]
        truth_area = sum(float(row["truth_area_px"]) for row in arm_rows)
        intersection = sum(float(row["intersection_area_px"]) for row in arm_rows)
        predicted = sum(float(row["predicted_area_px"]) for row in arm_rows)
        frame_area = sum(float(row["frame_area_px"]) for row in arm_rows)
        matched = sum(int(row["matched_truth_component_count"]) for row in arm_rows)
        truth_components = sum(int(row["truth_component_count"]) for row in arm_rows)
        predicted_components = sum(int(row["predicted_component_count"]) for row in arm_rows)
        unmatched = sum(int(row["unmatched_predicted_component_count"]) for row in arm_rows)
        ious = [float(row["previous_prediction_iou"]) for row in arm_rows if row["previous_prediction_iou"] is not None]
        summaries[arm] = {
            "frame_count": len(arm_rows),
            "coverage": None if truth_area == 0 else intersection / truth_area,
            "false_area_rate": (predicted - intersection) / frame_area,
            "component_recall": None if truth_components == 0 else matched / truth_components,
            "false_components_per_frame": unmatched / len(arm_rows),
            "fragmentation_ratio": None if matched == 0 else predicted_components / matched,
            "temporal_stability_median_iou": None if not ious else median(ious),
        }
    return summaries


def _load_trace_manifest(path: Path, cohort_sha256: str, contract: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(path)
    require(manifest.get("schema_version") == TRACE_MANIFEST_SCHEMA, "trace manifest: schema mismatch")
    require(manifest.get("protocol_id") == PROTOCOL_ID, "trace manifest: protocol mismatch")
    require(manifest.get("cohort_sha256") == cohort_sha256, "trace manifest: cohort hash mismatch")
    require(manifest.get("action_reviews_passed_before_trace_access") is True and manifest.get("full_event_facts_frozen_before_trace_access") is True, "trace manifest: action/full event review was not prior to trace access")
    shared = manifest.get("shared_execution")
    require(isinstance(shared, dict) and shared == contract["shared_execution"], "trace manifest: shared execution differs from contract")
    arms = manifest.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(ARMS), "trace manifest: arm coverage mismatch")
    require(arms["current_yolo"].get("input_kind") == "CURRENT_YOLO_OUTPUT", "trace manifest: YOLO input identity")
    require(arms["truth_box"].get("input_kind") == "SCENE_FACT_TRUTH_BOX", "trace manifest: truth box identity")
    require(arms["truth_mask"].get("input_kind") == "SCENE_FACT_TRUTH_MASK", "trace manifest: truth mask identity")
    require(arms["synthetic_oracle"].get("input_kind") == "EVENT_FACT_SYNTHETIC_DIRECTIVE", "trace manifest: synthetic oracle identity")
    return manifest


def _load_traces(path: Path, cohort: dict[str, dict[str, Any]]) -> dict[tuple[str, str, int], bool]:
    rows = read_jsonl(path)
    result: dict[tuple[str, str, int], bool] = {}
    for row in rows:
        require(row.get("schema_version") == TRACE_SCHEMA, f"{path}: trace schema mismatch")
        require(row.get("protocol_id") == PROTOCOL_ID, f"{path}: trace protocol mismatch")
        event_id, arm, frame, alert = row.get("parent_event_id"), row.get("arm"), row.get("frame_index"), row.get("feedback_alert")
        require(event_id in cohort and arm in ARMS and isinstance(frame, int) and isinstance(alert, bool), f"{path}: invalid trace row")
        require(frame in cohort[event_id]["frame_indices"], f"{path}: trace frame outside frozen event")
        key = (event_id, arm, frame)
        require(key not in result, f"{path}: duplicate trace row")
        result[key] = alert
    expected = {(event_id, arm, frame) for event_id, item in cohort.items() for arm in ARMS for frame in item["frame_indices"]}
    require(set(result) == expected, f"{path}: trace rows do not cover exactly the frozen event frames")
    return result


def _score_events(traces: dict[tuple[str, str, int], bool], facts: dict[str, dict[str, Any]], cohort: dict[str, dict[str, Any]], max_delay: int) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        by_bucket: dict[str, dict[str, int]] = {bucket: Counter() for bucket in sorted(BUCKETS)}
        per_event: dict[str, dict[str, Any]] = {}
        for event_id, item in cohort.items():
            bucket = item["bucket"]
            frames = item["frame_indices"]
            event_facts = facts[event_id]
            alertable_interval = event_facts["alertable_interval_frames"]
            passed_interval = event_facts["passed_interval_frames"]
            alertable = [] if alertable_interval is None else list(range(alertable_interval[0], alertable_interval[1] + 1))
            cleared = [] if passed_interval is None else list(range(passed_interval[0], passed_interval[1] + 1))
            fired = [frame for frame in frames if traces[(event_id, arm, frame)]]
            outcome: dict[str, Any] = {"bucket": bucket, "first_alert_frame": fired[0] if fired else None}
            if bucket in POSITIVE_BUCKETS:
                hit_frames = sorted(set(fired) & set(alertable))
                hit = bool(hit_frames)
                first_reference = min(alertable) if alertable else None
                delay = None if not hit or first_reference is None else hit_frames[0] - first_reference
                cleared_ok = bool(cleared) and not (set(fired) & set(cleared))
                premature_alert = bool(first_reference is not None and any(frame < first_reference for frame in fired))
                outcome.update({"hit": hit, "critical_miss": not hit, "premature_alert": premature_alert, "cleared": cleared_ok, "response_delay_frames": delay})
                by_bucket[bucket]["positive_events"] += 1
                by_bucket[bucket]["hits"] += int(hit)
                by_bucket[bucket]["critical_misses"] += int(not hit)
                by_bucket[bucket]["premature_alerts"] += int(premature_alert)
                by_bucket[bucket]["cleared"] += int(cleared_ok)
                by_bucket[bucket]["timely_hits"] += int(delay is not None and delay <= max_delay)
            else:
                false_alert = bool(fired)
                outcome.update({"false_alert": false_alert})
                by_bucket[bucket]["negative_events"] += 1
                by_bucket[bucket]["false_alerts"] += int(false_alert)
            per_event[event_id] = outcome
        aggregate = Counter()
        delays: list[int] = []
        for bucket, values in by_bucket.items():
            aggregate.update(values)
        for outcome in per_event.values():
            if isinstance(outcome.get("response_delay_frames"), int):
                delays.append(outcome["response_delay_frames"])
        by_arm[arm] = {
            "aggregate": dict(aggregate),
            "by_bucket": {bucket: dict(values) for bucket, values in by_bucket.items()},
            "median_response_delay_frames": None if not delays else median(delays),
            "per_event": per_event,
        }
    return by_arm


def _no_worse(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    candidate_values, baseline_values = candidate["aggregate"], baseline["aggregate"]
    problems: list[str] = []
    if candidate_values.get("hits", 0) < baseline_values.get("hits", 0):
        problems.append("lower_positive_hits")
    if candidate_values.get("critical_misses", 0) > baseline_values.get("critical_misses", 0):
        problems.append("more_critical_misses")
    if candidate_values.get("false_alerts", 0) > baseline_values.get("false_alerts", 0):
        problems.append("more_false_alert_events")
    if candidate_values.get("premature_alerts", 0) > baseline_values.get("premature_alerts", 0):
        problems.append("more_premature_alert_events")
    if candidate_values.get("cleared", 0) < baseline_values.get("cleared", 0):
        problems.append("fewer_cleared_events")
    for event_id, baseline_event in baseline["per_event"].items():
        candidate_event = candidate["per_event"][event_id]
        baseline_delay, candidate_delay = baseline_event.get("response_delay_frames"), candidate_event.get("response_delay_frames")
        if baseline_delay is not None and candidate_delay is not None and candidate_delay > baseline_delay:
            problems.append(f"later_common_hit:{event_id}")
    for bucket in BUCKETS:
        candidate_bucket, baseline_bucket = candidate["by_bucket"][bucket], baseline["by_bucket"][bucket]
        if bucket in POSITIVE_BUCKETS:
            if candidate_bucket.get("hits", 0) < baseline_bucket.get("hits", 0) or candidate_bucket.get("critical_misses", 0) > baseline_bucket.get("critical_misses", 0) or candidate_bucket.get("premature_alerts", 0) > baseline_bucket.get("premature_alerts", 0) or candidate_bucket.get("cleared", 0) < baseline_bucket.get("cleared", 0):
                problems.append(f"bucket_regression:{bucket}")
        elif candidate_bucket.get("false_alerts", 0) > baseline_bucket.get("false_alerts", 0):
            problems.append(f"bucket_regression:{bucket}")
    return not problems, sorted(set(problems))


def _oracle_ladder(events: dict[str, Any], cohort: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for candidate_arm, baseline_arm in (("truth_box", "current_yolo"), ("truth_mask", "truth_box"), ("synthetic_oracle", "truth_mask")):
        passed, problems = _no_worse(events[candidate_arm], events[baseline_arm])
        checks.append({"candidate": candidate_arm, "baseline": baseline_arm, "passed": passed, "problems": problems})
    synthetic = events["synthetic_oracle"]["aggregate"]
    positive_count = sum(1 for item in cohort.values() if item["bucket"] in POSITIVE_BUCKETS)
    negative_count = sum(1 for item in cohort.values() if item["bucket"] in NEGATIVE_BUCKETS)
    integrity_problems: list[str] = []
    if synthetic.get("hits", 0) != positive_count:
        integrity_problems.append("synthetic_oracle_missed_positive")
    if synthetic.get("critical_misses", 0) != 0:
        integrity_problems.append("synthetic_oracle_critical_miss")
    if synthetic.get("false_alerts", 0) != 0 or negative_count == 0:
        integrity_problems.append("synthetic_oracle_false_alert_or_missing_negative")
    if synthetic.get("premature_alerts", 0) != 0:
        integrity_problems.append("synthetic_oracle_premature_alert")
    if synthetic.get("cleared", 0) != positive_count:
        integrity_problems.append("synthetic_oracle_clearance_failure")
    return {
        "checks": checks,
        "synthetic_integrity_passed": not integrity_problems,
        "synthetic_integrity_problems": integrity_problems,
        "passed": all(item["passed"] for item in checks) and not integrity_problems,
    }


def run_audit(
    contract: dict[str, Any], registry: dict[str, Any], cohort: dict[str, Any], review_map: dict[str, Any], review_a: dict[str, Any], review_b: dict[str, Any], full_event_facts: dict[str, Any], scene_rows: list[dict[str, Any]], trace_manifest: dict[str, Any], trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_contract(contract)
    excluded = _validate_exclusions(registry)
    cohort_items, cohort_sha256 = _load_cohort(cohort, excluded, contract)
    require(isinstance(review_map, dict) and all(isinstance(key, str) and isinstance(value, dict) for key, value in review_map.items()), "review map: invalid")
    indexed_a = _review_index(review_a, review_map, cohort_items, "review A")
    indexed_b = _review_index(review_b, review_map, cohort_items, "review B")
    agreement, _ = _agreement_and_event_facts(indexed_a, indexed_b, cohort_items, contract)
    report: dict[str, Any] = {
        "schema_version": "blindassist.eval_validity_r0.report.v1",
        "protocol_id": PROTOCOL_ID,
        "cohort_sha256": cohort_sha256,
        "input_claim": "Evaluator-validity evidence only. It is not model training, selection, promotion, safety or default-App evidence.",
        "actionability_consistency": agreement,
        "scene_representation": None,
        "event_quality": None,
        "oracle_monotonicity": None,
        "status": None,
    }
    if not agreement["passed"]:
        report["status"] = "STOP_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED"
        report["next_action"] = "Do not read model or oracle traces. Revise the event-fact construct on a new, independently frozen cohort before any candidate comparison."
        return report
    # Parse after the actionability gate.  This ordering is deliberate and testable.
    event_facts = _load_full_event_facts(full_event_facts, cohort_items, cohort_sha256, agreement)
    full_event_facts_sha256 = sha256_json(full_event_facts)
    report["full_event_facts_sha256"] = full_event_facts_sha256
    scene_by_key = _load_scene_rows_from_rows(scene_rows, cohort_items)
    _validate_trace_manifest_value(trace_manifest, cohort_sha256, full_event_facts_sha256, contract)
    traces = _load_traces_from_rows(trace_rows, cohort_items)
    events = _score_events(traces, event_facts, cohort_items, int(contract["gates"]["maximum_response_delay_frames"]))
    ladder = _oracle_ladder(events, cohort_items)
    report["scene_representation"] = _representation_metrics(scene_by_key)
    report["event_quality"] = events
    report["oracle_monotonicity"] = ladder
    if not ladder["synthetic_integrity_passed"]:
        report["status"] = "STOP_EVALUATOR_INTEGRITY_NOT_ESTABLISHED"
        report["next_action"] = "Correct the evaluator or feedback-accounting path. Do not interpret any real arm comparison and do not train or retune a model."
    elif not ladder["passed"]:
        report["status"] = "STOP_ORACLE_MONOTONICITY_NOT_ESTABLISHED"
        report["next_action"] = "Correct the fact-to-evidence adapter or event scoring construct; do not train or retune a candidate model to repair this failure."
    else:
        report["status"] = "VALID_EVALUATION_CONSTRUCT_AND_ORACLE_LADDER"
        report["next_action"] = "Only after this validity result may a future candidate be assessed with the same two-layer report; this audit does not authorize training or default-App changes."
    return report


def _load_frozen_review_lineage(
    p0: dict[str, Any], p1: dict[str, Any], cohort: dict[str, dict[str, Any]], cohort_sha256: str, contract: dict[str, Any],
) -> dict[str, Any]:
    require(p0.get("schema_version") == P0_ANCHOR_AGREEMENT_SCHEMA and p0.get("protocol_id") == PROTOCOL_ID, "P0 receipt: schema/protocol mismatch")
    require(p0.get("status") == "P0_ANCHOR_CONSISTENCY_PASSED" and p0.get("candidate_outputs_opened") is False, "P0 receipt: not cleanly passed")
    screening_sha = next(iter(cohort.values())).get("screening_cohort_sha256")
    require(isinstance(screening_sha, str) and p0.get("screening_cohort_sha256") == screening_sha, "P0 receipt: screening cohort binding mismatch")
    agreement = p0.get("anchor_agreement")
    require(isinstance(agreement, dict) and agreement.get("passed") is True, "P0 receipt: anchor agreement not passed")
    metrics = agreement.get("metrics")
    require(isinstance(metrics, dict), "P0 receipt: missing metrics")
    gates = contract["gates"]
    require(
        metrics.get("reminder_now_exact_agreement") >= gates["minimum_actionability_exact_agreement"]
        and metrics.get("cleared_exact_agreement") >= gates["minimum_clearance_exact_agreement"]
        and metrics.get("knownness_exact_agreement") >= gates["minimum_knownness_exact_agreement"]
        and metrics.get("parent_event_sequence_exact_agreement") >= gates["minimum_sequence_exact_agreement"]
        and metrics.get("unknown_anchor_burden") <= gates["maximum_unknown_burden"],
        "P0 receipt: actionability gate failed",
    )
    p0_sha = sha256_json(p0)
    require(p1.get("schema_version") == P1_ACTION_FACTS_SCHEMA and p1.get("protocol_id") == PROTOCOL_ID, "P1 action facts: schema/protocol mismatch")
    require(p1.get("status") == "P1_ACTION_FACTS_FROZEN_AFTER_P0_CONSISTENCY" and p1.get("candidate_outputs_opened") is False, "P1 action facts: not cleanly passed")
    require(p1.get("screening_cohort_sha256") == screening_sha and p1.get("p0_anchor_agreement_sha256") == p0_sha, "P1 action facts: P0/screening binding mismatch")
    p1_evidence = p1.get("independent_full_review_evidence")
    require(isinstance(p1_evidence, dict) and p1_evidence.get("agreement_passed") is True and p1_evidence.get("unknown_or_disagreement_event_count") == 0, "P1 action facts: unresolved")
    rows = p1.get("items")
    require(isinstance(rows, list) and len(rows) == len(cohort), "P1 action facts: coverage mismatch")
    for row in rows:
        require(isinstance(row, dict) and row.get("screening_event_id") in cohort and row.get("resolved") is True and row.get("p0_anchor_compatible") is True, "P1 action facts: invalid event")
    require(len({row["screening_event_id"] for row in rows}) == len(cohort), "P1 action facts: duplicate event")
    return agreement


def run_frozen_audit(
    contract: dict[str, Any], registry: dict[str, Any], cohort: dict[str, Any], p0_agreement: dict[str, Any], p1_action_facts: dict[str, Any],
    full_event_facts: dict[str, Any], scene_rows: list[dict[str, Any]], trace_manifest: dict[str, Any], trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the real P0/P1 frozen route without reopening reviewer submissions."""
    _validate_contract(contract)
    excluded = _validate_exclusions(registry)
    cohort_items, cohort_sha256 = _load_cohort(cohort, excluded, contract)
    agreement = _load_frozen_review_lineage(p0_agreement, p1_action_facts, cohort_items, cohort_sha256, contract)
    require(full_event_facts.get("p0_anchor_agreement_receipt_sha256") == sha256_json(p0_agreement), "full event facts: P0 receipt binding mismatch")
    require(full_event_facts.get("p1_action_facts_sha256") == sha256_json(p1_action_facts), "full event facts: P1 action-facts binding mismatch")
    event_facts = _load_full_event_facts(full_event_facts, cohort_items, cohort_sha256, agreement)
    full_event_facts_sha256 = sha256_json(full_event_facts)
    scene_by_key = _load_scene_rows_from_rows(scene_rows, cohort_items)
    _validate_trace_manifest_value(trace_manifest, cohort_sha256, full_event_facts_sha256, contract)
    traces = _load_traces_from_rows(trace_rows, cohort_items)
    representation = _representation_metrics(scene_by_key)
    events = _score_events(traces, event_facts, cohort_items, int(contract["gates"]["maximum_response_delay_frames"]))
    monotonicity = _oracle_ladder(events, cohort_items)
    report = {
        "schema_version": "blindassist.eval_validity_r0.report.v1", "protocol_id": PROTOCOL_ID,
        "cohort_sha256": cohort_sha256,
        "input_claim": "Evaluator-validity evidence only. It is not model training, selection, promotion, safety or default-App evidence.",
        "actionability_consistency": agreement,
        "full_event_facts_sha256": full_event_facts_sha256,
        "scene_representation": representation, "event_quality": events, "oracle_monotonicity": monotonicity,
        "status": "VALID_EVALUATION_CONSTRUCT_AND_ORACLE_LADDER" if monotonicity["synthetic_integrity_passed"] and monotonicity["passed"] else "STOP_ORACLE_MONOTONICITY_NOT_ESTABLISHED",
    }
    report["next_action"] = (
        "The frozen evaluator can rank a separately authorized research candidate; no default-App or safety claim follows."
        if report["status"] == "VALID_EVALUATION_CONSTRUCT_AND_ORACLE_LADDER" else "Do not interpret model arms. Repair the evaluator/adapter chain on a new independent audit route; do not tune a candidate."
    )
    return report


def _load_scene_rows_from_rows(rows: list[dict[str, Any]], cohort: dict[str, dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    # Keep the pure function testable while using the exact same validation as the CLI loader.
    temporary = Path("__not_used__")
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        require(row.get("schema_version") == SCENE_FRAME_SCHEMA, "scene rows: schema mismatch")
        require(row.get("protocol_id") == PROTOCOL_ID, "scene rows: protocol mismatch")
        event_id, arm, frame = row.get("parent_event_id"), row.get("arm"), row.get("frame_index")
        require(event_id in cohort and arm in REPRESENTATION_ARMS and isinstance(frame, int), "scene rows: invalid identity")
        require(frame in cohort[event_id]["frame_indices"], "scene rows: frame outside cohort")
        require(row.get("scene_fact_manifest_sha256") == cohort[event_id]["scene_fact_manifest_sha256"], "scene rows: scene fact binding mismatch")
        key = (event_id, arm, frame)
        require(key not in result, "scene rows: duplicate row")
        area = require_finite_nonnegative(row.get("frame_area_px"), "scene rows: frame area")
        truth = require_finite_nonnegative(row.get("truth_area_px"), "scene rows: truth area")
        intersection = require_finite_nonnegative(row.get("intersection_area_px"), "scene rows: intersection")
        predicted = require_finite_nonnegative(row.get("predicted_area_px"), "scene rows: predicted area")
        require(area > 0 and truth <= area and intersection <= truth and intersection <= predicted and predicted <= area, "scene rows: impossible areas")
        for field in ("truth_component_count", "matched_truth_component_count", "predicted_component_count", "unmatched_predicted_component_count"):
            number = row.get(field)
            require(isinstance(number, int) and number >= 0, f"scene rows: invalid {field}")
        require(row["matched_truth_component_count"] <= row["truth_component_count"], "scene rows: matched components")
        require(row["unmatched_predicted_component_count"] <= row["predicted_component_count"], "scene rows: unmatched components")
        previous_iou = row.get("previous_prediction_iou")
        require(previous_iou is None or (isinstance(previous_iou, (int, float)) and not isinstance(previous_iou, bool) and 0 <= float(previous_iou) <= 1), "scene rows: previous IoU")
        result[key] = row
    expected = {(event_id, arm, frame) for event_id, item in cohort.items() for arm in REPRESENTATION_ARMS for frame in item["frame_indices"]}
    require(set(result) == expected, "scene rows: coverage mismatch")
    return result


def _validate_trace_manifest_value(manifest: dict[str, Any], cohort_sha256: str, full_event_facts_sha256: str, contract: dict[str, Any]) -> None:
    require(manifest.get("schema_version") == TRACE_MANIFEST_SCHEMA, "trace manifest: schema mismatch")
    require(manifest.get("protocol_id") == PROTOCOL_ID, "trace manifest: protocol mismatch")
    require(manifest.get("cohort_sha256") == cohort_sha256, "trace manifest: cohort hash mismatch")
    require(manifest.get("full_event_facts_sha256") == full_event_facts_sha256, "trace manifest: full event facts binding mismatch")
    require(manifest.get("action_reviews_passed_before_trace_access") is True and manifest.get("full_event_facts_frozen_before_trace_access") is True, "trace manifest: trace opened early")
    require(manifest.get("shared_execution") == contract["shared_execution"], "trace manifest: shared execution mismatch")
    arms = manifest.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(ARMS), "trace manifest: arm coverage")
    for arm, expected_input in {
        "current_yolo": "CURRENT_YOLO_OUTPUT", "truth_box": "SCENE_FACT_TRUTH_BOX", "truth_mask": "SCENE_FACT_TRUTH_MASK", "synthetic_oracle": "EVENT_FACT_SYNTHETIC_DIRECTIVE",
    }.items():
        require(isinstance(arms[arm], dict) and arms[arm].get("input_kind") == expected_input, f"trace manifest: {arm} identity")


def _load_traces_from_rows(rows: list[dict[str, Any]], cohort: dict[str, dict[str, Any]]) -> dict[tuple[str, str, int], bool]:
    result: dict[tuple[str, str, int], bool] = {}
    for row in rows:
        require(row.get("schema_version") == TRACE_SCHEMA, "trace rows: schema mismatch")
        require(row.get("protocol_id") == PROTOCOL_ID, "trace rows: protocol mismatch")
        event_id, arm, frame, alert = row.get("parent_event_id"), row.get("arm"), row.get("frame_index"), row.get("feedback_alert")
        require(event_id in cohort and arm in ARMS and isinstance(frame, int) and isinstance(alert, bool), "trace rows: invalid identity")
        require(frame in cohort[event_id]["frame_indices"], "trace rows: frame outside event")
        key = (event_id, arm, frame)
        require(key not in result, "trace rows: duplicate row")
        result[key] = alert
    expected = {(event_id, arm, frame) for event_id, item in cohort.items() for arm in ARMS for frame in item["frame_indices"]}
    require(set(result) == expected, "trace rows: coverage mismatch")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--review-map", type=Path)
    parser.add_argument("--review-a", type=Path)
    parser.add_argument("--review-b", type=Path)
    parser.add_argument("--p0-agreement", type=Path)
    parser.add_argument("--p1-action-facts", type=Path)
    parser.add_argument("--full-event-facts", type=Path, required=True)
    parser.add_argument("--scene-frames", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--feedback-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    contract, registry, cohort = read_json(args.contract), read_json(args.exclusion_registry), read_json(args.cohort)
    frozen_inputs = (args.p0_agreement, args.p1_action_facts)
    legacy_inputs = (args.review_map, args.review_a, args.review_b)
    if any(value is not None for value in frozen_inputs):
        if not all(value is not None for value in frozen_inputs) or any(value is not None for value in legacy_inputs):
            raise SystemExit("frozen P0/P1 route requires both --p0-agreement and --p1-action-facts, and no legacy review inputs")
        result = run_frozen_audit(
            contract, registry, cohort, read_json(args.p0_agreement), read_json(args.p1_action_facts), read_json(args.full_event_facts),
            read_jsonl(args.scene_frames), read_json(args.trace_manifest), read_jsonl(args.feedback_traces),
        )
        result["input_sha256"] = {
            "contract": sha256_file(args.contract), "exclusion_registry": sha256_file(args.exclusion_registry), "cohort": sha256_file(args.cohort),
            "p0_agreement": sha256_file(args.p0_agreement), "p1_action_facts": sha256_file(args.p1_action_facts), "full_event_facts": sha256_file(args.full_event_facts),
            "scene_frames": sha256_file(args.scene_frames), "trace_manifest": sha256_file(args.trace_manifest), "feedback_traces": sha256_file(args.feedback_traces),
        }
    else:
        if not all(value is not None for value in legacy_inputs):
            raise SystemExit("legacy route requires --review-map, --review-a, and --review-b")
        cohort_sha256 = sha256_json(cohort)
        review_a = _load_action_review(args.review_a, "ACTION_REVIEW_A", cohort_sha256)
        review_b = _load_action_review(args.review_b, "ACTION_REVIEW_B", cohort_sha256)
        result = run_audit(
            contract, registry, cohort, read_json(args.review_map), review_a, review_b, read_json(args.full_event_facts),
            read_jsonl(args.scene_frames), read_json(args.trace_manifest), read_jsonl(args.feedback_traces),
        )
        result["input_sha256"] = {
            "contract": sha256_file(args.contract), "exclusion_registry": sha256_file(args.exclusion_registry), "cohort": sha256_file(args.cohort),
            "review_map": sha256_file(args.review_map), "review_a": sha256_file(args.review_a), "review_b": sha256_file(args.review_b), "full_event_facts": sha256_file(args.full_event_facts),
            "scene_frames": sha256_file(args.scene_frames), "trace_manifest": sha256_file(args.trace_manifest), "feedback_traces": sha256_file(args.feedback_traces),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
