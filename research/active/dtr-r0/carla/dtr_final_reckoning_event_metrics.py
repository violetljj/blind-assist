"""Candidate shared R1 event evaluator, pending execution-contract freeze.

Input contact times are absolute evaluator-only times. Truth UNKNOWN splits
truth events; prediction UNKNOWN splits observed alert segments. Continuous
ACTIVE across truth UNKNOWN remains one alert, so it cannot match twice.
Known truth events remain recall opportunities despite prediction abstention.
No source loading, model inference, or outcome-dependent configuration lives here.
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

DEFAULT_BOOTSTRAP_REPLICATES = 10000
DEFAULT_BOOTSTRAP_SEED = 517999
EPSILON = 1e-8


def contract() -> dict[str, Any]:
    return {
        "status": "IMPLEMENTATION_CANDIDATE_PENDING_EXECUTION_CONTRACT_FREEZE",
        "intervals": "HALF_OPEN_TIME_TO_TIME_PLUS_SAMPLE_SECONDS",
        "truth_unknown": "SPLITS_TRUTH_EVENTS_EXCLUDED_FROM_FRAME_DENOMINATORS",
        "prediction_unknown": "SPLITS_ALERTS_EXCLUDED_FROM_FRAME_METRICS_PRESERVES_TRUTH_EVENT_DENOMINATOR",
        "continuous_alert_across_truth_unknown": "ONE_ALERT_IDENTITY_NOT_MULTIPLE_MATCHES",
        "matching": "DETERMINISTIC_MAXIMUM_CARDINALITY_EARLIEST_ALERT_ONSET_ORDER_WITH_JOINT_KNOWN_POSITIVE_OVERLAP",
        "contact_time": "EARLIEST_EXPLICIT_ABSOLUTE_TRUTH_CONTACT_TIME_IN_EVENT_ELSE_FIRST_CURRENT_CONTACT_IN_SAME_EVENT_ELSE_REJECT",
        "onset_eligibility": "ALERT_ONSET_NO_LATER_THAN_ASSOCIATED_CONTACT_TIME",
        "fully_unevaluable_alerts": "EXCLUDED_AND_COUNTED_SEPARATELY",
        "false_rate_primary_denominator": "JOINT_KNOWN_SECONDS",
        "false_rate_additional_denominator": "ALL_TRUTH_KNOWN_SECONDS",
        "empty_metric_denominator": "NONE_NOT_MEASURED_ZERO",
        "fragmentation": "OTHER_OBSERVED_ALERT_ONSETS_INSIDE_A_MATCHED_TRUTH_EVENT",
        "fragmented_event_rate_denominator": "ALL_KNOWN_TRUTH_EVENTS",
        "clear_delay": "FIRST_OBSERVED_INACTIVE_TIME_MINUS_TRUTH_EVENT_END_ZERO_IF_IMMEDIATELY_INACTIVE",
        "clear_censor": "FIRST_TRUTH_OR_PREDICTION_UNKNOWN_NEXT_TRUE_EVENT_OR_EPISODE_END",
        "bootstrap": {"cluster": "EPISODE", "paired": True,
                      "replicates": DEFAULT_BOOTSTRAP_REPLICATES, "seed": DEFAULT_BOOTSTRAP_SEED,
                      "interval": "PERCENTILE_2P5_97P5", "delta": "CANDIDATE_MINUS_REFERENCE_EVENT_F1",
                      "zero_total_event_replicate": "DISCARD_AS_UNEVALUABLE_REPORT_COUNT_NONE_INTERVAL_IF_ALL"},
    }


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def classification(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def intervals(rows: Sequence[Mapping[str, Any]], field: str, dt: float) -> list[dict[str, Any]]:
    output = []
    start: int | None = None
    for index in range(len(rows) + 1):
        active = index < len(rows) and rows[index][field] is True
        if active and start is None:
            start = index
        if not active and start is not None:
            output.append({"id": len(output), "start_index": start, "end_index_exclusive": index,
                           "start_s": float(rows[start]["time_s"]),
                           "end_s": float(rows[index - 1]["time_s"]) + dt})
            start = None
    return output


def evaluate_episode(rows: Sequence[Mapping[str, Any]], sample_seconds: float = 0.1) -> dict[str, Any]:
    dt = float(sample_seconds)
    require(math.isfinite(dt) and dt > 0, "sample_seconds")
    require(bool(rows), "empty_episode")
    for index, row in enumerate(rows):
        now = float(row["time_s"])
        require(math.isfinite(now), "nonfinite_time")
        if index:
            require(abs(now - float(rows[index - 1]["time_s"]) - dt) <= EPSILON, "nonuniform_sampling")
        for field in ("truth_risk", "predicted_active", "current_contact"):
            value = row.get(field)
            require(value is None or type(value) is bool, f"invalid_tristate:{field}")
        require("truth_risk" in row and "predicted_active" in row, "missing_state")
        if row.get("truth_contact_time_s") is not None:
            require(math.isfinite(float(row["truth_contact_time_s"])), "nonfinite_contact_time")
    truth = intervals(rows, "truth_risk", dt)
    alerts = intervals(rows, "predicted_active", dt)
    for event in truth:
        portion = rows[event["start_index"]:event["end_index_exclusive"]]
        times = [float(row["truth_contact_time_s"]) for row in portion if row.get("truth_contact_time_s") is not None]
        if not times:
            times = [float(row["time_s"]) for row in portion if row.get("current_contact") is True]
        require(bool(times), f"missing_truth_contact_time:event_{event['id']}")
        event["contact_time_s"] = min(times)
        require(event["contact_time_s"] + EPSILON >= event["start_s"], "contact_precedes_truth_event")
    for alert in alerts:
        alert["evaluable"] = any(rows[i]["truth_risk"] is not None
                                 for i in range(alert["start_index"], alert["end_index_exclusive"]))

    # Augmenting paths preserve maximum cardinality. Earlier observed onsets
    # enter first; candidates are explored in chronological truth-event order.
    eligible: dict[int, list[int]] = {}
    for alert in alerts:
        eligible[alert["id"]] = []
        if not alert["evaluable"]:
            continue
        for event in truth:
            if alert["start_s"] > event["contact_time_s"] + EPSILON:
                continue
            begin = max(alert["start_index"], event["start_index"])
            end = min(alert["end_index_exclusive"], event["end_index_exclusive"])
            if begin < end:
                eligible[alert["id"]].append(event["id"])
    assigned: dict[int, int] = {}

    def augment(alert_id: int, seen: set[int]) -> bool:
        for truth_id in eligible[alert_id]:
            if truth_id in seen:
                continue
            seen.add(truth_id)
            if truth_id not in assigned or augment(assigned[truth_id], seen):
                assigned[truth_id] = alert_id
                return True
        return False

    for alert in alerts:
        augment(alert["id"], set())
    matches = [{"truth_event_id": event_id, "alert_event_id": alert_id,
                "lead_s": truth[event_id]["contact_time_s"] - alerts[alert_id]["start_s"]}
               for event_id, alert_id in sorted(assigned.items())]
    matched_alerts = set(assigned.values())
    false_ids = [a["id"] for a in alerts if a["evaluable"] and a["id"] not in matched_alerts]
    unmatched_truth = [e["id"] for e in truth if e["id"] not in assigned]
    fragmented = []
    extra = 0
    for event_id, matched_alert in sorted(assigned.items()):
        event = truth[event_id]
        others = [a["id"] for a in alerts if a["id"] != matched_alert and
                  event["start_index"] <= a["start_index"] < event["end_index_exclusive"]]
        if others:
            fragmented.append({"truth_event_id": event_id, "extra_alert_ids": others})
            extra += len(others)
    clear_records = []
    for event in truth:
        record = {"truth_event_id": event["id"], "truth_end_s": event["end_s"], "delay_s": None,
                  "right_censored": True, "censor_reason": "EPISODE_END",
                  "observation_end_s": float(rows[-1]["time_s"]) + dt}
        for index in range(event["end_index_exclusive"], len(rows)):
            row = rows[index]
            record["observation_end_s"] = float(row["time_s"])
            if row["truth_risk"] is None or row["predicted_active"] is None:
                record["censor_reason"] = "UNKNOWN"
                break
            if row["truth_risk"] is True:
                record["censor_reason"] = "NEXT_TRUE_EVENT"
                break
            if row["predicted_active"] is False:
                record.update(delay_s=max(0.0, float(row["time_s"]) - event["end_s"]),
                              right_censored=False, censor_reason=None)
                break
        else:
            record["observation_end_s"] = float(rows[-1]["time_s"]) + dt
        clear_records.append(record)
    frame = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    truth_known = joint_known = unknown_positive = unknown_negative = 0
    for row in rows:
        target, prediction = row["truth_risk"], row["predicted_active"]
        if target is None:
            continue
        truth_known += 1
        if prediction is None:
            unknown_positive += int(target)
            unknown_negative += int(not target)
            continue
        joint_known += 1
        frame["tp" if target and prediction else "fp" if prediction else "fn" if target else "tn"] += 1
    result = {"event_tp": len(matches), "event_fp": len(false_ids), "event_fn": len(unmatched_truth),
              "truth_event_count": len(truth), "alert_event_count": len(alerts),
              "evaluable_alert_event_count": sum(a["evaluable"] for a in alerts),
              "fully_unevaluable_alert_count": sum(not a["evaluable"] for a in alerts),
              "false_segments": len(false_ids), "fragmentation_extra_onsets": extra,
              "fragmented_event_count": len(fragmented),
              "frame_count": len(rows), "truth_known_frames": truth_known, "joint_known_frames": joint_known,
              "prediction_unknown_truth_positive_frames": unknown_positive,
              "prediction_unknown_truth_negative_frames": unknown_negative,
              "joint_known_seconds": joint_known * dt, "truth_known_seconds": truth_known * dt,
              "frame": frame,
              "lead_samples_s": [m["lead_s"] for m in matches],
              "clear_delay_samples_s": [r["delay_s"] for r in clear_records if not r["right_censored"]],
              "clear_right_censored_count": sum(r["right_censored"] for r in clear_records),
              "truth_events": truth, "alert_events": alerts, "matches": matches,
              "false_alert_ids": false_ids, "unmatched_truth_ids": unmatched_truth,
              "fragmented_events": fragmented, "clear_records": clear_records}
    return _derived(result)


def _derived(result: dict[str, Any]) -> dict[str, Any]:
    for key, value in classification(result["event_tp"], result["event_fp"], result["event_fn"]).items():
        result["event_" + key] = value
    result["frame"].update(classification(result["frame"]["tp"], result["frame"]["fp"], result["frame"]["fn"]))
    for name in ("precision", "recall", "f1"):
        result["frame_" + name] = result["frame"][name]
    result.update(
        false_segments_per_minute=(60 * result["false_segments"] / result["joint_known_seconds"] if result["joint_known_seconds"] else None),
        false_segments_per_truth_known_minute=(60 * result["false_segments"] / result["truth_known_seconds"] if result["truth_known_seconds"] else None),
        joint_known_coverage=(result["joint_known_frames"] / result["truth_known_frames"] if result["truth_known_frames"] else None),
        fragmented_event_rate=(result["fragmented_event_count"] / result["truth_event_count"] if result["truth_event_count"] else None),
        median_first_alert_lead_s=quantile(result["lead_samples_s"], 0.5),
        p10_first_alert_lead_s=quantile(result["lead_samples_s"], 0.1),
        median_clear_delay_s=quantile(result["clear_delay_samples_s"], 0.5),
    )
    return result


def aggregate(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(bool(episodes), "empty_aggregate")
    additive = ("event_tp", "event_fp", "event_fn", "truth_event_count", "alert_event_count",
                "evaluable_alert_event_count", "fully_unevaluable_alert_count", "false_segments",
                "fragmentation_extra_onsets", "fragmented_event_count", "frame_count", "truth_known_frames",
                "joint_known_frames", "prediction_unknown_truth_positive_frames", "prediction_unknown_truth_negative_frames",
                "joint_known_seconds", "truth_known_seconds", "clear_right_censored_count")
    value = {key: sum(episode[key] for episode in episodes) for key in additive}
    value["episode_count"] = len(episodes)
    value["frame"] = {key: sum(episode["frame"][key] for episode in episodes) for key in ("tp", "fp", "fn", "tn")}
    for key in ("lead_samples_s", "clear_delay_samples_s"):
        value[key] = [item for episode in episodes for item in episode[key]]
    return _derived(value)


def paired_episode_bootstrap(reference: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]],
                             replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
                             seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    require(bool(reference) and reference.keys() == candidate.keys(), "paired_episode_set")
    require(type(replicates) is int and replicates > 0, "bootstrap_replicates")
    keys = sorted(reference)
    for key in keys:
        for field in ("truth_event_count", "truth_known_frames", "frame_count"):
            require(reference[key][field] == candidate[key][field], f"paired_truth_denominator:{key}:{field}")
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        scores = []
        for arm in (reference, candidate):
            counts = [sum(arm[key][field] for key in selected) for field in ("event_tp", "event_fp", "event_fn")]
            scores.append(classification(*counts)["f1"])
        if scores[0] is not None and scores[1] is not None:
            samples.append(scores[1] - scores[0])
    low, high = quantile(samples, 0.025), quantile(samples, 0.975)
    reference_point = aggregate(list(reference.values()))["event_f1"]
    candidate_point = aggregate(list(candidate.values()))["event_f1"]
    point = candidate_point - reference_point if candidate_point is not None and reference_point is not None else None
    return {"cluster_unit": "episode", "episode_count": len(keys), "replicates": replicates, "seed": seed,
            "evaluable_replicates": len(samples), "discarded_unevaluable_replicates": replicates - len(samples),
            "delta_event_f1": point, "ci95": [low, high] if samples else None,
            "candidate_significant_win": low is not None and low > 0,
            "method": "PAIRED_EPISODE_CLUSTER_PERCENTILE_BOOTSTRAP"}
