"""Recompute and validate the candidate-blind R2-L1 eligibility artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from contract import ContractError, load_json, sha256_file
from metric_eligibility import (
    EXPOSURE_EXCLUSION_REASONS,
    MASK_SCHEMA,
    METRICS,
    RECEIPT_SCHEMA,
    json_bytes,
    materialize,
    sha256_bytes,
    validate_config,
)


VALIDATION_SCHEMA = (
    "blindassist_ustrf_metric_eligibility_validation_receipt_r2_l1_v1"
)
VALID_CLASSIFICATIONS = frozenset({"eligible", "ineligible", "not_event_grain"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _validate_contributions(
    contributions: Mapping[str, Any], *, total: int, where: str
) -> None:
    for grouping in ("by_provenance_family", "by_source"):
        rows = contributions.get(grouping)
        require(isinstance(rows, list), f"{where}.{grouping} must be a list")
        amount_sum = 0
        ids: list[str] = []
        for row in rows:
            require(isinstance(row, dict), f"{where}.{grouping} row must be object")
            amount = row.get("amount")
            require(
                isinstance(amount, int) and not isinstance(amount, bool) and amount > 0,
                f"{where}.{grouping} amount must be positive integer",
            )
            identifier = row.get("id")
            require(
                isinstance(identifier, str) and identifier,
                f"{where}.{grouping} id must be non-empty",
            )
            ids.append(identifier)
            amount_sum += amount
            expected_share = amount / total if total else None
            actual_share = row.get("share")
            require(
                expected_share is not None
                and isinstance(actual_share, (int, float))
                and abs(float(actual_share) - expected_share) <= 1e-12,
                f"{where}.{grouping} share does not recompute",
            )
        require(ids == sorted(set(ids)), f"{where}.{grouping} ids drifted")
        require(amount_sum == total, f"{where}.{grouping} amounts do not sum")


def validate_materialized(
    config: Mapping[str, Any],
    mask: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    repo: Path,
    config_path: Path,
) -> list[str]:
    """Validate structural and denominator invariants without opening candidates."""

    validate_config(config, repo=repo)
    checks: list[str] = []
    require(mask.get("schema") == MASK_SCHEMA, "mask schema mismatch")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "receipt schema mismatch")
    require(
        mask.get("candidate_outputs_read") == []
        and mask.get("candidate_outputs_executed") is False,
        "mask claims candidate output access",
    )
    require(
        receipt.get("candidate_outputs_read") == []
        and receipt.get("candidate_outputs_executed") is False,
        "receipt claims candidate output access",
    )
    checks.append("candidate_outputs_not_read_or_executed")

    expected_count = int(config["event_universe"]["expected_event_count"])
    events = mask.get("events")
    require(isinstance(events, list), "mask events must be a list")
    require(len(events) == expected_count, "event universe count drifted")
    require(mask.get("event_count") == expected_count, "mask event_count drifted")
    require(mask.get("metric_count_per_event") == len(METRICS), "metric count drifted")
    require(
        mask.get("event_metric_classification_count")
        == expected_count * len(METRICS),
        "event metric cell count drifted",
    )
    unit_ids = [event.get("unit_id") for event in events]
    require(
        unit_ids == sorted(unit_ids) and len(set(unit_ids)) == len(unit_ids),
        "event unit ids must be unique and sorted",
    )

    reason_taxonomy = set(config["reason_taxonomy"])
    reason_counts = {metric: Counter() for metric in METRICS}
    censor_counts = {metric: Counter() for metric in METRICS}
    raw_exclusion_counts: Counter[str] = Counter()
    eligible_counts = Counter()
    terminal_observed = 0
    for event in events:
        metrics = event.get("metrics")
        require(
            isinstance(metrics, dict) and set(metrics) == set(METRICS),
            f"event {event.get('unit_id')} metric roster drifted",
        )
        raw = event.get("raw_exclusion_reasons", [])
        require(isinstance(raw, list), "raw_exclusion_reasons must be a list")
        raw_exclusion_counts.update(str(reason) for reason in raw)
        observability = event.get("observability")
        require(
            isinstance(observability, dict)
            and isinstance(
                observability.get("same_person_truth_terminal_clear_observed"),
                bool,
            ),
            "event terminal-clear observability must be explicit",
        )
        terminal_clear_observed = observability[
            "same_person_truth_terminal_clear_observed"
        ]
        if terminal_clear_observed:
            require(
                event["anchors"]["truth_terminal_clear_frame"] is not None,
                "terminal clear observed without truth clear anchor",
            )
            require(
                "not_evaluable_pre_clear"
                not in metrics["clearance"].get("reasons", []),
                "terminal clear observed on pre-clear event",
            )
        recall_eligible = metrics["event_recall"]["classification"] == "eligible"
        if recall_eligible and terminal_clear_observed:
            terminal_observed += 1
        for metric in METRICS:
            entry = metrics[metric]
            require(isinstance(entry, dict), "metric entry must be an object")
            classification = entry.get("classification")
            require(
                classification in VALID_CLASSIFICATIONS,
                f"event {event.get('unit_id')} has invalid {metric} classification",
            )
            reasons = entry.get("reasons")
            require(isinstance(reasons, list), "metric reasons must be a list")
            require(
                reasons == sorted(set(reasons)),
                f"event {event.get('unit_id')} {metric} reasons are not canonical",
            )
            require(
                set(reasons).issubset(reason_taxonomy),
                f"event {event.get('unit_id')} {metric} has unknown reason",
            )
            if classification != "eligible":
                require(
                    len(reasons) > 0,
                    f"event {event.get('unit_id')} {metric} exclusion lacks reason",
                )
            else:
                require(
                    len(reasons) == 0,
                    f"event {event.get('unit_id')} {metric} eligible with exclusions",
                )
                eligible_counts[metric] += 1
            reason_counts[metric].update(reasons)
            censor = entry.get("censor_state")
            if censor is not None:
                require(
                    censor in {
                        "not_evaluable_pre_clear",
                        "right_censored_administrative",
                        "right_censored_identity_loss",
                    },
                    "unknown censor state",
                )
                require(classification != "eligible", "censored event marked eligible")
                censor_counts[metric][str(censor)] += 1

        clear_entry = metrics["clearance"]
        if (
            "not_evaluable_pre_clear" in clear_entry["reasons"]
            or event["anchors"]["truth_terminal_clear_frame"] is None
        ):
            require(
                clear_entry["classification"] != "eligible",
                "pre-clear event entered clearance denominator",
            )
            require(
                clear_entry.get("censor_state") == "not_evaluable_pre_clear",
                "pre-clear clearance has optimistic censor state",
            )
        critical_entry = metrics["critical_miss"]
        if critical_entry["classification"] == "eligible":
            details = critical_entry.get("details", {})
            start = details.get("critical_interval_start_frame")
            end = details.get("critical_interval_end_frame")
            require(
                isinstance(start, int)
                and isinstance(end, int)
                and end - start + 1 >= 3,
                "critical boolean entered denominator without frozen interval",
            )
    checks.extend(
        [
            "all_events_have_exactly_eight_classifications",
            "event_ids_unique_and_sorted",
            "exclusions_and_censors_are_explicit",
            "pre_clear_events_excluded_from_clearance",
            "critical_boolean_requires_frozen_interval",
        ]
    )

    universe = receipt.get("event_universe", {})
    require(universe.get("total") == expected_count, "receipt event total drifted")
    require(
        universe.get("event_metric_classification_count")
        == expected_count * len(METRICS),
        "receipt event metric cell total drifted",
    )
    expected_groups = _counter_dict(
        Counter(str(event["dataset_group"]) for event in events)
    )
    require(
        universe.get("by_dataset_group") == expected_groups,
        "dataset group inventory does not recompute",
    )
    require(
        receipt.get("raw_exclusion_reason_counts")
        == _counter_dict(raw_exclusion_counts),
        "raw exclusion reason counts do not recompute",
    )
    require(
        receipt.get("reason_counts")
        == {
            metric: _counter_dict(reason_counts[metric]) for metric in METRICS
        },
        "metric exclusion reason counts do not recompute",
    )
    require(
        receipt.get("censor_counts")
        == {
            metric: _counter_dict(censor_counts[metric]) for metric in METRICS
        },
        "censor counts do not recompute",
    )
    checks.append("event_inventory_and_reason_counts_recompute")

    metrics_receipt = receipt.get("metrics")
    require(
        isinstance(metrics_receipt, dict) and set(metrics_receipt) == set(METRICS),
        "receipt metric roster drifted",
    )
    for metric in METRICS:
        row = metrics_receipt[metric]
        require(row.get("result_status") == "not_tested", "metric was tested")
        require(row.get("numerator") is None, "metric numerator must be null")
        require(row.get("value") is None, "metric value must be null")
        require(
            row.get("ci_method") is None
            and row.get("ci_lower") is None
            and row.get("ci_upper") is None,
            "metric confidence interval must be null",
        )
        require(row.get("bound_sufficient") is False, "bound marked sufficient")
        require(row.get("gate_result") == "not_applicable", "metric has gate result")
        denominator = row.get("denominator")
        if denominator == 0:
            require(
                row.get("support_status") == "not_evaluable",
                "empty denominator was not marked not_evaluable",
            )
        require(
            not (
                denominator in (0, None)
                and row.get("gate_result") in {"pass", "fail"}
            ),
            "zero-over-zero produced a pass/fail",
        )
    for metric in ("event_recall", "critical_miss", "clearance", "regeneration"):
        row = metrics_receipt[metric]
        require(
            row.get("denominator") == eligible_counts[metric],
            f"{metric} denominator does not recompute",
        )
        _validate_contributions(
            row["contributions"],
            total=int(row["denominator"]),
            where=f"metrics.{metric}.contributions",
        )
    repeat = metrics_receipt["repeat"]
    require(repeat.get("denominator") is None, "repeat preoutput pool became denominator")
    require(
        repeat.get("truth_observation_pool_count") == eligible_counts["repeat"],
        "repeat truth pool does not recompute",
    )
    _validate_contributions(
        repeat["truth_observation_pool_contributions"],
        total=int(repeat["truth_observation_pool_count"]),
        where="metrics.repeat.truth_observation_pool_contributions",
    )
    require(
        repeat["contributions"]["actual_candidate_denominator"]
        == {"by_provenance_family": [], "by_source": []},
        "repeat contains candidate denominator before execution",
    )
    checks.extend(
        [
            "zero_over_zero_cannot_pass_or_fail",
            "all_results_remain_not_tested",
            "event_denominators_and_source_shares_recompute",
            "repeat_truth_pool_not_promoted_to_candidate_denominator",
        ]
    )

    exposure_pairs = mask.get("negative_exposure_pair_audit")
    require(
        isinstance(exposure_pairs, list),
        "negative exposure pair audit must be a list",
    )
    pair_ids = [str(row.get("unit_id")) for row in exposure_pairs]
    require(
        pair_ids == sorted(set(pair_ids)),
        "negative exposure pair ids must be unique and sorted",
    )
    eligible_pairs: list[Mapping[str, Any]] = []
    ineligible_pairs: list[Mapping[str, Any]] = []
    primary_pair_exclusions: Counter[str] = Counter()
    all_pair_exclusions: Counter[str] = Counter()
    primary_pair_duration_ns: Counter[str] = Counter()
    for pair in exposure_pairs:
        start = pair.get("start_ns")
        end = pair.get("end_ns")
        require(
            isinstance(start, int) and isinstance(end, int),
            "negative exposure pair timestamps must be integers",
        )
        delta_ns = end - start
        expected_duration = delta_ns if 0 < delta_ns <= 1_000_000_000 else None
        require(
            pair.get("duration_ns") == expected_duration,
            "negative exposure pair duration drifted",
        )
        reasons = pair.get("exclusion_reasons")
        require(isinstance(reasons, list), "pair exclusion reasons must be a list")
        require(
            reasons
            == [
                reason
                for reason in EXPOSURE_EXCLUSION_REASONS
                if reason in set(reasons)
            ],
            "pair exclusion reasons are not canonical",
        )
        classification = pair.get("classification")
        if classification == "eligible":
            require(not reasons, "eligible exposure pair has exclusion reasons")
            require(
                pair.get("primary_exclusion_reason") is None
                and expected_duration is not None,
                "eligible exposure pair lacks a legal duration",
            )
            eligible_pairs.append(pair)
        else:
            require(
                classification == "ineligible" and bool(reasons),
                "ineligible exposure pair lacks reasons",
            )
            primary = str(pair.get("primary_exclusion_reason"))
            require(primary == reasons[0], "pair primary exclusion reason drifted")
            ineligible_pairs.append(pair)
            primary_pair_exclusions[primary] += 1
            all_pair_exclusions.update(str(reason) for reason in reasons)
            if expected_duration is not None:
                primary_pair_duration_ns[primary] += expected_duration

    grouped_pairs: dict[
        tuple[str, str, str], list[tuple[int, int]]
    ] = defaultdict(list)
    for pair in eligible_pairs:
        grouped_pairs[
            (
                str(pair["provenance_family"]),
                str(pair["source_id"]),
                str(pair["sequence_id"]),
            )
        ].append((int(pair["start_ns"]), int(pair["end_ns"])))
    expected_exposure: list[dict[str, Any]] = []
    for key, intervals in sorted(grouped_pairs.items()):
        family, source_id, sequence_id = key
        current_start: int | None = None
        current_end: int | None = None
        for start, end in sorted(set(intervals)):
            if current_start is None:
                current_start, current_end = start, end
            elif start == current_end:
                current_end = end
            elif start >= current_end:
                expected_exposure.append(
                    {
                        "unit_id": (
                            f"{family}::{source_id}::{sequence_id}::"
                            f"{current_start}-{current_end}"
                        ),
                        "provenance_family": family,
                        "source_id": source_id,
                        "sequence_id": sequence_id,
                        "start_ns": current_start,
                        "end_ns": current_end,
                        "duration_ns": current_end - current_start,
                    }
                )
                current_start, current_end = start, end
            else:
                raise ContractError("eligible exposure candidate pairs overlap")
        if current_start is not None and current_end is not None:
            expected_exposure.append(
                {
                    "unit_id": (
                        f"{family}::{source_id}::{sequence_id}::"
                        f"{current_start}-{current_end}"
                    ),
                    "provenance_family": family,
                    "source_id": source_id,
                    "sequence_id": sequence_id,
                    "start_ns": current_start,
                    "end_ns": current_end,
                    "duration_ns": current_end - current_start,
                }
            )

    exposure = mask.get("negative_exposure_intervals")
    require(isinstance(exposure, list), "negative exposure ledger must be a list")
    require(
        exposure == expected_exposure,
        "negative exposure intervals do not rebuild from pair audit",
    )
    previous_by_sequence: dict[tuple[str, str, str], int] = {}
    exposure_total = 0
    exposure_units: list[tuple[str, str, int]] = []
    exposure_ids: list[str] = []
    for interval in exposure:
        start = interval.get("start_ns")
        end = interval.get("end_ns")
        duration = interval.get("duration_ns")
        require(
            isinstance(start, int)
            and isinstance(end, int)
            and isinstance(duration, int)
            and start < end
            and duration == end - start,
            "invalid negative exposure interval",
        )
        key = (
            str(interval["provenance_family"]),
            str(interval["source_id"]),
            str(interval["sequence_id"]),
        )
        require(
            start >= previous_by_sequence.get(key, start),
            "negative exposure intervals overlap",
        )
        previous_by_sequence[key] = end
        exposure_total += duration
        exposure_units.append((key[0], key[1], duration))
        exposure_ids.append(str(interval["unit_id"]))
    require(exposure_ids == sorted(exposure_ids), "exposure ledger is not sorted")
    negative = receipt.get("negative_exposure", {})
    require(
        negative.get("candidate_pair_count") == len(exposure_pairs)
        and negative.get("eligible_pair_count") == len(eligible_pairs)
        and negative.get("ineligible_pair_count") == len(ineligible_pairs)
        and negative.get("eligible_interval_count") == len(exposure)
        and negative.get("eligible_duration_ns") == exposure_total,
        "negative exposure receipt does not recompute",
    )
    require(
        negative.get("primary_exclusion_reason_counts")
        == _counter_dict(primary_pair_exclusions)
        and negative.get("all_exclusion_reason_counts")
        == _counter_dict(all_pair_exclusions)
        and negative.get("positive_duration_ns_by_primary_exclusion_reason")
        == _counter_dict(primary_pair_duration_ns),
        "negative exposure pair exclusion counts do not recompute",
    )
    _validate_contributions(
        negative["candidate_pair_contributions"],
        total=len(exposure_pairs),
        where="negative_exposure.candidate_pair_contributions",
    )
    _validate_contributions(
        negative["ineligible_pair_contributions"],
        total=len(ineligible_pairs),
        where="negative_exposure.ineligible_pair_contributions",
    )
    false_alert = metrics_receipt["false_alerts_per_minute"]
    require(
        false_alert.get("denominator") == exposure_total,
        "false-alert denominator does not equal exposure ledger",
    )
    _validate_contributions(
        false_alert["contributions"],
        total=exposure_total,
        where="metrics.false_alerts_per_minute.contributions",
    )
    checks.append(
        "negative_exposure_pair_exclusions_and_intervals_recompute"
    )

    frame_ledger = mask.get("preoutput_frame_ledger")
    require(isinstance(frame_ledger, list), "preoutput frame ledger must be a list")
    frame_unit_ids = [str(row.get("unit_id")) for row in frame_ledger]
    frame_sort_keys = [
        (
            str(row.get("provenance_family")),
            str(row.get("source_id")),
            str(row.get("sequence_id")),
            row.get("frame_id"),
        )
        for row in frame_ledger
    ]
    require(
        len(set(frame_unit_ids)) == len(frame_unit_ids)
        and frame_sort_keys == sorted(frame_sort_keys),
        "frame ledger unit ids must be unique and semantically sorted",
    )
    grouped_frames: dict[
        tuple[str, str, str], list[tuple[int, int, str]]
    ] = defaultdict(list)
    for row in frame_ledger:
        frame_id = row.get("frame_id")
        timestamp_ns = row.get("source_capture_timestamp_ns")
        route_state = row.get("route_validity_state")
        require(
            isinstance(frame_id, int)
            and isinstance(timestamp_ns, int)
            and route_state in {"known", "unknown"},
            "explicit frame ledger row is incomplete",
        )
        grouped_frames[
            (
                str(row["provenance_family"]),
                str(row["source_id"]),
                str(row["sequence_id"]),
            )
        ].append((frame_id, timestamp_ns, str(route_state)))
    expected_frame_masks: list[dict[str, Any]] = []
    expected_pair_keys: set[tuple[str, str, str, int, int]] = set()
    for (family, source_id, sequence_id), frames in sorted(
        grouped_frames.items()
    ):
        frames.sort()
        require(
            len({frame[0] for frame in frames}) == len(frames),
            "duplicate frame id inside sequence ledger",
        )
        expected_pair_keys.update(
            (
                family,
                source_id,
                sequence_id,
                frames[index][0],
                frames[index + 1][0],
            )
            for index in range(len(frames) - 1)
        )
        encoded = json.dumps(
            frames, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        route_counts = Counter(frame[2] for frame in frames)
        expected_frame_masks.append(
            {
                "unit_id": f"{family}::{source_id}::{sequence_id}",
                "provenance_family": family,
                "source_id": source_id,
                "sequence_id": sequence_id,
                "frame_count": len(frames),
                "first_frame_id": frames[0][0],
                "last_frame_id": frames[-1][0],
                "first_timestamp_ns": frames[0][1],
                "last_timestamp_ns": frames[-1][1],
                "capture_timestamp_complete": True,
                "route_validity_complete": True,
                "route_state_counts": dict(sorted(route_counts.items())),
                "frame_mask_sha256": sha256_bytes(encoded),
            }
        )
    frame_masks = mask.get("preoutput_frame_masks")
    require(isinstance(frame_masks, list), "frame masks must be a list")
    require(
        frame_masks == expected_frame_masks,
        "frame mask summaries do not rebuild from explicit ledger",
    )
    actual_pair_keys = {
        (
            str(pair["provenance_family"]),
            str(pair["source_id"]),
            str(pair["sequence_id"]),
            int(pair["start_frame_id"]),
            int(pair["end_frame_id"]),
        )
        for pair in exposure_pairs
    }
    require(
        actual_pair_keys == expected_pair_keys,
        "negative exposure pair audit does not cover frame-ledger adjacency",
    )
    frame_count = len(frame_ledger)
    frame_ids = [str(row["unit_id"]) for row in frame_masks]
    require(
        frame_ids == sorted(set(frame_ids)),
        "frame mask unit ids must be unique and sorted",
    )
    frame_support = receipt.get("frame_support", {})
    require(
        frame_support.get("sequence_mask_count") == len(frame_masks)
        and frame_support.get("frame_count") == frame_count,
        "frame support does not recompute",
    )
    require(
        frame_support.get("capture_timestamp_complete") is True
        and frame_support.get("route_validity_complete") is True,
        "preoutput frame support is incomplete",
    )
    require(
        frame_support.get("explicit_frame_ledger_sha256")
        == sha256_bytes(json_bytes(frame_ledger)),
        "explicit frame ledger hash drifted",
    )
    for metric in ("evidence_age", "unknown_or_stale_alert"):
        require(
            metrics_receipt[metric].get("denominator") == frame_count,
            f"{metric} frame denominator drifted",
        )
        _validate_contributions(
            metrics_receipt[metric]["contributions"],
            total=frame_count,
            where=f"metrics.{metric}.contributions",
        )
    checks.append(
        "explicit_preoutput_frame_ledger_masks_and_source_shares_recompute"
    )

    terminal = receipt.get("terminal_clear_observability", {})
    terminal_inventory = receipt.get("terminal_clear_inventory", {})
    require(
        terminal_inventory.get("raw_clear_marker_count")
        == sum(
            event["anchors"]["truth_terminal_clear_frame"] is not None
            for event in events
        )
        and terminal_inventory.get("same_person_terminal_clear_observed_count")
        == sum(
            event["observability"][
                "same_person_truth_terminal_clear_observed"
            ]
            is True
            for event in events
        )
        and terminal_inventory.get("same_person_clearance_eligible_count")
        == eligible_counts["clearance"],
        "terminal-clear inventory does not recompute",
    )
    recall_denominator = eligible_counts["event_recall"]
    require(
        terminal.get("denominator_recall_eligible_events") == recall_denominator
        and terminal.get("numerator_same_person_terminal_clear_observed")
        == terminal_observed,
        "terminal-clear observability does not recompute",
    )
    expected_terminal_value = (
        terminal_observed / recall_denominator if recall_denominator else None
    )
    require(
        terminal.get("value") == expected_terminal_value,
        "terminal-clear observability value drifted",
    )
    require(
        recall_denominator != 0 or terminal.get("support_status") == "not_evaluable",
        "empty terminal-clear denominator was treated as evaluable",
    )
    checks.append("terminal_clear_observability_recomputes_without_preclear")

    ready = str(config["l1_readiness"]["ready_status"])
    conditional = str(config["l1_readiness"]["conditional_status"])
    l0 = str(config["l1_readiness"]["not_ready_status"])
    event_floor_metrics = (
        "event_recall",
        "critical_miss",
        "clearance",
        "regeneration",
    )
    for metric in event_floor_metrics:
        expected_status = (
            ready
            if int(metrics_receipt[metric]["denominator"])
            >= int(config["l1_readiness"][metric]["minimum_denominator"])
            else l0
        )
        require(
            metrics_receipt[metric]["l1_readiness"] == expected_status,
            f"{metric} L1 floor does not recompute",
        )
    expected_repeat_status = (
        conditional
        if int(repeat["truth_observation_pool_count"])
        >= int(
            config["l1_readiness"]["repeat"][
                "minimum_actual_candidate_delivered_denominator"
            ]
        )
        else l0
    )
    require(
        repeat["l1_readiness"] == expected_repeat_status,
        "repeat conditional L1 floor does not recompute",
    )
    expected_false_alert_status = (
        ready
        if exposure_total
        >= int(
            config["l1_readiness"]["false_alerts_per_minute"][
                "minimum_exposure_ns"
            ]
        )
        else l0
    )
    require(
        false_alert["l1_readiness"] == expected_false_alert_status,
        "false-alert L1 floor does not recompute",
    )
    require(
        metrics_receipt["evidence_age"]["l1_readiness"]
        == (conditional if frame_support["capture_timestamp_complete"] else l0),
        "evidence-age conditional readiness does not recompute",
    )
    require(
        metrics_receipt["unknown_or_stale_alert"]["l1_readiness"]
        == (ready if frame_support["route_validity_complete"] else l0),
        "unknown/stale readiness does not recompute",
    )
    require(
        all(
            metrics_receipt[metric]["l1_readiness"] in {ready, conditional, l0}
            for metric in METRICS
        ),
        "unknown L1 readiness status",
    )
    expected_routing = {
        "l1_exploratory_eligible_metrics": [
            metric
            for metric in METRICS
            if metrics_receipt[metric]["l1_readiness"] == ready
        ],
        "l1_conditional_on_candidate_observation_metrics": [
            metric
            for metric in METRICS
            if metrics_receipt[metric]["l1_readiness"] == conditional
        ],
        "l0_engineering_diagnostic_metrics": [
            metric
            for metric in METRICS
            if metrics_receipt[metric]["l1_readiness"] == l0
        ],
    }
    routing = receipt.get("l1_routing", {})
    for key, value in expected_routing.items():
        require(routing.get(key) == value, f"{key} does not recompute")
    require(
        set().union(*(set(value) for value in expected_routing.values()))
        == set(METRICS),
        "L1 routing does not partition all metrics",
    )
    require(routing.get("candidate_winner_allowed") is False, "winner authority opened")
    checks.append("l1_routing_recomputes_without_winner_authority")

    config_sha = sha256_file(config_path)
    mask_sha = sha256_bytes(json_bytes(mask))
    require(
        receipt.get("protocol_binding")
        == {
            "path": config_path.relative_to(repo).as_posix(),
            "sha256": config_sha,
        },
        "protocol binding drifted",
    )
    require(
        receipt.get("event_mask_binding", {}).get("sha256") == mask_sha,
        "event mask hash binding drifted",
    )
    authority = receipt.get("authority", {})
    for key in (
        "candidate_execution_in_this_task",
        "candidate_winner",
        "selection",
        "confirmation",
        "android_shadow",
        "h2",
        "human_outcome",
        "production",
    ):
        require(authority.get(key) is False, f"authority {key} opened")
    checks.extend(["protocol_and_mask_hashes_bound", "authority_remains_closed_above_l1"])
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def _resolve(repo: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    config_path = _resolve(repo, args.config)
    try:
        config = load_json(config_path)
        output_root = repo / str(config["outputs"]["root"])
        mask_path = (
            _resolve(repo, args.mask)
            if args.mask
            else output_root / str(config["outputs"]["event_mask"])
        )
        receipt_path = (
            _resolve(repo, args.receipt)
            if args.receipt
            else output_root / str(config["outputs"]["denominator_receipt"])
        )
        mask = load_json(mask_path)
        receipt = load_json(receipt_path)
        checks = validate_materialized(
            config,
            mask,
            receipt,
            repo=repo,
            config_path=config_path,
        )
        expected_mask, expected_receipt = materialize(
            copy.deepcopy(config), repo=repo, config_path=config_path
        )
        require(json_bytes(mask) == json_bytes(expected_mask), "mask recompute mismatch")
        require(
            json_bytes(receipt) == json_bytes(expected_receipt),
            "receipt recompute mismatch",
        )
        checks.append("full_candidate_blind_recomputation_exact_match")
        module_dir = Path(__file__).resolve().parent
        tool_paths = [
            module_dir / "metric_eligibility.py",
            module_dir / "materialize_metric_eligibility_r2_l1.py",
            Path(__file__).resolve(),
            module_dir / "test_metric_eligibility_r2_l1.py",
        ]
        validation = {
            "schema": VALIDATION_SCHEMA,
            "status": "VALID",
            "protocol_binding": {
                "path": config_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(config_path),
            },
            "event_mask_binding": {
                "path": mask_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(mask_path),
            },
            "denominator_receipt_binding": {
                "path": receipt_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(receipt_path),
            },
            "toolchain_bindings": [
                {
                    "path": path.relative_to(repo).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in tool_paths
            ],
            "checks": checks,
            "summary": {
                "event_count": mask["event_count"],
                "event_metric_classification_count": mask[
                    "event_metric_classification_count"
                ],
                "l1_exploratory_eligible_metrics": receipt["l1_routing"][
                    "l1_exploratory_eligible_metrics"
                ],
                "l1_conditional_on_candidate_observation_metrics": receipt[
                    "l1_routing"
                ]["l1_conditional_on_candidate_observation_metrics"],
                "l0_engineering_diagnostic_metrics": receipt["l1_routing"][
                    "l0_engineering_diagnostic_metrics"
                ],
            },
            "candidate_outputs_read": [],
            "candidate_outputs_executed": False,
            "authority": "validation_only_no_metric_result_or_promotion_authority",
        }
        output_root.mkdir(parents=True, exist_ok=True)
        validation_path = output_root / str(config["outputs"]["validation_receipt"])
        validation_path.write_bytes(json_bytes(validation))
    except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"INVALID_METRIC_ELIGIBILITY_R2_L1: {exc}", file=sys.stderr)
        return 2
    print(
        "VALID_METRIC_ELIGIBILITY_R2_L1 "
        f"checks={len(checks)} events={mask['event_count']} "
        f"validation_receipt={validation_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
