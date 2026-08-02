#!/usr/bin/env python3
"""Evaluate D40 continuous current-risk versus one-second track projection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_stage_c_d36_thor_magni_track_veto_event import (
    parse_bool,
    positive_event_groups,
    write_json_atomic,
)
from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from produce_stage_c_d36_thor_magni_track_veto_input import (
    DEFAULT_OUTPUT as DEFAULT_DETECTIONS,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_SAMPLES,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d40_thor_magni_"
    "continuous_track_projected_risk_v0"
)
SUPPORTED_STATUS = (
    "D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_"
    "SUPPORTED_DEVELOPMENT_ONLY"
)
EXPECTED_SAMPLES = 530
EXPECTED_SESSIONS = 19
EXPECTED_POSITIVE_ANCHORS = 157
EXPECTED_NEGATIVE_ANCHORS = 373
EXPECTED_POSITIVE_EVENTS = 107
EXPECTED_UNIQUE_FRAMES = 3_710
EXPECTED_DETECTIONS = 14_364
DEFAULT_PRODUCER_RECEIPT = DEFAULT_DETECTIONS.with_name(
    "producer_receipt.json"
)
DEFAULT_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d40-thor-magni-continuous-track-projected-risk-v0"
)
DEFAULT_KERNEL_REPLAY = DEFAULT_ROOT / "kernel_replay.tsv"
DEFAULT_OUTPUT = DEFAULT_ROOT / "report.json"


def load_kernel_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        maximum_slope = str(row["maximum_absolute_slope"])
        parsed.append(
            {
                "sample_id": str(row["sample_id"]),
                "source_session_id": str(row["source_session_id"]),
                "fold": int(row["fold"]),
                "anchor_scene_frame": int(row["anchor_scene_frame"]),
                "baseline_any_triggered": parse_bool(
                    str(row["baseline_any_triggered"])
                ),
                "candidate_any_triggered": parse_bool(
                    str(row["candidate_any_triggered"])
                ),
                "candidate_only_triggered_window": parse_bool(
                    str(row["candidate_only_triggered_window"])
                ),
                "baseline_only_triggered_window": parse_bool(
                    str(row["baseline_only_triggered_window"])
                ),
                "forecast_frames": int(row["forecast_frames"]),
                "positive_slope_frames": int(
                    row["positive_slope_frames"]
                ),
                "negative_slope_frames": int(
                    row["negative_slope_frames"]
                ),
                "maximum_absolute_slope": (
                    float(maximum_slope) if maximum_slope else None
                ),
                "segment_count": int(row["segment_count"]),
            }
        )
    return parsed


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if bool(row["positive"])]
    negative = [row for row in rows if not bool(row["positive"])]
    baseline_positive = sum(
        bool(row["baseline_any_triggered"]) for row in positive
    )
    candidate_positive = sum(
        bool(row["candidate_any_triggered"]) for row in positive
    )
    baseline_negative = sum(
        bool(row["baseline_any_triggered"]) for row in negative
    )
    candidate_negative = sum(
        bool(row["candidate_any_triggered"]) for row in negative
    )
    groups: list[list[dict[str, Any]]] = []
    for source in sorted(
        {str(row["source_session_id"]) for row in positive}
    ):
        groups.extend(
            positive_event_groups(
                [
                    row
                    for row in positive
                    if str(row["source_session_id"]) == source
                ]
            )
        )
    baseline_event_hits = sum(
        any(bool(row["baseline_any_triggered"]) for row in group)
        for group in groups
    )
    candidate_event_hits = sum(
        any(bool(row["candidate_any_triggered"]) for row in group)
        for group in groups
    )
    event_losses = sum(
        any(bool(row["baseline_any_triggered"]) for row in group)
        and not any(bool(row["candidate_any_triggered"]) for row in group)
        for group in groups
    )
    event_gains = sum(
        any(bool(row["candidate_any_triggered"]) for row in group)
        and not any(bool(row["baseline_any_triggered"]) for row in group)
        for group in groups
    )
    return {
        "anchors": len(rows),
        "positive_anchors": len(positive),
        "negative_anchors": len(negative),
        "positive_events": len(groups),
        "baseline_positive_alerted": baseline_positive,
        "candidate_positive_alerted": candidate_positive,
        "positive_anchor_losses": sum(
            bool(row["baseline_any_triggered"])
            and not bool(row["candidate_any_triggered"])
            for row in positive
        ),
        "positive_anchor_gains": sum(
            bool(row["candidate_any_triggered"])
            and not bool(row["baseline_any_triggered"])
            for row in positive
        ),
        "positive_anchor_recall_delta": (
            (candidate_positive - baseline_positive) / len(positive)
            if positive
            else None
        ),
        "baseline_positive_event_hits": baseline_event_hits,
        "candidate_positive_event_hits": candidate_event_hits,
        "positive_event_hit_delta": (
            candidate_event_hits - baseline_event_hits
        ),
        "positive_event_losses": event_losses,
        "positive_event_gains": event_gains,
        "baseline_negative_alerted": baseline_negative,
        "candidate_negative_alerted": candidate_negative,
        "negative_alert_delta": candidate_negative - baseline_negative,
        "negative_alert_reduction": baseline_negative
        - candidate_negative,
        "candidate_only_negative_windows": sum(
            bool(row["candidate_only_triggered_window"])
            for row in negative
        ),
        "baseline_only_negative_windows": sum(
            bool(row["baseline_only_triggered_window"])
            for row in negative
        ),
        "forecast_windows": sum(
            int(row["forecast_frames"]) > 0 for row in rows
        ),
        "forecast_frames": sum(
            int(row["forecast_frames"]) for row in rows
        ),
        "positive_slope_frames": sum(
            int(row["positive_slope_frames"]) for row in rows
        ),
        "negative_slope_frames": sum(
            int(row["negative_slope_frames"]) for row in rows
        ),
    }


def build_gate(
    pooled: dict[str, Any],
    rows: list[dict[str, Any]],
    by_fold: list[dict[str, Any]],
    producer_receipt: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, bool], str]:
    forecast_rows = [
        row for row in rows if int(row["forecast_frames"]) > 0
    ]
    evaluability = {
        "complete_cohort": (
            len(rows) == EXPECTED_SAMPLES
            and len({row["source_session_id"] for row in rows})
            == EXPECTED_SESSIONS
        ),
        "source_census": (
            int(producer_receipt["unique_requested_frames"])
            == EXPECTED_UNIQUE_FRAMES
            and int(producer_receipt["detection_rows"])
            == EXPECTED_DETECTIONS
        ),
        "anchor_detector_parity": (
            int(producer_receipt["anchor_count_mismatches"]) == 0
            and int(producer_receipt["anchor_mask_mismatches"]) == 0
            and float(producer_receipt["maximum_anchor_slot_error"])
            <= float(producer_receipt["anchor_parity_tolerance"])
        ),
        "all_anchor_windows_mapped": all(
            int(row["segment_count"]) >= 1 for row in rows
        ),
        "forecast_opportunity": (
            len(forecast_rows) >= 50
            and len(
                {
                    str(row["source_session_id"])
                    for row in forecast_rows
                }
            )
            >= 5
        ),
        "baseline_positive_opportunity": (
            int(pooled["baseline_positive_alerted"]) >= 20
        ),
        "baseline_negative_opportunity": (
            int(pooled["baseline_negative_alerted"]) >= 20
        ),
        "finite_forecast_output": all(
            row["maximum_absolute_slope"] is None
            or (
                float(row["maximum_absolute_slope"])
                == float(row["maximum_absolute_slope"])
                and abs(float(row["maximum_absolute_slope"]))
                < float("inf")
            )
            for row in rows
        ),
    }
    strict_fold_improvements = sum(
        (
            int(row["positive_event_hit_delta"]) > 0
            and int(row["negative_alert_delta"]) <= 0
        )
        or (
            int(row["negative_alert_reduction"]) > 0
            and int(row["positive_event_hit_delta"]) >= 0
        )
        for row in by_fold
    )
    support = {
        "positive_event_noninferiority": (
            int(pooled["positive_event_hit_delta"]) >= 0
        ),
        "positive_anchor_noninferiority": (
            float(pooled["positive_anchor_recall_delta"]) >= -0.01
        ),
        "negative_alert_noninferiority": (
            int(pooled["negative_alert_delta"]) <= 0
        ),
        "candidate_only_negative_bound": (
            int(pooled["candidate_only_negative_windows"]) <= 5
        ),
        "meaningful_strict_gain": (
            int(pooled["positive_event_gains"]) >= 5
            or int(pooled["negative_alert_reduction"]) >= 20
        ),
        "strict_gain_across_folds": strict_fold_improvements >= 3,
    }
    if not all(evaluability.values()):
        status = (
            "D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_"
            "NOT_EVALUABLE"
        )
    elif all(support.values()):
        status = SUPPORTED_STATUS
    else:
        status = (
            "D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_"
            "NOT_SUPPORTED"
        )
    return evaluability, support, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--detections",
        type=Path,
        default=DEFAULT_DETECTIONS,
    )
    parser.add_argument(
        "--producer-receipt",
        type=Path,
        default=DEFAULT_PRODUCER_RECEIPT,
    )
    parser.add_argument(
        "--kernel-replay",
        type=Path,
        default=DEFAULT_KERNEL_REPLAY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_kernel_rows(args.kernel_replay)
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError("D40 kernel replay row count drift")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("D40 kernel replay sample IDs are not unique")
    samples = load_jsonl(args.samples)
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("D40 D12 sample IDs are not unique")
    joined: list[dict[str, Any]] = []
    for row in rows:
        sample = samples_by_id.get(str(row["sample_id"]))
        if sample is None:
            raise ValueError("D40 kernel-to-D12 join failed")
        if not bool(sample["future_onset_target"]["proximity_eligible"]):
            raise ValueError("D40 kernel row is not proximity eligible")
        if (
            str(sample["source_session_id"])
            != str(row["source_session_id"])
            or int(sample["fold"]) != int(row["fold"])
            or int(sample["anchor_scene_frame"])
            != int(row["anchor_scene_frame"])
        ):
            raise ValueError("D40 kernel source identity drift")
        joined.append(
            {
                **row,
                "positive": bool(
                    sample["future_onset_target"]["proximity_onset"]
                ),
            }
        )
    positive = sum(bool(row["positive"]) for row in joined)
    negative = len(joined) - positive
    if (
        positive != EXPECTED_POSITIVE_ANCHORS
        or negative != EXPECTED_NEGATIVE_ANCHORS
    ):
        raise ValueError("D40 frozen label census drift")

    with args.producer_receipt.open("r", encoding="utf-8") as handle:
        producer_receipt = json.load(handle)
    if (
        int(producer_receipt["sample_count"]) != EXPECTED_SAMPLES
        or int(producer_receipt["session_count"]) != EXPECTED_SESSIONS
        or bool(producer_receipt["future_truth_consumed"])
        or str(producer_receipt["detections_tsv_sha256"])
        != sha256(args.detections)
    ):
        raise ValueError("D40 producer receipt drift")

    pooled = summarize(joined)
    if int(pooled["positive_events"]) != EXPECTED_POSITIVE_EVENTS:
        raise ValueError("D40 positive-event census drift")
    by_fold: list[dict[str, Any]] = []
    for fold in range(5):
        summary = summarize(
            [row for row in joined if int(row["fold"]) == fold]
        )
        summary["fold"] = fold
        by_fold.append(summary)
    by_source: list[dict[str, Any]] = []
    for source in sorted(
        {str(row["source_session_id"]) for row in joined}
    ):
        source_rows = [
            row
            for row in joined
            if str(row["source_session_id"]) == source
        ]
        summary = summarize(source_rows)
        summary["source_session_id"] = source
        summary["fold"] = int(source_rows[0]["fold"])
        by_source.append(summary)

    evaluability, support, status = build_gate(
        pooled,
        joined,
        by_fold,
        producer_receipt,
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "supported": status == SUPPORTED_STATUS,
        "evaluable": all(evaluability.values()),
        "cohort": {
            "samples": len(joined),
            "sessions": len(
                {str(row["source_session_id"]) for row in joined}
            ),
            "positive_anchors": positive,
            "negative_anchors": negative,
            "positive_events": int(pooled["positive_events"]),
            "post_d39_adaptive_outcome_open_development": True,
        },
        "pooled": pooled,
        "by_fold": by_fold,
        "by_source": by_source,
        "evaluability_gates": evaluability,
        "support_gates": support,
        "bindings": {
            "samples_sha256": sha256(args.samples),
            "detections_tsv_sha256": sha256(args.detections),
            "producer_receipt_sha256": sha256(
                args.producer_receipt
            ),
            "kernel_replay_sha256": sha256(args.kernel_replay),
        },
        "claims": {
            "production_kotlin_kernel": True,
            "continuous_session_replay": True,
            "causal_track_one_second_box_projection": True,
            "default_app_changed": False,
            "android_device_runtime": False,
            "independent_generalization": False,
            "mainline_promotion": False,
            "product_or_safety": False,
        },
    }
    write_json_atomic(args.output, payload)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256(args.output)}  {args.output.name}\n",
        encoding="ascii",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
