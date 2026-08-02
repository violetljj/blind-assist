#!/usr/bin/env python3
"""Evaluate D36 production-kernel OFF versus track-veto replay."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

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
    "blindassist_hftf_stage_c_d36_thor_magni_"
    "production_track_veto_event_v0"
)
EXPECTED_SAMPLES = 530
EXPECTED_SESSIONS = 19
EXPECTED_POSITIVE_ANCHORS = 157
EXPECTED_NEGATIVE_ANCHORS = 373
EXPECTED_POSITIVE_EVENTS = 107
EVENT_MAX_GAP_FRAMES = 45
DEFAULT_KERNEL_REPLAY = DEFAULT_DETECTIONS.with_name("kernel_replay.tsv")
DEFAULT_PRODUCER_RECEIPT = DEFAULT_DETECTIONS.with_name(
    "producer_receipt.json"
)
DEFAULT_OUTPUT = DEFAULT_DETECTIONS.with_name("report.json")


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"D36 invalid boolean: {value}")


def load_kernel_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    parsed: list[dict[str, Any]] = []
    for row in rows:
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
                "candidate_only_triggered_frames": int(
                    row["candidate_only_triggered_frames"]
                ),
                "suppressed_frames": int(row["suppressed_frames"]),
                "admitted_contradict_frames": int(
                    row["admitted_contradict_frames"]
                ),
                "admitted_confirm_frames": int(
                    row["admitted_confirm_frames"]
                ),
                "track_confirm_frames": int(row["track_confirm_frames"]),
                "track_contradict_frames": int(
                    row["track_contradict_frames"]
                ),
                "track_abstain_frames": int(row["track_abstain_frames"]),
                "evidence_absent_frames": int(
                    row["evidence_absent_frames"]
                ),
                "raw_risk_mismatches": int(row["raw_risk_mismatches"]),
                "stable_risk_mismatches": int(
                    row["stable_risk_mismatches"]
                ),
                "non_track_admitted_evidence": int(
                    row["non_track_admitted_evidence"]
                ),
            }
        )
    return parsed


def positive_event_groups(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    positives = sorted(
        (row for row in rows if bool(row["positive"])),
        key=lambda row: int(row["anchor_scene_frame"]),
    )
    groups: list[list[dict[str, Any]]] = []
    for row in positives:
        frame = int(row["anchor_scene_frame"])
        if (
            not groups
            or frame - int(groups[-1][-1]["anchor_scene_frame"])
            > EVENT_MAX_GAP_FRAMES
        ):
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


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
                    for row in rows
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
        "baseline_positive_anchor_recall": (
            baseline_positive / len(positive) if positive else None
        ),
        "candidate_positive_anchor_recall": (
            candidate_positive / len(positive) if positive else None
        ),
        "positive_anchor_recall_delta": (
            (candidate_positive - baseline_positive) / len(positive)
            if positive
            else None
        ),
        "baseline_positive_event_hits": baseline_event_hits,
        "candidate_positive_event_hits": candidate_event_hits,
        "positive_event_losses": event_losses,
        "positive_event_gains": event_gains,
        "baseline_positive_event_recall": (
            baseline_event_hits / len(groups) if groups else None
        ),
        "candidate_positive_event_recall": (
            candidate_event_hits / len(groups) if groups else None
        ),
        "baseline_negative_alerted": baseline_negative,
        "candidate_negative_alerted": candidate_negative,
        "negative_alert_reduction": baseline_negative
        - candidate_negative,
        "negative_alert_relative_reduction": (
            (baseline_negative - candidate_negative) / baseline_negative
            if baseline_negative
            else None
        ),
        "baseline_negative_false_active_rate": (
            baseline_negative / len(negative) if negative else None
        ),
        "candidate_negative_false_active_rate": (
            candidate_negative / len(negative) if negative else None
        ),
        "candidate_only_triggered_windows": sum(
            bool(row["candidate_only_triggered_window"]) for row in rows
        ),
        "candidate_only_triggered_frames": sum(
            int(row["candidate_only_triggered_frames"]) for row in rows
        ),
        "suppressed_frames": sum(
            int(row["suppressed_frames"]) for row in rows
        ),
        "admitted_contradict_frames": sum(
            int(row["admitted_contradict_frames"]) for row in rows
        ),
        "admitted_confirm_frames": sum(
            int(row["admitted_confirm_frames"]) for row in rows
        ),
    }


def build_gate(
    pooled: dict[str, Any],
    rows: list[dict[str, Any]],
    by_fold: list[dict[str, Any]],
    producer_receipt: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, bool], str]:
    contradicted = [
        row for row in rows if int(row["admitted_contradict_frames"]) > 0
    ]
    evaluability = {
        "complete_cohort": (
            len(rows) == EXPECTED_SAMPLES
            and len({row["source_session_id"] for row in rows})
            == EXPECTED_SESSIONS
        ),
        "anchor_detector_parity": (
            int(producer_receipt["anchor_count_mismatches"]) == 0
            and int(producer_receipt["anchor_mask_mismatches"]) == 0
            and float(producer_receipt["maximum_anchor_slot_error"])
            <= float(producer_receipt["anchor_parity_tolerance"])
        ),
        "risk_path_parity": (
            sum(int(row["raw_risk_mismatches"]) for row in rows) == 0
            and sum(int(row["stable_risk_mismatches"]) for row in rows) == 0
            and sum(
                int(row["non_track_admitted_evidence"]) for row in rows
            )
            == 0
        ),
        "baseline_positive_opportunity": (
            int(pooled["baseline_positive_alerted"]) >= 20
        ),
        "baseline_negative_opportunity": (
            int(pooled["baseline_negative_alerted"]) >= 20
        ),
        "contradict_opportunity": (
            len(contradicted) >= 10
            and len({row["source_session_id"] for row in contradicted}) >= 5
        ),
    }
    support = {
        "zero_positive_event_losses": (
            int(pooled["positive_event_losses"]) == 0
        ),
        "positive_anchor_noninferiority": (
            int(pooled["positive_anchor_losses"]) <= 1
            and float(pooled["positive_anchor_recall_delta"]) >= -0.01
        ),
        "negative_alert_absolute_reduction": (
            int(pooled["negative_alert_reduction"]) >= 10
        ),
        "negative_alert_relative_reduction": (
            float(pooled["negative_alert_relative_reduction"]) >= 0.20
        ),
        "negative_reduction_across_folds": (
            sum(
                int(row["negative_alert_reduction"]) > 0
                for row in by_fold
            )
            >= 3
        ),
        "candidate_creates_no_window_alert": (
            int(pooled["candidate_only_triggered_windows"]) == 0
        ),
    }
    if not all(evaluability.values()):
        status = (
            "D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_"
            "NOT_EVALUABLE"
        )
    elif all(support.values()):
        status = "D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_SUPPORTED"
    else:
        status = (
            "D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_"
            "NOT_SUPPORTED"
        )
    return evaluability, support, status


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


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

    kernel_rows = load_kernel_rows(args.kernel_replay)
    if len(kernel_rows) != EXPECTED_SAMPLES:
        raise ValueError("D36 kernel replay row count drift")
    if len({row["sample_id"] for row in kernel_rows}) != len(kernel_rows):
        raise ValueError("D36 kernel replay sample IDs are not unique")
    samples = load_jsonl(args.samples)
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("D36 D12 sample IDs are not unique")
    joined = []
    for row in kernel_rows:
        sample = samples_by_id.get(str(row["sample_id"]))
        if sample is None:
            raise ValueError("D36 kernel-to-D12 join failed")
        if not bool(sample["future_onset_target"]["proximity_eligible"]):
            raise ValueError("D36 kernel row is not proximity eligible")
        if (
            str(sample["source_session_id"])
            != str(row["source_session_id"])
            or int(sample["fold"]) != int(row["fold"])
            or int(sample["anchor_scene_frame"])
            != int(row["anchor_scene_frame"])
        ):
            raise ValueError("D36 kernel source identity drift")
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
        raise ValueError("D36 frozen label census drift")

    with args.producer_receipt.open("r", encoding="utf-8") as handle:
        producer_receipt = json.load(handle)
    if (
        int(producer_receipt["sample_count"]) != EXPECTED_SAMPLES
        or int(producer_receipt["session_count"]) != EXPECTED_SESSIONS
        or bool(producer_receipt["future_truth_consumed"])
    ):
        raise ValueError("D36 producer receipt drift")
    if str(producer_receipt["detections_tsv_sha256"]) != sha256(
        args.detections
    ):
        raise ValueError("D36 detection input hash drift")

    pooled = summarize(joined)
    if int(pooled["positive_events"]) != EXPECTED_POSITIVE_EVENTS:
        raise ValueError("D36 positive-event census drift")
    by_fold = []
    for fold in range(5):
        summary = summarize(
            [row for row in joined if int(row["fold"]) == fold]
        )
        summary["fold"] = fold
        by_fold.append(summary)
    by_source = []
    for source in sorted(
        {str(row["source_session_id"]) for row in joined}
    ):
        summary = summarize(
            [
                row
                for row in joined
                if str(row["source_session_id"]) == source
            ]
        )
        summary["source_session_id"] = source
        summary["fold"] = int(
            next(
                row["fold"]
                for row in joined
                if str(row["source_session_id"]) == source
            )
        )
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
        "supported": status.endswith("_SUPPORTED"),
        "evaluable": all(evaluability.values()),
        "cohort": {
            "samples": len(joined),
            "sessions": len(
                {str(row["source_session_id"]) for row in joined}
            ),
            "positive_anchors": positive,
            "negative_anchors": negative,
            "positive_events": int(pooled["positive_events"]),
            "outcome_open_development": True,
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
            "production_track_source": True,
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
