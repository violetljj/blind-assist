#!/usr/bin/env python3
"""Evaluate D38 OFF versus 250 ms bounded temporal scene-veto replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_stage_c_d37_thor_magni_scene_scale_veto_event import (
    DEFAULT_DETECTIONS,
    DEFAULT_PRODUCER_RECEIPT,
    EXPECTED_NEGATIVE_ANCHORS,
    EXPECTED_POSITIVE_ANCHORS,
    EXPECTED_POSITIVE_EVENTS,
    EXPECTED_SAMPLES,
    EXPECTED_SESSIONS,
    build_gate as build_d37_gate,
    load_kernel_rows as load_d37_kernel_rows,
    summarize,
    write_json_atomic,
)
from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_SAMPLES,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d38_thor_magni_"
    "bounded_temporal_veto_event_v0"
)
SUPPORTED_STATUS = (
    "D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_"
    "SUPPORTED_DEVELOPMENT_ONLY"
)
DEFAULT_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d38-thor-magni-bounded-temporal-veto-event-v0"
)
DEFAULT_KERNEL_REPLAY = DEFAULT_ROOT / "kernel_replay.tsv"
DEFAULT_OUTPUT = DEFAULT_ROOT / "report.json"


def load_kernel_rows(path: Path) -> list[dict[str, Any]]:
    parsed = load_d37_kernel_rows(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    if len(raw) != len(parsed):
        raise ValueError("D38 raw/parsed kernel row count drift")
    for target, source in zip(parsed, raw, strict=True):
        if str(target["sample_id"]) != str(source["sample_id"]):
            raise ValueError("D38 raw/parsed kernel row identity drift")
        target["latch_only_suppressed_frames"] = int(
            source["latch_only_suppressed_frames"]
        )
    return parsed


def build_gate(
    pooled: dict[str, Any],
    rows: list[dict[str, Any]],
    by_fold: list[dict[str, Any]],
    producer_receipt: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, bool], str]:
    evaluability, support, _ = build_d37_gate(
        pooled,
        rows,
        by_fold,
        producer_receipt,
    )
    latched = [
        row
        for row in rows
        if int(row["latch_only_suppressed_frames"]) > 0
    ]
    evaluability["latch_only_suppression_opportunity"] = (
        len(latched) >= 10
        and len({row["source_session_id"] for row in latched}) >= 5
    )
    if not all(evaluability.values()):
        status = (
            "D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_"
            "NOT_EVALUABLE"
        )
    elif all(support.values()):
        status = SUPPORTED_STATUS
    else:
        status = (
            "D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_"
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

    kernel_rows = load_kernel_rows(args.kernel_replay)
    if len(kernel_rows) != EXPECTED_SAMPLES:
        raise ValueError("D38 kernel replay row count drift")
    if len({row["sample_id"] for row in kernel_rows}) != len(kernel_rows):
        raise ValueError("D38 kernel replay sample IDs are not unique")
    samples = load_jsonl(args.samples)
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("D38 D12 sample IDs are not unique")
    joined: list[dict[str, Any]] = []
    for row in kernel_rows:
        sample = samples_by_id.get(str(row["sample_id"]))
        if sample is None:
            raise ValueError("D38 kernel-to-D12 join failed")
        if not bool(sample["future_onset_target"]["proximity_eligible"]):
            raise ValueError("D38 kernel row is not proximity eligible")
        if (
            str(sample["source_session_id"])
            != str(row["source_session_id"])
            or int(sample["fold"]) != int(row["fold"])
            or int(sample["anchor_scene_frame"])
            != int(row["anchor_scene_frame"])
        ):
            raise ValueError("D38 kernel source identity drift")
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
        raise ValueError("D38 frozen label census drift")

    with args.producer_receipt.open("r", encoding="utf-8") as handle:
        producer_receipt = json.load(handle)
    if (
        int(producer_receipt["sample_count"]) != EXPECTED_SAMPLES
        or int(producer_receipt["session_count"]) != EXPECTED_SESSIONS
        or bool(producer_receipt["future_truth_consumed"])
    ):
        raise ValueError("D38 producer receipt drift")
    if str(producer_receipt["detections_tsv_sha256"]) != sha256(
        args.detections
    ):
        raise ValueError("D38 detection input hash drift")

    pooled = summarize(joined)
    if int(pooled["positive_events"]) != EXPECTED_POSITIVE_EVENTS:
        raise ValueError("D38 positive-event census drift")
    pooled["latch_only_suppressed_frames"] = sum(
        int(row["latch_only_suppressed_frames"]) for row in joined
    )
    by_fold: list[dict[str, Any]] = []
    for fold in range(5):
        fold_rows = [row for row in joined if int(row["fold"]) == fold]
        summary = summarize(fold_rows)
        summary["fold"] = fold
        summary["latch_only_suppressed_frames"] = sum(
            int(row["latch_only_suppressed_frames"])
            for row in fold_rows
        )
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
        summary["latch_only_suppressed_frames"] = sum(
            int(row["latch_only_suppressed_frames"])
            for row in source_rows
        )
        by_source.append(summary)

    evaluability, support, status = build_gate(
        pooled,
        joined,
        by_fold,
        producer_receipt,
    )
    latched_rows = [
        row
        for row in joined
        if int(row["latch_only_suppressed_frames"]) > 0
    ]
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
            "post_d37_adaptive_outcome_open_development": True,
        },
        "pooled": pooled,
        "by_fold": by_fold,
        "by_source": by_source,
        "evaluability_gates": evaluability,
        "support_gates": support,
        "diagnostics": {
            "latch_only_suppression_anchors": len(latched_rows),
            "latch_only_suppression_sessions": len(
                {
                    str(row["source_session_id"])
                    for row in latched_rows
                }
            ),
            "hold_duration_ns": 250_000_000,
            "hold_duration_source": "production_scene_evidence_ttl",
            "duration_search_performed": False,
        },
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
            "production_scene_scale_source": True,
            "bounded_feedback_only_veto": True,
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
