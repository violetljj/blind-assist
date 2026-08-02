#!/usr/bin/env python3
"""Evaluate D37 production-kernel OFF versus scene-scale-veto replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_stage_c_d36_thor_magni_track_veto_event import (
    parse_bool,
    summarize,
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
    "blindassist_hftf_stage_c_d37_thor_magni_"
    "production_scene_scale_veto_event_v0"
)
SUPPORTED_STATUS = (
    "D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_SUPPORTED"
)
EXPECTED_SAMPLES = 530
EXPECTED_SESSIONS = 19
EXPECTED_POSITIVE_ANCHORS = 157
EXPECTED_NEGATIVE_ANCHORS = 373
EXPECTED_POSITIVE_EVENTS = 107
DEFAULT_PRODUCER_RECEIPT = DEFAULT_DETECTIONS.with_name(
    "producer_receipt.json"
)
DEFAULT_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d37-thor-magni-production-scene-scale-veto-event-v0"
)
DEFAULT_KERNEL_REPLAY = DEFAULT_ROOT / "kernel_replay.tsv"
DEFAULT_OUTPUT = DEFAULT_ROOT / "report.json"


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
                "scene_contradict_observations": int(
                    row["scene_contradict_observations"]
                ),
                "scene_abstain_observations": int(
                    row["scene_abstain_observations"]
                ),
                "evidence_absent_frames": int(
                    row["evidence_absent_frames"]
                ),
                "raw_risk_mismatches": int(row["raw_risk_mismatches"]),
                "stable_risk_mismatches": int(
                    row["stable_risk_mismatches"]
                ),
                "non_scene_source_observations": int(
                    row["non_scene_source_observations"]
                ),
            }
        )
    return parsed


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
            and sum(int(row["stable_risk_mismatches"]) for row in rows)
            == 0
            and sum(
                int(row["non_scene_source_observations"]) for row in rows
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
    prefix = "D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_"
    if not all(evaluability.values()):
        status = prefix + "NOT_EVALUABLE"
    elif all(support.values()):
        status = prefix + "SUPPORTED"
    else:
        status = prefix + "NOT_SUPPORTED"
    return evaluability, support, status


def is_supported_status(status: str) -> bool:
    return status == SUPPORTED_STATUS


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
        raise ValueError("D37 kernel replay row count drift")
    if len({row["sample_id"] for row in kernel_rows}) != len(kernel_rows):
        raise ValueError("D37 kernel replay sample IDs are not unique")
    samples = load_jsonl(args.samples)
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("D37 D12 sample IDs are not unique")
    joined: list[dict[str, Any]] = []
    for row in kernel_rows:
        sample = samples_by_id.get(str(row["sample_id"]))
        if sample is None:
            raise ValueError("D37 kernel-to-D12 join failed")
        if not bool(sample["future_onset_target"]["proximity_eligible"]):
            raise ValueError("D37 kernel row is not proximity eligible")
        if (
            str(sample["source_session_id"])
            != str(row["source_session_id"])
            or int(sample["fold"]) != int(row["fold"])
            or int(sample["anchor_scene_frame"])
            != int(row["anchor_scene_frame"])
        ):
            raise ValueError("D37 kernel source identity drift")
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
        raise ValueError("D37 frozen label census drift")

    with args.producer_receipt.open("r", encoding="utf-8") as handle:
        producer_receipt = json.load(handle)
    if (
        int(producer_receipt["sample_count"]) != EXPECTED_SAMPLES
        or int(producer_receipt["session_count"]) != EXPECTED_SESSIONS
        or bool(producer_receipt["future_truth_consumed"])
    ):
        raise ValueError("D37 producer receipt drift")
    if str(producer_receipt["detections_tsv_sha256"]) != sha256(
        args.detections
    ):
        raise ValueError("D37 detection input hash drift")

    pooled = summarize(joined)
    if int(pooled["positive_events"]) != EXPECTED_POSITIVE_EVENTS:
        raise ValueError("D37 positive-event census drift")
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
        "supported": is_supported_status(status),
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
        "diagnostics": {
            "scene_contradict_observations": sum(
                int(row["scene_contradict_observations"])
                for row in joined
            ),
            "scene_abstain_observations": sum(
                int(row["scene_abstain_observations"])
                for row in joined
            ),
            "evidence_absent_frames": sum(
                int(row["evidence_absent_frames"]) for row in joined
            ),
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
