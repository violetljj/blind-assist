#!/usr/bin/env python3
"""Diagnose relative directional structure on consumed SANPO events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    CENTRAL_DIRECTIONS,
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    ManifestFrames,
    infer_manifest_probabilities,
    load_model,
)
from train_stage_c_d5_tartanground_development_student import sha256


POSITIVE_BUCKETS = (
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
)
LATERAL_DIRECTIONS = (0, 1, 4, 5)


def direction_features(
    risk: np.ndarray,
    known: np.ndarray,
) -> dict[str, np.ndarray]:
    future = risk[:, 1:, 1:, :, :]
    future_known = known[:, 1:, 1:, :, :]
    body = risk[:, 1:, 1, :, :]
    head = risk[:, 1:, 2, :, :]
    head_known = known[:, 1:, 2, :, :]
    body_k3 = np.partition(body, -3, axis=-1)[..., -3].max(axis=1)
    return {
        "risk_mean": future.mean(axis=(1, 2, 4)),
        "risk_max": future.max(axis=(1, 2, 4)),
        "body_k3_support": body_k3,
        "head_known_risk_support": np.minimum(
            head,
            head_known,
        ).max(axis=(1, 3)),
        "known_mean": future_known.mean(axis=(1, 2, 4)),
    }


def profile_channels(
    risk: np.ndarray,
    known: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {}
    for name, values in direction_features(risk, known).items():
        central_mean = values[:, CENTRAL_DIRECTIONS].mean(axis=1)
        lateral_mean = values[:, LATERAL_DIRECTIONS].mean(axis=1)
        central_peak = values[:, CENTRAL_DIRECTIONS].max(axis=1)
        lateral_peak = values[:, LATERAL_DIRECTIONS].max(axis=1)
        output[f"{name}/central_mean"] = central_mean
        output[f"{name}/central_minus_lateral_mean"] = (
            central_mean - lateral_mean
        )
        output[f"{name}/central_peak_minus_lateral_peak"] = (
            central_peak - lateral_peak
        )
    return output


def event_phase_indices(event: dict[str, Any]) -> list[int]:
    if event["bucket"] in POSITIVE_BUCKETS:
        start, end = map(int, event["alertable_interval_frames"])
        return list(range(start, end + 1))
    return list(range(len(event["frames"])))


def pairwise_auc(
    positives: list[float],
    negatives: list[float],
) -> float:
    if not positives or not negatives:
        raise ValueError("AUC needs positive and negative events")
    score = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return score / (len(positives) * len(negatives))


def summarize_channel(
    rows: list[dict[str, Any]],
    channel: str,
) -> dict[str, Any]:
    positives = [
        float(row["profiles"][channel])
        for row in rows
        if row["bucket"] in POSITIVE_BUCKETS
    ]
    negatives = [
        float(row["profiles"][channel])
        for row in rows
        if row["bucket"] not in POSITIVE_BUCKETS
    ]
    parallel = [
        float(row["profiles"][channel])
        for row in rows
        if row["bucket"] == "parallel_curb_negative"
    ]
    normal = [
        float(row["profiles"][channel])
        for row in rows
        if row["bucket"] == "normal_walkable_negative"
    ]
    return {
        "positive_event_count": len(positives),
        "negative_event_count": len(negatives),
        "positive_median": float(np.median(positives)),
        "negative_median": float(np.median(negatives)),
        "median_delta": float(
            np.median(positives) - np.median(negatives)
        ),
        "auc_vs_all_negative": pairwise_auc(positives, negatives),
        "auc_vs_parallel_curb": pairwise_auc(positives, parallel),
        "auc_vs_normal_walkable": pairwise_auc(positives, normal),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        int(manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in manifest["events"])
        != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    model, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    dataset = ManifestFrames(args.manifest, manifest)
    risks, knowns = infer_manifest_probabilities(
        model,
        dataset,
        manifest,
        args.batch_size,
    )
    event_rows = []
    for event_index, event in enumerate(manifest["events"]):
        channels = profile_channels(
            risks[event_index],
            knowns[event_index],
        )
        indices = event_phase_indices(event)
        event_rows.append(
            {
                "parent_event_id": event["parent_event_id"],
                "source_session_id": event["source_session_id"],
                "bucket": event["bucket"],
                "phase": (
                    "alertable"
                    if event["bucket"] in POSITIVE_BUCKETS
                    else "full_negative_event"
                ),
                "frame_count": len(indices),
                "profiles": {
                    name: float(np.median(values[indices]))
                    for name, values in channels.items()
                },
            }
        )
    channel_names = sorted(event_rows[0]["profiles"])
    summaries = {
        channel: summarize_channel(event_rows, channel)
        for channel in channel_names
    }
    ranked = sorted(
        (
            {
                "channel": channel,
                **summary,
            }
            for channel, summary in summaries.items()
        ),
        key=lambda row: (
            row["auc_vs_all_negative"],
            row["auc_vs_parallel_curb"],
            row["median_delta"],
        ),
        reverse=True,
    )
    result = {
        "schema": (
            "blindassist_hftf_stage_c_d6_sanpo_"
            "direction_profile_diagnostic_v0"
        ),
        "status": "SANPO_DIRECTION_PROFILE_DIAGNOSTIC_COMPLETE",
        "policy": {
            "data_role": "consumed_development_diagnostic",
            "changes_alert_output": False,
            "searches_alert_threshold": False,
            "positive_phase": "human_reviewed_alertable_interval",
            "negative_phase": "entire_negative_event",
            "event_unit": "source_session_parent_event",
            "central_direction_indices": list(CENTRAL_DIRECTIONS),
            "lateral_direction_indices": list(LATERAL_DIRECTIONS),
        },
        "model": {
            "name": args.name,
            "architecture": checkpoint.get("architecture", "pooled"),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "inputs": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "event_count": manifest["event_count"],
            "frame_count": len(dataset),
        },
        "ranked_channels": ranked,
        "channels": summaries,
        "events": event_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(ranked[:5], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
