#!/usr/bin/env python3
"""Consumed-only diagnostic for a fast observer with causal metric anchors."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluate_metric3d_clearance_field_a0 import HORIZONS_M, summarize

ANCHOR_PERIOD_FRAMES = 5
SCHEMA = "blindassist_hftf_dual_frequency_clearance_replay_r0"


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report.get("frames"), list) or not report["frames"]:
        raise ValueError(f"{path}: report has no frames")
    return report


def frame_key(frame: dict[str, Any]) -> tuple[str, float, str]:
    return (
        str(frame["sequence_id"]),
        float(frame["timestamp"]),
        str(frame["frame_path"]),
    )


def adjusted_field(
    field: dict[str, Any],
    band_offsets: dict[str, float] | None,
    height_offset: float | None,
) -> dict[str, Any]:
    if field.get("status") != "VALID" or not band_offsets:
        return {"status": "UNKNOWN_METRIC_ANCHOR"}
    output = copy.deepcopy(field)
    corrected_bands = 0
    for band, band_output in output["bands"].items():
        clearance = band_output.get("clearance_m")
        if band not in band_offsets or clearance is None:
            band_output["clearance_m"] = None
            band_output["occupied_by_horizon"] = {
                str(horizon): None for horizon in HORIZONS_M
            }
            band_output["anchor_status"] = "UNKNOWN_METRIC_ANCHOR"
            continue
        offset = band_offsets[band]
        corrected = max(0.0, float(clearance) + offset)
        band_output["clearance_m"] = corrected
        band_output["occupied_by_horizon"] = {
            str(horizon): corrected <= horizon for horizon in HORIZONS_M
        }
        band_output["anchor_status"] = "VALID"
        corrected_bands += 1
    if corrected_bands == 0:
        return {"status": "UNKNOWN_METRIC_ANCHOR"}
    if height_offset is not None and output.get("camera_height_m") is not None:
        output["camera_height_m"] = max(
            0.0, float(output["camera_height_m"]) + height_offset
        )
    output["anchor_adjusted"] = True
    return output


def replay(
    metric_report: dict[str, Any],
    fast_report: dict[str, Any],
) -> dict[str, Any]:
    metric_frames = sorted(metric_report["frames"], key=frame_key)
    fast_frames = sorted(fast_report["frames"], key=frame_key)
    if [frame_key(frame) for frame in metric_frames] != [
        frame_key(frame) for frame in fast_frames
    ]:
        raise ValueError("metric and fast reports do not contain identical frames")

    output_frames = []
    sequence_position: dict[str, int] = {}
    band_offsets: dict[str, float] | None = None
    height_offset: float | None = None
    previous_sequence = None
    anchor_frames = 0
    for metric_frame, fast_frame in zip(
        metric_frames, fast_frames, strict=True
    ):
        sequence = str(metric_frame["sequence_id"])
        if sequence != previous_sequence:
            band_offsets = None
            height_offset = None
            previous_sequence = sequence
        position = sequence_position.get(sequence, 0)
        is_anchor = position % ANCHOR_PERIOD_FRAMES == 0
        sequence_position[sequence] = position + 1
        metric_field = metric_frame["candidate"]
        fast_field = fast_frame["candidate"]
        if is_anchor:
            anchor_frames += 1
            band_offsets = None
            height_offset = None
            if (
                metric_field.get("status") == "VALID"
                and fast_field.get("status") == "VALID"
            ):
                candidate_offsets: dict[str, float] = {}
                for band, metric_band in metric_field["bands"].items():
                    metric_clearance = metric_band.get("clearance_m")
                    fast_clearance = fast_field["bands"][band].get(
                        "clearance_m"
                    )
                    if metric_clearance is None or fast_clearance is None:
                        continue
                    candidate_offsets[band] = float(
                        metric_clearance - fast_clearance
                    )
                if candidate_offsets:
                    band_offsets = candidate_offsets
                    if (
                        metric_field.get("camera_height_m") is not None
                        and fast_field.get("camera_height_m") is not None
                    ):
                        height_offset = float(
                            metric_field["camera_height_m"]
                            - fast_field["camera_height_m"]
                        )
            candidate = copy.deepcopy(metric_field)
        else:
            candidate = adjusted_field(
                fast_field, band_offsets, height_offset
            )
        if metric_frame["sensor"] != fast_frame["sensor"]:
            raise ValueError(f"sensor comparator mismatch at {frame_key(metric_frame)}")
        output_frames.append(
            {
                "sequence_root": metric_frame["sequence_root"],
                "sequence_id": sequence,
                "timestamp": metric_frame["timestamp"],
                "frame_path": metric_frame["frame_path"],
                "latency_ms": float(fast_frame["latency_ms"])
                + (float(metric_frame["latency_ms"]) if is_anchor else 0.0),
                "sensor": copy.deepcopy(metric_frame["sensor"]),
                "candidate": candidate,
                "anchor_frame": is_anchor,
            }
        )

    summary = summarize(output_frames)
    gates_passed = all(summary["gates"].values())
    summary.update(
        {
            "schema": SCHEMA,
            "status": (
                "DUAL_FREQUENCY_CONSUMED_DIAGNOSTIC_TASK_GATES_PASS"
                if gates_passed
                else "DUAL_FREQUENCY_CONSUMED_DIAGNOSTIC_TASK_GATES_FAIL"
            ),
            "claim_ceiling": (
                "consumed-only causal replay; no deployment, fresh-transfer, "
                "alert, or safety authority"
            ),
            "anchor_period_frames": ANCHOR_PERIOD_FRAMES,
            "anchor_frames": anchor_frames,
            "metric_model_id": metric_report.get("candidate_model_id"),
            "fast_model_id": fast_report.get("candidate_model_id"),
            "memory": "NOT_MEASURED_FOR_CORESIDENT_MODELS",
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-report", type=Path, required=True)
    parser.add_argument("--fast-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(
        load_report(args.metric_report), load_report(args.fast_report)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "frames"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
