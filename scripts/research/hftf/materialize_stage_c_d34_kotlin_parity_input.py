#!/usr/bin/env python3
"""Materialize source-only D33 detector tracks for Kotlin parity replay."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    DEFAULT_PACKETS,
    HISTORY_COUNT,
    REPO_ROOT,
    sha256,
    source_decision,
)
from evaluate_stage_c_d33_jrdb_detector_track_future_range import (
    load_json,
    load_jsonl,
)
from produce_stage_c_d33_jrdb_detector_tracks import (
    DEFAULT_RECEIPT as DEFAULT_PRODUCER_RECEIPT,
    DEFAULT_TRACKS,
)


SCHEMA = "blindassist_hftf_stage_c_d34_kotlin_parity_input_v0"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts.local/evidence/hftf/"
    "stage-c-d34-kotlin-shadow-state-parity-v0"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "parity_input.tsv"
DEFAULT_RECEIPT = DEFAULT_OUTPUT_ROOT / "input_receipt.json"
FIELDS = (
    "sequence",
    "track_id",
    "frame_index",
    "timestamp_ns",
    "left",
    "top",
    "right",
    "bottom",
    "expected_decision",
    "expected_slope_per_s",
)


def packet_timestamps(
    packet_paths: tuple[Path, ...],
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for path in packet_paths:
        packet = load_json(path)
        sequence = str(packet["sequence"])
        for frame in packet["frames"]:
            key = (sequence, int(frame["frame_index"]))
            if key in result:
                raise ValueError(f"D34 duplicate packet timestamp: {key}")
            result[key] = int(frame["time"]["image_timestamp_ns"])
    return result


def build_rows(
    packet_paths: tuple[Path, ...],
    tracks_path: Path,
    producer_receipt_path: Path,
) -> list[dict[str, Any]]:
    producer = load_json(producer_receipt_path)
    if (
        producer.get("status") != "COMPLETE"
        or producer.get("tracks_sha256") != sha256(tracks_path)
    ):
        raise ValueError("D34 D33 producer binding drift")
    timestamps = packet_timestamps(packet_paths)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in load_jsonl(tracks_path):
        grouped[(str(row["sequence"]), int(row["track_id"]))].append(row)
    output = []
    for (sequence, track_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
        history: list[dict[str, Any]] = []
        previous_frame: int | None = None
        for row in ordered:
            frame_index = int(row["frame_index"])
            if previous_frame is None or frame_index != previous_frame + 1:
                history.clear()
            previous_frame = frame_index
            timestamp_ns = timestamps.get((sequence, frame_index))
            if timestamp_ns is None:
                raise ValueError(
                    f"D34 source timestamp missing: {sequence}/{frame_index}"
                )
            box = [float(value) for value in row["bbox_xyxy"]]
            history.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ns": timestamp_ns,
                    "height": box[3] - box[1],
                }
            )
            if len(history) > HISTORY_COUNT:
                history.pop(0)
            if len(history) == HISTORY_COUNT:
                decision, slope = source_decision(history)
                slope_text = format(slope, ".17g")
            else:
                decision = "ABSTAIN"
                slope_text = ""
            output.append(
                {
                    "sequence": sequence,
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "timestamp_ns": timestamp_ns,
                    "left": format(box[0], ".9g"),
                    "top": format(box[1], ".9g"),
                    "right": format(box[2], ".9g"),
                    "bottom": format(box[3], ".9g"),
                    "expected_decision": decision,
                    "expected_slope_per_s": slope_text,
                }
            )
    return output


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        action="append",
        type=Path,
        dest="packets",
    )
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument(
        "--producer-receipt",
        type=Path,
        default=DEFAULT_PRODUCER_RECEIPT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    packets = tuple(args.packets) if args.packets else DEFAULT_PACKETS
    rows = build_rows(packets, args.tracks, args.producer_receipt)
    write_tsv(args.output, rows)
    receipt = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "source_only": True,
        "row_count": len(rows),
        "distinct_tracks": len(
            {(row["sequence"], row["track_id"]) for row in rows}
        ),
        "input_path": str(args.output.resolve()),
        "input_sha256": sha256(args.output),
        "d33_tracks_sha256": sha256(args.tracks),
        "future_truth_included": False,
    }
    write_json(args.receipt, receipt)
    digest = sha256(args.receipt)
    args.receipt.with_suffix(args.receipt.suffix + ".sha256").write_text(
        f"{digest}  {args.receipt.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
