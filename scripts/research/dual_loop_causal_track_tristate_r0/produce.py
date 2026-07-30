from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from .common import (
    HISTORY_FRAMES,
    IMPLEMENTATION_ID,
    PROTOCOL_ID,
    frame_detection_id,
    immutable_roi_id,
    read_json,
    sha256_file,
    source_decision,
    source_parameter_sha256,
    source_parameters,
    write_exclusive,
    write_jsonl_exclusive,
)


def _valid_box(item: dict[str, Any]) -> list[float] | None:
    try:
        box = [float(value) for value in item["box"]]
    except (KeyError, TypeError, ValueError):
        return None
    if (
        len(box) != 4
        or not all(math.isfinite(value) for value in box)
        or box[2] <= 0
        or box[3] <= 0
    ):
        return None
    return box


def run(
    freeze_path: Path,
    source_root: Path,
    source_acquisition_receipt: Path,
    truth_root: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if truth_root.exists():
        raise ValueError("truth payload exists before source sealing")
    freeze = read_json(freeze_path)
    acquisition = read_json(source_acquisition_receipt)
    if (
        acquisition.get("role") != "source_2d"
        or acquisition.get("freeze_sha256") != sha256_file(freeze_path)
    ):
        raise ValueError("source acquisition binding drift")
    acquisition_by_sequence = {
        str(row["sequence"]): row for row in acquisition["records"]
    }
    rows: list[dict[str, Any]] = []
    dispositions: Counter[str] = Counter()
    for selected in freeze["selected"]:
        sequence = str(selected["sequence"])
        source_path = source_root / "source_2d" / f"{sequence}.json"
        binding = acquisition_by_sequence.get(sequence)
        if binding is None or sha256_file(source_path) != binding["sha256"]:
            raise ValueError(f"source payload binding drift: {sequence}")
        labels = read_json(source_path)["labels"]
        histories: dict[str, deque[tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=HISTORY_FRAMES)
        )
        for local_position, frame in enumerate(selected["frames"]):
            stem = str(frame["frame_stem"])
            timestamp_ns = int(frame["timestamp_ns"])
            items = labels.get(f"{stem}.jpg")
            if not isinstance(items, list):
                raise ValueError(f"source frame absent: {sequence}/{stem}")
            for item in sorted(items, key=lambda value: str(value["label_id"])):
                label_id = str(item["label_id"])
                box = _valid_box(item)
                if box is None:
                    dispositions["INVALID_BOX"] += 1
                    continue
                history = histories[label_id]
                if history and local_position != history[-1][0] + 1:
                    history.clear()
                history.append(
                    (local_position, timestamp_ns / 1_000_000_000.0, math.log(box[3]))
                )
                decision, slope, reason = source_decision(
                    [row[1] for row in history],
                    [row[2] for row in history],
                )
                detection_id = frame_detection_id(sequence, stem, label_id)
                row = {
                    "protocol_id": PROTOCOL_ID,
                    "implementation_id": IMPLEMENTATION_ID,
                    "parameter_sha256": source_parameter_sha256(),
                    "sequence": sequence,
                    "local_position": local_position,
                    "frame_stem": stem,
                    "captured_at_ns": timestamp_ns,
                    "frame_detection_id": detection_id,
                    "track_id": label_id,
                    "track_epoch": f"{sequence}:{label_id}",
                    "immutable_roi_id": immutable_roi_id(detection_id, box),
                    "box_xywh": box,
                    "history_count": len(history),
                    "decision": decision,
                    "signed_log_height_slope_per_s": slope,
                    "reason": reason,
                }
                rows.append(row)
                dispositions[decision] += 1
    write_jsonl_exclusive(output_path, rows)
    receipt = {
        "schema": "blindassist.dual_loop_causal_track_tristate_producer_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "COMPLETE",
        "freeze_sha256": sha256_file(freeze_path),
        "source_acquisition_receipt_sha256": sha256_file(
            source_acquisition_receipt
        ),
        "parameter_sha256": source_parameter_sha256(),
        "parameters": source_parameters(),
        "truth_payload_opened": False,
        "truth_path_absent_at_start": True,
        "output_rows": len(rows),
        "dispositions": dict(sorted(dispositions.items())),
        "output_path": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "identity_contract": [
            "frame_detection_id",
            "track_id",
            "track_epoch",
            "immutable_roi_id",
        ],
    }
    write_exclusive(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-acquisition-receipt", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.freeze,
                args.source_root,
                args.source_acquisition_receipt,
                args.truth_root,
                args.output,
                args.receipt,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
