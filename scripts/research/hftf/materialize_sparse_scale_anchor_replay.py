#!/usr/bin/env python3
"""Materialize prefix-only sparse scale anchors from a consumed report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from evaluate_camera_scale_calibrated_clearance_r0 import (
    CALIBRATION_FRAMES,
    calibration_ratios,
)
from sparse_scale_anchor_io import SCHEMA


def manifest_timestamp_binding(paths: list[Path]) -> dict[str, int]:
    binding = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            frame = Path(str(row["frame_path"]))
            if not frame.is_absolute():
                frame = (path.parent / frame).resolve()
            value = int(row["timestamp_ns"])
            previous = binding.setdefault(str(frame), value)
            if previous != value:
                raise ValueError(f"{path}:{line_number}: conflicting frame binding")
    if not binding:
        raise ValueError("manifests contain no timestamp bindings")
    return binding


def materialize(report: dict, timestamp_by_frame: dict[str, int]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in report["frames"]:
        groups.setdefault(str(row["sequence_id"]), []).append(row)
    output = []
    for sequence, rows in sorted(groups.items()):
        rows.sort(key=lambda row: float(row["timestamp"]))
        if len(rows) <= CALIBRATION_FRAMES:
            raise ValueError(f"{sequence} has insufficient frames")
        ratios = calibration_ratios(rows)
        if not ratios:
            raise ValueError(f"{sequence} has no calibration pairs")
        scale = float(statistics.median(ratios))
        anchor_frame = str(Path(str(rows[CALIBRATION_FRAMES - 1]["frame_path"])).resolve())
        if anchor_frame not in timestamp_by_frame:
            raise ValueError(f"{sequence} anchor frame has no manifest timestamp binding")
        bound_timestamp = timestamp_by_frame[anchor_frame]
        output.append(
            {
                "schema": SCHEMA,
                "sequence_id": sequence,
                "timestamp_ns": bound_timestamp,
                "scale": scale,
                "pair_count": len(ratios),
                "median_abs_ratio_residual": float(
                    statistics.median(abs(value - scale) for value in ratios)
                ),
                "source": "consumed_sensor_prefix_three_band_proxy",
                "claim_ceiling": "replay fixture only; not a real ToF measurement",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = materialize(
        json.loads(args.report.read_text(encoding="utf-8")),
        manifest_timestamp_binding(args.manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"anchors": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
