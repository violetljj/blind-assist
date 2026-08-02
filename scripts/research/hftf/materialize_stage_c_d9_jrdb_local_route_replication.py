#!/usr/bin/env python3
"""Materialize a bounded JRDB local-route replication corpus."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import sha256


SCHEMA = "blindassist_hftf_stage_c_d9_jrdb_local_route_replication_v0"
SEQUENCES = (
    "clark-center-2019-02-28_0",
    "gates-basement-elevators-2019-01-17_1",
    "meyer-green-2019-03-16_0",
    "stlc-111-2019-04-19_0",
)
FOLDS = {
    SEQUENCES[0]: 0,
    SEQUENCES[1]: 0,
    SEQUENCES[2]: 1,
    SEQUENCES[3]: 1,
}
HISTORY_OFFSETS = (-12, -9, -6, -3, 0)
FUTURE_FRAMES = 30
ANCHOR_STRIDE = 3


def find_label(root: Path, sequence: str) -> Path:
    matches = list(root.rglob(f"labels_3d/{sequence}.json"))
    if len(matches) != 1:
        raise ValueError(
            f"{sequence}: expected one labels_3d file, found {matches}"
        )
    return matches[0]


def radial_bin(distance: float) -> int | None:
    if not 0.0 <= distance < 4.0:
        return None
    return min(3, int(distance))


def direction_bin(x: float, y: float) -> int:
    angle = math.atan2(y, x)
    return min(5, int((angle + math.pi) / (2.0 * math.pi) * 6.0))


def future_target(
    labels: dict[int, list[dict[str, Any]]],
    anchor: int,
) -> dict[str, Any] | None:
    occupancy = np.zeros((2, 6, 4), dtype=np.int64)
    closest: dict[str, Any] | None = None
    observations = 0
    corridor = False
    for offset in range(1, FUTURE_FRAMES + 1):
        horizon = 0 if offset <= FUTURE_FRAMES // 2 else 1
        for person in labels.get(anchor + offset, []):
            box = person["box"]
            x = float(box["cx"])
            y = float(box["cy"])
            if not np.isfinite([x, y]).all():
                continue
            observations += 1
            distance = math.hypot(x, y)
            ring = radial_bin(distance)
            if ring is not None:
                occupancy[horizon, direction_bin(x, y), ring] = 1
            if 0.0 < x <= 4.0 and abs(y) <= 0.9:
                corridor = True
            if closest is None or distance < closest["distance_m"]:
                closest = {
                    "label_id": str(person["label_id"]),
                    "frame_offset": offset,
                    "distance_m": distance,
                    "forward_m": x,
                    "lateral_m": y,
                }
    if closest is None:
        return None
    return {
        "future_minimum_synchronized_distance_m": closest["distance_m"],
        "future_proximity_le_1_25m": closest["distance_m"] <= 1.25,
        "future_corridor_intrusion": corridor,
        "occupancy_target": occupancy.tolist(),
        "occupancy_positive_cells": int(np.sum(occupancy)),
        "future_person_observations": observations,
        "closest": closest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(
            "artifacts.local/work/dual-loop-jrdb-dev-r0/images"
        ),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite JRDB replication output")
    label_roots = args.label_root or [
        Path(
            "artifacts.local/datasets/"
            "jrdb-person-3d-trajectory-sensor-support-and-bias-"
            "cross-sequence-replication-r0"
        ),
        Path(
            "artifacts.local/datasets/"
            "jrdb-single-sequence-native-multisensor-person-"
            "geometry-canary-r0"
        ),
    ]

    records = []
    sequence_rows = []
    for sequence in SEQUENCES:
        image_dir = args.image_root / sequence
        images = sorted(image_dir.glob("*.jpg"))
        expected_names = [f"{index:06d}.jpg" for index in range(120)]
        if [path.name for path in images] != expected_names:
            raise ValueError(f"{sequence}: expected contiguous frames 0..119")
        label_matches = []
        for root in label_roots:
            label_matches.extend(
                list(root.rglob(f"labels_3d/{sequence}.json"))
            )
        if len(label_matches) != 1:
            raise ValueError(
                f"{sequence}: expected one label file, got {label_matches}"
            )
        label_path = label_matches[0]
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        labels = {
            int(key.split(".")[0]): value
            for key, value in payload["labels"].items()
        }
        image_hashes = {
            path.name: sha256(path)
            for path in images
        }
        sequence_records = []
        for anchor in range(
            -HISTORY_OFFSETS[0],
            len(images) - FUTURE_FRAMES,
            ANCHOR_STRIDE,
        ):
            target = future_target(labels, anchor)
            if target is None:
                continue
            history_indices = [
                anchor + offset for offset in HISTORY_OFFSETS
            ]
            sample_id = (
                f"jrdb-route-{sequence}-{anchor:06d}"
            )
            sequence_records.append(
                {
                    "sample_id": sample_id,
                    "dataset_id": "JRDB",
                    "source_session_id": sequence,
                    "ancestry_group": sequence,
                    "fold": FOLDS[sequence],
                    "anchor_frame_index": anchor,
                    "history_frame_indices": history_indices,
                    "history_image_paths": [
                        str(images[index].resolve())
                        for index in history_indices
                    ],
                    "history_image_sha256": [
                        image_hashes[images[index].name]
                        for index in history_indices
                    ],
                    "source": {
                        "image_semantics": "JRDB image_stitched RGB360",
                        "labels_3d_frame": (
                            "jrdb_logical_rgb360_metric_frame"
                        ),
                        "label_path": str(label_path.resolve()),
                        "label_sha256": sha256(label_path),
                        "frame_rate_hz": 15,
                    },
                    "target": target,
                    "authority": {
                        "source_native_geometric_proxy": True,
                        "human_event_truth": False,
                        "promotion": False,
                        "app_or_safety": False,
                    },
                }
            )
        records.extend(sequence_records)
        sequence_rows.append(
            {
                "sequence": sequence,
                "fold": FOLDS[sequence],
                "sample_count": len(sequence_records),
                "proximity_positive": sum(
                    int(
                        row["target"][
                            "future_proximity_le_1_25m"
                        ]
                    )
                    for row in sequence_records
                ),
                "corridor_positive": sum(
                    int(
                        row["target"]["future_corridor_intrusion"]
                    )
                    for row in sequence_records
                ),
                "label_path": str(label_path.resolve()),
                "label_sha256": sha256(label_path),
            }
        )

    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 104:
        raise ValueError(f"Expected 104 JRDB samples, got {len(records)}")
    args.output_root.mkdir(parents=True)
    samples_path = args.output_root / "samples.jsonl"
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "D9_JRDB_LOCAL_ROUTE_REPLICATION_MATERIALIZED",
        "authority": {
            "role": "Development independent-dataset replication proxy",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "design": {
            "frame_rate_hz": 15,
            "history_frame_offsets": list(HISTORY_OFFSETS),
            "future_frames": FUTURE_FRAMES,
            "future_seconds": 2.0,
            "anchor_stride_frames": ANCHOR_STRIDE,
            "heldout_folds": {
                "0": list(SEQUENCES[:2]),
                "1": list(SEQUENCES[2:]),
            },
            "fold_note": (
                "Development pair folds selected after geometry-only class "
                "census because single-sequence AUROC is not evaluable"
            ),
            "primary_replication_target": "future_corridor_intrusion",
            "negative_control_target": "future_proximity_le_1_25m",
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(SEQUENCES),
            "proximity_positive": sum(
                int(
                    row["target"]["future_proximity_le_1_25m"]
                )
                for row in records
            ),
            "corridor_positive": sum(
                int(row["target"]["future_corridor_intrusion"])
                for row in records
            ),
        },
        "sequences": sequence_rows,
        "samples": {
            "path": str(samples_path.resolve()),
            "sha256": sha256(samples_path),
        },
        "next_scientific_gate": (
            "equal-capacity temporal-spatial history must improve corridor "
            "AUROC and AP in both source-pair folds; proximity is a "
            "negative-control target"
        ),
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(report_path)
    Path(str(report_path) + ".sha256").write_text(
        f"{digest}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "counts": report["counts"],
                "sequences": sequence_rows,
                "samples_sha256": report["samples"]["sha256"],
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
