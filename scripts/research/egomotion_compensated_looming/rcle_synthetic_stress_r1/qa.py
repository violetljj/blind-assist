from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.io_utils import (
    read_jsonl,
)


def build_condition_montages(
    root: Path, output_dir: Path, sample_step: int = 5
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    metadata = {
        row["sequence_id"]: row
        for row in read_jsonl(root / "stress_sequence_manifest.jsonl")
    }
    by_condition_sequence: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(root / "algorithm_input_manifest.jsonl"):
        sequence = metadata[row["sequence_id"]]
        by_condition_sequence[sequence["condition_id"]][
            row["sequence_id"]
        ].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    total_sampled = 0
    total_frames = 0
    for condition_id in sorted(by_condition_sequence):
        sequence_map = by_condition_sequence[condition_id]
        first_row = next(iter(sequence_map.values()))[0]
        first = cv2.imread(
            str(root / first_row["rgb_path"]), cv2.IMREAD_COLOR
        )
        if first is None:
            raise ValueError("QA_FIRST_RGB_READ")
        thumb_width = 128
        thumb_height = int(
            round(first.shape[0] * thumb_width / first.shape[1])
        )
        label_width = 260
        rows_selected: list[tuple[str, list[dict[str, Any]]]] = []
        for sequence_id in sorted(sequence_map):
            rows = sorted(
                sequence_map[sequence_id],
                key=lambda row: row["frame_index"],
            )
            selected = rows[::sample_step]
            rows_selected.append((sequence_id, selected))
            total_frames += len(rows)
            total_sampled += len(selected)
        columns = max(len(rows) for _, rows in rows_selected)
        canvas = np.full(
            (
                24 + len(rows_selected) * thumb_height,
                label_width + columns * thumb_width,
                3,
            ),
            245,
            dtype=np.uint8,
        )
        for column in range(columns):
            cv2.putText(
                canvas,
                str(column * sample_step),
                (label_width + column * thumb_width + 3, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        for row_index, (sequence_id, selected) in enumerate(rows_selected):
            y0 = 24 + row_index * thumb_height
            base_id = metadata[sequence_id]["base_sequence_id"]
            cv2.putText(
                canvas,
                base_id,
                (4, y0 + thumb_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            for column, row in enumerate(selected):
                image = cv2.imread(
                    str(root / row["rgb_path"]), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError(f"QA_RGB_READ:{row['rgb_path']}")
                thumb = cv2.resize(
                    image,
                    (thumb_width, thumb_height),
                    interpolation=cv2.INTER_AREA,
                )
                x0 = label_width + column * thumb_width
                canvas[
                    y0 : y0 + thumb_height,
                    x0 : x0 + thumb_width,
                ] = thumb
        output = output_dir / f"{condition_id}.png"
        if not cv2.imwrite(str(output), canvas):
            raise OSError(f"QA_WRITE:{output}")
        reports.append(
            {
                "condition_id": condition_id,
                "output": str(output),
                "sampled_frames": sum(
                    len(rows) for _, rows in rows_selected
                ),
            }
        )
    return {
        "condition_montages": reports,
        "sampled_frames": total_sampled,
        "total_frames": total_frames,
        "sampling_fraction": total_sampled / total_frames,
    }
