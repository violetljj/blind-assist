from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .io_utils import read_jsonl


def build_rgb_timeline_montage(
    dataset_root: Path,
    output: Path,
    *,
    sample_step: int = 5,
    thumbnail_width: int = 128,
) -> dict[str, int | str]:
    """Render a deterministic timeline sampling at least 20% of every sequence."""
    root = dataset_root.resolve()
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(root / "frame_manifest.jsonl"):
        by_sequence[str(row["sequence_id"])].append(row)
    selected_by_sequence: list[tuple[str, list[dict[str, object]]]] = []
    for sequence_id in sorted(by_sequence):
        rows = sorted(
            by_sequence[sequence_id], key=lambda item: int(item["frame_index"])
        )
        selected = rows[::sample_step]
        if selected[-1] is not rows[-1]:
            selected.append(rows[-1])
        selected_by_sequence.append((sequence_id, selected))
    if not selected_by_sequence:
        raise ValueError("NO_FRAMES")

    first_path = root / str(selected_by_sequence[0][1][0]["rgb_path"])
    first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"RGB_READ:{first_path}")
    thumb_height = int(round(first.shape[0] * thumbnail_width / first.shape[1]))
    label_width = 260
    header_height = 24
    max_columns = max(len(rows) for _, rows in selected_by_sequence)
    canvas = np.full(
        (
            header_height + len(selected_by_sequence) * thumb_height,
            label_width + max_columns * thumbnail_width,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    for column in range(max_columns):
        cv2.putText(
            canvas,
            str(column * sample_step),
            (label_width + column * thumbnail_width + 3, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    sampled = 0
    for row_index, (sequence_id, rows) in enumerate(selected_by_sequence):
        y0 = header_height + row_index * thumb_height
        cv2.putText(
            canvas,
            sequence_id,
            (4, y0 + thumb_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        for column, frame in enumerate(rows):
            image_path = root / str(frame["rgb_path"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"RGB_READ:{image_path}")
            thumb = cv2.resize(
                image,
                (thumbnail_width, thumb_height),
                interpolation=cv2.INTER_AREA,
            )
            x0 = label_width + column * thumbnail_width
            canvas[y0 : y0 + thumb_height, x0 : x0 + thumbnail_width] = thumb
            sampled += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"MONTAGE_WRITE:{output}")
    total = sum(len(rows) for rows in by_sequence.values())
    return {
        "output": str(output.resolve()),
        "sequence_count": len(selected_by_sequence),
        "sampled_frame_count": sampled,
        "total_frame_count": total,
        "sampling_fraction": sampled / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_rgb_timeline_montage(args.dataset, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
