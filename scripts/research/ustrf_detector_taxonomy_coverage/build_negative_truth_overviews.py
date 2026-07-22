from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite review output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    outputs: list[str] = []
    cell_w, cell_h, label_h, columns = 160, 112, 14, 20
    for source in candidate["sources"]:
        for window in source["windows"]:
            if window["window_type"] != "negative":
                continue
            rows = window["frames"]
            canvas = np.full(
                (math.ceil(len(rows) / columns) * cell_h, columns * cell_w, 3),
                255,
                dtype=np.uint8,
            )
            for position, row in enumerate(rows):
                image = cv2.imread(row["image_path"])
                if image is None:
                    raise ValueError(f"cannot read {row['image_path']}")
                scale = min(cell_w / image.shape[1], (cell_h - label_h) / image.shape[0])
                resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
                ox = position % columns * cell_w
                oy = position // columns * cell_h + label_h
                canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
                for proposal in row["all_person_candidates"]:
                    x1, y1, x2, y2 = [round(value * scale) for value in proposal["bbox_xyxy"]]
                    cv2.rectangle(canvas, (ox + x1, oy + y1), (ox + x2, oy + y2), (0, 200, 0), 1)
                count = len(row["all_person_candidates"])
                color = (0, 0, 180) if count else (0, 100, 0)
                cv2.putText(
                    canvas,
                    f"{row['frame_id']} p={count}",
                    (ox + 2, oy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            output = args.output_dir / f"{window['window_id']}-all-frames.jpg"
            if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise ValueError(f"failed to write {output}")
            outputs.append(str(output))
    print(json.dumps({"negative_windows": len(outputs), "outputs": outputs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
