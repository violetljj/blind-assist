"""Measure line-center versus word-carrier geometry on HierText validation GT."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def centroid(vertices: Iterable[Iterable[float]]) -> tuple[float, float]:
    points = [(float(point[0]), float(point[1])) for point in vertices]
    if not points:
        raise ValueError("polygon has no vertices")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def direction(normalized_x: float) -> str:
    centered_x = 2.0 * normalized_x - 1.0
    if centered_x < -0.14:
        return "LEFT"
    if centered_x > 0.14:
        return "RIGHT"
    return "FORWARD"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def valid_words(line: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        word
        for word in line.get("words", [])
        if isinstance(word, dict)
        and word.get("legible") is True
        and str(word.get("text", "")).strip()
        and word.get("vertices")
    ]


def run(gt_path: Path, adapter_limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(gt_path, "rt", encoding="utf-8") as stream:
        root = json.load(stream)
    annotations = root.get("annotations")
    if not isinstance(annotations, list):
        raise TypeError("HierText root has no annotations list")

    image_count = 0
    multiword_lines = 0
    evaluated_words = 0
    direction_disagreements = 0
    normalized_x_errors: list[float] = []
    adapter_rows: list[dict[str, Any]] = []

    for image in annotations:
        image_count += 1
        width = int(image["image_width"])
        height = int(image["image_height"])
        for paragraph_index, paragraph in enumerate(image.get("paragraphs", [])):
            if paragraph.get("legible") is not True:
                continue
            for line_index, line in enumerate(paragraph.get("lines", [])):
                if line.get("legible") is not True or line.get("handwritten") is True:
                    continue
                words = valid_words(line)
                if len(words) < 2:
                    continue
                multiword_lines += 1
                line_center = centroid(line["vertices"])
                line_x = line_center[0] / width
                for word_index, word in enumerate(words):
                    word_center = centroid(word["vertices"])
                    word_x = word_center[0] / width
                    evaluated_words += 1
                    normalized_x_errors.append(abs(word_x - line_x))
                    if direction(word_x) != direction(line_x):
                        direction_disagreements += 1
                    if len(adapter_rows) < adapter_limit:
                        adapter_rows.append(
                            {
                                "schema_version": "blindassist-l10-word-carrier-source-v1",
                                "image_id": image["image_id"],
                                "image_width": width,
                                "image_height": height,
                                "paragraph_index": paragraph_index,
                                "line_index": line_index,
                                "word_index": word_index,
                                "transcript": word["text"],
                                "legible": True,
                                "line_polygon_xy": line["vertices"],
                                "word_polygon_xy": word["vertices"],
                                "line_center_normalized": [
                                    line_center[0] / width,
                                    line_center[1] / height,
                                ],
                                "word_center_normalized": [
                                    word_center[0] / width,
                                    word_center[1] / height,
                                ],
                                "line_direction": direction(line_x),
                                "word_direction": direction(word_x),
                            }
                        )

    disagreement_rate = direction_disagreements / evaluated_words if evaluated_words else 0.0
    result = {
        "schema_version": "blindassist-l10-hiertext-word-carrier-canary-v1",
        "source": {
            "dataset": "HierText validation v1.0",
            "ground_truth_path": str(gt_path.resolve()),
            "ground_truth_sha256": sha256_file(gt_path),
            "license": "CC-BY-SA-4.0",
        },
        "cohort": {
            "policy": "All legible, printed, multi-word validation lines in source order.",
            "image_count": image_count,
            "multiword_line_count": multiword_lines,
            "evaluated_word_count": evaluated_words,
        },
        "metrics": {
            "line_vs_word_direction_disagreement_count": direction_disagreements,
            "line_vs_word_direction_disagreement_rate": round(disagreement_rate, 6),
            "absolute_normalized_x_error_mean": round(
                statistics.fmean(normalized_x_errors) if normalized_x_errors else 0.0, 6
            ),
            "absolute_normalized_x_error_p50": round(percentile(normalized_x_errors, 0.50), 6),
            "absolute_normalized_x_error_p90": round(percentile(normalized_x_errors, 0.90), 6),
        },
        "adapter_row_count": len(adapter_rows),
        "verdict": (
            "HIERTEXT_WORD_CARRIER_GEOMETRY_ADMITTED"
            if evaluated_words and direction_disagreements
            else "HIERTEXT_WORD_CARRIER_GEOMETRY_NOT_ADMITTED"
        ),
        "decision": (
            "Use word polygons as evaluator geometry for the merged-line carrier gap. "
            "Do not use this static dataset as semantic reacquisition evidence."
        ),
        "claim_ceiling": (
            "Static word-versus-line steering geometry only; no temporal presence, "
            "identity reacquisition, action utility, arrival, or handoff claim."
        ),
    }
    if not math.isfinite(result["metrics"]["absolute_normalized_x_error_mean"]):
        raise ValueError("non-finite metric")
    return result, adapter_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--adapter-limit", type=int, default=256)
    args = parser.parse_args()

    result, rows = run(args.ground_truth, args.adapter_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.adapter_output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
