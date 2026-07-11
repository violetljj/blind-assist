#!/usr/bin/env python3
"""Benchmark BlindAssist traversability rules on local SANPO segmentation masks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SAFE_IDS = {3, 5, 6, 17, 30}
NAVIGATION_HAZARD_IDS = {2, 4, 9, 10, 11, 15, 18, 20, 24, 26}
OBSTACLE_IDS = {4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 28, 29}
LABELS = {
    2: "curb",
    4: "road barrier",
    9: "hand rail",
    10: "opening door",
    11: "opening gate",
    15: "stairs",
    18: "inaccessible surface",
    20: "generic obstacle",
    24: "pole",
    26: "bike rack",
}
MASK_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("test-artifacts.local/datasets/blindassist-sanpo-pilot-20260711"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--corridor-top-ratio", type=float, default=0.42)
    parser.add_argument("--corridor-top-half-width-ratio", type=float, default=0.16)
    parser.add_argument("--corridor-bottom-half-width-ratio", type=float, default=0.42)
    return parser.parse_args()


def corridor_mask(height: int, width: int, args: argparse.Namespace) -> np.ndarray:
    result = np.zeros((height, width), dtype=bool)
    top = min(height - 1, max(0, int(height * args.corridor_top_ratio)))
    denominator = max(1, height - 1 - top)
    for y in range(top, height):
        progress = (y - top) / denominator
        half = args.corridor_top_half_width_ratio + (
            args.corridor_bottom_half_width_ratio - args.corridor_top_half_width_ratio
        ) * progress
        left = max(0, int(width * (0.5 - half)))
        right = min(width, max(left + 1, int(width * (0.5 + half))))
        result[y, left:right] = True
    return result


def extract_components(class_ids: np.ndarray, corridor: np.ndarray) -> list[dict]:
    height, width = class_ids.shape
    components = []
    for class_id in sorted(NAVIGATION_HAZARD_IDS):
        selected = (class_ids == class_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, connectivity=4)
        for component_id in range(1, count):
            left, top, component_width, component_height, pixels = stats[component_id]
            if pixels < 12 or pixels / class_ids.size < 0.0008:
                continue
            component_mask = labels == component_id
            corridor_overlap = float((component_mask & corridor).sum() / max(1, pixels))
            bottom_ratio = float((top + component_height) / height)
            minimum_overlap = 0.35 if class_id == 2 else 0.10
            minimum_bottom = 0.62 if class_id == 2 else 0.42
            # v2 treats curb as boundary evidence. It is retained for diagnostics but
            # cannot become a frame-level obstacle without depth or temporal support.
            accepted = class_id != 2 and corridor_overlap >= minimum_overlap and bottom_ratio >= minimum_bottom
            components.append(
                {
                    "class_id": class_id,
                    "label": LABELS[class_id],
                    "pixel_count": int(pixels),
                    "corridor_overlap_ratio": round(corridor_overlap, 4),
                    "bottom_ratio": round(bottom_ratio, 4),
                    "accepted": accepted,
                    "gate_reason": "accepted" if accepted else (
                        "curb_requires_corroboration" if class_id == 2 else "geometry_gate_rejected"
                    ),
                    "bbox_xyxy": [
                        int(left),
                        int(top),
                        int(left + component_width),
                        int(top + component_height),
                    ],
                }
            )
    return components


def intersection_over_actual(actual: list[int], expected: list[int]) -> float:
    left = max(actual[0], expected[0])
    top = max(actual[1], expected[1])
    right = min(actual[2], expected[2])
    bottom = min(actual[3], expected[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    actual_area = max(1, actual[2] - actual[0]) * max(1, actual[3] - actual[1])
    return intersection / actual_area


def scale_bbox(box: list[int], scale_x: float, scale_y: float) -> list[int]:
    return [
        int(round(box[0] * scale_x)),
        int(round(box[1] * scale_y)),
        int(round(box[2] * scale_x)),
        int(round(box[3] * scale_y)),
    ]


def load_rows(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def benchmark(args: argparse.Namespace) -> dict:
    dataset = args.dataset.resolve()
    rows = load_rows(dataset / "manifest.jsonl")
    coverage_rows = []
    class_counts: Counter[str] = Counter()
    primary_total = 0
    primary_selected = 0
    frames_with_hazard = 0
    for row in rows:
        mask_path = dataset / "source_masks" / "test" / Path(row["image_path"]).name
        rgb = np.asarray(Image.open(mask_path).convert("RGB"))
        class_ids = cv2.resize(rgb[:, :, 0], (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_NEAREST)
        corridor = corridor_mask(class_ids.shape[0], class_ids.shape[1], args)
        corridor_ids = class_ids[corridor]
        safe = np.isin(corridor_ids, tuple(SAFE_IDS))
        obstacle = np.isin(corridor_ids, tuple(OBSTACLE_IDS))
        not_safe = ~(safe | obstacle)
        total = max(1, corridor_ids.size)
        all_candidates = extract_components(class_ids, corridor)
        candidates = [candidate for candidate in all_candidates if candidate["accepted"]]
        if candidates:
            frames_with_hazard += 1
        class_counts.update(component["label"] for component in candidates)
        primary_id = row.get("source_primary_region_id")
        if primary_id:
            primary_total += 1
            expected = next((region for region in row.get("source_regions", []) if region["id"] == primary_id), None)
            expected_label = LABELS.get(expected.get("sanpo_class_id")) if expected else None
            if expected and any(
                component["label"] == expected_label
                and intersection_over_actual(
                    scale_bbox(
                        component["bbox_xyxy"],
                        row["width"] / MASK_SIZE,
                        row["height"] / MASK_SIZE,
                    ),
                    expected["bbox_xyxy"],
                ) >= 0.5
                for component in candidates
            ):
                primary_selected += 1
        coverage_rows.append(
            {
                "id": row["id"],
                "safe_coverage": round(float(safe.sum() / total), 4),
                "not_safe_coverage": round(float(not_safe.sum() / total), 4),
                "obstacle_coverage": round(float(obstacle.sum() / total), 4),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
    return {
        "dataset": str(dataset),
        "frame_count": len(rows),
        "frames_with_corridor_hazard": frames_with_hazard,
        "source_primary_region_coverage": round(primary_selected / max(1, primary_total), 4),
        "source_primary_region_selected": primary_selected,
        "source_primary_region_total": primary_total,
        "mean_safe_coverage": round(float(np.mean([row["safe_coverage"] for row in coverage_rows])), 4),
        "mean_not_safe_coverage": round(float(np.mean([row["not_safe_coverage"] for row in coverage_rows])), 4),
        "mean_obstacle_coverage": round(float(np.mean([row["obstacle_coverage"] for row in coverage_rows])), 4),
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "mapping_note": "BlindAssist override: stairs is an obstacle hazard; SANPO paper A.6.1 maps stairs to safe-to-walk.",
        "per_frame": coverage_rows,
    }


def markdown(result: dict) -> str:
    counts = ", ".join(f"{name}={count}" for name, count in result["candidate_class_counts"].items()) or "none"
    return "\n".join(
        [
            "# SANPO Traversability Oracle Baseline",
            "",
            f"- Frames: {result['frame_count']}",
            f"- Frames with corridor hazard: {result['frames_with_corridor_hazard']}",
            f"- Primary region coverage: {result['source_primary_region_coverage']} "
            f"({result['source_primary_region_selected']}/{result['source_primary_region_total']})",
            f"- Mean corridor coverage: safe={result['mean_safe_coverage']}, "
            f"not-safe={result['mean_not_safe_coverage']}, obstacle={result['mean_obstacle_coverage']}",
            f"- Candidate classes: {counts}",
            f"- Mapping note: {result['mapping_note']}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    output_dir = (args.output_dir or args.dataset / "qa" / "traversability-baseline").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "baseline.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
