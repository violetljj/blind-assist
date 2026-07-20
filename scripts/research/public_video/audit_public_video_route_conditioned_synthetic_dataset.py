#!/usr/bin/env python3
"""Audit hashes, exact composition geometry, and route labels for r812."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from build_public_video_route_conditioned_synthetic_dataset import (
    load_json,
    load_jsonl,
    point_bbox_overlap_fraction,
    reject_independent_direction,
    sha256_file,
    write_json,
)


SCHEMA = "blindassist_route_conditioned_synthetic_geometry_audit_v1"
COLORS = {"LEFT": (255, 80, 80), "STRAIGHT": (80, 220, 255), "RIGHT": (80, 255, 80)}


def mask_bbox(mask: np.ndarray) -> list[int]:
    values = mask.max(axis=2) if mask.ndim == 3 else mask
    points = np.argwhere(values > 0)
    if not len(points):
        raise ValueError("mask is empty")
    y1, x1 = points.min(axis=0)
    y2, x2 = points.max(axis=0) + 1
    return [int(x1), int(y1), int(x2), int(y2)]


def outside_bbox_equal(clear: np.ndarray, composite: np.ndarray, bbox: Sequence[int]) -> bool:
    if clear.shape != composite.shape:
        return False
    x1, y1, x2, y2 = [int(value) for value in bbox]
    equal = clear == composite
    equal[y1:y2, x1:x2] = True
    return bool(equal.all())


def render_route_overlay(image: np.ndarray, rows: Sequence[dict[str, Any]]) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    bbox = rows[0].get("bbox_xyxy")
    if bbox:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 128, 255), max(2, width // 500))
    for index, row in enumerate(sorted(rows, key=lambda item: item["route_choice"])):
        choice = row["route_choice"]
        points = np.asarray([[round(x * width), round(y * height)] for x, y in row["route_waypoints_xy_norm"]], dtype=np.int32)
        cv2.polylines(output, [points], False, COLORS[choice], max(3, width // 350), cv2.LINE_AA)
        status = "BLOCK" if row["route_blocked"] else "CLEAR"
        cv2.putText(output, f"{choice}:{status}", (18, 35 + index * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, width / 2200), COLORS[choice], max(2, width // 900), cv2.LINE_AA)
    return output


def contact_sheet(images: Sequence[np.ndarray], *, columns: int = 3, cell_width: int = 560) -> np.ndarray:
    if not images:
        raise ValueError("contact sheet needs images")
    cells = []
    for image in images:
        scale = cell_width / image.shape[1]
        cells.append(cv2.resize(image, (cell_width, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    cell_height = max(image.shape[0] for image in cells)
    rows = (len(cells) + columns - 1) // columns
    canvas = np.full((rows * cell_height, columns * cell_width, 3), 245, dtype=np.uint8)
    for index, image in enumerate(cells):
        row, column = divmod(index, columns)
        canvas[row * cell_height:row * cell_height + image.shape[0], column * cell_width:(column + 1) * cell_width] = image
    return canvas


def audit(dataset: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    reject_independent_direction(dataset)
    generation = load_jsonl(dataset / "generation_records.jsonl")
    examples = load_jsonl(dataset / "route_examples.jsonl")
    compositions = load_json(dataset / "composition_records.json")["records"]
    receipt = load_json(dataset / "build_receipt.json")
    by_image = {row["id"]: row for row in generation}
    examples_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        examples_by_image.setdefault(row["image_id"], []).append(row)
    errors: list[str] = []
    overlays: list[np.ndarray] = []
    checked_labels = 0
    checked_composites = 0
    for composition in compositions:
        image_id = composition["image_id"]
        row = by_image[image_id]
        image_path = dataset / row["image_path"]
        mask_path = dataset / row["objects"][0]["mask_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            errors.append(f"decode_failed:{image_id}")
            continue
        if sha256_file(image_path) != composition["image_sha256"] or sha256_file(mask_path) != composition["mask_sha256"]:
            errors.append(f"hash_mismatch:{image_id}")
        bbox = [int(value) for value in composition["bbox_xyxy"]]
        if mask_bbox(mask) != bbox:
            errors.append(f"mask_bbox_mismatch:{image_id}")
        clear_id = f"{row['attributes']['parent_slug']}_clear_{int(row['attributes']['distance_index']):02d}"
        clear = cv2.imread(str(dataset / by_image[clear_id]["image_path"]), cv2.IMREAD_COLOR)
        if clear is None or not outside_bbox_equal(clear, image, bbox):
            errors.append(f"pixels_changed_outside_bbox:{image_id}")
        route_rows = examples_by_image.get(image_id, [])
        if len(route_rows) != 3:
            errors.append(f"route_choice_count:{image_id}")
        for example in route_rows:
            expected = point_bbox_overlap_fraction(example["route_waypoints_xy_norm"], bbox, image.shape[1], image.shape[0])
            if abs(expected - float(example["intersection_fraction"])) > 1e-12:
                errors.append(f"route_fraction_mismatch:{example['example_id']}")
            if bool(expected >= 1 / 3) != bool(example["route_blocked"]):
                errors.append(f"route_label_mismatch:{example['example_id']}")
            checked_labels += 1
        overlays.append(render_route_overlay(image, route_rows))
        checked_composites += 1
    clear_examples = [row for row in examples if row["obstacle_direction"] == "NONE"]
    if any(row["route_blocked"] or float(row["intersection_fraction"]) != 0.0 for row in clear_examples):
        errors.append("clear_examples_not_clear")
    qa_root = dataset / "qa"
    qa_root.mkdir(exist_ok=True)
    overlay_path = qa_root / "route_overlay_contact_sheet.jpg"
    if not cv2.imwrite(str(overlay_path), contact_sheet(overlays)):
        raise ValueError("cannot write route overlay contact sheet")
    report = {
        "schema": SCHEMA, "dataset": str(dataset), "build_receipt_sha256": sha256_file(dataset / "build_receipt.json"),
        "generated_image_count": len(generation), "checked_composite_count": checked_composites,
        "checked_route_label_count": checked_labels, "checked_clear_route_example_count": len(clear_examples),
        "route_overlay_contact_sheet": str(overlay_path), "route_overlay_contact_sheet_sha256": sha256_file(overlay_path),
        "errors": errors, "ok": not errors and receipt.get("geometry_gate_passed") is True,
        "train_only": True, "real_event_truth": False, "provider_evaluation_credit": False,
        "production_model_replacement_authorized": False,
    }
    report_path = qa_root / "automated_geometry_audit.json"
    write_json(report_path, report)
    Path(str(report_path) + ".sha256").write_text(sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = audit(parse_args(argv).dataset)
    print(json.dumps({"ok": report["ok"], "checked_composite_count": report["checked_composite_count"],
                      "checked_route_label_count": report["checked_route_label_count"], "errors": report["errors"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
