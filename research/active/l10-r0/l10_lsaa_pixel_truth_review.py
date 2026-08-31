#!/usr/bin/env python3
"""Create evaluator-only review crops for freezing LSAA visible-text truth.

The output is deliberately model-free: it reads only already materialized pixels,
the frozen public source rows, and official LSAA single-door boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expanded_crop(
    image: Image.Image,
    box: list[float],
) -> tuple[Image.Image, list[float], list[int]]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    crop_box = [
        max(0, int(x1 - 4.5 * width)),
        max(0, int(y1 - 3.5 * height)),
        min(image.width, int(x2 + 4.5 * width)),
        min(image.height, int(y2 + 0.75 * height)),
    ]
    crop = image.crop(tuple(crop_box))
    local_box = [
        x1 - crop_box[0],
        y1 - crop_box[1],
        x2 - crop_box[0],
        y2 - crop_box[1],
    ]
    return crop, local_box, crop_box


def render_cell(
    crop: Image.Image,
    local_box: list[float],
    label: str,
    cell_size: tuple[int, int] = (1400, 1100),
) -> Image.Image:
    cell_width, cell_height = cell_size
    header = 90
    usable_height = cell_height - header
    scale = min(cell_width / crop.width, usable_height / crop.height)
    rendered = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    cell = Image.new("RGB", cell_size, "white")
    left = (cell_width - rendered.width) // 2
    top = header + (usable_height - rendered.height) // 2
    cell.paste(rendered, (left, top))
    draw = ImageDraw.Draw(cell)
    draw.rectangle(
        [
            left + local_box[0] * scale,
            top + local_box[1] * scale,
            left + local_box[2] * scale,
            top + local_box[3] * scale,
        ],
        outline=(255, 0, 0),
        width=6,
    )
    draw.text((18, 20), label, fill="black", font=ImageFont.load_default(size=28))
    return cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--materialization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source.resolve()
    truth_path = args.truth.resolve()
    materialization_dir = args.materialization_dir.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"refusing to overwrite output directory: {output_dir}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    truth_doc = json.loads(truth_path.read_text(encoding="utf-8"))
    manifest_path = materialization_dir / "materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_id = {row["item_id"]: row for row in source["rows"]}
    selected_ids = [row["item_id"] for row in manifest["rows"]]
    require(selected_ids, "materialization manifest contains no rows")
    require(all(item_id in rows_by_id for item_id in selected_ids), "source row mismatch")

    output_dir.mkdir(parents=True)
    receipts: list[dict[str, Any]] = []
    cells: list[Image.Image] = []
    sheet_paths: list[str] = []
    per_sheet = 4
    for item_id in selected_ids:
        row = rows_by_id[item_id]
        image_path = materialization_dir / f"{item_id}.jpg"
        require(image_path.is_file(), f"missing materialized image: {image_path}")
        official_box = truth_doc["truth"][item_id]["official_single_door_box_xyxy"]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        crop, local_box, crop_box = expanded_crop(image, official_box)
        crop_path = output_dir / f"{item_id}-door-context.jpg"
        crop.save(crop_path, quality=96)
        label = (
            f"{item_id}  EXPECT={row['mission']['house_number']}  "
            f"CF={row['counterfactual_mission']['house_number']}  RED=LSAA_DOOR"
        )
        cells.append(render_cell(crop, local_box, label))
        receipts.append(
            {
                "item_id": item_id,
                "source_image": str(image_path),
                "source_image_sha256": sha256_file(image_path),
                "expected_house_number": row["mission"]["house_number"],
                "counterfactual_house_number": row["counterfactual_mission"]["house_number"],
                "official_single_door_box_xyxy": official_box,
                "review_crop_xyxy": crop_box,
                "review_crop": str(crop_path),
                "review_crop_sha256": sha256_file(crop_path),
            }
        )
        if len(cells) == per_sheet or item_id == selected_ids[-1]:
            sheet = Image.new("RGB", (2800, 2200), (230, 230, 230))
            for offset, cell in enumerate(cells):
                sheet.paste(cell, ((offset % 2) * 1400, (offset // 2) * 1100))
            sheet_path = output_dir / f"review_sheet_{len(sheet_paths) + 1:02d}.jpg"
            sheet.save(sheet_path, quality=94)
            sheet_paths.append(str(sheet_path))
            cells = []

    receipt = {
        "schema": "blindassist-l10-lsaa-pixel-truth-review-v1",
        "model_free": True,
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "truth_draft": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
        "materialization_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "rows": receipts,
        "review_sheets": sheet_paths,
    }
    receipt_path = output_dir / "review_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(receipts), "review_sheets": sheet_paths}, indent=2))


if __name__ == "__main__":
    main()
