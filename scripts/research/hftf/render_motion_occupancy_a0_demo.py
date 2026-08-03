#!/usr/bin/env python3
"""Render a deterministic public-data contact sheet for supported A0.1 output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BANDS = ("left", "center", "right")
HORIZON_M = 1.5


def select_middle_frames(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in details:
        if float(row["horizon_m"]) != HORIZON_M:
            continue
        frame = grouped[str(row["sequence_id"])].setdefault(
            str(row["frame_path"]),
            {
                "sequence_id": row["sequence_id"],
                "frame_path": row["frame_path"],
                "timestamp": float(row["timestamp"]),
                "bands": {},
            },
        )
        frame["bands"][row["band"]] = {
            "probability": float(row["scores"]["motion_probability_field"]),
            "truth": bool(row["label_occupied"]),
        }
    selected = []
    for sequence in sorted(grouped):
        complete = [
            frame
            for frame in grouped[sequence].values()
            if set(frame["bands"]) == set(BANDS)
        ]
        complete.sort(key=lambda value: value["timestamp"])
        if complete:
            selected.append(complete[len(complete) // 2])
    return selected


def probability_color(probability: float) -> tuple[int, int, int]:
    if probability >= 0.50:
        return (190, 45, 45)
    if probability >= 0.20:
        return (205, 135, 30)
    return (45, 145, 80)


def render_tile(frame: dict[str, Any], width: int = 480, image_height: int = 360) -> Image.Image:
    image = Image.open(frame["frame_path"]).convert("RGB")
    image.thumbnail((width, image_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, image_height + 92), "white")
    x = (width - image.width) // 2
    canvas.paste(image, (x, 28 + (image_height - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, 28), fill=(25, 25, 25))
    draw.text((8, 7), str(frame["sequence_id"]), fill="white", font=font)
    band_width = width // 3
    for index, band in enumerate(BANDS):
        value = frame["bands"][band]
        left = index * band_width
        right = width if index == 2 else (index + 1) * band_width
        color = probability_color(value["probability"])
        draw.rectangle((left, image_height + 28, right, image_height + 72), fill=color)
        label = f"{band[0].upper()} P={value['probability']:.2f} T={int(value['truth'])}"
        draw.text((left + 8, image_height + 44), label, fill="white", font=font)
    draw.text(
        (8, image_height + 76),
        "CURRENT occupancy within 1.5 m; metric-band strip is schematic; not future prediction",
        fill=(25, 25, 25),
        font=font,
    )
    return canvas


def render(report: dict[str, Any], output: Path) -> dict[str, Any]:
    selected = select_middle_frames(report["details"])
    if not selected:
        raise ValueError("no complete 1.5 m frames")
    tiles = [render_tile(frame) for frame in selected]
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    tile_width, tile_height = tiles[0].size
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height + 42), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, 42), fill=(12, 35, 55))
    draw.text(
        (12, 14),
        "BlindAssist M3D-CF A0.1 | frozen walking_halfsphere median-frame demo",
        fill="white",
        font=ImageFont.load_default(),
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_width, 42 + (index // columns) * tile_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {
        "output": str(output),
        "sequences": len(selected),
        "selection": "middle complete frame per sequence at horizon 1.5 m",
        "sequence_ids": [frame["sequence_id"] for frame in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(json.dumps(render(report, args.output), indent=2))


if __name__ == "__main__":
    main()
