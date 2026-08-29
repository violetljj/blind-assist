"""Render PB2-A source-only contact sheets before any model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]


def render(manifest_path: Path, output_root: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    outputs = []
    tile_width, tile_height, label_height = 320, 240, 74
    for split in ("development", "test"):
        entities = [row for row in manifest["entities"] if row["split"] == split]
        canvas = Image.new(
            "RGB", (tile_width * 5, (tile_height + label_height) * len(entities)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, entity in enumerate(entities):
            images = [("reference", None, entity["references"][0])] + [
                ("query", query["facet"], query) for query in entity["queries"]
            ]
            for column, (role, facet, image_row) in enumerate(images):
                with Image.open(image_row["local_path"]).convert("RGB") as image:
                    image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
                    x = column * tile_width + (tile_width - image.width) // 2
                    y0 = row_index * (tile_height + label_height)
                    y = y0 + (tile_height - image.height) // 2
                    canvas.paste(image, (x, y))
                label = f"{entity['id']}\n{role}" + (f" / {facet}" if facet else "")
                draw.multiline_text(
                    (column * tile_width + 5, y0 + tile_height + 4), label, fill="black", font=font
                )
        destination = output_root / f"{split}-source-audit.jpg"
        canvas.save(destination, quality=92)
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts.local/knowledge/named-poi-place-identity-v1/dataset_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/knowledge/named-poi-place-identity-v1/audit",
    )
    args = parser.parse_args()
    for output in render(args.manifest, args.output_root):
        print(output)


if __name__ == "__main__":
    main()
