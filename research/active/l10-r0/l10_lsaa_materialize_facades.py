#!/usr/bin/env python3
"""Range-read only the frozen LSAA facade members needed for a split."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from remotezip import RemoteZip


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def contact_sheets(
    rows: list[dict[str, Any]],
    truth: dict[str, Any],
    output_dir: Path,
    per_sheet: int = 6,
) -> list[str]:
    written = []
    for sheet_index in range(0, len(rows), per_sheet):
        page = rows[sheet_index : sheet_index + per_sheet]
        cells = []
        for row in page:
            image = Image.open(output_dir / f"{row['item_id']}.jpg").convert("RGB")
            scale = min(1.0, 1320 / image.width, 760 / image.height)
            resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1360, 850), "white")
            left = (1360 - resized.width) // 2
            top = 70 + (760 - resized.height) // 2
            canvas.paste(resized, (left, top))
            box = truth[row["item_id"]]["official_single_door_box_xyxy"]
            draw = ImageDraw.Draw(canvas)
            draw.rectangle(
                [left + box[0] * scale, top + box[1] * scale, left + box[2] * scale, top + box[3] * scale],
                outline=(255, 0, 0),
                width=5,
            )
            label = (
                f"{row['item_id']}  TRUE={row['mission']['house_number']}  "
                f"CF={row['counterfactual_mission']['house_number']}  {row['facade_name']}"
            )
            draw.text((12, 18), label, fill="black", font=ImageFont.load_default(size=22))
            cells.append(canvas)
        sheet = Image.new("RGB", (2720, 850 * 3), (230, 230, 230))
        for offset, cell in enumerate(cells):
            sheet.paste(cell, ((offset % 2) * 1360, (offset // 2) * 850))
        path = output_dir / f"contact_sheet_{sheet_index // per_sheet + 1:02d}.jpg"
        sheet.save(path, quality=92)
        written.append(str(path.resolve()))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--split", choices=("DEVELOPMENT", "CONFIRMATION_HOLDOUT"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require(not args.output_dir.exists(), f"refusing to overwrite output directory: {args.output_dir}")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    truth_doc = json.loads(args.truth.read_text(encoding="utf-8"))
    rows = [row for row in source["rows"] if row["split"] == args.split]
    require(rows, f"no rows for split: {args.split}")
    args.output_dir.mkdir(parents=True)

    archive = source["remote_facade_archive"]
    file_id = archive["google_drive_file_id"]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    receipts = []
    with RemoteZip(url, initial_buffer_size=1024 * 1024) as remote:
        info_by_name = {row.filename: row for row in remote.infolist()}
        for row in rows:
            member = row["remote_archive_member"]
            require(member in info_by_name, f"missing remote archive member: {member}")
            payload = remote.read(member)
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            output = args.output_dir / f"{row['item_id']}.jpg"
            output.write_bytes(payload)
            with Image.open(io.BytesIO(payload)) as image:
                shape_hw = [image.height, image.width]
            info = info_by_name[member]
            receipts.append(
                {
                    "item_id": row["item_id"],
                    "member": member,
                    "member_bytes": len(payload),
                    "member_crc32": f"{info.CRC:08x}",
                    "member_sha256": sha256_bytes(payload),
                    "image_shape_hw": shape_hw,
                    "local_path": str(output.resolve()),
                }
            )

    sheets = contact_sheets(rows, truth_doc["truth"], args.output_dir)
    manifest = {
        "schema": "blindassist-l10-lsaa-facade-materialization-v1",
        "source": {"path": str(args.source.resolve()), "sha256": sha256_file(args.source)},
        "truth_draft": {"path": str(args.truth.resolve()), "sha256": sha256_file(args.truth)},
        "split": args.split,
        "remote_archive": archive,
        "rows": receipts,
        "contact_sheets": sheets,
    }
    manifest_path = args.output_dir / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "rows": len(receipts), "contact_sheets": sheets}, indent=2))


if __name__ == "__main__":
    main()
