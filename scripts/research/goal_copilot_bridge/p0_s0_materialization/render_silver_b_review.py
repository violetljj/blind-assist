"""Render score-blind review sheets for Silver-B referent annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


def _place_names(source_report: dict[str, Any]) -> dict[str, str]:
    names: dict[str, list[str]] = {}
    for row in source_report.get("place_building_crosswalk_candidates", []):
        if row.get("status") != "CANDIDATE_ONLY" or len(row.get("building_ids", [])) != 1:
            continue
        names.setdefault(str(row["building_ids"][0]), []).append(str(row.get("place_name") or row["place_id"]))
    return {building_id: values[0] for building_id, values in names.items() if len(values) == 1}


def render_review(
    silver_b: dict[str, Any],
    metadata: dict[str, Any],
    proposal_receipt: dict[str, Any],
    source_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    metadata_by_id = {str(item["id"]): item for item in metadata["images"]}
    proposals_by_id = {str(item["image_id"]): item for item in proposal_receipt["images"]}
    names = _place_names(source_report)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=22)
    for episode in silver_b.get("episodes", []):
        frame_id = str(episode["frame_id"])
        meta = metadata_by_id[frame_id]
        receipt = proposals_by_id[frame_id]
        if len({meta["image_sha256"], receipt["image_sha256"], episode["image_sha256"]}) != 1:
            raise ValueError("image identity drift")
        with Image.open(meta["path"]) as source:
            image = source.convert("RGB")
        weak_ids = {str(item["candidate_id"]) for item in episode["weak_positive_candidates"]}
        draw = ImageDraw.Draw(image)
        candidates = []
        for rank, proposal in enumerate(receipt["proposals"], start=1):
            candidate_id = f"gdino-{frame_id}-{rank:03d}"
            box = [float(value) for value in proposal["bbox_xyxy"]]
            weak = candidate_id in weak_ids
            color = (0, 210, 60) if weak else (235, 50, 50)
            width = 6 if weak else 3
            draw.rectangle(box, outline=color, width=width)
            label = f"{rank}{'*' if weak else ''}"
            label_box = draw.textbbox((box[0], box[1]), label, font=font, stroke_width=2)
            draw.rectangle(label_box, fill=(255, 255, 255))
            draw.text((box[0], box[1]), label, fill=(0, 0, 0), font=font, stroke_width=1, stroke_fill=(255, 255, 255))
            candidates.append({
                "candidate_id": candidate_id,
                "review_label": label,
                "bbox_xyxy": box,
                "proposal_label": proposal["label"],
                "map_geometry_weak_positive": weak,
            })
        title_height = 72
        canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
        canvas.paste(image, (0, title_height))
        title = ImageDraw.Draw(canvas)
        target_name = names.get(str(episode["target_building_id"]), str(episode["target_building_id"]))
        title.text((8, 5), f"TARGET POI: {target_name}", fill=(0, 0, 0), font=title_font)
        title.text((8, 36), "Green * = map+geometry weak positive; red = other DINO proposal; scores hidden", fill=(0, 0, 0), font=font)
        output_path = output_dir / f"{episode['episode_id']}.jpg"
        canvas.save(output_path, format="JPEG", quality=94)
        index.append({
            "episode_id": episode["episode_id"],
            "frame_id": frame_id,
            "target_name": target_name,
            "review_image_path": str(output_path.resolve()),
            "source_image_sha256": episode["image_sha256"],
            "candidates": candidates,
        })
    report = {
        "schema_version": 1,
        "review_mode": "FULL_FRAME_SCORE_BLIND_NUMBERED_CANDIDATES",
        "episode_count": len(index),
        "episodes": index,
    }
    contact_sheets = []
    for page_index in range(0, len(index), 4):
        page_items = index[page_index:page_index + 4]
        opened = [Image.open(item["review_image_path"]).convert("RGB") for item in page_items]
        try:
            cell_width = max(image.width for image in opened)
            cell_height = max(image.height for image in opened)
            sheet = Image.new("RGB", (cell_width * 2, cell_height * 2), "white")
            for cell_index, image in enumerate(opened):
                sheet.paste(image, ((cell_index % 2) * cell_width, (cell_index // 2) * cell_height))
            sheet_path = output_dir / f"contact-{page_index // 4 + 1:02d}.jpg"
            sheet.save(sheet_path, format="JPEG", quality=92)
            contact_sheets.append(str(sheet_path.resolve()))
        finally:
            for image in opened:
                image.close()
    report["contact_sheets"] = contact_sheets
    report["report_sha256"] = materializer.content_sha256(report)
    materializer.write_json(output_dir / "review-index.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-b", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--proposal-receipt", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.silver_b, args.metadata, args.proposal_receipt, args.source_report)]
    report = render_review(*inputs, args.output_dir)
    print(json.dumps({"episode_count": report["episode_count"], "report_sha256": report["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
