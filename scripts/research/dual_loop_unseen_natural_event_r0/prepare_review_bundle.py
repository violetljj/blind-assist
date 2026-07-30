from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


EXPECTED_VIDEO_SHA256 = (
    "58971199576c01d675080f8592b0ca69870054108418fefe5bdff73e199e0f49"
)
SOURCE_ID = "commons_shanghai_shopping_street_night_2024"
SAMPLE_HZ = 1
SHEET_COLUMNS = 5
SHEET_ROWS = 4
THUMB_SIZE = (320, 180)
LABEL_HEIGHT = 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(video: Path, output: Path, ffmpeg: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    actual_sha256 = sha256_file(video)
    if actual_sha256 != EXPECTED_VIDEO_SHA256:
        raise ValueError("selected 480p source identity mismatch")

    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    frames = temporary / "frames_1hz"
    sheets = temporary / "contact_sheets"
    frames.mkdir(parents=True)
    sheets.mkdir()

    subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={SAMPLE_HZ},scale=640:360:force_original_aspect_ratio=decrease",
            "-q:v",
            "3",
            "-start_number",
            "0",
            str(frames / "frame_%06d.jpg"),
        ],
        check=True,
    )
    images = sorted(frames.glob("frame_*.jpg"))
    if not images:
        raise ValueError("ffmpeg produced no review frames")

    manifest_path = temporary / "review_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, image_path in enumerate(images):
            expected = f"frame_{index:06d}.jpg"
            if image_path.name != expected:
                raise ValueError(f"non-contiguous frame sequence at {image_path.name}")
            row = {
                "schema_version": "blindassist.model_review_frame.v1",
                "source_id": SOURCE_ID,
                "frame_id": index,
                "source_capture_timestamp_ns": index * 1_000_000_000,
                "image_path": f"frames_1hz/{image_path.name}",
                "image_sha256": sha256_file(image_path),
            }
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    page_size = SHEET_COLUMNS * SHEET_ROWS
    for page_start in range(0, len(images), page_size):
        canvas = Image.new(
            "RGB",
            (
                SHEET_COLUMNS * THUMB_SIZE[0],
                SHEET_ROWS * (THUMB_SIZE[1] + LABEL_HEIGHT),
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for offset, image_path in enumerate(images[page_start : page_start + page_size]):
            index = page_start + offset
            x = (offset % SHEET_COLUMNS) * THUMB_SIZE[0]
            y = (offset // SHEET_COLUMNS) * (THUMB_SIZE[1] + LABEL_HEIGHT)
            with Image.open(image_path) as image:
                thumb = ImageOps.fit(image.convert("RGB"), THUMB_SIZE)
                canvas.paste(thumb, (x, y))
            draw.text(
                (x + 6, y + THUMB_SIZE[1] + 6),
                f"t={index:06d}.0s  frame={index:06d}",
                fill="black",
            )
        sheet_index = page_start // page_size
        canvas.save(
            sheets / f"sheet_{sheet_index:03d}_{page_start:06d}s.jpg",
            quality=90,
        )

    receipt = {
        "schema_version": "blindassist.model_review_bundle_receipt.v1",
        "protocol_id": "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0",
        "status": "COMPLETE",
        "source_id": SOURCE_ID,
        "video_sha256": actual_sha256,
        "video_size_bytes": video.stat().st_size,
        "sample_hz": SAMPLE_HZ,
        "review_frame_count": len(images),
        "contact_sheet_count": (len(images) + page_size - 1) // page_size,
        "review_manifest_sha256": sha256_file(manifest_path),
        "truth_read": False,
        "baseline_output_read": False,
        "candidate_output_read": False,
    }
    (temporary / "review_bundle_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.video, args.output, args.ffmpeg),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
