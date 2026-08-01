from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SCHEMA = "blindassist.riskseg_r0.sparse_rgb_review_bundle.v1"
THUMBNAIL = (320, 180)
COLUMNS = 5
MINIMUM_FRAME_COUNT = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_timeline(root: Path) -> dict[str, Any]:
    spec_path = root / "candidate_spec.json"
    manifest_path = root / "manifest.rgb_timeline.jsonl"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = load_jsonl(manifest_path)
    if spec.get("status") != "output_blind_event_eval_rgb_screening_only":
        raise ValueError(f"{root.name}: wrong screening role")
    if spec.get("training_authorized") is not False:
        raise ValueError(f"{root.name}: training must be forbidden")
    if len(rows) < MINIMUM_FRAME_COUNT:
        raise ValueError(f"{root.name}: sparse timeline has too few frames")
    frames: list[Path] = []
    source_indices: list[int] = []
    for row in rows:
        frame = root / row["image_path"]
        if not frame.is_file() or sha256_file(frame) != row["image_sha256"]:
            raise ValueError(f"{root.name}: RGB asset/hash mismatch")
        frames.append(frame)
        source_indices.append(int(row["source_frame_index"]))
    if any(a >= b for a, b in zip(source_indices, source_indices[1:])):
        raise ValueError(f"{root.name}: source frame order is not increasing")
    return {
        "root": root,
        "spec": spec,
        "rows": rows,
        "frames": frames,
        "source_indices": source_indices,
        "spec_sha256": sha256_file(spec_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def render_sheet(timeline: dict[str, Any], destination: Path, ordinal: int) -> None:
    header = 44
    rows = (len(timeline["frames"]) + COLUMNS - 1) // COLUMNS
    canvas = Image.new(
        "RGB",
        (THUMBNAIL[0] * COLUMNS, header + THUMBNAIL[1] * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    session = timeline["spec"]["source"]["session_id"]
    draw.text((6, 5), f"{ordinal:03d} | session {session}", fill="black")
    draw.text(
        (6, 23),
        (
            f"RGB-only sparse screen | source frames "
            f"{timeline['source_indices'][0]}..{timeline['source_indices'][-1]}"
        ),
        fill="black",
    )
    for index, (frame, source_index) in enumerate(
        zip(timeline["frames"], timeline["source_indices"])
    ):
        x = (index % COLUMNS) * THUMBNAIL[0]
        y = header + (index // COLUMNS) * THUMBNAIL[1]
        with Image.open(frame) as image:
            image = image.convert("RGB")
            image.thumbnail(THUMBNAIL)
            canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + 72, y + 17), fill="white")
        draw.text((x + 3, y + 2), f"f={source_index}", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=88)


def prepare(input_root: Path, output_root: Path) -> dict[str, Any]:
    timelines = [
        load_timeline(path)
        for path in sorted(input_root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and (path / "candidate_spec.json").is_file()
    ]
    if not timelines:
        raise ValueError("no completed sparse timelines found")
    items: list[dict[str, Any]] = []
    for ordinal, timeline in enumerate(timelines, start=1):
        session = timeline["spec"]["source"]["session_id"]
        candidate_id = f"sparse-{ordinal:03d}"
        sheet = output_root / "contact_sheets" / f"{candidate_id}.jpg"
        render_sheet(timeline, sheet, ordinal)
        items.append({
            "event_candidate_id": candidate_id,
            "source_session_id": session,
            "source_frame_start": timeline["source_indices"][0],
            "source_frame_end": timeline["source_indices"][-1],
            "sparse_frame_count": len(timeline["frames"]),
            "contact_sheet": sheet.relative_to(output_root).as_posix(),
            "contact_sheet_sha256": sha256_file(sheet),
            "source_spec_sha256": timeline["spec_sha256"],
            "source_manifest_sha256": timeline["manifest_sha256"],
            "role": "output_blind_rgb_cost_control_screen_only",
            "event_truth": None,
        })
    bundle = {
        "schema": SCHEMA,
        "status": "RGB_SCREENING_ONLY_NOT_EVENT_TRUTH",
        "candidate_output_accessed": False,
        "source_masks_in_review_bundle": False,
        "training_authorized": False,
        "items": items,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_index.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bundle = prepare(args.input_root, args.output_root)
    print(json.dumps({
        "ok": True,
        "candidate_count": len(bundle["items"]),
        "output_root": str(args.output_root.resolve()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
