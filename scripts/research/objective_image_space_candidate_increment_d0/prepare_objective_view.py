"""Strip consumed event/action fields and freeze an objective-only media view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .common import sha256_file, write_json, write_jsonl


def prepare(source: Path, output: Path, receipt: Path) -> dict[str, object]:
    if output.exists() or receipt.exists():
        raise ValueError("refusing to overwrite objective view")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    root = source.parent
    rows: list[dict[str, object]] = []
    sessions: set[str] = set()
    for event in manifest["events"]:
        session = str(event["source_session_id"])
        sessions.add(session)
        for observation_index, frame in enumerate(event["frames"]):
            image = (root / frame["image_path"]).resolve()
            mask = (root / frame["oracle_mask_path"]).resolve()
            if sha256_file(image) != frame["image_sha256"]:
                raise ValueError(f"image SHA drift: {image}")
            if sha256_file(mask) != frame["oracle_mask_sha256"]:
                raise ValueError(f"mask SHA drift: {mask}")
            with Image.open(image) as opened:
                image_size = opened.size
            with Image.open(mask) as opened:
                mask_size = opened.size
            rows.append(
                {
                    "schema_version": "blindassist.objective_image_space_view.v1",
                    "source_session_id": session,
                    "observation_index": observation_index,
                    "source_frame_index": int(frame["source_frame_index"]),
                    "timestamp_ns": int(frame["timestamp_ms"]) * 1_000_000,
                    "image_path": str(image),
                    "image_sha256": frame["image_sha256"],
                    "oracle_mask_path": str(mask),
                    "oracle_mask_sha256": frame["oracle_mask_sha256"],
                    "image_width": image_size[0],
                    "image_height": image_size[1],
                    "mask_width": mask_size[0],
                    "mask_height": mask_size[1],
                }
            )
    write_jsonl(output, rows)
    result = {
        "schema_version": "blindassist.objective_image_space_view_receipt.v1",
        "status": "OBJECTIVE_VIEW_FROZEN",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "output_manifest": str(output.resolve()),
        "output_manifest_sha256": sha256_file(output),
        "frame_count": len(rows),
        "source_session_count": len(sessions),
        "source_sessions": sorted(sessions),
        "forbidden_fields_excluded": [
            "positive",
            "bucket",
            "alertable_interval_frames",
            "passed_interval_frames",
            "event_candidate_id",
            "parent_event_id",
        ],
        "model_output_accessed": False,
    }
    write_json(receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source.resolve(), args.output.resolve(), args.receipt.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
