"""Create the exact normalized YOLO input view from the objective-only ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_objective_view, sha256_file, write_json, write_jsonl


def prepare(source: Path, output: Path, receipt: Path) -> dict[str, object]:
    if output.exists() or receipt.exists():
        raise ValueError("refusing to overwrite normalized YOLO view")
    rows = load_objective_view(source)
    normalized = [
        {
            "schema_version": (
                "blindassist.objective_image_space_candidate_increment_d0."
                "normalized_yolo_manifest.v1"
            ),
            "source_id": row["source_session_id"],
            "frame_id": row["source_frame_index"],
            "source_capture_timestamp_ns": row["timestamp_ns"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "width": row["image_width"],
            "height": row["image_height"],
        }
        for row in rows
    ]
    write_jsonl(output, normalized)
    result = {
        "schema_version": (
            "blindassist.objective_image_space_candidate_increment_d0."
            "normalized_yolo_manifest_receipt.v1"
        ),
        "status": "YOLO_INPUT_VIEW_FROZEN",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "output_manifest": str(output.resolve()),
        "output_manifest_sha256": sha256_file(output),
        "frame_count": len(normalized),
        "source_session_count": len(
            {str(row["source_session_id"]) for row in rows}
        ),
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
