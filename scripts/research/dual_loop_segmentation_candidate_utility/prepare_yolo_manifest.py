"""Create a normalized host-YOLO manifest from the canonical SANPO manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate_candidate_utility import load_manifest, sha256_file


SCHEMA_VERSION = "blindassist.dual_loop_segmentation_candidate_utility_r0.normalized_yolo_manifest.v1"


def write_normalized_manifest(
    *,
    input_manifest: Path,
    dataset_root: Path,
    split: str | None,
    output: Path,
    receipt: Path,
) -> dict[str, Any]:
    observations = load_manifest(
        input_manifest,
        dataset_root=dataset_root,
        split=split,
        require_truth=False,
    )
    if output.exists() or receipt.exists():
        raise ValueError("refusing to overwrite normalized manifest or receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for observation in observations:
            handle.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_id": observation["source_id"],
                        "frame_id": observation["frame_id"],
                        "source_capture_timestamp_ns": observation["source_capture_timestamp_ns"],
                        "image_path": str(observation["image_path"]),
                        "image_sha256": observation["image_sha256"],
                        "width": observation["width"],
                        "height": observation["height"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_candidate_utility_r0.normalized_yolo_manifest_receipt.v1",
        "status": "COMPLETE",
        "source_manifest": str(input_manifest),
        "source_manifest_sha256": sha256_file(input_manifest),
        "dataset_root": str(dataset_root),
        "split": split,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "frame_count": len(observations),
        "source_ids": sorted({observation["source_id"] for observation in observations}),
        "timestamp_rule": "source_capture_timestamp_ns if present, otherwise frame_id as derived order",
    }
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_normalized_manifest(
        input_manifest=args.manifest.resolve(),
        dataset_root=args.dataset_root.resolve(),
        split=args.split,
        output=args.output.resolve(),
        receipt=args.receipt.resolve(),
    )
    print(json.dumps({"status": result["status"], "frame_count": result["frame_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
