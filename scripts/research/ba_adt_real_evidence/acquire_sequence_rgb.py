#!/usr/bin/env python3
"""Acquire explicitly selected ADT preview RGB for ADT-1 development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acquire_sample import MANIFEST_URL, acquire, digest, fetch_manifest


DEFAULT_MAX_TOTAL_MIB = 256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-total-mib", type=int, default=DEFAULT_MAX_TOTAL_MIB)
    parser.add_argument("--manifest-file", type=Path)
    args = parser.parse_args()

    if len(set(args.sequence_id)) != len(args.sequence_id):
        raise ValueError("duplicate --sequence-id")
    manifest = fetch_manifest(args.manifest_file)
    sequences = manifest["sequences"]
    missing = [sequence_id for sequence_id in args.sequence_id if sequence_id not in sequences]
    if missing:
        raise ValueError(f"unknown ADT sequences: {missing}")
    selected = [(sequence_id, sequences[sequence_id]["video_main_rgb"]) for sequence_id in args.sequence_id]
    total_bytes = sum(int(member["file_size_bytes"]) for _, member in selected)
    maximum = args.max_total_mib * 1024 * 1024
    if total_bytes > maximum:
        raise RuntimeError(f"selection is {total_bytes} bytes, above {maximum}-byte cap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence_id, member in selected:
        path, action = acquire(member, args.output_dir)
        rows.append(
            {
                "sequence_id": sequence_id,
                "role": "RGB_SYSTEM_INPUT",
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha1": digest(path, "sha1"),
                "sha256": digest(path, "sha256"),
                "action": action,
            }
        )
    receipt = {
        "schema_version": "ba_adt_selected_rgb_acquisition_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-1",
        "manifest_url": MANIFEST_URL,
        "manifest_file_sha256": digest(args.manifest_file, "sha256") if args.manifest_file else None,
        "selection_source": "ADT0_GT_ONLY_EPISODE_MINING",
        "sequence_count": len(rows),
        "total_bytes": total_bytes,
        "members": rows,
        "gt_payload_count": 0,
        "claim_ceiling": "rgb_input_acquisition_no_perception_result",
        "terminal": "ADT1_SELECTED_RGB_ACQUIRED",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": receipt["terminal"], "sequence_count": len(rows), "total_bytes": total_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
