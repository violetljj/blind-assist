#!/usr/bin/env python3
"""Acquire explicitly selected ADT main-groundtruth archives for ADT-0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acquire_sample import MANIFEST_URL, acquire, digest, fetch_manifest


DEFAULT_MAX_TOTAL_MIB = 128


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
    selected = [(sequence_id, sequences[sequence_id]["main_groundtruth"]) for sequence_id in args.sequence_id]
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
                "role": "GT_MINING_AND_EVALUATION_ONLY",
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha1": digest(path, "sha1"),
                "sha256": digest(path, "sha256"),
                "action": action,
            }
        )

    receipt = {
        "schema_version": "ba_adt_selected_groundtruth_acquisition_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-0",
        "manifest_url": MANIFEST_URL,
        "manifest_file_sha256": digest(args.manifest_file, "sha256") if args.manifest_file else None,
        "selection_role": "DEVELOPMENT_PRIORITIZED_FROM_DISCLOSED_CONSUMED_GT_GEOMETRY_PRESCREEN",
        "sequence_count": len(rows),
        "total_bytes": total_bytes,
        "members": rows,
        "rgb_payload_count": 0,
        "gt_may_enter_rgb_estimator": False,
        "claim_ceiling": "development_gt_only_episode_selection_not_fresh_confirmation",
        "terminal": "ADT0_SELECTED_GROUNDTRUTH_ACQUIRED",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": receipt["terminal"], "sequence_count": len(rows), "total_bytes": total_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
