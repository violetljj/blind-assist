#!/usr/bin/env python3
"""Stage a hash-bound R1.1 target-aware Android replay input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .contract import load_json, sha256_file, write_json
    from .diagnostic_contract import DIAGNOSTIC_ROLE, load_projection, load_target_ledger, require
except ImportError:
    from contract import load_json, sha256_file, write_json
    from diagnostic_contract import DIAGNOSTIC_ROLE, load_projection, load_target_ledger, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--target-ledger", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="JSON with sources[{event_id,video_path,video_sha256}]")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    ledger = load_target_ledger(args.target_ledger)
    load_projection(args.projection_receipt, args.target_ledger, ledger)
    source_manifest = load_json(args.source_manifest)
    sources = source_manifest.get("sources")
    require(isinstance(sources, list), "source manifest needs sources")
    source_by_event = {row.get("event_id"): row for row in sources}
    require(len(source_by_event) == len(sources), "source manifest repeats event")
    require(set(source_by_event) == {event["event_id"] for event in ledger["events"]}, "source manifest inventory mismatch")
    remote_root = "ustrf-crosscam-r11"
    staged_sources = []
    staged_files = []
    for event in ledger["events"]:
        row = source_by_event[event["event_id"]]
        local_video = (repo / row["video_path"]).resolve()
        require(local_video.is_relative_to(repo) and local_video.is_file(), f"{event['event_id']}: video unavailable")
        require(sha256_file(local_video) == row["video_sha256"], f"{event['event_id']}: video SHA mismatch")
        remote_video = f"{remote_root}/videos/{event['event_id']}{local_video.suffix.lower()}"
        staged_sources.append({"event_id": event["event_id"], "source_id": event["source_id"],
                               "video_path": remote_video, "video_sha256": row["video_sha256"]})
        staged_files.append({"host_path": str(local_video), "device_relative_path": remote_video,
                             "sha256": row["video_sha256"]})
    remote_ledger = f"{remote_root}/target_instance_ledger.json"
    remote_projection = f"{remote_root}/frame_projection_receipt.json"
    android_input = {
        "schema": "blindassist_ustrf_crosscam_target_aware_android_input_v2",
        "diagnostic_set_role": DIAGNOSTIC_ROLE,
        "target_ledger_path": remote_ledger, "target_ledger_sha256": sha256_file(args.target_ledger),
        "projection_receipt_path": remote_projection, "projection_receipt_sha256": sha256_file(args.projection_receipt),
        "uncertainty_frame_ratios": [0.01, 0.02, 0.03], "target_match_contract_id": "target_label_allowlist_max_iou_030_v1",
        "threshold_fit": False, "parameter_search": False, "training_authorized": False,
        "production_model_replacement_authorized": False, "sources": staged_sources,
    }
    output_path = args.output_dir / "android_r11_input.json"
    write_json(output_path, android_input)
    staged_files.extend([
        {"host_path": str(args.target_ledger.resolve()), "device_relative_path": remote_ledger,
         "sha256": sha256_file(args.target_ledger)},
        {"host_path": str(args.projection_receipt.resolve()), "device_relative_path": remote_projection,
         "sha256": sha256_file(args.projection_receipt)},
        {"host_path": str(output_path.resolve()), "device_relative_path": f"{remote_root}/input.json",
         "sha256": sha256_file(output_path)},
    ])
    write_json(args.output_dir / "host_staging_receipt.json", {
        "schema": "blindassist_ustrf_crosscam_r11_android_staging_v1", "diagnostic_set_role": DIAGNOSTIC_ROLE,
        "input_sha256": sha256_file(output_path), "risk_or_detector_outputs_read": False, "files": staged_files,
    })
    print(json.dumps({"ok": True, "input": str(output_path), "sha256": sha256_file(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
