#!/usr/bin/env python3
"""Stage hash-bound R1.2 target-aware Android replay input after oracle clearance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import load_json, sha256_file, write_json
from diagnostic_contract import load_projection, load_target_ledger, require


ROLE = "new_held_out_unscored"
REMOTE_ROOT = "ustrf-crosscam-r12"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--target-ledger", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--source-preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    ledger = load_target_ledger(args.target_ledger)
    require(ledger["diagnostic_set_role"] == ROLE, "R1.2 ledger role mismatch")
    load_projection(args.projection_receipt, args.target_ledger, ledger)
    oracle = load_json(args.oracle_output)
    require(oracle.get("target_ledger_sha256") == sha256_file(args.target_ledger), "oracle ledger hash mismatch")
    require(oracle.get("projection_receipt_sha256") == sha256_file(args.projection_receipt), "oracle projection hash mismatch")
    require(all(item.get("oracle_geometry_passed") is True for item in oracle.get("sources", [])),
            "all six sources must pass oracle before Android materialization")
    prereg = load_json(args.source_preregistration)
    require(prereg.get("dataset_role") == ROLE, "source preregistration role mismatch")
    require(ledger.get("source_preregistration_sha256") == sha256_file(args.source_preregistration),
            "ledger/source preregistration hash mismatch")
    source_by_event = {row["event_id"]: row for row in prereg["held_out_events"]}
    require(set(source_by_event) == {event["event_id"] for event in ledger["events"]},
            "source inventory mismatch")

    staged_sources = []
    staged_files = []
    for event in ledger["events"]:
        row = source_by_event[event["event_id"]]
        local_video = (repo / row["local_video_path"]).resolve()
        require(local_video.is_relative_to(repo) and local_video.is_file(), f"{event['event_id']}: video unavailable")
        require(sha256_file(local_video) == row["video_sha256"], f"{event['event_id']}: video SHA mismatch")
        remote_video = f"{REMOTE_ROOT}/videos/{event['event_id']}{local_video.suffix.lower()}"
        staged_sources.append({
            "event_id": event["event_id"], "source_id": event["source_id"],
            "video_path": remote_video, "video_sha256": row["video_sha256"],
        })
        staged_files.append({
            "host_path": str(local_video), "device_relative_path": remote_video,
            "sha256": row["video_sha256"],
        })

    remote_ledger = f"{REMOTE_ROOT}/target_instance_ledger.json"
    remote_projection = f"{REMOTE_ROOT}/frame_projection_receipt.json"
    android_input = {
        "schema": "blindassist_ustrf_crosscam_target_aware_android_input_v2",
        "diagnostic_set_role": ROLE,
        "target_ledger_path": remote_ledger,
        "target_ledger_sha256": sha256_file(args.target_ledger),
        "projection_receipt_path": remote_projection,
        "projection_receipt_sha256": sha256_file(args.projection_receipt),
        "oracle_output_sha256": sha256_file(args.oracle_output),
        "source_preregistration_sha256": sha256_file(args.source_preregistration),
        "uncertainty_frame_ratios": [0.01, 0.02, 0.03],
        "target_match_contract_id": "target_label_allowlist_max_iou_030_v1",
        "threshold_fit": False,
        "parameter_search": False,
        "training_authorized": False,
        "production_model_replacement_authorized": False,
        "sources": staged_sources,
    }
    output_path = args.output_dir / "android_r12_input.json"
    write_json(output_path, android_input)
    staged_files.extend([
        {"host_path": str(args.target_ledger.resolve()), "device_relative_path": remote_ledger,
         "sha256": sha256_file(args.target_ledger)},
        {"host_path": str(args.projection_receipt.resolve()), "device_relative_path": remote_projection,
         "sha256": sha256_file(args.projection_receipt)},
        {"host_path": str(output_path.resolve()), "device_relative_path": f"{REMOTE_ROOT}/input.json",
         "sha256": sha256_file(output_path)},
    ])
    write_json(args.output_dir / "host_staging_receipt.json", {
        "schema": "blindassist_ustrf_crosscam_r12_android_staging_v1",
        "diagnostic_set_role": ROLE,
        "input_sha256": sha256_file(output_path),
        "risk_or_detector_outputs_read": False,
        "files": staged_files,
    })
    print(json.dumps({"ok": True, "input": str(output_path), "sha256": sha256_file(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
