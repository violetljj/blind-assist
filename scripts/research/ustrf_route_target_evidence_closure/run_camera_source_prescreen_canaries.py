#!/usr/bin/env python3
"""Materialize only the frozen candidate-blind camera-source canaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file
from materialize_crowdbot_holdout_sources import unlink_with_retry, valid_bundle, write_json
from validate_camera_source_prescreen import validate_roster


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    roster_path = args.roster.resolve()
    roster = load_json(roster_path)
    validation = validate_roster(repo, roster)
    roster_sha256 = sha256_file(roster_path)
    transport = roster["transport_and_storage"]
    logical_root = repo / transport["logical_root"]
    dataset_root = logical_root / "dataset"
    evidence_root = logical_root / "evidence"
    cache_root = logical_root / "compressed-cache"
    state_path = evidence_root / "materialization-state-r1.json"
    script_root = Path(__file__).resolve().parent

    state: dict[str, Any] = {
        "schema": "blindassist_ustrf_route_target_camera_source_prescreen_materialization_state_r1",
        "authority": "candidate_blind_reject_only_canary_materialization_not_holdout_admission",
        "source_id": roster["source_id"],
        "roster_sha256": roster_sha256,
        "policy_sha256": validation["policy_sha256"],
        "candidate_outputs_executed": False,
        "app_detector_or_event_outputs_exposed": False,
        "sequence_total": len(roster["selected_canaries"]),
        "sequence_completed": 0,
        "status": "running",
        "transport": {
            "range_workers": transport["range_workers"],
            "range_parts": transport["range_parts"],
            "request_timeout_seconds": transport["request_timeout_seconds"],
        },
        "sequences": [],
        "android_shadow": "closed",
        "h2_depth_ttc_route_risk_flip": "closed",
    }
    write_json(state_path, state)

    for canary in roster["selected_canaries"]:
        sequence_id = canary["sequence_id"]
        sequence_dir = dataset_root / roster["source_id"] / "sequences" / sequence_id
        partial_sequence_dir = sequence_dir.with_name(sequence_dir.name + ".partial")
        raw_path = dataset_root / roster["source_id"] / "raw" / canary["entry_name"]
        receipt_path = evidence_root / f"{roster['source_id']}_{sequence_id}_bag-receipt-r1.json"
        if valid_bundle(sequence_dir):
            status = "sequence_already_complete"
        else:
            if partial_sequence_dir.exists():
                raise RuntimeError(
                    f"partial sequence directory requires explicit audit before retry: {partial_sequence_dir}"
                )
            if receipt_path.exists() and not raw_path.exists():
                raise RuntimeError(f"receipt exists but raw bag and verified bundle are missing: {receipt_path}")
            if not raw_path.exists():
                run_checked(
                    [
                        sys.executable,
                        str(script_root / "stream_remote_zip_entry.py"),
                        "--inventory",
                        str(repo / canary["archive_inventory_path"]),
                        "--entry",
                        canary["entry_name"],
                        "--output",
                        str(raw_path),
                        "--receipt",
                        str(receipt_path),
                        "--max-compressed-bytes",
                        str(canary["compressed_size"]),
                        "--max-uncompressed-bytes",
                        str(canary["uncompressed_size"]),
                        "--range-workers",
                        str(transport["range_workers"]),
                        "--range-parts",
                        str(transport["range_parts"]),
                        "--request-timeout-seconds",
                        str(transport["request_timeout_seconds"]),
                        "--compressed-cache-root",
                        str(cache_root),
                    ]
                )
            run_checked(
                [
                    sys.executable,
                    str(script_root / "materialize_crowdbot_rgbd_sequence.py"),
                    "--bag",
                    str(raw_path),
                    "--bag-receipt",
                    str(receipt_path),
                    "--source-id",
                    roster["source_id"],
                    "--sequence-id",
                    sequence_id,
                    "--output-dir",
                    str(partial_sequence_dir),
                ]
            )
            if not valid_bundle(partial_sequence_dir):
                raise RuntimeError(f"derived canary sequence failed validation: {partial_sequence_dir}")
            sequence_dir.parent.mkdir(parents=True, exist_ok=True)
            partial_sequence_dir.replace(sequence_dir)
            receipt = load_json(receipt_path)
            if receipt["output_sha256"] != sha256_file(raw_path):
                raise RuntimeError(f"raw canary receipt mismatch before cleanup: {raw_path}")
            unlink_with_retry(raw_path)
            write_json(
                sequence_dir / "raw-cleanup-receipt.json",
                {
                    "schema": "blindassist_crowdbot_raw_cleanup_receipt_r1",
                    "deleted_path": raw_path.as_posix(),
                    "deleted_sha256": receipt["output_sha256"],
                    "recoverable_from": receipt["url"],
                    "derived_bundle": (sequence_dir / "bundle.json").as_posix(),
                },
            )
            status = "sequence_complete"
        bundle = load_json(sequence_dir / "bundle.json")
        state["sequence_completed"] += 1
        state["sequences"].append(
            {
                "source_sequence_key": canary["source_sequence_key"],
                "purpose": canary["purpose"],
                "status": status,
                "bundle_path": (sequence_dir / "bundle.json").as_posix(),
                "bundle_sha256": sha256_file(sequence_dir / "bundle.json"),
                "rgb_frame_count": bundle["rgb_frame_count"],
                "aligned_depth_frame_count": bundle["aligned_depth_frame_count"],
                "exact_rgb_depth_pair_count": bundle["exact_rgb_depth_pair_count"],
            }
        )
        write_json(state_path, state)
        print(
            json.dumps(
                {
                    "status": status,
                    "source_sequence_key": canary["source_sequence_key"],
                    "completed": state["sequence_completed"],
                    "total": state["sequence_total"],
                }
            ),
            flush=True,
        )

    state["status"] = "complete"
    write_json(state_path, state)
    print(json.dumps({"status": "complete", "sequence_completed": state["sequence_completed"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
