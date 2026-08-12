#!/usr/bin/env python3
"""Offline validator for the D3 Phase-A 48-pool and first-32 lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d3_phase_a import (
    ASSETS,
    CHECKPOINT_SCHEMA,
    MANIFEST_SCHEMA,
    PROTOCOL_SCHEMA,
    first_continuous_window,
    first_qualified,
    load_json,
    lookup_assets,
    parse_trajectory,
    portrait_runs,
    require,
    sha256_file,
    validate_intrinsics_archive,
    write_all,
)


RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_validation_v1"


def checkpoint_path(receipts_root: Path, identity: dict[str, Any]) -> Path:
    return receipts_root / f"{int(identity['pool_order']):02d}-{identity['video_id']}.json"


def validate_checkpoint_seal(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256.json")
    require(path.is_file() and sidecar.is_file(), f"checkpoint or sidecar missing: {path}")
    seal = load_json(sidecar)
    require(path.stat().st_size == int(seal["bytes"]), f"checkpoint bytes drift: {path}")
    require(sha256_file(path) == seal["sha256"], f"checkpoint SHA drift: {path}")
    value = load_json(path)
    require(value.get("schema") == CHECKPOINT_SCHEMA, f"checkpoint schema drift: {path}")
    return value


def verify_file(entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    require(path.is_file(), f"retained file missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"retained bytes drift: {path}")
    require(sha256_file(path) == entry["sha256"], f"retained SHA drift: {path}")
    return path


def expected_selected_rows(processed: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return first_qualified(processed, count)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return len(payload), hashlib.sha256(payload).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = load_json(args.protocol)
    roster = load_json(args.roster)
    head_result = load_json(args.head_result)
    manifest = load_json(args.manifest)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    require(
        protocol["validator"]["sha256"] == sha256_file(Path(__file__)),
        "validator SHA drift",
    )
    require(manifest["protocol_sha256"] == sha256_file(args.protocol), "manifest protocol drift")
    require(manifest["roster_sha256"] == sha256_file(args.roster), "manifest roster drift")
    require(
        manifest["head_result_sha256"] == sha256_file(args.head_result),
        "manifest HEAD drift",
    )
    require(
        manifest["terminal"]
        == "D3_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED",
        "manifest is not the Phase-A PASS terminal",
    )
    require(
        manifest["rgb_depth_confidence_read"] is False
        and manifest["truth_or_model_output_read"] is False
        and manifest["train_development_roles_assigned"] is False,
        "manifest authority drift",
    )
    require(manifest["r2_cohort_access"] == "NONE", "R2 access drift")
    require(manifest["processed_identity_count"] == 48, "processed count drift")
    require(manifest["phase_a_selection_locked"] is True, "selection lock drift")
    require(manifest["selected_identity_count"] == 32, "selected count drift")
    require(manifest["source_archives_retained_for_offline_validation"] is True, "source retention drift")
    require(not (args.manifest.parent / "_temporary_downloads").exists(), "temporary downloads remain")

    pool = roster["pool"]
    require(len(pool) == 48, "roster count drift")
    head_lookup = lookup_assets(head_result)
    receipts_root = args.manifest.parent / "receipts"
    processed: list[dict[str, Any]] = []
    total_body_bytes = 0
    total_pincam_payloads = 0
    for identity in pool:
        value = validate_checkpoint_seal(checkpoint_path(receipts_root, identity))
        for key in ("pool_order", "visit_id", "video_id", "fold"):
            require(str(value[key]) == str(identity[key]), f"identity drift: {key}")
        require(value["attempt_sha256"] == manifest["attempt_sha256"], "attempt SHA drift")
        require(
            value["rgb_depth_confidence_read"] is False
            and value["truth_or_model_output_read"] is False,
            "checkpoint authority drift",
        )
        source_by_asset = {entry["asset"]: entry for entry in value["source_assets"]}
        require(tuple(sorted(source_by_asset)) == tuple(sorted(ASSETS)), "source asset drift")
        video_id = str(identity["video_id"])
        for asset in ASSETS:
            entry = source_by_asset[asset]
            head = head_lookup[(video_id, asset)]
            require(entry["url"] == head["url"], "source URL drift")
            require(int(entry["bytes"]) == int(head["content_length_bytes"]), "source bytes drift")
            require(entry["response_http_status"] == 200, "GET status drift")
            require(
                int(entry["response_content_length_bytes"])
                == int(head["content_length_bytes"]),
                "GET Content-Length drift",
            )
            require(entry["response_etag"] == head["etag"], "GET/HEAD ETag drift")
            require(entry["frozen_head_etag"] == head["etag"], "frozen ETag drift")
            require(
                entry["response_last_modified"] == head["last_modified"],
                "GET/HEAD Last-Modified drift",
            )
            require(
                entry["frozen_head_last_modified"] == head["last_modified"],
                "frozen Last-Modified drift",
            )
            verify_file(entry)
            total_body_bytes += int(entry["bytes"])
        trajectory_path = verify_file(value["trajectory"])
        trajectory = parse_trajectory(trajectory_path)
        archive_path = Path(source_by_asset["lowres_wide_intrinsics.zip"]["path"])
        members, dimension_counts = validate_intrinsics_archive(archive_path)
        total_pincam_payloads += len(members)
        require(
            len(members) == int(value["intrinsics_payload_validated_count"]),
            "pincam payload count drift",
        )
        require(
            dimension_counts == value["source_intrinsics_dimension_counts"],
            "pincam dimension summary drift",
        )
        runs, coverage = portrait_runs(
            list(members),
            trajectory,
            float(protocol["maximum_pose_bracketing_gap_seconds"]),
            float(protocol["maximum_adjacent_frame_gap_seconds"]),
            {int(index) for index in protocol["portrait_orientation_indices"]},
        )
        try:
            selected_stems = first_continuous_window(
                runs, int(protocol["continuous_portrait_frame_count"])
            )
        except ValueError:
            selected_stems = []
        require(bool(selected_stems) == bool(value["eligible"]), "eligibility recompute drift")
        require(selected_stems == value["selected_frame_stems"], "selected stems recompute drift")
        require(coverage == value["coverage"], "coverage recompute drift")
        processed.append(value)

    expected = expected_selected_rows(processed, int(protocol["selected_identity_count"]))
    require(len(expected) == 32, "fewer than 32 recomputed eligible identities")
    expected_orders = [int(row["pool_order"]) for row in expected]
    manifest_orders = [int(row["pool_order"]) for row in manifest["selected_phase_a"]]
    require(manifest_orders == expected_orders, "manifest first-32 lock drift")
    require(
        [int(row["pool_order"]) for row in manifest["eligible_candidates"]]
        == [int(row["pool_order"]) for row in processed if row["eligible"]],
        "eligible candidate summary drift",
    )
    require(total_body_bytes == int(manifest["downloaded_body_bytes"]), "body byte sum drift")
    require(total_body_bytes == int(manifest["declared_download_bytes"]), "declared/body sum drift")
    result = {
        "schema": RESULT_SCHEMA,
        "status": "D3_PHASE_A_OFFLINE_VALIDATION_PASS_48_PROCESSED_32_LOCKED",
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": sha256_file(args.roster),
        "head_result_sha256": sha256_file(args.head_result),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "bytes": args.manifest.stat().st_size,
            "sha256": sha256_file(args.manifest),
        },
        "checkpoint_count": len(processed),
        "source_asset_count": len(processed) * 2,
        "validated_pincam_payload_count": total_pincam_payloads,
        "eligible_identity_count": sum(bool(row["eligible"]) for row in processed),
        "selected_identity_count": len(expected),
        "selected_pool_orders": expected_orders,
        "body_bytes_reverified": total_body_bytes,
        "rgb_depth_confidence_read": False,
        "truth_or_model_output_read": False,
        "train_development_roles_assigned": False,
        "r2_cohort_access": "NONE",
        "next_gate": "EXPLICIT_D3_PHASE_B_DEPTH_CONFIDENCE_HEAD_ONLY_PREFLIGHT_ACTIVATION",
        "authority": "Offline D3 Phase-A source-integrity and label-blind continuity validation only.",
    }
    result_bytes, result_sha = write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_bytes": result_bytes,
                "result_sha256": result_sha,
                "selected_pool_orders": expected_orders,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
