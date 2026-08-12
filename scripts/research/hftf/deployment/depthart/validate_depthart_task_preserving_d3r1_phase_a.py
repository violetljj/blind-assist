#!/usr/bin/env python3
"""Offline terminal validator for D3R1 Phase-A PASS or legitimate FAIL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d3r1_phase_a import (
    ASSETS,
    ATTEMPT_SCHEMA,
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
    validate_bindings,
    verify_frozen_file,
    write_all,
)
from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_a_assets import (
    roster_rows as frozen_roster_rows,
)


RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_terminal_validation_v1"
PASS_TERMINAL = "D3R1_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED"
FAIL_TERMINAL = "D3R1_PHASE_A_FAIL_FEWER_THAN_32_ELIGIBLE_IDENTITIES"
PASS_SUCCESSOR = (
    "EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_SOURCE_SCOPE_REGISTRATION_"
    "FOR_EXACT_32_PHASE_A_SELECTION"
)


def checkpoint_path(receipts_root: Path, identity: dict[str, Any]) -> Path:
    return receipts_root / f"{int(identity['pool_order']):03d}-{identity['video_id']}.json"


def validate_json_seal(path: Path, expected_schema: str) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256.json")
    require(path.is_file() and sidecar.is_file(), f"checkpoint or sidecar missing: {path}")
    seal = load_json(sidecar)
    require(path.stat().st_size == int(seal["bytes"]), f"checkpoint bytes drift: {path}")
    require(sha256_file(path) == seal["sha256"], f"checkpoint SHA drift: {path}")
    value = load_json(path)
    require(value.get("schema") == expected_schema, f"sealed JSON schema drift: {path}")
    return value


def validate_checkpoint_seal(path: Path) -> dict[str, Any]:
    return validate_json_seal(path, CHECKPOINT_SCHEMA)


def verify_file(entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    require(path.is_file(), f"retained file missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"retained bytes drift: {path}")
    require(sha256_file(path) == entry["sha256"], f"retained SHA drift: {path}")
    return path


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


def expected_terminal_selection(
    processed: list[dict[str, Any]], selected_count: int
) -> tuple[str, list[dict[str, Any]]]:
    selected = first_qualified(processed, selected_count)
    if len(selected) == selected_count:
        return PASS_TERMINAL, selected
    return FAIL_TERMINAL, []


def expected_selected_rows(
    processed: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    return first_qualified(processed, count)


def expected_candidate_summary(processed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
        }
        for row in processed
        if row["eligible"]
    ]


def roster_rows(roster: dict[str, Any]) -> list[dict[str, Any]]:
    return frozen_roster_rows(roster)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")

    protocol, roster, head_result = validate_bindings(
        args.protocol,
        args.roster,
        args.head_result,
        args.source_scope,
        args.activation,
    )
    manifest = validate_json_seal(args.manifest, MANIFEST_SCHEMA)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    for entry, label in [
        (protocol["owning_protocol"], "owning protocol"),
        (protocol["roster"], "roster"),
        (protocol["source_scope"], "source scope"),
        (protocol["head_protocol"], "HEAD protocol"),
        (protocol["head_activation"], "HEAD activation"),
        (protocol["head_governed_result"], "HEAD governed result"),
        (protocol["head_result"], "HEAD machine result"),
        (protocol["head_validation"], "HEAD validation"),
        (protocol["producer"], "producer"),
        (protocol["validator"], "validator"),
        *[(entry, "test") for entry in protocol["tests"]],
        *[(entry, "dependency") for entry in protocol["dependencies"]],
    ]:
        verify_frozen_file(entry, label)
    require(protocol["roster"]["sha256"] == sha256_file(args.roster), "passed roster drift")
    require(protocol["head_result"]["sha256"] == sha256_file(args.head_result), "passed HEAD result drift")
    require(protocol["source_scope"]["sha256"] == sha256_file(args.source_scope), "passed source-scope drift")
    require(protocol["validator"]["sha256"] == sha256_file(Path(__file__)), "validator self SHA drift")
    require(manifest["protocol_sha256"] == sha256_file(args.protocol), "manifest protocol drift")
    require(manifest["roster_sha256"] == sha256_file(args.roster), "manifest roster drift")
    require(manifest["head_result_sha256"] == sha256_file(args.head_result), "manifest HEAD drift")
    require(manifest["source_scope_sha256"] == protocol["source_scope"]["sha256"], "manifest source drift")
    require(manifest["activation_sha256"] == sha256_file(args.activation), "manifest activation drift")
    require(
        manifest["rgb_depth_confidence_read"] is False
        and manifest["truth_or_model_output_read"] is False
        and manifest["train_development_roles_assigned"] is False,
        "manifest authority drift",
    )
    require(manifest["r2_cohort_access"] == "NONE", "manifest R2 access drift")
    require(manifest["source_archives_retained_for_offline_validation"] is True, "source retention drift")
    require(manifest["temporary_archives_retained"] is False, "temporary retention drift")
    require(not (args.manifest.parent / "_temporary_downloads").exists(), "temporary downloads remain")

    attempt_path = args.manifest.parent / "attempt.json"
    require(attempt_path.is_file(), "attempt receipt missing")
    require(sha256_file(attempt_path) == manifest["attempt_sha256"], "attempt SHA drift")
    attempt = load_json(attempt_path)
    require(attempt.get("schema") == ATTEMPT_SCHEMA, "attempt schema drift")
    for name in ("protocol", "roster", "head_result", "source_scope", "activation"):
        require(
            attempt["bindings"][f"{name}_sha256"] == manifest[f"{name}_sha256"],
            f"attempt {name} drift",
        )

    pool = roster_rows(roster)
    require(
        attempt["identity_plan"]
        == [
            {
                "pool_order": row["pool_order"],
                "visit_id": row["visit_id"],
                "video_id": row["video_id"],
                "fold": row["fold"],
            }
            for row in pool
        ],
        "attempt identity plan drift",
    )
    head_lookup = lookup_assets(head_result, 254, roster)
    receipts_root = args.manifest.parent / "receipts"
    processed: list[dict[str, Any]] = []
    total_body_bytes = 0
    total_pincam_payloads = 0
    for identity in pool:
        value = validate_checkpoint_seal(checkpoint_path(receipts_root, identity))
        for key in ("pool_order", "visit_id", "video_id", "fold"):
            require(str(value[key]) == str(identity[key]), f"identity drift: {key}")
        require(value["attempt_sha256"] == manifest["attempt_sha256"], "checkpoint attempt drift")
        require(
            value["rgb_depth_confidence_read"] is False
            and value["truth_or_model_output_read"] is False,
            "checkpoint authority drift",
        )
        source_by_asset = {entry["asset"]: entry for entry in value["source_assets"]}
        require(len(source_by_asset) == 2 and set(source_by_asset) == set(ASSETS), "source asset drift")
        video_id = identity["video_id"]
        expected_source_root = (args.manifest.parent / "source" / "Training" / video_id).resolve()
        for asset in ASSETS:
            entry = source_by_asset[asset]
            head = head_lookup[(video_id, asset)]
            require(entry["url"] == head["url"], "source URL drift")
            require(int(entry["bytes"]) == int(head["content_length_bytes"]), "source bytes drift")
            require(entry["response_http_status"] == 200, "GET status drift")
            require(entry["response_final_url"] == head["url"], "GET final URL drift")
            require(
                len(entry["attempt_history"]) == int(entry["attempts"])
                and entry["attempt_history"][-1]["error"] is None
                and all(item["method"] == "GET" for item in entry["attempt_history"]),
                "GET attempt history drift",
            )
            require(int(entry["response_content_length_bytes"]) == int(head["content_length_bytes"]), "GET length drift")
            require(entry["response_etag"] == head["etag"] == entry["frozen_head_etag"], "GET/HEAD ETag drift")
            require(
                entry["response_last_modified"]
                == head["last_modified"]
                == entry["frozen_head_last_modified"],
                "GET/HEAD Last-Modified drift",
            )
            source_path = verify_file(entry)
            require(source_path.parent.resolve() == expected_source_root, "retained source root drift")
            require(source_path.name == asset, "retained source filename drift")
            total_body_bytes += int(entry["bytes"])

        trajectory_path = verify_file(value["trajectory"])
        trajectory_entry = source_by_asset["lowres_wide.traj"]
        require(
            trajectory_path == Path(trajectory_entry["path"])
            and int(value["trajectory"]["bytes"]) == int(trajectory_entry["bytes"])
            and value["trajectory"]["sha256"] == trajectory_entry["sha256"],
            "trajectory receipt drift",
        )
        trajectory = parse_trajectory(trajectory_path)
        require(int(value["trajectory"]["row_count"]) == int(trajectory.shape[0]), "trajectory row count drift")
        archive_entry = source_by_asset["lowres_wide_intrinsics.zip"]
        require(archive_entry["zip_crc_and_member_safety_checked"] is True, "archive check flag drift")
        members, dimension_counts = validate_intrinsics_archive(Path(archive_entry["path"]))
        total_pincam_payloads += len(members)
        require(len(members) == int(value["intrinsics_payload_validated_count"]), "pincam count drift")
        require(dimension_counts == value["source_intrinsics_dimension_counts"], "dimension summary drift")
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
            reason = "PASS"
        except ValueError as error:
            selected_stems = []
            reason = str(error)
        times = [float(stem.rsplit("_", 1)[-1]) for stem in selected_stems]
        maximum_gap = (
            max(right - left for left, right in zip(times, times[1:]))
            if len(times) > 1
            else None
        )
        require(bool(selected_stems) == bool(value["eligible"]), "eligibility recompute drift")
        require(reason == value["eligibility_reason"], "eligibility reason drift")
        require(selected_stems == value["selected_frame_stems"], "selected stems drift")
        require(len(selected_stems) == int(value["selected_frame_count"]), "selected count drift")
        require((times[0] if times else None) == value["selected_start_timestamp"], "selected start drift")
        require((times[-1] if times else None) == value["selected_end_timestamp"], "selected end drift")
        require(maximum_gap == value["maximum_selected_adjacent_gap_seconds"], "selected gap drift")
        require(coverage == value["coverage"], "coverage recompute drift")
        processed.append(value)

    require(processed == manifest["processed"], "manifest/checkpoint payload drift")
    require(int(manifest["pool_count"]) == 127, "manifest pool count drift")
    require(int(manifest["processed_identity_count"]) == 127, "manifest processed count drift")
    eligible_count = sum(bool(row["eligible"]) for row in processed)
    require(int(manifest["eligible_identity_count"]) == eligible_count, "eligible count drift")
    require(manifest["eligible_candidates"] == expected_candidate_summary(processed), "candidate summary drift")
    terminal, selected = expected_terminal_selection(processed, int(protocol["selected_identity_count"]))
    require(manifest["terminal"] == terminal, "terminal recompute drift")
    expected_orders = [int(row["pool_order"]) for row in selected]
    require([int(row["pool_order"]) for row in manifest["selected_phase_a"]] == expected_orders, "selection drift")
    require(bool(manifest["phase_a_selection_locked"]) == (terminal == PASS_TERMINAL), "lock drift")
    require(int(manifest["selected_identity_count"]) == len(selected), "selected count drift")
    if terminal == FAIL_TERMINAL:
        require(eligible_count < int(protocol["selected_identity_count"]), "FAIL denominator drift")
        require(manifest["selected_phase_a"] == [], "FAIL published partial selection")
    require(total_body_bytes == int(manifest["downloaded_body_bytes"]), "body byte sum drift")
    require(total_body_bytes == int(manifest["declared_download_bytes"]), "declared byte sum drift")
    require(total_body_bytes == int(protocol["expected_total_content_length_bytes"]), "protocol byte sum drift")
    expected_receipts = {
        checkpoint_path(receipts_root, identity).name for identity in pool
    } | {
        checkpoint_path(receipts_root, identity).with_suffix(".sha256.json").name
        for identity in pool
    }
    require(
        {path.name for path in receipts_root.iterdir()} == expected_receipts,
        "checkpoint inventory drift",
    )
    require(
        len(list((args.manifest.parent / "source" / "Training").glob("*/*"))) == 254,
        "retained source inventory drift",
    )
    expected_selected_payload = [
        {
            "selection_order": index + 1,
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
            "role": "D3R1_PHASE_A_SELECTED_IDENTITY_ONLY",
            "selected_frame_stems": row["selected_frame_stems"],
        }
        for index, row in enumerate(selected)
    ]
    require(manifest["selected_phase_a"] == expected_selected_payload, "selected payload drift")

    result = {
        "schema": RESULT_SCHEMA,
        "status": "D3R1_PHASE_A_OFFLINE_TERMINAL_VALIDATION_PASS",
        "execution_validity": "VALID",
        "scientific_terminal": terminal,
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
        "eligible_identity_count": eligible_count,
        "eligible_pool_orders": [int(row["pool_order"]) for row in processed if row["eligible"]],
        "phase_a_selection_locked": terminal == PASS_TERMINAL,
        "selected_identity_count": len(selected),
        "selected_pool_orders": expected_orders,
        "body_bytes_reverified": total_body_bytes,
        "rgb_depth_confidence_read": False,
        "truth_or_model_output_read": False,
        "train_development_roles_assigned": False,
        "r2_cohort_access": "NONE",
        "phase_b_authorized": False,
        "next_gate": PASS_SUCCESSOR if terminal == PASS_TERMINAL else None,
        "authority": (
            "Offline D3R1 Phase-A retained-source integrity and label-blind continuity "
            "terminal validation only; no Phase-B asset, role, outcome, performance, product "
            "or safety authority."
        ),
    }
    result_bytes, result_sha = write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scientific_terminal": terminal,
                "eligible_identity_count": eligible_count,
                "selected_identity_count": len(selected),
                "result_bytes": result_bytes,
                "result_sha256": result_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
