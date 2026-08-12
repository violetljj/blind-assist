#!/usr/bin/env python3
"""Read-only terminal audit for D3 Phase-A, including a legitimate FAIL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d3_phase_a import (
    ASSETS,
    ATTEMPT_SCHEMA,
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
    verify_frozen_file,
)
from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d3_phase_a import (
    checkpoint_path,
    validate_checkpoint_seal,
    verify_file,
    write_json_exclusive,
)


REPAIR_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_terminal_validator_repair_v1"
RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_terminal_audit_v1"
PASS_TERMINAL = "D3_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED"
FAIL_TERMINAL = "D3_PHASE_A_FAIL_FEWER_THAN_32_ELIGIBLE_IDENTITIES"


def expected_terminal_selection(
    processed: list[dict[str, Any]], selected_count: int
) -> tuple[str, list[dict[str, Any]]]:
    candidates = first_qualified(processed, selected_count)
    if len(candidates) == selected_count:
        return PASS_TERMINAL, candidates
    return FAIL_TERMINAL, []


def validate_json_seal(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256.json")
    require(path.is_file() and sidecar.is_file(), f"sealed JSON missing: {path}")
    seal = load_json(sidecar)
    require(path.stat().st_size == int(seal["bytes"]), f"sealed JSON bytes drift: {path}")
    require(sha256_file(path) == seal["sha256"], f"sealed JSON SHA drift: {path}")
    return load_json(path)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")

    repair = load_json(args.repair_receipt)
    protocol = load_json(args.protocol)
    roster = load_json(args.roster)
    head_result = load_json(args.head_result)
    manifest = validate_json_seal(args.manifest)
    require(repair.get("schema") == REPAIR_SCHEMA, "repair receipt schema drift")
    require(
        repair.get("status") == "POST_TERMINAL_VALIDATOR_COVERAGE_REPAIR_FROZEN",
        "repair receipt status drift",
    )
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    for name, path in (
        ("protocol", args.protocol),
        ("roster", args.roster),
        ("head_result", args.head_result),
        ("manifest", args.manifest),
    ):
        binding = repair["bindings"][name]
        require(path.stat().st_size == int(binding["bytes"]), f"repair {name} bytes drift")
        require(sha256_file(path) == binding["sha256"], f"repair {name} SHA drift")
    verify_frozen_file(repair["bindings"]["original_validator"], "original validator")
    verify_frozen_file(repair["bindings"]["terminal_auditor"], "terminal auditor")
    verify_frozen_file(repair["bindings"]["terminal_auditor_test"], "terminal auditor test")
    require(
        repair["bindings"]["terminal_auditor"]["sha256"] == sha256_file(Path(__file__)),
        "terminal auditor self SHA drift",
    )
    require(
        protocol["validator"] == repair["bindings"]["original_validator"],
        "original frozen validator binding drift",
    )
    require(repair["authority"]["body_redownload"] is False, "repair redownload scope drift")
    require(repair["authority"]["gate_or_selection_change"] is False, "repair gate scope drift")
    require(repair["authority"]["phase_b"] is False, "repair Phase-B scope drift")
    require(repair["authority"]["truth_or_model"] is False, "repair outcome scope drift")

    require(manifest["protocol_sha256"] == sha256_file(args.protocol), "manifest protocol drift")
    require(manifest["roster_sha256"] == sha256_file(args.roster), "manifest roster drift")
    require(manifest["head_result_sha256"] == sha256_file(args.head_result), "manifest HEAD drift")
    require(
        manifest["source_scope_sha256"] == protocol["source_scope"]["sha256"],
        "manifest source-scope drift",
    )
    require(
        manifest["activation_sha256"] == repair["bindings"]["activation"]["sha256"],
        "manifest activation drift",
    )
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
    require(attempt["bindings"]["protocol_sha256"] == manifest["protocol_sha256"], "attempt protocol drift")
    require(attempt["bindings"]["roster_sha256"] == manifest["roster_sha256"], "attempt roster drift")
    require(attempt["bindings"]["head_result_sha256"] == manifest["head_result_sha256"], "attempt HEAD drift")
    require(attempt["bindings"]["source_scope_sha256"] == manifest["source_scope_sha256"], "attempt source drift")
    require(attempt["bindings"]["activation_sha256"] == manifest["activation_sha256"], "attempt activation drift")

    pool = roster["pool"]
    require(
        len(pool) == 48
        and [int(row["pool_order"]) for row in pool] == list(range(1, 49)),
        "roster pool drift",
    )
    require(attempt["identity_plan"] == [
        {
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
        }
        for row in pool
    ], "attempt identity plan drift")
    head_lookup = lookup_assets(head_result)
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
        video_id = str(identity["video_id"])
        expected_source_root = (
            args.manifest.parent / "source" / "Training" / video_id
        ).resolve()
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
        require(int(value["trajectory"]["row_count"]) == int(trajectory.shape[0]), "trajectory row-count drift")
        archive_entry = source_by_asset["lowres_wide_intrinsics.zip"]
        require(archive_entry["zip_crc_and_member_safety_checked"] is True, "archive check flag drift")
        archive_path = Path(archive_entry["path"])
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
            eligibility_reason = "PASS"
        except ValueError as error:
            selected_stems = []
            eligibility_reason = str(error)
        times = [float(stem.rsplit("_", 1)[-1]) for stem in selected_stems]
        maximum_gap = (
            max(right - left for left, right in zip(times, times[1:]))
            if len(times) > 1
            else None
        )
        require(bool(selected_stems) == bool(value["eligible"]), "eligibility recompute drift")
        require(eligibility_reason == value["eligibility_reason"], "eligibility reason drift")
        require(selected_stems == value["selected_frame_stems"], "selected stems recompute drift")
        require(len(selected_stems) == int(value["selected_frame_count"]), "selected frame count drift")
        require((times[0] if times else None) == value["selected_start_timestamp"], "selected start drift")
        require((times[-1] if times else None) == value["selected_end_timestamp"], "selected end drift")
        require(maximum_gap == value["maximum_selected_adjacent_gap_seconds"], "selected gap drift")
        require(coverage == value["coverage"], "coverage recompute drift")
        processed.append(value)

    require(processed == manifest["processed"], "manifest/checkpoint payload drift")
    require(int(manifest["pool_count"]) == 48, "manifest pool count drift")
    require(int(manifest["processed_identity_count"]) == 48, "manifest processed count drift")
    eligible_count = sum(bool(row["eligible"]) for row in processed)
    require(int(manifest["eligible_identity_count"]) == eligible_count, "eligible count drift")
    require(manifest["eligible_candidates"] == expected_candidate_summary(processed), "candidate summary drift")
    expected_terminal, expected_selected = expected_terminal_selection(
        processed, int(protocol["selected_identity_count"])
    )
    require(manifest["terminal"] == expected_terminal, "terminal recompute drift")
    expected_orders = [int(row["pool_order"]) for row in expected_selected]
    manifest_orders = [int(row["pool_order"]) for row in manifest["selected_phase_a"]]
    require(manifest_orders == expected_orders, "manifest selected order drift")
    require(
        bool(manifest["phase_a_selection_locked"]) == (expected_terminal == PASS_TERMINAL),
        "selection lock drift",
    )
    require(int(manifest["selected_identity_count"]) == len(expected_selected), "selected count drift")
    if expected_terminal == FAIL_TERMINAL:
        require(eligible_count < int(protocol["selected_identity_count"]), "FAIL denominator drift")
        require(manifest["selected_phase_a"] == [], "FAIL published a partial selection")
    require(total_body_bytes == int(manifest["downloaded_body_bytes"]), "body byte sum drift")
    require(total_body_bytes == int(manifest["declared_download_bytes"]), "declared/body sum drift")
    require(total_body_bytes == int(protocol["expected_total_content_length_bytes"]), "protocol/body sum drift")

    result = {
        "schema": RESULT_SCHEMA,
        "status": "D3_PHASE_A_OFFLINE_TERMINAL_AUDIT_PASS",
        "execution_validity": "VALID_WITH_POST_TERMINAL_VALIDATOR_COVERAGE_REPAIR",
        "scientific_terminal": expected_terminal,
        "repair_receipt": {
            "path": str(args.repair_receipt.resolve()),
            "bytes": args.repair_receipt.stat().st_size,
            "sha256": sha256_file(args.repair_receipt),
        },
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
        "phase_a_selection_locked": expected_terminal == PASS_TERMINAL,
        "selected_identity_count": len(expected_selected),
        "selected_pool_orders": expected_orders,
        "body_bytes_reverified": total_body_bytes,
        "rgb_depth_confidence_read": False,
        "truth_or_model_output_read": False,
        "train_development_roles_assigned": False,
        "r2_cohort_access": "NONE",
        "phase_b_authorized": expected_terminal == PASS_TERMINAL,
        "next_gate": (
            "EXPLICIT_D3_PHASE_B_DEPTH_CONFIDENCE_HEAD_ONLY_PREFLIGHT_ACTIVATION"
            if expected_terminal == PASS_TERMINAL
            else "NONE_D3_PHASE_A_FAIL_TERMINAL"
        ),
        "authority": "Read-only D3 Phase-A terminal and retained-source integrity audit only; no new data, gate, selection, Phase-B, truth, model, training, R2, performance, production or safety authority.",
    }
    result_bytes, result_sha = write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "execution_validity": result["execution_validity"],
                "scientific_terminal": result["scientific_terminal"],
                "eligible_identity_count": eligible_count,
                "selected_identity_count": len(expected_selected),
                "result_bytes": result_bytes,
                "result_sha256": result_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
